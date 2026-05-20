"""
core/graph.py

LangGraph pipeline graph -- the orchestration layer.

Current pipeline:
  START
    -> parse_resume          (Agent 1 -- Resume Parser, Claude Sonnet)
    -> recommend_profiles    (Agent 2 -- Profile Recommender, Claude Haiku)
    -> [INTERRUPT: user_confirmation]
    -> search_jobs           (Agent 3 -- Job Search, JSearch API)
    -> prune_urls            (Agent 4 -- URL Pruner, Claude Haiku)
    -> rank_results          (Agent 6 -- Ranker, Claude Haiku x 60 jobs)
    -> END

Hiring signals agent (Option A -- disconnected):
  Code preserved in agents/signals/. Removed from active graph due to
  low signal reliability for Indian mid-market companies and sparse
  NewsAPI coverage. Reinstate by adding node + edge when a better
  data source is available.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agents.parser.resume_parser import run_resume_parser
from agents.recommender.profile_recommender import (
    apply_user_confirmation,
    run_profile_recommender,
)
from agents.job_search.job_search_agent import node_job_search
from agents.pruner.url_pruner import node_url_pruner
from agents.ranker.ranker_agent import node_ranker
# agents.signals intentionally not imported -- Option A (disconnected, code preserved)
from core.state.session_state import SessionState

logger = logging.getLogger(__name__)


# -- Node functions ------------------------------------------------------------
# Each node receives the full state dict, calls the agent, returns updated fields.
# LangGraph merges returned fields back into the state automatically.

def node_parse_resume(state: dict) -> dict:
    """Node 1: Resume Parser."""
    session = SessionState(**state)
    updated = run_resume_parser(session)
    return updated.model_dump()


def node_recommend_profiles(state: dict) -> dict:
    """Node 2: Profile Recommender."""
    session = SessionState(**state)

    # Skip if parsing failed upstream
    if session.parse_failed:
        logger.warning("[graph] Skipping profile recommendation -- parse failed")
        return state

    updated = run_profile_recommender(session)
    return updated.model_dump()


def node_user_confirmation(state: dict) -> dict:
    """
    Node 3: Human-in-the-loop gate.

    LangGraph interrupt() pauses graph execution here and persists full state
    to the checkpointer. The graph resumes when the API layer calls
    graph.invoke() again with the user's confirmation payload in the state.

    The interrupt payload is the list of suggested profile titles -- sent to
    the frontend so the user can see what they're confirming.
    """
    session = SessionState(**state)

    # If an error occurred upstream, skip the gate
    if session.error:
        logger.warning(
            f"[graph] Skipping confirmation gate -- upstream error: {session.error}"
        )
        return state

    suggested_titles = [p.title for p in session.suggested_profiles]

    logger.info(
        f"[graph] Interrupting for user confirmation -- "
        f"{len(suggested_titles)} profiles to confirm"
    )

    # This call pauses execution. The value passed to interrupt() is available
    # to the caller as the interrupt payload -- use it to drive the frontend UI.
    user_input = interrupt({
        "action":   "confirm_profiles",
        "profiles": [p.model_dump() for p in session.suggested_profiles],
        "message":  "Select the job profiles you want to search for.",
    })

    # user_input is whatever the API layer passes back when resuming.
    # Expected format:
    # {
    #   "selected_titles": ["Lead Data Scientist", "ML Engineer"],
    #   "custom_profiles": ["Data Analytics Lead"]   # optional
    # }
    selected_titles  = user_input.get("selected_titles", [])
    custom_profiles  = user_input.get("custom_profiles", [])

    updated = apply_user_confirmation(session, selected_titles, custom_profiles)
    return updated.model_dump()


# -- Conditional edge: should we proceed after parsing? ------------------------

def route_after_parse(state: dict) -> str:
    if state.get("parse_failed"):
        logger.info("[graph] Parse failed -- routing to END")
        return "end"
    return "recommend"


def route_after_recommend(state: dict) -> str:
    if state.get("error"):
        logger.info("[graph] Recommendation error -- routing to END")
        return "end"
    return "confirm"


# -- Observability finalisation node ------------------------------------------

def node_finalise(state: dict) -> dict:
    """
    Final node before END -- aggregates per-agent metrics into SessionMetrics,
    logs to MLflow, and stores the serialised payload in state.session_metrics
    for Postgres persistence by the API layer.

    Runs synchronously -- no LLM calls, just in-memory aggregation + I/O.
    Fail-open: any error here is logged and skipped; pipeline state is unaffected.
    """
    session = SessionState(**state)

    try:
        from core.observability.metrics import SessionMetrics, _tokens_to_inr
        from core.observability.mlflow_logger import log_session_metrics
        from core.config.config_loader import cfg

        # Reconstruct SessionMetrics from agent_metrics accumulated in state
        metrics = SessionMetrics(session_id=session.session_id)

        for agent, data in session.agent_metrics.items():
            # Reconstruct latency
            if "latency_secs" in data:
                # Fake the start time to produce correct latency
                metrics._agent_start[agent]   = 0.0
                metrics.agent_latencies[agent] = data["latency_secs"]

            # Reconstruct token usage
            in_tok  = data.get("input_tokens",  0)
            out_tok = data.get("output_tokens", 0)
            model   = data.get("model", "claude-haiku-4-5-20251001")
            calls   = max(int(data.get("llm_calls", 1) or 1), 1)
            if in_tok or out_tok:
                # Distribute tokens across N calls without losing the
                # remainder to integer division. The first `extra_in` calls
                # get one additional input token; same for output. Total
                # recorded tokens equal the original aggregate exactly.
                base_in,  extra_in  = divmod(in_tok,  calls)
                base_out, extra_out = divmod(out_tok, calls)
                for i in range(calls):
                    metrics.record_llm_call(
                        agent         = agent,
                        input_tokens  = base_in  + (1 if i < extra_in  else 0),
                        output_tokens = base_out + (1 if i < extra_out else 0),
                        model         = model,
                    )

        # Quality metrics from ranked results
        ranked     = session.ranked_jobs
        fit_scores = [j.fit_score for j in ranked]
        raw_count  = state.get("_raw_jobs_count", len(session.raw_jobs))

        # Per-stage funnel stats from ranker (Task #9)
        ranker_metrics = session.agent_metrics.get("ranker", {})
        jobs_by_stage  = ranker_metrics.get("jobs_by_stage", {})

        quality = {
            "raw_jobs_fetched":         raw_count,
            "ranked_jobs":              len(ranked),
            "high_fit_count":           sum(1 for s in fit_scores if s >= 0.70),
            "moderate_fit_count":       sum(1 for s in fit_scores if 0.50 <= s < 0.70),
            "mean_fit_score":           round(sum(fit_scores) / len(fit_scores), 3) if fit_scores else 0.0,
            "score_p25":                round(sorted(fit_scores)[len(fit_scores)//4],   3) if len(fit_scores) >= 3 else None,
            "score_p75":                round(sorted(fit_scores)[3*len(fit_scores)//4], 3) if len(fit_scores) >= 3 else None,
            "fallback_activated":       int(session.fallback_activated),
            "india_accessible_dropped": ranker_metrics.get("india_accessible_dropped", 0),
            "pre_filter_skill_dropped": ranker_metrics.get("pre_filter_skill_dropped", 0),
            # Per-stage counts surfaced as individual MLflow metrics for funnel analysis
            **{f"stage_{k}": v for k, v in jobs_by_stage.items()},
        }

        # Total latency: sum of per-agent latencies. The pipeline is
        # sequential between agents (parallelism happens *inside* the ranker
        # and job-search agents, already reflected in their per-agent
        # latency_secs values). Wall-clock cannot be measured here because
        # SessionMetrics was just constructed -- started_at is "now", not
        # the actual session start time.
        total_latency = sum(metrics.agent_latencies.values())

        metrics.finish(
            quality_data           = quality,
            total_latency_override = total_latency,
        )
        payload = metrics.to_dict()

        # Store in state for API layer to persist to Postgres
        session.session_metrics   = payload
        session.pipeline_complete = True
        session.results_ready     = True

        # Log to MLflow (fail-open internally)
        log_session_metrics(metrics)

        logger.info(
            f"[graph:finalise] Metrics captured -- "
            f"latency={payload['total_latency_secs']}s | "
            f"cost=INR {payload['total_cost_inr']} | "
            f"calls={payload['total_llm_calls']}"
        )

    except Exception as e:
        logger.error(f"[graph:finalise] Metrics aggregation failed: {e}", exc_info=False)
        session.pipeline_complete = True
        session.results_ready     = True

    return session.model_dump()


# -- Graph builder -------------------------------------------------------------

def build_graph(use_postgres: bool = False):
    """
    Build and compile the LangGraph pipeline.

    Args:
        use_postgres: If True, use Postgres checkpointer (production).
                      If False, use in-memory checkpointer (development/testing).

    Returns compiled LangGraph graph.
    """
    builder = StateGraph(dict)

    # Register nodes
    builder.add_node("parse_resume",       node_parse_resume)
    builder.add_node("recommend_profiles", node_recommend_profiles)
    builder.add_node("user_confirmation",  node_user_confirmation)
    builder.add_node("search_jobs",        node_job_search)
    builder.add_node("prune_urls",         node_url_pruner)
    builder.add_node("rank_results",       node_ranker)
    builder.add_node("finalise",           node_finalise)   # observability aggregation
    # hiring_signals intentionally excluded -- Option A. Code in agents/signals/.

    # Entry point
    builder.add_edge(START, "parse_resume")

    # Conditional routing after parse
    builder.add_conditional_edges(
        "parse_resume",
        route_after_parse,
        {"recommend": "recommend_profiles", "end": END},
    )

    # Conditional routing after recommend
    builder.add_conditional_edges(
        "recommend_profiles",
        route_after_recommend,
        {"confirm": "user_confirmation", "end": END},
    )

    # After confirmation -> job search -> URL prune -> rank -> finalise -> END
    builder.add_edge("user_confirmation", "search_jobs")
    builder.add_edge("search_jobs",       "prune_urls")
    builder.add_edge("prune_urls",        "rank_results")
    builder.add_edge("rank_results",      "finalise")
    builder.add_edge("finalise",          END)

    # -- Checkpointer ---------------------------------------------------------
    if use_postgres:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from psycopg_pool import AsyncConnectionPool
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                raise ValueError("DATABASE_URL not set")
            # Create pool with open=False -- lifespan will open it async
            pool = AsyncConnectionPool(
                conninfo=db_url,
                max_size=10,
                open=False,
            )
            checkpointer = AsyncPostgresSaver(pool)
            # Store pool on checkpointer so lifespan can access it
            checkpointer._pool = pool
            logger.info("[graph] Using Postgres checkpointer (async)")
        except Exception as e:
            logger.warning(
                f"[graph] Postgres checkpointer unavailable ({e}), "
                f"falling back to in-memory checkpointer"
            )
            checkpointer = MemorySaver()
    else:
        checkpointer = MemorySaver()
        logger.info("[graph] Using in-memory checkpointer")

    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["user_confirmation"],  # pause BEFORE this node runs
    )

    logger.info("[graph] Graph compiled successfully")
    return graph


# -- Convenience runner --------------------------------------------------------

def create_session_config(session_id: str) -> dict:
    """
    Create the LangGraph thread config for a session.
    All graph calls for the same session must use the same thread_id.
    """
    return {"configurable": {"thread_id": session_id}}


async def run_until_confirmation(
    graph,
    initial_state: dict,
    session_id: str,
) -> dict:
    """
    Run the graph from START until it hits the confirmation interrupt.
    Uses ainvoke to support async nodes (e.g. job search agent).
    """
    config = create_session_config(session_id)
    state = await graph.ainvoke(initial_state, config=config)
    logger.info(
        f"[graph] Pipeline paused at confirmation gate -- "
        f"session_id={session_id}"
    )
    return state


async def resume_after_confirmation(
    graph,
    session_id: str,
    selected_titles: list[str],
    custom_profiles: list[str] | None = None,
) -> dict:
    """
    Resume the graph after the user has confirmed their profile selections.
    Uses ainvoke to support async nodes (e.g. job search agent).
    """
    from langgraph.types import Comman