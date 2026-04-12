"""
core/graph.py

LangGraph pipeline graph — the orchestration layer.

Wires all agents as graph nodes and defines the transitions between them.
The human-in-the-loop confirmation gate lives here as a LangGraph interrupt.

Current graph (Phase 1):
  START
    → parse_resume
    → recommend_profiles
    → [INTERRUPT: awaiting_user_confirmation]
    → apply_confirmation
    → END   (Agents 3–6 will extend from here in Phase 2)
"""

from __future__ import annotations

import logging
import os
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
from agents.ranker.ranker_agent import node_ranker
from agents.signals.signals_agent import node_signals
from core.state.session_state import SessionState

logger = logging.getLogger(__name__)


# ── Node functions ────────────────────────────────────────────────────────────
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
        logger.warning("[graph] Skipping profile recommendation — parse failed")
        return state

    updated = run_profile_recommender(session)
    return updated.model_dump()


def node_user_confirmation(state: dict) -> dict:
    """
    Node 3: Human-in-the-loop gate.

    LangGraph interrupt() pauses graph execution here and persists full state
    to the checkpointer. The graph resumes when the API layer calls
    graph.invoke() again with the user's confirmation payload in the state.

    The interrupt payload is the list of suggested profile titles — sent to
    the frontend so the user can see what they're confirming.
    """
    session = SessionState(**state)

    # If an error occurred upstream, skip the gate
    if session.error:
        logger.warning(
            f"[graph] Skipping confirmation gate — upstream error: {session.error}"
        )
        return state

    suggested_titles = [p.title for p in session.suggested_profiles]

    logger.info(
        f"[graph] Interrupting for user confirmation — "
        f"{len(suggested_titles)} profiles to confirm"
    )

    # This call pauses execution. The value passed to interrupt() is available
    # to the caller as the interrupt payload — use it to drive the frontend UI.
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


# ── Conditional edge: should we proceed after parsing? ────────────────────────

def route_after_parse(state: dict) -> str:
    """
    After parsing, route to recommender if successful, else END.
    """
    if state.get("parse_failed"):
        logger.info("[graph] Parse failed — routing to END")
        return "end"
    return "recommend"


def route_after_recommend(state: dict) -> str:
    """
    After recommendation, route to confirmation gate if successful, else END.
    """
    if state.get("error"):
        logger.info("[graph] Recommendation error — routing to END")
        return "end"
    return "confirm"


# ── Graph builder ─────────────────────────────────────────────────────────────

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
    builder.add_node("rank_results",       node_ranker)
    builder.add_node("hiring_signals",     node_signals)

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

    # After confirmation → job search → rank → signals → END
    builder.add_edge("user_confirmation", "search_jobs")
    builder.add_edge("search_jobs",       "rank_results")
    builder.add_edge("rank_results",      "hiring_signals")
    builder.add_edge("hiring_signals",     END)

    # ── Checkpointer ─────────────────────────────────────────────────────────
    if use_postgres:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from psycopg_pool import AsyncConnectionPool
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                raise ValueError("DATABASE_URL not set")
            # Create pool with open=False — lifespan will open it async
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


# ── Convenience runner ────────────────────────────────────────────────────────

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
        f"[graph] Pipeline paused at confirmation gate — "
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
    from langgraph.types import Command

    config = create_session_config(session_id)

    resume_payload = {
        "selected_titles": selected_titles,
        "custom_profiles": custom_profiles or [],
    }

    state = await graph.ainvoke(
        Command(resume=resume_payload),
        config=config,
    )

    logger.info(
        f"[graph] Pipeline resumed and completed — "
        f"session_id={session_id}, "
        f"confirmed={len(selected_titles)} profiles"
    )
    return state
