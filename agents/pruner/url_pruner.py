"""
agents/pruner/url_pruner.py

Agent 5 -- URL Pruner

Responsibility:
  Receive raw_jobs[] from the job search agent, send all apply_urls to Claude
  in a single LLM call, and return only the jobs whose URLs are classified as
  coming from reliable, high-quality sources.

Why LLM not hardcoded rules:
  The hardcoded inclusion list in jsearch.py covers known quality publishers
  (LinkedIn, Indeed, Glassdoor) and known ATS domains. But the long tail of
  direct company career pages on unknown domains cannot be enumerated.
  An LLM reading the URL classifies these correctly using world knowledge --
  "careers.stripe.com" is obviously a direct company page even if "stripe"
  is not in any hardcoded list.

Coverage:
  All job sources are classified -- jsearch, remoteok, active_jobs_db,
  linkedin_jobs, jobs_search_api, and any future sources added to the pipeline.
  No source is bypassed. The LLM handles aggregator detection uniformly
  regardless of where a job was fetched from.

Cost:
  ~1 Haiku call per session regardless of job count.
  Input: ~30 chars x N URLs ~ 1,400 tokens for 150 jobs.
  Output: filtered ID list ~ 400 tokens.
  Total: ~$0.003 / 0.25 per session. Negligible.

LangSmith tracing:
  - @traceable on run_url_pruner for full session-level visibility
  - Logs: input count, kept count, dropped count, dropped job IDs
  - All LLM inputs/outputs captured via LangSmith SDK

Input  (from SessionState): raw_jobs[]
Output (to SessionState)  : raw_jobs[] (filtered -- only quality-sourced jobs)
"""

from __future__ import annotations

import asyncio
import time
import json
import logging
import re

import anthropic
from langsmith import traceable
from tenacity import retry, stop_after_attempt, wait_exponential

from core.prompts.pruner_prompts import URL_PRUNER_PROMPT
from core.config.config_loader import cfg
from core.state.session_state import RawJob, SessionState

logger = logging.getLogger(__name__)

# -- Config ---------------------------------------------------------------------
# LLM parameters loaded from core/config/llm_config.yaml -- edit there, not here.

_CFG = cfg.url_pruner


# -- LLM call -----------------------------------------------------------------

@traceable(
    name="url-pruner-llm",
    run_type="llm",
    metadata={"model": cfg.url_pruner.model, "agent": "url_pruner"},
)
@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(min=1, max=6),
    reraise=False,
)
async def _classify_urls(jobs: list[RawJob]) -> tuple[list[str], dict]:
    """
    Send all job URLs to Claude in one call.
    Returns (kept_ids, token_counts) where token_counts = {"input": int, "output": int}.
    Falls back to (all_ids, empty_counts) on any error.
    """
    _fallback_tokens: dict = {"input": 0, "output": 0}

    if not jobs:
        return [], _fallback_tokens

    url_lines = "\n".join(
        f'{{"job_id": "{job.job_id}", "apply_url": "{job.apply_url}"}}'
        for job in sorted(jobs, key=lambda j: j.job_id)
    )

    prompt = URL_PRUNER_PROMPT.replace("{job_url_list}", url_lines)

    logger.info(
        f"[url_pruner] Sending {len(jobs)} URLs to Claude {_CFG.model} "
        f"(temp={_CFG.temperature}) for quality classification"
    )

    try:
        client = anthropic.AsyncAnthropic()
        response = await asyncio.wait_for(
            client.messages.create(
                model       = _CFG.model,
                max_tokens  = _CFG.max_tokens,
                temperature = _CFG.temperature,
                messages    = [{"role": "user", "content": prompt}],
            ),
            timeout=_CFG.timeout_secs,
        )

        # Capture token usage
        usage   = getattr(response, "usage", None)
        tok_counts = {
            "input":  getattr(usage, "input_tokens",  0) if usage else 0,
            "output": getattr(usage, "output_tokens", 0) if usage else 0,
        }

        raw = response.content[0].text.strip()
        logger.debug(f"[url_pruner] Raw LLM response: {raw[:300]}")

        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$",           "", raw, flags=re.MULTILINE)

        kept_ids = json.loads(raw)

        if not isinstance(kept_ids, list):
            logger.warning(
                f"[url_pruner] Unexpected response type: {type(kept_ids)} -- "
                f"keeping all jobs as fallback"
            )
            return [job.job_id for job in jobs], tok_counts

        valid_ids = {job.job_id for job in jobs}
        kept_ids  = [jid for jid in kept_ids if jid in valid_ids]

        logger.info(
            f"[url_pruner] Classification complete: "
            f"{len(kept_ids)}/{len(jobs)} jobs kept | "
            f"dropped={len(jobs) - len(kept_ids)}"
        )
        return kept_ids, tok_counts

    except asyncio.TimeoutError:
        logger.error(f"[url_pruner] Timed out after {_CFG.timeout_secs}s -- fallback")
        return [job.job_id for job in jobs], _fallback_tokens

    except json.JSONDecodeError as e:
        logger.error(f"[url_pruner] JSON parse failed ({e}) -- fallback")
        return [job.job_id for job in jobs], _fallback_tokens

    except Exception as e:
        logger.error(f"[url_pruner] Unexpected error ({e}) -- fallback")
        return [job.job_id for job in jobs], _fallback_tokens


# -- Main agent function -------------------------------------------------------

@traceable(
    name="agent-5-url-pruner",
    run_type="chain",
    metadata={"agent": "url_pruner"},
)
async def run_url_pruner(state: SessionState) -> SessionState:
    """
    Agent 5 -- URL Pruner.

    Single-pass LLM filter: URL quality only.
    Drops aggregator/low-quality URLs, keeps direct company pages and
    trusted ATS domains. Eligibility filtering (experience gap + function
    mismatch) is handled downstream by the ranker prompt Step 0.

    All sources are classified uniformly -- no source bypasses the LLM check.

    On any LLM failure: returns all jobs unchanged -- pipeline continues.
    """
    state.current_agent = "url_pruner"
    _t0 = time.perf_counter()

    if not state.raw_jobs:
        logger.warning("[url_pruner] No raw jobs to prune -- skipping")
        return state

    input_count = len(state.raw_jobs)

    # Log per-source volume for LangSmith visibility
    source_counts: dict[str, int] = {}
    for j in state.raw_jobs:
        source_counts[j.source] = source_counts.get(j.source, 0) + 1

    logger.info(
        f"[url_pruner] URL quality pass -- session_id={state.session_id} | "
        f"total={input_count} | sources={source_counts}"
    )

    # -- URL Quality Filter: all sources ----------------------------------------
    # Every job, regardless of source, goes through the LLM URL classifier.
    # The LLM uses world knowledge to distinguish direct company/ATS URLs from
    # aggregator redirect URLs -- correctly handling the long tail of company
    # career page domains that cannot be enumerated in a hardcoded list.
    kept_ids, token_counts = await _classify_urls(state.raw_jobs)
    kept_set  = set(kept_ids)
    kept_jobs = [j for j in state.raw_jobs if j.job_id in kept_set]
    dropped   = [j for j in state.raw_jobs if j.job_id not in kept_set]

    if dropped:
        logger.info(f"[url_pruner] Dropped {len(dropped)} low-quality URLs:")
        for job in dropped:
            logger.info(
                f"  [url_pruner:drop] '{job.title}' @ {job.company} | "
                f"source={job.source} | url={job.apply_url[:80]}"
            )

    state.raw_jobs = kept_jobs

    # -- Record observability metrics -----------------------------------------
    elapsed = round(time.perf_counter() - _t0, 2)
    state.agent_metrics["url_pruner"] = {
        "model":         _CFG.model,
        "input_tokens":  token_counts.get("input",  0),
        "output_tokens": token_counts.get("output", 0),
        "llm_calls":     1,
        "latency_secs":  elapsed,
        "jobs_in":       input_count,
        "jobs_kept":     len(kept_jobs),
        "jobs_dropped":  len(dropped),
        "sources":       source_counts,
    }
    logger.info(
        f"[url_pruner] Complete -- kept={len(kept_jobs)} | dropped={len(dropped)} | "
        f"tokens={token_counts.get('input',0)}in/{token_counts.get('output',0)}out | "
        f"latency={elapsed}s"
    )
    return state


# -- LangGraph node wrapper ----------------------------------------------------

async def node_url_pruner(state: dict) -> dict:
    """
    LangGraph async node wrapper for the URL pruner agent.

    Must be async -- graph.ainvoke() runs inside an event loop already.
    Using asyncio.run() inside a node called from ainvoke raises
    'RuntimeError: This event loop is already running'.
    """
    session = SessionState(**state)

    if not session.raw_jobs:
        logger.warning("[graph] Skipping URL pruner -- no raw jobs")
        return state

    updated = await run_url_pruner(session)
    return updated.model_dump()