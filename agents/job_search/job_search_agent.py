"""
agents/job_search/job_search_agent.py

Agent 3 — Job Search

Responsibility:
  - Read confirmed_profiles[] + preferences from SessionState
  - Run parallel searches across all sources for each profile
  - Normalize all results into RawJob objects
  - Write raw_jobs[] to SessionState for Agent 4 (ranker)

Execution model:
  asyncio.gather() runs all (profile × source) combinations simultaneously.
  2 profiles × 2 sources = 4 concurrent API calls.
  Total wall time ≈ slowest single API call (~3-5s) not sum of all calls.

Input  (from SessionState): confirmed_profiles[], preferences
Output (to SessionState)  : raw_jobs[]
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Coroutine

from agents.job_search.normalizer import normalize_jobs
from agents.job_search.sources.jsearch import search_jsearch
from agents.job_search.sources.remoteok import search_remoteok
from core.state.session_state import RawJob, SessionState

logger = logging.getLogger(__name__)

# Max raw jobs to carry forward to Agent 4
# Beyond this, we truncate lowest-relevance results
MAX_RAW_JOBS = 60


# ── Source activation logic ───────────────────────────────────────────────────

def _should_include_remoteok(work_type: str, location: str) -> bool:
    """Activate RemoteOK when user prefers remote work."""
    return (
        work_type == "remote"
        or "remote" in location.lower()
    )


# ── Per-profile search ────────────────────────────────────────────────────────

async def _search_one_profile(
    profile_title: str,
    location:      str,
    work_type:     str,
) -> list[RawJob]:
    """
    Run all active sources for a single profile in parallel.
    Returns normalized RawJob list. Never raises — errors are logged.
    """
    tasks: list[tuple[str, Coroutine]] = [
        ("jsearch", search_jsearch(
            profile_title = profile_title,
            location      = location,
            work_type     = work_type,
            num_pages     = 2,
        )),
    ]

    if _should_include_remoteok(work_type, location):
        tasks.append((
            "remoteok", search_remoteok(
                profile_title = profile_title,
                max_results   = 15,
            )
        ))

    # Run all sources concurrently
    sources, coroutines = zip(*tasks)
    results = await asyncio.gather(*coroutines, return_exceptions=True)

    all_jobs: list[RawJob] = []
    for source, result in zip(sources, results):
        if isinstance(result, Exception):
            logger.error(
                f"[job_search] Source '{source}' failed for "
                f"'{profile_title}': {result}"
            )
            continue

        normalized = normalize_jobs(
            raw_jobs        = result,
            source          = source,
            matched_profile = profile_title,
        )
        logger.info(
            f"[job_search] {source}: {len(normalized)} jobs "
            f"for '{profile_title}'"
        )
        all_jobs.extend(normalized)

    return all_jobs


# ── Main agent function ───────────────────────────────────────────────────────

async def run_job_search_agent(state: SessionState) -> SessionState:
    """
    Agent 3 — Job Search Agent.

    Runs parallel searches across all active sources for each
    confirmed profile. Writes raw_jobs[] to session state.

    Returns updated SessionState.
    """
    state.current_agent = "job_search"
    logger.info(
        f"[job_search] Starting — session_id={state.session_id}, "
        f"profiles={[p.title for p in state.confirmed_profiles]}"
    )

    if not state.confirmed_profiles:
        logger.warning("[job_search] No confirmed profiles — skipping search")
        state.error = "No confirmed profiles to search for."
        return state

    prefs    = state.preferences
    location = prefs.location
    work_type = prefs.work_type

    # Run search for all profiles concurrently
    profile_tasks = [
        _search_one_profile(
            profile_title = profile.title,
            location      = location,
            work_type     = work_type,
        )
        for profile in state.confirmed_profiles
    ]

    profile_results = await asyncio.gather(
        *profile_tasks, return_exceptions=True
    )

    # Collect all results
    all_jobs: list[RawJob] = []
    for i, result in enumerate(profile_results):
        profile_title = state.confirmed_profiles[i].title
        if isinstance(result, Exception):
            logger.error(
                f"[job_search] Profile search failed for "
                f"'{profile_title}': {result}"
            )
            continue
        all_jobs.extend(result)

    # Cap raw jobs before passing to ranker
    if len(all_jobs) > MAX_RAW_JOBS:
        logger.info(
            f"[job_search] Capping {len(all_jobs)} → {MAX_RAW_JOBS} raw jobs"
        )
        all_jobs = all_jobs[:MAX_RAW_JOBS]

    state.raw_jobs = all_jobs
    state.error    = None

    logger.info(
        f"[job_search] Complete — {len(all_jobs)} raw jobs collected "
        f"across {len(state.confirmed_profiles)} profiles"
    )

    return state


# ── LangGraph node wrapper ────────────────────────────────────────────────────

async def node_job_search(state: dict) -> dict:
    """
    LangGraph node wrapper for the job search agent.
    Defined as async so LangGraph runs it directly in the
    existing event loop — avoids asyncio.run() conflict
    with FastAPI/uvicorn's running loop.
    """
    session = SessionState(**state)

    if session.error or not session.confirmed_profiles:
        logger.warning("[graph] Skipping job search — no confirmed profiles")
        return state

    updated = await run_job_search_agent(session)
    return updated.model_dump()