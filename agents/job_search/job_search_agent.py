"""
agents/job_search/job_search_agent.py

Agent 3 -- Job Search

Responsibility:
  - Read confirmed_profiles[] + preferences from SessionState
  - Run parallel searches across all sources for each profile
  - Normalize all results into RawJob objects
  - Write raw_jobs[] to SessionState for Agent 4 (ranker)

Execution model:
  asyncio.gather() runs all (profile x source) combinations simultaneously.
  Total wall time ~ slowest single API call (~3-5s) not sum of all calls.

Sources (all on separate free-tier RapidAPI keys -- quotas are independent):
  JSearch          -- Google Jobs aggregator        JSEARCH_API_KEY
  Active Jobs DB   -- Same schema as JSearch        ACTIVE_JOBS_DB_API_KEY
  LinkedIn Jobs    -- LinkedIn listings             LINKEDIN_JOBS_API_KEY
  Techmap          -- 240-country, full JDs         TECHMAP_API_KEY
  Jobs Search API  -- Broad ATS coverage            JOBS_SEARCH_API_KEY
  RemoteOK         -- CONDITIONAL: remote work only

Input  (from SessionState): confirmed_profiles[], preferences
Output (to SessionState)  : raw_jobs[]
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Coroutine

from agents.job_search.normalizer import normalize_jobs
from agents.job_search.sources.jsearch import search_jsearch
from agents.job_search.sources.remoteok import search_remoteok
from agents.job_search.sources.active_jobs_db import search_active_jobs_db
from agents.job_search.sources.linkedin_jobs import search_linkedin_jobs
from agents.job_search.sources.techmap import search_techmap
from agents.job_search.sources.jobs_search_api import search_jobs_search_api
from core.config.config_loader import cfg
from core.state.session_state import RawJob, SessionState

logger = logging.getLogger(__name__)


# -- Title canonicalisation (#12) ---------------------------------------------

def _canonicalize_title(title: str) -> str:
    """
    Map an alias title to its canonical search form using llm_config.yaml
    [title_canonical_map]. Pass-through if the title is not in the map.

    Example: "ML Engineer" → "Machine Learning Engineer" (more API results).
    Domain-agnostic -- the map is configured externally, not hardcoded here.
    """
    canonical = cfg.title_canonical_map.get(title, title)
    if canonical != title:
        logger.info(
            f"[job_search] Title canonicalised: '{title}' → '{canonical}'"
        )
    return canonical


# -- Source activation logic ---------------------------------------------------

def _should_include_remoteok(work_type: str, location: str) -> bool:
    """Activate RemoteOK when user prefers remote work."""
    if work_type == "remote":
        return True
    loc = (location or "").lower()
    return "remote" in loc


# -- Per-profile search --------------------------------------------------------

async def _search_one_profile(
    profile_title: str,
    location:      str,
    work_type:     str,
) -> tuple[list[RawJob], dict[str, dict[str, int]]]:
    """
    Run all active sources for a single profile in parallel.

    Returns (jobs, source_stats) where source_stats maps each attempted
    source name to {"jobs": int, "errors": int}. Never raises --
    individual source failures are logged and counted as errors.
    """
    tasks: list[tuple[str, Coroutine]] = []

    # JSearch -- Google Jobs aggregator (500 calls/month free)
    tasks.append((
        "jsearch", search_jsearch(
            profile_title = profile_title,
            location      = location,
            work_type     = work_type,
            num_pages     = cfg.job_search.num_pages,
        )
    ))

    # Active Jobs DB -- same JSearch schema, separate key (~200 calls/month free)
    if cfg.job_search.active_jobs_db_enabled:
        tasks.append((
            "active_jobs_db", search_active_jobs_db(
                profile_title = profile_title,
                location      = location,
                work_type     = work_type,
                num_pages     = cfg.job_search.active_jobs_db_pages,
            )
        ))

    # LinkedIn Jobs API -- LinkedIn listings (~100 calls/month free)
    if cfg.job_search.linkedin_jobs_enabled:
        tasks.append((
            "linkedin_jobs", search_linkedin_jobs(
                profile_title = profile_title,
                location      = location,
                work_type     = work_type,
                num_pages     = cfg.job_search.linkedin_jobs_pages,
            )
        ))

    # Techmap -- 240-country coverage, full JDs (1,000 jobs/month free)
    if cfg.job_search.techmap_enabled:
        tasks.append((
            "techmap", search_techmap(
                profile_title = profile_title,
                location      = location,
                work_type     = work_type,
                num_pages     = cfg.job_search.techmap_pages,
            )
        ))

    # Jobs Search API -- broad ATS coverage (trial plan free)
    if cfg.job_search.jobs_search_api_enabled:
        tasks.append((
            "jobs_search_api", search_jobs_search_api(
                profile_title = profile_title,
                location      = location,
                work_type     = work_type,
                num_pages     = cfg.job_search.jobs_search_api_pages,
            )
        ))

    # RemoteOK -- activate when work_type == "remote"
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
    # Initialise stats for every attempted source so callers can tell apart
    # "source returned 0 jobs" from "source was disabled / not attempted".
    source_stats: dict[str, dict[str, int]] = {
        s: {"jobs": 0, "errors": 0} for s in sources
    }
    for source, result in zip(sources, results):
        if isinstance(result, Exception):
            source_stats[source]["errors"] += 1
            logger.error(
                f"[job_search] Source '{source}' failed for "
                f"'{profile_title}': {type(result).__name__}: {result}"
            )
            continue

        normalized = normalize_jobs(
            raw_jobs        = result,
            source          = source,
            matched_profile = profile_title,
        )
        source_stats[source]["jobs"] += len(normalized)
        logger.info(
            f"[job_search] {source}: {len(normalized)} jobs "
            f"for '{profile_title}'"
        )
        all_jobs.extend(normalized)

    return all_jobs, source_stats


# -- Main agent function -------------------------------------------------------

async def run_job_search_agent(state: SessionState) -> SessionState:
    """
    Agent 3 -- Job Search Agent.

    Runs parallel searches across all active sources for each
    confirmed profile. Writes raw_jobs[] to session state.

    Also writes per-source contribution counts to
    state.agent_metrics["job_search"] so silent-zero sources (e.g.
    quota-exhausted or mis-keyed APIs) become visible at the session
    level. Domain-agnostic -- no assumptions about resume content or
    role type.

    Returns updated SessionState.
    """
    state.current_agent = "job_search"
    _t0 = time.perf_counter()
    logger.info(
        f"[job_search] Starting -- session_id={state.session_id}, "
        f"profiles={[p.title for p in state.confirmed_profiles]}"
    )

    if not state.confirmed_profiles:
        logger.warning("[job_search] No confirmed profiles -- skipping search")
        state.error = "No confirmed profiles to search for."
        return state

    prefs     = state.preferences
    location  = prefs.location
    work_type = prefs.work_type

    # Run search for all profiles concurrently.
    # Canonicalise each title before querying (#12): "ML Engineer" →
    # "Machine Learning Engineer" to maximise API result coverage.
    profile_tasks = [
        _search_one_profile(
            profile_title = _canonicalize_title(profile.title),
            location      = location,
            work_type     = work_type,
        )
        for profile in state.confirmed_profiles
    ]

    profile_results = await asyncio.gather(
        *profile_tasks, return_exceptions=True
    )

    all_jobs: list[RawJob] = []
    # Aggregate per-source contribution counts across every confirmed
    # profile. Silent-zero sources (configured but contributing nothing)
    # are surfaced as a separate WARNING below.
    source_stats_total: dict[str, dict[str, int]] = {}
    for i, result in enumerate(profile_results):
        profile_title = state.confirmed_profiles[i].title
        if isinstance(result, Exception):
            logger.error(
                f"[job_search] Profile search failed for "
                f"'{profile_title}': {type(result).__name__}: {result}"
            )
            continue
        jobs, src_stats = result
        all_jobs.extend(jobs)
        for src, stats in src_stats.items():
            agg = source_stats_total.setdefault(src, {"jobs": 0, "errors": 0})
            agg["jobs"]   += stats["jobs"]
            agg["errors"] += stats["errors"]

    # -- Dedupe by job_id ----------------------------------------------------
    # Sources occasionally return the same posting on multiple pages, and the
    # same posting may be returned for different confirmed profiles. The
    # ranker has its own (looser) company::title fingerprint dedup, but
    # collapsing exact job_id duplicates here saves Haiku tokens in the URL
    # pruner and produces a cleaner state payload. Domain-agnostic.
    pre_dedup_count = len(all_jobs)
    unique_by_id: dict[str, RawJob] = {}
    for job in all_jobs:
        existing = unique_by_id.get(job.job_id)
        if existing is None:
            unique_by_id[job.job_id] = job
            continue
        # Same job_id appeared again. Keep the longer JD (richer text) and
        # union the matched_profile attribution so we don't lose the fact
        # that this job matched multiple profiles.
        keeper = job if len(job.jd_text or "") > len(existing.jd_text or "") else existing
        existing_profiles = {
            p.strip()
            for p in (existing.matched_profile or "").split("|")
            if p.strip()
        }
        new_profiles = {
            p.strip()
            for p in (job.matched_profile or "").split("|")
            if p.strip()
        }
        keeper.matched_profile = " | ".join(sorted(existing_profiles | new_profiles))
        unique_by_id[job.job_id] = keeper

    deduped_jobs = list(unique_by_id.values())
    dropped = pre_dedup_count - len(deduped_jobs)
    if dropped:
        logger.info(
            f"[job_search] Deduped {dropped} exact-job_id duplicate(s): "
            f"{pre_dedup_count} -> {len(deduped_jobs)}"
        )

    state.raw_jobs = deduped_jobs
    state.error    = None
    all_jobs       = deduped_jobs   # keep local var consistent for the warning below

    # -- Per-source observability --------------------------------------------
    # Surface contribution counts so silent-zero sources (e.g. exhausted
    # quotas, missing keys, schema drift) become visible in MLflow and in
    # the session_metrics payload returned to the API layer.
    elapsed = round(time.perf_counter() - _t0, 2)
    state.agent_metrics["job_search"] = {
        "latency_secs":       elapsed,
        "total_jobs":         len(deduped_jobs),
        "pre_dedup_count":    pre_dedup_count,
        "duplicates_dropped": pre_dedup_count - len(deduped_jobs),
        "source_stats":       source_stats_total,
        "profiles_searched":  [p.title for p in state.confirmed_profiles],
    }

    # One-line per-source summary at INFO so it is visible in the docker logs.
    summary_parts = [
        f"{src}={s['jobs']}" + (f" (errors={s['errors']})" if s["errors"] else "")
        for src, s in source_stats_total.items()
    ]
    silent_zero_sources = [
        src for src, s in source_stats_total.items()
        if s["jobs"] == 0 and s["errors"] == 0
    ]

    if not all_jobs:
        # All upstream sources returned zero jobs. Most common causes:
        #   1. RapidAPI free-tier quota exhausted for the month
        #   2. Confirmed profiles too narrow for the configured location
        #   3. All sources transiently unavailable
        # We don't fail the pipeline -- downstream nodes skip on empty raw_jobs,
        # finalise marks results_ready=True, and the API surfaces an empty-state
        # message via the /status and /results endpoints.
        logger.warning(
            f"[job_search] ZERO jobs returned across "
            f"{len(state.confirmed_profiles)} confirmed profile(s). "
            f"Per-source: {', '.join(summary_parts)}. "
            f"Likely causes: API quota exhausted, narrow profile titles, or "
            f"transient source outage."
        )
    else:
        logger.info(
            f"[job_search] Complete -- {len(all_jobs)} unique jobs across "
            f"{len(state.confirmed_profiles)} profile(s) | "
            f"per-source: {', '.join(summary_parts)} | latency={elapsed}s"
        )
        if silent_zero_sources:
            # These sources were configured + attempted but contributed 0 jobs
            # without raising errors -- usually a missing/invalid API key or
            # an exhausted quota. Surface as a single WARNING so it is easy
            # to spot during a demo.
            logger.warning(
                f"[job_search] Silent-zero source(s): {silent_zero_sources}. "
                f"Check API keys / quotas in .env -- these contributed nothing "
                f"to this session."
            )

    return state


# -- LangGraph node wrapper ----------------------------------------------------

async def node_job_search(state: dict) -> dict:
    """
    LangGraph node wrapper for the job search agent.
    Defined as async so LangGraph runs it directly in the
    existing event loop -- avoids asyncio.run() conflict
    with FastAPI/uvicorn's running loop.
    """
    session = SessionState(**state)

    if session.error or not session.confirmed_profiles:
        logger.warning("[graph] Skipping job search -- no confirmed profiles")
        return state

    updated = await run_job_search_agent(session)
    return updated.model_dump()
