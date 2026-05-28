"""
agents/job_search/job_search_agent.py

Agent 3 -- Job Search

Responsibility:
  - Read confirmed_profiles[] + preferences from SessionState
  - Run parallel searches across all active sources for each profile
  - Normalize all results into RawJob objects
  - Write raw_jobs[] to SessionState for downstream agents

Execution model:
  asyncio.gather() runs all (profile × source) combinations simultaneously.
  Total wall time ≈ slowest single API call (~3-5s), not sum of all calls.

Sources are registered in agents/job_search/registry.py (SOURCES list).
Each source exposes search() + normalize() + SOURCE_SPEC.
Enable/disable sources and set page counts in llm_config.yaml [job_search.sources].
API keys are set per SOURCE_SPEC["env_key"] in your .env file.

Input  (from SessionState): confirmed_profiles[], preferences
Output (to SessionState)  : raw_jobs[]
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Coroutine, Optional

from agents.job_search.registry import SOURCES, validate_source_keys
from core.config.config_loader import cfg
from core.state.session_state import RawJob, SessionState

logger = logging.getLogger(__name__)

# Build normalizer dispatch table from registry at import time.
# Maps source name -> normalize() function so the agent never imports
# individual source modules directly.
_NORMALIZERS = {m.SOURCE_SPEC["name"]: m.normalize for m in SOURCES}

# Warn at startup about enabled sources with missing API keys.
validate_source_keys()


# -- Title canonicalisation ----------------------------------------------------

def _canonicalize_title(title: str) -> str:
    """
    Map an alias title to its canonical search form using llm_config.yaml
    [title_canonical_map]. Pass-through if the title is not in the map.

    Example: "ML Engineer" -> "Machine Learning Engineer" (more API results).
    Domain-agnostic -- the map is configured externally, not hardcoded here.
    """
    canonical = cfg.title_canonical_map.get(title, title)
    if canonical != title:
        logger.info(
            f"[job_search] Title canonicalised: '{title}' -> '{canonical}'"
        )
    return canonical


# -- Normalize helper ----------------------------------------------------------

def _normalize_batch(
    raw_jobs:        list[dict],
    source:          str,
    matched_profile: str,
) -> list[RawJob]:
    """
    Normalize a list of raw job dicts from a named source.

    Dispatches to the source's own normalize() function via _NORMALIZERS.
    Jobs failing any quality gate return None and are excluded.
    Logs a per-source summary for observability.
    """
    normalizer = _NORMALIZERS.get(source)
    if not normalizer:
        logger.error(f"[job_search] No normalizer registered for source '{source}'")
        return []

    results = []
    dropped = 0
    for raw in raw_jobs:
        job = normalizer(raw, matched_profile)
        if job:
            results.append(job)
        else:
            dropped += 1

    logger.info(
        f"[normalizer] {source}: "
        f"{len(results)} passed quality gates, {dropped} dropped | "
        f"profile='{matched_profile}'"
    )
    return results


# -- Per-profile search --------------------------------------------------------

async def _search_one_profile(
    profile_title: str,
    location:      Optional[str],
) -> tuple[list[RawJob], dict[str, dict[str, int]]]:
    """
    Run all active sources for a single profile in parallel.

    Iterates the SOURCES registry, skipping sources that are disabled in
    llm_config.yaml. Each enabled source's search() is launched as a
    concurrent coroutine. Results are normalized and collected.

    Returns (jobs, source_stats) where source_stats maps each attempted
    source name to {"jobs": int, "errors": int}. Never raises --
    individual source failures are logged and counted as errors.
    """
    tasks: list[tuple[str, Coroutine]] = []

    for source_module in SOURCES:
        spec       = source_module.SOURCE_SPEC
        name       = spec["name"]
        always_on  = spec.get("always_on", False)
        source_cfg = cfg.job_search.source(name)

        # Skip disabled sources (unless always_on)
        if not always_on and not source_cfg.enabled:
            continue

        pages = source_cfg.pages if not always_on else cfg.job_search.num_pages
        loc   = (location or "") if spec.get("requires_location", True) else ""

        tasks.append((
            name,
            source_module.search(profile_title, loc, pages),
        ))

    if not tasks:
        logger.warning(
            f"[job_search] No sources enabled for '{profile_title}'. "
            f"Enable at least one source in llm_config.yaml [job_search.sources]."
        )
        return [], {}

    sources, coroutines = zip(*tasks)
    results = await asyncio.gather(*coroutines, return_exceptions=True)

    all_jobs: list[RawJob] = []
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

        normalized = _normalize_batch(
            raw_jobs        = result,
            source          = source,
            matched_profile = profile_title,
        )
        source_stats[source]["jobs"] += len(normalized)
        logger.info(
            f"[job_search] {source}: {len(normalized)} jobs for '{profile_title}'"
        )
        all_jobs.extend(normalized)

    return all_jobs, source_stats


# -- Main agent function -------------------------------------------------------

async def run_job_search_agent(state: SessionState) -> SessionState:
    """
    Agent 3 -- Job Search Agent.

    Runs parallel searches across all active sources for each confirmed
    profile. Writes raw_jobs[] to session state.

    Also writes per-source contribution counts to
    state.agent_metrics["job_search"] so silent-zero sources (e.g.
    quota-exhausted or mis-keyed APIs) become visible at the session level.
    Domain-agnostic -- no assumptions about resume content or role type.

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

    prefs    = state.preferences
    location = prefs.location

    # Build search tasks: one per (confirmed profile + each search variant).
    # Canonicalise each title before querying: "ML Engineer" ->
    # "Machine Learning Engineer" to maximise API result coverage.
    # Variants are searched with the parent profile title as matched_profile
    # so downstream attribution stays correct.
    search_tasks: list[tuple[str, str]] = []  # (search_title, matched_profile_title)
    for profile in state.confirmed_profiles:
        canonical = _canonicalize_title(profile.title)
        search_tasks.append((canonical, profile.title))
        for variant in (profile.search_variants or []):
            variant_canonical = _canonicalize_title(variant)
            search_tasks.append((variant_canonical, profile.title))

    profile_tasks = [
        _search_one_profile(
            profile_title = search_title,
            location      = location,
        )
        for search_title, _ in search_tasks
    ]

    profile_results = await asyncio.gather(
        *profile_tasks, return_exceptions=True
    )

    all_jobs: list[RawJob] = []
    # Aggregate per-source contribution counts across every confirmed
    # profile and variant. Silent-zero sources (configured but contributing
    # nothing) are surfaced as a separate WARNING below.
    source_stats_total: dict[str, dict[str, int]] = {}
    for i, result in enumerate(profile_results):
        search_title, matched_profile_title = search_tasks[i]
        if isinstance(result, Exception):
            logger.error(
                f"[job_search] Profile search failed for "
                f"'{search_title}': {type(result).__name__}: {result}"
            )
            continue
        jobs, src_stats = result
        # Override matched_profile on each job so attribution uses the parent
        # confirmed profile title, not the variant title.
        for job in jobs:
            job.matched_profile = matched_profile_title
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

    # -- Initialize pipeline audit for every job entering the pipeline ----------
    # Done here (after exact job_id dedup) so each unique job gets exactly one
    # entry. All score/filter fields start as None and are filled in by
    # downstream agents (url_pruner, dedup, hyde, ranker).
    for job in deduped_jobs:
        state.pipeline_audit[job.job_id] = {
            # Identity
            "job_id":          job.job_id,
            "title":           job.title or "",
            "company":         job.company or "",
            "location":        job.location or "",
            "work_type":       job.work_type or "",
            "source":          job.source or "",
            "apply_url":       (job.apply_url or "")[:120],
            "matched_profile": job.matched_profile or "",
            "jd_word_count":   len((job.jd_text or "").split()),
            "posted_date":     job.posted_date.isoformat() if job.posted_date else None,
            # Journey tracking
            "status":          "passed",   # updated to "dropped" at each stage
            "dropped_at":      None,       # stage name where dropped
            "drop_reason":     None,       # human-readable reason
            # HyDE prefilter (filled by hyde_agent)
            "jd1_emb_score":   None,
            "jd2_emb_score":   None,
            "hyde_section":    None,       # "S1" | "S2" | "dropped"
            # Ranker scores (filled by ranker_agent)
            "fit_score":       None,
            "title_relevance": None,
            "experience_score":  None,
            "skill_match_score": None,
            "domain_score":    None,
            "education_score": None,
            "sparse_jd":       False,
            "final_rank":      None,       # 1-based rank in final output
        }

    # -- Per-source observability --------------------------------------------
    elapsed = round(time.perf_counter() - _t0, 2)
    confirmed_titles = [p.title for p in state.confirmed_profiles]
    variant_searches = [st for st, mp in search_tasks if st not in confirmed_titles]

    state.agent_metrics["job_search"] = {
        "latency_secs":       elapsed,
        "total_jobs":         len(deduped_jobs),
        "pre_dedup_count":    pre_dedup_count,
        "duplicates_dropped": pre_dedup_count - len(deduped_jobs),
        "source_stats":       source_stats_total,
        "profiles_searched":  confirmed_titles,
        "variant_searches":   variant_searches,
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
        logger.warning(
            f"[job_search] ZERO jobs returned across "
            f"{len(state.confirmed_profiles)} confirmed profile(s). "
            f"Per-source: {', '.join(summary_parts) or 'no sources attempted'}. "
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
    with FastAPI/uvicorn running loop.
    """
    session = SessionState(**state)

    if session.error or not session.confirmed_profiles:
        logger.warning("[graph] Skipping job search -- no confirmed profiles")
        return state

    updated = await run_job_search_agent(session)
    return updated.model_dump()
