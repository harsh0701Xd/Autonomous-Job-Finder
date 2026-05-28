"""
core/url_validator.py

Lightweight async HTTP-based URL validator.

Replaces the LLM URL Pruner (Agent 5) with a deterministic, zero-cost check.
No hardcoded domain lists. No candidate data. Resume-agnostic by design.

Decision logic (two rules, applied in order):
  1. Dead link  -- HTTP response is 404 or 410 → drop.
  2. Aggregator -- following all redirects, the final URL lands on a
                   different domain than the original URL → drop.
                   (Aggregators redirect to the real job page or to nothing.
                   Direct company/ATS pages do not redirect to a foreign domain.)
  3. Everything else → keep.
     Includes: 200 OK, 403 Forbidden, 429 Too Many Requests, timeouts,
     network errors. Benefit of the doubt — we would rather pass a marginally
     bad URL to HyDE than silently drop a real job.

Properties:
  - No hardcoding. No domain blocklist or allowlist.
  - Resume-agnostic. Only the apply_url string is examined — no candidate data.
  - Fully async (httpx). All URLs are checked concurrently up to MAX_CONCURRENCY.
  - Fail-open. Any unexpected error keeps the job in the pipeline.

Limitations:
  Aggregators that serve their own 200 OK page (no redirect) are not detected
  here. They are caught downstream by HyDE's semantic similarity floor, which
  drops jobs whose JD text is semantically dissimilar to the candidate's profile.
"""

from __future__ import annotations

import asyncio
import logging
import time
from urllib.parse import urlparse

import httpx

from core.state.session_state import RawJob, SessionState

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────

TIMEOUT_SECS    = 8      # per-URL request timeout
MAX_CONCURRENCY = 20     # max simultaneous HTTP connections
DROP_STATUSES   = {404, 410}   # unambiguous "not found" / "gone"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; job-pipeline-url-check/1.0)"
}


# ── Per-URL check ─────────────────────────────────────────────────────────────

async def _check_url(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    job: RawJob,
) -> tuple[str, bool, str]:
    """
    Check one URL. Returns (job_id, is_valid, reason).

    is_valid=True  → keep this job in the pipeline
    is_valid=False → drop this job (dead link or aggregator redirect)
    """
    url = job.apply_url
    original_host = _normalise_host(urlparse(url).netloc)

    async with sem:
        try:
            resp = await client.head(
                url,
                follow_redirects=True,
                timeout=TIMEOUT_SECS,
            )

            # Rule 1: Dead link
            if resp.status_code in DROP_STATUSES:
                return job.job_id, False, f"dead_link (HTTP {resp.status_code})"

            # Rule 2: Aggregator redirect
            # httpx exposes the final URL via resp.url after following redirects
            final_host = _normalise_host(urlparse(str(resp.url)).netloc)
            if original_host and final_host and original_host != final_host:
                return (
                    job.job_id,
                    False,
                    f"aggregator_redirect ({original_host} → {final_host})",
                )

            return job.job_id, True, f"ok (HTTP {resp.status_code})"

        except httpx.TimeoutException:
            # Slow page — give benefit of the doubt
            return job.job_id, True, "timeout (kept)"

        except httpx.ConnectError:
            # DNS failure / connection refused — could be transient
            return job.job_id, True, "connection_error (kept)"

        except Exception as e:
            # Catch-all: never drop due to an unexpected error
            return job.job_id, True, f"check_error (kept): {type(e).__name__}"


def _normalise_host(netloc: str) -> str:
    """Strip port and leading 'www.' for apples-to-apples domain comparison."""
    host = netloc.split(":")[0].lower()          # drop port
    if host.startswith("www."):
        host = host[4:]
    return host


# ── Batch validator ───────────────────────────────────────────────────────────

async def validate_urls(jobs: list[RawJob]) -> list[tuple[str, bool, str]]:
    """
    Validate all apply_urls concurrently.
    Returns list of (job_id, is_valid, reason) tuples.
    """
    if not jobs:
        return []

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    # Shared httpx async client — connection pooling, redirects enabled
    async with httpx.AsyncClient(
        headers=_HEADERS,
        verify=False,      # skip SSL verification to avoid cert errors on career pages
        limits=httpx.Limits(max_connections=MAX_CONCURRENCY),
    ) as client:
        tasks = [_check_url(sem, client, job) for job in jobs]
        results = await asyncio.gather(*tasks)

    return list(results)


# ── LangGraph node ────────────────────────────────────────────────────────────

async def node_url_validate(state: dict) -> dict:
    """
    LangGraph async node: URL Validator.

    Replaces the LLM URL Pruner. Drops dead links and aggregator-redirect URLs
    using HTTP behaviour only. No LLM, no hardcoded domains, no candidate data.

    On any unexpected failure: keeps all jobs (fail-open).
    Pipeline audit label is preserved as 'url_pruner' for metric continuity.
    """
    session = SessionState(**state)

    if not session.raw_jobs:
        logger.warning("[url_validate] No raw jobs — skipping")
        return state

    _t0 = time.perf_counter()
    input_count = len(session.raw_jobs)

    source_counts: dict[str, int] = {}
    for j in session.raw_jobs:
        source_counts[j.source] = source_counts.get(j.source, 0) + 1

    logger.info(
        f"[url_validate] Checking {input_count} URLs | sources={source_counts}"
    )

    # Run async HTTP checks
    try:
        results = await validate_urls(session.raw_jobs)
    except Exception as e:
        logger.error(f"[url_validate] Unexpected batch error ({e}) — keeping all jobs")
        results = [(j.job_id, True, "batch_error (kept)") for j in session.raw_jobs]

    # Apply results
    drop_map: dict[str, str] = {
        jid: reason for jid, valid, reason in results if not valid
    }
    kept_jobs = [j for j in session.raw_jobs if j.job_id not in drop_map]
    dropped   = [j for j in session.raw_jobs if j.job_id in drop_map]

    # Log drops
    for job in dropped:
        reason = drop_map[job.job_id]
        logger.info(
            f"  [url_validate:drop] '{job.title}' @ {job.company} | "
            f"source={job.source} | reason={reason} | url={job.apply_url[:80]}"
        )
        # Update pipeline audit — label preserved as 'url_pruner' for
        # backward compatibility with funnel metrics in node_finalise.
        if job.job_id in session.pipeline_audit:
            session.pipeline_audit[job.job_id]["status"]     = "dropped"
            session.pipeline_audit[job.job_id]["dropped_at"] = "url_pruner"
            session.pipeline_audit[job.job_id]["drop_reason"] = reason

    session.raw_jobs = kept_jobs

    elapsed = round(time.perf_counter() - _t0, 2)
    session.agent_metrics["url_pruner"] = {
        "model":         None,   # no LLM
        "input_tokens":  0,
        "output_tokens": 0,
        "llm_calls":     0,
        "latency_secs":  elapsed,
        "jobs_in":       input_count,
        "jobs_kept":     len(kept_jobs),
        "jobs_dropped":  len(dropped),
        "sources":       source_counts,
    }

    logger.info(
        f"[url_validate] Complete — kept={len(kept_jobs)} | "
        f"dropped={len(dropped)} | latency={elapsed}s"
    )
    return session.model_dump()
