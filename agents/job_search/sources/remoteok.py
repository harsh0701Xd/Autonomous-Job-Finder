"""
agents/job_search/sources/remoteok.py

RemoteOK public API client.
No API key required. Returns remote-only tech jobs globally.

API: https://remoteok.com/remote-jobs.json
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from agents.job_search.quality_gates import (
    clean_text, is_english, is_jd_sufficient, make_job_id, parse_date,
)
from core.state.session_state import RawJob

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source descriptor — read by registry.py to build the SOURCES list.
# ---------------------------------------------------------------------------
SOURCE_SPEC = {
    "name":              "remoteok",
    "env_key":           None,    # no API key required
    "always_on":         False,   # controlled via llm_config.yaml sources.remoteok
    "requires_location": False,   # global remote feed — location not used in search
}

REMOTEOK_URL    = "https://remoteok.com/remote-jobs.json"
DEFAULT_TIMEOUT = 20.0
_RESULTS_PER_PAGE = 15   # effective jobs per "page" for this pageless source


def _matches_profile(job: dict, profile_title: str) -> bool:
    """
    Simple keyword match — RemoteOK has no server-side search,
    so we filter client-side on title and tags.
    """
    title_lower = (job.get("position") or "").lower()
    tags        = [t.lower() for t in (job.get("tags") or [])]
    query_words = profile_title.lower().split()

    return any(
        word in title_lower or any(word in tag for tag in tags)
        for word in query_words
    )


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    reraise=False,
)
async def search(
    profile_title: str,
    location:      str,
    pages:         int,
    **kwargs,
) -> list[dict]:
    """
    Fetch and filter RemoteOK jobs matching a profile title.

    RemoteOK returns all recent jobs — we filter locally by matching
    profile keywords against job title and tags. `location` is ignored
    (RemoteOK is a global remote-only feed). `pages` scales max_results:
    pages=1 → 15 results, pages=2 → 30, etc.

    Returns list of raw job dicts. Empty list on any error.
    """
    max_results = max(pages, 1) * _RESULTS_PER_PAGE
    logger.info(f"[remoteok] Fetching jobs for '{profile_title}' (max_results={max_results})")

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            # RemoteOK requires a User-Agent header or returns 403
            response = await client.get(
                REMOTEOK_URL,
                headers={"User-Agent": "AutonomousJobFinder/1.0"},
                follow_redirects=True,
            )
            response.raise_for_status()
            all_jobs = response.json()

        # First item is metadata — skip it
        jobs = [j for j in all_jobs if isinstance(j, dict) and j.get("position")]

        # Filter by profile keywords
        matched = [j for j in jobs if _matches_profile(j, profile_title)]

        result = matched[:max_results]
        logger.info(
            f"[remoteok] {len(jobs)} total jobs, "
            f"{len(matched)} matched '{profile_title}', "
            f"returning {len(result)}"
        )
        return result

    except httpx.TimeoutException:
        logger.error("[remoteok] Request timed out")
        return []
    except Exception as e:
        logger.error(f"[remoteok] Error fetching jobs: {e}")
        return []


def normalize(raw: dict, matched_profile: str) -> Optional[RawJob]:
    """
    Normalize a RemoteOK job dict into RawJob.

    RemoteOK field reference:
      id, position, company, url, description,
      date, salary_min, salary_max, tags, location
    """
    try:
        job_id    = str(raw.get("id")   or "")
        title     = raw.get("position") or ""
        company   = raw.get("company")  or ""
        apply_url = raw.get("url")      or ""

        if not job_id or not title or not apply_url:
            return None

        raw_jd  = raw.get("description") or ""
        jd_text = clean_text(raw_jd)

        if not is_jd_sufficient(jd_text, title, company):
            return None
        if not is_english(jd_text, title, company):
            return None

        return RawJob(
            job_id          = make_job_id("remoteok", job_id),
            title           = title,
            company         = company,
            location        = raw.get("location") or "Remote",
            work_type       = "remote",
            jd_text         = jd_text,
            apply_url       = apply_url,
            source          = "remoteok",
            posted_date     = parse_date(raw.get("date")),
            matched_profile = matched_profile,
        )

    except Exception as e:
        logger.warning(f"[remoteok] Failed to normalize job: {e}")
        return None
