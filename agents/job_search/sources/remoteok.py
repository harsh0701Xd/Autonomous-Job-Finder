"""
agents/job_search/sources/remoteok.py

RemoteOK public API client.
No API key required. Returns remote-only tech jobs globally.

API: https://remoteok.com/remote-jobs.json
"""

from __future__ import annotations

import logging

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

REMOTEOK_URL    = "https://remoteok.com/remote-jobs.json"
DEFAULT_TIMEOUT = 20.0
MAX_RESULTS     = 15   # cap to avoid flooding the pipeline


def _matches_profile(job: dict, profile_title: str) -> bool:
    """
    Simple keyword match — RemoteOK has no server-side search,
    so we filter client-side on title and tags.
    """
    title_lower   = (job.get("position") or "").lower()
    tags          = [t.lower() for t in (job.get("tags") or [])]
    query_words   = profile_title.lower().split()

    return any(
        word in title_lower or any(word in tag for tag in tags)
        for word in query_words
    )


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    reraise=False,
)
async def search_remoteok(
    profile_title: str,
    max_results:   int = MAX_RESULTS,
) -> list[dict]:
    """
    Fetch and filter RemoteOK jobs matching a profile title.

    RemoteOK returns all recent jobs — we filter locally by
    matching profile keywords against job title and tags.

    Args:
        profile_title: e.g. "ML Engineer", "Data Scientist"
        max_results:   maximum number of filtered results to return

    Returns:
        List of raw job dicts. Empty list on any error.
    """
    logger.info(f"[remoteok] Fetching jobs for '{profile_title}'")

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
