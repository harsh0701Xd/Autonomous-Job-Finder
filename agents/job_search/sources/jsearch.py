"""
agents/job_search/sources/jsearch.py

JSearch API client (via RapidAPI).
Aggregates Indeed, LinkedIn, Glassdoor, ZipRecruiter globally.

API docs: https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

JSEARCH_BASE    = "https://jsearch.p.rapidapi.com"
JSEARCH_HOST    = "jsearch.p.rapidapi.com"
DEFAULT_PAGES   = 2       # 10 results per page → 20 results per call
DEFAULT_TIMEOUT = 20.0    # seconds


def _get_headers() -> dict:
    key = os.getenv("JSEARCH_API_KEY")
    if not key:
        raise ValueError("JSEARCH_API_KEY not set in environment")
    return {
        "X-RapidAPI-Key":  key,
        "X-RapidAPI-Host": JSEARCH_HOST,
    }


def _build_query(
    profile_title: str,
    location:      str,
    work_type:     str,
) -> str:
    """
    Build a natural language search query from profile + preferences.

    JSearch's native API only supports work_from_home=true for remote.
    For hybrid and on-site we enrich the query string — Google for Jobs
    indexes these terms and uses them for relevance ranking.
    """
    if work_type == "remote":
        return f"{profile_title} remote"
    elif work_type == "hybrid":
        return f"{profile_title} hybrid {location}"
    elif work_type == "on-site":
        return f"{profile_title} onsite {location}"
    return f"{profile_title} {location}"


def _should_include(job: dict, work_type: str) -> bool:
    """
    Post-fetch work type filter using JSearch's job_is_remote field.

    JSearch returns job_is_remote (bool) per listing. We use this to
    filter results after fetching since the API has no native hybrid/
    on-site filter parameter.

    - remote:  keep only remote jobs
    - on-site: keep only non-remote jobs
    - hybrid:  keep all (hybrid is ambiguous in listings — many hybrid
               jobs are not flagged as remote, so excluding remote is
               too aggressive)
    - any:     keep all
    """
    if work_type == "remote":
        return job.get("job_is_remote", False) is True
    if work_type == "on-site":
        return job.get("job_is_remote", False) is not True
    return True  # hybrid and any — no filtering


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    reraise=True,
)
async def search_jsearch(
    profile_title:  str,
    location:       str,
    work_type:      str = "any",
    num_pages:      int = DEFAULT_PAGES,
) -> list[dict]:
    """
    Search JSearch API for jobs matching a profile title and location.

    Args:
        profile_title: e.g. "ML Engineer", "Senior Data Scientist"
        location:      e.g. "Delhi NCR", "London", "Remote"
        work_type:     "remote" | "hybrid" | "on-site" | "any"
        num_pages:     number of result pages (10 results per page)

    Returns:
        List of raw job dicts from JSearch API, filtered by work_type.
        Empty list on error (logged, not raised — don't block pipeline).
    """
    query = _build_query(profile_title, location, work_type)

    params = {
        "query":            query,
        "num_pages":        str(num_pages),
        "employment_types": "FULLTIME",
    }

    # JSearch native remote filter — only valid signal available from API
    if work_type == "remote":
        params["work_from_home"] = "true"

    logger.info(
        f"[jsearch] Searching: query='{query}' pages={num_pages}"
    )

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(
                f"{JSEARCH_BASE}/search",
                headers=_get_headers(),
                params=params,
            )
            response.raise_for_status()
            data = response.json()

        jobs = data.get("data", [])

        # Post-fetch filter by work type using job_is_remote field
        before = len(jobs)
        jobs = [j for j in jobs if _should_include(j, work_type)]
        if before != len(jobs):
            logger.info(
                f"[jsearch] work_type filter '{work_type}': "
                f"{before} → {len(jobs)} jobs"
            )

        logger.info(f"[jsearch] Returned {len(jobs)} jobs for '{query}'")
        return jobs

    except httpx.HTTPStatusError as e:
        logger.error(
            f"[jsearch] HTTP error {e.response.status_code} "
            f"for query '{query}': {e.response.text[:200]}"
        )
        return []
    except httpx.TimeoutException:
        logger.error(f"[jsearch] Timeout for query '{query}'")
        return []
    except Exception as e:
        logger.error(f"[jsearch] Unexpected error for query '{query}': {e}")
        return []
