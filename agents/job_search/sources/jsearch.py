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
    JSearch handles free-text queries natively — no routing needed.
    """
    if work_type == "remote":
        return f"{profile_title} remote"
    return f"{profile_title} {location}"


def _work_type_filter(work_type: str) -> Optional[str]:
    """Map our work_type to JSearch employment_types param."""
    mapping = {
        "remote":  "FULLTIME",
        "hybrid":  "FULLTIME",
        "on-site": "FULLTIME",
        "any":     "FULLTIME",
    }
    return mapping.get(work_type, "FULLTIME")


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
        List of raw job dicts from JSearch API.
        Empty list on error (logged, not raised — don't block pipeline).
    """
    query = _build_query(profile_title, location, work_type)

    params = {
        "query":            query,
        "num_pages":        str(num_pages),
        "employment_types": _work_type_filter(work_type),
    }

    # Add remote filter if applicable
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
