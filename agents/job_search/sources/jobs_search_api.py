"""
agents/job_search/sources/jobs_search_api.py

Jobs Search API — RapidAPI source adapter.

API:    POST https://jobs-search-api.p.rapidapi.com/getjobs
Auth:   x-rapidapi-key header
Free:   100 requests/month hard limit
Body:   JSON payload (NOT query params — this API is POST not GET)
Key:    JOBS_SEARCH_API_KEY in .env

Request body fields:
  search_term         str   -- job title keywords
  location            str   -- city name
  country_indeed      str   -- "India" for IN
  results_wanted      int   -- jobs per request (max ~10 on free tier)
  site_name           list  -- sources to search
  is_remote           bool
  linkedin_fetch_description bool -- must be TRUE to get full JD text
  hours_old           int   -- freshness window from cfg.job_search.hours_old (default 168 = 7 days)

Response fields (JobSpy-style schema):
  id, title, company, location, description (FULL when linkedin_fetch_description=true),
  job_url, date_posted, min_amount, max_amount, is_remote
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

from core.config.config_loader import cfg

logger = logging.getLogger(__name__)

_BASE_URL = "https://jobs-search-api.p.rapidapi.com/getjobs"
_TIMEOUT  = 30.0   # POST with description fetch can be slower


def _build_payload(
    profile_title: str,
    location:      str,
    results:       int,
) -> dict[str, Any]:
    """Build the POST request body."""
    return {
        "search_term":                profile_title,
        "location":                   location,
        "country_indeed":             "India",
        "results_wanted":             results,
        "site_name":                  ["indeed", "linkedin", "naukri", "glassdoor"],
        "linkedin_fetch_description": True,   # REQUIRED for full JD text
        "hours_old":                  cfg.job_search.hours_old,
        "distance":                   50,
    }


async def search_jobs_search_api(
    profile_title: str,
    location:      str,
    num_pages:     int = 2,
) -> list[dict[str, Any]]:
    """
    Search Jobs Search API for jobs matching profile_title + location.
    Returns list of raw job dicts. Empty list on any error.

    Note: This API uses POST with a JSON body, not GET with query params.
    Each request returns results_wanted jobs. num_pages is used to
    make multiple calls with increasing hours_old to get more variety.
    """
    api_key = os.environ.get("JOBS_SEARCH_API_KEY", "")
    if not api_key:
        logger.warning("[jobs_search_api] JOBS_SEARCH_API_KEY not set — skipping")
        return []

    headers = {
        "x-rapidapi-key":  api_key,
        "x-rapidapi-host": "jobs-search-api.p.rapidapi.com",
        "Content-Type":    "application/json",
    }

    all_jobs: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for page in range(num_pages):
            payload = _build_payload(profile_title, location, results=10)

            logger.info(
                f"[jobs_search_api] Request {page + 1} — "
                f"profile='{profile_title}' location='{location}' "
                f"hours_old={cfg.job_search.hours_old}"
            )

            try:
                r = await client.post(_BASE_URL, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.error(
                        f"[jobs_search_api] 404 — endpoint may be wrong. "
                        f"Check RapidAPI → JOBS SEARCH API → Endpoints tab."
                    )
                elif e.response.status_code == 429:
                    logger.warning("[jobs_search_api] 429 rate limit — stopping")
                else:
                    logger.error(
                        f"[jobs_search_api] HTTP {e.response.status_code} "
                        f"request {page + 1}: {e.response.text[:200]}"
                    )
                break
            except httpx.TimeoutException:
                logger.warning(f"[jobs_search_api] Timeout on request {page + 1}")
                break
            except Exception as e:
                logger.error(f"[jobs_search_api] Error request {page + 1}: {e}")
                break

            # Response: {"jobs": [...]} or a list directly
            jobs = (
                data.get("jobs") or
                data.get("data") or
                (data if isinstance(data, list) else [])
            )
            all_jobs.extend(jobs)

            logger.info(
                f"[jobs_search_api] Request {page + 1}: {len(jobs)} jobs "
                f"(total: {len(all_jobs)})"
            )

            if len(jobs) == 0:
                break
            if page < num_pages - 1:
                await asyncio.sleep(1.0)   # slightly longer pause — POST with fetch

    logger.info(
        f"[jobs_search_api] Complete — {len(all_jobs)} raw jobs for '{profile_title}'"
    )
    return all_jobs
