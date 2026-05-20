"""
agents/job_search/sources/linkedin_jobs.py

LinkedIn Job Search API — RapidAPI source adapter (Fantastic Jobs product).

API:    GET https://linkedin-job-search-api.p.rapidapi.com/active-jb-1h
Auth:   x-rapidapi-key header (lowercase)
Free:   25 requests/month, 250 jobs/month hard limit
Params: title_filter, location_filter, description_type=text, limit, offset
Key:    LINKEDIN_JOBS_API_KEY in .env

Schema: Same as Active Jobs DB (both are Fantastic Jobs products):
  id, title, organization, description (FULL when description_type=text),
  url, date_posted, locations_raw (list), remote_derived,
  min_annual_salary, max_annual_salary
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_BASE_URL  = "https://linkedin-job-search-api.p.rapidapi.com/active-jb-1h"
_TIMEOUT   = 20.0
_PAGE_SIZE = 10   # conservative — stay within 250/month job cap


async def search_linkedin_jobs(
    profile_title: str,
    location:      str,
    num_pages:     int = 1,
    limit:         int = _PAGE_SIZE,
) -> list[dict[str, Any]]:
    """
    Search LinkedIn Job Search API for jobs matching profile_title + location.
    Returns list of raw job dicts. Empty list on any error.
    """
    api_key = os.environ.get("LINKEDIN_JOBS_API_KEY", "")
    if not api_key:
        logger.warning("[linkedin_jobs] LINKEDIN_JOBS_API_KEY not set — skipping")
        return []

    headers = {
        "x-rapidapi-key":  api_key,
        "x-rapidapi-host": "linkedin-job-search-api.p.rapidapi.com",
    }

    all_jobs: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for page in range(num_pages):
            params: dict[str, Any] = {
                "title_filter":     profile_title,
                "location_filter":  f'"{location}"',
                "description_type": "text",
                "limit":            limit,
                "offset":           page * limit,
            }

            logger.info(
                f"[linkedin_jobs] Page {page + 1} — "
                f"profile='{profile_title}' location='{location}'"
            )

            try:
                r = await client.get(_BASE_URL, headers=headers, params=params)
                r.raise_for_status()
                data = r.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning("[linkedin_jobs] 429 rate limit — stopping")
                else:
                    logger.error(
                        f"[linkedin_jobs] HTTP {e.response.status_code} page {page + 1}: "
                        f"{e.response.text[:200]}"
                    )
                break
            except httpx.TimeoutException:
                logger.warning(f"[linkedin_jobs] Timeout on page {page + 1}")
                break
            except Exception as e:
                logger.error(f"[linkedin_jobs] Error page {page + 1}: {e}")
                break

            jobs = data if isinstance(data, list) else data.get("data", [])
            all_jobs.extend(jobs)

            logger.info(
                f"[linkedin_jobs] Page {page + 1}: {len(jobs)} jobs "
                f"(total: {len(all_jobs)})"
            )

            if len(jobs) == 0:
                break
            if page < num_pages - 1:
                await asyncio.sleep(0.3)

    logger.info(
        f"[linkedin_jobs] Complete — {len(all_jobs)} raw jobs for '{profile_title}'"
    )
    return all_jobs
