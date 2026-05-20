"""
agents/job_search/sources/active_jobs_db.py

Active Jobs DB — RapidAPI source adapter.

API:    GET https://active-jobs-db.p.rapidapi.com/active-ats-1h
Auth:   X-RapidAPI-Key header
Free:   25 requests/month, 250 jobs/month hard limit
Params: title_filter, location_filter, description_type=text, offset
Key:    ACTIVE_JOBS_DB_API_KEY in .env

Notes:
  - title_filter uses quoted strings e.g. '"Senior Data Scientist"'
  - location_filter uses full names e.g. '"India"' (no abbreviations)
  - description_type=text returns full plain-text JD (required for ranker)
  - offset paginates in steps of 100 (not pages)
  - Returns up to 100 jobs per call but monthly job cap is 250
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://active-jobs-db.p.rapidapi.com/active-ats-1h"
_TIMEOUT  = 20.0
_PAGE_SIZE = 10   # conservative — stay within 250/month job cap


async def search_active_jobs_db(
    profile_title: str,
    location:      str,
    num_pages:     int = 1,
) -> list[dict[str, Any]]:
    """
    Search Active Jobs DB for jobs matching profile_title + location.
    Returns list of raw job dicts. Empty list on any error.
    """
    api_key = os.environ.get("ACTIVE_JOBS_DB_API_KEY", "")
    if not api_key:
        logger.warning("[active_jobs_db] ACTIVE_JOBS_DB_API_KEY not set — skipping")
        return []

    headers = {
        "x-rapidapi-key":  api_key,
        "x-rapidapi-host": "active-jobs-db.p.rapidapi.com",
    }

    all_jobs: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for page in range(num_pages):
            params: dict[str, Any] = {
                "title_filter":    profile_title,
                "location_filter": f'"{location}"',
                "description_type": "text",
                "offset":          page * _PAGE_SIZE,
            }

            logger.info(
                f"[active_jobs_db] Page {page + 1} — "
                f"profile='{profile_title}' location='{location}'"
            )

            try:
                r = await client.get(_BASE_URL, headers=headers, params=params)
                r.raise_for_status()
                data = r.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning("[active_jobs_db] 429 rate limit — stopping")
                else:
                    logger.error(
                        f"[active_jobs_db] HTTP {e.response.status_code} page {page + 1}: "
                        f"{e.response.text[:200]}"
                    )
                break
            except httpx.TimeoutException:
                logger.warning(f"[active_jobs_db] Timeout on page {page + 1}")
                break
            except Exception as e:
                logger.error(f"[active_jobs_db] Error page {page + 1}: {e}")
                break

            # Response is a list of job objects directly
            jobs = data if isinstance(data, list) else data.get("data", [])
            all_jobs.extend(jobs)

            logger.info(
                f"[active_jobs_db] Page {page + 1}: {len(jobs)} jobs "
                f"(total: {len(all_jobs)})"
            )

            if len(jobs) == 0:
                break
            if page < num_pages - 1:
                await asyncio.sleep(0.3)

    logger.info(
        f"[active_jobs_db] Complete — {len(all_jobs)} raw jobs for '{profile_title}'"
    )
    return all_jobs
