"""
agents/job_search/sources/techmap.py

Techmap Daily International Job Postings — RapidAPI source adapter.

API:    GET https://daily-international-job-postings.p.rapidapi.com/api/v2/jobs/search
Auth:   Authorization: Bearer <token>  (NOT X-RapidAPI-Key)
Free:   100 requests × 10 jobs = 1,000 jobs/month
Fields: Unique JSON-LD wrapped structure inside result[]:
          result[n].jsonLD.title
          result[n].jsonLD.description  (FULL — markdown formatted)
          result[n].jsonLD.url          (apply URL)
          result[n].jsonLD.datePosted
          result[n].jsonLD.hiringOrganization.name
          result[n].jsonLD.jobLocation.address.addressLocality  (city)
          result[n].jsonLD.employmentType
          result[n].jsonLD.baseSalary.value.minValue / maxValue
          result[n].city                (top-level shortcut)
          result[n].company             (top-level shortcut)
          result[n].workPlace           (remote|hybrid|onsite)
Key:    TECHMAP_API_KEY in .env
Note:   countryCode=in for India. title filter via ?title= query param.
        dateCreated defaults to today-2 if not specified.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://daily-international-job-postings.p.rapidapi.com/api/v2/jobs/search"
_TIMEOUT  = 20.0

# City aliases for Techmap's free-text city filter
_CITY_ALIASES: dict[str, list[str]] = {
    "bengaluru": ["bangalore", "bengaluru"],
    "delhi ncr": ["delhi", "gurgaon", "noida"],
    "mumbai":    ["mumbai"],
    "hyderabad": ["hyderabad"],
    "chennai":   ["chennai"],
    "pune":      ["pune"],
}


def _resolve_city(location: str) -> str:
    """Return the primary city name Techmap expects for a given location string."""
    loc = location.strip().lower()
    for canonical, aliases in _CITY_ALIASES.items():
        if loc in canonical or any(loc in a for a in aliases):
            return aliases[0]
    return location.strip()


async def search_techmap(
    profile_title: str,
    location:      str,
    work_type:     str,
    num_pages:     int = 3,
) -> list[dict[str, Any]]:
    """
    Search Techmap for jobs matching profile_title + India location.
    Returns list of raw Techmap result objects (each has a 'jsonLD' sub-dict).
    Empty list on any error.
    """
    api_key = os.environ.get("TECHMAP_API_KEY", "")
    if not api_key:
        logger.warning("[techmap] TECHMAP_API_KEY not set — skipping")
        return []

    headers = {
        "Authorization":   f"Bearer {api_key}",
        "X-RapidAPI-Host": "daily-international-job-postings.p.rapidapi.com",
    }

    city = _resolve_city(location)
    work_place_map = {"remote": "remote", "hybrid": "hybrid", "office": "onsite"}
    work_place = work_place_map.get(work_type.lower(), "onsite")

    all_jobs: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for page in range(1, num_pages + 1):
            params: dict[str, Any] = {
                "title":       profile_title,
                "countryCode": "in",
                "city":        city,
                "workPlace":   work_place,
                "page":        page,
            }

            logger.info(
                f"[techmap] Page {page} — "
                f"profile='{profile_title}' city='{city}'"
            )

            try:
                r = await client.get(_BASE_URL, headers=headers, params=params)
                r.raise_for_status()
                data = r.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning("[techmap] 429 rate limit — stopping")
                elif e.response.status_code == 401:
                    logger.error("[techmap] 401 Unauthorized — check TECHMAP_API_KEY")
                else:
                    logger.error(f"[techmap] HTTP {e.response.status_code} page {page}")
                break
            except httpx.TimeoutException:
                logger.warning(f"[techmap] Timeout on page {page}")
                break
            except Exception as e:
                logger.error(f"[techmap] Error page {page}: {e}")
                break

            jobs = data.get("result", [])
            all_jobs.extend(jobs)

            logger.info(
                f"[techmap] Page {page}: {len(jobs)} jobs "
                f"(total: {len(all_jobs)}, totalCount: {data.get('totalCount', '?')})"
            )

            if len(jobs) == 0:
                break
            if page < num_pages:
                await asyncio.sleep(0.3)

    logger.info(f"[techmap] Complete — {len(all_jobs)} raw jobs for '{profile_title}'")
    return all_jobs
