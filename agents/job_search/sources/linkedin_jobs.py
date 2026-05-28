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
from typing import Any, Optional

import httpx

from agents.job_search.quality_gates import (
    clean_text, is_english, is_jd_sufficient, make_job_id, parse_date,
)
from core.state.session_state import RawJob

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source descriptor — read by registry.py to build the SOURCES list.
# ---------------------------------------------------------------------------
SOURCE_SPEC = {
    "name":              "linkedin_jobs",
    "env_key":           "LINKEDIN_JOBS_API_KEY",
    "always_on":         False,
    "requires_location": True,
}

_BASE_URL  = "https://linkedin-job-search-api.p.rapidapi.com/active-jb-1h"
_TIMEOUT   = 20.0
_PAGE_SIZE = 10   # conservative — stay within 250/month job cap


async def search(
    profile_title: str,
    location:      str,
    pages:         int,
    **kwargs,
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
        for page in range(pages):
            params: dict[str, Any] = {
                "title_filter":     profile_title,
                "location_filter":  f'"{location}"',
                "description_type": "text",
                "limit":            _PAGE_SIZE,
                "offset":           page * _PAGE_SIZE,
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
            if page < pages - 1:
                await asyncio.sleep(0.3)

    logger.info(
        f"[linkedin_jobs] Complete — {len(all_jobs)} raw jobs for '{profile_title}'"
    )
    return all_jobs


def normalize(raw: dict, matched_profile: str) -> Optional[RawJob]:
    """
    Normalize a LinkedIn Job Search API response object into RawJob.

    LinkedIn Job Search API is also a Fantastic Jobs product, same schema
    as Active Jobs DB:
      id, title, organization, description (FULL — same field as Active Jobs DB),
      url, date_posted, locations_raw (list), remote_derived,
      min_annual_salary, max_annual_salary
    """
    try:
        job_id    = str(raw.get("id") or "")
        title     = (raw.get("title") or "").strip()
        company   = (raw.get("organization") or "").strip()
        apply_url = (raw.get("url") or "").strip()

        if not job_id or not title or not apply_url:
            return None

        jd_text = clean_text(raw.get("description") or "")

        if not is_jd_sufficient(jd_text, title, company):
            return None
        if not is_english(jd_text, title, company):
            return None

        # Extract location from locations_raw list
        locations_raw = raw.get("locations_raw") or []
        if locations_raw and isinstance(locations_raw, list):
            first_loc = locations_raw[0]
            location  = (
                first_loc.get("location") or
                first_loc.get("city") or
                "Not specified"
            ) if isinstance(first_loc, dict) else str(first_loc)
        else:
            location = raw.get("location") or "Not specified"

        work_type = "remote" if raw.get("remote_derived", False) else "office"

        return RawJob(
            job_id          = make_job_id("linkedin_jobs", job_id),
            title           = title,
            company         = company,
            location        = location,
            work_type       = work_type,
            jd_text         = jd_text,
            apply_url       = apply_url,
            source          = "linkedin_jobs",
            posted_date     = parse_date(raw.get("date_posted")),
            matched_profile = matched_profile,
        )

    except Exception as e:
        logger.warning(f"[linkedin_jobs] Failed to normalize job: {e}")
        return None
