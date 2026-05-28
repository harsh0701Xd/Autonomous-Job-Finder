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
import hashlib
import logging
import os
from typing import Any, Optional

import httpx

from agents.job_search.quality_gates import (
    clean_text, is_english, is_jd_sufficient, make_job_id, parse_date,
)
from core.config.config_loader import cfg
from core.state.session_state import RawJob

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source descriptor — read by registry.py to build the SOURCES list.
# ---------------------------------------------------------------------------
SOURCE_SPEC = {
    "name":              "jobs_search_api",
    "env_key":           "JOBS_SEARCH_API_KEY",
    "always_on":         False,
    "requires_location": True,
}

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


async def search(
    profile_title: str,
    location:      str,
    pages:         int,
    **kwargs,
) -> list[dict[str, Any]]:
    """
    Search Jobs Search API for jobs matching profile_title + location.
    Returns list of raw job dicts. Empty list on any error.

    Note: This API uses POST with a JSON body, not GET with query params.
    Each request returns results_wanted jobs. `pages` controls how many
    POST requests are made (each fetching 10 jobs).
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
        for page in range(pages):
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
            if page < pages - 1:
                await asyncio.sleep(1.0)   # slightly longer pause — POST with fetch

    logger.info(
        f"[jobs_search_api] Complete — {len(all_jobs)} raw jobs for '{profile_title}'"
    )
    return all_jobs


def normalize(raw: dict, matched_profile: str) -> Optional[RawJob]:
    """
    Normalize a Jobs Search API (JobSpy-style) response object into RawJob.

    JobSpy schema (NOT Fantastic Jobs schema):
      id, title, company, description (FULL when linkedin_fetch_description=True),
      job_url, date_posted, location, is_remote,
      min_amount, max_amount (salary)
    """
    try:
        job_id    = str(raw.get("id") or "")
        title     = (raw.get("title") or "").strip()
        company   = (raw.get("company") or "").strip()
        apply_url = (raw.get("job_url") or "").strip()

        if not title or not apply_url:
            return None

        # Fallback ID if none provided
        if not job_id:
            job_id = hashlib.md5(f"{title}{company}{apply_url}".encode()).hexdigest()[:12]

        jd_text = clean_text(raw.get("description") or "")

        if not is_jd_sufficient(jd_text, title, company):
            return None
        if not is_english(jd_text, title, company):
            return None

        location  = (raw.get("location") or "Not specified").strip()
        work_type = "remote" if raw.get("is_remote", False) else "office"

        return RawJob(
            job_id          = make_job_id("jobs_search_api", job_id),
            title           = title,
            company         = company,
            location        = location,
            work_type       = work_type,
            jd_text         = jd_text,
            apply_url       = apply_url,
            source          = "jobs_search_api",
            posted_date     = parse_date(raw.get("date_posted")),
            matched_profile = matched_profile,
        )

    except Exception as e:
        logger.warning(f"[jobs_search_api] Failed to normalize job: {e}")
        return None
