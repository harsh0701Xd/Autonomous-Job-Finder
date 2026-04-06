"""
agents/job_search/normalizer.py

Maps raw API responses from each source into the unified RawJob schema.

Each source returns different field names — this is the single place
that handles all the mapping. Agent 4 (ranker) only ever sees RawJob objects.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional

from core.state.session_state import RawJob

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_job_id(source: str, raw_id: str) -> str:
    """
    Generate a stable job ID from source + raw identifier.
    Ensures IDs are unique across sources without collisions.
    """
    return f"{source}_{raw_id}"


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Safely parse an ISO date string into datetime."""
    if not date_str:
        return None
    try:
        # Handle both "2024-01-15" and "2024-01-15T10:30:00Z"
        date_str = date_str.replace("Z", "+00:00")
        return datetime.fromisoformat(date_str)
    except (ValueError, AttributeError):
        return None


def _clean_text(text: Optional[str], max_chars: int = 3000) -> str:
    """Strip and truncate job description text."""
    if not text:
        return ""
    return text.strip()[:max_chars]


def _extract_salary(value: Optional[float | int | str]) -> Optional[int]:
    """Safely convert salary value to int."""
    if value is None:
        return None
    try:
        return int(float(str(value).replace(",", "").replace("$", "")))
    except (ValueError, TypeError):
        return None


# ── JSearch normalizer ────────────────────────────────────────────────────────

def normalize_jsearch(raw: dict, matched_profile: str) -> Optional[RawJob]:
    """
    Normalize a JSearch API job dict into RawJob.

    JSearch field reference:
      job_id, job_title, employer_name, job_city, job_country,
      job_description, job_apply_link, job_posted_at_datetime_utc,
      job_min_salary, job_max_salary, job_employment_type,
      job_is_remote
    """
    try:
        job_id   = raw.get("job_id") or ""
        title    = raw.get("job_title") or ""
        company  = raw.get("employer_name") or ""
        apply_url = raw.get("job_apply_link") or ""

        if not job_id or not title or not apply_url:
            return None

        # Build location string
        city    = raw.get("job_city") or ""
        country = raw.get("job_country") or ""
        location = ", ".join(filter(None, [city, country])) or "Not specified"

        # Work type
        is_remote    = raw.get("job_is_remote", False)
        employment   = (raw.get("job_employment_type") or "").lower()
        if is_remote:
            work_type = "remote"
        elif "hybrid" in (raw.get("job_job_title") or "").lower():
            work_type = "hybrid"
        else:
            work_type = "on-site"

        return RawJob(
            job_id          = _make_job_id("jsearch", job_id),
            title           = title,
            company         = company,
            location        = location,
            work_type       = work_type,
            jd_text         = _clean_text(raw.get("job_description")),
            apply_url       = apply_url,
            source          = "jsearch",
            posted_date     = _parse_date(raw.get("job_posted_at_datetime_utc")),
            salary_min      = _extract_salary(raw.get("job_min_salary")),
            salary_max      = _extract_salary(raw.get("job_max_salary")),
            matched_profile = matched_profile,
        )

    except Exception as e:
        logger.warning(f"[normalizer] Failed to normalize JSearch job: {e}")
        return None


# ── RemoteOK normalizer ───────────────────────────────────────────────────────

def normalize_remoteok(raw: dict, matched_profile: str) -> Optional[RawJob]:
    """
    Normalize a RemoteOK job dict into RawJob.

    RemoteOK field reference:
      id, position, company, url, description,
      date, salary_min, salary_max, tags, location
    """
    try:
        job_id    = str(raw.get("id") or "")
        title     = raw.get("position") or ""
        company   = raw.get("company") or ""
        apply_url = raw.get("url") or ""

        if not job_id or not title or not apply_url:
            return None

        return RawJob(
            job_id          = _make_job_id("remoteok", job_id),
            title           = title,
            company         = company,
            location        = raw.get("location") or "Remote",
            work_type       = "remote",
            jd_text         = _clean_text(raw.get("description")),
            apply_url       = apply_url,
            source          = "remoteok",
            posted_date     = _parse_date(raw.get("date")),
            salary_min      = _extract_salary(raw.get("salary_min")),
            salary_max      = _extract_salary(raw.get("salary_max")),
            matched_profile = matched_profile,
        )

    except Exception as e:
        logger.warning(f"[normalizer] Failed to normalize RemoteOK job: {e}")
        return None


# ── Dispatcher ────────────────────────────────────────────────────────────────

NORMALIZERS = {
    "jsearch":  normalize_jsearch,
    "remoteok": normalize_remoteok,
}


def normalize_jobs(
    raw_jobs:        list[dict],
    source:          str,
    matched_profile: str,
) -> list[RawJob]:
    """
    Normalize a list of raw job dicts from a given source.
    Skips jobs that fail normalization — never raises.

    Args:
        raw_jobs:        raw API response list
        source:          "jsearch" | "remoteok"
        matched_profile: which confirmed profile triggered this search

    Returns:
        List of valid RawJob objects.
    """
    normalizer = NORMALIZERS.get(source)
    if not normalizer:
        logger.error(f"[normalizer] Unknown source: {source}")
        return []

    results = []
    skipped = 0

    for raw in raw_jobs:
        job = normalizer(raw, matched_profile)
        if job:
            results.append(job)
        else:
            skipped += 1

    if skipped:
        logger.debug(
            f"[normalizer] {source}: {len(results)} normalized, "
            f"{skipped} skipped (missing required fields)"
        )

    return results
