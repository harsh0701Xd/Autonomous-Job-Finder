"""
agents/job_search/sources/jsearch.py

JSearch API client (via RapidAPI).
Aggregates Indeed, LinkedIn, Glassdoor, ZipRecruiter globally.

API docs: https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch

Source quality filter:
  Keeps ONLY jobs from:
    1. Known quality publishers   LinkedIn, Indeed, Glassdoor
    2. Known ATS domains          Greenhouse, Workday, Lever, etc.
    3. Direct company pages       apply_url contains /careers/ or /jobs/
                                   on a non-aggregator domain
  Everything else (jobrapido, trabajo.org, etc.) is discarded.
  All filter decisions are logged for evaluation.
"""

from __future__ import annotations

import logging
import os
from typing import Optional
from urllib.parse import urlparse

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from agents.job_search.quality_gates import (
    clean_text, is_english, is_expired, is_jd_sufficient, make_job_id, parse_date,
)
from core.state.session_state import RawJob

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source descriptor — read by registry.py to build the SOURCES list.
# ---------------------------------------------------------------------------
SOURCE_SPEC = {
    "name":              "jsearch",
    "env_key":           "JSEARCH_API_KEY",   # env var that must be set when enabled
    "always_on":         False,               # controlled entirely by llm_config.yaml
    "requires_location": True,                # passes location to search()
}

JSEARCH_BASE    = "https://jsearch.p.rapidapi.com"
JSEARCH_HOST    = "jsearch.p.rapidapi.com"
DEFAULT_TIMEOUT = 20.0   # seconds


#  Source quality filter constants 
#
# Policy (Option A  confirmed by product owner):
#   KEEP:  LinkedIn | Indeed | Glassdoor publishers
#          + ATS-hosted direct apply (Greenhouse, Lever, Workday, etc.)
#          + Company-owned career pages (/careers/, /jobs/ on non-aggregator domains)
#   DROP:  All other aggregators and job boards

QUALITY_PUBLISHERS = [
    "linkedin",
    "indeed",
    "glassdoor",
]

QUALITY_ATS_DOMAINS = [
    # Global ATS platforms
    "greenhouse.io",
    "boards.greenhouse.io",
    "jobs.greenhouse.io",
    "lever.co",
    "jobs.lever.co",
    "workday.com",
    "myworkdayjobs.com",
    "workable.com",
    "jobs.workable.com",
    "smartrecruiters.com",
    "jobs.smartrecruiters.com",
    "icims.com",
    "ashbyhq.com",
    "jobs.ashbyhq.com",
    "bamboohr.com",
    "taleo.net",
    "jobvite.com",
    "breezy.hr",
    "recruitee.com",
    "clearcompany.com",
    "jazzhr.com",
    "successfactors.com",
    "personio.de",
    "jobs.personio.de",
    "teamtailor.com",
    "ultipro.com",
    "ukg.com",
    "paylocity.com",
    "rippling.com",
    "pinpointhq.com",
    # Indian ATS platforms
    "darwinbox.com",
    "keka.com",
    "freshteam.com",
    "springrecruit.com",
]

CAREER_PAGE_PATHS = ["/careers/", "/en/careers/", "/career/", "/jobs/", "/join/", "/openings/"]

AGGREGATOR_DOMAINS = [
    # Indian job boards (blocked  only LinkedIn/Indeed/Glassdoor allowed via Signal 1)
    "naukri.com", "naukrigulf.com", "foundit.in", "monsterindia.com",
    "instahyre.com", "iimjobs.com", "cutshort.io", "wellfound.com",
    "internshala.com", "hasjob.co", "apna.co", "hirist.tech",
    "shine.com", "timesjobs.com", "wisdomjobs.com", "freshersworld.com",
    "placementindia.com", "freshteam.com",
    # Global job boards (blocked)
    "ziprecruiter.com", "monster.com", "dice.com", "builtin.com",
    "careerbuilder.com", "simplyhired.com", "jobrapido.com",
    "careerjet.com", "jooble.org", "neuvoo.com", "talent.com",
    "efinancialcareers.com", "jobsora.com", "trovit.com", "mitula.com",
    "trabajo.org", "jobberman.com", "in.jobrapido.com", "in.trabajo.org",
    "jobs.towardsai.net", "wellfound.com", "angel.co",
]


def _get_headers() -> dict:
    key = os.getenv("JSEARCH_API_KEY")
    if not key:
        raise ValueError("JSEARCH_API_KEY not set in environment")
    return {
        "X-RapidAPI-Key":  key,
        "X-RapidAPI-Host": JSEARCH_HOST,
    }


def _build_query(profile_title: str, location: Optional[str]) -> str:
    """
    Build a JSearch natural language query from profile + location.

    Always appends the city name so JSearch surfaces geographically relevant
    results for all job types (on-site, hybrid, and remote).
    The ranker's E3 location filter handles final geographic gating.
    """
    if not location:
        return profile_title.strip()
    return f"{profile_title.strip()} {location.strip()}"


def _is_quality_source(job: dict) -> tuple[bool, str]:
    """
    Three-signal source quality check.

    Returns (keep: bool, reason: str) for logging.

    Signal 1  Known quality publisher (LinkedIn, Indeed, Glassdoor):
        job_publisher contains a quality publisher name.

    Signal 2  Known ATS domain in apply_url:
        URL points to Greenhouse, Workday, Lever, etc.
        These are always direct company applications.

    Signal 3  Direct company career page:
        apply_url contains /careers/ or /jobs/ path on a non-aggregator domain.
        Catches company-hosted career pages that don't use a standard ATS.

    Anything not matching any signal is rejected.
    """
    publisher = (job.get("job_publisher") or "").lower()
    apply_url = (job.get("job_apply_link") or "").lower()

    # Signal 1: Known quality publisher
    for p in QUALITY_PUBLISHERS:
        if p in publisher:
            return True, f"publisher:{p}"

    # Signal 2: Known ATS domain
    for ats in QUALITY_ATS_DOMAINS:
        if ats in apply_url:
            return True, f"ats:{ats}"

    # Signal 3: Direct company career page
    has_career_path = any(path in apply_url for path in CAREER_PAGE_PATHS)
    if has_career_path:
        try:
            domain = urlparse(apply_url).netloc
        except Exception:
            domain = ""
        is_aggregator = any(agg in domain for agg in AGGREGATOR_DOMAINS)
        if not is_aggregator:
            return True, f"direct_career_page:{domain}"

    return False, f"rejected:publisher='{job.get('job_publisher', '')}' url='{apply_url[:60]}'"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    reraise=True,
)
async def search(
    profile_title: str,
    location:      str,
    pages:         int,
    **kwargs,
) -> list[dict]:
    """
    Search JSearch API for jobs matching a profile title and location.

    Always queries with city context so JSearch surfaces geographically
    relevant listings. Applies source quality filter (LinkedIn/Indeed/
    Glassdoor/ATS/direct career page) to ensure only trustworthy listings
    are returned. Geographic and remote-work gating is handled downstream
    by the ranker's E3 location filter.

    Returns only quality-sourced jobs. All filter decisions are logged.
    """
    query = _build_query(profile_title, location or None)

    params = {
        "query":     query,
        "num_pages": str(pages),
    }

    logger.info(f"[jsearch] Searching: query='{query}' pages={pages}")

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
        raw_count = len(jobs)

        #  Source quality filter
        kept         = []
        rejected_log = []

        for job in jobs:
            keep, reason = _is_quality_source(job)
            if keep:
                kept.append(job)
                logger.debug(
                    f"[jsearch:quality] KEEP  '{job.get('job_title','')}' "
                    f"@ {job.get('employer_name','')} | {reason}"
                )
            else:
                rejected_log.append(reason)
                logger.debug(
                    f"[jsearch:quality] DROP  '{job.get('job_title','')}' "
                    f"@ {job.get('employer_name','')} | {reason}"
                )

        logger.info(
            f"[jsearch] Source quality filter: "
            f"{raw_count} → {len(kept)} jobs kept "
            f"({raw_count - len(kept)} rejected) | "
            f"query='{query}'"
        )

        if rejected_log:
            from collections import Counter
            rejection_summary = dict(Counter(
                r.split(":")[0] + ":" + r.split(":")[1].split("=")[0]
                if "=" in r else r
                for r in rejected_log
            ))
            logger.info(f"[jsearch] Rejection summary: {rejection_summary}")

        return kept

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


def normalize(raw: dict, matched_profile: str) -> Optional[RawJob]:
    """
    Normalize a JSearch API job dict into RawJob.

    Quality gates applied in order:
      1. Required fields present (job_id, title, apply_url)
      2. Listing not expired (expiry date in past)
      3. JD not null/empty and >= MIN_JD_CHARS / MIN_JD_WORDS
      4. JD is English (non-ASCII ratio <= 30%)

    Returns None for any gate failure -- job is silently dropped.

    JSearch field reference:
      job_id, job_title, employer_name, job_city, job_country,
      job_description, job_apply_link, job_posted_at_datetime_utc,
      job_min_salary, job_max_salary, job_employment_type, job_is_remote
    """
    try:
        job_id    = raw.get("job_id")    or ""
        title     = raw.get("job_title") or ""
        company   = raw.get("employer_name") or ""
        apply_url = raw.get("job_apply_link") or ""

        if not job_id or not title or not apply_url:
            return None

        if is_expired(raw, title, company):
            return None

        raw_jd  = raw.get("job_description") or ""
        jd_text = clean_text(raw_jd)

        if not is_jd_sufficient(jd_text, title, company):
            return None
        if not is_english(jd_text, title, company):
            return None

        city     = raw.get("job_city")    or ""
        country  = raw.get("job_country") or ""
        location = ", ".join(filter(None, [city, country])) or "Not specified"

        is_remote = raw.get("job_is_remote", False)
        work_type = "remote" if is_remote else "office"

        return RawJob(
            job_id          = make_job_id("jsearch", job_id),
            title           = title,
            company         = company,
            location        = location,
            work_type       = work_type,
            jd_text         = jd_text,
            apply_url       = apply_url,
            source          = "jsearch",
            posted_date     = parse_date(raw.get("job_posted_at_datetime_utc")),
            matched_profile = matched_profile,
        )

    except Exception as e:
        logger.warning(f"[jsearch] Failed to normalize job: {e}")
        return None
