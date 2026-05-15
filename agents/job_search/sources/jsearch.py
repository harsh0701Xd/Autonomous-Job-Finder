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

logger = logging.getLogger(__name__)

JSEARCH_BASE    = "https://jsearch.p.rapidapi.com"
JSEARCH_HOST    = "jsearch.p.rapidapi.com"
DEFAULT_PAGES   = 5       # 10 results per page  50 raw per profile
                          # increased to compensate for source quality filter
DEFAULT_TIMEOUT = 20.0    # seconds


#  Source quality filter constants 
#
# Policy (Option A  confirmed by product owner):
#   KEEP:  LinkedIn | Indeed | Glassdoor publishers
#          + ATS-hosted direct apply (Greenhouse, Lever, Workday, etc.)
#          + Company-owned career pages (/careers/, /jobs/ on non-aggregator domains)
#   DROP:  All other aggregators and job boards
#
# This ensures users only see listings they can apply to directly
# or that are verified by a major trusted platform.

# Signal 1: Trusted publisher names  ONLY these three
# Matched as substrings (case-insensitive) against job_publisher field
QUALITY_PUBLISHERS = [
    "linkedin",
    "indeed",
    "glassdoor",
]

# Signal 2: ATS / direct hiring platform domains (Option A  keep these)
# These are direct company applications hosted on third-party ATS platforms.
# The candidate applies directly to the company  not through an aggregator.
# Matched as substrings against the apply_url.
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

# Signal 3: Direct company career page URL path patterns
# Matched against apply_url  catches company-hosted pages not on standard ATS.
# The aggregator check below rejects false positives from known job boards
# that also use /jobs/ or /careers/ in their URLs.
CAREER_PAGE_PATHS = ["/careers/", "/en/careers/", "/career/", "/jobs/", "/join/", "/openings/"]

# Aggregator and job board domains  reject Signal 3 matches from these.
# This list is comprehensive: all non-LinkedIn/Indeed/Glassdoor job boards
# are explicitly blocked so only genuine company-owned pages pass Signal 3.
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


def _build_query(
    profile_title: str,
    location:      Optional[str],
    work_type:     str,
) -> str:
    """
    Build a JSearch natural language query from profile + preferences.

    Remote mode (location=None):
        Query is profile title only  no location suffix.
        JSearch remote_jobs_only parameter handles geography filtering.
        Returns global results  no restriction to Indian remote roles.

    Office/hybrid mode (location=str):
        Query appends the selected city  surfaces on-site and hybrid roles
        in that geography.
    """
    if work_type == "remote" or not location:
        return profile_title.strip()
    return f"{profile_title.strip()} {location.strip()}"


def _should_include_work_type(job: dict, work_type: str) -> bool:
    """
    Work type filter.

    remote  trust JSearch's server-side work_from_home=true filter.
             Do NOT check job_is_remote  JSearch frequently returns valid remote
             jobs with job_is_remote=false even when queried with work_from_home=true.
             Applying a client-side check drops legitimate remote listings.

    office  drop jobs explicitly flagged as remote (job_is_remote=True).
             Missing/null is treated as office  conservative, keeps more results.
    """
    if work_type == "remote":
        return True   # already filtered server-side by work_from_home=true
    # office  only drop confirmed remote jobs
    return job.get("job_is_remote", False) is not True


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
    # Must have a career-page path AND not be a known aggregator domain
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
async def search_jsearch(
    profile_title:  str,
    location:       Optional[str],   # None for remote mode  no geography constraint
    work_type:      str = "office",
    num_pages:      int = DEFAULT_PAGES,
) -> list[dict]:
    """
    Search JSearch API for jobs matching a profile title and location.

    Applies two post-fetch filters:
    1. Work type filter  (remote/hybrid/on-site)
    2. Source quality filter (LinkedIn/Indeed/Glassdoor/ATS/direct career page)

    Returns only quality-sourced jobs. All filter decisions are logged.
    """
    query = _build_query(profile_title, location, work_type)

    params = {
        "query":     query,
        "num_pages": str(num_pages),
    }

    if work_type == "remote":
        params["work_from_home"] = "true"

    logger.info(f"[jsearch] Searching: query='{query}' pages={num_pages}")

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

        #  Filter 1: Work type 
        jobs = [j for j in jobs if _should_include_work_type(j, work_type)]
        after_work_type = len(jobs)
        if raw_count != after_work_type:
            logger.info(
                f"[jsearch] work_type filter '{work_type}': "
                f"{raw_count}  {after_work_type} jobs"
            )

        #  Filter 2: Source quality 
        kept   = []
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

        # Summary log for LangSmith evaluation
        logger.info(
            f"[jsearch] Source quality filter: "
            f"{after_work_type}  {len(kept)} jobs kept "
            f"({after_work_type - len(kept)} rejected) | "
            f"query='{query}'"
        )

        if rejected_log:
            # Log unique rejection reasons for source analysis
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