"""
agents/job_search/normalizer.py

Maps raw API responses from each source into the unified RawJob schema.

Each source returns different field names -- this is the single place
that handles all the mapping. Agent 4 (ranker) only ever sees RawJob objects.

Quality gates applied here (before any downstream processing):
  1. JD null/empty          -> drop (1.1 / 1.2)
  2. JD below 200 chars     -> drop (1.1 / 1.2)
  3. JD below 60 words      -> drop (1.3)
  4. Non-English JD         -> drop (3.3)

These gates fire at normalisation time, returning None, so the job
never enters raw_jobs[] and never costs a Haiku or Cohere call.
All drops are logged with a reason code for LangSmith evaluation.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from core.config.config_loader import cfg
from core.state.session_state import RawJob

logger = logging.getLogger(__name__)

# -- Constants -----------------------------------------------------------------
# Loaded from llm_config.yaml [job_search] -- edit there, not here.

MIN_JD_CHARS = cfg.job_search.min_jd_chars   # chars gate -- JDs below this are dropped
MIN_JD_WORDS = cfg.job_search.min_jd_words   # word gate  -- JDs below this are dropped
MAX_JD_CHARS = cfg.job_search.max_jd_chars   # truncation ceiling for JD text

# Characters that are overwhelmingly Latin/English punctuation and digits.
# Used for the language heuristic -- no external library needed.
_LATIN_PATTERN     = re.compile(r"[a-zA-Z]")
_NON_ASCII_PATTERN = re.compile(r"[^\x00-\x7F]")


# -- Helpers -------------------------------------------------------------------

def _make_job_id(source: str, raw_id: str) -> str:
    """Generate a stable job ID from source + raw identifier."""
    return f"{source}_{raw_id}"


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Safely parse an ISO date string into datetime."""
    if not date_str:
        return None
    try:
        date_str = date_str.replace("Z", "+00:00")
        return datetime.fromisoformat(date_str)
    except (ValueError, AttributeError):
        return None


# Garbled cp1252-as-UTF-8 sequences sometimes found in scraped job text.
# These are Windows-1252 characters incorrectly decoded as UTF-8,
# producing mojibake sequences like (right single quote) or (left double quote).
# Mapping: garbled -> correct replacement
_MOJIBAKE_MAP = {
    "ΓÇÖ": "'",   # right single quotation mark
    "ΓÇÿ": "'",   # left single quotation mark
    "ΓÇ£": '"',   # left double quotation mark
    "ΓÇ¥": '"',   # right double quotation mark
    "ΓÇô": "–",   # en dash
    "ΓÇö": "—",   # em dash
    "ΓÇó": "·",   # bullet / middle dot
    "ΓÇñ": "-",   # hyphen
    "GÇÖ": "'",   # alternate encoding of right single quote
    "GÇô": "–",   # alternate en dash
}

# Unicode private-use area and other non-printable non-ASCII chars to strip.
_PRIVATE_USE_PATTERN = re.compile(r"[-]")


def _clean_text(text: Optional[str], max_chars: int = MAX_JD_CHARS) -> str:
    """
    Strip HTML tags, normalise whitespace, fix garbled UTF-8, and truncate JD text.

    HTML stripping: some sources return jd_text with raw HTML markup
    (<p>, <ul>, <li>, <strong> etc). Tags are stripped before length
    gating so they don't inflate char counts or pollute Claude input.

    Mojibake fix: cp1252-decoded-as-UTF-8 sequences and
    private-use-area Unicode bullets are replaced/stripped before storage.
    """
    if not text:
        return ""
    # Fix mojibake sequences first (before any other processing)
    for garbled, replacement in _MOJIBAKE_MAP.items():
        text = text.replace(garbled, replacement)
    # Strip private-use area Unicode characters
    text = _PRIVATE_USE_PATTERN.sub("", text)
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse multiple whitespace into single spaces
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()[:max_chars]


def _is_expired(raw: dict, job_title: str, company: str) -> bool:
    """
    N-2 gate -- expired job listing.

    JSearch returns job_offer_expiration_datetime_utc on most listings.
    If the expiry date is in the past (relative to UTC now), the role
    is likely filled or the posting is stale. Drop it before normalisation
    so it never costs a Haiku extraction call or a Cohere embedding slot.

    Returns True (expired, should drop) if expiry is present and past.
    Returns False (keep) if expiry is absent or in the future.
    Absence of expiry is treated as non-expired -- conservative choice,
    avoids dropping listings that simply don't set an expiry date.
    """
    expiry_str = raw.get("job_offer_expiration_datetime_utc")
    if not expiry_str:
        return False   # no expiry set -- treat as active

    expiry = _parse_date(expiry_str)
    if expiry is None:
        return False   # unparseable -- treat as active

    now = datetime.now(expiry.tzinfo or __import__("datetime").timezone.utc)
    if expiry < now:
        logger.info(
            f"[normalizer:drop] '{job_title}' @ {company} | "
            f"reason=expired | expiry={expiry_str}"
        )
        return True
    return False


# -- Quality gates -------------------------------------------------------------

def _is_jd_sufficient(jd_text: str, job_title: str, company: str) -> bool:
    """
    Gates 1+2+3 -- JD null / too short (chars) / too sparse (words).

    Returns False (drop) if:
    - jd_text is empty or whitespace-only
    - jd_text length after stripping is below MIN_JD_CHARS
    - jd_text word count is below MIN_JD_WORDS

    Gate rationale:
    - Char gate: drops truly empty stubs and one-liners.
    - Word gate: drops padded stubs where long words/punctuation inflate
      char count past the threshold but the JD still has no requirements section,
      no named skills, and no experience expectation. Such JDs produce unreliably
      inflated scores (experience_score -> 1.0, skill_score -> 1.0) because
      Claude finds no contradicting evidence in the near-empty text.
    """
    stripped = (jd_text or "").strip()
    if not stripped:
        logger.info(
            f"[normalizer:drop] '{job_title}' @ {company} | "
            f"reason=jd_empty"
        )
        return False
    if len(stripped) < MIN_JD_CHARS:
        logger.info(
            f"[normalizer:drop] '{job_title}' @ {company} | "
            f"reason=jd_too_short | chars={len(stripped)} < {MIN_JD_CHARS}"
        )
        return False
    word_count = len(stripped.split())
    if word_count < MIN_JD_WORDS:
        logger.info(
            f"[normalizer:drop] '{job_title}' @ {company} | "
            f"reason=jd_too_sparse | words={word_count} < {MIN_JD_WORDS}"
        )
        return False
    return True


def _is_english(jd_text: str, job_title: str, company: str) -> bool:
    """
    Gate 3 -- Non-English JD detection.

    Heuristic: if more than 30% of characters are non-ASCII (Unicode),
    the JD is likely in a non-Latin script (Hindi, Chinese, Arabic, etc.)
    and will produce unreliable extraction across all scoring dimensions.

    No external library -- pure character ratio check.
    Threshold chosen conservatively: English JDs with a few Unicode chars
    (e.g. bullets, em-dashes, salary symbols) stay well below 30%.
    A predominantly Hindi JD using Devanagari script will exceed 60%+.

    Mixed Hindi-English JDs (common on Indian boards): if enough Devanagari
    is present to exceed 30%, the JD's extractable signal is unreliable.
    Drop it. If it's mostly English with a Hindi sentence or two (<30%),
    pass it -- Claude handles mixed text well.
    """
    if not jd_text:
        return False

    total     = len(jd_text)
    non_ascii = len(_NON_ASCII_PATTERN.findall(jd_text))
    ratio     = non_ascii / total if total > 0 else 0

    if ratio > 0.30:
        logger.info(
            f"[normalizer:drop] '{job_title}' @ {company} | "
            f"reason=non_english | non_ascii_ratio={ratio:.2f}"
        )
        return False
    return True


# -- JSearch normalizer --------------------------------------------------------

def normalize_jsearch(raw: dict, matched_profile: str) -> Optional[RawJob]:
    """
    Normalize a JSearch API job dict into RawJob.

    Quality gates applied in order:
      1. Required fields present (job_id, title, apply_url)
      2. JD not null/empty and >= MIN_JD_CHARS
      3. JD is English (non-ASCII ratio <= 30%)

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

        # Gate 0 -- required structural fields
        if not job_id or not title or not apply_url:
            return None

        # Gate 0.5 -- expired listing (N-2)
        if _is_expired(raw, title, company):
            return None

        # Extract and clean JD before gates (N-1: HTML stripped inside _clean_text)
        raw_jd  = raw.get("job_description") or ""
        jd_text = _clean_text(raw_jd)

        # Gate 1 + 2 -- JD sufficient
        if not _is_jd_sufficient(jd_text, title, company):
            return None

        # Gate 3 -- English
        if not _is_english(jd_text, title, company):
            return None

        # Location
        city     = raw.get("job_city")    or ""
        country  = raw.get("job_country") or ""
        location = ", ".join(filter(None, [city, country])) or "Not specified"

        # Work type -- binary
        is_remote = raw.get("job_is_remote", False)
        work_type = "remote" if is_remote else "office"

        return RawJob(
            job_id          = _make_job_id("jsearch", job_id),
            title           = title,
            company         = company,
            location        = location,
            work_type       = work_type,
            jd_text         = jd_text,
            apply_url       = apply_url,
            source          = "jsearch",
            posted_date     = _parse_date(raw.get("job_posted_at_datetime_utc")),
            matched_profile = matched_profile,
        )

    except Exception as e:
        logger.warning(f"[normalizer] Failed to normalize JSearch job: {e}")
        return None


# -- RemoteOK normalizer -------------------------------------------------------

def normalize_remoteok(raw: dict, matched_profile: str) -> Optional[RawJob]:
    """
    Normalize a RemoteOK job dict into RawJob.

    Same quality gates as JSearch apply.

    RemoteOK field reference:
      id, position, company, url, description,
      date, salary_min, salary_max, tags, location
    """
    try:
        job_id    = str(raw.get("id")       or "")
        title     = raw.get("position")     or ""
        company   = raw.get("company")      or ""
        apply_url = raw.get("url")          or ""

        if not job_id or not title or not apply_url:
            return None

        raw_jd  = raw.get("description") or ""
        jd_text = _clean_text(raw_jd)

        if not _is_jd_sufficient(jd_text, title, company):
            return None

        if not _is_english(jd_text, title, company):
            return None

        return RawJob(
            job_id          = _make_job_id("remoteok", job_id),
            title           = title,
            company         = company,
            location        = raw.get("location") or "Remote",
            work_type       = "remote",
            jd_text         = jd_text,
            apply_url       = apply_url,
            source          = "remoteok",
            posted_date     = _parse_date(raw.get("date")),
            matched_profile = matched_profile,
        )

    except Exception as e:
        logger.warning(f"[normalizer] Failed to normalize RemoteOK job: {e}")
        return None


# -- Active Jobs DB normalizer -------------------------------------------------

def normalize_active_jobs_db(raw: dict, matched_profile: str) -> Optional[RawJob]:
    """
    Normalize an Active Jobs DB (Fantastic Jobs) response object into RawJob.

    Active Jobs DB uses the Fantastic Jobs schema -- same field names as
    LinkedIn Job Search API (both are Fantastic Jobs products):
      id, title, organization, description (FULL text when description_type=text),
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

        # Full JD returned when description_type=text is passed in request
        jd_text = _clean_text(raw.get("description") or "")

        if not _is_jd_sufficient(jd_text, title, company):
            return None
        if not _is_english(jd_text, title, company):
            return None

        # Location from locations_raw list
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
            job_id          = _make_job_id("active_jobs_db", job_id),
            title           = title,
            company         = company,
            location        = location,
            work_type       = work_type,
            jd_text         = jd_text,
            apply_url       = apply_url,
            source          = "active_jobs_db",
            posted_date     = _parse_date(raw.get("date_posted")),
            matched_profile = matched_profile,
        )

    except Exception as e:
        logger.warning(f"[normalizer] Failed to normalize Active Jobs DB job: {e}")
        return None


# -- LinkedIn Job Search API normalizer ----------------------------------------

def normalize_linkedin_jobs(raw: dict, matched_profile: str) -> Optional[RawJob]:
    """
    Normalize a LinkedIn Job Search API response object into RawJob.

    LinkedIn Job Search API is also a Fantastic Jobs product, same schema
    as Active Jobs DB:
      id, title, organization, description (FULL -- same field as Active Jobs DB),
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

        # Field is 'description' not 'description_text'
        jd_text = _clean_text(raw.get("description") or "")

        if not _is_jd_sufficient(jd_text, title, company):
            return None
        if not _is_english(jd_text, title, company):
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
            job_id          = _make_job_id("linkedin_jobs", job_id),
            title           = title,
            company         = company,
            location        = location,
            work_type       = work_type,
            jd_text         = jd_text,
            apply_url       = apply_url,
            source          = "linkedin_jobs",
            posted_date     = _parse_date(raw.get("date_posted")),
            matched_profile = matched_profile,
        )

    except Exception as e:
        logger.warning(f"[normalizer] Failed to normalize LinkedIn Jobs job: {e}")
        return None


# -- Techmap normalizer --------------------------------------------------------

def normalize_techmap(raw: dict, matched_profile: str) -> Optional[RawJob]:
    """
    Normalize a Techmap Daily International Job Postings result into RawJob.

    Techmap wraps job data in a JSON-LD sub-object (raw.jsonLD):
      jsonLD.title, jsonLD.description (FULL), jsonLD.url,
      jsonLD.datePosted, jsonLD.hiringOrganization.name,
      jsonLD.jobLocation.address.addressLocality,
      jsonLD.baseSalary.value.{minValue,maxValue}
    Top-level shortcuts: raw.city, raw.company, raw.workPlace
    """
    try:
        jld       = raw.get("jsonLD") or {}
        title     = (jld.get("title") or raw.get("occupation") or "").strip()
        apply_url = (jld.get("url") or "").strip()
        company   = (
            (jld.get("hiringOrganization") or {}).get("name") or
            raw.get("company") or ""
        ).strip()

        if not title or not apply_url:
            return None

        jd_text = _clean_text(jld.get("description") or "")

        if not _is_jd_sufficient(jd_text, title, company):
            return None
        if not _is_english(jd_text, title, company):
            return None

        raw_id = (
            str(jld.get("identifier") or "") or
            hashlib.md5(f"{title}{company}{apply_url}".encode()).hexdigest()[:12]
        )

        job_location = jld.get("jobLocation") or {}
        address      = job_location.get("address") or {}
        city         = (address.get("addressLocality") or raw.get("city") or "").strip()
        country      = (address.get("addressCountry") or "India").strip()
        location     = ", ".join(filter(None, [city, country])) or "India"

        work_place_raw = (raw.get("workPlace") or "").lower()
        if work_place_raw == "remote":
            work_type = "remote"
        elif work_place_raw == "hybrid":
            work_type = "hybrid"
        else:
            work_type = "office"

        posted_date = None
        date_str    = jld.get("datePosted") or raw.get("dateCreated")
        if date_str:
            posted_date = _parse_date(str(date_str)[:10])

        return RawJob(
            job_id          = _make_job_id("techmap", raw_id),
            title           = title,
            company         = company,
            location        = location,
            work_type       = work_type,
            jd_text         = jd_text,
            apply_url       = apply_url,
            source          = "techmap",
            posted_date     = posted_date,
            matched_profile = matched_profile,
        )

    except Exception as e:
        logger.warning(f"[normalizer] Failed to normalize Techmap job: {e}")
        return None


# -- Jobs Search API normalizer ------------------------------------------------

def normalize_jobs_search_api(raw: dict, matched_profile: str) -> Optional[RawJob]:
    """
    Normalize a Jobs Search API (JobSpy-style) response object into RawJob.

    This API returns JobSpy schema (NOT Fantastic Jobs schema):
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

        # Full JD -- populated when linkedin_fetch_description=True in request
        jd_text = _clean_text(raw.get("description") or "")

        if not _is_jd_sufficient(jd_text, title, company):
            return None
        if not _is_english(jd_text, title, company):
            return None

        location  = (raw.get("location") or "Not specified").strip()
        work_type = "remote" if raw.get("is_remote", False) else "office"

        return RawJob(
            job_id          = _make_job_id("jobs_search_api", job_id),
            title           = title,
            company         = company,
            location        = location,
            work_type       = work_type,
            jd_text         = jd_text,
            apply_url       = apply_url,
            source          = "jobs_search_api",
            posted_date     = _parse_date(raw.get("date_posted")),
            matched_profile = matched_profile,
        )

    except Exception as e:
        logger.warning(f"[normalizer] Failed to normalize Jobs Search API job: {e}")
        return None


# -- Dispatcher ----------------------------------------------------------------

NORMALIZERS = {
    "jsearch":          normalize_jsearch,
    "remoteok":         normalize_remoteok,
    "active_jobs_db":   normalize_active_jobs_db,
    "linkedin_jobs":    normalize_linkedin_jobs,
    "techmap":          normalize_techmap,
    "jobs_search_api":  normalize_jobs_search_api,
}


def normalize_jobs(
    raw_jobs:        list[dict],
    source:          str,
    matched_profile: str,
) -> list[RawJob]:
    """
    Normalize a list of raw job dicts from a given source.

    Quality gates run inside each normalizer -- jobs that fail any gate
    return None and are silently excluded from the result.
    Logs a summary per source for LangSmith evaluation.

    Args:
        raw_jobs:        raw API response list
        source:          "jsearch" | "remoteok" | "active_jobs_db" | "linkedin_jobs" | "techmap" | "jobs_search_api"
        matched_profile: which confirmed profile triggered this search

    Returns:
        List of valid RawJob objects that passed all quality gates.
    """
    normalizer = NORMALIZERS.get(source)
    if not normalizer:
        logger.error(f"[normalizer] Unknown source: {source}")
        return []

    results = []
    dropped = 0

    for raw in raw_jobs:
        job = normalizer(raw, matched_profile)
        if job:
            results.append(job)
        else:
            dropped += 1

    logger.info(
        f"[normalizer] {source}: "
        f"{len(results)} passed quality gates, "
        f"{dropped} dropped | "
        f"profile='{matched_profile}'"
    )
    return results
