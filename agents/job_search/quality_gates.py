"""
agents/job_search/quality_gates.py

Shared quality gates and text utilities used by all job search source normalizers.
Previously part of normalizer.py — extracted so each source file can be self-contained
without duplicating this logic.

Quality gates applied by every normalizer:
  1. Required structural fields present (job_id, title, apply_url)
  2. JD not null / empty / below MIN_JD_CHARS
  3. JD word count not below MIN_JD_WORDS
  4. JD is English (non-ASCII ratio <= 30%)
  5. Listing not expired (expiry date in the past)
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from core.config.config_loader import cfg

logger = logging.getLogger(__name__)

# -- Constants -----------------------------------------------------------------
MIN_JD_CHARS = cfg.job_search.min_jd_chars
MIN_JD_WORDS = cfg.job_search.min_jd_words
MAX_JD_CHARS = cfg.job_search.max_jd_chars

_LATIN_PATTERN     = re.compile(r"[a-zA-Z]")
_NON_ASCII_PATTERN = re.compile(r"[^\x00-\x7F]")
_PRIVATE_USE_PATTERN = re.compile(r"[-]")

# Garbled cp1252-as-UTF-8 sequences sometimes found in scraped job text.
_MOJIBAKE_MAP = {
    "ΓÇÖ": "'",
    "ΓÇÿ": "'",
    "ΓÇ£": '"',
    "ΓÇ¥": '"',
    "ΓÇô": "–",
    "ΓÇö": "—",
    "ΓÇó": "·",
    "ΓÇñ": "-",
    "GÇÖ": "'",
    "GÇô": "–",
}


# -- Helpers -------------------------------------------------------------------

def make_job_id(source: str, raw_id: str) -> str:
    """Generate a stable job ID from source + raw identifier."""
    return f"{source}_{raw_id}"


def make_job_id_hash(source: str, title: str, company: str, url: str) -> str:
    """Fallback job ID when the source provides no stable identifier."""
    return f"{source}_{hashlib.md5(f'{title}{company}{url}'.encode()).hexdigest()[:12]}"


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Safely parse an ISO date string into a timezone-aware datetime."""
    if not date_str:
        return None
    try:
        date_str = date_str.replace("Z", "+00:00")
        return datetime.fromisoformat(date_str)
    except (ValueError, AttributeError):
        return None


def clean_text(text: Optional[str], max_chars: int = MAX_JD_CHARS) -> str:
    """
    Strip HTML tags, fix garbled UTF-8, normalise whitespace, and truncate.
    """
    if not text:
        return ""
    for garbled, replacement in _MOJIBAKE_MAP.items():
        text = text.replace(garbled, replacement)
    text = _PRIVATE_USE_PATTERN.sub("", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()[:max_chars]


def is_expired(raw: dict, job_title: str, company: str) -> bool:
    """
    Returns True if the listing has an explicit past expiry date.
    Treats absent or unparseable expiry as non-expired (conservative).
    """
    expiry_str = raw.get("job_offer_expiration_datetime_utc")
    if not expiry_str:
        return False
    expiry = parse_date(expiry_str)
    if expiry is None:
        return False
    now = datetime.now(expiry.tzinfo or timezone.utc)
    if expiry < now:
        logger.info(
            f"[quality_gate:drop] '{job_title}' @ {company} | "
            f"reason=expired | expiry={expiry_str}"
        )
        return True
    return False


def is_jd_sufficient(jd_text: str, job_title: str, company: str) -> bool:
    """
    Returns False (drop) if JD is empty, below MIN_JD_CHARS, or below MIN_JD_WORDS.
    """
    stripped = (jd_text or "").strip()
    if not stripped:
        logger.info(
            f"[quality_gate:drop] '{job_title}' @ {company} | reason=jd_empty"
        )
        return False
    if len(stripped) < MIN_JD_CHARS:
        logger.info(
            f"[quality_gate:drop] '{job_title}' @ {company} | "
            f"reason=jd_too_short | chars={len(stripped)} < {MIN_JD_CHARS}"
        )
        return False
    word_count = len(stripped.split())
    if word_count < MIN_JD_WORDS:
        logger.info(
            f"[quality_gate:drop] '{job_title}' @ {company} | "
            f"reason=jd_too_sparse | words={word_count} < {MIN_JD_WORDS}"
        )
        return False
    return True


def is_english(jd_text: str, job_title: str, company: str) -> bool:
    """
    Returns False (drop) if more than 30% of characters are non-ASCII.
    Heuristic for non-Latin scripts (Devanagari, Chinese, Arabic, etc.).
    """
    if not jd_text:
        return False
    total     = len(jd_text)
    non_ascii = len(_NON_ASCII_PATTERN.findall(jd_text))
    ratio     = non_ascii / total if total > 0 else 0
    if ratio > 0.30:
        logger.info(
            f"[quality_gate:drop] '{job_title}' @ {company} | "
            f"reason=non_english | non_ascii_ratio={ratio:.2f}"
        )
        return False
    return True
