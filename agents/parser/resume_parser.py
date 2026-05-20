"""
agents/parser/resume_parser.py

Agent 1 -- Resume Parser

Responsibility:
  - Accept raw file bytes (PDF or DOCX)
  - Extract plain text from the file
  - Call Claude API with a strict JSON schema prompt
  - Return a validated CandidateProfile object
  - Handle parse failures gracefully with a fallback path

Input  (from SessionState): resume_raw_text or file bytes + file_type
Output (to SessionState)  : candidate_profile | parse_failed + parse_failure_reason
"""

from __future__ import annotations

import json
import logging
import re
import time
from io import BytesIO
from typing import Optional

import anthropic
import fitz  # PyMuPDF
import docx  # python-docx
from langsmith import traceable
from tenacity import retry, stop_after_attempt, wait_exponential

from core.prompts.parser_prompts import (
    RESUME_PARSE_FALLBACK_PROMPT,
    RESUME_PARSE_PROMPT,
)
from core.config.config_loader import cfg
from core.state.session_state import (
    CandidateProfile,
    CareerGap,
    Education,
    SessionState,
    SkillSet,
    WorkExperience,
)

logger = logging.getLogger(__name__)

# -- Constants -----------------------------------------------------------------
# LLM parameters loaded from core/config/llm_config.yaml -- edit there, not here.

_CFG            = cfg.resume_parser
MIN_TEXT_LENGTH = _CFG.min_text_length   # loaded from llm_config.yaml [resume_parser]

# Markdown-link cleanup
# PyMuPDF dict-mode and python-docx occasionally emit hyperlinks in
# Markdown form, e.g. '[B.Tech](http://B.Tech)' or
# '[name@x.com](mailto:name@x.com)'. The label is the visible text that
# should remain in the resume; the URL is noise that pollutes the LLM
# prompt and the user-visible scoring_notes downstream. This is a generic,
# domain-agnostic cleanup -- works for any resume content.
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _strip_markdown_links(text: str) -> str:
    """
    Replace Markdown link syntax '[label](url)' with just 'label'.
    Domain-agnostic. Handles emails, URLs, and any other linked text.
    """
    if not text:
        return text
    return _MD_LINK_RE.sub(r"\1", text)


# -- Text extraction ----------------------------------------------------------

def _extract_layout_aware_text(page) -> str:
    """
    Extract text from a single PDF page using layout geometry.

    Uses font size and y-coordinate signals -- no hardcoded assumptions
    about section names, bullet characters, or resume structure.

    Logic:
      - A new line is detected when y-coordinate increases by more than
        the median line height (relative threshold, not a fixed pixel value).
      - A blank line separator is inserted when the y-gap is more than
        2x the median line height -- indicating a visual gap in the layout.
      - Spans on the same line are joined with a space.
      - Non-printable / zero-width characters are stripped.

    This is purely geometric -- it works for any document layout.
    """
    blocks = page.get_text("dict")["blocks"]

    # Collect all (y0, x0, text, size) from every span across all blocks
    spans = []
    for block in blocks:
        if block.get("type") != 0:   # type 0 = text, type 1 = image
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                spans.append({
                    "y":    round(span["origin"][1], 1),
                    "x":    round(span["origin"][0], 1),
                    "size": round(span["size"], 1),
                    "text": text,
                })

    if not spans:
        return ""

    # Sort by vertical position then horizontal position
    spans.sort(key=lambda s: (s["y"], s["x"]))

    # Compute median line height from unique y values
    y_values = sorted(set(s["y"] for s in spans))
    if len(y_values) > 1:
        gaps = [y_values[i+1] - y_values[i] for i in range(len(y_values)-1)]
        gaps = [g for g in gaps if g > 0]
        median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 12.0
    else:
        median_gap = 12.0

    # Group spans into lines by y-coordinate proximity
    lines: list[list[dict]] = []
    current_line: list[dict] = []
    prev_y = None

    for span in spans:
        if prev_y is None or abs(span["y"] - prev_y) <= median_gap * 0.6:
            current_line.append(span)
        else:
            if current_line:
                lines.append(current_line)
            current_line = [span]
        prev_y = span["y"]

    if current_line:
        lines.append(current_line)

    # Build output -- insert blank line when vertical gap > 2x median
    output_lines: list[str] = []
    prev_line_y = None

    for line_spans in lines:
        line_y = line_spans[0]["y"]

        if prev_line_y is not None:
            gap = line_y - prev_line_y
            if gap > median_gap * 2.0:
                output_lines.append("")   # blank line = visual section gap

        line_text = " ".join(s["text"] for s in line_spans)
        # Strip non-printable artifacts (e.g. , \x00, zero-width chars)
        line_text = "".join(c for c in line_text if c.isprintable() and ord(c) > 31)
        line_text = line_text.strip()

        if line_text:
            output_lines.append(line_text)

        prev_line_y = line_y

    return "\n".join(output_lines)


# -- User-facing parse-failure messages ----------------------------------------
# These strings are returned verbatim to the frontend (parse_failure_reason).
# Keep them short, actionable, and free of library-internal jargon.

ERR_PDF_ENCRYPTED = (
    "This PDF is password-protected. "
    "Please remove the password (open it in a PDF reader, then File > Save As "
    "without encryption) and re-upload."
)
ERR_PDF_NO_TEXT = (
    "We couldn't extract any text from this PDF -- it looks like a scanned "
    "image or photo of a resume. Please upload a text-based PDF (export "
    "directly from Word, Google Docs, LinkedIn, or your resume builder) "
    "or a DOCX file instead."
)
ERR_PDF_UNREADABLE = (
    "We couldn't read this PDF. The file may be corrupted -- please try "
    "re-exporting it from your resume editor and upload again."
)
ERR_DOCX_UNREADABLE = (
    "We couldn't read this DOCX file. It may be corrupted or use an "
    "unsupported format -- please re-save it from Word and try again."
)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from PDF bytes using layout-aware geometry (PyMuPDF dict mode).

    Falls back to plain text extraction if dict mode returns nothing
    (e.g. corrupted or unusual PDF structure).

    Raises ValueError with a user-friendly message for:
      - password-protected / encrypted PDFs
      - scanned / image-only PDFs (no extractable text)
      - corrupted or otherwise unreadable PDFs
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        logger.error(f"[parser] PDF open failed: {type(e).__name__}: {e}")
        raise ValueError(ERR_PDF_UNREADABLE) from e

    # Encrypted / password-protected PDFs
    # PyMuPDF: needs_pass=True until authenticate() succeeds.
    if getattr(doc, "needs_pass", False) or getattr(doc, "is_encrypted", False):
        try:
            doc.close()
        except Exception:
            pass
        logger.warning("[parser] Rejected encrypted PDF upload")
        raise ValueError(ERR_PDF_ENCRYPTED)

    try:
        pages_text = []
        for page in doc:
            page_text = _extract_layout_aware_text(page)
            if page_text:
                pages_text.append(page_text)
        doc.close()

        text = "\n\n".join(pages_text)

        # Fallback to plain text if layout extraction yielded nothing
        if not text.strip():
            logger.warning(
                "[parser] Layout extraction yielded empty result -- "
                "falling back to plain text"
            )
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            pages = [page.get_text("text") for page in doc]
            doc.close()
            text = "\n".join(pages)

        # Both passes produced no usable text -- almost certainly a scanned PDF.
        # We don't have OCR in the pipeline; tell the user clearly.
        if len(text.strip()) < 20:
            logger.warning(
                f"[parser] PDF yielded only {len(text.strip())} chars after "
                f"both extraction passes -- treating as scanned/image-only"
            )
            raise ValueError(ERR_PDF_NO_TEXT)

        # Strip Markdown link syntax that PyMuPDF leaks for hyperlinks.
        # Domain-agnostic -- runs on any extracted resume text.
        text = _strip_markdown_links(text)

        logger.debug(f"PDF extracted: {len(text)} chars across {len(pages_text)} pages")
        return text

    except ValueError:
        # Already a user-friendly error -- don't re-wrap.
        raise
    except Exception as e:
        logger.error(f"[parser] PDF extraction failed: {type(e).__name__}: {e}")
        raise ValueError(ERR_PDF_UNREADABLE) from e


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract plain text from DOCX bytes using python-docx."""
    try:
        document = docx.Document(BytesIO(file_bytes))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs)
        # Strip Markdown link syntax just in case the source DOCX or any
        # upstream conversion produced inline '[label](url)' patterns.
        # Domain-agnostic.
        text = _strip_markdown_links(text)
        logger.debug(f"DOCX extracted: {len(text)} chars, {len(paragraphs)} paragraphs")
        return text
    except Exception as e:
        logger.error(f"[parser] DOCX extraction failed: {type(e).__name__}: {e}")
        raise ValueError(ERR_DOCX_UNREADABLE) from e


def extract_text(file_bytes: bytes, file_type: str) -> str:
    """
    Route to the correct extractor based on file type.
    Raises ValueError (with a user-friendly message) for unsupported types.

    Note: legacy ".doc" is intentionally NOT supported -- python-docx only
    handles the modern OOXML ".docx" container. The API layer rejects .doc
    uploads before they reach here (see api/dependencies.py).
    """
    file_type = file_type.lower().strip(".")
    if file_type == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif file_type == "docx":
        return extract_text_from_docx(file_bytes)
    else:
        raise ValueError(
            f"Unsupported file type: '.{file_type}'. Only PDF and DOCX are accepted."
        )


# -- Experience calculator -----------------------------------------------------

def _calculate_duration_months(start: Optional[str], end: Optional[str]) -> Optional[int]:
    """
    Calculate duration in months between two YYYY-MM strings.
    If end is None, uses today's date (role is current).
    Returns None if start is unparseable.
    """
    import datetime
    if not start:
        return None
    try:
        start_dt = datetime.date(int(start[:4]), int(start[5:7]), 1)
        if end:
            end_dt = datetime.date(int(end[:4]), int(end[5:7]), 1)
        else:
            end_dt = datetime.date.today().replace(day=1)
        months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
        return max(0, months)
    except (ValueError, IndexError):
        return None


def _calculate_experience_split(
    work_experience: list,
) -> tuple[float, float]:
    """
    Calculate full-time and other experience years separately.

    Uses role_type field set by Claude -- LLM-classified per role using
    title + company context, avoiding brittle keyword matching.

    Role type buckets:
      "full_time"  -> permanent employment -> counted in full_time total
      "internship" -> intern/trainee/placement -> counted in other total
      "other"      -> freelance/contract/part-time -> counted in other total

    Both buckets use merge-interval arithmetic to handle overlapping
    concurrent roles correctly (two simultaneous freelance contracts
    don't double-count time in the "other" bucket).

    Fallback for roles with no start_date: excluded from both totals
    and logged. P-5 guard: if full_time calculates to 0.0 but non-
    internship roles are present with duration_months set, use the
    sum of those durations as a fallback estimate.

    Returns:
        (full_time_years, other_years) -- both rounded to 1 decimal place.
    """
    import datetime

    def _to_interval(exp) -> Optional[tuple]:
        if not exp.start_date:
            return None
        try:
            s = datetime.date(int(exp.start_date[:4]), int(exp.start_date[5:7]), 1)
            e = (
                datetime.date(int(exp.end_date[:4]), int(exp.end_date[5:7]), 1)
                if exp.end_date
                else datetime.date.today().replace(day=1)
            )
            return (s, e) if e > s else None
        except (ValueError, IndexError, AttributeError):
            return None

    def _merge_and_sum_years(intervals: list) -> float:
        if not intervals:
            return 0.0
        intervals.sort(key=lambda x: x[0])
        merged = [list(intervals[0])]
        for s, e in intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        total_days = sum((e - s).days for s, e in merged)
        return round(total_days / 365.25, 1)

    fulltime_intervals = []
    other_intervals    = []
    skipped            = 0

    for exp in work_experience:
        interval = _to_interval(exp)
        if interval is None:
            skipped += 1
            logger.debug(
                f"[parser:exp] No parseable start_date for '{exp.title}' "
                f"@ {exp.company} -- excluded from experience totals"
            )
            continue

        role_type = getattr(exp, "role_type", "full_time") or "full_time"
        if role_type == "full_time":
            fulltime_intervals.append(interval)
        else:
            other_intervals.append(interval)
            logger.debug(
                f"[parser:exp] '{exp.title}' @ {exp.company} -> "
                f"role_type={role_type}, counted in 'other' bucket"
            )

    if skipped:
        logger.warning(
            f"[parser:exp] {skipped} role(s) excluded from experience totals "
            f"due to missing/unparseable start_date"
        )

    full_time_years = _merge_and_sum_years(fulltime_intervals)
    other_years     = _merge_and_sum_years(other_intervals)

    # -- P-5 guard: fallback when interval calc returns 0 ---------------------
    # If full_time_years = 0 but full_time roles exist with duration_months set,
    # sum those durations as a fallback rather than passing 0 to the ranker.
    if full_time_years == 0.0 and fulltime_intervals == [] and work_experience:
        fallback_months = sum(
            exp.duration_months or 0
            for exp in work_experience
            if (getattr(exp, "role_type", "full_time") or "full_time") == "full_time"
            and exp.duration_months
        )
        if fallback_months > 0:
            full_time_years = round(fallback_months / 12, 1)
            logger.warning(
                f"[parser:exp] Interval calculation returned 0 for full_time roles -- "
                f"falling back to duration_months sum: {full_time_years}yr"
            )

    return full_time_years, other_years


@traceable(
    name="claude-resume-parser",
    run_type="llm",
    metadata={"model": cfg.resume_parser.model, "agent": "resume_parser"},
)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def call_claude_for_parse(resume_text: str, use_fallback: bool = False):
    """
    Call Claude API to extract structured profile from resume text.
    Retries up to 3 times with exponential backoff on transient failures.
    Returns (raw_json_str, response) tuple -- caller reads response.usage for metrics.
    """
    client = anthropic.Anthropic()

    prompt_template = RESUME_PARSE_FALLBACK_PROMPT if use_fallback else RESUME_PARSE_PROMPT
    prompt = prompt_template.replace("{resume_text}", resume_text)

    logger.info(f"Calling Claude for resume parse (fallback={use_fallback})")

    response = client.messages.create(
        model       = _CFG.model,
        max_tokens  = _CFG.max_tokens,
        temperature = _CFG.temperature,
        messages    = [{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    logger.debug(f"Claude response length: {len(raw)} chars")
    return raw, response


# -- JSON cleaning & validation ------------------------------------------------

def clean_json_response(raw: str) -> str:
    """
    Strip markdown fences and extract a parseable JSON object.

    Handles four cases:
    1. Clean JSON object -- returned as-is
    2. Fenced JSON -- fence lines stripped then parsed
    3. JSON content without outer braces -- wraps in {} and parses
       (happens when Claude returns object body without the {} wrapper)
    4. JSON with preamble -- finds first { ... last } boundary
    """
    # Strip fence lines by line (avoids MULTILINE regex edge cases)
    lines = raw.split("\n")
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    raw = "\n".join(lines).strip()

    # Case 1: parses cleanly as-is
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass

    # Case 3: content without outer braces
    # Claude occasionally returns the object body without the {} wrapper,
    # producing responses like: \n  "current_title": "Data Scientist", ...
    if not raw.startswith("{") and raw.strip().startswith('"'):
        candidate = "{" + raw
        if not candidate.rstrip().endswith("}"):
            candidate = candidate + "}"
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Case 2 / 4: find first { ... last } boundary
    obj_start = raw.find("{")
    arr_start = raw.find("[")

    if obj_start == -1 and arr_start == -1:
        return raw  # no JSON found -- caller raises clear error

    if obj_start == -1:
        start = arr_start
        end = raw.rfind("]") + 1
    elif arr_start == -1:
        start = obj_start
        end = raw.rfind("}") + 1
    else:
        start = min(obj_start, arr_start)
        if start == obj_start:
            end = raw.rfind("}") + 1
        else:
            end = raw.rfind("]") + 1

    if end <= start:
        return raw

    return raw[start:end]


def parse_profile_from_json(raw_json: str, resume_text: str) -> CandidateProfile:
    """
    Parse the LLM JSON response into a validated CandidateProfile.
    Raises ValueError if JSON is malformed or fails schema validation.
    """
    cleaned = clean_json_response(raw_json)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM returned invalid JSON: {e}\n"
            f"Cleaned response (first 300 chars): {cleaned[:300]}"
        ) from e

    # Build nested Pydantic objects defensively
    skills_data = data.get("skills") or {}
    skills = SkillSet(
        technical=skills_data.get("technical") or [],
        tools=skills_data.get("tools") or [],
        soft=skills_data.get("soft") or [],
    )

    education = [
        Education(
            degree      = edu.get("degree", ""),
            field       = edu.get("field") or edu.get("major"),  # Claude sometimes returns "major"
            institution = edu.get("institution", ""),
            year        = edu.get("year"),
        )
        for edu in (data.get("education") or [])
        if isinstance(edu, dict) and edu.get("degree") and edu.get("institution")
    ]

    work_experience = [
        WorkExperience(**exp)
        for exp in (data.get("work_experience") or [])
        if isinstance(exp, dict)
    ]

    # Calculate duration_months per role in Python -- deterministic arithmetic
    for exp in work_experience:
        if exp.duration_months is None:
            exp.duration_months = _calculate_duration_months(exp.start_date, exp.end_date)

    # Split experience by role_type -- LLM-classified, Python-calculated
    full_time_years, other_years = _calculate_experience_split(work_experience)

    career_gaps = [
        CareerGap(**gap)
        for gap in (data.get("career_gaps") or [])
        if isinstance(gap, dict)
    ]

    profile = CandidateProfile(
        current_title               = data.get("current_title"),
        years_experience_full_time  = full_time_years,
        years_experience_other      = other_years,
        seniority_level             = data.get("seniority_level")
                                      if data.get("seniority_level") in
                                      {"intern","junior","mid","senior","lead","principal","executive",None}
                                      else None,
        ats_summary                 = data.get("ats_summary") or None,
        skills                      = skills,
        education                   = education,
        work_experience             = work_experience,
        career_trajectory           = data.get("career_trajectory")
                                      if data.get("career_trajectory") in
                                      {"ascending","lateral","pivot","re-entry",None}
                                      else None,
        pivot_signals               = data.get("pivot_signals") or [],
        domain_expertise            = data.get("domain_expertise") or [],
        career_gaps                 = career_gaps,
        raw_text                    = resume_text,
    )

    return profile


# -- Main agent function -------------------------------------------------------

@traceable(name="agent-1-resume-parser", run_type="chain")
def run_resume_parser(
    state: SessionState,
    file_bytes: Optional[bytes] = None,
    file_type: Optional[str] = None,
) -> SessionState:
    """
    Agent 1 -- Resume Parser.

    Accepts either:
      - file_bytes + file_type: extracts text then parses
      - state.resume_raw_text already set: skips extraction, goes straight to LLM

    Updates SessionState with:
      - candidate_profile (on success)
      - parse_failed + parse_failure_reason (on failure)

    Returns updated SessionState.
    """
    state.current_agent = "resume_parser"
    logger.info(f"[resume_parser] Starting -- session_id={state.session_id}")
    _t0 = time.perf_counter()
    _last_response = None   # captured for token usage metrics

    # -- Step 1: Get raw text ------------------------------------------------
    resume_text = state.resume_raw_text

    if not resume_text and file_bytes and file_type:
        try:
            resume_text = extract_text(file_bytes, file_type)
            state.resume_raw_text = resume_text
            logger.info(f"[resume_parser] Extracted {len(resume_text)} chars from {file_type}")
        except ValueError as e:
            state.parse_failed = True
            state.parse_failure_reason = str(e)
            logger.error(f"[resume_parser] Text extraction failed: {e}")
            return state

    if not resume_text:
        state.parse_failed = True
        state.parse_failure_reason = "No resume text available. Upload a PDF/DOCX or paste plain text."
        return state

    # -- Step 2: Check minimum length ----------------------------------------
    if len(resume_text.strip()) < MIN_TEXT_LENGTH:
        state.parse_failed = True
        state.parse_failure_reason = (
            f"Extracted text is too short ({len(resume_text)} chars). "
            "The file may be a scanned image or heavily formatted. "
            "Please paste your resume as plain text."
        )
        logger.warning(f"[resume_parser] Text too short: {len(resume_text)} chars")
        return state

    # -- Step 3: LLM parse ---------------------------------------------------
    try:
        raw_json, _last_response = call_claude_for_parse(resume_text, use_fallback=False)
        profile = parse_profile_from_json(raw_json, resume_text)

    except (ValueError, json.JSONDecodeError) as e:
        logger.warning(f"[resume_parser] Primary parse failed ({e}), trying fallback prompt")
        try:
            raw_json, _last_response = call_claude_for_parse(resume_text, use_fallback=True)
            profile = parse_profile_from_json(raw_json, resume_text)
        except Exception as fallback_err:
            state.parse_failed = True
            state.parse_failure_reason = (
                f"Could not extract structured data from resume. "
                f"Error: {fallback_err}"
            )
            logger.error(f"[resume_parser] Fallback parse also failed: {fallback_err}")
            return state

    except Exception as e:
        state.parse_failed = True
        state.parse_failure_reason = f"Unexpected error during parsing: {e}"
        logger.error(f"[resume_parser] Unexpected error: {e}", exc_info=True)
        return state

    # -- Step 4: Write to state -----------------------------------------------
    state.candidate_profile = profile
    state.parse_failed = False
    state.parse_failure_reason = None

    # -- Step 5: Record observability metrics ---------------------------------
    elapsed = round(time.perf_counter() - _t0, 2)
    if _last_response:
        usage   = getattr(_last_response, "usage", None)
        in_tok  = getattr(usage, "input_tokens",  0) if usage else 0
        out_tok = getattr(usage, "output_tokens", 0) if usage else 0
        state.agent_metrics["resume_parser"] = {
            "model":         _CFG.model,
            "input_tokens":  in_tok,
            "output_tokens": out_tok,
            "llm_calls":     1,
            "latency_secs":  elapsed,
        }
        logger.info(
            f"[resume_parser] tokens={in_tok}in/{out_tok}out | "
            f"latency={elapsed}s"
        )

    logger.info(
        f"[resume_parser] Success -- "
        f"title={profile.current_title}, "
        f"seniority={profile.seniority_level}, "
        f"skills={len(profile.skills.technical)} technical"
    )

    return state
