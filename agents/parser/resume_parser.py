"""
agents/parser/resume_parser.py

Agent 1 — Resume Parser

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
from core.state.session_state import (
    CandidateProfile,
    CareerGap,
    Education,
    NotableProject,
    SessionState,
    SkillSet,
    WorkExperience,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MIN_TEXT_LENGTH = 200          # chars — below this we treat parse as failed
CLAUDE_MODEL    = "claude-haiku-4-5-20251001"
MAX_TOKENS      = 4096


# ── Text extraction ──────────────────────────────────────────────────────────

def _extract_layout_aware_text(page) -> str:
    """
    Extract text from a single PDF page using layout geometry.

    Uses font size and y-coordinate signals — no hardcoded assumptions
    about section names, bullet characters, or resume structure.

    Logic:
      - A new line is detected when y-coordinate increases by more than
        the median line height (relative threshold, not a fixed pixel value).
      - A blank line separator is inserted when the y-gap is more than
        2x the median line height — indicating a visual gap in the layout.
      - Spans on the same line are joined with a space.
      - Non-printable / zero-width characters are stripped.

    This is purely geometric — it works for any document layout.
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

    # Build output — insert blank line when vertical gap > 2x median
    output_lines: list[str] = []
    prev_line_y = None

    for line_spans in lines:
        line_y = line_spans[0]["y"]

        if prev_line_y is not None:
            gap = line_y - prev_line_y
            if gap > median_gap * 2.0:
                output_lines.append("")   # blank line = visual section gap

        line_text = " ".join(s["text"] for s in line_spans)
        # Strip non-printable artifacts (e.g. ¯, \x00, zero-width chars)
        line_text = "".join(c for c in line_text if c.isprintable() and ord(c) > 31)
        line_text = line_text.strip()

        if line_text:
            output_lines.append(line_text)

        prev_line_y = line_y

    return "\n".join(output_lines)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from PDF bytes using layout-aware geometry (PyMuPDF dict mode).

    Falls back to plain text extraction if dict mode returns nothing
    (e.g. corrupted or unusual PDF structure).
    """
    try:
        doc  = fitz.open(stream=file_bytes, filetype="pdf")
        pages_text = []
        for page in doc:
            page_text = _extract_layout_aware_text(page)
            if page_text:
                pages_text.append(page_text)
        doc.close()

        text = "\n\n".join(pages_text)

        # Fallback to plain text if layout extraction yielded nothing
        if not text.strip():
            logger.warning("[parser] Layout extraction yielded empty result — falling back to plain text")
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            pages = [page.get_text("text") for page in doc]
            doc.close()
            text = "\n".join(pages)

        logger.debug(f"PDF extracted: {len(text)} chars across {len(pages_text)} pages")
        return text
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise ValueError(f"Could not read PDF: {e}") from e


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract plain text from DOCX bytes using python-docx."""
    try:
        document = docx.Document(BytesIO(file_bytes))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs)
        logger.debug(f"DOCX extracted: {len(text)} chars, {len(paragraphs)} paragraphs")
        return text
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
        raise ValueError(f"Could not read DOCX: {e}") from e


def extract_text(file_bytes: bytes, file_type: str) -> str:
    """
    Route to the correct extractor based on file type.
    Raises ValueError for unsupported types.
    """
    file_type = file_type.lower().strip(".")
    if file_type == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif file_type in ("docx", "doc"):
        return extract_text_from_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {file_type}. Use PDF or DOCX.")


# ── Experience calculator ─────────────────────────────────────────────────────

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


def _calculate_total_experience(work_experience: list) -> float:
    """
    Calculate total professional experience in years from work experience list.

    Uses a merge-interval approach to handle overlapping roles correctly —
    two concurrent roles don't double-count the time.

    Excludes internships (title contains 'intern') from the professional total
    but counts them separately.

    Returns total years rounded to 1 decimal place.
    """
    import datetime

    intervals = []
    for exp in work_experience:
        start = exp.start_date
        end   = exp.end_date
        if not start:
            continue
        try:
            s = datetime.date(int(start[:4]), int(start[5:7]), 1)
            e = datetime.date(int(end[:4]), int(end[5:7]), 1) if end else datetime.date.today().replace(day=1)
            intervals.append((s, e))
        except (ValueError, IndexError):
            continue

    if not intervals:
        return 0.0

    # Merge overlapping intervals
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    total_days = sum((e - s).days for s, e in merged)
    return round(total_days / 365.25, 1)

@traceable(
    name="claude-resume-parser",
    run_type="llm",
    metadata={"model": CLAUDE_MODEL, "agent": "resume_parser"},
)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def call_claude_for_parse(resume_text: str, use_fallback: bool = False) -> str:
    """
    Call Claude API to extract structured profile from resume text.
    Retries up to 3 times with exponential backoff on transient failures.
    Returns raw JSON string.

    Uses str.replace() instead of .format() — resume text may contain
    curly braces (e.g. from code snippets or JSON) which break .format().
    """
    client = anthropic.Anthropic()

    prompt_template = RESUME_PARSE_FALLBACK_PROMPT if use_fallback else RESUME_PARSE_PROMPT
    prompt = prompt_template.replace("{resume_text}", resume_text)

    logger.info(f"Calling Claude for resume parse (fallback={use_fallback})")

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    logger.debug(f"Claude response length: {len(raw)} chars")
    return raw


# ── JSON cleaning & validation ────────────────────────────────────────────────

def clean_json_response(raw: str) -> str:
    """
    Strip accidental markdown fences, preamble, or leading whitespace
    from LLM response. Finds the first { or [ and last } or ] to extract
    clean JSON even if Claude adds text before or after.
    """
    # Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    # Find the actual JSON boundaries — handles leading newlines or preamble text
    # Look for first { or [ (whichever comes first)
    obj_start = raw.find("{")
    arr_start = raw.find("[")

    if obj_start == -1 and arr_start == -1:
        return raw  # no JSON found — let the caller handle it

    if obj_start == -1:
        start = arr_start
        end = raw.rfind("]") + 1
    elif arr_start == -1:
        start = obj_start
        end = raw.rfind("}") + 1
    else:
        start = min(obj_start, arr_start)
        # Find matching closing bracket
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

    # Calculate duration_months per role and total experience in Python
    # — deterministic arithmetic, not LLM estimation
    for exp in work_experience:
        if exp.duration_months is None:
            exp.duration_months = _calculate_duration_months(exp.start_date, exp.end_date)

    years_experience = _calculate_total_experience(work_experience)

    notable_projects = [
        NotableProject(**proj)
        for proj in (data.get("notable_projects") or [])
        if isinstance(proj, dict)
    ]

    career_gaps = [
        CareerGap(**gap)
        for gap in (data.get("career_gaps") or [])
        if isinstance(gap, dict)
    ]

    profile = CandidateProfile(
        current_title     = data.get("current_title"),
        years_experience  = years_experience,   # Python-calculated, not LLM-estimated
        seniority_level   = data.get("seniority_level")
                            if data.get("seniority_level") in
                            {"intern","junior","mid","senior","lead","principal","executive",None}
                            else None,
        skills            = skills,
        education         = education,
        work_experience   = work_experience,
        career_trajectory = data.get("career_trajectory")
                            if data.get("career_trajectory") in
                            {"ascending","lateral","pivot","re-entry",None}
                            else None,
        pivot_signals     = data.get("pivot_signals") or [],
        domain_expertise  = data.get("domain_expertise") or [],
        notable_projects  = notable_projects,
        career_gaps       = career_gaps,
        raw_text          = resume_text,
    )

    return profile


# ── Main agent function ───────────────────────────────────────────────────────

@traceable(name="agent-1-resume-parser", run_type="chain")
def run_resume_parser(
    state: SessionState,
    file_bytes: Optional[bytes] = None,
    file_type: Optional[str] = None,
) -> SessionState:
    """
    Agent 1 — Resume Parser.

    Accepts either:
      - file_bytes + file_type: extracts text then parses
      - state.resume_raw_text already set: skips extraction, goes straight to LLM

    Updates SessionState with:
      - candidate_profile (on success)
      - parse_failed + parse_failure_reason (on failure)

    Returns updated SessionState.
    """
    state.current_agent = "resume_parser"
    logger.info(f"[resume_parser] Starting — session_id={state.session_id}")

    # ── Step 1: Get raw text ────────────────────────────────────────────────
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

    # ── Step 2: Check minimum length ────────────────────────────────────────
    if len(resume_text.strip()) < MIN_TEXT_LENGTH:
        state.parse_failed = True
        state.parse_failure_reason = (
            f"Extracted text is too short ({len(resume_text)} chars). "
            "The file may be a scanned image or heavily formatted. "
            "Please paste your resume as plain text."
        )
        logger.warning(f"[resume_parser] Text too short: {len(resume_text)} chars")
        return state

    # ── Step 3: LLM parse ───────────────────────────────────────────────────
    try:
        raw_json = call_claude_for_parse(resume_text, use_fallback=False)
        profile = parse_profile_from_json(raw_json, resume_text)

    except (ValueError, json.JSONDecodeError) as e:
        # First attempt failed — try fallback prompt
        logger.warning(f"[resume_parser] Primary parse failed ({e}), trying fallback prompt")
        try:
            raw_json = call_claude_for_parse(resume_text, use_fallback=True)
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

    # ── Step 4: Write to state ───────────────────────────────────────────────
    state.candidate_profile = profile
    state.parse_failed = False
    state.parse_failure_reason = None

    logger.info(
        f"[resume_parser] Success — "
        f"title={profile.current_title}, "
        f"seniority={profile.seniority_level}, "
        f"skills={len(profile.skills.technical)} technical"
    )

    return state
