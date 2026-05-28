"""
test_scripts/test_resume_parser.py

Live test for Agent 2 -- Resume Parser.
Extracts text from a PDF/DOCX resume and calls Claude Sonnet to produce
a structured CandidateProfile. Prints every parsed field.

Usage:
    python test_scripts/test_resume_parser.py <path/to/resume.pdf>

Example:
    python test_scripts/test_resume_parser.py resume.pdf
"""

import sys
import uuid
import json
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

import fitz  # PyMuPDF

from core.state.session_state import SessionState
from agents.parser.resume_parser import run_resume_parser


def extract_text_from_pdf(path: str) -> str:
    """Extract plain text from a PDF using PyMuPDF."""
    doc  = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text.strip()


async def main(resume_path: str):
    print("\n" + "=" * 60)
    print("  Agent 2 -- Resume Parser Live Test")
    print("=" * 60)

    # -- Extract text ----------------------------------------------------------
    print(f"\nFile : {resume_path}")
    try:
        if resume_path.lower().endswith(".pdf"):
            raw_text = extract_text_from_pdf(resume_path)
        else:
            print("ERROR: Only PDF files are supported in this test script.")
            return
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return

    print(f"Chars extracted : {len(raw_text)}")
    if len(raw_text) < 200:
        print("WARNING: Very short text — may cause a parse failure.")

    # -- Build minimal session state ------------------------------------------
    session = SessionState(
        session_id      = str(uuid.uuid4()),
        resume_raw_text = raw_text,
        resume_file_name= resume_path.split("/")[-1].split("\\")[-1],
    )

    # -- Run parser ------------------------------------------------------------
    print("\nCalling Claude Sonnet (resume parser)...\n")
    result = await run_resume_parser(session)

    # -- Results ---------------------------------------------------------------
    if result.parse_failed:
        print(f"PARSE FAILED: {result.parse_failure_reason}")
        return

    p = result.candidate_profile
    print("=" * 60)
    print("  Parsed Candidate Profile")
    print("=" * 60)

    print(f"\nCurrent title        : {p.current_title}")
    print(f"Seniority            : {p.seniority_level}")
    print(f"Full-time experience : {p.years_experience_full_time} years")
    print(f"Other experience     : {p.years_experience_other} years")
    print(f"Career trajectory    : {p.career_trajectory}")

    print(f"\nTechnical skills ({len(p.skills.technical)}):")
    for s in p.skills.technical:
        print(f"  - {s}")

    print(f"\nTools ({len(p.skills.tools)}):")
    for t in p.skills.tools:
        print(f"  - {t}")

    print(f"\nDomain expertise:")
    for d in p.domain_expertise:
        print(f"  - {d}")

    print(f"\nEducation:")
    for edu in p.education:
        line = edu.degree
        if edu.field:       line += f" in {edu.field}"
        if edu.institution: line += f", {edu.institution}"
        if edu.year:        line += f" ({edu.year})"
        print(f"  - {line}")

    print(f"\nWork experience ({len(p.work_experience)} roles):")
    for w in p.work_experience:
        dur = f"{w.duration_months}mo" if w.duration_months else "?"
        print(f"  [{w.role_type}] {w.title} @ {w.company} ({dur})")
        for sig in w.impact_signals[:2]:
            print(f"    • {sig}")

    if p.career_gaps:
        print(f"\nCareer gaps: {len(p.career_gaps)}")

    if p.pivot_signals:
        print(f"\nPivot signals:")
        for sig in p.pivot_signals:
            print(f"  - {sig}")

    print(f"\nATS summary preview:")
    print(f"  {(p.ats_summary or '')[:300]}...")

    tokens_in  = result.agent_metrics.get("parser", {}).get("input_tokens",  0)
    tokens_out = result.agent_metrics.get("parser", {}).get("output_tokens", 0)
    latency    = result.agent_metrics.get("parser", {}).get("latency_secs",  0)
    print(f"\nTokens: {tokens_in} in / {tokens_out} out | Latency: {latency}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import asyncio
    if len(sys.argv) < 2:
        print("Usage: python test_scripts/test_resume_parser.py <path/to/resume.pdf>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
