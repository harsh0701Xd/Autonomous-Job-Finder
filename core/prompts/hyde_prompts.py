"""
core/prompts/hyde_prompts.py

Prompt builders for the HyDE Prefilter agent (Agent 5b).

Two hypothetical JD types are generated per session:

  JD1 — Domain-anchored
    Full profile context. Tightly anchored to the candidate's current domain
    (e.g. fintech / AML) via domain_description and all impact signals.
    Primary filter: surfaces roles in the candidate's own domain.

  JD2 — Transferable-skills
    Skills-only context. No domain_expertise, no company names, no domain-
    specific impact signals, no ats_summary. Instructs Claude to produce a
    completely industry-neutral JD, distributing responsibilities evenly across
    all skill clusters (ML, NLP/LLM, time-series, data engineering, etc.).
    Broader filter: surfaces roles the candidate qualifies for across any domain.

CRITICAL CONSTRAINT (preserved from architecture decision):
  Resume-agnostic design — no domain-specific embedding models in production.
  voyage-3 only. Domain specificity comes from HyDE content, not model choice.
"""

from __future__ import annotations

import textwrap


# =============================================================================
# JD1 — Domain-anchored prompt builder
# =============================================================================

def build_domain_anchored_jd_prompt(
    confirmed_profile: str,
    step_up:           bool,
    domain_description: str,
    free_text:         str | None,
    profile:           dict,          # raw CandidateProfile dict from session state
    years_ft:          float,
    years_other:       float,
) -> str:
    """
    Build the JD1 (domain-anchored) prompt for Claude Sonnet.

    Full profile context — no field truncation. Domain is explicitly anchored
    via domain_description so that the resulting embedding tightly matches
    roles in the candidate's current industry.

    Args:
        confirmed_profile: e.g. "Data Scientist" (user-confirmed title)
        step_up:           True → target title prefixed with "Senior"
        domain_description: from session preferences (e.g. "fintech, AML, fraud detection")
        free_text:         optional extra context from user (passed through)
        profile:           CandidateProfile.model_dump() dict
        years_ft:          computed full-time years of experience
        years_other:       computed internship/other years of experience
    """
    target_title    = f"Senior {confirmed_profile}" if step_up else confirmed_profile
    seniority_note  = "step-up target" if step_up else "current level"

    domain_line = (
        f"The role must be in this domain: {domain_description}."
        if domain_description
        else "The domain should match the candidate's primary area of expertise."
    )
    extra_line = (
        f'The candidate specifically mentioned: "{free_text}". '
        "Incorporate this into the JD where appropriate."
        if free_text
        else ""
    )

    tech_skills      = profile.get("skills", {}).get("technical", [])
    tools            = profile.get("skills", {}).get("tools", [])
    domain_expertise = profile.get("domain_expertise", [])
    ats_summary      = profile.get("ats_summary", "")
    work_exp         = profile.get("work_experience", [])
    seniority        = profile.get("seniority_level", "mid")

    exp_lines = []
    for exp in work_exp:
        signals = "; ".join(exp.get("impact_signals", []))
        exp_lines.append(
            f"  • {exp.get('title')} @ {exp.get('company')} "
            f"({exp.get('role_type', '')}) — {signals}"
        )
    exp_block  = "\n".join(exp_lines) if exp_lines else "  (none)"
    total_yrs  = round(years_ft + years_other, 1)
    exp_bar    = f"{total_yrs}–{round(total_yrs + 2.0, 1)} years of relevant experience"
    domain_str = ", ".join(domain_expertise) if domain_expertise else "the candidate's background"

    return textwrap.dedent(f"""
    You are an experienced technical recruiter writing a realistic job description.

    Generate a job description for the role of **{target_title}** that would be an
    excellent match for the following candidate.

    {domain_line}
    {extra_line}

    CANDIDATE STRUCTURED PROFILE:
    - Current title     : {profile.get('current_title')}
    - Seniority         : {seniority} ({seniority_note})
    - Full-time exp     : {years_ft} yrs  |  Other exp: {years_other} yrs
    - ATS summary       : {ats_summary}
    - Technical skills  : {', '.join(tech_skills)}
    - Tools             : {', '.join(tools)}
    - Domain expertise  : {domain_str}
    - Work experience   :
    {exp_block}

    INSTRUCTIONS:
    - Write like a real JD posted on LinkedIn / job boards — NOT a summary of the candidate
    - Include: role overview, 5-8 responsibilities, 5-8 required qualifications / skills
    - Experience requirement: target {exp_bar} in the qualifications section
    - Reflect ALL of the candidate's domain expertise areas: {domain_str}
      Do NOT over-index on any single domain. Weight each area in proportion to the
      evidence in their work experience — breadth matters as much as depth here
    - Ground the JD in the candidate's actual skills and projects
    - Length: 350–500 words
    - Output ONLY the job description text. No preamble, no explanation, no markdown headers.
    """).strip()


# =============================================================================
# JD2 — Transferable-skills prompt builder
# =============================================================================

def build_transferable_jd_prompt(
    confirmed_profile: str,
    step_up:           bool,
    profile:           dict,          # raw CandidateProfile dict from session state
    years_ft:          float,
    years_other:       float,
) -> str:
    """
    Build the JD2 (transferable-skills) prompt for Claude Sonnet.

    Deliberately omits ALL domain signals:
      - domain_expertise  (contains fintech/AML/healthcare etc.)
      - company names     (AmEx → financial domain signal)
      - impact_signals    (mention AML, SAR, FIU, HIPAA — domain leakage)
      - ats_summary       (domain-heavy by design)

    Passes only:
      - Technical skills and tools  (domain-neutral method/tool names)
      - Seniority and experience years
      - Education (degree/field — domain-neutral)

    CRITICAL: Claude is explicitly instructed NOT to use any industry-specific
    terminology and to describe a generic technology organisation. This ensures
    the resulting embedding captures transferable skill similarity without
    anchoring to the candidate's specific industry.

    Args:
        confirmed_profile: e.g. "Data Scientist"
        step_up:           True → target title prefixed with "Senior"
        profile:           CandidateProfile.model_dump() dict
        years_ft:          computed full-time years of experience
        years_other:       computed internship/other years of experience
    """
    target_title = f"Senior {confirmed_profile}" if step_up else confirmed_profile

    tech_skills = profile.get("skills", {}).get("technical", [])
    tools       = profile.get("skills", {}).get("tools", [])
    seniority   = profile.get("seniority_level", "mid")
    education   = profile.get("education", [])

    edu_lines = []
    for edu in education:
        line = edu.get("degree", "")
        if edu.get("field"):       line += f" in {edu['field']}"
        if edu.get("institution"): line += f", {edu['institution']}"
        edu_lines.append(line)
    edu_block = "; ".join(edu_lines) if edu_lines else "Not specified"

    total_yrs = round(years_ft + years_other, 1)
    exp_bar   = f"{total_yrs}–{round(total_yrs + 2.0, 1)} years of relevant experience"

    # High-level skill cluster hints derived from tech_skills for even breadth distribution
    cluster_hints = (
        "machine learning & anomaly detection, NLP & large language models, "
        "time-series forecasting, data engineering & pipeline orchestration, "
        "model interpretability & validation"
    )

    return textwrap.dedent(f"""
    You are an experienced technical recruiter writing a realistic job description.

    Generate a job description for the role of **{target_title}** that would be an
    excellent match for a candidate with the following technical capabilities.

    CRITICAL CONSTRAINTS — you must follow all of these exactly:
    - Do NOT mention any specific industry, business domain, or company type
      (no finance, fintech, banking, healthcare, manufacturing, retail, etc.)
    - Do NOT use domain-specific terminology (no AML, fraud, compliance, clinical,
      regulatory, SAR, HIPAA, or any other industry jargon)
    - The role title must be exactly "{target_title}" — no domain subtitle
    - The company "About" blurb must describe a generic technology organisation
      with no industry affiliation stated or implied
    - Focus purely on transferable technical capabilities

    CANDIDATE TECHNICAL PROFILE (domain-neutral):
    - Role target       : {target_title}
    - Seniority         : {seniority}
    - Full-time exp     : {years_ft} yrs  |  Other exp: {years_other} yrs
    - Technical skills  : {', '.join(tech_skills)}
    - Tools & platforms : {', '.join(tools)}
    - Education         : {edu_block}

    INSTRUCTIONS:
    - Write like a real JD posted on LinkedIn / job boards
    - Include: brief industry-neutral company intro, role overview,
      5-8 responsibilities, 5-8 required qualifications / skills
    - Experience requirement: target {exp_bar} in the qualifications section
    - Distribute responsibilities PROPORTIONALLY across ALL skill clusters:
      {cluster_hints}
      Do NOT let any single skill cluster dominate — equal weighting across all
    - Required qualifications must map directly to the technical skills listed above
    - Length: 300–450 words
    - Output ONLY the job description text. No preamble, no explanation, no markdown headers.
    """).strip()
