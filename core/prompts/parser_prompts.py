"""
core/prompts/parser_prompts.py

All prompts used by the Resume Parser agent.
Keeping prompts isolated makes them independently testable and easy to iterate.
"""

RESUME_PARSE_PROMPT = """
You are an expert resume parser. Extract structured information from the resume text below.

Return ONLY valid JSON that exactly matches the schema provided.

Rules:
- If a field cannot be determined, set it to null. Never guess or hallucinate.

- For start_date and end_date:
  - Return in YYYY-MM format if you can determine both year and month with confidence.
  - Return YYYY-01 if the resume gives only a year.
  - Return null if the date is missing or ambiguous. Do not fabricate.
  - If the role is current ("Present", "now", "current", "ongoing"), set end_date to null.
  - Always set duration_months to null  calculated in code.

- For role_type, classify each work experience role into exactly one of:
  "full_time"    permanent, salaried employment regardless of seniority or title.
                  When genuinely uncertain between full_time and other, default to full_time.
  "internship"   intern, trainee, industrial placement, summer analyst, graduate scheme,
                  industrial attachment, co-op student, apprentice.
  "other"        freelance, contract, consulting (self-employed), part-time, volunteer,
                  research assistant, teaching assistant, casual or temporary work.
  Use both title AND company for context. "Consultant" at McKinsey  full_time.
  "Consultant" at own practice or without a company  other.
  "Summer Analyst" at Goldman Sachs  internship.

- For seniority_level, infer purely from scope, ownership, and impact described:
  "intern"      explicitly an intern, trainee, or apprentice
  "junior"      entry-level responsibilities, limited autonomy, no production ownership
  "mid"         independently owns deliverables, some cross-team influence
  "senior"      drives technical decisions, large-scale impact, mentors others
  "lead"        explicitly leads a team or function
  "principal"   cross-team technical authority, staff-level
  "executive"   VP, Director, C-suite
  Do NOT apply year-based thresholds. Assess from what the person actually did.
  IMPORTANT - production-scale impact signals OVERRIDE short tenure:
    If the candidate's full-time work shows ANY of the following, classify as "mid" or above,
    regardless of years of experience:
      - Production systems covering millions of users or transactions
      - Enterprise-wide model or system adoption (regulatory approval, org-wide rollout)
      - Quantified business impact (cost savings, efficiency gains, revenue influence)
      - Ownership of end-to-end ML/data pipelines in a regulated or high-stakes domain
      - Cross-functional stakeholder delivery (not just individual contributor tasks)
    A candidate with 1-2 years who independently owns a production system at scale
    in a regulated domain (finance, healthcare, legal) is "mid", not "junior".
  If uncertain, return null.

- For current_title, return the most recent job title exactly as written. Do not paraphrase.

- For impact_signals, extract quantifiable achievements only (numbers, %, scale, revenue).

- For ats_summary, write a 150-200 word ATS-optimised summary of the candidate's skills
  and professional experience. Write in third person. Be specific about technologies,
  domains, and seniority. This will be used for semantic matching against job descriptions
  so include all relevant technical and domain keywords naturally. Do NOT invent anything
  not present in the resume.

- For career_trajectory:
  "ascending"  progressively more senior roles
  "lateral"    similar level across roles
  "pivot"      clear domain or function change
  "re-entry"   gap followed by return to workforce

- For career_gaps, identify periods > 3 months where the candidate was not employed.
  STRICT rules for gap calculation:
    1. Only measure gaps between FULL-TIME roles, or between graduation date and
       first full-time role. Never measure gaps between internships.
    2. Pre-graduation internships are part of the academic phase. Do NOT treat the
       period between a pre-graduation internship end and graduation date as a gap.
    3. Graduation to first full-time job is a valid gap ONLY if it exceeds 3 months.
    4. Post-graduation internships (after graduation date) count as regular roles.
    5. If graduation to first full-time start is 3 months or under, report NO gap.
  Example: internship ends Jul 2023, graduation May 2024, full-time starts Sept 2024.
    - Jul 2023 to May 2024: NOT a gap (pre-graduation academic phase).
    - May 2024 to Sept 2024: 4-month gap. Report this one only.

Schema:
{
  "current_title": string | null,
  "seniority_level": "intern" | "junior" | "mid" | "senior" | "lead" | "principal" | "executive" | null,
  "ats_summary": string | null,
  "skills": {
    "technical": [string],
    "tools": [string],
    "soft": [string]
  },
  "education": [
    {
      "degree": string,
      "field": string | null,
      "institution": string,
      "year": number | null
    }
  ],
  "work_experience": [
    {
      "title": string,
      "company": string,
      "role_type": "full_time" | "internship" | "other",
      "start_date": "YYYY-MM" | null,
      "end_date": "YYYY-MM" | null,
      "duration_months": null,
      "impact_signals": [string]
    }
  ],
  "career_trajectory": "ascending" | "lateral" | "pivot" | "re-entry" | null,
  "pivot_signals": [string],
  "domain_expertise": [string],
  "career_gaps": [
    {
      "approx_duration_months": number,
      "position_in_timeline": "early" | "mid" | "recent"
    }
  ]
}

Resume text:
\"\"\"
{resume_text}
\"\"\"

Return ONLY the JSON object. No explanation, no markdown, no code fences.
""".strip()


RESUME_PARSE_FALLBACK_PROMPT = """
The resume text provided appears to be incomplete or poorly formatted.
Extract whatever structured information is available. Set missing or uncertain fields to null.
For dates, use YYYY-MM if confident, null if not. Never guess. Set duration_months to null always.
For role_type, default to "full_time" if you cannot determine it from context.
Return ONLY valid JSON matching the schema above.

Partial resume text:
\"\"\"
{resume_text}
\"\"\"
""".strip()