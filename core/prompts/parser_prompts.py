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
  - Return YYYY-01 only if the resume gives a year but no month — make clear it is approximate.
  - Return null if the date is ambiguous, missing, or unparseable. Do not fabricate a date.
  - If the role is current ("Present", "now", "current", "ongoing"), set end_date to null.
  - Always set duration_months to null — it will be calculated in code from the dates.
- For seniority_level:
  - Infer purely from the scope, ownership, scale, and impact described in the resume text.
  - "intern" only if the role is explicitly labelled as intern, trainee, or apprentice.
  - For all other levels, use your judgment based on what the person actually did,
    not how long they have been working. Do not apply year-based thresholds.
  - If seniority cannot be determined confidently, return null.
- For current_title, return the most recent job title exactly as written. Do not paraphrase.
- For impact_signals, extract quantifiable achievements only (numbers, percentages, scale).
- For career_trajectory, assess the overall pattern across all roles:
  "ascending" → progressively more senior roles or responsibilities
  "lateral"   → similar level across roles
  "pivot"     → clear domain or function change
  "re-entry"  → gap followed by return to workforce
- For career_gaps, identify gaps > 3 months between consecutive roles.

Schema:
{
  "current_title": string | null,
  "seniority_level": "intern" | "junior" | "mid" | "senior" | "lead" | "principal" | "executive" | null,
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
      "start_date": "YYYY-MM" | null,
      "end_date": "YYYY-MM" | null,
      "duration_months": null,
      "responsibilities": [string],
      "impact_signals": [string]
    }
  ],
  "career_trajectory": "ascending" | "lateral" | "pivot" | "re-entry" | null,
  "pivot_signals": [string],
  "domain_expertise": [string],
  "notable_projects": [
    {
      "name": string | null,
      "description": string,
      "tech_used": [string]
    }
  ],
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
Extract whatever structured information is available. Set any uncertain or missing fields to null.
For dates, use YYYY-MM if confident, null if not. Never guess. Set duration_months to null always.
Return ONLY valid JSON matching the schema above.

Partial resume text:
\"\"\"
{resume_text}
\"\"\"
""".strip()
