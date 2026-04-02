"""
core/prompts/parser_prompts.py

All prompts used by the Resume Parser agent.
Keeping prompts isolated makes them independently testable and easy to iterate.
"""

RESUME_PARSE_PROMPT = """
You are an expert resume parser. Extract structured information from the resume text below.

Return ONLY valid JSON that exactly matches the schema provided.
- If a field cannot be determined from the resume, set it to null.
- Do not hallucinate values. Only extract what is explicitly stated or clearly implied.
- For duration_months, estimate based on date ranges if exact months are not given.
- For impact_signals, extract quantifiable achievements (e.g. "reduced latency by 40%", "led team of 8").
- For career_trajectory, assess the overall pattern:
    "ascending"  → progressively more senior roles
    "lateral"    → similar seniority across roles
    "pivot"      → clear domain or function change (e.g. finance → data science)
    "re-entry"   → gap followed by return to workforce
- For pivot_signals, list specific evidence of a domain shift if present.
- For career_gaps, identify gaps > 3 months between roles.

Schema:
{
  "current_title": string | null,
  "years_experience": number | null,
  "seniority_level": "intern" | "junior" | "mid" | "senior" | "lead" | "principal" | "executive" | null,
  "skills": {
    "technical": [string],
    "tools": [string],
    "soft": [string]
  },
  "education": [
    {
      "degree": string,
      "field": string,
      "institution": string,
      "year": number | null
    }
  ],
  "work_experience": [
    {
      "title": string,
      "company": string,
      "duration_months": number | null,
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
Extract whatever structured information is available, filling all other fields with null.
Return ONLY valid JSON matching the schema above.

Partial resume text:
\"\"\"
{resume_text}
\"\"\"
""".strip()
