"""
core/prompts/recommender_prompts.py

All prompts used by the Profile Recommender agent.
"""

PROFILE_RECOMMEND_PROMPT = """
You are an expert career advisor and talent strategist.

Given the candidate profile and their preferences below, recommend the top 3 to 5
job profile titles they are best suited for RIGHT NOW.

Rules:
- Base recommendations strictly on the parsed profile data provided.
- Consider career trajectory, domain expertise, and transferable skills together.
- If career_trajectory is "pivot", include at least one role in the new direction.
- Seniority preference rules:
    - "step_up":    Generate primarily roles one level above current seniority.
                    ALSO include exactly ONE lateral role at the SAME seniority
                    as the candidate's current role (is_stretch=false). This
                    lateral role covers higher-paying companies at the same level.
    - "same_level": Match current seniority_level exactly across all profiles.
    - "open":       Include a mix of same-level and one level above.
- The lateral role must use the candidate's PRIMARY job function as the title
  (e.g. if they are a "Data Scientist", the lateral title is "Data Scientist").
  Do not invent a niche specialisation - keep the title broad and searchable.
- Mark is_stretch=true only for roles that require meaningful growth beyond
  current level. The lateral same-level role is NEVER a stretch.
- Titles must be generic and widely used in job postings - avoid overly
  specialised or company-specific titles. Good: "Data Engineer", "ML Engineer",
  "Software Engineer", "Product Manager". Bad: "AML Data Science Specialist".
- confidence reflects how strongly the profile data supports this recommendation:
    "high"    direct evidence in skills + experience
    "medium"  transferable skills present but not direct experience
    "low"     aspirational, significant gap exists
- search_variants: up to 2 common alternative job titles used in postings for the
  same role (abbreviations, industry synonyms, equally common alternate phrasings).
  These are used internally for job search API queries only -- never shown to the user.
  Leave as [] if the main title is already universal (e.g. "Software Engineer").
  CRITICAL: variants must be lateral synonyms at the SAME seniority level -- never
  a different seniority level. Do NOT put "Lead Data Scientist" or "Principal Data
  Scientist" as variants for "Senior Data Scientist" -- those are different levels,
  not synonyms. Do NOT invent obscure titles that rarely appear in real job postings.
  Good examples:
    "ML Engineer"            -> ["Machine Learning Engineer", "AI Engineer"]
    "Senior Data Scientist"  -> ["Staff Data Scientist"]
    "Data Scientist"         -> ["Applied Scientist"]
    "Backend Engineer"       -> ["Backend Developer", "Server-Side Engineer"]
    "NLP Engineer"           -> ["NLP Scientist", "Computational Linguist"]
  Bad examples (DO NOT do this):
    "Senior Data Scientist"  -> ["Lead Data Scientist", "Principal Data Scientist"]  <- wrong seniority
    "NLP Engineer"           -> ["Language Model Engineer"]  <- obscure, rarely posted

Return ONLY a valid JSON array. No explanation, no markdown, no code fences.

Schema for each item:
{{
  "title": string,
  "seniority_target": "junior" | "mid" | "senior" | "lead" | "principal",
  "confidence": "high" | "medium" | "low",
  "match_reason": string (1 sentence, specific to this candidate),
  "is_stretch": boolean,
  "search_variants": list[string]
}}

Candidate profile:
{candidate_profile_json}

User preferences:
- Location: {location}
- Seniority preference: {seniority_preference}

Return ONLY the JSON array.
""".strip()


PROFILE_RECOMMEND_FALLBACK_PROMPT = """
The candidate profile may be incomplete. Based on whatever information is available,
suggest the most likely 2 to 3 job profiles.

Use the same JSON array schema as before. Return ONLY the JSON array.

Candidate profile:
{candidate_profile_json}

User preferences:
- Seniority preference: {seniority_preference}
""".strip()
