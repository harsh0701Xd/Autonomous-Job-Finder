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
- If career_trajectory is "pivot", include at least one role in the new domain.
- If seniority_preference is "step_up", prioritise roles one level above current.
- If seniority_preference is "same_level", match current seniority_level exactly.
- If seniority_preference is "open", include a mix of same-level and stretch roles.
- Mark is_stretch=true for any role that requires meaningful growth beyond current level.
- confidence reflects how strongly the profile data supports this recommendation:
    "high"   → direct evidence in skills + experience
    "medium" → transferable skills present but not direct experience
    "low"    → aspirational, significant gap exists

Return ONLY a valid JSON array. No explanation, no markdown, no code fences.

Schema for each item:
{{
  "title": string,
  "seniority_target": "junior" | "mid" | "senior" | "lead" | "principal",
  "confidence": "high" | "medium" | "low",
  "match_reason": string (1 sentence, specific to this candidate),
  "is_stretch": boolean
}}

Candidate profile:
{candidate_profile_json}

User preferences:
- Location: {location}
- Work type: {work_type}
- Seniority preference: {seniority_preference}
- Salary range: {salary_range}

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
