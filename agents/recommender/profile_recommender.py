"""
agents/recommender/profile_recommender.py

Agent 2 — Profile Recommender

Responsibility:
  - Read candidate_profile + user preferences from SessionState
  - Call Claude API to generate 3-5 suggested job profiles
  - Write suggested_profiles to SessionState
  - Set awaiting_confirmation = True to trigger the human-in-the-loop gate

The LangGraph interrupt() gate lives in the graph layer (graph.py), not here.
This agent's job is purely: profile → suggestions. Clean separation.

Input  (from SessionState): candidate_profile, preferences
Output (to SessionState)  : suggested_profiles, awaiting_confirmation=True
                            OR error on failure
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import anthropic
from langsmith import traceable
from tenacity import retry, stop_after_attempt, wait_exponential

from core.prompts.recommender_prompts import (
    PROFILE_RECOMMEND_FALLBACK_PROMPT,
    PROFILE_RECOMMEND_PROMPT,
)
from core.state.session_state import SessionState, SuggestedProfile, UserPreferences

logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS   = 1024
MIN_PROFILES = 2
MAX_PROFILES = 5


# ── Formatting helpers ────────────────────────────────────────────────────────


def _profile_to_prompt_json(state: SessionState) -> str:
    """
    Serialize the candidate profile to a compact JSON string for the prompt.
    Excludes raw_text (too long, not needed for profile recommendation).
    """
    profile = state.candidate_profile
    data = {
        "current_title":     profile.current_title,
        "years_experience":  profile.years_experience,
        "seniority_level":   profile.seniority_level,
        "skills":            profile.skills.model_dump(),
        "domain_expertise":  profile.domain_expertise,
        "career_trajectory": profile.career_trajectory,
        "pivot_signals":     profile.pivot_signals,
        "work_experience": [
            {
                "title":          exp.title,
                "company":        exp.company,
                "duration_months": exp.duration_months,
                "impact_signals": exp.impact_signals,
            }
            for exp in profile.work_experience
        ],
        "notable_projects": [
            {
                "name":        proj.name,
                "description": proj.description,
                "tech_used":   proj.tech_used,
            }
            for proj in profile.notable_projects
        ],
        "education": [
            {
                "degree":      edu.degree,
                "field":       edu.field,
                "institution": edu.institution,
            }
            for edu in profile.education
        ],
        "career_gaps": [
            {
                "approx_duration_months":  gap.approx_duration_months,
                "position_in_timeline":    gap.position_in_timeline,
            }
            for gap in profile.career_gaps
        ],
    }
    return json.dumps(data, indent=2)


# ── LLM call ─────────────────────────────────────────────────────────────────

@traceable(
    name="claude-profile-recommender",
    run_type="llm",
    metadata={"model": CLAUDE_MODEL, "agent": "profile_recommender"},
)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _call_claude(prompt: str) -> str:
    """Call Claude and return the raw response string."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


# ── Response parsing ──────────────────────────────────────────────────────────

def _clean_response(raw: str) -> str:
    """
    Strip markdown fences, preamble, or leading whitespace from LLM response.
    Finds the first [ and last ] to extract the JSON array even if Claude
    adds text before or after the array.
    """
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$",          "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    # Extract JSON array boundaries
    start = raw.find("[")
    end   = raw.rfind("]") + 1

    if start != -1 and end > start:
        return raw[start:end]

    # Fallback: maybe it returned a single object instead of array
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end > start:
        return "[" + raw[start:end] + "]"

    return raw


def _parse_profiles(raw_json: str) -> list[SuggestedProfile]:
    """
    Parse the LLM JSON array response into validated SuggestedProfile objects.
    Raises ValueError on malformed JSON or schema violations.
    """
    cleaned = _clean_response(raw_json)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nRaw: {cleaned[:300]}") from e

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")

    profiles = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            logger.warning(f"Skipping non-dict item at index {i}: {item}")
            continue
        try:
            profile = SuggestedProfile(
                title            = item["title"],
                seniority_target = item["seniority_target"],
                confidence       = item["confidence"],
                match_reason     = item["match_reason"],
                is_stretch       = item.get("is_stretch", False),
                source           = "system",
            )
            profiles.append(profile)
        except (KeyError, ValueError) as e:
            logger.warning(f"Skipping malformed profile item {i}: {e} — {item}")
            continue

    if len(profiles) < MIN_PROFILES:
        raise ValueError(
            f"Too few valid profiles returned: {len(profiles)} "
            f"(minimum {MIN_PROFILES})"
        )

    # Cap at MAX_PROFILES
    return profiles[:MAX_PROFILES]


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_preconditions(state: SessionState) -> Optional[str]:
    """
    Check that the state is ready for profile recommendation.
    Returns an error message string if not ready, else None.
    """
    if state.parse_failed:
        return "Cannot recommend profiles: resume parsing failed."

    if not state.candidate_profile:
        return "Cannot recommend profiles: candidate_profile is missing from state."

    if not state.preferences:
        return "Cannot recommend profiles: user preferences are missing from state."

    profile = state.candidate_profile
    has_skills = bool(
        profile.skills.technical
        or profile.skills.tools
        or profile.domain_expertise
    )
    if not has_skills and not profile.current_title:
        return (
            "Candidate profile appears empty — no skills or title found. "
            "Resume parsing may have partially failed."
        )

    return None


# ── Main agent function ───────────────────────────────────────────────────────

@traceable(name="agent-2-profile-recommender", run_type="chain")
def run_profile_recommender(state: SessionState) -> SessionState:
    """
    Agent 2 — Profile Recommender.

    Reads candidate_profile and preferences from state.
    Calls Claude to generate 3–5 job profile suggestions.
    Writes suggested_profiles to state and sets awaiting_confirmation=True.

    The actual LangGraph interrupt() pause happens in graph.py, not here.
    This agent only produces the suggestions — the graph decides when to pause.

    Returns updated SessionState.
    """
    state.current_agent = "profile_recommender"
    logger.info(f"[profile_recommender] Starting — session_id={state.session_id}")

    # ── Validate preconditions ───────────────────────────────────────────────
    error = _validate_preconditions(state)
    if error:
        state.error = error
        logger.error(f"[profile_recommender] Precondition failed: {error}")
        return state

    prefs = state.preferences

    # ── Build prompt ─────────────────────────────────────────────────────────
    profile_json = _profile_to_prompt_json(state)

    # Use .replace() not .format() — profile_json contains {} from JSON
    # which breaks Python's str.format() with KeyError
    prompt = (
        PROFILE_RECOMMEND_PROMPT
        .replace("{candidate_profile_json}", profile_json)
        .replace("{location}",             prefs.location)
        .replace("{work_type}",            prefs.work_type)
        .replace("{seniority_preference}", prefs.seniority_preference)
    )

    # ── LLM call + parse ─────────────────────────────────────────────────────
    try:
        raw = _call_claude(prompt)
        profiles = _parse_profiles(raw)

    except ValueError as e:
        # Primary attempt failed — try fallback prompt
        logger.warning(
            f"[profile_recommender] Primary attempt failed ({e}), trying fallback"
        )
        try:
            fallback_prompt = (
                PROFILE_RECOMMEND_FALLBACK_PROMPT
                .replace("{candidate_profile_json}", profile_json)
                .replace("{seniority_preference}",   prefs.seniority_preference)
            )
            raw = _call_claude(fallback_prompt)
            profiles = _parse_profiles(raw)

        except Exception as fallback_err:
            state.error = (
                f"Profile recommendation failed after fallback: {fallback_err}"
            )
            logger.error(
                f"[profile_recommender] Fallback also failed: {fallback_err}"
            )
            return state

    except Exception as e:
        state.error = f"Unexpected error in profile recommender: {e}"
        logger.error(f"[profile_recommender] Unexpected error: {e}")
        return state

    # ── Write to state ───────────────────────────────────────────────────────
    state.suggested_profiles   = profiles
    state.awaiting_confirmation = True
    state.error                = None

    high_conf  = [p for p in profiles if p.confidence == "high"]
    stretch    = [p for p in profiles if p.is_stretch]

    logger.info(
        f"[profile_recommender] Success — "
        f"{len(profiles)} profiles suggested, "
        f"{len(high_conf)} high confidence, "
        f"{len(stretch)} stretch roles"
    )
    for p in profiles:
        logger.debug(
            f"  → [{p.confidence}] {p.title} "
            f"({p.seniority_target})"
            f"{' [stretch]' if p.is_stretch else ''}"
        )

    return state


# ── User confirmation handler ─────────────────────────────────────────────────

def apply_user_confirmation(
    state: SessionState,
    selected_titles: list[str],
    custom_profiles: Optional[list[str]] = None,
) -> SessionState:
    """
    Called after the human-in-the-loop gate resolves.

    Takes the user's selected profile titles and any custom additions,
    writes confirmed_profiles to state, and clears the awaiting flag.

    Args:
        state:            current SessionState
        selected_titles:  list of titles the user selected from suggestions
        custom_profiles:  list of custom title strings the user added manually

    Returns updated SessionState.
    """
    confirmed: list[SuggestedProfile] = []

    # Map selected titles back to their full SuggestedProfile objects
    suggested_map = {p.title: p for p in state.suggested_profiles}

    for title in selected_titles:
        if title in suggested_map:
            confirmed.append(suggested_map[title])
        else:
            logger.warning(
                f"[profile_recommender] Selected title not in suggestions: '{title}'"
            )

    # Add any user-defined custom profiles
    for title in (custom_profiles or []):
        title = title.strip()
        if not title:
            continue
        # Inherit seniority from the candidate profile where possible
        seniority = (
            state.candidate_profile.seniority_level
            if state.candidate_profile and state.candidate_profile.seniority_level
            else "mid"
        )
        # Ensure seniority is one of the valid SuggestedProfile values
        valid_seniority = {"junior", "mid", "senior", "lead", "principal"}
        if seniority not in valid_seniority:
            seniority = "mid"

        confirmed.append(
            SuggestedProfile(
                title            = title,
                seniority_target = seniority,
                confidence       = "medium",
                match_reason     = "Added manually by user.",
                is_stretch       = False,
                source           = "user_custom",
            )
        )

    state.confirmed_profiles    = confirmed
    state.awaiting_confirmation = False

    logger.info(
        f"[profile_recommender] Confirmation applied — "
        f"{len(confirmed)} profiles confirmed "
        f"({len(custom_profiles or [])} user-custom)"
    )

    return state
