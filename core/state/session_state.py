"""
core/state/session_state.py

The single shared state object that flows through the entire LangGraph pipeline.
Every agent reads from and writes to this object via the Postgres checkpointer.
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_serializer
from datetime import datetime


# ─────────────────────────────────────────────
# Sub-schemas: Resume Parser output
# ─────────────────────────────────────────────

class SkillSet(BaseModel):
    technical: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    soft: list[str] = Field(default_factory=list)


class WorkExperience(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: str
    company: str
    start_date: Optional[str] = None       # YYYY-MM format, from LLM
    end_date: Optional[str] = None         # YYYY-MM format, null = present
    duration_months: Optional[int] = None  # calculated in Python post-parse
    responsibilities: list[str] = Field(default_factory=list)
    impact_signals: list[str] = Field(default_factory=list)


class Education(BaseModel):
    model_config = ConfigDict(extra="ignore")  # silently drop gpa, major, etc.
    degree: str
    field: Optional[str] = None          # Claude sometimes returns "major" instead
    institution: str
    year: Optional[int] = None


class NotableProject(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: Optional[str] = None
    description: str
    tech_used: list[str] = Field(default_factory=list)


class CareerGap(BaseModel):
    model_config = ConfigDict(extra="ignore")
    approx_duration_months: int
    position_in_timeline: Literal["early", "mid", "recent"]


class CandidateProfile(BaseModel):
    current_title: Optional[str] = None
    years_experience: Optional[float] = None
    seniority_level: Optional[
        Literal["intern", "junior", "mid", "senior", "lead", "principal", "executive"]
    ] = None
    skills: SkillSet = Field(default_factory=SkillSet)
    education: list[Education] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    career_trajectory: Optional[
        Literal["ascending", "lateral", "pivot", "re-entry"]
    ] = None
    pivot_signals: list[str] = Field(default_factory=list)
    domain_expertise: list[str] = Field(default_factory=list)
    notable_projects: list[NotableProject] = Field(default_factory=list)
    career_gaps: list[CareerGap] = Field(default_factory=list)
    raw_text: Optional[str] = None  # kept for downstream embedding use


# ─────────────────────────────────────────────
# Sub-schemas: Profile Recommender output
# ─────────────────────────────────────────────

class SuggestedProfile(BaseModel):
    title: str
    seniority_target: Literal["junior", "mid", "senior", "lead", "principal"]
    confidence: Literal["high", "medium", "low"]
    match_reason: str
    is_stretch: bool = False
    source: Literal["system", "user_custom"] = "system"


# ─────────────────────────────────────────────
# Sub-schemas: Job Search output
# ─────────────────────────────────────────────

class RawJob(BaseModel):
    job_id: str
    title: str
    company: str
    location: str
    work_type: Optional[str] = None
    jd_text: str
    apply_url: str
    source: str                         # adzuna / jsearch / remoteok / firecrawl
    posted_date: Optional[datetime] = None
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    matched_profile: str                # which confirmed profile triggered this


# ─────────────────────────────────────────────
# Sub-schemas: Ranker output
# ─────────────────────────────────────────────

class RankedJob(BaseModel):
    job_id: str
    title: str
    company: str
    location: str
    work_type: Optional[str] = None
    jd_text: str
    apply_url: str
    source: str
    posted_date: Optional[datetime] = None
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    matched_via: list[str] = Field(default_factory=list)  # profile badge list
    fit_score: float                    # 0.0 – 1.0 composite score
    semantic_score: float
    seniority_score: float
    recency_score: float
    gap_skills: list[str] = Field(default_factory=list)
    recommended_action: Literal["apply_now", "apply_with_note", "monitor", "skip"] = "monitor"


# ─────────────────────────────────────────────
# Sub-schemas: Hiring Signals output
# ─────────────────────────────────────────────

class HiringSignal(BaseModel):
    model_config = ConfigDict(extra="ignore")
    company:              str
    signal_type:          Literal[
        "funding", "expansion", "product_launch",
        "headcount_growth", "hiring_freeze", "layoff", "neutral"
    ]
    signal_strength:      Literal["high", "medium", "low"]
    summary:              str
    is_positive:          bool = True
    confidence:           float = 0.5
    source_url:           str = ""
    source_date:          Optional[datetime] = None
    source_name:          str = ""
    jobs_you_matched:     int = 0        # ties signal back to user's ranked results
    relevant_to_profiles: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────
# User preferences (Step 1 input)
# ─────────────────────────────────────────────

class UserPreferences(BaseModel):
    location: str
    work_type: Literal["remote", "hybrid", "on-site", "any"] = "any"
    seniority_preference: Literal["same_level", "step_up", "open"] = "open"


# ─────────────────────────────────────────────
# Master session state — the LangGraph state object
# ─────────────────────────────────────────────

class SessionState(BaseModel):
    """
    Single source of truth for the entire pipeline.
    Persisted to Postgres via LangGraph checkpointer.
    Every agent reads from and writes to this object.
    """

    # Identity
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Step 1: Raw inputs
    resume_file_name: Optional[str] = None
    resume_raw_text: Optional[str] = None
    preferences: Optional[UserPreferences] = None

    # Step 2: Parser output
    candidate_profile: Optional[CandidateProfile] = None
    parse_failed: bool = False
    parse_failure_reason: Optional[str] = None

    # Step 3: Recommender output + user confirmation
    suggested_profiles: list[SuggestedProfile] = Field(default_factory=list)
    confirmed_profiles: list[SuggestedProfile] = Field(default_factory=list)
    awaiting_confirmation: bool = False

    # Step 4: Job search output
    raw_jobs: list[RawJob] = Field(default_factory=list)

    # Step 5: Ranked output
    ranked_jobs: list[RankedJob] = Field(default_factory=list)

    # Step 6: Hiring signals
    hiring_signals: list[HiringSignal] = Field(default_factory=list)
    watch_list:     list[HiringSignal] = Field(default_factory=list)

    # Step 7: Final assembled payload ready for frontend
    results_ready: bool = False

    # Pipeline control
    current_agent: Optional[str] = None
    error: Optional[str] = None
    pipeline_complete: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("created_at", "updated_at")
    def serialize_datetime(self, v: datetime) -> str:
        return v.isoformat()
