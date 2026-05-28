"""
api/models.py

Pydantic models for all API request and response payloads.
Kept separate from core/state so the API contract can evolve
independently of the internal pipeline state schema.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ── Shared ────────────────────────────────────────────────────────────────────

class APIError(BaseModel):
    detail: str
    code: str


# ── POST /sessions ────────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    location: Optional[str] = Field(None, description="Preferred job location, e.g. 'Bangalore'. Required — all searches use city context.")
    seniority_preference: Literal["same_level", "step_up", "open"] = "open"


class CreateSessionResponse(BaseModel):
    session_id: str
    message: str = "Session created. Upload your resume to begin."


# ── POST /sessions/{session_id}/resume ────────────────────────────────────────

class ResumeUploadResponse(BaseModel):
    session_id: str
    status: Literal["parsing", "parse_failed", "awaiting_confirmation"]
    message: str
    # Populated only when status == "awaiting_confirmation"
    suggested_profiles: list[SuggestedProfileResponse] = Field(default_factory=list)
    parse_failure_reason: Optional[str] = None


# ── Profile shapes ────────────────────────────────────────────────────────────

class SuggestedProfileResponse(BaseModel):
    title: str
    seniority_target: str
    confidence: Literal["high", "medium", "low"]
    match_reason: str
    is_stretch: bool
    source: Literal["system", "user_custom"] = "system"


# ── POST /sessions/{session_id}/confirm ──────────────────────────────────────

class ConfirmProfilesRequest(BaseModel):
    selected_titles: list[str] = Field(
        ...,
        description="Titles selected from the suggested profiles list",
        min_length=1,
    )
    custom_profiles: list[str] = Field(
        default_factory=list,
        description="Any additional custom job titles added by the user",
    )


class ConfirmProfilesResponse(BaseModel):
    session_id: str
    status: Literal["confirmed", "searching"]
    confirmed_profiles: list[SuggestedProfileResponse]
    message: str


# ── GET /sessions/{session_id}/status ────────────────────────────────────────

class PipelineStatus(BaseModel):
    session_id: str
    status: Literal[
        "created",
        "parsing",
        "parse_failed",
        "awaiting_confirmation",
        "searching",
        "ranking",
        "complete",
        "error",
    ]
    current_agent: Optional[str] = None
    message: str
    # Populated progressively as pipeline runs
    suggested_profiles: list[SuggestedProfileResponse] = Field(default_factory=list)
    confirmed_profiles: list[SuggestedProfileResponse] = Field(default_factory=list)
    jobs_found: int = 0
    results_ready: bool = False
    error: Optional[str] = None


# ── GET /sessions/{session_id}/results ───────────────────────────────────────

class JobResult(BaseModel):
    job_id: str
    title: str
    company: str
    location: str
    work_type: Optional[str] = None
    apply_url: str
    source: str
    posted_date: Optional[str] = None
    matched_via: list[str] = Field(default_factory=list)
    matched_profile: str = ""

    # HyDE section — drives results UI grouping
    # "S1" → "Roles in your domain"  |  "S2" → "Broader opportunities"
    hyde_section:    Optional[str]   = None
    jd1_emb_score:   Optional[float] = None
    jd2_emb_score:   Optional[float] = None

    # Composite score
    fit_score: float = 0.0

    # Sub-scores
    experience_score: float = 0.0
    skill_score: float = 0.0
    domain_score: float = 0.0
    recency_score: float = 0.0
    education_score: Optional[float] = None
    education_required: bool = False

    # Scoring notes and gap analysis
    scoring_notes: str = ""
    experience_gap: Optional[str] = None
    skill_gaps: list[str] = Field(default_factory=list)
    domain_gap: Optional[str] = None
    education_gap: Optional[str] = None


class ResultsResponse(BaseModel):
    session_id: str
    total_jobs: int
    # Sectioned output from HyDE prefilter
    section1_jobs: list[JobResult] = Field(default_factory=list)   # "Roles in your domain"
    section2_jobs: list[JobResult] = Field(default_factory=list)   # "Broader opportunities"
    # Backward-compatible flat list: S1 + S2 concatenated (for clients not yet section-aware)
    jobs: list[JobResult] = Field(default_factory=list)
    message: str
    # Pipeline observability: session_metrics.quality dict from MLflow logger.
    # Includes per-stage job counts (jobs_by_stage), fallback flag, etc.
    session_metrics: Optional[dict] = None


# ── Health check ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str = "0.1.0"
    agents_ready: bool = True