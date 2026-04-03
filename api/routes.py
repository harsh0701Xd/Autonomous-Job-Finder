"""
api/routes.py

All FastAPI route handlers for the Autonomous Job Finder pipeline.

Endpoints:
  POST   /sessions                          → create session + set preferences
  POST   /sessions/{session_id}/resume      → upload resume, trigger pipeline
  POST   /sessions/{session_id}/confirm     → submit profile confirmation
  GET    /sessions/{session_id}/status      → poll pipeline status
  GET    /sessions/{session_id}/results     → fetch final job results
  GET    /health                            → health check
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, status

from agents.parser.resume_parser import extract_text

from api.dependencies import (
    create_session_record,
    generate_session_id,
    get_graph,
    get_session_record,
    update_session_status,
    validate_resume_file,
)
from api.models import (
    ConfirmProfilesRequest,
    ConfirmProfilesResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    HealthResponse,
    JobResult,
    PipelineStatus,
    ResumeUploadResponse,
    ResultsResponse,
    SuggestedProfileResponse,
)
from core.state.session_state import SessionState, UserPreferences

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Helper: state → response model ───────────────────────────────────────────

def _profile_to_response(p) -> SuggestedProfileResponse:
    return SuggestedProfileResponse(
        title            = p.title,
        seniority_target = p.seniority_target,
        confidence       = p.confidence,
        match_reason     = p.match_reason,
        is_stretch       = p.is_stretch,
        source           = p.source,
    )


def _derive_pipeline_status(state: SessionState, session_record: dict) -> str:
    """Derive a clean status string from pipeline state."""
    if state.pipeline_complete:
        return "complete"
    if state.error:
        return "error"
    if state.parse_failed:
        return "parse_failed"
    if state.awaiting_confirmation:
        return "awaiting_confirmation"
    if state.confirmed_profiles and not state.ranked_jobs:
        return "searching"
    if state.ranked_jobs and not state.results_ready:
        return "ranking"
    if state.results_ready:
        return "complete"
    if state.resume_raw_text and not state.candidate_profile:
        return "parsing"
    return session_record.get("status", "created")


# ── POST /sessions ────────────────────────────────────────────────────────────

@router.post(
    "/sessions",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new job search session",
)
async def create_session(body: CreateSessionRequest) -> CreateSessionResponse:
    """
    Step 1: Create a session and record user preferences.
    Returns a session_id used in all subsequent requests.
    """
    session_id = generate_session_id()

    create_session_record(
        session_id  = session_id,
        preferences = body.model_dump(),
    )

    logger.info(
        f"[routes] Session created: {session_id} — "
        f"location={body.location}, work_type={body.work_type}"
    )

    return CreateSessionResponse(
        session_id = session_id,
        message    = "Session created. Upload your resume to begin.",
    )


# ── POST /sessions/{session_id}/resume ───────────────────────────────────────

@router.post(
    "/sessions/{session_id}/resume",
    response_model=ResumeUploadResponse,
    summary="Upload resume and trigger the parsing + profile recommendation pipeline",
)
async def upload_resume(
    session_id:       str,
    background_tasks: BackgroundTasks,
    file:             UploadFile = File(..., description="Resume file (PDF or DOCX, max 5MB)"),
) -> ResumeUploadResponse:
    """
    Step 2: Upload the resume file.

    Triggers Agent 1 (Resume Parser) → Agent 2 (Profile Recommender) synchronously.
    The pipeline pauses at the confirmation gate and returns suggested profiles
    for the user to review.

    The graph interrupt() persists full state — the pipeline can resume
    even if the server restarts before the user confirms.
    """
    # Validate session exists
    session_record = get_session_record(session_id)

    # Validate file
    file_bytes, file_ext = await validate_resume_file(file)

    # Build initial pipeline state from session preferences
    prefs_data = session_record["preferences"]
    preferences = UserPreferences(
        location             = prefs_data["location"],
        work_type            = prefs_data["work_type"],
        seniority_preference = prefs_data["seniority_preference"],
        salary_min           = prefs_data.get("salary_min"),
        salary_max           = prefs_data.get("salary_max"),
        currency             = prefs_data.get("currency", "USD"),
    )

    initial_state = SessionState(
        session_id        = session_id,
        resume_file_name  = file.filename,
        preferences       = preferences,
    ).model_dump()

    # Extract text before entering the graph
    try:
        raw_text = extract_text(file_bytes, file_ext)
        initial_state["resume_raw_text"] = raw_text
    except Exception as e:
        update_session_status(session_id, "parse_failed")
        return ResumeUploadResponse(
            session_id           = session_id,
            status               = "parse_failed",
            message              = "Could not extract text from the uploaded file.",
            parse_failure_reason = str(e),
        )

    update_session_status(session_id, "parsing")
    logger.info(f"[routes] Starting pipeline for session {session_id}")

    # Run graph until it hits the confirmation interrupt
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    try:
        final_state_dict = graph.invoke(initial_state, config=config)
    except Exception as e:
        logger.error(f"[routes] Pipeline error for session {session_id}: {e}")
        update_session_status(session_id, "error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {e}",
        )

    # Reconstruct state object from dict
    final_state = SessionState(**final_state_dict)

    # Handle parse failure
    if final_state.parse_failed:
        update_session_status(session_id, "parse_failed")
        return ResumeUploadResponse(
            session_id           = session_id,
            status               = "parse_failed",
            message              = "Resume parsing failed. Please try again.",
            parse_failure_reason = final_state.parse_failure_reason,
        )

    # Handle recommendation error
    if final_state.error:
        update_session_status(session_id, "error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=final_state.error,
        )

    # Success — pipeline paused at confirmation gate
    update_session_status(session_id, "awaiting_confirmation")

    suggested = [_profile_to_response(p) for p in final_state.suggested_profiles]

    logger.info(
        f"[routes] Pipeline paused at confirmation gate for {session_id} — "
        f"{len(suggested)} profiles suggested"
    )

    return ResumeUploadResponse(
        session_id        = session_id,
        status            = "awaiting_confirmation",
        message           = (
            f"Resume parsed successfully. "
            f"We found {len(suggested)} job profiles that match your background. "
            f"Select the ones you want to search for."
        ),
        suggested_profiles = suggested,
    )


# ── POST /sessions/{session_id}/confirm ──────────────────────────────────────

@router.post(
    "/sessions/{session_id}/confirm",
    response_model=ConfirmProfilesResponse,
    summary="Submit user's profile confirmation and resume the pipeline",
)
async def confirm_profiles(
    session_id: str,
    body:       ConfirmProfilesRequest,
) -> ConfirmProfilesResponse:
    """
    Step 3: User confirms which profiles to search for.

    Resumes the LangGraph pipeline from the interrupt point.
    The graph applies the confirmation and prepares for job search
    (Agent 3 and beyond — Phase 2).

    Returns the confirmed profiles so the frontend can show
    what's being searched.
    """
    session_record = get_session_record(session_id)

    # Validate session is in the right state
    current_status = session_record.get("status")
    if current_status != "awaiting_confirmation":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Session is not awaiting confirmation. "
                f"Current status: '{current_status}'. "
                f"Upload a resume first."
            ),
        )

    if not body.selected_titles and not body.custom_profiles:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one profile title must be selected or provided.",
        )

    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    # Resume the graph from the interrupt point using LangGraph Command
    # The graph paused at user_confirmation node — we resume by passing
    # the user's selection as the interrupt resume value via Command
    from langgraph.types import Command

    resume_payload = {
        "selected_titles": body.selected_titles,
        "custom_profiles": body.custom_profiles,
    }

    try:
        final_state_dict = graph.invoke(
            Command(resume=resume_payload),
            config=config,
        )
    except Exception as e:
        logger.error(f"[routes] Resume error for session {session_id}: {e}")
        update_session_status(session_id, "error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline resume error: {e}",
        )

    # Fetch full state from checkpointer and reconstruct SessionState
    try:
        checkpoint = graph.get_state(config)
        state_dict = checkpoint.values if checkpoint else {}
        final_state = SessionState(**state_dict)
    except Exception as e:
        logger.error(f"[routes] State reconstruction error: {e}")
        # Fallback — build minimal confirmed response from body directly
        from agents.recommender.profile_recommender import apply_user_confirmation
        # Get state from what we know
        checkpoint = graph.get_state(config)
        state_dict = checkpoint.values if checkpoint else {}
        final_state = SessionState(**state_dict) if state_dict else SessionState(
            session_id=session_id,
            confirmed_profiles=[],
        )

    if final_state.error:
        update_session_status(session_id, "error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=final_state.error,
        )

    update_session_status(session_id, "searching")
    confirmed = [_profile_to_response(p) for p in final_state.confirmed_profiles]

    logger.info(
        f"[routes] Confirmation applied for {session_id} — "
        f"{len(confirmed)} profiles confirmed"
    )

    return ConfirmProfilesResponse(
        session_id         = session_id,
        status             = "confirmed",
        confirmed_profiles = confirmed,
        message            = (
            f"Confirmed {len(confirmed)} profile(s). "
            f"Job search will begin shortly."
        ),
    )


# ── GET /sessions/{session_id}/status ────────────────────────────────────────

@router.get(
    "/sessions/{session_id}/status",
    response_model=PipelineStatus,
    summary="Poll the current status of the pipeline for a session",
)
async def get_status(session_id: str) -> PipelineStatus:
    """
    Poll endpoint for the frontend to track pipeline progress.

    The frontend calls this every 2–3 seconds after resume upload
    to show a live progress indicator.

    Returns current agent, status, and any available partial results.
    """
    session_record = get_session_record(session_id)
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    # Fetch current persisted state from checkpointer
    try:
        checkpoint = graph.get_state(config)
        state_dict = checkpoint.values if checkpoint else {}
    except Exception as e:
        logger.warning(f"[routes] Could not fetch graph state for {session_id}: {e}")
        state_dict = {}

    if not state_dict:
        # Session exists but pipeline hasn't started yet
        return PipelineStatus(
            session_id    = session_id,
            status        = session_record.get("status", "created"),
            current_agent = None,
            message       = "Session ready. Upload your resume to begin.",
        )

    state = SessionState(**state_dict)
    pipeline_status = _derive_pipeline_status(state, session_record)

    status_messages = {
        "created":               "Session ready. Upload your resume to begin.",
        "parsing":               "Parsing your resume...",
        "parse_failed":          f"Resume parsing failed: {state.parse_failure_reason}",
        "awaiting_confirmation": "Profiles ready. Waiting for your selection.",
        "searching":             "Searching job boards across your selected profiles...",
        "ranking":               "Ranking and deduplicating results...",
        "complete":              f"Done! Found {len(state.ranked_jobs)} matching jobs.",
        "error":                 f"An error occurred: {state.error}",
    }

    return PipelineStatus(
        session_id         = session_id,
        status             = pipeline_status,
        current_agent      = state.current_agent,
        message            = status_messages.get(pipeline_status, "Processing..."),
        suggested_profiles = [_profile_to_response(p) for p in state.suggested_profiles],
        confirmed_profiles = [_profile_to_response(p) for p in state.confirmed_profiles],
        jobs_found         = len(state.ranked_jobs),
        results_ready      = state.results_ready,
        error              = state.error,
    )


# ── GET /sessions/{session_id}/results ───────────────────────────────────────

@router.get(
    "/sessions/{session_id}/results",
    response_model=ResultsResponse,
    summary="Fetch final ranked job results for a completed session",
)
async def get_results(session_id: str) -> ResultsResponse:
    """
    Returns the final ranked job list and hiring signal watch list.

    Only available once the pipeline has completed (results_ready=True).
    Poll /status first to know when results are ready.
    """
    get_session_record(session_id)  # validates session exists
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    try:
        checkpoint = graph.get_state(config)
        state_dict = checkpoint.values if checkpoint else {}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve session state: {e}",
        )

    if not state_dict:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No pipeline state found for this session.",
        )

    state = SessionState(**state_dict)

    if not state.results_ready and not state.ranked_jobs:
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail="Results not ready yet. Poll /status for progress.",
        )

    # Map ranked jobs to response model
    jobs = [
        JobResult(
            job_id             = j.job_id,
            title              = j.title,
            company            = j.company,
            location           = j.location,
            work_type          = j.work_type,
            apply_url          = j.apply_url,
            source             = j.source,
            posted_date        = j.posted_date.isoformat() if j.posted_date else None,
            salary_min         = j.salary_min,
            salary_max         = j.salary_max,
            matched_via        = j.matched_via,
            fit_score          = round(j.fit_score, 3),
            gap_skills         = j.gap_skills,
            recommended_action = j.recommended_action,
        )
        for j in state.ranked_jobs
    ]

    # Map hiring signals
    watch_list = []
    for s in state.hiring_signals:
        from api.models import HiringSignalResult
        watch_list.append(HiringSignalResult(
            company                = s.company,
            signal_type            = s.signal_type,
            signal_strength        = s.signal_strength,
            summary                = s.summary,
            source_url             = s.source_url,
            source_date            = s.source_date.isoformat(),
            hiring_momentum_score  = round(s.hiring_momentum_score, 3),
        ))

    return ResultsResponse(
        session_id  = session_id,
        total_jobs  = len(jobs),
        jobs        = jobs,
        watch_list  = watch_list,
        message     = f"Found {len(jobs)} matching jobs across your selected profiles.",
    )


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API health check",
)
async def health_check() -> HealthResponse:
    """Returns API health status."""
    return HealthResponse(status="ok")
