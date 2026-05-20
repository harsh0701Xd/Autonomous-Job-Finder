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
from fastapi.responses import JSONResponse

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
from langgraph.types import Command
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
        f"location={body.location}, seniority={body.seniority_preference}"
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
    file:             UploadFile = File(..., description="Resume file (PDF or DOCX only, max 5MB; legacy .doc not supported)"),
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
        seniority_preference = prefs_data["seniority_preference"],
    )

    initial_state = SessionState(
        session_id        = session_id,
        resume_file_name  = file.filename,
        preferences       = preferences,
    ).model_dump()

    # Store raw file bytes under a key the graph node can access
    # We pass them directly into the initial state as a special key
    # The parse node reads from resume_raw_text OR file bytes
    # Here we trigger extraction immediately before graph entry
    from agents.parser.resume_parser import extract_text
    try:
        raw_text = extract_text(file_bytes, file_ext)
        initial_state["resume_raw_text"] = raw_text
    except ValueError as e:
        update_session_status(session_id, "parse_failed")
        return ResumeUploadResponse(
            session_id          = session_id,
            status              = "parse_failed",
            message             = str(e),
            parse_failure_reason = str(e),
        )

    update_session_status(session_id, "parsing")
    logger.info(f"[routes] Starting pipeline for session {session_id}")

    # Run graph until it hits the confirmation interrupt
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    try:
        final_state_dict = await graph.ainvoke(initial_state, config=config)
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
#
# NON-BLOCKING DESIGN:
#   Returns HTTP 202 immediately. Pipeline runs as a BackgroundTask so
#   the HTTP connection is never held open during the 45-90s ranker.
#   Streamlit polls /status every 2s (already implemented) — zero frontend changes.
#
# PREVIOUS DESIGN (blocking):
#   await graph.ainvoke(Command(resume=...))  ← held connection for ~90s
#   Risk: uvicorn timeout-keep-alive=120s could drop the connection if
#   ranker took longer than expected, silently losing results.

async def _run_pipeline_in_background(
    session_id:     str,
    resume_payload: dict,
    config:         dict,
) -> None:
    """
    Resume the LangGraph pipeline from the confirmation interrupt.

    Called as a BackgroundTask — the HTTP connection is already closed.
    Results are written to Postgres via the checkpointer. Streamlit polls
    /status until results_ready=true, then fetches /results.

    On any exception: sets status="error" in Postgres so the poll loop
    can surface it to the frontend via the existing error branch.
    """
    import traceback

    logger.info(
        f"[routes:bg] Pipeline resuming in background — session_id={session_id}"
    )
    graph = get_graph()

    try:
        await graph.ainvoke(
            Command(resume=resume_payload),
            config=config,
        )
        logger.info(
            f"[routes:bg] Pipeline complete — session_id={session_id}"
        )

    except Exception as e:
        logger.error(
            f"[routes:bg] Pipeline error for {session_id}\n"
            f"Type: {type(e).__name__} | {e!r}\n"
            f"{traceback.format_exc()}"
        )
        update_session_status(session_id, "error")


@router.post(
    "/sessions/{session_id}/confirm",
    status_code=202,
    summary="Submit user's profile confirmation and resume the pipeline",
)
async def confirm_profiles(
    session_id:       str,
    body:             ConfirmProfilesRequest,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Step 3: User confirms which profiles to search for.

    Returns HTTP 202 immediately. The pipeline (job search → url pruner →
    ranker → finalise) runs as a BackgroundTask, completely decoupled from
    this HTTP connection. The frontend polls GET /status for progress.

    This eliminates the race condition where a slow ranker (~90s) could
    collide with uvicorn's timeout-keep-alive=120s and drop the connection.
    """
    session_record = get_session_record(session_id)

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

    config = {"configurable": {"thread_id": session_id}}

    resume_payload = {
        "selected_titles": body.selected_titles,
        "custom_profiles": body.custom_profiles,
    }

    # Set status immediately so /status shows "searching" before first poll
    update_session_status(session_id, "searching")

    # Fire the pipeline — returns before it runs
    background_tasks.add_task(
        _run_pipeline_in_background,
        session_id=session_id,
        resume_payload=resume_payload,
        config=config,
    )

    logger.info(
        f"[routes] Confirmation received for {session_id} — "
        f"pipeline queued as BackgroundTask | profiles={body.selected_titles}"
    )

    return JSONResponse(
        status_code=202,
        content={
            "session_id": session_id,
            "status":     "pipeline_started",
            "message":    (
                f"Confirmed {len(body.selected_titles)} profile(s). "
                f"Job search started. Poll /status for progress."
            ),
        },
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
        checkpoint = await graph.aget_state(config)
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

    # Empty-results case: distinguish "nothing came back from job sources" from
    # "everything got filtered by the experience-score floor" so the user gets
    # an actionable message instead of just "Done! Found 0 matching jobs."
    if pipeline_status == "complete" and len(state.ranked_jobs) == 0:
        if not state.raw_jobs:
            complete_msg = (
                "No jobs were returned from the search APIs for your selected "
                "profiles. This usually means the API quota is exhausted for "
                "this month, the profile titles are too narrow for your "
                "location, or a job source is temporarily down. "
                "Try broader profile titles or check your JSearch API quota."
            )
        else:
            complete_msg = (
                f"Searched {len(state.raw_jobs)} jobs but none passed the "
                f"experience-fit threshold for your profile. Try selecting "
                f"different profile titles or relax the seniority filter."
            )
    else:
        complete_msg = f"Done! Found {len(state.ranked_jobs)} matching jobs."

    status_messages = {
        "created":               "Session ready. Upload your resume to begin.",
        "parsing":               "Parsing your resume...",
        "parse_failed":          f"Resume parsing failed: {state.parse_failure_reason}",
        "awaiting_confirmation": "Profiles ready. Waiting for your selection.",
        "searching":             "Searching job boards across your selected profiles...",
        "ranking":               "Ranking and deduplicating results...",
        "complete":              complete_msg,
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
    session_record = get_session_record(session_id)  # validates session exists
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    try:
        checkpoint = await graph.aget_state(config)
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

    pipeline_status = _derive_pipeline_status(state, session_record)
    if pipeline_status not in ("complete", "ranking", "searching") and not state.results_ready and not state.ranked_jobs:
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
            matched_via        = j.matched_via,
            matched_profile    = j.matched_profile,
            fit_score          = round(j.fit_score,          3),
            experience_score   = round(j.experience_score,   3),
            skill_score        = round(j.skill_score,        3),
            domain_score       = round(j.domain_score,       3),
            recency_score      = round(j.recency_score,      3),
            education_score    = round(j.education_score, 3) if j.education_score is not None else None,
            education_required = j.education_required,
            scoring_notes      = j.scoring_notes,
            experience_gap     = j.experience_gap,
            skill_gaps         = j.skill_gaps,
            domain_gap         = j.domain_gap,
            education_gap      = j.education_gap,
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

    if jobs:
        msg = f"Found {len(jobs)} matching jobs across your selected profiles."
    elif not state.raw_jobs:
        msg = (
            "No jobs were returned from the search APIs. Likely an exhausted "
            "API quota or transient source outage -- try again later or with "
            "broader profile titles."
        )
    else:
        msg = (
            f"Searched {len(state.raw_jobs)} jobs but none passed the "
            f"experience-fit threshold for your profile."
        )

    return ResultsResponse(
        session_id      = session_id,
        total_jobs      = len(jobs),
        jobs            = jobs,
        watch_list      = watch_list,
        message         = msg,
        session_metrics = state.session_metrics,
    )


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API health check",
)
async def health_check() -> HealthResponse:
    """Returns API health status."""
    return Hea