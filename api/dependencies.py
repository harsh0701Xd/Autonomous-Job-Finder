"""
api/dependencies.py

FastAPI dependency injection layer.

Provides:
  - A single shared LangGraph instance (built once at startup)
  - A lightweight in-memory session store for development
    (swap for Redis/Postgres in production)
  - File validation helpers for resume uploads
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from typing import Optional

from fastapi import HTTPException, UploadFile, status

logger = logging.getLogger(__name__)

# ── Allowed file types ────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {"pdf", "docx", "doc"}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB


# ── LangGraph singleton ───────────────────────────────────────────────────────

_graph = None


def get_graph():
    """
    Returns the compiled LangGraph instance.
    Built once at application startup and reused for all requests.

    Uses Postgres checkpointer in production (APP_ENV=production),
    in-memory checkpointer in development.
    """
    global _graph
    if _graph is None:
        from core.graph import build_graph
        use_postgres = os.getenv("APP_ENV", "development") == "production"
        _graph = build_graph(use_postgres=use_postgres)
        logger.info(
            f"[dependencies] Graph initialised "
            f"({'postgres' if use_postgres else 'memory'} checkpointer)"
        )
    return _graph


# ── In-memory session metadata store ─────────────────────────────────────────
# Stores lightweight metadata per session_id.
# In production this would be a Redis hash or Postgres row.
# The full pipeline state lives in LangGraph's checkpointer —
# this store only tracks things the API layer needs (status, created_at).

_session_store: dict[str, dict] = {}


def create_session_record(session_id: str, preferences: dict) -> dict:
    """Create and store a new session record."""
    record = {
        "session_id":  session_id,
        "status":      "created",
        "preferences": preferences,
        "created_at":  datetime.utcnow().isoformat(),
        "updated_at":  datetime.utcnow().isoformat(),
    }
    _session_store[session_id] = record
    return record


def get_session_record(session_id: str) -> dict:
    """
    Retrieve session metadata.
    Raises 404 if session doesn't exist.
    """
    record = _session_store.get(session_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return record


def update_session_status(session_id: str, new_status: str) -> None:
    """Update the status field of an existing session record."""
    if session_id in _session_store:
        _session_store[session_id]["status"] = new_status
        _session_store[session_id]["updated_at"] = datetime.utcnow().isoformat()


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())


# ── File validation ───────────────────────────────────────────────────────────

async def validate_resume_file(file: UploadFile) -> tuple[bytes, str]:
    """
    Validate an uploaded resume file.

    Checks:
      - File extension is PDF or DOCX
      - File size is within the 5MB limit
      - File is not empty

    Returns (file_bytes, file_extension) on success.
    Raises HTTPException on validation failure.
    """
    # Check extension
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported file type: '.{ext}'. "
                f"Please upload a PDF or DOCX file."
            ),
        )

    # Read file bytes
    file_bytes = await file.read()

    # Check not empty
    if not file_bytes:
        raise HTTPException(
            status_code=422,
            detail="Uploaded file is empty.",
        )

    # Check file size
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large: {size_mb:.1f}MB. "
                f"Maximum allowed size is 5MB."
            ),
        )

    logger.debug(
        f"[dependencies] File validated: {filename} "
        f"({len(file_bytes) / 1024:.1f}KB, .{ext})"
    )

    return file_bytes, ext
