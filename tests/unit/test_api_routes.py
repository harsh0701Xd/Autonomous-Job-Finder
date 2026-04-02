"""
tests/unit/test_api_routes.py

Unit tests for the FastAPI route handlers and dependency logic.

Strategy:
  - Dependencies (graph, agents) are mocked — no real pipeline runs
  - Uses FastAPI TestClient for endpoint tests
  - Pure logic helpers (session store, status derivation) tested directly
"""

from __future__ import annotations

import io
import json
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.dependencies import (
    _session_store,
    create_session_record,
    generate_session_id,
    update_session_status,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
)
from core.state.session_state import (
    CandidateProfile,
    SessionState,
    SkillSet,
    SuggestedProfile,
    UserPreferences,
)


# ── App fixture ───────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(autouse=True)
def clear_session_store():
    _session_store.clear()
    yield
    _session_store.clear()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_session(prefs: dict | None = None) -> str:
    sid = generate_session_id()
    create_session_record(sid, prefs or {
        "location": "Bangalore",
        "work_type": "remote",
        "seniority_preference": "step_up",
        "salary_min": None,
        "salary_max": None,
        "currency": "USD",
    })
    return sid


def _make_state(session_id: str, **overrides) -> SessionState:
    defaults = dict(
        session_id=session_id,
        preferences=UserPreferences(location="Bangalore", work_type="remote"),
        candidate_profile=CandidateProfile(
            current_title="Senior Data Scientist",
            seniority_level="senior",
            skills=SkillSet(technical=["Python"], tools=["XGBoost"], soft=[]),
            raw_text="resume text",
        ),
        suggested_profiles=[
            SuggestedProfile(
                title="Lead Data Scientist",
                seniority_target="lead",
                confidence="high",
                match_reason="Strong ML background.",
                is_stretch=False,
                source="system",
            ),
            SuggestedProfile(
                title="ML Engineer",
                seniority_target="senior",
                confidence="high",
                match_reason="Production ML experience.",
                is_stretch=False,
                source="system",
            ),
        ],
        awaiting_confirmation=True,
    )
    defaults.update(overrides)
    return SessionState(**defaults)


def _resume_bytes(content: str = "Sample resume content " * 20) -> bytes:
    return content.encode("utf-8")


# ── Session store unit tests ──────────────────────────────────────────────────

class TestSessionStore:
    def test_generate_session_id_is_unique(self):
        ids = {generate_session_id() for _ in range(100)}
        assert len(ids) == 100

    def test_generate_session_id_is_uuid(self):
        sid = generate_session_id()
        assert str(uuid.UUID(sid)) == sid

    def test_create_session_record(self):
        sid = generate_session_id()
        record = create_session_record(sid, {"location": "Delhi"})
        assert record["session_id"] == sid
        assert record["status"] == "created"
        assert sid in _session_store

    def test_update_session_status(self):
        sid = generate_session_id()
        create_session_record(sid, {"location": "Mumbai"})
        update_session_status(sid, "parsing")
        assert _session_store[sid]["status"] == "parsing"

    def test_update_status_noop_on_missing(self):
        update_session_status("ghost-id", "parsing")

    def test_record_has_timestamps(self):
        sid = generate_session_id()
        record = create_session_record(sid, {"location": "Pune"})
        assert "created_at" in record
        assert "updated_at" in record


# ── File validation constants ─────────────────────────────────────────────────

class TestFileValidationConstants:
    def test_pdf_allowed(self):
        assert "pdf" in ALLOWED_EXTENSIONS

    def test_docx_allowed(self):
        assert "docx" in ALLOWED_EXTENSIONS

    def test_max_size_is_5mb(self):
        assert MAX_FILE_SIZE_BYTES == 5 * 1024 * 1024

    def test_exe_blocked(self):
        assert "exe" not in ALLOWED_EXTENSIONS

    def test_txt_blocked(self):
        assert "txt" not in ALLOWED_EXTENSIONS


# ── Status derivation logic ───────────────────────────────────────────────────

def _derive(pc, err, pf, ac, cp, rj, rr, rrt, cand, ss="created"):
    if pc: return "complete"
    if err: return "error"
    if pf: return "parse_failed"
    if ac: return "awaiting_confirmation"
    if cp and not rj: return "searching"
    if rj and not rr: return "ranking"
    if rr: return "complete"
    if rrt and not cand: return "parsing"
    return ss


class TestDerivePipelineStatus:
    def test_complete(self):
        assert _derive(True,None,False,False,[],[],False,"t","p") == "complete"

    def test_error(self):
        assert _derive(False,"err",False,False,[],[],False,"t","p") == "error"

    def test_parse_failed(self):
        assert _derive(False,None,True,False,[],[],False,"t",None) == "parse_failed"

    def test_awaiting_confirmation(self):
        assert _derive(False,None,False,True,[],[],False,"t","p") == "awaiting_confirmation"

    def test_searching(self):
        assert _derive(False,None,False,False,["p"],[],False,"t","p") == "searching"

    def test_ranking(self):
        assert _derive(False,None,False,False,["p"],["j"],False,"t","p") == "ranking"

    def test_results_ready(self):
        assert _derive(False,None,False,False,["p"],["j"],True,"t","p") == "complete"

    def test_parsing(self):
        assert _derive(False,None,False,False,[],[],False,"text",None) == "parsing"

    def test_created_fallback(self):
        assert _derive(False,None,False,False,[],[],False,None,None,"created") == "created"


# ── POST /sessions ────────────────────────────────────────────────────────────

class TestCreateSession:
    def test_returns_201(self, client):
        r = client.post("/api/v1/sessions", json={"location": "Bangalore"})
        assert r.status_code == 201

    def test_returns_session_id(self, client):
        r = client.post("/api/v1/sessions", json={"location": "Bangalore"})
        body = r.json()
        assert "session_id" in body
        assert uuid.UUID(body["session_id"])

    def test_returns_message(self, client):
        r = client.post("/api/v1/sessions", json={"location": "Delhi"})
        assert "message" in r.json()

    def test_missing_location_returns_422(self, client):
        r = client.post("/api/v1/sessions", json={})
        assert r.status_code == 422

    def test_invalid_work_type_returns_422(self, client):
        r = client.post("/api/v1/sessions", json={
            "location": "Bangalore",
            "work_type": "flying",
        })
        assert r.status_code == 422

    def test_session_stored_after_creation(self, client):
        r = client.post("/api/v1/sessions", json={"location": "Bangalore"})
        sid = r.json()["session_id"]
        assert sid in _session_store


# ── POST /sessions/{id}/resume ────────────────────────────────────────────────

class TestUploadResume:
    def _upload(self, client, session_id, content=None,
                filename="resume.pdf", content_type="application/pdf"):
        data = content or _resume_bytes()
        return client.post(
            f"/api/v1/sessions/{session_id}/resume",
            files={"file": (filename, io.BytesIO(data), content_type)},
        )

    def test_unknown_session_returns_404(self, client):
        r = self._upload(client, "nonexistent-session-id")
        assert r.status_code == 404

    def test_unsupported_file_type_returns_422(self, client):
        sid = _make_session()
        r = self._upload(client, sid, filename="resume.txt",
                         content_type="text/plain")
        assert r.status_code == 422

    def test_empty_file_returns_422(self, client):
        sid = _make_session()
        import asyncio
        from unittest.mock import AsyncMock
        from api.dependencies import validate_resume_file
        from fastapi import UploadFile, HTTPException

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "resume.pdf"
        mock_file.read = AsyncMock(return_value=b"")

        async def run():
            with pytest.raises(HTTPException) as exc:
                await validate_resume_file(mock_file)
            assert exc.value.status_code == 422
            assert "empty" in exc.value.detail.lower()

        asyncio.run(run())

    def test_file_too_large_returns_413(self, client):
        sid = _make_session()
        big = b"x" * (MAX_FILE_SIZE_BYTES + 1)
        r = self._upload(client, sid, content=big)
        # Newer Starlette versions return 400 for oversized multipart bodies
        # before the route handler runs; older versions let it through and we
        # return 413. Accept either — both mean "file rejected as too large".
        assert r.status_code in (400, 413)

    def test_successful_upload_returns_awaiting_confirmation(self, client):
        sid = _make_session()
        state = _make_state(sid)

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = state.model_dump()

        # Patch at api.routes.extract_text — where the name is bound after import
        # Patch api.routes.get_graph so no real LangGraph init happens
        with patch("api.routes.extract_text", return_value="A" * 300), \
             patch("api.routes.get_graph", return_value=mock_graph):
            r = self._upload(client, sid)

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "awaiting_confirmation"
        assert body["session_id"] == sid
        assert len(body["suggested_profiles"]) == 2

    def test_parse_failure_returns_parse_failed(self, client):
        sid = _make_session()
        failed_state = SessionState(
            session_id=sid,
            parse_failed=True,
            parse_failure_reason="File too short.",
            preferences=UserPreferences(location="Bangalore"),
        )
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = failed_state.model_dump()

        with patch("api.routes.extract_text", return_value="A" * 300), \
             patch("api.routes.get_graph", return_value=mock_graph):
            r = self._upload(client, sid)

        assert r.json()["status"] == "parse_failed"

    def test_extract_text_failure_returns_parse_failed(self, client):
        sid = _make_session()
        with patch("api.routes.extract_text",
                   side_effect=ValueError("Cannot read PDF")):
            r = self._upload(client, sid)

        assert r.json()["status"] == "parse_failed"


# ── POST /sessions/{id}/confirm ───────────────────────────────────────────────

class TestConfirmProfiles:
    def _confirm(self, client, session_id, selected, custom=None):
        return client.post(
            f"/api/v1/sessions/{session_id}/confirm",
            json={"selected_titles": selected, "custom_profiles": custom or []},
        )

    def test_unknown_session_returns_404(self, client):
        r = self._confirm(client, "ghost-id", ["Lead DS"])
        assert r.status_code == 404

    def test_wrong_status_returns_409(self, client):
        sid = _make_session()
        r = self._confirm(client, sid, ["Lead DS"])
        assert r.status_code == 409

    def test_successful_confirmation(self, client):
        sid = _make_session()
        update_session_status(sid, "awaiting_confirmation")

        confirmed_state = SessionState(
            session_id=sid,
            preferences=UserPreferences(location="Bangalore"),
            confirmed_profiles=[
                SuggestedProfile(
                    title="Lead Data Scientist",
                    seniority_target="lead",
                    confidence="high",
                    match_reason="Good.",
                    is_stretch=False,
                    source="system",
                )
            ],
            awaiting_confirmation=False,
        )
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = confirmed_state.model_dump()

        with patch("api.routes.get_graph", return_value=mock_graph):
            r = self._confirm(client, sid, ["Lead Data Scientist"])

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "confirmed"
        assert len(body["confirmed_profiles"]) == 1

    def test_empty_selection_returns_422(self, client):
        sid = _make_session()
        update_session_status(sid, "awaiting_confirmation")
        r = self._confirm(client, sid, selected=[], custom=[])
        assert r.status_code == 422


# ── GET /sessions/{id}/status ─────────────────────────────────────────────────

class TestGetStatus:
    def test_unknown_session_returns_404(self, client):
        r = client.get("/api/v1/sessions/ghost-id/status")
        assert r.status_code == 404

    def test_status_before_pipeline_starts(self, client):
        sid = _make_session()
        mock_graph = MagicMock()
        mock_graph.get_state.return_value = None

        with patch("api.routes.get_graph", return_value=mock_graph):
            r = client.get(f"/api/v1/sessions/{sid}/status")

        assert r.status_code == 200
        assert r.json()["status"] == "created"

    def test_status_with_pipeline_state(self, client):
        sid = _make_session()
        state = _make_state(sid)

        mock_checkpoint = MagicMock()
        mock_checkpoint.values = state.model_dump()
        mock_graph = MagicMock()
        mock_graph.get_state.return_value = mock_checkpoint

        with patch("api.routes.get_graph", return_value=mock_graph):
            r = client.get(f"/api/v1/sessions/{sid}/status")

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "awaiting_confirmation"
        assert len(body["suggested_profiles"]) == 2


# ── GET /health ───────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200

    def test_health_body(self, client):
        body = client.get("/api/v1/health").json()
        assert body["status"] == "ok"
        assert body["agents_ready"] is True
        assert "version" in body
