"""
tests/unit/test_job_search_agent.py

Unit tests for Agent 3 — Job Search Agent.
All HTTP calls are mocked — no real API calls in unit tests.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.job_search.normalizer import (
    normalize_jobs,
    normalize_jsearch,
    normalize_remoteok,
    _make_job_id,
    _parse_date,
    _extract_salary,
)
from agents.job_search.sources.jsearch import _build_query
from agents.job_search.job_search_agent import (
    _should_include_remoteok,
    run_job_search_agent,
)
from core.state.session_state import (
    SessionState,
    SuggestedProfile,
    UserPreferences,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def session() -> SessionState:
    return SessionState(
        session_id=str(uuid.uuid4()),
        preferences=UserPreferences(
            location="Delhi NCR",
            work_type="hybrid",
            seniority_preference="step_up",
        ),
        confirmed_profiles=[
            SuggestedProfile(
                title="Senior Data Scientist",
                seniority_target="senior",
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
    )


@pytest.fixture
def sample_jsearch_job() -> dict:
    return {
        "job_id":                      "jsearch_abc123",
        "job_title":                   "Senior Data Scientist",
        "employer_name":               "Accenture",
        "job_city":                    "Delhi",
        "job_country":                 "IN",
        "job_description":             "We are looking for a Senior DS...",
        "job_apply_link":              "https://accenture.com/apply/123",
        "job_posted_at_datetime_utc":  "2024-01-15T10:00:00Z",
        "job_min_salary":              1500000,
        "job_max_salary":              2500000,
        "job_employment_type":         "FULLTIME",
        "job_is_remote":               False,
    }


@pytest.fixture
def sample_remoteok_job() -> dict:
    return {
        "id":          "123456",
        "position":    "ML Engineer",
        "company":     "RemoteCo",
        "url":         "https://remoteok.com/jobs/123456",
        "description": "Remote ML Engineer role...",
        "date":        "2024-01-15T10:00:00Z",
        "salary_min":  "80000",
        "salary_max":  "120000",
        "tags":        ["machine-learning", "python"],
        "location":    "Remote",
    }


# ── Normalizer helper tests ───────────────────────────────────────────────────

class TestNormalizerHelpers:
    def test_make_job_id_format(self):
        result = _make_job_id("jsearch", "abc123")
        assert result == "jsearch_abc123"

    def test_parse_date_iso(self):
        result = _parse_date("2024-01-15T10:00:00Z")
        assert result is not None
        assert result.year == 2024

    def test_parse_date_none(self):
        assert _parse_date(None) is None

    def test_parse_date_invalid(self):
        assert _parse_date("not-a-date") is None

    def test_extract_salary_int(self):
        assert _extract_salary(1500000) == 1500000

    def test_extract_salary_string(self):
        assert _extract_salary("80000") == 80000

    def test_extract_salary_none(self):
        assert _extract_salary(None) is None

    def test_extract_salary_malformed(self):
        assert _extract_salary("N/A") is None


# ── JSearch normalizer tests ──────────────────────────────────────────────────

class TestNormalizeJSearch:
    def test_normalizes_complete_job(self, sample_jsearch_job):
        result = normalize_jsearch(sample_jsearch_job, "Senior Data Scientist")
        assert result is not None
        assert result.title == "Senior Data Scientist"
        assert result.company == "Accenture"
        assert result.location == "Delhi, IN"
        assert result.source == "jsearch"
        assert result.apply_url == "https://accenture.com/apply/123"
        assert result.matched_profile == "Senior Data Scientist"
        assert result.salary_min == 1500000
        assert result.salary_max == 2500000

    def test_remote_job_work_type(self, sample_jsearch_job):
        sample_jsearch_job["job_is_remote"] = True
        result = normalize_jsearch(sample_jsearch_job, "DS")
        assert result.work_type == "remote"

    def test_returns_none_on_missing_required_fields(self):
        result = normalize_jsearch({"job_title": "DS"}, "DS")
        assert result is None

    def test_job_id_prefixed_with_source(self, sample_jsearch_job):
        result = normalize_jsearch(sample_jsearch_job, "DS")
        assert result.job_id.startswith("jsearch_")

    def test_truncates_long_description(self, sample_jsearch_job):
        sample_jsearch_job["job_description"] = "x" * 5000
        result = normalize_jsearch(sample_jsearch_job, "DS")
        assert len(result.jd_text) <= 3000


# ── RemoteOK normalizer tests ─────────────────────────────────────────────────

class TestNormalizeRemoteOK:
    def test_normalizes_complete_job(self, sample_remoteok_job):
        result = normalize_remoteok(sample_remoteok_job, "ML Engineer")
        assert result is not None
        assert result.title == "ML Engineer"
        assert result.company == "RemoteCo"
        assert result.work_type == "remote"
        assert result.source == "remoteok"
        assert result.matched_profile == "ML Engineer"

    def test_job_id_prefixed_with_source(self, sample_remoteok_job):
        result = normalize_remoteok(sample_remoteok_job, "MLE")
        assert result.job_id.startswith("remoteok_")

    def test_returns_none_on_missing_fields(self):
        result = normalize_remoteok({"position": "MLE"}, "MLE")
        assert result is None

    def test_salary_parsed_from_string(self, sample_remoteok_job):
        result = normalize_remoteok(sample_remoteok_job, "MLE")
        assert result.salary_min == 80000
        assert result.salary_max == 120000


# ── normalize_jobs dispatcher tests ──────────────────────────────────────────

class TestNormalizeJobs:
    def test_dispatches_to_correct_normalizer(self, sample_jsearch_job):
        results = normalize_jobs([sample_jsearch_job], "jsearch", "DS")
        assert len(results) == 1
        assert results[0].source == "jsearch"

    def test_skips_invalid_jobs(self):
        results = normalize_jobs([{"broken": True}], "jsearch", "DS")
        assert results == []

    def test_unknown_source_returns_empty(self, sample_jsearch_job):
        results = normalize_jobs([sample_jsearch_job], "unknown_source", "DS")
        assert results == []

    def test_handles_mixed_valid_invalid(self, sample_jsearch_job):
        jobs = [sample_jsearch_job, {"broken": True}, sample_jsearch_job]
        # Two valid, one invalid
        results = normalize_jobs(jobs, "jsearch", "DS")
        assert len(results) == 2


# ── Source activation logic ───────────────────────────────────────────────────

class TestSourceActivation:
    def test_remoteok_on_for_remote_work_type(self):
        assert _should_include_remoteok("remote", "Delhi NCR") is True

    def test_remoteok_on_for_remote_in_location(self):
        assert _should_include_remoteok("hybrid", "Remote") is True

    def test_remoteok_off_for_hybrid_onsite(self):
        assert _should_include_remoteok("hybrid", "Delhi NCR") is False

    def test_remoteok_off_for_onsite(self):
        assert _should_include_remoteok("on-site", "London") is False


# ── Agent orchestration tests ─────────────────────────────────────────────────

class TestRunJobSearchAgent:
    def test_successful_search_populates_raw_jobs(self, session, sample_jsearch_job):
        mock_jsearch_results  = [sample_jsearch_job]
        mock_remoteok_results = []

        async def mock_jsearch(*args, **kwargs):
            return mock_jsearch_results

        async def mock_remoteok(*args, **kwargs):
            return mock_remoteok_results

        with patch("agents.job_search.job_search_agent.search_jsearch",
                   side_effect=mock_jsearch), \
             patch("agents.job_search.job_search_agent.search_remoteok",
                   side_effect=mock_remoteok):
            result = asyncio.run(run_job_search_agent(session))

        # 1 job × 2 profiles = 2 raw jobs
        assert len(result.raw_jobs) == 2
        assert result.error is None
        assert result.current_agent == "job_search"

    def test_no_confirmed_profiles_sets_error(self):
        session = SessionState(
            session_id=str(uuid.uuid4()),
            preferences=UserPreferences(location="Delhi NCR"),
            confirmed_profiles=[],
        )
        result = asyncio.run(run_job_search_agent(session))
        assert result.error is not None
        assert result.raw_jobs == []

    def test_source_failure_doesnt_crash_pipeline(self, session):
        async def failing_jsearch(*args, **kwargs):
            raise Exception("API rate limited")

        with patch("agents.job_search.job_search_agent.search_jsearch",
                   side_effect=failing_jsearch):
            result = asyncio.run(run_job_search_agent(session))

        # Should complete without crashing, raw_jobs may be empty
        assert result.current_agent == "job_search"

    def test_session_id_preserved(self, session, sample_jsearch_job):
        original_id = session.session_id

        async def mock_jsearch(*args, **kwargs):
            return [sample_jsearch_job]

        with patch("agents.job_search.job_search_agent.search_jsearch",
                   side_effect=mock_jsearch):
            result = asyncio.run(run_job_search_agent(session))

        assert result.session_id == original_id

    def test_raw_jobs_capped_at_max(self, session):
        # Generate 100 fake jobs to test the cap
        fake_jobs = [
            {
                "job_id":    f"job_{i}",
                "job_title": "Data Scientist",
                "employer_name": f"Company {i}",
                "job_city": "Delhi",
                "job_country": "IN",
                "job_description": "Job desc",
                "job_apply_link": f"https://example.com/{i}",
                "job_posted_at_datetime_utc": None,
                "job_min_salary": None,
                "job_max_salary": None,
                "job_is_remote": False,
            }
            for i in range(100)
        ]

        async def mock_jsearch(*args, **kwargs):
            return fake_jobs

        with patch("agents.job_search.job_search_agent.search_jsearch",
                   side_effect=mock_jsearch):
            result = asyncio.run(run_job_search_agent(session))

        assert len(result.raw_jobs) <= 60  # MAX_RAW_JOBS
