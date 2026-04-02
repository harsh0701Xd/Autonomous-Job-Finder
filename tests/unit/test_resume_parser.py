"""
tests/unit/test_resume_parser.py

Unit tests for Agent 1 — Resume Parser.
All Claude API calls are mocked — no real API calls in unit tests.
"""

import json
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from agents.parser.resume_parser import (
    clean_json_response,
    extract_text_from_pdf,
    parse_profile_from_json,
    run_resume_parser,
)
from core.state.session_state import SessionState, UserPreferences


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def session() -> SessionState:
    return SessionState(
        session_id=str(uuid.uuid4()),
        preferences=UserPreferences(location="Bangalore", work_type="remote"),
    )


@pytest.fixture
def sample_resume_text() -> str:
    return """
    Harsh Sharma
    Senior Data Scientist | harshsharma@email.com | Delhi NCR

    EXPERIENCE
    Senior Analyst - Data Science | American Express | Jan 2024 - Present
    - Built ML models for credit risk scoring using XGBoost and LightGBM
    - Reduced false positive rate by 23% through feature engineering
    - Led cross-functional team of 5 to deliver real-time fraud detection pipeline

    Data Analyst | HDFC Bank | Jun 2021 - Dec 2023 (30 months)
    - Designed ETL pipelines using Python and Airflow
    - Built dashboards in Tableau for senior leadership

    SKILLS
    Technical: Python, SQL, Machine Learning, Deep Learning, NLP
    Tools: XGBoost, LightGBM, scikit-learn, TensorFlow, Airflow, Tableau, Git
    Soft: Leadership, Communication, Problem-solving

    EDUCATION
    B.Tech Computer Science | IIT Delhi | 2021

    PROJECTS
    - Credit Risk Scoring Engine: End-to-end ML pipeline using XGBoost, deployed on AWS
    - NLP Sentiment Analyzer: BERT-based model for customer feedback analysis
    """


@pytest.fixture
def sample_parsed_json() -> dict:
    return {
        "current_title": "Senior Analyst - Data Science",
        "years_experience": 3.5,
        "seniority_level": "senior",
        "skills": {
            "technical": ["Python", "SQL", "Machine Learning", "Deep Learning", "NLP"],
            "tools": ["XGBoost", "LightGBM", "scikit-learn", "TensorFlow", "Airflow"],
            "soft": ["Leadership", "Communication", "Problem-solving"],
        },
        "education": [
            {
                "degree": "B.Tech",
                "field": "Computer Science",
                "institution": "IIT Delhi",
                "year": 2021,
            }
        ],
        "work_experience": [
            {
                "title": "Senior Analyst - Data Science",
                "company": "American Express",
                "duration_months": 15,
                "responsibilities": ["Built ML models for credit risk scoring"],
                "impact_signals": ["Reduced false positive rate by 23%"],
            }
        ],
        "career_trajectory": "ascending",
        "pivot_signals": [],
        "domain_expertise": ["Financial Services", "Credit Risk", "Fraud Detection"],
        "notable_projects": [
            {
                "name": "Credit Risk Scoring Engine",
                "description": "End-to-end ML pipeline using XGBoost, deployed on AWS",
                "tech_used": ["XGBoost", "AWS"],
            }
        ],
        "career_gaps": [],
    }


# ── Text cleaning tests ───────────────────────────────────────────────────────

class TestCleanJsonResponse:
    def test_strips_markdown_fences(self):
        raw = "```json\n{\"key\": \"value\"}\n```"
        assert clean_json_response(raw) == '{"key": "value"}'

    def test_strips_plain_fences(self):
        raw = "```\n{\"key\": \"value\"}\n```"
        assert clean_json_response(raw) == '{"key": "value"}'

    def test_passes_through_clean_json(self):
        raw = '{"key": "value"}'
        assert clean_json_response(raw) == raw

    def test_strips_whitespace(self):
        raw = '  \n{"key": "value"}\n  '
        assert clean_json_response(raw) == '{"key": "value"}'


# ── JSON parsing tests ────────────────────────────────────────────────────────

class TestParseProfileFromJson:
    def test_parses_complete_profile(self, sample_parsed_json, sample_resume_text):
        raw = json.dumps(sample_parsed_json)
        profile = parse_profile_from_json(raw, sample_resume_text)

        assert profile.current_title == "Senior Analyst - Data Science"
        assert profile.seniority_level == "senior"
        assert profile.years_experience == 3.5
        assert "Python" in profile.skills.technical
        assert "XGBoost" in profile.skills.tools
        assert len(profile.education) == 1
        assert profile.education[0].institution == "IIT Delhi"
        assert profile.career_trajectory == "ascending"
        assert profile.raw_text == sample_resume_text

    def test_handles_null_fields(self, sample_resume_text):
        minimal = {
            "current_title": None,
            "years_experience": None,
            "seniority_level": None,
            "skills": {"technical": [], "tools": [], "soft": []},
            "education": [],
            "work_experience": [],
            "career_trajectory": None,
            "pivot_signals": [],
            "domain_expertise": [],
            "notable_projects": [],
            "career_gaps": [],
        }
        profile = parse_profile_from_json(json.dumps(minimal), sample_resume_text)
        assert profile.current_title is None
        assert profile.skills.technical == []

    def test_raises_on_invalid_json(self, sample_resume_text):
        with pytest.raises(ValueError, match="invalid JSON"):
            parse_profile_from_json("not json at all {{{", sample_resume_text)

    def test_handles_missing_skills_key(self, sample_resume_text):
        data = {
            "current_title": "Data Scientist",
            "years_experience": 3,
            "seniority_level": "mid",
            "education": [],
            "work_experience": [],
            "career_trajectory": "ascending",
            "pivot_signals": [],
            "domain_expertise": [],
            "notable_projects": [],
            "career_gaps": [],
        }
        # skills key missing entirely — should default to empty
        profile = parse_profile_from_json(json.dumps(data), sample_resume_text)
        assert profile.skills.technical == []


# ── Agent orchestration tests ─────────────────────────────────────────────────

class TestRunResumeParser:
    def test_successful_parse_from_text(
        self, session, sample_resume_text, sample_parsed_json
    ):
        session.resume_raw_text = sample_resume_text

        with patch(
            "agents.parser.resume_parser.call_claude_for_parse",
            return_value=json.dumps(sample_parsed_json),
        ):
            result = run_resume_parser(session)

        assert result.parse_failed is False
        assert result.candidate_profile is not None
        assert result.candidate_profile.current_title == "Senior Analyst - Data Science"
        assert result.current_agent == "resume_parser"

    def test_parse_failure_on_short_text(self, session):
        session.resume_raw_text = "Too short"
        result = run_resume_parser(session)

        assert result.parse_failed is True
        assert result.parse_failure_reason is not None
        assert "too short" in result.parse_failure_reason.lower()
        assert result.candidate_profile is None

    def test_parse_failure_on_no_input(self, session):
        result = run_resume_parser(session)

        assert result.parse_failed is True
        assert result.candidate_profile is None

    def test_fallback_prompt_used_on_json_error(
        self, session, sample_resume_text, sample_parsed_json
    ):
        session.resume_raw_text = sample_resume_text

        call_count = {"n": 0}

        def mock_claude(text, use_fallback=False):
            call_count["n"] += 1
            if not use_fallback:
                return "INVALID JSON {{{"
            return json.dumps(sample_parsed_json)

        with patch("agents.parser.resume_parser.call_claude_for_parse", side_effect=mock_claude):
            result = run_resume_parser(session)

        assert result.parse_failed is False
        assert result.candidate_profile is not None
        assert call_count["n"] == 2  # primary + fallback

    def test_parse_failed_when_both_attempts_fail(self, session, sample_resume_text):
        session.resume_raw_text = sample_resume_text

        with patch(
            "agents.parser.resume_parser.call_claude_for_parse",
            return_value="INVALID {{{",
        ):
            result = run_resume_parser(session)

        assert result.parse_failed is True
        assert result.candidate_profile is None

    def test_state_preserves_session_id(self, session, sample_resume_text, sample_parsed_json):
        original_id = session.session_id
        session.resume_raw_text = sample_resume_text

        with patch(
            "agents.parser.resume_parser.call_claude_for_parse",
            return_value=json.dumps(sample_parsed_json),
        ):
            result = run_resume_parser(session)

        assert result.session_id == original_id
