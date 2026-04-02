"""
tests/unit/test_profile_recommender.py

Unit tests for Agent 2 — Profile Recommender.
All Claude API calls are mocked — no real API calls in unit tests.
"""

import json
import uuid
from unittest.mock import patch

import pytest

from agents.recommender.profile_recommender import (
    _clean_response,
    _format_salary_range,
    _parse_profiles,
    _validate_preconditions,
    apply_user_confirmation,
    run_profile_recommender,
)
from core.state.session_state import (
    CandidateProfile,
    Education,
    SessionState,
    SkillSet,
    SuggestedProfile,
    UserPreferences,
    WorkExperience,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def preferences() -> UserPreferences:
    return UserPreferences(
        location="Bangalore",
        work_type="remote",
        seniority_preference="step_up",
        salary_min=3200000,
        salary_max=5000000,
        currency="INR",
    )


@pytest.fixture
def candidate_profile() -> CandidateProfile:
    return CandidateProfile(
        current_title="Senior Analyst - Data Science",
        years_experience=3.5,
        seniority_level="senior",
        skills=SkillSet(
            technical=["Python", "SQL", "Machine Learning", "NLP"],
            tools=["XGBoost", "scikit-learn", "Airflow", "Tableau"],
            soft=["Leadership", "Communication"],
        ),
        domain_expertise=["Financial Services", "Credit Risk", "Fraud Detection"],
        career_trajectory="ascending",
        pivot_signals=[],
        work_experience=[
            WorkExperience(
                title="Senior Analyst - Data Science",
                company="American Express",
                duration_months=15,
                responsibilities=["Built ML models for credit risk"],
                impact_signals=["Reduced false positive rate by 23%"],
            )
        ],
        education=[
            Education(
                degree="B.Tech",
                field="Computer Science",
                institution="IIT Delhi",
                year=2021,
            )
        ],
        raw_text="sample resume text",
    )


@pytest.fixture
def session(preferences, candidate_profile) -> SessionState:
    return SessionState(
        session_id=str(uuid.uuid4()),
        preferences=preferences,
        candidate_profile=candidate_profile,
    )


@pytest.fixture
def sample_profiles_json() -> str:
    return json.dumps([
        {
            "title": "Lead Data Scientist",
            "seniority_target": "lead",
            "confidence": "high",
            "match_reason": "Strong ML experience with measurable impact at Amex.",
            "is_stretch": False,
        },
        {
            "title": "ML Engineer",
            "seniority_target": "senior",
            "confidence": "high",
            "match_reason": "Production ML pipeline experience bridges DS and engineering.",
            "is_stretch": False,
        },
        {
            "title": "AI Product Manager",
            "seniority_target": "mid",
            "confidence": "medium",
            "match_reason": "Cross-functional leadership signals PM aptitude.",
            "is_stretch": True,
        },
    ])


# ── Formatting tests ──────────────────────────────────────────────────────────

class TestFormatSalaryRange:
    def test_formats_full_range(self):
        prefs = UserPreferences(
            location="Bangalore",
            salary_min=3200000,
            salary_max=5000000,
            currency="INR",
        )
        result = _format_salary_range(prefs)
        assert "3,200,000" in result
        assert "5,000,000" in result
        assert "INR" in result

    def test_formats_min_only(self):
        prefs = UserPreferences(location="Delhi", salary_min=80000, currency="USD")
        result = _format_salary_range(prefs)
        assert "80,000+" in result

    def test_formats_not_specified(self):
        prefs = UserPreferences(location="Mumbai")
        result = _format_salary_range(prefs)
        assert result == "not specified"


# ── Response cleaning tests ───────────────────────────────────────────────────

class TestCleanResponse:
    def test_strips_json_fence(self):
        raw = "```json\n[{\"title\": \"DS\"}]\n```"
        assert "```" not in _clean_response(raw)

    def test_passes_through_clean_array(self):
        raw = '[{"title": "DS"}]'
        assert _clean_response(raw) == raw


# ── Profile parsing tests ─────────────────────────────────────────────────────

class TestParseProfiles:
    def test_parses_valid_profiles(self, sample_profiles_json):
        profiles = _parse_profiles(sample_profiles_json)
        assert len(profiles) == 3
        assert profiles[0].title == "Lead Data Scientist"
        assert profiles[0].confidence == "high"
        assert profiles[0].is_stretch is False
        assert profiles[0].source == "system"

    def test_identifies_stretch_roles(self, sample_profiles_json):
        profiles = _parse_profiles(sample_profiles_json)
        stretch = [p for p in profiles if p.is_stretch]
        assert len(stretch) == 1
        assert stretch[0].title == "AI Product Manager"

    def test_caps_at_max_profiles(self):
        many = [
            {
                "title": f"Role {i}",
                "seniority_target": "senior",
                "confidence": "high",
                "match_reason": "Good match.",
                "is_stretch": False,
            }
            for i in range(10)
        ]
        profiles = _parse_profiles(json.dumps(many))
        assert len(profiles) == 5  # MAX_PROFILES

    def test_raises_on_invalid_json(self):
        with pytest.raises(ValueError, match="invalid JSON"):
            _parse_profiles("not json {{{")

    def test_raises_on_non_array(self):
        with pytest.raises(ValueError, match="Expected JSON array"):
            _parse_profiles('{"title": "DS"}')

    def test_raises_when_too_few_valid_profiles(self):
        one = json.dumps([{
            "title": "DS",
            "seniority_target": "senior",
            "confidence": "high",
            "match_reason": "Good.",
            "is_stretch": False,
        }])
        with pytest.raises(ValueError, match="Too few valid profiles"):
            _parse_profiles(one)

    def test_skips_malformed_items_but_keeps_valid(self):
        mixed = json.dumps([
            {"title": "DS", "seniority_target": "senior",
             "confidence": "high", "match_reason": "Good.", "is_stretch": False},
            {"broken": True},   # missing required fields
            {"title": "MLE", "seniority_target": "senior",
             "confidence": "medium", "match_reason": "Good.", "is_stretch": False},
        ])
        profiles = _parse_profiles(mixed)
        assert len(profiles) == 2
        assert profiles[0].title == "DS"
        assert profiles[1].title == "MLE"


# ── Precondition validation tests ─────────────────────────────────────────────

class TestValidatePreconditions:
    def test_passes_with_complete_state(self, session):
        assert _validate_preconditions(session) is None

    def test_fails_when_parse_failed(self, session):
        session.parse_failed = True
        session.parse_failure_reason = "bad pdf"
        result = _validate_preconditions(session)
        assert result is not None
        assert "parsing failed" in result.lower()

    def test_fails_when_no_candidate_profile(self, session):
        session.candidate_profile = None
        result = _validate_preconditions(session)
        assert result is not None
        assert "candidate_profile" in result

    def test_fails_when_no_preferences(self, session):
        session.preferences = None
        result = _validate_preconditions(session)
        assert result is not None
        assert "preferences" in result

    def test_fails_when_profile_is_empty(self, session):
        session.candidate_profile = CandidateProfile(raw_text="something")
        result = _validate_preconditions(session)
        assert result is not None
        assert "empty" in result.lower()


# ── Agent orchestration tests ─────────────────────────────────────────────────

class TestRunProfileRecommender:
    def test_successful_recommendation(self, session, sample_profiles_json):
        with patch(
            "agents.recommender.profile_recommender._call_claude",
            return_value=sample_profiles_json,
        ):
            result = run_profile_recommender(session)

        assert result.error is None
        assert len(result.suggested_profiles) == 3
        assert result.awaiting_confirmation is True
        assert result.current_agent == "profile_recommender"

    def test_fallback_used_on_primary_failure(self, session, sample_profiles_json):
        call_count = {"n": 0}

        def mock_claude(prompt):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return "INVALID JSON {{{"
            return sample_profiles_json

        with patch(
            "agents.recommender.profile_recommender._call_claude",
            side_effect=mock_claude,
        ):
            result = run_profile_recommender(session)

        assert result.error is None
        assert len(result.suggested_profiles) == 3
        assert call_count["n"] == 2

    def test_error_set_when_both_attempts_fail(self, session):
        with patch(
            "agents.recommender.profile_recommender._call_claude",
            return_value="INVALID {{{",
        ):
            result = run_profile_recommender(session)

        assert result.error is not None
        assert result.suggested_profiles == []

    def test_precondition_failure_returns_error(self, session):
        session.candidate_profile = None
        result = run_profile_recommender(session)
        assert result.error is not None
        assert result.suggested_profiles == []

    def test_awaiting_confirmation_set_true(self, session, sample_profiles_json):
        with patch(
            "agents.recommender.profile_recommender._call_claude",
            return_value=sample_profiles_json,
        ):
            result = run_profile_recommender(session)

        assert result.awaiting_confirmation is True

    def test_session_id_preserved(self, session, sample_profiles_json):
        original_id = session.session_id
        with patch(
            "agents.recommender.profile_recommender._call_claude",
            return_value=sample_profiles_json,
        ):
            result = run_profile_recommender(session)

        assert result.session_id == original_id


# ── Confirmation handler tests ────────────────────────────────────────────────

class TestApplyUserConfirmation:
    @pytest.fixture
    def session_with_suggestions(self, session, sample_profiles_json) -> SessionState:
        """Session that already has suggested profiles from the recommender."""
        with patch(
            "agents.recommender.profile_recommender._call_claude",
            return_value=sample_profiles_json,
        ):
            return run_profile_recommender(session)

    def test_confirms_selected_profiles(self, session_with_suggestions):
        result = apply_user_confirmation(
            session_with_suggestions,
            selected_titles=["Lead Data Scientist", "ML Engineer"],
        )
        assert len(result.confirmed_profiles) == 2
        titles = [p.title for p in result.confirmed_profiles]
        assert "Lead Data Scientist" in titles
        assert "ML Engineer" in titles

    def test_awaiting_confirmation_cleared(self, session_with_suggestions):
        result = apply_user_confirmation(
            session_with_suggestions,
            selected_titles=["Lead Data Scientist"],
        )
        assert result.awaiting_confirmation is False

    def test_adds_custom_profiles(self, session_with_suggestions):
        result = apply_user_confirmation(
            session_with_suggestions,
            selected_titles=["ML Engineer"],
            custom_profiles=["Data Analytics Lead"],
        )
        assert len(result.confirmed_profiles) == 2
        custom = [p for p in result.confirmed_profiles if p.source == "user_custom"]
        assert len(custom) == 1
        assert custom[0].title == "Data Analytics Lead"

    def test_custom_profile_source_is_user_custom(self, session_with_suggestions):
        result = apply_user_confirmation(
            session_with_suggestions,
            selected_titles=[],
            custom_profiles=["My Custom Role"],
        )
        assert result.confirmed_profiles[0].source == "user_custom"

    def test_ignores_unknown_selected_titles(self, session_with_suggestions):
        result = apply_user_confirmation(
            session_with_suggestions,
            selected_titles=["Nonexistent Role"],
        )
        assert result.confirmed_profiles == []

    def test_ignores_blank_custom_profiles(self, session_with_suggestions):
        result = apply_user_confirmation(
            session_with_suggestions,
            selected_titles=["ML Engineer"],
            custom_profiles=["", "  ", "Real Role"],
        )
        custom = [p for p in result.confirmed_profiles if p.source == "user_custom"]
        assert len(custom) == 1
        assert custom[0].title == "Real Role"

    def test_confirmed_profiles_inherit_seniority(self, session_with_suggestions):
        result = apply_user_confirmation(
            session_with_suggestions,
            selected_titles=[],
            custom_profiles=["New Role"],
        )
        # Should inherit seniority_level from candidate_profile ("senior")
        assert result.confirmed_profiles[0].seniority_target == "senior"
