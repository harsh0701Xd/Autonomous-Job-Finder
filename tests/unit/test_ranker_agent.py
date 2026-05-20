"""
tests/unit/test_ranker_agent.py

Unit tests for Agent 4 -- Ranker + Deduplication Agent.
No external API calls -- pure logic testing.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from agents.ranker.ranker_agent import (
    _build_candidate_keywords,
    _fingerprint,
    _gap_skills,
    _recommended_action,
    _recency_score,
    _semantic_score,
    _seniority_score,
    deduplicate,
    rank_jobs,
    run_ranker_agent,
)
from core.state.session_state import (
    CandidateProfile,
    RankedJob,
    RawJob,
    SessionState,
    SkillSet,
    SuggestedProfile,
    UserPreferences,
    WorkExperience,
)


# -- Fixtures ------------------------------------------------------------------

@pytest.fixture
def candidate_profile() -> CandidateProfile:
    return CandidateProfile(
        current_title="Senior Data Scientist",
        seniority_level="senior",
        skills=SkillSet(
            technical=["Python", "Machine Learning", "SQL", "NLP", "XGBoost"],
            tools=["Airflow", "Docker", "FastAPI", "scikit-learn"],
            soft=["Leadership"],
        ),
        domain_expertise=["AML", "Financial Services", "Fraud Detection"],
        career_trajectory="ascending",
        work_experience=[
            WorkExperience(
                title="Senior Analyst",
                company="American Express",
                duration_months=15,
                responsibilities=[],
                impact_signals=[],
            )
        ],
        raw_text="resume text",
    )


@pytest.fixture
def confirmed_profiles() -> list[SuggestedProfile]:
    return [
        SuggestedProfile(
            title="Senior Data Scientist",
            seniority_target="senior",
            confidence="high",
            match_reason="Strong ML.",
            is_stretch=False,
            source="system",
        ),
        SuggestedProfile(
            title="ML Engineer",
            seniority_target="senior",
            confidence="high",
            match_reason="Production ML.",
            is_stretch=False,
            source="system",
        ),
    ]


def _make_job(
    title:           str = "Senior Data Scientist",
    company:         str = "Accenture",
    jd_text:         str = "We need Python machine learning SQL experience",
    days_old:        int = 5,
    matched_profile: str = "Senior Data Scientist",
    job_id:          str | None = None,
) -> RawJob:
    return RawJob(
        job_id          = job_id or f"jsearch_{uuid.uuid4().hex[:8]}",
        title           = title,
        company         = company,
        location        = "Bengaluru, IN",
        work_type       = "hybrid",
        jd_text         = jd_text,
        apply_url       = f"https://example.com/{uuid.uuid4().hex[:6]}",
        source          = "jsearch",
        posted_date     = datetime.now(timezone.utc) - timedelta(days=days_old),
        matched_profile = matched_profile,
    )


@pytest.fixture
def session(candidate_profile, confirmed_profiles) -> SessionState:
    return SessionState(
        session_id       = str(uuid.uuid4()),
        preferences      = UserPreferences(location="Bangalore"),
        candidate_profile = candidate_profile,
        confirmed_profiles = confirmed_profiles,
        raw_jobs         = [
            _make_job("Senior Data Scientist", "Accenture",
                      "Python machine learning SQL AML fraud detection",
                      days_old=2),
            _make_job("ML Engineer", "Flipkart",
                      "Python pytorch machine learning production deployment",
                      days_old=7, matched_profile="ML Engineer"),
            _make_job("Data Analyst", "Infosys",
                      "SQL excel reporting dashboards",
                      days_old=20),
        ],
    )


# -- Deduplication tests -------------------------------------------------------

class TestDeduplication:
    def test_removes_exact_duplicates(self):
        jobs = [
            _make_job("Senior DS", "Accenture", job_id="jsearch_1"),
            _make_job("Senior DS", "Accenture", job_id="remoteok_1"),
        ]
        result = deduplicate(jobs)
        assert len(result) == 1

    def test_normalises_sr_vs_senior(self):
        jobs = [
            _make_job("Sr Data Scientist", "TCS", job_id="j1"),
            _make_job("Senior Data Scientist", "TCS", job_id="j2"),
        ]
        result = deduplicate(jobs)
        assert len(result) == 1

    def test_keeps_different_companies(self):
        jobs = [
            _make_job("Senior DS", "Accenture"),
            _make_job("Senior DS", "Infosys"),
        ]
        result = deduplicate(jobs)
        assert len(result) == 2

    def test_merges_matched_profile_tags(self):
        jobs = [
            _make_job("Senior DS", "Accenture",
                      matched_profile="Senior Data Scientist",
                      job_id="j1"),
            _make_job("Senior DS", "Accenture",
                      matched_profile="ML Engineer",
                      job_id="j2"),
        ]
        result = deduplicate(jobs)
        assert len(result) == 1
        assert "Senior Data Scientist" in result[0].matched_profile
        assert "ML Engineer" in result[0].matched_profile

    def test_keeps_longer_jd_on_dedup(self):
        jobs = [
            _make_job("Senior DS", "Accenture", jd_text="short", job_id="j1"),
            _make_job("Senior DS", "Accenture",
                      jd_text="much longer description with more details",
                      job_id="j2"),
        ]
        result = deduplicate(jobs)
        assert "much longer" in result[0].jd_text


# -- Semantic scoring tests ----------------------------------------------------

class TestSemanticScore:
    def test_high_score_on_matching_jd(self, candidate_profile):
        keywords = _build_candidate_keywords(candidate_profile)
        job = _make_job(
            jd_text="Python machine learning SQL AML fraud airflow docker fastapi"
        )
        score = _semantic_score(job, keywords)
        assert score > 0.5

    def test_low_score_on_unrelated_jd(self, candidate_profile):
        keywords = _build_candidate_keywords(candidate_profile)
        job = _make_job(jd_text="Excel PowerPoint accounting finance marketing")
        score = _semantic_score(job, keywords)
        assert score < 0.4

    def test_neutral_on_empty_jd(self, candidate_profile):
        keywords = _build_candidate_keywords(candidate_profile)
        job = _make_job(jd_text="")
        score = _semantic_score(job, keywords)
        assert score == 0.3

    def test_score_between_0_and_1(self, candidate_profile):
        keywords = _build_candidate_keywords(candidate_profile)
        job = _make_job(jd_text="Python SQL machine learning deep learning")
        score = _semantic_score(job, keywords)
        assert 0.0 <= score <= 1.0


# -- Seniority scoring tests ---------------------------------------------------

class TestSeniorityScore:
    def test_perfect_match_senior(self, candidate_profile, confirmed_profiles):
        job = _make_job(title="Senior Data Scientist")
        score = _seniority_score(job, candidate_profile, confirmed_profiles)
        assert score == 1.0

    def test_one_level_off(self, candidate_profile, confirmed_profiles):
        job = _make_job(title="Lead Data Scientist")
        score = _seniority_score(job, candidate_profile, confirmed_profiles)
        assert score == 0.7

    def test_junior_role_low_score(self, candidate_profile, confirmed_profiles):
        job = _make_job(title="Junior Data Scientist")
        score = _seniority_score(job, candidate_profile, confirmed_profiles)
        assert score < 0.5

    def test_mid_level_default(self, candidate_profile, confirmed_profiles):
        job = _make_job(title="Data Scientist")   # no seniority prefix
        score = _seniority_score(job, candidate_profile, confirmed_profiles)
        assert 0.0 <= score <= 1.0


# -- Recency scoring tests -----------------------------------------------------

class TestRecencyScore:
    def test_fresh_job_scores_high(self):
        job = _make_job(days_old=1)
        assert _recency_score(job) > 0.9

    def test_old_job_scores_low(self):
        job = _make_job(days_old=35)
        assert _recency_score(job) == 0.0

    def test_two_week_old_job_mid_score(self):
        job = _make_job(days_old=15)
        score = _recency_score(job)
        assert 0.3 < score < 0.7

    def test_no_date_returns_neutral(self):
        job = _make_job()
        job.posted_date = None
        assert _recency_score(job) == 0.5


# -- Recommended action tests --------------------------------------------------

class TestRecommendedAction:
    def test_high_score_apply_now(self):
        assert _recommended_action(0.80) == "apply_now"

    def test_mid_score_apply_with_note(self):
        assert _recommended_action(0.60) == "apply_with_note"

    def test_low_score_monitor(self):
        assert _recommended_action(0.40) == "monitor"

    def test_very_low_score_skip(self):
        assert _recommended_action(0.20) == "skip"


# -- Gap skills tests ----------------------------------------------------------

class TestGapSkills:
    def test_identifies_missing_skills(self, candidate_profile):
        keywords = _build_candidate_keywords(candidate_profile)
        job = _make_job(
            jd_text="We require kubernetes spark snowflake databricks experience"
        )
        gaps = _gap_skills(job, keywords)
        assert len(gaps) > 0
        assert any(s in gaps for s in ["kubernetes", "spark", "snowflake"])

    def test_no_gaps_when_fully_matched(self, candidate_profile):
        keywords = _build_candidate_keywords(candidate_profile)
        job = _make_job(jd_text="Python machine learning SQL experience required")
        gaps = _gap_skills(job, keywords)
        assert len(gaps) == 0

    def test_caps_at_five_gaps(self, candidate_profile):
        keywords = _build_candidate_keywords(candidate_profile)
        job = _make_job(
            jd_text="kubernetes spark kafka snowflake databricks "
                    "terraform flink elasticsearch mongodb cassandra"
        )
        gaps = _gap_skills(job, keywords)
        assert len(gaps) <= 5


# -- Full rank_jobs tests ------------------------------------------------------

class TestRankJobs:
    def test_returns_ranked_job_objects(
        self, candidate_profile, confirmed_profiles
    ):
        jobs = [_make_job() for _ in range(5)]
        result = rank_jobs(jobs, candidate_profile, confirmed_profiles)
        assert all(isinstance(j, RankedJob) for j in result)

    def test_sorted_by_fit_score_descending(
        self, candidate_profile, confirmed_profiles
    ):
        jobs = [
            _make_job("Senior Data Scientist", "A",
                      "Python ML SQL AML fraud airflow docker"),
            _make_job("Excel Analyst", "B", "excel powerpoint reporting"),
        ]
        result = rank_jobs(jobs, candidate_profile, confirmed_profiles)
        scores = [j.fit_score for j in result]
        assert scores == sorted(scores, reverse=True)

    def test_capped_at_max_results(
        self, candidate_profile, confirmed_profiles
    ):
        jobs = [_make_job() for _ in range(50)]
        result = rank_jobs(jobs, candidate_profile, confirmed_profiles)
        assert len(result) <= 25

    def test_fit_score_between_0_and_1(
        self, candidate_profile, confirmed_profiles
    ):
        jobs = [_make_job()]
        result = rank_jobs(jobs, candidate_profile, confirmed_profiles)
        for job in result:
            assert 0.0 <= job.fit_score <= 1.0

    def test_matched_via_populated(
        self, candidate_profile, confirmed_profiles
    ):
        job = _make_job(matched_profile="Senior Data Scientist | ML Engineer")
        result = rank_jobs([job], candidate_profile, confirmed_profiles)
        assert len(result[0].matched_via) >= 1


# -- Full agent orchestration tests --------------------------------------------

class TestRunRankerAgent:
    def test_successful_ranking(self, session):
        result = run_ranker_agent(session)
        assert result.results_ready is True
        assert len(result.ranked_jobs) > 0
        assert result.error is None
        assert result.current_agent == "ranker"

    def test_no_raw_jobs_returns_empty(self, session):
        session.raw_jobs = []
        result = run_ranker_agent(session)
        assert result.ranked_jobs == []
        assert result.results_ready is True

    def test_missing_profile_sets_error(self, session):
        session.candidate_profile = None
        result = run_ranker_agent(session)
        assert result.error is not None
        assert result.ranked_jobs == []

    def test_results_sorted_highest_first(self, session):
        result = run_ranker_agent(session)
        scores = [j.fit_score for j in result.ranked_jobs]
        assert scores == sorted(scores, reverse=True)

    def test_session_id_preserved(self, session):
        original_id = session.session_id
        result = run_ranker_agent(session)
        assert result.session_id == original_id
