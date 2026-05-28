"""
test_scripts/test_profile_recommender.py

Live test for Agent 2b -- Profile Recommender.
Uses a hardcoded CandidateProfile (AmEx DS/ML profile) and calls
run_profile_recommender() to generate suggested job profiles.
Prints every suggested profile with its confidence and match reason.

Usage:
    python test_scripts/test_profile_recommender.py
"""

import sys
import uuid
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from core.state.session_state import (
    CandidateProfile, SessionState, SkillSet,
    SuggestedProfile, UserPreferences, WorkExperience, Education,
)
from agents.recommender.profile_recommender import run_profile_recommender


def main():
    print("\n" + "=" * 60)
    print("  Agent 2b -- Profile Recommender Live Test")
    print("=" * 60)

    # -- Build a hardcoded candidate profile (AmEx DS/ML background) ----------
    profile = CandidateProfile(
        current_title              = "Senior Analyst - Data Science",
        years_experience_full_time = 1.7,
        years_experience_other     = 0.5,
        seniority_level            = "mid",
        skills = SkillSet(
            technical = [
                "Python", "Machine Learning", "SQL", "NLP",
                "XGBoost", "Deep Learning", "Statistical Modelling",
                "A/B Testing", "Feature Engineering",
            ],
            tools = [
                "Airflow", "Docker", "FastAPI", "scikit-learn",
                "TensorFlow", "Spark", "Tableau", "Git",
            ],
            soft = ["Leadership", "Communication", "Cross-functional collaboration"],
        ),
        domain_expertise = [
            "AML (Anti-Money Laundering)", "Financial Services",
            "Fraud Detection", "Credit Risk", "Transaction Monitoring",
        ],
        career_trajectory = "ascending",
        education = [
            Education(
                degree      = "B.Tech",
                field       = "Computer Science",
                institution = "IIT Kharagpur",
                year        = 2023,
            ),
        ],
        work_experience = [
            WorkExperience(
                title           = "Senior Analyst - Data Science",
                company         = "American Express",
                start_date      = "2023-07",
                end_date        = None,
                duration_months = 22,
                role_type       = "full_time",
                impact_signals  = [
                    "Built real-time AML transaction monitoring models serving 1M+ transactions/day",
                    "Reduced false positive rate by 34% using XGBoost ensemble",
                    "Led cross-functional team of 4 across data engineering and compliance",
                ],
            ),
            WorkExperience(
                title           = "Data Science Intern",
                company         = "American Express",
                start_date      = "2023-01",
                end_date        = "2023-06",
                duration_months = 6,
                role_type       = "internship",
                impact_signals  = [
                    "Developed credit risk scoring model for SME segment",
                ],
            ),
        ],
        ats_summary = (
            "Mid-level Data Scientist with 1.7 years full-time experience at American Express "
            "in AML and fraud detection. Strong Python/ML skills with production deployment "
            "experience. IIT Kharagpur graduate with a trajectory toward senior DS or ML roles."
        ),
        pivot_signals  = [],
        career_gaps    = [],
    )

    session = SessionState(
        session_id       = str(uuid.uuid4()),
        candidate_profile = profile,
        preferences = UserPreferences(
            location             = "Bangalore",
            seniority_preference = "step_up",
        ),
    )

    print(f"\nCandidate   : {profile.current_title}")
    print(f"Experience  : {profile.years_experience_full_time} yrs full-time")
    print(f"Seniority   : {profile.seniority_level}")
    print(f"Preference  : {session.preferences.seniority_preference}")
    print(f"Location    : {session.preferences.location}")
    print("\nCalling Claude (profile recommender)...\n")

    # -- Run recommender -------------------------------------------------------
    result = run_profile_recommender(session)

    # -- Display results -------------------------------------------------------
    if result.error:
        print(f"ERROR: {result.error}")
        return

    if not result.suggested_profiles:
        print("No profiles returned (check logs above).")
        return

    print("=" * 60)
    print(f"  Suggested Profiles ({len(result.suggested_profiles)} returned)")
    print("=" * 60)

    for i, p in enumerate(result.suggested_profiles, 1):
        stretch = " [STRETCH]" if p.is_stretch else ""
        print(f"\n#{i}  {p.title} ({p.seniority_target}){stretch}")
        print(f"     Confidence   : {p.confidence}")
        print(f"     Match reason : {p.match_reason}")
        if p.search_variants:
            print(f"     Search alts  : {', '.join(p.search_variants)}")

    metrics = result.agent_metrics.get("profile_recommender", {})
    tokens_in  = metrics.get("input_tokens",  0)
    tokens_out = metrics.get("output_tokens", 0)
    latency    = metrics.get("latency_secs",  0)
    llm_calls  = metrics.get("llm_calls",     0)

    print(f"\nTokens : {tokens_in} in / {tokens_out} out | Latency: {latency}s | LLM calls: {llm_calls}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
