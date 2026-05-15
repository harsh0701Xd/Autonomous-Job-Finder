"""
scripts/test_ranker_agent_live.py

Live integration test for Agent 4 -- Ranker + Dedup.
Runs Agent 3 (real API calls) then pipes output through Agent 4.
Shows ranked job results with fit scores.

Usage:
    python scripts/test_ranker_agent_live.py
"""

import asyncio
import sys
import uuid
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from core.state.session_state import (
    CandidateProfile, SessionState, SkillSet,
    SuggestedProfile, UserPreferences, WorkExperience,
)
from agents.job_search.job_search_agent import run_job_search_agent
from agents.ranker.ranker_agent import run_ranker_agent


async def main():
    print("\n" + "="*60)
    print("  Agent 4 -- Ranker Live Integration Test")
    print("="*60)

    session = SessionState(
        session_id = str(uuid.uuid4()),
        preferences = UserPreferences(
            location             = "Bangalore",
            work_type            = "hybrid",
            seniority_preference = "step_up",
        ),
        candidate_profile = CandidateProfile(
            current_title    = "Senior Data Scientist",
            seniority_level  = "senior",
            years_experience = 2.0,
            skills = SkillSet(
                technical = ["Python", "Machine Learning", "SQL",
                             "NLP", "XGBoost", "Deep Learning"],
                tools     = ["Airflow", "Docker", "FastAPI",
                             "scikit-learn", "TensorFlow"],
                soft      = ["Leadership", "Communication"],
            ),
            domain_expertise = ["AML", "Financial Services",
                                "Fraud Detection", "Credit Risk"],
            career_trajectory = "ascending",
            work_experience   = [
                WorkExperience(
                    title            = "Senior Analyst - Data Science",
                    company          = "American Express",
                    duration_months  = 15,
                    responsibilities = [],
                    impact_signals   = ["Reduced fraud by 23%"],
                )
            ],
            raw_text = "",
        ),
        confirmed_profiles = [
            SuggestedProfile(
                title            = "Senior Data Scientist",
                seniority_target = "senior",
                confidence       = "high",
                match_reason     = "Strong ML background.",
                is_stretch       = False,
                source           = "system",
            ),
            SuggestedProfile(
                title            = "ML Engineer",
                seniority_target = "senior",
                confidence       = "high",
                match_reason     = "Production ML experience.",
                is_stretch       = False,
                source           = "system",
            ),
        ],
    )

    # Step 1: Run Agent 3
    print("\nStep 1: Running job search (Agent 3)...")
    session = await run_job_search_agent(session)
    print(f"  Raw jobs collected: {len(session.raw_jobs)}")

    if not session.raw_jobs:
        print("  ERROR: No jobs returned. Check your JSEARCH_API_KEY.")
        return

    # Step 2: Run Agent 4
    print("\nStep 2: Ranking and deduplicating (Agent 4)...")
    session = run_ranker_agent(session)
    print(f"  Ranked jobs: {len(session.ranked_jobs)}")

    # Display results
    print("\n" + "="*60)
    print(f"  Top {len(session.ranked_jobs)} Jobs -- Ranked by Fit Score")
    print("="*60)

    for i, job in enumerate(session.ranked_jobs, 1):
        action_icons = {
            "apply_now":       " APPLY NOW",
            "apply_with_note": "~ Apply with note",
            "monitor":         " Monitor",
            "skip":            " Skip",
        }
        action = action_icons.get(job.recommended_action, job.recommended_action)

        print(f"\n#{i}  [{job.fit_score:.0%} fit]  {action}")
        print(f"     {job.title} @ {job.company}")
        print(f"     {job.location} | {job.work_type} | {job.source}")
        print(f"     Matched via: {' + '.join(job.matched_via)}")
        if job.gap_skills:
            print(f"     Skill gaps : {', '.join(job.gap_skills)}")
        print(f"     Apply      : {job.apply_url[:65]}...")

    # Score distribution
    print("\n" + "="*60)
    print("  Score distribution:")
    apply_now  = sum(1 for j in session.ranked_jobs
                     if j.recommended_action == "apply_now")
    with_note  = sum(1 for j in session.ranked_jobs
                     if j.recommended_action == "apply_with_note")
    monitor    = sum(1 for j in session.ranked_jobs
                     if j.recommended_action == "monitor")
    skip_count = sum(1 for j in session.ranked_jobs
                     if j.recommended_action == "skip")

    print(f"  Apply now       : {apply_now}")
    print(f"  Apply with note : {with_note}")
    print(f"  Monitor         : {monitor}")
    print(f"  Skip            : {skip_count}")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
