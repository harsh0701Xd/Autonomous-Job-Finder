"""
scripts/test_signals_agent_live.py

Live integration test for Agent 5 -- Hiring Signals Agent.
Runs Agents 3 + 4 + 5 in sequence using real API calls.

Usage:
    python scripts/test_signals_agent_live.py
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
from agents.signals.signals_agent import run_signals_agent


async def main():
    print("\n" + "="*60)
    print("  Agent 5 -- Hiring Signals Live Integration Test")
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
                tools     = ["Airflow", "Docker", "FastAPI", "scikit-learn"],
                soft      = ["Leadership"],
            ),
            domain_expertise = ["AML", "Financial Services", "Fraud Detection"],
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

    # Step 1: Job search
    print("\nStep 1: Job search (Agent 3)...")
    session = await run_job_search_agent(session)
    print(f"  Raw jobs: {len(session.raw_jobs)}")

    # Step 2: Rank
    print("Step 2: Ranking (Agent 4)...")
    session = run_ranker_agent(session)
    print(f"  Ranked jobs: {len(session.ranked_jobs)}")

    # Show top 3 companies we'll scan
    companies = list(dict.fromkeys(j.company for j in session.ranked_jobs))[:5]
    print(f"  Top companies to scan: {', '.join(companies)}")

    # Step 3: Signals
    print("\nStep 3: Fetching hiring signals (Agent 5)...")
    print("  (This may take 10-20 seconds -- fetching from NewsAPI + RSS feeds)")
    session = await run_signals_agent(session)

    # Display results
    print("\n" + "="*60)
    print(f"  Hiring Signals -- {len(session.hiring_signals)} found")
    print("="*60)

    if session.hiring_signals:
        for sig in session.hiring_signals:
            strength_icon = {"high": "!!!", "medium": "!!", "low": "!"}.get(
                sig.signal_strength, "!"
            )
            direction = "+" if sig.is_positive else "-"
            print(f"\n  [{direction}] [{strength_icon}] {sig.company}")
            print(f"       Type    : {sig.signal_type}")
            print(f"       Summary : {sig.summary}")
            print(f"       Jobs    : {sig.jobs_you_matched} matched job(s)")
            if sig.relevant_to_profiles:
                print(f"       Profiles: {', '.join(sig.relevant_to_profiles)}")
            print(f"       Source  : {sig.source_name}")
            if sig.source_url:
                print(f"       URL     : {sig.source_url}")
    else:
        print("\n  No signals found.")
        print("  Possible reasons:")
        print("  - NEWSAPI_API_KEY not set (only RSS feeds ran)")
        print("  - Companies in results have no recent news")
        print("  - RSS feeds had no matching articles")

    # Watch list
    print(f"\n{'='*60}")
    print(f"  Watch List -- {len(session.watch_list)} proactive signals")
    print("="*60)

    if session.watch_list:
        for sig in session.watch_list:
            print(f"\n  [watch] {sig.company}")
            print(f"          {sig.summary}")
            if sig.source_url:
                print(f"          {sig.source_url}")
    else:
        print("\n  No watch list entries found.")

    # Summary
    print(f"\n{'='*60}")
    print("  Pipeline summary:")
    print(f"  Raw jobs collected  : {len(session.raw_jobs)}")
    print(f"  Ranked jobs         : {len(session.ranked_jobs)}")
    print(f"  Hiring signals      : {len(session.hiring_signals)}")
    print(f"  Watch list entries  : {len(session.watch_list)}")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
