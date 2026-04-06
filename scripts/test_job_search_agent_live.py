"""
scripts/test_job_search_agent_live.py

Live integration test for Agent 3 — Job Search Agent.
Makes real API calls to JSearch and RemoteOK.
Run this to verify Agent 3 is returning real job results
before wiring it into the full pipeline.

Usage:
    python scripts/test_job_search_agent_live.py
"""

import asyncio
import os
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from core.state.session_state import (
    SessionState, SuggestedProfile, UserPreferences
)
from agents.job_search.job_search_agent import run_job_search_agent
import uuid


async def main():
    print("\n" + "="*60)
    print("  Job Search Agent — Live Integration Test")
    print("="*60)

    # Build a test session matching your real confirmed profiles
    session = SessionState(
        session_id = str(uuid.uuid4()),
        preferences = UserPreferences(
            location             = "Bangalore",
            work_type            = "hybrid",
            seniority_preference = "step_up",
        ),
        confirmed_profiles = [
            SuggestedProfile(
                title            = "Senior Data Scientist",
                seniority_target = "senior",
                confidence       = "high",
                match_reason     = "Test profile.",
                is_stretch       = False,
                source           = "system",
            ),
            SuggestedProfile(
                title            = "ML Engineer",
                seniority_target = "senior",
                confidence       = "high",
                match_reason     = "Test profile.",
                is_stretch       = False,
                source           = "system",
            ),
        ],
    )

    print(f"\nSearching for {len(session.confirmed_profiles)} profiles:")
    for p in session.confirmed_profiles:
        print(f"  → {p.title}")
    print(f"\nLocation : {session.preferences.location}")
    print(f"Work type: {session.preferences.work_type}")
    print("\nRunning job search...\n")

    result = await run_job_search_agent(session)

    print("="*60)
    print(f"  Results: {len(result.raw_jobs)} jobs found")
    print("="*60)

    if result.error:
        print(f"\nERROR: {result.error}")
        return

    # Group by source
    by_source = {}
    for job in result.raw_jobs:
        by_source.setdefault(job.source, []).append(job)

    for source, jobs in by_source.items():
        print(f"\n[{source.upper()}] — {len(jobs)} jobs")
        print("-" * 40)
        for job in jobs[:3]:   # show first 3 per source
            print(f"  Title   : {job.title}")
            print(f"  Company : {job.company}")
            print(f"  Location: {job.location}")
            print(f"  Profile : {job.matched_profile}")
            print(f"  Apply   : {job.apply_url[:60]}...")
            if job.salary_min:
                print(f"  Salary  : {job.salary_min:,} – {job.salary_max:,}")
            print()

    # Summary by matched profile
    print("="*60)
    print("  By confirmed profile:")
    for profile in session.confirmed_profiles:
        count = sum(1 for j in result.raw_jobs
                    if j.matched_profile == profile.title)
        print(f"  {profile.title}: {count} jobs")

    print(f"\n  Total raw jobs → Agent 4 (ranker): {len(result.raw_jobs)}")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())