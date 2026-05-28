"""
test_scripts/test_job_search.py

Live test for Agent 3 -- Job Search Agent.
Makes real API calls across all enabled sources for a set of confirmed profiles.
Prints raw job counts broken down by source and by matched profile.

Usage:
    python test_scripts/test_job_search.py

Requires API keys set in .env (e.g. JSEARCH_API_KEY, TECHMAP_API_KEY, etc.)
Enable/disable sources and set page counts in llm_config.yaml [job_search.sources].
"""

import asyncio
import sys
import uuid
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from core.state.session_state import (
    SessionState, SuggestedProfile, UserPreferences,
)
from agents.job_search.job_search_agent import run_job_search_agent


async def main():
    print("\n" + "=" * 60)
    print("  Agent 3 -- Job Search Live Test")
    print("=" * 60)

    session = SessionState(
        session_id = str(uuid.uuid4()),
        preferences = UserPreferences(
            location             = "Bangalore",
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
                is_stretch       = True,
                source           = "system",
            ),
        ],
    )

    print(f"\nSearching for {len(session.confirmed_profiles)} profiles:")
    for p in session.confirmed_profiles:
        stretch = " [stretch]" if p.is_stretch else ""
        print(f"  -> {p.title}{stretch}")
    print(f"\nLocation    : {session.preferences.location}")
    print(f"Seniority   : {session.preferences.seniority_preference}")
    print("\nRunning job search (real API calls)...\n")

    result = await run_job_search_agent(session)

    # -- Results ---------------------------------------------------------------
    print("=" * 60)
    print(f"  {len(result.raw_jobs)} raw jobs returned")
    print("=" * 60)

    if result.error:
        print(f"\nERROR: {result.error}")
        return

    if not result.raw_jobs:
        print("\nNo jobs found. Check that at least one source is enabled in llm_config.yaml")
        print("and that the corresponding API key is set in .env")
        return

    # By source
    by_source: dict[str, list] = {}
    for job in result.raw_jobs:
        by_source.setdefault(job.source, []).append(job)

    for source, jobs in sorted(by_source.items()):
        print(f"\n[{source.upper()}]  {len(jobs)} jobs")
        print("-" * 40)
        for job in jobs[:3]:
            jd_words = len((job.jd_text or "").split())
            print(f"  {job.title} @ {job.company}")
            print(f"  Location : {job.location}")
            print(f"  Profile  : {job.matched_profile}")
            print(f"  JD words : {jd_words}")
            print(f"  URL      : {job.apply_url[:70]}...")
            print()

    # By profile
    print("=" * 60)
    print("  Jobs per confirmed profile:")
    for p in session.confirmed_profiles:
        count = sum(1 for j in result.raw_jobs if j.matched_profile == p.title)
        print(f"  {p.title}: {count} jobs")

    print(f"\n  Total -> HyDE / Ranker: {len(result.raw_jobs)} jobs")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
