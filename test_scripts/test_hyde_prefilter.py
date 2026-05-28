"""
test_scripts/test_hyde_prefilter.py

Live test for Agent 5b -- HyDE Prefilter.
Runs a real job search to populate raw_jobs, then pipes the output through
the HyDE prefilter. Shows the S1/S2 partition, per-job scores, and the
two hypothetical JDs that were generated.

Usage:
    python test_scripts/test_hyde_prefilter.py

Requires: ANTHROPIC_API_KEY + VOYAGE_API_KEY + at least one job source key.
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
from agents.ranker.ranker_agent import deduplicate
from agents.hyde.hyde_agent import run_hyde_prefilter


async def main():
    print("\n" + "=" * 60)
    print("  Agent 5b -- HyDE Prefilter Live Test")
    print("=" * 60)

    # -- Candidate profile (AmEx DS/ML) ----------------------------------------
    profile = CandidateProfile(
        current_title              = "Senior Analyst - Data Science",
        years_experience_full_time = 1.7,
        years_experience_other     = 0.5,
        seniority_level            = "mid",
        skills = SkillSet(
            technical = [
                "Python", "Machine Learning", "SQL", "NLP",
                "XGBoost", "Deep Learning", "Statistical Modelling",
            ],
            tools = [
                "Airflow", "Docker", "FastAPI", "scikit-learn",
                "TensorFlow", "Spark", "Git",
            ],
            soft = ["Leadership", "Communication"],
        ),
        domain_expertise = [
            "AML (Anti-Money Laundering)", "Financial Services",
            "Fraud Detection", "Credit Risk",
        ],
        career_trajectory = "ascending",
        work_experience = [
            WorkExperience(
                title           = "Senior Analyst - Data Science",
                company         = "American Express",
                duration_months = 22,
                role_type       = "full_time",
                impact_signals  = [
                    "Built real-time AML transaction monitoring models",
                    "Reduced false positive rate by 34% using XGBoost ensemble",
                ],
            ),
        ],
        ats_summary = (
            "Mid-level Data Scientist at American Express in AML and fraud detection. "
            "Strong Python/ML background with production experience."
        ),
    )

    confirmed = SuggestedProfile(
        title            = "Senior Data Scientist",
        seniority_target = "senior",
        confidence       = "high",
        match_reason     = "Strong ML background, step up from current role.",
        is_stretch       = True,
        source           = "system",
    )

    session = SessionState(
        session_id        = str(uuid.uuid4()),
        candidate_profile = profile,
        preferences = UserPreferences(
            location             = "Bangalore",
            seniority_preference = "step_up",
        ),
        confirmed_profiles = [confirmed],
    )

    # -- Step 1: Job search ----------------------------------------------------
    print("\nStep 1: Running job search (real API calls)...")
    session = await run_job_search_agent(session)
    print(f"  Raw jobs fetched: {len(session.raw_jobs)}")

    if not session.raw_jobs:
        print("  ERROR: No jobs returned. Check source config and API keys.")
        return

    # -- Step 2: Dedup ---------------------------------------------------------
    before_dedup = len(session.raw_jobs)
    session.raw_jobs = deduplicate(session.raw_jobs)
    print(f"  After dedup     : {len(session.raw_jobs)} (removed {before_dedup - len(session.raw_jobs)} duplicates)")

    # -- Step 3: HyDE prefilter ------------------------------------------------
    print("\nStep 2: Running HyDE prefilter (JD generation + Voyage embedding)...")
    session = await run_hyde_prefilter(session)

    # -- Results ---------------------------------------------------------------
    s1_jobs = [j for j in session.raw_jobs if j.hyde_section == "S1"]
    s2_jobs = [j for j in session.raw_jobs if j.hyde_section == "S2"]

    print(f"\n{'=' * 60}")
    print(f"  HyDE Partition Results")
    print(f"{'=' * 60}")
    print(f"  Section 1 (domain roles)      : {len(s1_jobs)} jobs")
    print(f"  Section 2 (broader opps)      : {len(s2_jobs)} jobs")
    print(f"  Total passed to ranker        : {len(session.raw_jobs)} jobs")

    metrics = session.agent_metrics.get("hyde_prefilter", {})
    print(f"\n  Floor used    : {metrics.get('floor_used', 'n/a')}")
    print(f"  Fallback used : {metrics.get('fallback_activated', False)}")
    print(f"  Emb latency   : {metrics.get('emb_latency_secs', 'n/a')}s")
    print(f"  Total latency : {metrics.get('latency_secs', 'n/a')}s")

    # Hypothetical JDs
    if session.hypo_jd1:
        print(f"\n  JD1 (domain-anchored) preview:")
        print(f"  {session.hypo_jd1[:300].strip()}...")
    if session.hypo_jd2:
        print(f"\n  JD2 (transferable-skills) preview:")
        print(f"  {session.hypo_jd2[:300].strip()}...")

    # Top jobs per section
    if s1_jobs:
        print(f"\n{'=' * 60}")
        print(f"  Section 1 -- Top 5 (by JD1 RRF score)")
        print(f"{'=' * 60}")
        for job in sorted(s1_jobs, key=lambda j: j.rrf_jd1 or 0, reverse=True)[:5]:
            print(f"  [{job.rrf_jd1:.4f} rrf1 | {job.jd1_emb_score:.3f} cos] "
                  f"{job.title} @ {job.company} ({job.location})")

    if s2_jobs:
        print(f"\n{'=' * 60}")
        print(f"  Section 2 -- Top 5 (by JD2 RRF score)")
        print(f"{'=' * 60}")
        for job in sorted(s2_jobs, key=lambda j: j.rrf_jd2 or 0, reverse=True)[:5]:
            print(f"  [{job.rrf_jd2:.4f} rrf2 | {job.jd2_emb_score:.3f} cos] "
                  f"{job.title} @ {job.company} ({job.location})")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
