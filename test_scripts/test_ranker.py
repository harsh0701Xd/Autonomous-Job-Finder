"""
test_scripts/test_ranker.py

Live test for Agent 6 -- Ranker (full chain).
Runs the complete pipeline up to and including the ranker:
  Job Search → URL Validator → Dedup → HyDE Prefilter → Ranker

Prints the top ranked jobs with fit scores, sub-scores, skill gaps,
and the per-source discard rate log.

Usage:
    python test_scripts/test_ranker.py

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
from agents.ranker.ranker_agent import deduplicate, run_ranker_agent
from agents.hyde.hyde_agent import run_hyde_prefilter
from core.url_validator import validate_urls


async def main():
    print("\n" + "=" * 60)
    print("  Agent 6 -- Ranker Full-Chain Live Test")
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
        work_experience = [
            WorkExperience(
                title           = "Senior Analyst - Data Science",
                company         = "American Express",
                start_date      = "2023-07",
                end_date        = None,
                duration_months = 22,
                role_type       = "full_time",
                impact_signals  = [
                    "Built real-time AML transaction monitoring models serving 1M+ txns/day",
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
            "experience. IIT Kharagpur graduate with trajectory toward senior DS/ML roles."
        ),
        pivot_signals  = [],
        career_gaps    = [],
    )

    confirmed_profiles = [
        SuggestedProfile(
            title            = "Senior Data Scientist",
            seniority_target = "senior",
            confidence       = "high",
            match_reason     = "Strong ML background, step up from current role.",
            is_stretch       = True,
            source           = "system",
        ),
        SuggestedProfile(
            title            = "ML Engineer",
            seniority_target = "senior",
            confidence       = "high",
            match_reason     = "Production ML and pipeline experience.",
            is_stretch       = True,
            source           = "system",
        ),
    ]

    session = SessionState(
        session_id        = str(uuid.uuid4()),
        candidate_profile = profile,
        preferences = UserPreferences(
            location             = "Bangalore",
            seniority_preference = "step_up",
        ),
        confirmed_profiles = confirmed_profiles,
    )

    # -- Step 1: Job Search ----------------------------------------------------
    print("\nStep 1: Job search (real API calls)...")
    session = await run_job_search_agent(session)
    print(f"  Raw jobs: {len(session.raw_jobs)}")

    if not session.raw_jobs:
        print("  ERROR: No jobs. Check source config in llm_config.yaml and .env keys.")
        return

    # -- Step 2: URL Validator --------------------------------------------------
    print("\nStep 2: URL validation...")
    url_results = await validate_urls(session.raw_jobs)
    valid_ids   = {job_id for job_id, is_valid, _ in url_results if is_valid}
    before_url  = len(session.raw_jobs)
    session.raw_jobs = [j for j in session.raw_jobs if j.job_id in valid_ids]
    print(f"  Passed: {len(session.raw_jobs)} | Dropped (dead/redirected): {before_url - len(session.raw_jobs)}")

    # -- Step 3: Dedup ---------------------------------------------------------
    print("\nStep 3: Deduplication...")
    before_dedup  = len(session.raw_jobs)
    session.raw_jobs = deduplicate(session.raw_jobs)
    print(f"  Unique: {len(session.raw_jobs)} | Duplicates removed: {before_dedup - len(session.raw_jobs)}")

    # -- Step 4: HyDE Prefilter ------------------------------------------------
    print("\nStep 4: HyDE prefilter (JD generation + Voyage embedding)...")
    session = await run_hyde_prefilter(session)
    s1 = sum(1 for j in session.raw_jobs if j.hyde_section == "S1")
    s2 = sum(1 for j in session.raw_jobs if j.hyde_section == "S2")
    print(f"  S1 (domain roles): {s1} | S2 (broader opps): {s2} | Total: {len(session.raw_jobs)}")

    if not session.raw_jobs:
        print("  ERROR: All jobs dropped by HyDE. Lower hyde_min_floor or check sources.")
        return

    # -- Step 5: Ranker --------------------------------------------------------
    print(f"\nStep 5: Ranking {len(session.raw_jobs)} jobs via Claude Haiku...")
    session = await run_ranker_agent(session)
    print(f"  Ranked jobs returned: {len(session.ranked_jobs)}")

    # -- Results ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  Top {min(len(session.ranked_jobs), 20)} Ranked Jobs")
    print("=" * 60)

    for i, job in enumerate(session.ranked_jobs[:20], 1):
        section   = f"[{job.hyde_section}]" if job.hyde_section else ""
        sparse    = " [sparse JD]"  if job.sparse_jd         else ""
        overq     = " [overqualified]" if job.overqualified  else ""
        print(f"\n#{i:02d} {section} {job.fit_score:.0%} fit {sparse}{overq}")
        print(f"     {job.title} @ {job.company}")
        print(f"     Location : {job.location}")
        print(f"     Source   : {job.source} | {job.matched_profile}")
        print(f"     Scores   : exp={job.experience_score:.2f} skill={job.skill_score:.2f} "
              f"domain={job.domain_score:.2f}", end="")
        if job.education_required:
            print(f" edu={job.education_score:.2f}", end="")
        print()
        if job.skill_gaps:
            print(f"     Skill gaps: {', '.join(job.skill_gaps[:5])}")
        if job.experience_gap:
            print(f"     Exp gap   : {job.experience_gap}")
        print(f"     Apply     : {job.apply_url[:70]}...")

    # -- Score summary ---------------------------------------------------------
    scores = [j.fit_score for j in session.ranked_jobs]
    if scores:
        print(f"\n{'=' * 60}")
        print(f"  Score summary")
        print(f"{'=' * 60}")
        print(f"  Count       : {len(scores)}")
        print(f"  Mean fit    : {sum(scores)/len(scores):.2%}")
        print(f"  Max fit     : {max(scores):.2%}")
        print(f"  Min fit     : {min(scores):.2%}")
        above_70 = sum(1 for s in scores if s >= 0.70)
        above_55 = sum(1 for s in scores if s >= 0.55)
        print(f"  >= 70%      : {above_70}")
        print(f"  >= 55%      : {above_55}")

    # -- Per-source discard rates (from ranker metrics) -----------------------
    ranker_metrics = session.agent_metrics.get("ranker", {})
    source_rates   = ranker_metrics.get("source_discard_rates", {})
    if source_rates:
        print(f"\n{'=' * 60}")
        print(f"  Per-source discard rates")
        print(f"{'=' * 60}")
        print(f"  {'Source':<20} {'Retrieved':>10} {'Passed':>8} {'Dropped':>8} {'Pass%':>7} {'AvgFit':>8}")
        print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*8} {'-'*7} {'-'*8}")
        for src, stats in sorted(source_rates.items()):
            print(f"  {src:<20} {stats.get('retrieved',0):>10} {stats.get('passed',0):>8} "
                  f"{stats.get('dropped',0):>8} {stats.get('pass_rate',0):>6.0%}  "
                  f"{stats.get('avg_fit_score',0):>8.2%}")

    # -- Agent timing ----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Agent latencies")
    print(f"{'=' * 60}")
    for agent_name in ["hyde_prefilter", "ranker"]:
        m = session.agent_metrics.get(agent_name, {})
        print(f"  {agent_name:<20}: {m.get('latency_secs', 'n/a')}s")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
