"""
agents/ranker/ranker_agent.py

Agent 6 -- Ranker
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone

import anthropic
from langsmith import traceable
from tenacity import retry, stop_after_attempt, wait_exponential

from core.prompts.ranker_prompts import JOB_SCORING_PROMPT
from core.config.config_loader import cfg
from core.state.session_state import (
    CandidateProfile,
    RankedJob,
    RawJob,
    SessionState,
    SuggestedProfile,
)

logger = logging.getLogger(__name__)

_CFG               = cfg.ranker
_SCORING_SEMAPHORE = asyncio.Semaphore(_CFG.semaphore_size)


# -- Deduplication -------------------------------------------------------------

def _fingerprint(job: RawJob) -> str:
    company = re.sub(r"\s+", " ", (job.company or "").lower().strip())
    title   = re.sub(r"\s+", " ", (job.title   or "").lower().strip())
    title   = re.sub(r"^(sr\.?|senior|junior|jr\.?|lead|principal|staff)\s+", "", title)
    return f"{company}::{title}"


def deduplicate(jobs: list[RawJob]) -> list[RawJob]:
    seen:        dict[str, RawJob]   = {}
    matched_via: dict[str, set[str]] = {}

    for job in jobs:
        fp = _fingerprint(job)
        if fp not in seen:
            seen[fp]        = job
            matched_via[fp] = {job.matched_profile}
        else:
            if len(job.jd_text) > len(seen[fp].jd_text):
                seen[fp] = job
            matched_via[fp].add(job.matched_profile)

    result = []
    for fp, job in seen.items():
        job.matched_profile = " | ".join(sorted(matched_via[fp]))
        result.append(job)

    logger.info(f"[ranker] Dedup: {len(jobs)} -> {len(result)} unique jobs")
    return result


# -- Profile-based caps --------------------------------------------------------

def apply_profile_caps(
    jobs:               list[RawJob],
    confirmed_profiles: list[SuggestedProfile],
) -> list[RawJob]:
    """
    Cap the number of jobs per confirmed profile, applied independently within
    each HyDE section (S1 and S2) so caps are not shared across sections.

    Section 1 jobs are processed first, then Section 2 — preserving the
    section ordering that the ranker and results UI rely on.
    """
    def _cap_bucket(bucket: list[RawJob]) -> list[RawJob]:
        """Apply profile caps to a single HyDE section bucket."""
        profile_rank: dict[str, int] = {
            p.title: i for i, p in enumerate(confirmed_profiles)
        }
        ranked_buckets: dict[int, list[RawJob]] = {
            i: [] for i in range(len(confirmed_profiles))
        }
        unmatched: list[RawJob] = []

        for job in bucket:
            primary = job.matched_profile.split("|")[0].strip()
            rank    = profile_rank.get(primary)
            if rank is not None and rank in ranked_buckets:
                ranked_buckets[rank].append(job)
            else:
                unmatched.append(job)

        def _recency_key(job: RawJob) -> datetime:
            if job.posted_date:
                return (
                    job.posted_date.replace(tzinfo=timezone.utc)
                    if job.posted_date.tzinfo is None
                    else job.posted_date
                )
            return datetime.min.replace(tzinfo=timezone.utc)

        capped: list[RawJob] = []
        for rank, rjobs in ranked_buckets.items():
            cap     = _CFG.profile_caps.as_dict().get(rank, 10)
            kept    = sorted(rjobs, key=_recency_key, reverse=True)[:cap]
            dropped = len(rjobs) - len(kept)
            if rjobs:
                logger.info(
                    f"[ranker:cap] Profile rank {rank + 1} "
                    f"('{confirmed_profiles[rank].title}'): "
                    f"{len(rjobs)} jobs -> cap {cap} -> kept {len(kept)}"
                    + (f" (dropped {dropped} older)" if dropped else "")
                )
            capped.extend(kept)

        if unmatched:
            last_cap = _CFG.profile_caps.as_dict().get(len(confirmed_profiles) - 1, 10)
            capped.extend(unmatched[:last_cap])

        return capped

    # Split by HyDE section, cap each independently, then reassemble in order
    s1 = [j for j in jobs if j.hyde_section == "S1"]
    s2 = [j for j in jobs if j.hyde_section == "S2"]
    # Jobs without a section (e.g. HyDE was skipped via fail-open) go to s1
    s_other = [j for j in jobs if j.hyde_section not in ("S1", "S2")]

    s1_capped    = _cap_bucket(s1 + s_other)
    s2_capped    = _cap_bucket(s2)
    result       = s1_capped + s2_capped

    logger.info(
        f"[ranker:cap] Total after capping: {len(result)} jobs "
        f"(S1={len(s1_capped)}, S2={len(s2_capped)})"
    )
    return result


# -- Recency scoring -----------------------------------------------------------

def _recency_score(job: RawJob) -> float:
    if not job.posted_date:
        return 0.5
    now    = datetime.now(timezone.utc)
    posted = job.posted_date
    if posted.tzinfo is None:
        posted = posted.replace(tzinfo=timezone.utc)
    age_days = (now - posted).days
    if age_days <= 0:
        return 1.0
    if age_days >= _CFG.recency_decay_days:
        return 0.0
    return round(1.0 - (age_days / _CFG.recency_decay_days), 3)


# -- Candidate profile formatter -----------------------------------------------

def _format_candidate_for_prompt(profile: CandidateProfile) -> dict[str, str]:
    education_lines = []
    for edu in profile.education:
        line = edu.degree
        if edu.field:       line += f" in {edu.field}"
        if edu.institution: line += f", {edu.institution}"
        if edu.year:        line += f" ({edu.year})"
        education_lines.append(line)

    ats = (profile.ats_summary or "")[:800] or (profile.raw_text[:800] if profile.raw_text else "Not provided")

    return {
        "current_title":    profile.current_title or "Not specified",
        "full_time_years":  str(profile.years_experience_full_time),
        "other_years":      str(profile.years_experience_other),
        "seniority_level":  profile.seniority_level or "Not specified",
        "education":        "\n".join(education_lines) if education_lines else "Not specified",
        "technical_skills": ", ".join(profile.skills.technical[:30]) or "Not specified",
        "tools":            ", ".join(profile.skills.tools[:20]) or "Not specified",
        "domain_expertise": ", ".join(profile.domain_expertise[:10]) or "Not specified",
        "ats_summary":      ats,
    }


# -- Per-job LLM scoring -------------------------------------------------------

@traceable(
    name="job-scorer-llm",
    run_type="llm",
    metadata={"model": cfg.ranker.model, "agent": "ranker"},
)
@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4), reraise=False)
async def _score_job(
    job:            RawJob,
    candidate_subs: dict[str, str],
) -> tuple[dict, dict]:
    _neutral = {
        "title_relevance":        1.0,   # neutral fallback: don't filter on scoring failure
        "education_required":     False,
        "education_match":        None,
        "experience_match":       0.5,
        "skill_match":            0.5,
        "domain_match":           0.5,
        "overqualified":          None,   # neutral: unknown on scoring failure
        "sparse_jd":              False,  # neutral: don't penalise on scoring failure
        "india_accessible":       True,   # neutral: don't drop on scoring failure
        "job_location_extracted": None,  # neutral: fall back to metadata location
        "scoring_notes":          "Scoring unavailable -- neutral scores applied.",
        "experience_gap":         None,
        "skill_gaps":             [],
        "domain_gap":             None,
        "education_gap":          None,
    }
    _zero_tokens: dict = {"input": 0, "output": 0}

    prompt = JOB_SCORING_PROMPT
    for key, value in candidate_subs.items():
        prompt = prompt.replace(f"{{{key}}}", value)
    prompt = prompt.replace("{target_role}", job.matched_profile.split("|")[0].strip())
    prompt = prompt.replace("{jd_text}", job.jd_text[:6000])

    _raw_response: str = ""

    async with _SCORING_SEMAPHORE:
        try:
            client   = anthropic.AsyncAnthropic()
            response = await asyncio.wait_for(
                client.messages.create(
                    model       = _CFG.model,
                    max_tokens  = _CFG.max_tokens,
                    temperature = _CFG.temperature,
                    messages    = [{"role": "user", "content": prompt}],
                ),
                timeout=_CFG.timeout_secs,
            )

            usage = getattr(response, "usage", None)
            tok   = {
                "input":  getattr(usage, "input_tokens",  0) if usage else 0,
                "output": getattr(usage, "output_tokens", 0) if usage else 0,
            }

            raw = response.content[0].text.strip()
            _raw_response = raw

            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\s*```$",           "", raw, flags=re.MULTILINE)

            result = json.loads(raw)

            edu_required = bool(result.get("education_required", False))
            edu_raw      = result.get("education_match")
            edu_score    = (
                max(0.0, min(1.0, float(edu_raw)))
                if edu_required and edu_raw is not None
                else None
            )

            title_rel_raw = result.get("title_relevance")
            title_rel     = (
                max(0.0, min(1.0, float(title_rel_raw)))
                if title_rel_raw is not None
                else 1.0   # missing field -> don't penalise
            )

            # job_location_extracted: clean and normalise, None if missing/blank
            _loc_raw = result.get("job_location_extracted")
            _loc_extracted = str(_loc_raw).strip() if _loc_raw and str(_loc_raw).strip() else None

            scores = {
                "title_relevance":        title_rel,
                "education_required":     edu_required,
                "education_match":        edu_score,
                "experience_match":       max(0.0, min(1.0, float(result.get("experience_match", 0.5)))),
                "skill_match":            max(0.0, min(1.0, float(result.get("skill_match",      0.5)))),
                "domain_match":           max(0.0, min(1.0, float(result.get("domain_match",     0.5)))),
                "overqualified":          bool(result.get("overqualified", False)),
                "sparse_jd":              bool(result.get("sparse_jd",       False)),
                "india_accessible":       bool(result.get("india_accessible", True)),
                "job_location_extracted": _loc_extracted,
                "scoring_notes":          str(result.get("scoring_notes", "")),
                "experience_gap":         result.get("experience_gap") or None,
                "skill_gaps":             [s for s in (result.get("skill_gaps") or []) if isinstance(s, str)][:4],
                "domain_gap":             result.get("domain_gap") or None,
                "education_gap":          result.get("education_gap") or None,
            }

            edu_str = f"edu={scores['education_match']:.2f}" if scores["education_match"] is not None else "edu=n/a"
            logger.debug(
                f"[ranker:score] '{job.title}' @ {job.company} | "
                f"title_rel={scores['title_relevance']:.2f} "
                f"exp={scores['experience_match']:.2f} "
                f"skill={scores['skill_match']:.2f} "
                f"domain={scores['domain_match']:.2f} "
                f"{edu_str}"
            )
            return scores, tok

        except asyncio.TimeoutError:
            logger.error(
                f"[ranker:score] TIMEOUT for '{job.title}' @ {job.company} "
                f"(timeout={_CFG.timeout_secs}s) -- neutral fallback"
            )
        except json.JSONDecodeError as e:
            logger.error(
                f"[ranker:score] JSON parse FAILED for '{job.title}' @ {job.company} | "
                f"error={e} | raw_response={_raw_response[:800]!r}"
            )
        except Exception as e:
            logger.error(
                f"[ranker:score] Unexpected error for '{job.title}' @ {job.company}: "
                f"{type(e).__name__}: {e} -- neutral fallback"
            )

        return _neutral, _zero_tokens


# -- Main async ranking function -----------------------------------------------

@traceable(
    name="agent-6-ranker",
    run_type="chain",
    metadata={"agent": "ranker"},
)
async def rank_jobs_async(
    jobs:      list[RawJob],
    profile:   CandidateProfile,
    confirmed: list[SuggestedProfile],
) -> tuple[list[RankedJob], dict]:
    if not jobs:
        return [], {"input": 0, "output": 0}

    capped_jobs    = apply_profile_caps(jobs, confirmed)
    candidate_subs = _format_candidate_for_prompt(profile)

    logger.info(
        f"[ranker] Scoring {len(capped_jobs)} jobs with {_CFG.model} "
        f"(temp={_CFG.temperature}, timeout={_CFG.timeout_secs}s)..."
    )

    BATCH_SIZE  = _CFG.batch_size
    BATCH_DELAY = _CFG.batch_delay_secs

    score_results = []
    for i in range(0, len(capped_jobs), BATCH_SIZE):
        batch = capped_jobs[i : i + BATCH_SIZE]
        batch_scores = await asyncio.gather(*[
            _score_job(job, candidate_subs)
            for job in batch
        ])
        score_results.extend(batch_scores)
        if i + BATCH_SIZE < len(capped_jobs):
            await asyncio.sleep(BATCH_DELAY)

    total_in_tok  = 0
    total_out_tok = 0
    all_scores    = []
    for scores, tok in score_results:
        all_scores.append(scores)
        total_in_tok  += tok.get("input",  0)
        total_out_tok += tok.get("output", 0)

    ranked: list[RankedJob] = []
    for job, scores in zip(capped_jobs, all_scores):
        title_rel_raw = scores.get("title_relevance")
        title_rel     = round(title_rel_raw, 3) if title_rel_raw is not None else None
        exp_score     = round(scores["experience_match"], 3)
        skill_score   = round(scores["skill_match"],      3)
        domain_score  = round(scores["domain_match"],     3)
        rec_score     = round(_recency_score(job),        3)
        edu_required  = scores.get("education_required", False)
        edu_score_raw = scores.get("education_match")
        edu_score     = round(edu_score_raw, 3) if edu_score_raw is not None else None

        if edu_required and edu_score is not None:
            w = _CFG.weights_with_education
            fit_score = round(
                w.experience * exp_score   +
                w.skill      * skill_score +
                w.domain     * domain_score +
                w.education  * edu_score,
                3,
            )
        else:
            w = _CFG.weights_without_education
            fit_score = round(
                w.experience * exp_score   +
                w.skill      * skill_score +
                w.domain     * domain_score,
                3,
            )

        matched_via = [p.strip() for p in job.matched_profile.split("|") if p.strip()]

        ranked.append(RankedJob(
            job_id                 = job.job_id,
            title                  = job.title,
            company                = job.company,
            location               = job.location,
            work_type              = job.work_type,
            jd_text                = job.jd_text,
            apply_url              = job.apply_url,
            source                 = job.source,
            posted_date            = job.posted_date,
            matched_via            = matched_via,
            matched_profile        = job.matched_profile,
            # HyDE section fields -- must propagate from RawJob so the API can
            # split results into section1_jobs / section2_jobs correctly.
            hyde_section           = job.hyde_section,
            jd1_emb_score          = job.jd1_emb_score,
            jd2_emb_score          = job.jd2_emb_score,
            fit_score              = fit_score,
            title_relevance        = title_rel,
            experience_score       = exp_score,
            skill_score            = skill_score,
            domain_score           = domain_score,
            recency_score          = rec_score,
            education_score        = edu_score,
            education_required     = edu_required,
            sparse_jd              = scores.get("sparse_jd", False),
            india_accessible       = scores.get("india_accessible", True),
            job_location_extracted = scores.get("job_location_extracted"),
            scoring_notes          = scores.get("scoring_notes", ""),
            experience_gap         = scores.get("experience_gap"),
            skill_gaps             = scores.get("skill_gaps", []),
            domain_gap             = scores.get("domain_gap"),
            education_gap          = scores.get("education_gap"),
            overqualified          = scores.get("overqualified"),
        ))

    ranked.sort(key=lambda j: j.fit_score, reverse=True)

    edu_count = sum(1 for j in ranked if j.education_required)
    w_e = _CFG.weights_with_education
    w_n = _CFG.weights_without_education

    if ranked:
        logger.info(
            f"[ranker] Complete: {len(ranked)} ranked | "
            f"top={ranked[0].fit_score:.3f} bottom={ranked[-1].fit_score:.3f} | "
            f"tokens={total_in_tok}in/{total_out_tok}out | "
            f"edu_required={edu_count}/{len(ranked)} | "
            f"weights(edu): exp={w_e.experience:.0%} skill={w_e.skill:.0%} "
            f"domain={w_e.domain:.0%} edu={w_e.education:.0%} | "
            f"weights(no_edu): exp={w_n.experience:.0%} skill={w_n.skill:.0%} "
            f"domain={w_n.domain:.0%}"
        )
    else:
        logger.warning(f"[ranker] Complete: 0 ranked | tokens={total_in_tok}in/{total_out_tok}out")

    return ranked, {"input": total_in_tok, "output": total_out_tok}


# -- Main agent function -------------------------------------------------------

async def run_ranker_agent(state: SessionState) -> SessionState:
    """
    Agent 6 -- Ranker.

    Receives deduplicated, HyDE-partitioned jobs from the graph.
    Dedup runs in node_dedup; semantic prefiltering runs in node_hyde_prefilter —
    both upstream of this agent.

    Scores each job via Claude Haiku, then applies experience / title /
    fit_score / location / india_accessible filters before returning ranked_jobs.
    """
    state.current_agent = "ranker"
    _t0 = time.perf_counter()
    logger.info(
        f"[ranker] Starting -- session_id={state.session_id} | "
        f"raw_jobs={len(state.raw_jobs)}"
    )

    if not state.raw_jobs:
        logger.warning("[ranker] No raw jobs -- skipping")
        state.ranked_jobs   = []
        state.results_ready = True
        return state

    if not state.candidate_profile:
        logger.error("[ranker] No candidate profile -- cannot rank")
        state.error = "Cannot rank jobs: candidate profile missing."
        return state

    # ── Stage tracking dict ───────────────────────────────────────────────────
    jobs_by_stage: dict[str, int] = {}

    # Dedup ran upstream (node_dedup in graph.py) — use raw_jobs directly.
    unique_jobs = state.raw_jobs
    jobs_by_stage["raw_into_ranker"] = len(unique_jobs)

    # Snapshot for per-job audit trail.
    _audit_all_jobs: list[RawJob] = list(unique_jobs)

    jobs_by_stage["post_skill_prefilter"] = len(unique_jobs)

    # ── Initialise per-job audit dict ─────────────────────────────────────────
    # Keyed by job_id. HyDE already dropped irrelevant jobs upstream — every
    # job here has a hyde_section ("S1" or "S2"). dropped_at starts as None
    # and is updated to the actual drop stage as jobs flow through filters.
    _job_audit: dict[str, dict] = {}
    for job in _audit_all_jobs:
        _job_audit[job.job_id] = {
            "job_id":          job.job_id,
            "title":           job.title or "",
            "company":         job.company or "",
            "url":             job.apply_url or "",
            "source":          job.source or "",
            "work_type":       job.work_type or "",
            "location":        job.location or "",
            # HyDE Prefilter fields (Agent 5b) — for observability and audit UI
            "hyde_section":    job.hyde_section or "unknown",
            "jd1_emb_score":   job.jd1_emb_score,
            "jd2_emb_score":   job.jd2_emb_score,
            "dropped_at":      None,
            # Score fields -- populated after LLM scoring
            "title_relevance":   None,
            "experience_score":  None,
            "skill_match_score": None,
            "domain_score":      None,
            "fit_score":         None,
            "skill_gaps":        [],
            "experience_gap":    None,
            "domain_gap":        None,
        }

    # ── LLM scoring ───────────────────────────────────────────────────────────
    ranked, ranker_tokens = await rank_jobs_async(
        jobs      = unique_jobs,
        profile   = state.candidate_profile,
        confirmed = state.confirmed_profiles,
    )
    jobs_by_stage["post_llm_score"] = len(ranked)

    # Update audit dict with LLM scores for every scored job
    for _j in ranked:
        if _j.job_id in _job_audit:
            _job_audit[_j.job_id].update({
                "title_relevance":   _j.title_relevance,
                "experience_score":  _j.experience_score,
                "skill_match_score": _j.skill_score,   # RankedJob field is skill_score
                "domain_score":      _j.domain_score,
                "fit_score":         _j.fit_score,
                "skill_gaps":        _j.skill_gaps or [],
                "experience_gap":    _j.experience_gap,
                "domain_gap":        _j.domain_gap,
            })

    # Snapshot after scoring -- used for zero-results fallback (Task #5)
    ranked_scored = ranked.copy()

    # ── Fix 2: Neutral scoring fallback gate ──────────────────────────────────
    # Drop jobs where LLM scoring failed and neutral scores were applied.
    # Detected by the exact scoring_notes string written by the _neutral fallback.
    # These jobs have 0.5/0.5/0.5 exp/skill/domain -- inflated neutral fit_score
    # (~0.50) that would push genuinely irrelevant jobs up the leaderboard.
    _NEUTRAL_NOTES_SIG = "Scoring unavailable -- neutral scores applied."
    _ids_pre_neutral   = {j.job_id for j in ranked}
    ranked             = [j for j in ranked if j.scoring_notes != _NEUTRAL_NOTES_SIG]
    _dropped_neutral   = len(_ids_pre_neutral) - len(ranked)
    if _dropped_neutral:
        logger.info(
            f"[ranker] scoring_fallback filter: dropped {_dropped_neutral} job(s) "
            f"with neutral scores (LLM scoring failed for those jobs)"
        )
    # Also purge from ranked_scored so fallback logic doesn't restore neutral jobs
    ranked_scored = [j for j in ranked_scored if j.scoring_notes != _NEUTRAL_NOTES_SIG]
    jobs_by_stage["post_neutral_filter"] = len(ranked)
    # Audit: mark neutral-scoring-dropped jobs
    _ids_post_neutral = {j.job_id for j in ranked}
    for _jid in _ids_pre_neutral - _ids_post_neutral:
        if _job_audit.get(_jid, {}).get("dropped_at") is None:
            _job_audit[_jid]["dropped_at"] = "scoring_fallback"

    # ── Experience score threshold filter ─────────────────────────────────────
    # Two thresholds, selected by seniority_preference:
    #   same_level -> min_experience_score_same   (stricter)
    #   step_up    -> min_experience_score_step_up (lenient: gap is expected)
    # Set either to 0.0 in llm_config.yaml to disable that threshold.
    seniority_pref = (
        state.preferences.seniority_preference if state.preferences else "same_level"
    )
    min_exp = (
        _CFG.min_experience_score_step_up
        if seniority_pref == "step_up"
        else _CFG.min_experience_score_same
    )
    if min_exp > 0.0:
        before  = len(ranked)
        ranked  = [j for j in ranked if (j.experience_score or 0.0) >= min_exp]
        dropped = before - len(ranked)
        if dropped:
            logger.info(
                f"[ranker] exp_score filter (>={min_exp}, seniority={seniority_pref}): "
                f"{before} -> {len(ranked)} jobs ({dropped} dropped)"
            )
    jobs_by_stage["post_exp_filter"] = len(ranked)

    # Audit: mark jobs dropped by exp filter
    _ranked_ids_after_llm = {j.job_id for j in ranked_scored}
    _ranked_ids_after_exp = {j.job_id for j in ranked}
    for _jid in _ranked_ids_after_llm - _ranked_ids_after_exp:
        if _job_audit.get(_jid, {}).get("dropped_at") is None:
            _job_audit[_jid]["dropped_at"] = "exp_filter"

    # Snapshot after exp filter -- used by fallback to detect which filter was binding.
    # If ranked_after_exp is non-empty but ranked later becomes empty, title was binding.
    # If ranked_after_exp is empty, exp was binding.
    ranked_after_exp = ranked.copy()

    # ── Title relevance threshold filter (S1 only) ────────────────────────────
    # S1 (domain-anchored) jobs must meet min_title_relevance.
    # S2 (transferable-skill) jobs are fully exempt -- adjacent titles are
    # expected and intentional in S2 by design.
    # Set min_title_relevance to 0.0 in llm_config.yaml to disable entirely.
    min_title_rel = _CFG.min_title_relevance
    _s1_before_title = [j for j in ranked if j.hyde_section == "S1"]
    _s2_jobs         = [j for j in ranked if j.hyde_section != "S1"]

    if min_title_rel > 0.0:
        _s1_after_title = [
            j for j in _s1_before_title
            if (j.title_relevance if j.title_relevance is not None else 1.0) >= min_title_rel
        ]
        dropped_title = len(_s1_before_title) - len(_s1_after_title)
        if dropped_title:
            logger.info(
                f"[ranker] title_relevance filter S1 (>={min_title_rel}): "
                f"{len(_s1_before_title)} S1 -> {len(_s1_after_title)} "
                f"({dropped_title} dropped) | S2: {len(_s2_jobs)} exempt"
            )
        ranked = _s1_after_title + _s2_jobs
    jobs_by_stage["post_title_filter"] = len(ranked)

    # Audit: mark S1 jobs dropped by title filter
    _ranked_ids_after_title = {j.job_id for j in ranked}
    for _jid in _ranked_ids_after_exp - _ranked_ids_after_title:
        if _job_audit.get(_jid, {}).get("dropped_at") is None:
            _job_audit[_jid]["dropped_at"] = "title_filter"

    # ── Fallback logic ─────────────────────────────────────────────────────────
    # Two independent cases:
    #
    # Case A — S1 title fallback (S1-specific):
    #   Title filter wiped all S1 exp-passing jobs but S2 may still have results.
    #   Relax title floor to 0.2 for S1 only. S2 is untouched.
    #
    # Case B — Global exp fallback (catch-all):
    #   Exp filter wiped ALL jobs (both S1 and S2). Relax exp to 0.0,
    #   then re-apply title filter to S1 only at the configured threshold.
    #
    # Cases are checked independently: A fires on S1 depletion, B fires on
    # total depletion. Both set fallback_activated=True.
    _TITLE_FALLBACK_FLOOR = 0.2  # absolute minimum title relevance during S1 fallback

    # Case A: S1-specific title fallback
    _s1_after_exp_ids    = {j.job_id for j in ranked_after_exp if j.hyde_section == "S1"}
    _s1_after_title_ids  = {j.job_id for j in ranked          if j.hyde_section == "S1"}
    if (min_title_rel > 0.0
            and _s1_after_exp_ids           # S1 had exp-passing jobs
            and not _s1_after_title_ids):   # but title filter killed all of them
        logger.warning(
            f"[ranker] S1 fallback: title_relevance filter (>={min_title_rel}) "
            f"eliminated all {len(_s1_after_exp_ids)} S1 exp-passing jobs. "
            f"Re-applying S1 with floor={_TITLE_FALLBACK_FLOOR}."
        )
        _s1_fallback = [
            j for j in ranked_after_exp
            if j.hyde_section == "S1"
            and (j.title_relevance if j.title_relevance is not None else 1.0) >= _TITLE_FALLBACK_FLOOR
        ]
        _s2_current = [j for j in ranked if j.hyde_section != "S1"]
        ranked = _s1_fallback + _s2_current
        state.fallback_activated = True
        state.fallback_reason    = "title_relaxed"
        logger.info(
            f"[ranker] S1 fallback result: {len(_s1_fallback)} S1 jobs restored "
            f"(reason=title_relaxed) | S2 unchanged: {len(_s2_current)}"
        )

    # Case B: global exp fallback (all sections wiped by exp filter)
    if not ranked and ranked_scored:
        logger.warning(
            "[ranker] Global fallback: exp_score filter eliminated all jobs "
            f"(seniority={seniority_pref}, threshold={min_exp}). "
            "Re-applying with exp_score relaxed to 0.0."
        )
        ranked = ranked_scored.copy()
        # Re-apply title filter to S1 only at the configured threshold
        if min_title_rel > 0.0:
            _s1_exp_fallback = [
                j for j in ranked
                if j.hyde_section == "S1"
                and (j.title_relevance if j.title_relevance is not None else 1.0) >= min_title_rel
            ]
            _s2_exp_fallback = [j for j in ranked if j.hyde_section != "S1"]
            ranked = _s1_exp_fallback + _s2_exp_fallback
        state.fallback_activated = True
        state.fallback_reason    = "exp_score_relaxed"
        logger.info(
            f"[ranker] Global fallback result: {len(ranked)} jobs (reason=exp_score_relaxed)"
        )

    # ── Sparse JD logging ─────────────────────────────────────────────────────
    # sparse_jd is now set by the LLM scorer (not word-count heuristic).
    # No score cap applied -- sparse jobs are handled by the min_fit_score gate.
    # The flag is surfaced as a UI banner on the job card.
    _sparse_count = sum(1 for j in ranked if j.sparse_jd)
    if _sparse_count:
        logger.info(f"[ranker] sparse_jd: {_sparse_count} job(s) flagged by LLM as thin descriptions")

    # ── Fix 3: Minimum fit_score gate ─────────────────────────────────────────
    # Drop jobs whose fit_score falls below the configured floor.
    # Set min_fit_score to 0.0 in llm_config.yaml to disable.
    min_fit = getattr(_CFG, "min_fit_score", 0.0)
    if min_fit > 0.0:
        _before_fit = len(ranked)
        ranked      = [j for j in ranked if j.fit_score >= min_fit]
        _dropped_fit = _before_fit - len(ranked)
        if _dropped_fit:
            logger.info(
                f"[ranker] min_fit_score filter (>={min_fit}): "
                f"{_before_fit} -> {len(ranked)} jobs ({_dropped_fit} dropped)"
            )
        # Audit: mark min_fit_score-dropped jobs
        _ids_after_fit = {j.job_id for j in ranked}
        for _jid in set(_job_audit) - _ids_after_fit:
            if _job_audit.get(_jid, {}).get("dropped_at") is None:
                _job_audit[_jid]["dropped_at"] = "min_fit_score"
    jobs_by_stage["post_min_fit_filter"] = len(ranked)

    # -- Post-score compliance filters -----------------------------------------
    prefs = state.preferences

    # E3: location filter
    _LOCATION_ALIASES: dict[str, list[str]] = {
        "bengaluru":  ["bengaluru", "bangalore"],
        "delhi ncr":  ["delhi", "gurgaon", "gurugram", "noida", "faridabad", "ncr"],
        "mumbai":     ["mumbai", "bombay"],
        "hyderabad":  ["hyderabad"],
        "chennai":    ["chennai", "madras"],
        "pune":       ["pune"],
    }

    def _location_matches(job_loc: str, pref_loc: str) -> bool:
        if not job_loc:
            return True
        jloc = job_loc.lower()
        ploc = pref_loc.lower().strip()
        if ploc in jloc:
            return True
        for canonical, aliases in _LOCATION_ALIASES.items():
            if ploc == canonical or any(ploc == a for a in aliases):
                return any(alias in jloc for alias in aliases)
        return False

    if prefs and prefs.location and prefs.location.strip():
        before_loc = len(ranked)
        ranked = [
            j for j in ranked
            if j.work_type == 'remote'
            # Use LLM-extracted location first (more reliable than API metadata).
            # Fall back to metadata location if extraction returned None.
            # _location_matches returns True on empty string = benefit of doubt.
            or _location_matches(
                j.job_location_extracted or j.location,
                prefs.location,
            )
        ]
        dropped_loc = before_loc - len(ranked)
        if dropped_loc:
            logger.info(
                f"[ranker] location filter (E3): dropped {dropped_loc} jobs "
                f"(preference='{prefs.location}', using LLM-extracted location with metadata fallback)"
            )
        # Audit: mark location-dropped jobs
        _passed_loc_ids = {j.job_id for j in ranked}
        for _jid in set(_job_audit) - _passed_loc_ids:
            if _job_audit.get(_jid, {}).get("dropped_at") is None:
                _job_audit[_jid]["dropped_at"] = "location_filter"
    jobs_by_stage['post_location_filter'] = len(ranked)

    # India-accessible filter
    before_ia = len(ranked)
    ranked = [
        j for j in ranked
        if j.work_type != 'remote' or j.india_accessible
    ]
    dropped_ia = before_ia - len(ranked)
    if dropped_ia:
        logger.info(
            f"[ranker] india_accessible filter: dropped {dropped_ia} remote job(s) "
            f"requiring US work authorisation"
        )
    # Audit: mark india_accessible-dropped jobs
    _passed_ia_ids = {j.job_id for j in ranked}
    for _jid in set(_job_audit) - _passed_ia_ids:
        if _job_audit.get(_jid, {}).get("dropped_at") is None:
            _job_audit[_jid]["dropped_at"] = "india_accessible_filter"
    jobs_by_stage['post_india_accessible_filter'] = len(ranked)
    jobs_by_stage['final_ranked']                 = len(ranked)

    # -- Audit finalization ----------------------------------------------------
    # Sort: passed jobs first (by fit_score desc), then each drop stage.
    # skill_prefilter removed — HyDE prefilter (Agent 5b) handles that upstream.
    _STAGE_ORDER = {
        "passed":                  0,
        "scoring_fallback":        1,
        "exp_filter":              2,
        "title_filter":            3,
        "location_filter":         4,
        "india_accessible_filter": 5,
        "min_fit_score":           6,
        "profile_cap":             7,
    }

    # Detect profile_cap drops: in _job_audit, no scores + no dropped_at
    # means apply_profile_caps() removed them before LLM scoring.
    _final_ranked_ids = {j.job_id for j in ranked}
    for _jid, _entry in _job_audit.items():
        if _entry.get("dropped_at") is None and _entry.get("fit_score") is None:
            _job_audit[_jid]["dropped_at"] = "profile_cap"

    job_audit_list = sorted(
        _job_audit.values(),
        key=lambda e: (
            _STAGE_ORDER.get(e.get("dropped_at") or "passed", 9),
            -(e.get("fit_score") or 0),
        ),
    )

    # -- Propagate ranker results into global pipeline_audit -------------------
    # Merge _job_audit scores + drop stages into state.pipeline_audit so the
    # frontend has a single source of truth for the full per-job journey.
    # Also assign final_rank (1-based) to jobs that made it to the output.
    _rank_counter = 1
    for _j in ranked:  # ranked is sorted by fit_score desc
        jid = _j.job_id
        if jid in state.pipeline_audit:
            state.pipeline_audit[jid].update({
                "fit_score":         round(_j.fit_score, 3),
                "title_relevance":   round(_j.title_relevance or 0, 3),
                "experience_score":  round(_j.experience_score, 3),
                "skill_match_score": round(_j.skill_score, 3),   # RankedJob field is skill_score
                "domain_score":      round(_j.domain_score, 3),
                "education_score":   round(_j.education_score, 3) if _j.education_score is not None else None,
                "sparse_jd":         _j.sparse_jd,
                "final_rank":        _rank_counter,
                "status":            "passed",
                "dropped_at":        None,
            })
        _rank_counter += 1

    for _jid, _entry in _job_audit.items():
        if _jid in state.pipeline_audit and _entry.get("dropped_at"):
            pa = state.pipeline_audit[_jid]
            pa["status"]     = "dropped"
            pa["dropped_at"] = _entry["dropped_at"]
            # Write any LLM scores that were computed before the drop
            for _score_key in ("fit_score", "title_relevance", "experience_score",
                               "skill_match_score", "domain_score"):
                if _entry.get(_score_key) is not None:
                    pa[_score_key] = _entry[_score_key]

    state.ranked_jobs   = ranked
    state.results_ready = True
    state.error         = None

    elapsed    = round(time.perf_counter() - _t0, 2)
    fit_scores = [j.fit_score for j in ranked] if ranked else []

    # ── Fix 8: Per-source discard rate logging ────────────────────────────────
    # Compute {source -> {retrieved, passed, dropped, pass_rate, avg_fit_score}}
    # from pipeline_audit so operators can spot low-signal sources at a glance.
    _source_stats: dict[str, dict] = {}
    for _jid, _entry in state.pipeline_audit.items():
        _src = _entry.get("source") or "unknown"
        if _src not in _source_stats:
            _source_stats[_src] = {"retrieved": 0, "passed": 0, "dropped": 0, "fit_scores": []}
        _source_stats[_src]["retrieved"] += 1
        if _entry.get("status") == "passed":
            _source_stats[_src]["passed"] += 1
            _fs = _entry.get("fit_score")
            if _fs is not None:
                _source_stats[_src]["fit_scores"].append(_fs)
        else:
            _source_stats[_src]["dropped"] += 1
    source_discard_rates: dict[str, dict] = {}
    for _src, _st in _source_stats.items():
        _ret = _st["retrieved"]
        _pas = _st["passed"]
        _fss = _st["fit_scores"]
        source_discard_rates[_src] = {
            "retrieved":     _ret,
            "passed":        _pas,
            "dropped":       _st["dropped"],
            "pass_rate":     round(_pas / _ret, 3) if _ret else 0.0,
            "avg_fit_score": round(sum(_fss) / len(_fss), 3) if _fss else None,
        }
    if source_discard_rates:
        logger.info("[ranker] per-source discard rates:")
        for _src, _st in sorted(source_discard_rates.items()):
            logger.info(
                f"  {_src}: retrieved={_st['retrieved']} passed={_st['passed']} "
                f"pass_rate={_st['pass_rate']:.0%} "
                f"avg_fit={_st['avg_fit_score']}"
            )

    existing_ranker_metrics = state.agent_metrics.get('ranker', {})
    existing_ranker_metrics.update({
        'model':                    _CFG.model,
        'input_tokens':             ranker_tokens.get('input',  0),
        'output_tokens':            ranker_tokens.get('output', 0),
        'llm_calls':                len(unique_jobs),
        'latency_secs':             elapsed,
        'jobs_ranked':              len(ranked),
        'fallback_activated':       state.fallback_activated,
        'india_accessible_dropped': dropped_ia,
        'mean_fit_score':           round(sum(fit_scores) / len(fit_scores), 3) if fit_scores else 0.0,
        'score_p25':                round(sorted(fit_scores)[len(fit_scores)//4], 3) if len(fit_scores) >= 3 else None,
        'score_p75':                round(sorted(fit_scores)[3*len(fit_scores)//4], 3) if len(fit_scores) >= 3 else None,
        'jobs_by_stage':            jobs_by_stage,
        'job_audit':                job_audit_list,
        'source_discard_rates':     source_discard_rates,
    })
    state.agent_metrics['ranker'] = existing_ranker_metrics

    logger.info(
        f"[ranker] Done -- {len(ranked)} ranked jobs | latency={elapsed}s | "
        f"stages: {jobs_by_stage}"
    )
    return state


# -- LangGraph node wrapper ----------------------------------------------------

async def node_ranker(state: dict) -> dict:
    """LangGraph async node wrapper for Agent 6 (Ranker)."""
    session = SessionState(**state)
    if not session.raw_jobs:
        logger.warning('[graph] Skipping ranker -- no raw jobs')
        return state
    updated = await run_ranker_agent(session)
    return updated.model_dump()
