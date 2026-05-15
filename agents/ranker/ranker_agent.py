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
    profile_rank: dict[str, int] = {
        p.title: i for i, p in enumerate(confirmed_profiles)
    }

    buckets: dict[int, list[RawJob]] = {i: [] for i in range(len(confirmed_profiles))}
    unmatched: list[RawJob] = []

    for job in jobs:
        primary = job.matched_profile.split("|")[0].strip()
        rank    = profile_rank.get(primary)
        if rank is not None and rank in buckets:
            buckets[rank].append(job)
        else:
            unmatched.append(job)

    def _recency_key(job: RawJob) -> datetime:
        if job.posted_date:
            return job.posted_date.replace(tzinfo=timezone.utc) if job.posted_date.tzinfo is None else job.posted_date
        return datetime.min.replace(tzinfo=timezone.utc)

    capped: list[RawJob] = []
    for rank, bucket in buckets.items():
        cap     = _CFG.profile_caps.as_dict().get(rank, 10)
        kept    = sorted(bucket, key=_recency_key, reverse=True)[:cap]
        dropped = len(bucket) - len(kept)
        logger.info(
            f"[ranker:cap] Profile rank {rank + 1} "
            f"('{confirmed_profiles[rank].title}'): "
            f"{len(bucket)} jobs -> cap {cap} -> kept {len(kept)}"
            + (f" (dropped {dropped} older)" if dropped else "")
        )
        capped.extend(kept)

    if unmatched:
        last_cap = _CFG.profile_caps.as_dict().get(len(confirmed_profiles) - 1, 10)
        capped.extend(unmatched[:last_cap])

    logger.info(f"[ranker:cap] Total after capping: {len(capped)} jobs")
    return capped


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
        "title_relevance":    1.0,   # neutral fallback: don't filter on scoring failure
        "education_required": False,
        "education_match":    None,
        "experience_match":   0.5,
        "skill_match":        0.5,
        "domain_match":       0.5,
        "scoring_notes":      "Scoring unavailable -- neutral scores applied.",
        "experience_gap":     None,
        "skill_gaps":         [],
        "domain_gap":         None,
        "education_gap":      None,
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

            scores = {
                "title_relevance":    title_rel,
                "education_required": edu_required,
                "education_match":    edu_score,
                "experience_match":   max(0.0, min(1.0, float(result.get("experience_match", 0.5)))),
                "skill_match":        max(0.0, min(1.0, float(result.get("skill_match",      0.5)))),
                "domain_match":       max(0.0, min(1.0, float(result.get("domain_match",     0.5)))),
                "scoring_notes":      str(result.get("scoring_notes", "")),
                "experience_gap":     result.get("experience_gap") or None,
                "skill_gaps":         [s for s in (result.get("skill_gaps") or []) if isinstance(s, str)][:4],
                "domain_gap":         result.get("domain_gap") or None,
                "education_gap":      result.get("education_gap") or None,
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
            job_id             = job.job_id,
            title              = job.title,
            company            = job.company,
            location           = job.location,
            work_type          = job.work_type,
            jd_text            = job.jd_text,
            apply_url          = job.apply_url,
            source             = job.source,
            posted_date        = job.posted_date,
            matched_via        = matched_via,
            matched_profile    = job.matched_profile,
            fit_score          = fit_score,
            title_relevance    = title_rel,
            experience_score   = exp_score,
            skill_score        = skill_score,
            domain_score       = domain_score,
            recency_score      = rec_score,
            education_score    = edu_score,
            education_required = edu_required,
            scoring_notes      = scores.get("scoring_notes", ""),
            experience_gap     = scores.get("experience_gap"),
            skill_gaps         = scores.get("skill_gaps", []),
            domain_gap         = scores.get("domain_gap"),
            education_gap      = scores.get("education_gap"),
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
    """Agent 6 -- Ranker. Deduplicates, applies caps, scores, ranks."""
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

    unique_jobs           = deduplicate(state.raw_jobs)
    ranked, ranker_tokens = await rank_jobs_async(
        jobs      = unique_jobs,
        profile   = state.candidate_profile,
        confirmed = state.confirmed_profiles,
    )

    # -- Experience score threshold filter -------------------------------------
    # Two thresholds, selected by seniority_preference:
    #   same_level -> min_experience_score_same   (stricter: candidate should be competitive)
    #   step_up    -> min_experience_score_step_up (lenient: experience gap is expected)
    # Configurable via ranker.min_experience_score_same / min_experience_score_step_up
    # in llm_config.yaml. Set either to 0.0 to disable that threshold.
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

    # -- Title relevance threshold filter --------------------------------------
    # Drop jobs where the LLM-assigned title_relevance is below the configured
    # minimum. These are roles returned by APIs that are semantically related
    # but don't match the profile the candidate searched for
    # (e.g. "Data Scientist" returned when searching "ML Engineer").
    # Configurable via ranker.min_title_relevance in llm_config.yaml.
    # Set to 0.0 to disable the filter entirely.
    # NOTE: uses explicit None check (not `or`) so that a genuine score of 0.0
    # is treated as 0.0 and filtered, not as falsy and replaced with 1.0.
    min_title_rel = _CFG.min_title_relevance
    if min_title_rel > 0.0:
        before  = len(ranked)
        ranked  = [j for j in ranked if (j.title_relevance if j.title_relevance is not None else 1.0) >= min_title_rel]
        dropped = before - len(ranked)
        if dropped:
            logger.info(
                f"[ranker] title_relevance filter (>={min_title_rel}): "
                f"{before} -> {len(ranked)} jobs ({dropped} dropped)"
            )

    # -- Sparse JD detection + fit_score cap ----------------------------------
    # JDs that slipped through the normalizer's 60-word gate but are still thin
    # (< sparse_jd_word_threshold words) receive unreliably inflated scores
    # because Claude finds no contradicting evidence in the near-empty text.
    # Flag these as sparse_jd=True and cap their fit_score so they cannot
    # top the leaderboard over richer, genuinely better-matched listings.
    # Re-sort after capping so rankings reflect the corrected scores.
    # Configurable via ranker.sparse_jd_word_threshold / sparse_jd_fit_score_cap
    # in llm_config.yaml. Set sparse_jd_word_threshold=0 to disable.
    sparse_threshold = _CFG.sparse_jd_word_threshold
    sparse_cap       = _CFG.sparse_jd_fit_score_cap
    if sparse_threshold > 0:
        capped = 0
        for j in ranked:
            word_count = len(j.jd_text.split())
            if word_count < sparse_threshold:
                j.sparse_jd = True
                if sparse_cap < 1.0 and j.fit_score > sparse_cap:
                    j.fit_score = round(sparse_cap, 3)
                    capped += 1
        if capped:
            logger.info(
                f"[ranker] sparse_jd cap (<{sparse_threshold} words, cap={sparse_cap}): "
                f"{capped} job(s) fit_score capped"
            )
        flagged = sum(1 for j in ranked if j.sparse_jd)
        if flagged:
            logger.info(
                f"[ranker] sparse_jd flagged: {flagged} job(s) marked sparse_jd=True"
            )
        # Re-sort after capping so order reflects corrected scores
        ranked.sort(key=lambda j: j.fit_score, reverse=True)

    # -- Post-score compliance filters -----------------------------------------
    # Applied AFTER LLM scoring so every job has a fit_score, but BEFORE
    # storing to state.ranked_jobs.  These enforce the user's explicit
    # location and work_type preferences using authoritative post-ranking
    # checks -- more reliable than API-level filters which are inconsistent
    # across sources.

    prefs = state.preferences

    # E2: work_type filter
    # "any" bypasses entirely.  "on-site", "onsite", "office" are treated as
    # equivalent (all mean the user wants an in-office role).
    # Jobs with work_type=None are kept (benefit of the doubt -- unknown).
    _OFFICE_VARIANTS = {"office", "on-site", "onsite"}

    def _work_type_matches(job_wt: str | None, pref_wt: str) -> bool:
        if job_wt is None:
            return True
        jnorm = job_wt.lower().strip()
        pnorm = pref_wt.lower().strip()
        if jnorm in _OFFICE_VARIANTS:
            jnorm = "office"
        if pnorm in _OFFICE_VARIANTS:
            pnorm = "office"
        return jnorm == pnorm

    if prefs and prefs.work_type and prefs.work_type != "any":
        before_wt = len(ranked)
        ranked = [j for j in ranked if _work_type_matches(j.work_type, prefs.work_type)]
        dropped_wt = before_wt - len(ranked)
        if dropped_wt:
            logger.info(
                f"[ranker] work_type filter: dropped {dropped_wt} jobs "
                f"(preference='{prefs.work_type}')"
            )

    # E3: location filter
    # Skipped when preference is blank/None.
    # Remote jobs (work_type=="remote") bypass location check -- they are
    # location-agnostic by definition.
    # Jobs whose location string is empty are kept (benefit of the doubt).
    # Matching uses a city alias map so "Bangalore" matches "Bengaluru,Karnataka"
    # and "Delhi NCR" matches "Noida", "Gurugram", etc.
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
        # Direct substring match
        if ploc in jloc:
            return True
        # Alias map lookup: find the canonical group the preference belongs to,
        # then check if any alias of that group appears in the job location.
        for canonical, aliases in _LOCATION_ALIASES.items():
            if ploc == canonical or any(ploc == a for a in aliases):
                return any(alias in jloc for alias in aliases)
        return False

    if prefs and prefs.location and prefs.location.strip():
        before_loc = len(ranked)
        ranked = [
            j for j in ranked
            if j.work_type == "remote"          # remote jobs are location-agnostic
            or _location_matches(j.location, prefs.location)
        ]
        dropped_loc = before_loc - len(ranked)
        if dropped_loc:
            logger.info(
                f"[ranker] location filter: dropped {dropped_loc} jobs "
                f"(preference='{prefs.location}')"
            )

    state.ranked_jobs   = ranked
    state.results_ready = True
    state.error         = None

    elapsed    = round(time.perf_counter() - _t0, 2)
    fit_scores = [j.fit_score for j in ranked] if ranked else []
    state.agent_metrics["ranker"] = {
        "model":          _CFG.model,
        "input_tokens":   ranker_tokens.get("input",  0),
        "output_tokens":  ranker_tokens.get("output", 0),
        "llm_calls":      len(unique_jobs),
        "latency_secs":   elapsed,
        "jobs_ranked":    len(ranked),
        "mean_fit_score": round(sum(fit_scores) / len(fit_scores), 3) if fit_scores else 0.0,
        # p25/p75 are meaningless with fewer than 3 data points -- null signals
        # "insufficient sample" to dashboards rather than a misleading value.
        "score_p25":      round(sorted(fit_scores)[len(fit_scores)//4], 3) if len(fit_scores) >= 3 else None,
        "score_p75":      round(sorted(fit_scores)[3*len(fit_scores)//4], 3) if len(fit_scores) >= 3 else None,
    }

    logger.info(f"[ranker] Done -- {len(ranked)} ranked jobs | latency={elapsed}s")
    return state


# -- LangGraph node wrapper ----------------------------------------------------

async def node_ranker(state: dict) -> dict:
    """LangGraph async node wrapper for Agent 6 (Ranker)."""
    session = SessionState(**state)
    if not session.raw_jobs:
        logger.warning("[graph] Skipping ranker -- no raw jobs")
        return state
    updated = await run_ranker_agent(session)
    return updated.model_dump()
