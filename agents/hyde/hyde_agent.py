"""
agents/hyde/hyde_agent.py

Agent 5b -- HyDE (Hypothetical Document Embeddings) Prefilter

Responsibility:
  Receive deduplicated raw_jobs[] from the dedup node, generate two
  hypothetical job descriptions (JD1 domain-anchored, JD2 transferable) in
  parallel via Claude Sonnet, embed all job JDs + both hypotheticals in a
  single Voyage AI batch call, and partition jobs into:

    Section 1 -- "Roles in your domain"
      max(jd1_emb, jd2_emb) >= hyde_min_floor  AND  jd1_emb >= jd2_emb
      Ranked downstream by JD1 RRF signal.

    Section 2 -- "Broader opportunities"
      max(jd1_emb, jd2_emb) >= hyde_min_floor  AND  jd2_emb > jd1_emb
      Ranked downstream by JD2 RRF signal.

    Dropped
      max(jd1_emb, jd2_emb) < hyde_min_floor -- removed before ranker.

Why delta-based partition (not two independent thresholds):
  Absolute thresholds are session-dependent (score distributions shift with
  profile and job pool size). The relative JD1 vs JD2 signal is self-
  calibrating: whichever hypothetical JD is semantically closer to a real job
  determines whether that job belongs in a domain-specific or transferable
  section. A single absolute floor (hyde_min_floor) eliminates noise-level
  matches regardless of which JD they scraped higher on.

CRITICAL CONSTRAINT:
  Resume-agnostic design -- voyage-3 only. No domain-specific embedding models
  (voyage-finance-2, voyage-law-2) in production. Domain specificity comes
  from HyDE content, not model choice.

Cost:
  ~2 Sonnet calls (parallel, ~$0.01-0.02) + 1 Voyage batch (~$0.002 / 100 jobs).
  Negligible compared to the downstream ranker's Haiku calls it replaces.

LangSmith tracing:
  - @traceable on run_hyde_prefilter (chain) and _generate_dual_jds (llm)
  - Logs: jd1/jd2 word counts, job counts by section, floor used, latency
  - All LLM inputs/outputs captured via LangSmith SDK

Fallback:
  If Section 1 is empty after partition: lower floor to fallback_floor and
  retry once. If still empty: pass all jobs through as Section 1 (fail-open)
  so the ranker always has something to score.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from typing import Optional

import anthropic
from langsmith import traceable

from core.config.config_loader import cfg
from core.prompts.hyde_prompts import (
    build_domain_anchored_jd_prompt,
    build_transferable_jd_prompt,
)
from core.state.session_state import RawJob, SessionState

logger = logging.getLogger(__name__)

_CFG = cfg.hyde_prefilter


# =============================================================================
# Experience helper
# =============================================================================

def _compute_experience_years(profile_dict: dict) -> tuple[float, float]:
    """
    Compute (years_full_time, years_other) from a CandidateProfile dict.
    Mirrors agents/parser/resume_parser.py:_calculate_experience_split.
    """
    import datetime
    today        = datetime.date.today()
    ft_months    = 0
    other_months = 0

    for exp in profile_dict.get("work_experience", []):
        start     = exp.get("start_date")
        end       = exp.get("end_date")
        role_type = exp.get("role_type", "full_time")
        if not start:
            continue
        try:
            s = datetime.date(int(start[:4]), int(start[5:7]), 1)
            e = datetime.date(int(end[:4]), int(end[5:7]), 1) if end else today
            months = max(0, (e.year - s.year) * 12 + (e.month - s.month))
        except (ValueError, TypeError):
            continue
        if role_type == "full_time":
            ft_months    += months
        else:
            other_months += months

    return round(ft_months / 12, 1), round(other_months / 12, 1)


# =============================================================================
# Dual JD generation (parallel Sonnet calls)
# =============================================================================

@traceable(
    name="hyde-dual-jd-generation",
    run_type="llm",
    metadata={"agent": "hyde_prefilter"},
)
async def _generate_dual_jds(
    confirmed_profile:  str,
    step_up:            bool,
    domain_description: str,
    free_text:          Optional[str],
    profile_dict:       dict,
    years_ft:           float,
    years_other:        float,
) -> tuple[str, str]:
    """
    Generate JD1 (domain-anchored) and JD2 (transferable) in parallel.
    Returns (jd1_text, jd2_text).
    """
    prompt_jd1 = build_domain_anchored_jd_prompt(
        confirmed_profile  = confirmed_profile,
        step_up            = step_up,
        domain_description = domain_description,
        free_text          = free_text,
        profile            = profile_dict,
        years_ft           = years_ft,
        years_other        = years_other,
    )
    prompt_jd2 = build_transferable_jd_prompt(
        confirmed_profile = confirmed_profile,
        step_up           = step_up,
        profile           = profile_dict,
        years_ft          = years_ft,
        years_other       = years_other,
    )

    client = anthropic.AsyncAnthropic()

    logger.info(
        f"[hyde] Generating JD1 + JD2 in parallel via {_CFG.jd_gen_model} "
        f"(profile={confirmed_profile!r}, step_up={step_up})"
    )

    resp1, resp2 = await asyncio.gather(
        asyncio.wait_for(
            client.messages.create(
                model       = _CFG.jd_gen_model,
                max_tokens  = _CFG.max_tokens,
                temperature = _CFG.temperature,
                messages    = [{"role": "user", "content": prompt_jd1}],
            ),
            timeout=_CFG.timeout_secs,
        ),
        asyncio.wait_for(
            client.messages.create(
                model       = _CFG.jd_gen_model,
                max_tokens  = _CFG.max_tokens,
                temperature = _CFG.temperature,
                messages    = [{"role": "user", "content": prompt_jd2}],
            ),
            timeout=_CFG.timeout_secs,
        ),
    )

    jd1 = resp1.content[0].text.strip()
    jd2 = resp2.content[0].text.strip()

    logger.info(
        f"[hyde] JD1 generated -- {len(jd1.split())} words | "
        f"JD2 generated -- {len(jd2.split())} words"
    )
    return jd1, jd2


# =============================================================================
# Voyage AI embedding (single paid-tier batch -- no sleep needed)
# =============================================================================

def _voyage_embed(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using Voyage AI (voyage-3, paid tier).

    Paid tier: 8M TPM / 2K RPM -- all texts fit in one or two large batches
    with no inter-batch sleep. voyage_batch_size=128 by default.

    Uses symmetric document-to-document comparison (both hypothetical JDs and
    real job JDs are embedded as 'document' type -- same format, same model).
    """
    import voyageai
    import os

    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY env var not set.")

    client     = voyageai.Client(api_key=api_key)
    batch_size = getattr(_CFG, "voyage_batch_size", 128)
    embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch  = texts[i : i + batch_size]
        result = client.embed(batch, model=_CFG.voyage_model, input_type="document")
        embeddings.extend(result.embeddings)

    return embeddings


# =============================================================================
# Scoring utilities
# =============================================================================

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_scores(query_text: str, corpus_texts: list[str]) -> list[float]:
    """BM25 scores for query_text against each text in corpus_texts."""
    from rank_bm25 import BM25Okapi
    corpus = [_tokenise(t) for t in corpus_texts]
    bm25   = BM25Okapi(corpus)
    query  = _tokenise(query_text)
    return bm25.get_scores(query).tolist()


def _rrf_fuse(score_lists: list[list[float]], k: int = 60) -> list[float]:
    """Reciprocal Rank Fusion over multiple score lists."""
    n_docs     = len(score_lists[0])
    rrf_scores = [0.0] * n_docs
    for scores in score_lists:
        ranked = sorted(range(n_docs), key=lambda i: scores[i], reverse=True)
        rank_of = [0] * n_docs
        for pos, idx in enumerate(ranked):
            rank_of[idx] = pos + 1
        for idx in range(n_docs):
            rrf_scores[idx] += 1.0 / (k + rank_of[idx])
    return rrf_scores


# =============================================================================
# Partition logic
# =============================================================================

def _partition_jobs(
    jobs:      list[RawJob],
    emb_jd1:   list[float],
    emb_jd2:   list[float],
    rrf_jd1:   list[float],
    rrf_jd2:   list[float],
    floor:     float,
) -> tuple[list[RawJob], list[RawJob], list[RawJob]]:
    """
    Delta-based partition:
      max(jd1_emb, jd2_emb) >= floor AND jd1_emb >= jd2_emb -> S1
      max(jd1_emb, jd2_emb) >= floor AND jd2_emb  > jd1_emb -> S2
      max < floor                                             -> DROPPED

    Annotates each job in-place with hyde_section, jd1_emb_score,
    jd2_emb_score, rrf_jd1, rrf_jd2.

    Returns (s1_jobs, s2_jobs, dropped_jobs).
    """
    s1: list[RawJob] = []
    s2: list[RawJob] = []
    dr: list[RawJob] = []

    for i, job in enumerate(jobs):
        e1 = emb_jd1[i]
        e2 = emb_jd2[i]

        # Annotate in-place -- these fields land in session state + audit trail
        job.jd1_emb_score = round(e1, 4)
        job.jd2_emb_score = round(e2, 4)
        job.rrf_jd1       = round(rrf_jd1[i], 5)
        job.rrf_jd2       = round(rrf_jd2[i], 5)

        if max(e1, e2) < floor:
            job.hyde_section = None   # will be excluded from raw_jobs
            dr.append(job)
        elif e1 >= e2:
            job.hyde_section = "S1"
            s1.append(job)
        else:
            job.hyde_section = "S2"
            s2.append(job)

    # Sort within sections by respective RRF signal (ranker re-sorts by fit_score later)
    s1.sort(key=lambda j: j.rrf_jd1 or 0.0, reverse=True)
    s2.sort(key=lambda j: j.rrf_jd2 or 0.0, reverse=True)
    dr.sort(key=lambda j: max(j.jd1_emb_score or 0.0, j.jd2_emb_score or 0.0), reverse=True)

    return s1, s2, dr


# =============================================================================
# Main agent function
# =============================================================================

@traceable(
    name="agent-5b-hyde-prefilter",
    run_type="chain",
    metadata={"agent": "hyde_prefilter"},
)
async def run_hyde_prefilter(state: SessionState) -> SessionState:
    """
    Agent 5b -- HyDE Prefilter.

    Pipeline position: after dedup node, before rank_results.

    Steps:
      1. Generate JD1 + JD2 in parallel (Sonnet, asyncio.gather)
      2. Embed all job JDs + JD1 + JD2 in one Voyage batch
      3. Compute BM25 + RRF per JD
      4. Partition: delta-based (JD1 vs JD2 signal strength)
      5. Annotate raw_jobs; drop below-floor jobs from state.raw_jobs
      6. Store hypo_jd1, hypo_jd2 in state
      7. Write agent_metrics["hyde_prefilter"]

    Fail-open: on any error, returns state unmodified so ranker still runs.
    """
    state.current_agent = "hyde_prefilter"
    _t0 = time.perf_counter()

    if not state.raw_jobs:
        logger.warning("[hyde] No raw jobs -- skipping")
        return state

    if not state.candidate_profile:
        logger.warning("[hyde] No candidate profile -- skipping (fail-open)")
        return state

    if not state.confirmed_profiles:
        logger.warning("[hyde] No confirmed profiles -- skipping (fail-open)")
        return state

    jobs       = state.raw_jobs
    input_count = len(jobs)
    profile    = state.candidate_profile
    confirmed  = state.confirmed_profiles[0]   # primary confirmed profile

    logger.info(
        f"[hyde] Starting -- session_id={state.session_id} | "
        f"jobs={input_count} | profile={confirmed.title!r} | step_up={confirmed.is_stretch}"
    )

    try:
        # -- Step 1: Parallel JD generation ------------------------------------
        profile_dict  = profile.model_dump()
        years_ft, years_other = _compute_experience_years(profile_dict)

        # domain_description: from session preferences (free-text field)
        # Passed through from HITL; empty string if not set
        domain_description = ""
        if state.preferences and hasattr(state.preferences, "domain_description"):
            domain_description = state.preferences.domain_description or ""

        jd1, jd2 = await _generate_dual_jds(
            confirmed_profile  = confirmed.title,
            step_up            = confirmed.is_stretch,
            domain_description = domain_description,
            free_text          = None,   # free_text not in current SessionState schema
            profile_dict       = profile_dict,
            years_ft           = years_ft,
            years_other        = years_other,
        )

        state.hypo_jd1 = jd1
        state.hypo_jd2 = jd2

        # -- Step 2: Voyage AI embedding (single batch) ------------------------
        jd_texts   = [j.jd_text or "" for j in jobs]
        all_texts  = [jd1, jd2] + jd_texts   # JD1=[0], JD2=[1], jobs=[2:]

        logger.info(
            f"[hyde] Embedding {len(all_texts)} texts via {_CFG.voyage_model} "
            f"(batch_size={getattr(_CFG, 'voyage_batch_size', 128)})"
        )
        _t_emb = time.perf_counter()
        all_embeddings = await asyncio.to_thread(_voyage_embed, all_texts)
        emb_latency    = round(time.perf_counter() - _t_emb, 2)
        logger.info(f"[hyde] Embedding complete in {emb_latency}s")

        jd1_emb        = all_embeddings[0]
        jd2_emb        = all_embeddings[1]
        job_embeddings = all_embeddings[2:]   # reused for both JD comparisons

        emb_jd1 = [_cosine_similarity(jd1_emb, e) for e in job_embeddings]
        emb_jd2 = [_cosine_similarity(jd2_emb, e) for e in job_embeddings]

        # -- Step 3: BM25 + RRF per JD ----------------------------------------
        bm25_jd1 = _bm25_scores(jd1, jd_texts)
        bm25_jd2 = _bm25_scores(jd2, jd_texts)
        rrf_jd1  = _rrf_fuse([bm25_jd1, emb_jd1])
        rrf_jd2  = _rrf_fuse([bm25_jd2, emb_jd2])

        # -- Step 4: Partition with fallback retry -----------------------------
        floor = getattr(_CFG, "hyde_min_floor", 0.50)
        s1_jobs, s2_jobs, dropped = _partition_jobs(
            jobs, emb_jd1, emb_jd2, rrf_jd1, rrf_jd2, floor
        )

        # Fallback: if S1 empty, lower floor and retry once
        fallback_used = False
        if not s1_jobs:
            fallback_floor = getattr(_CFG, "fallback_floor", 0.45)
            logger.warning(
                f"[hyde] Section 1 empty at floor={floor} -- "
                f"retrying with fallback_floor={fallback_floor}"
            )
            s1_jobs, s2_jobs, dropped = _partition_jobs(
                jobs, emb_jd1, emb_jd2, rrf_jd1, rrf_jd2, fallback_floor
            )
            fallback_used = True
            floor = fallback_floor

        # Initialise cap-drop counters -- set to 0 here so the metrics block
        # below always has these defined regardless of which branch runs.
        _s1_cap_dropped = 0
        _s2_cap_dropped = 0

        # Hard fallback: if still empty, pass all jobs as S1
        if not s1_jobs and not s2_jobs:
            logger.error(
                "[hyde] Both sections empty even after fallback -- "
                "passing all jobs as S1 (fail-open)"
            )
            for i, job in enumerate(jobs):
                job.hyde_section  = "S1"
                job.jd1_emb_score = round(emb_jd1[i], 4)
                job.jd2_emb_score = round(emb_jd2[i], 4)
                job.rrf_jd1       = round(rrf_jd1[i], 5)
                job.rrf_jd2       = round(rrf_jd2[i], 5)
            state.raw_jobs = jobs
        else:
            # -- Per-section caps applied AFTER floor filtering ----------------
            # Caps are enforced here (in HyDE) rather than inside the ranker so
            # the ranker's apply_profile_caps() still works correctly within the
            # already-capped sets. Jobs are already sorted by RRF signal within
            # each section, so truncation keeps the highest-signal jobs.
            s1_cap = getattr(_CFG, "s1_max_jobs", 40)
            s2_cap = getattr(_CFG, "s2_max_jobs", 20)

            s1_dropped_by_cap: list[RawJob] = []
            s2_dropped_by_cap: list[RawJob] = []

            if len(s1_jobs) > s1_cap:
                s1_dropped_by_cap  = s1_jobs[s1_cap:]
                s1_jobs            = s1_jobs[:s1_cap]
                _s1_cap_dropped    = len(s1_dropped_by_cap)
                logger.info(
                    f"[hyde] S1 cap applied: {len(s1_jobs) + _s1_cap_dropped} → {len(s1_jobs)} "
                    f"(cap={s1_cap}, dropped {_s1_cap_dropped} lowest-RRF jobs)"
                )

            if len(s2_jobs) > s2_cap:
                s2_dropped_by_cap  = s2_jobs[s2_cap:]
                s2_jobs            = s2_jobs[:s2_cap]
                _s2_cap_dropped    = len(s2_dropped_by_cap)
                logger.info(
                    f"[hyde] S2 cap applied: {len(s2_jobs) + _s2_cap_dropped} → {len(s2_jobs)} "
                    f"(cap={s2_cap}, dropped {_s2_cap_dropped} lowest-RRF jobs)"
                )

            # Mark cap-dropped jobs in pipeline_audit
            for job in s1_dropped_by_cap:
                if job.job_id in state.pipeline_audit:
                    state.pipeline_audit[job.job_id]["status"]        = "dropped"
                    state.pipeline_audit[job.job_id]["dropped_at"]    = "hyde_section_cap"
                    state.pipeline_audit[job.job_id]["drop_reason"]   = f"S1 cap={s1_cap}: low RRF score"
                    state.pipeline_audit[job.job_id]["jd1_emb_score"] = job.jd1_emb_score
                    state.pipeline_audit[job.job_id]["jd2_emb_score"] = job.jd2_emb_score
                    state.pipeline_audit[job.job_id]["hyde_section"]  = "S1"
            for job in s2_dropped_by_cap:
                if job.job_id in state.pipeline_audit:
                    state.pipeline_audit[job.job_id]["status"]        = "dropped"
                    state.pipeline_audit[job.job_id]["dropped_at"]    = "hyde_section_cap"
                    state.pipeline_audit[job.job_id]["drop_reason"]   = f"S2 cap={s2_cap}: low RRF score"
                    state.pipeline_audit[job.job_id]["jd1_emb_score"] = job.jd1_emb_score
                    state.pipeline_audit[job.job_id]["jd2_emb_score"] = job.jd2_emb_score
                    state.pipeline_audit[job.job_id]["hyde_section"]  = "S2"

            state.raw_jobs = s1_jobs + s2_jobs   # DROPPED jobs excluded

        # -- Step 6: Pipeline audit -------------------------------------------
        # Annotate passed jobs with their embedding scores and section.
        # Mark dropped jobs with dropped_at=hyde_floor + their scores.
        for job in s1_jobs:
            if job.job_id in state.pipeline_audit:
                state.pipeline_audit[job.job_id]["jd1_emb_score"] = job.jd1_emb_score
                state.pipeline_audit[job.job_id]["jd2_emb_score"] = job.jd2_emb_score
                state.pipeline_audit[job.job_id]["hyde_section"]  = "S1"
        for job in s2_jobs:
            if job.job_id in state.pipeline_audit:
                state.pipeline_audit[job.job_id]["jd1_emb_score"] = job.jd1_emb_score
                state.pipeline_audit[job.job_id]["jd2_emb_score"] = job.jd2_emb_score
                state.pipeline_audit[job.job_id]["hyde_section"]  = "S2"
        for job in dropped:
            if job.job_id in state.pipeline_audit:
                state.pipeline_audit[job.job_id]["status"]        = "dropped"
                state.pipeline_audit[job.job_id]["dropped_at"]    = "hyde_floor"
                state.pipeline_audit[job.job_id]["drop_reason"]   = (
                    f"max_emb={max(job.jd1_emb_score or 0, job.jd2_emb_score or 0):.4f}"
                    f" < floor={floor}"
                )
                state.pipeline_audit[job.job_id]["jd1_emb_score"] = job.jd1_emb_score
                state.pipeline_audit[job.job_id]["jd2_emb_score"] = job.jd2_emb_score
                state.pipeline_audit[job.job_id]["hyde_section"]  = "dropped"

        # -- Step 7: Observability --------------------------------------------
        elapsed = round(time.perf_counter() - _t0, 2)
        logger.info(
            f"[hyde] Complete -- "
            f"S1={len(s1_jobs)} (cap_dropped={_s1_cap_dropped}) | "
            f"S2={len(s2_jobs)} (cap_dropped={_s2_cap_dropped}) | "
            f"floor_dropped={len(dropped)} | "
            f"floor={floor} | fallback={fallback_used} | latency={elapsed}s"
        )

        if dropped:
            for job in dropped[:10]:   # log top-10 nearest-miss drops
                logger.debug(
                    f"  [hyde:drop] '{job.title}' @ {job.company} | "
                    f"jd1_emb={job.jd1_emb_score:.4f} | jd2_emb={job.jd2_emb_score:.4f}"
                )

        state.agent_metrics["hyde_prefilter"] = {
            "model":            _CFG.jd_gen_model,
            "voyage_model":     _CFG.voyage_model,
            "llm_calls":        2,          # JD1 + JD2 (parallel)
            "latency_secs":     elapsed,
            "emb_latency_secs": emb_latency,
            "jobs_in":          input_count,
            "section1_jobs":    len(s1_jobs),
            "section2_jobs":    len(s2_jobs),
            "jobs_dropped":     len(dropped),
            "s1_cap_dropped":   _s1_cap_dropped,
            "s2_cap_dropped":   _s2_cap_dropped,
            "floor_used":       floor,
            "fallback_used":    fallback_used,
            "jd1_words":        len(jd1.split()),
            "jd2_words":        len(jd2.split()),
        }

    except asyncio.TimeoutError:
        logger.error(f"[hyde] JD generation timed out after {_CFG.timeout_secs}s -- fail-open")
    except Exception as e:
        logger.error(f"[hyde] Unexpected error ({e}) -- fail-open, all jobs passed to ranker")

    return state


# =============================================================================
# LangGraph node wrapper
# =============================================================================

async def node_hyde_prefilter(state: dict) -> dict:
    """
    LangGraph async node wrapper for Agent 5b (HyDE Prefilter).
    Runs between node_dedup and node_ranker in the graph.
    """
    session = SessionState(**state)

    if not session.raw_jobs:
        logger.warning("[graph] Skipping HyDE prefilter -- no raw jobs")
        return state

    updated = await run_hyde_prefilter(session)
    return updated.model_dump()
