"""
core/config/config_loader.py
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

def _find_config() -> Path:
    env_path = os.getenv("AGENTCFG")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    here = Path(__file__).parent
    local = here / "llm_config.yaml"
    if local.exists():
        return local
    root = here.parent.parent / "llm_config.yaml"
    if root.exists():
        return root
    raise FileNotFoundError(
        "llm_config.yaml not found. Set AGENTCFG env var or place the file in "
        "core/config/ or the project root."
    )


@dataclass
class AgentLLMConfig:
    model:        str
    max_tokens:   int
    temperature:  float
    timeout_secs: int
    cost_per_million_input_tokens:  float = 0.80
    cost_per_million_output_tokens: float = 4.00

    def input_cost_usd(self, tokens: int) -> float:
        return (tokens / 1_000_000) * self.cost_per_million_input_tokens

    def output_cost_usd(self, tokens: int) -> float:
        return (tokens / 1_000_000) * self.cost_per_million_output_tokens

    def total_cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return self.input_cost_usd(input_tokens) + self.output_cost_usd(output_tokens)


@dataclass
class ResumeParserConfig(AgentLLMConfig):
    """Extended config for the resume parser -- includes text quality gate."""
    min_text_length: int = 200   # chars -- below this resume is treated as a parse failure


@dataclass
class RankerWeightsWithEducation:
    """Default weights when the JD specifies an education requirement.
    Must sum to 1.0. See llm_config.yaml [ranker.weights_with_education]."""
    experience: float = 0.40
    skill:      float = 0.30
    domain:     float = 0.20
    recency:    float = 0.00   # displayed in UI, not used in scoring
    education:  float = 0.10

    def validate(self) -> None:
        total = round(
            self.experience + self.skill + self.domain + self.recency + self.education, 6
        )
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"weights_with_education must sum to 1.0, got {total}. "
                f"Check llm_config.yaml [ranker.weights_with_education]."
            )


@dataclass
class RankerWeightsWithoutEducation:
    """Default weights when the JD has no explicit education requirement.
    Must sum to 1.0. See llm_config.yaml [ranker.weights_without_education]."""
    experience: float = 0.50
    skill:      float = 0.30
    domain:     float = 0.20
    recency:    float = 0.00   # displayed in UI, not used in scoring

    def validate(self) -> None:
        total = round(self.experience + self.skill + self.domain + self.recency, 6)
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"weights_without_education must sum to 1.0, got {total}. "
                f"Check llm_config.yaml [ranker.weights_without_education]."
            )


@dataclass
class ProfileCaps:
    rank_1: int = 30
    rank_2: int = 20
    rank_3: int = 10

    def as_dict(self) -> dict[int, int]:
        return {0: self.rank_1, 1: self.rank_2, 2: self.rank_3}


@dataclass
class RankerConfig(AgentLLMConfig):
    """Extended config for the ranker -- includes weights, caps, decay, and filters."""
    weights_with_education:    RankerWeightsWithEducation    = field(default_factory=RankerWeightsWithEducation)
    weights_without_education: RankerWeightsWithoutEducation = field(default_factory=RankerWeightsWithoutEducation)
    profile_caps:              ProfileCaps                   = field(default_factory=ProfileCaps)
    recency_decay_days:           int   = 45
    semaphore_size:               int   = 4     # max concurrent LLM calls (shared across all sessions)
    batch_size:                   int   = 4     # jobs per asyncio.gather() call
    batch_delay_secs:             float = 8.0   # sleep between batches (seconds)
    min_experience_score_same:    float = 0.5   # threshold when seniority_preference == same_level
    min_experience_score_step_up: float = 0.3   # threshold when seniority_preference == step_up
    min_title_relevance:          float = 0.40
    min_skill_overlap_ratio:      float = 0.0   # disabled -- HyDE prefilter supersedes keyword overlap
    min_fit_score:                float = 0.45  # drop jobs below this fit_score


@dataclass
class HydePrefilterConfig:
    """Config for Agent 5b -- HyDE prefilter (JD generation + Voyage AI embedding)."""
    jd_gen_model:      str   = "claude-sonnet-4-6"
    max_tokens:        int   = 1024
    temperature:       float = 0.3
    timeout_secs:      int   = 60
    voyage_model:      str   = "voyage-3"
    voyage_batch_size: int   = 128
    hyde_min_floor:    float = 0.45   # absolute noise floor -- drops irrelevant jobs
    fallback_floor:    float = 0.40   # retry floor when Section 1 is empty after first pass
    s1_max_jobs:       int   = 40     # max S1 jobs passed to ranker (sorted by RRF score)
    s2_max_jobs:       int   = 20     # max S2 jobs passed to ranker (sorted by RRF score)


@dataclass
class SourceConfig:
    """Per-source enable/pages config. One entry per source in llm_config.yaml [job_search.sources]."""
    enabled: bool = False
    pages:   int  = 0


@dataclass
class JobSearchConfig:
    num_pages:    int  = 2       # fallback pages for sources not in sources dict
    min_jd_chars: int  = 200    # chars gate -- JDs below this are dropped
    min_jd_words: int  = 60     # word gate  -- JDs below this are dropped
    max_jd_chars: int  = 8000   # truncation ceiling for JD text
    hours_old:    int  = 168    # Jobs Search API freshness window (hours)
    # Plug-and-play source registry. Keyed by source name (matches SOURCE_SPEC["name"]).
    # Sources absent from this dict are treated as disabled (enabled=False, pages=0).
    sources:      dict = field(default_factory=dict)  # dict[str, SourceConfig]

    def source(self, name: str) -> SourceConfig:
        """Return SourceConfig for a named source, or a disabled default if absent."""
        return self.sources.get(name, SourceConfig(enabled=False, pages=0))


@dataclass
class PipelineConfig:
    resume_parser:        ResumeParserConfig
    profile_recommender:  AgentLLMConfig
    job_search:           JobSearchConfig
    hyde_prefilter:       HydePrefilterConfig
    ranker:               RankerConfig
    usd_to_inr:           float = 83.5

    # Maps alias title -> canonical search title.
    # Loaded from llm_config.yaml [title_canonical_map].
    # Used by recommender (dedup) and job search (query normalisation).
    title_canonical_map:  dict = field(default_factory=dict)


def _load_config() -> PipelineConfig:
    config_path = _find_config()
    logger.info(f"[config] Loading LLM config from {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    def _agent(key: str) -> AgentLLMConfig:
        d = raw[key]
        return AgentLLMConfig(
            model        = d["model"],
            max_tokens   = int(d["max_tokens"]),
            temperature  = float(d["temperature"]),
            timeout_secs = int(d["timeout_secs"]),
            cost_per_million_input_tokens  = float(d.get("cost_per_million_input_tokens",  0.80)),
            cost_per_million_output_tokens = float(d.get("cost_per_million_output_tokens", 4.00)),
        )

    def _parser_agent() -> ResumeParserConfig:
        d = raw["resume_parser"]
        return ResumeParserConfig(
            model           = d["model"],
            max_tokens      = int(d["max_tokens"]),
            temperature     = float(d["temperature"]),
            timeout_secs    = int(d["timeout_secs"]),
            cost_per_million_input_tokens  = float(d.get("cost_per_million_input_tokens",  0.80)),
            cost_per_million_output_tokens = float(d.get("cost_per_million_output_tokens", 4.00)),
            min_text_length = int(d.get("min_text_length", 200)),
        )

    r  = raw["ranker"]
    we = r.get("weights_with_education", {})
    wn = r.get("weights_without_education", {})
    c  = r.get("profile_caps", {})

    weights_with_edu = RankerWeightsWithEducation(
        experience = float(we.get("experience", 0.30)),
        skill      = float(we.get("skill",      0.25)),
        domain     = float(we.get("domain",     0.20)),
        recency    = float(we.get("recency",    0.15)),
        education  = float(we.get("education",  0.10)),
    )
    weights_with_edu.validate()

    weights_without_edu = RankerWeightsWithoutEducation(
        experience = float(wn.get("experience", 0.35)),
        skill      = float(wn.get("skill",      0.30)),
        domain     = float(wn.get("domain",     0.20)),
        recency    = float(wn.get("recency",    0.15)),
    )
    weights_without_edu.validate()

    caps = ProfileCaps(
        rank_1 = int(c.get("rank_1", 30)),
        rank_2 = int(c.get("rank_2", 20)),
        rank_3 = int(c.get("rank_3", 10)),
    )

    ranker_cfg = RankerConfig(
        model               = r["model"],
        max_tokens          = int(r["max_tokens"]),
        temperature         = float(r["temperature"]),
        timeout_secs        = int(r["timeout_secs"]),
        cost_per_million_input_tokens  = float(r.get("cost_per_million_input_tokens",  0.80)),
        cost_per_million_output_tokens = float(r.get("cost_per_million_output_tokens", 4.00)),
        weights_with_education    = weights_with_edu,
        weights_without_education = weights_without_edu,
        profile_caps              = caps,
        recency_decay_days           = int(r.get("recency_decay_days",           45)),
        semaphore_size               = int(r.get("semaphore_size",               4)),
        batch_size                   = int(r.get("batch_size",                   4)),
        batch_delay_secs             = float(r.get("batch_delay_secs",           8.0)),
        min_experience_score_same    = float(r.get("min_experience_score_same",    0.5)),
        min_experience_score_step_up = float(r.get("min_experience_score_step_up", 0.3)),
        min_title_relevance          = float(r.get("min_title_relevance",          0.40)),
        min_skill_overlap_ratio      = float(r.get("min_skill_overlap_ratio",      0.0)),
        min_fit_score                = float(r.get("min_fit_score",                0.45)),
    )

    h = raw.get("hyde_prefilter", {})
    hyde_cfg = HydePrefilterConfig(
        jd_gen_model      = str(h.get("jd_gen_model",     "claude-sonnet-4-6")),
        max_tokens        = int(h.get("max_tokens",        1024)),
        temperature       = float(h.get("temperature",     0.3)),
        timeout_secs      = int(h.get("timeout_secs",      60)),
        voyage_model      = str(h.get("voyage_model",      "voyage-3")),
        voyage_batch_size = int(h.get("voyage_batch_size", 128)),
        hyde_min_floor    = float(h.get("hyde_min_floor",  0.45)),
        fallback_floor    = float(h.get("fallback_floor",  0.40)),
        s1_max_jobs       = int(h.get("s1_max_jobs",       40)),
        s2_max_jobs       = int(h.get("s2_max_jobs",       20)),
    )

    js = raw.get("job_search", {})

    # Parse plug-and-play source registry from job_search.sources block.
    # Each key is a source name; value has enabled + pages.
    sources_raw = js.get("sources", {}) or {}
    sources: dict[str, SourceConfig] = {}
    for src_name, src_cfg in sources_raw.items():
        src_cfg = src_cfg or {}
        sources[src_name] = SourceConfig(
            enabled = bool(src_cfg.get("enabled", False)),
            pages   = int(src_cfg.get("pages",   0)),
        )

    job_search_cfg = JobSearchConfig(
        num_pages    = int(js.get("num_pages",    2)),
        min_jd_chars = int(js.get("min_jd_chars", 200)),
        min_jd_words = int(js.get("min_jd_words", 60)),
        max_jd_chars = int(js.get("max_jd_chars", 8000)),
        hours_old    = int(js.get("hours_old",    168)),
        sources      = sources,
    )

    config = PipelineConfig(
        resume_parser        = _parser_agent(),
        profile_recommender  = _agent("profile_recommender"),
        job_search           = job_search_cfg,
        hyde_prefilter       = hyde_cfg,
        ranker               = ranker_cfg,
        usd_to_inr           = float(raw.get("usd_to_inr", 90.5)),
        title_canonical_map  = dict(raw.get("title_canonical_map", {})),
    )

    # Build source summary for log line
    src_summary = " | ".join(
        f"{name}={'on' if sc.enabled else 'off'}(p={sc.pages})"
        for name, sc in config.job_search.sources.items()
    ) or "no sources configured"

    logger.info(
        "[config] Loaded -- "
        "parser=%s temp=%s min_text_length=%s | recommender=%s temp=%s | ranker=%s temp=%s "
        "weights(edu)=%s/%s/%s/%s/%s | weights(no_edu)=%s/%s/%s/%s | "
        "min_exp_score(same)=%s min_exp_score(step_up)=%s | min_title_relevance=%s | "
        "min_fit_score=%s | "
        "ranker.batch_size=%s batch_delay_secs=%s semaphore_size=%s | "
        "job_search.num_pages=%s min_jd_chars=%s min_jd_words=%s max_jd_chars=%s hours_old=%s | "
        "sources: %s | title_canonical_map=%d entries",
        config.resume_parser.model, config.resume_parser.temperature,
        config.resume_parser.min_text_length,
        config.profile_recommender.model, config.profile_recommender.temperature,
        config.ranker.model, config.ranker.temperature,
        config.ranker.weights_with_education.experience,
        config.ranker.weights_with_education.skill,
        config.ranker.weights_with_education.domain,
        config.ranker.weights_with_education.recency,
        config.ranker.weights_with_education.education,
        config.ranker.weights_without_education.experience,
        config.ranker.weights_without_education.skill,
        config.ranker.weights_without_education.domain,
        config.ranker.weights_without_education.recency,
        config.ranker.min_experience_score_same,
        config.ranker.min_experience_score_step_up,
        config.ranker.min_title_relevance,
        config.ranker.min_fit_score,
        config.ranker.batch_size,
        config.ranker.batch_delay_secs,
        config.ranker.semaphore_size,
        config.job_search.num_pages,
        config.job_search.min_jd_chars,
        config.job_search.min_jd_words,
        config.job_search.max_jd_chars,
        config.job_search.hours_old,
        src_summary,
        len(config.title_canonical_map),
    )
    return config


cfg: PipelineConfig = _load_config()
