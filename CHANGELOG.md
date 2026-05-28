# Changelog

All notable changes to this project are documented here, ordered from most recent to earliest. Each entry captures the architectural decision, what was changed, and why.

---

## [Unreleased] — May 2026

### Added
- **Seniority preference UI redesign** — replaced "Same level" / "Step up" radio labels with "Find roles that match who I am now" / "Find roles that match who I'm becoming". Added dynamic context-aware callout cards that change based on the selected option, explaining the effect on both HyDE JD generation and the experience filter threshold. Early-career guidance (sub-1.5yr candidates) surfaced directly in the UI.
- **README overhaul** — full project page with tech stack badges, pipeline summary table, design decision explanations, project structure tree, quickstart, environment variable reference, test instructions, and deployment notes.
- **`docs/architecture.md`** — 400-line deep-dive covering every agent, every filter gate, the HyDE partition logic, fit score formula, cost model, configuration reference, and known limitations.

---

## [v1.4] — May 2026 · Pipeline quality gates

### Added
- **Overqualified flag** (`#49`) — ranker LLM now returns an `overqualified` boolean when the candidate is significantly more senior than the role requires. Surfaced as an info banner on the job card in the UI. Does not affect fit_score or rank — purely informational.
- **Scoring fallback gate** (`#50`) — jobs where the LLM returns a neutral/failed scoring response (all scores at default 0.0) are now explicitly dropped with `dropped_at: "scoring_fallback"` in the pipeline audit, rather than passing through and appearing as low-fit results.
- **Configurable `min_fit_score` gate** (`#51`) — a final noise floor (default 0.45) drops any job that cleared all filter gates but scored poorly across all dimensions. Configurable in `llm_config.yaml`. Set to 0.0 to disable.
- **Per-source discard rates** (`#52`) — `agent_metrics["ranker"]["source_discard_rates"]` now logs, per job source, how many jobs were retrieved vs. how many survived to final output. Surfaced in test_scripts and MLflow metrics. Useful for deciding which RapidAPI subscriptions to retain.

### Changed
- Audit priority ordering updated — `scoring_fallback` gate precedes `min_fit_score` in the drop sequence, ensuring fallback-scored jobs are correctly labelled in the audit trail.

---

## [v1.3] — May 2026 · Production hardening

### Added
- **GitHub Actions CI/CD pipeline** — on every push to `main`: builds `api-runtime` and `streamlit-runtime` Docker images from the shared multi-stage Dockerfile, pushes both to Amazon ECR, SSHs into EC2 and restarts with `docker-compose.prod.yml`. Layer cache shared across both builds via GitHub Actions cache.
- **MLflow observability** — per-agent latency, token usage, estimated cost (INR), and pipeline funnel metrics logged to MLflow on every run. `job_audit.json` artifact captures the complete per-job record.
- **Pipeline audit UI** — "Pipeline Audit" expander on the Streamlit results page renders the full per-job funnel: jobs at each gate, drop reasons, HyDE scores, fit scores.

### Fixed
- MLflow 3.x browser API compatibility — added `/ajax-api/` nginx location block for MLflow 3.x AJAX calls.
- LangGraph Postgres checkpointer — switched to autocommit connection for setup to avoid transaction state conflicts.
- `AsyncPostgresSaver` — replaced all sync `get_state` calls with `aget_state`; fixed async pool lifecycle in lifespan handler.

---

## [v1.2] — May 2026 · Architectural redesign (HyDE + filter refactor)

This version represents the largest single architectural change in the project. The keyword-based prefilter was replaced with a semantic HyDE system, four filter gates were added or redesigned, and the job source layer was refactored into a registry pattern.

### Added
- **HyDE dual-JD prefilter** (`#22–#26`) — Agent 5b generates two hypothetical job descriptions (JD1 domain-anchored, JD2 transferable-skills) via Claude Sonnet in parallel, then embeds all job JDs + both hypotheticals in a single Voyage AI batch call. Jobs are partitioned into Section 1 ("Roles in your domain") and Section 2 ("Broader opportunities") by comparing cosine similarity against JD1 vs. JD2. A single absolute floor (`hyde_min_floor=0.45`) drops noise-level matches. RRF fusion (dense + BM25 sparse) ranks jobs within each section before passing to the ranker.
- **Deduplication before HyDE** (`#17`) — moved `deduplicate()` from inside the ranker to before the HyDE node. Prevents Voyage AI from embedding duplicate JD texts, which wasted embedding budget and distorted RRF score distributions.
- **S1/S2 per-section caps** (`#40`) — configurable caps (S1: 40 jobs, S2: 20 jobs) applied within each section before the ranker. Jobs exceeding the cap are dropped by RRF rank, not randomly.
- **LLM-based job location extraction** (`#46`) — ranker scoring prompt now extracts the actual job location from JD text and uses it for the location filter, falling back to API metadata. More reliable than API-reported location fields.
- **india_accessible filter** — LLM judges whether a job is accessible to India-based remote candidates. Drops only when the JD explicitly requires US work authorisation or states no visa sponsorship. Benefit of the doubt given to jobs that don't mention restrictions.
- **Title relevance gate (S1-only)** (`#47`) — title relevance score from the LLM gates S1 inclusion (threshold: 0.4). Not applied to S2 — S2 is inherently transferable-skill-based, so strict title matching would defeat its purpose. S1-specific fallback logic relaxes the title gate when it is the binding constraint preventing any S1 results.
- **Job source registry pattern** (`#48`) — each job source is now a self-contained adapter in `agents/job_search/sources/`. Adding a new source requires only implementing the `fetch()` interface and registering it in `registry.py`. Sources can be enabled/disabled and configured (pages) in `llm_config.yaml` without touching orchestration code.
- **Section-aware profile caps** (`#21`) — `apply_profile_caps()` now applies caps independently within each HyDE section rather than across the combined pool, preserving S1/S2 balance.
- **Two-section results UI** (`#29`) — Streamlit results page renders S1 and S2 as distinct labeled sections with separate headers and descriptions. Fit score filter slider works across both sections simultaneously.

### Removed
- **Signals classifier** (`#44`) — entire hiring signals layer removed: NewsAPI fetching, RSS parsing, company watch list, `signals_classifier.py`, and all associated state fields and API routes. The layer added latency and cost without improving job ranking quality. Removed across 7 files.
- **LLM URL pruner** (`#45`) — replaced with HTTP-based `url_validator.py`. The LLM-based pruner was slower, more expensive, and less accurate than a direct HTTP HEAD/GET check. HTTP responses are ground truth for URL liveness; LLMs are not.
- **BM25 as primary prefilter gate** (`#16`) — `skill_prefilter` retired as a primary gate. BM25 keyword overlap was too sensitive to terminology mismatches (e.g. "ML Engineer" vs. "Machine Learning Engineer") and produced high false-negative rates. Replaced by HyDE semantic filtering.
- **`domain_preference` HITL field** (`#19`) — removed from the user-facing confirmation step. Domain preference is now implicit in the candidate's confirmed profile and the HyDE JD content, not an explicit dropdown.

### Changed
- **Experience filter thresholds recalibrated** (`#43`) — `step_up` threshold raised from 0.15 to 0.30 (was too permissive, letting through roles 3–4 levels above the candidate). `same_level` threshold remains 0.50.
- **Title relevance threshold raised** from 0.30 to 0.40.
- **HyDE JD prompts** (`#6–#8`) — removed hardcoded domain vocabulary lists, removed truncation of technical skills/tools/impact signals, added explicit breadth instruction. JDs now reflect the candidate's full profile rather than a filtered subset.
- **JD2 prompt engineered** (`#10`) — JD2 now explicitly avoids domain-specific language and focuses on transferable functional and technical skills. This sharpens the S1/S2 partition signal.

---

## [v1.1] — April 2026 · Core pipeline

### Added
- **Resume parser** (Agent 1, Claude Sonnet) — structured extraction of work history, education, skills, domain expertise, career trajectory, and ATS summary from PDF/DOCX.
- **Profile recommender** (Agent 2, Claude Haiku) — suggests 2–3 target job titles with seniority targets, confidence levels, match reasons, and search variant expansions.
- **Human-in-the-loop gate** — LangGraph `interrupt_before` pause between recommender and job search. User selects one confirmed profile via Streamlit UI. State persists across the interrupt via Postgres checkpointer.
- **Job search** (Agent 3) — parallel async queries to JSearch, Active Jobs DB, Techmap, Jobs Search API (RapidAPI), and RemoteOK. ~100–150 raw listings per run.
- **Ranker** (Agent 6, Claude Haiku) — per-job LLM scoring on experience, skills, domain, and education fit. Fit score formula with configurable weights. Gap analysis (experience gap, skill gaps, domain gap, education gap) returned at zero additional cost from the same scoring call.
- **LangSmith tracing** — `@traceable` on all LLM calls; run_name and metadata tags per pipeline phase.
- **Docker + Postgres checkpointer** — containerised stack with `docker-compose.yml`. LangGraph state persisted to Postgres via psycopg3 connection pool.
- **Streamlit frontend** — three-step UI: upload + preferences → profile confirmation → ranked results with fit score tiles, gap analysis expander, sparse JD and overqualified banners.

### Changed
- Switched ranker from Claude Sonnet to Claude Haiku — ~5× cost reduction with acceptable quality for structured scoring tasks.
- Experience calculator made deterministic — replaced LLM-based year estimation with date arithmetic from parsed work history.

---

## [v1.0] — April 2026 · Initial build

- Initial LangGraph pipeline: resume parser → recommender → job search (JSearch + RemoteOK) → ranker → results.
- FastAPI backend with session management.
- Basic Streamlit frontend.
- PostgreSQL state persistence via LangGraph checkpointer.
