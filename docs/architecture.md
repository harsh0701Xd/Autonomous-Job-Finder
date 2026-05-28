# Pipeline Architecture

This document describes the full design of the Autonomous Job Finder pipeline — every agent, every filter gate, every design decision, and the reasoning behind each choice.

---

## Pipeline overview

```
                         ┌─────────────────────────────────────┐
                         │           SessionState               │
                         │  (Pydantic model, persisted to       │
                         │   Postgres via LangGraph checkpointer)│
                         └─────────────────────────────────────┘
                                          │
          ┌───────────────────────────────▼────────────────────────────────┐
          │                         LangGraph Graph                        │
          │                                                                │
          │  START                                                         │
          │    │                                                           │
          │    ▼                                                           │
          │  [Agent 1] Resume Parser         Claude Sonnet                │
          │    │                                                           │
          │    ▼                                                           │
          │  [Agent 2] Profile Recommender   Claude Haiku                 │
          │    │                                                           │
          │    ▼                                                           │
          │  [INTERRUPT] User Confirmation   ← Human selects one profile  │
          │    │                                                           │
          │    ▼                                                           │
          │  [Agent 3] Job Search            5 live APIs in parallel      │
          │    │                                                           │
          │    ▼                                                           │
          │  [Agent 4] URL Validator         HTTP-only, no LLM            │
          │    │                                                           │
          │    ▼                                                           │
          │  [Agent 5a] Dedup                In-memory fingerprint dedup  │
          │    │                                                           │
          │    ▼                                                           │
          │  [Agent 5b] HyDE Prefilter       Claude Sonnet + Voyage AI    │
          │    │                                                           │
          │    ▼                                                           │
          │  [Agent 6] Ranker                Claude Haiku × N jobs        │
          │    │                                                           │
          │    ▼                                                           │
          │  [Finalise] Metrics + MLflow     No LLM                       │
          │    │                                                           │
          │   END                                                          │
          └────────────────────────────────────────────────────────────────┘
```

**Typical run:** 100–150 jobs retrieved → 20–30 after HyDE → 10–20 after ranker filters → 8–15 displayed.

**Typical wall-clock time:** 2–3 minutes end-to-end (dominated by job search API latency and ranker LLM calls).

---

## Shared state: SessionState

Every agent in the pipeline reads from and writes to a single `SessionState` Pydantic object (`core/state/session_state.py`). LangGraph serialises this object and persists it to Postgres at every node boundary via the Postgres checkpointer.

This means:
- State survives server restarts between the confirmation interrupt and the job search
- Any agent can be re-run in isolation by loading the checkpointed state
- The pipeline audit trail (per-job drop reasons and scores) accumulates incrementally as state flows through nodes

Key state fields written at each stage:

| Stage | Fields written |
|---|---|
| Parser | `candidate_profile`, `parse_failed`, `resume_raw_text` |
| Recommender | `suggested_profiles` |
| Confirmation | `confirmed_profiles` |
| Job Search | `raw_jobs`, `pipeline_audit` (initial entries) |
| URL Validator | `pipeline_audit` (url_pruner drops) |
| Dedup | `raw_jobs` (deduplicated), `pipeline_audit` (dedup drops) |
| HyDE | `raw_jobs` (with hyde_section, jd1/jd2 scores), `hypo_jd1`, `hypo_jd2`, `pipeline_audit` |
| Ranker | `ranked_jobs`, `pipeline_audit` (scores + filter drops) |
| Finalise | `session_metrics`, `results_ready` |

---

## Agent 1: Resume Parser

**File:** `agents/parser/resume_parser.py`
**Model:** Claude Sonnet
**Input:** Raw PDF or DOCX bytes
**Output:** `CandidateProfile` — structured representation of the candidate

The parser extracts:
- Work experience history (title, company, start/end dates, role type: full_time / internship / other)
- Education (degree, institution, year)
- Skills (technical, tools, soft)
- Domain expertise (industry verticals inferred from experience)
- Career trajectory classification: ascending / lateral / pivot / re-entry
- Career gaps

The most important output is the **ATS summary** — a Claude-written narrative paragraph that synthesises the candidate's experience into a dense, searchable paragraph. This is the primary input to the ranker scoring prompt. It is written by the parser rather than the ranker to avoid re-processing the full resume text in every downstream scoring call.

**Experience split:** The parser separates `years_experience_full_time` from `years_experience_other` (internships, freelance, other). Both are passed to downstream agents but only full-time experience is used in the experience filter threshold checks.

---

## Agent 2: Profile Recommender

**File:** `agents/recommender/profile_recommender.py`
**Model:** Claude Haiku
**Input:** `CandidateProfile`, `UserPreferences`
**Output:** 2–3 `SuggestedProfile` objects

The recommender produces target job titles that the pipeline will search for. Each suggested profile includes:
- `title`: the canonical job title (e.g. "Data Scientist")
- `seniority_target`: junior / mid / senior / lead / principal
- `confidence`: high / medium / low
- `match_reason`: displayed to the user in the confirmation UI
- `is_stretch`: whether this is a reach profile given the candidate's experience
- `search_variants`: alternative phrasings used to expand job search queries (e.g. "ML Engineer" → ["Machine Learning Engineer", "AI Engineer", "Applied ML Scientist"])

**Design note:** `search_variants` are never shown to the user — they exist purely to widen job API queries. The canonical `title` is what's displayed and what the ranker scores against.

---

## Human-in-the-loop confirmation gate

**Implementation:** LangGraph `interrupt_before=["user_confirmation"]`

After the recommender runs, the graph pauses before the `user_confirmation` node. The FastAPI layer detects this interrupt, reads the suggested profiles from checkpointed state, and returns them to the Streamlit frontend in the `/sessions/{id}/resume` response (`status: "awaiting_confirmation"`).

The user selects one profile. The frontend calls `/sessions/{id}/confirm`, which resumes the graph by passing a `Command(resume={"selected_titles": [...], "custom_profiles": []})` payload. LangGraph resumes from exactly where it paused, with the full prior state intact.

**Why a single-profile confirmation:** Multi-profile selection was originally supported. It was reduced to one profile because multi-profile runs multiply job search volume, HyDE embedding cost, and ranker LLM calls without proportional result quality improvement. A single confirmed profile produces a tighter, more coherent result set.

---

## Agent 3: Job Search

**File:** `agents/job_search/job_search_agent.py`
**Model:** None (API calls only)
**Input:** `confirmed_profiles`, `UserPreferences.location`
**Output:** `raw_jobs[]` (~100–150 listings)

The job search agent queries five sources in parallel using `asyncio.gather`:

| Source | Type | Notes |
|---|---|---|
| JSearch | RapidAPI | Large general index; primary source |
| Active Jobs DB | RapidAPI | Real-time listings, India-heavy |
| Techmap | RapidAPI | Structured data, strong metadata |
| Jobs Search API | RapidAPI | Supplementary general source |
| RemoteOK | Direct API | Remote-only listings; filtered for India accessibility downstream |

**Registry pattern:** Each source is a self-contained adapter in `agents/job_search/sources/`. Adding a new source requires only implementing the `fetch()` interface and registering it in `agents/job_search/registry.py`. The orchestrator calls all enabled sources without knowing their implementation details.

**Query construction:** For each confirmed profile, the agent generates queries using the canonical title plus its `search_variants`. This expands recall without requiring the user to know alternative job title phrasings.

**Pipeline audit initialisation:** Every raw job gets an entry in `pipeline_audit` at this stage with its identity fields and `status: "passed"`. Subsequent stages update this entry as jobs are dropped or scored.

---

## Agent 4: URL Validator

**File:** `core/url_validator.py`
**Model:** None (HTTP HEAD/GET only)
**Input:** `raw_jobs[]`
**Output:** Filtered `raw_jobs[]`

Drops jobs whose apply URLs are dead or redirect to aggregator homepages. Uses async `httpx` with a concurrency semaphore. Classifies each URL as:

- **Alive** (`200`, `201`, `301` with valid destination, `403`): kept
- **Dead** (`404`, `410`, `400`, `5xx`): dropped, logged as `url_pruner` in audit
- **Aggregator redirect**: domain-level check against a known list of job aggregators whose "apply" links redirect to their homepage rather than the actual posting — dropped

**Why HTTP-only:** The original implementation used an LLM to judge URL quality. This was replaced with a pure HTTP check because: it's faster (no LLM latency), cheaper (no token cost), and more accurate (LLMs hallucinate URL status; HTTP responses are ground truth). The only thing lost is nuanced judgment about content — which the HyDE floor and ranker handle downstream.

---

## Agent 5a: Deduplication

**File:** `agents/ranker/ranker_agent.py` (`deduplicate()` function)
**Model:** None (in-memory)
**Input:** URL-validated `raw_jobs[]`
**Output:** Deduplicated `raw_jobs[]`

Deduplication uses a fingerprint: `company + normalised_title` (after stripping seniority prefixes like "Senior", "Jr.", "Lead"). When duplicates are found, the job with the **longer JD text** is kept (more signal for embedding and scoring), and the `matched_profile` strings from all duplicates are merged.

**Why before HyDE:** Dedup was originally inside the ranker. It was moved before the HyDE node so that Voyage AI does not embed duplicate JD texts. Embedding 4 identical Accenture listings wastes embedding budget and distorts RRF score distributions.

---

## Agent 5b: HyDE Prefilter

**File:** `agents/hyde/hyde_agent.py`
**Model:** Claude Sonnet (JD generation) + Voyage AI `voyage-3` (embeddings)
**Input:** Deduplicated `raw_jobs[]`, `CandidateProfile`, `UserPreferences`
**Output:** `raw_jobs[]` partitioned into S1/S2 with embedding scores; irrelevant jobs dropped

This is the core semantic filtering stage. It has three sub-steps:

### Step 1: Dual hypothetical JD generation

Two hypothetical job descriptions are generated in parallel via Claude Sonnet:

**JD1 — Domain-anchored:** Written as if for a candidate whose primary value is their domain expertise. Emphasises the specific industry vertical, domain-specific tools, and the type of problems they've solved. For a credit risk analyst at a fintech, JD1 describes a risk analyst role at a financial services company.

**JD2 — Transferable-skills:** Written to capture the candidate's analytical, technical, and functional skills independent of domain. Emphasises cross-cutting capabilities (Python, SQL, statistical modelling, stakeholder communication) without domain language. For the same candidate, JD2 describes a data analyst or analytics engineer role that could be in any industry.

The `seniority_preference` is passed into both prompts — `step_up` frames the JDs one level above the candidate's current title; `same_level` frames them at their current level.

### Step 2: Batch embedding

All real job JDs, JD1, and JD2 are embedded in a **single Voyage AI batch call**. This is a deliberate cost and latency optimisation — one API call for 100+ texts vs. 100+ individual calls.

**Embedding model:** `voyage-3` (general-purpose). No domain-specific models are used. Domain specificity is achieved through HyDE content, not model choice. This is a hard constraint: `voyage-finance-2` or `voyage-law-2` would improve recall for specific domains but would require model selection logic per candidate profile, adding complexity and coupling the pipeline to domain classification.

### Step 3: Section partitioning

For each job, two cosine similarities are computed: `jd1_score = cosine(job_embedding, JD1_embedding)` and `jd2_score = cosine(job_embedding, JD2_embedding)`.

Partition logic:
```
floor = hyde_min_floor (default: 0.45)

If max(jd1_score, jd2_score) < floor:
    → Dropped (below noise floor)
Elif jd1_score >= jd2_score:
    → Section 1: "Roles in your domain"
Else:
    → Section 2: "Broader opportunities"
```

**Why delta-based partition (not two independent thresholds):** Absolute cosine similarity scores are session-dependent — they shift based on profile vocabulary, job pool diversity, and embedding space density. The relative JD1 vs JD2 signal is self-calibrating: whichever hypothetical JD is semantically closer to a real job determines its section. A single absolute floor eliminates noise regardless of which JD the job scored higher against.

### Step 4: RRF scoring and section caps

Within each section, jobs are ranked by **Reciprocal Rank Fusion (RRF)** combining the cosine similarity rank with a BM25 keyword rank computed against the relevant hypothetical JD (JD1 for S1, JD2 for S2). RRF combines dense and sparse signals without requiring manual weight tuning.

Section caps (configurable in `llm_config.yaml`):
- **S1 cap:** top 40 jobs by RRF score
- **S2 cap:** top 20 jobs by RRF score

Jobs exceeding the cap are dropped with `dropped_at: "hyde_section_cap"` in the audit trail. Excess S2 jobs are dropped first because S2 contains transferable-skill matches (lower confidence) and the cap protects the ranker from being flooded by bulk postings from a single company.

**Fallback:** If Section 1 is empty after the first partition pass (all jobs scored higher against JD2 or all below the floor), the floor is lowered to `fallback_floor` (default: 0.40) and the partition is retried once. If S1 is still empty, all jobs above the fallback floor are passed through as S1. The `fallback_activated` flag is set in state.

---

## Agent 6: Ranker

**File:** `agents/ranker/ranker_agent.py`
**Model:** Claude Haiku (one call per job, up to `semaphore_size=4` concurrent)
**Input:** HyDE-filtered `raw_jobs[]` (S1 + S2, ~20–60 jobs), `CandidateProfile`, confirmed profile title
**Output:** `ranked_jobs[]` with fit scores and gap analysis

### Scoring

Each job is scored by a single Claude Haiku call that reads:
- The candidate's ATS summary (from Agent 1)
- The full job description
- The target role title the candidate confirmed

The LLM returns a JSON object with six fields:

| Field | Range | Used in fit score | Notes |
|---|---|---|---|
| `experience_match` | 0–1 | Yes (weight: 0.40–0.50) | Candidate years/level vs. JD requirement |
| `skill_match` | 0–1 | Yes (weight: 0.30) | Technical skill alignment |
| `domain_match` | 0–1 | Yes (weight: 0.20) | Industry/domain background alignment |
| `education_match` | 0–1 or null | Yes (weight: 0.10, if required) | null when JD has no edu requirement |
| `title_relevance` | 0–1 | No (filter gate) | How closely job title matches confirmed profile |
| `india_accessible` | bool | No (filter gate) | False only when JD requires US work auth |
| `job_location_extracted` | string or null | No (filter gate) | Extracted location from JD text |

The LLM also returns free-text gap analysis:
- `scoring_notes`: one-sentence overall assessment
- `experience_gap`: e.g. "Requires 5yr, candidate has 0.8yr"
- `skill_gaps[]`: e.g. ["Kubernetes", "dbt"]
- `domain_gap`: e.g. "Role in healthcare; background in fintech"
- `education_gap`: e.g. "Requires Master's; candidate has Bachelor's"
- `sparse_jd`: boolean — thin JD with insufficient detail to score confidently
- `overqualified`: boolean — candidate is significantly more senior than role requires

These are displayed on job cards in the UI at zero additional cost — they come from the same scoring call.

### Fit score formula

```python
# When JD has explicit education requirement:
fit_score = 0.40 * experience + 0.30 * skill + 0.20 * domain + 0.10 * education

# When JD has no education requirement:
fit_score = 0.50 * experience + 0.30 * skill + 0.20 * domain
```

Weights are configurable in `llm_config.yaml` under `ranker.weights_with_education` and `ranker.weights_without_education`. The config loader validates that both weight sets sum to 1.0.

`recency` has weight 0.00 — it is computed and displayed on job cards but has no effect on ranking. This is intentional: penalising older postings would disadvantage companies that post infrequently, and the information value of recency is already partially captured by the job search agent's sort order.

### Filter gates (applied after scoring)

Gates are applied in this order:

**1. Experience filter (`exp_filter`)**
Drops jobs where `experience_score < threshold`.
- `same_level` preference → threshold = **0.5**
- `step_up` preference → threshold = **0.3**

This is the highest-volume drop gate. The threshold difference between `same_level` and `step_up` is the primary mechanism by which seniority preference affects result quantity and quality.

**2. Title relevance filter (`title_filter`, S1 only)**
Drops S1 jobs where `title_relevance < 0.4`. Not applied to S2 jobs — S2 is inherently transferable-skill-based, so strict title matching would defeat its purpose.

**3. Location filter (`location_filter`)**
Drops jobs where the extracted or metadata location doesn't match the user's specified city. Uses `job_location_extracted` (from JD text) first, falling back to the raw API location metadata.

**4. India accessibility filter (`india_accessible_filter`)**
Drops remote jobs where the JD explicitly requires US work authorisation, US residency, or states no visa sponsorship. The LLM is instructed to give the benefit of the doubt to jobs that simply don't mention location restrictions — only explicit exclusions trigger this filter.

**5. Minimum fit score gate (`min_fit_score`)**
Drops any job with `fit_score < 0.45`. This is a final noise floor — jobs that cleared all other gates but scored poorly across all dimensions are removed here.

### Final ordering

Within each section, jobs are sorted by `fit_score` descending. Section 1 jobs are presented first, then Section 2 — regardless of the relative fit scores between sections. This is a known limitation: a 0.82-fit S2 job will display below a 0.62-fit S1 job. Future versions will use global fit_score ordering with section label as metadata.

---

## Observability: pipeline audit trail

Every job that enters the pipeline gets an entry in `session.pipeline_audit` (keyed by `job_id`). This entry is updated by each stage as the job flows through. At the end of the pipeline, the full audit trail is:

1. Logged to MLflow as `job_audit.json` artifact
2. Exposed through the Streamlit UI's "Pipeline Audit" expander on the results page

Each audit entry contains:
- Identity: `title`, `company`, `source`, `jd_word_count`, `matched_profile`
- Status: `"passed"` or `"dropped"`
- `dropped_at`: gate name (url_pruner / dedup / hyde_floor / hyde_section_cap / exp_filter / title_filter / location_filter / india_accessible_filter / min_fit_score)
- `drop_reason`: human-readable string
- HyDE fields: `jd1_emb_score`, `jd2_emb_score`, `hyde_section`, `rrf_jd1`, `rrf_jd2`
- Ranker fields: `fit_score`, `experience_score`, `skill_score`, `domain_score`, `domain_score`, `title_relevance`, `sparse_jd`, `overqualified`
- `final_rank`: 1-based rank in the presented output (null if dropped)

This trail makes the pipeline fully explainable — for any result set, you can trace exactly why each job was surfaced or eliminated.

---

## Cost model (per pipeline run)

| Stage | Model | Approximate cost |
|---|---|---|
| Resume parser | Claude Sonnet | ~$0.02–0.04 |
| Profile recommender | Claude Haiku | ~$0.001 |
| HyDE JD generation | Claude Sonnet | ~$0.01–0.02 |
| HyDE embeddings (100 jobs) | Voyage AI voyage-3 | ~$0.002 |
| Ranker (25 jobs × Haiku) | Claude Haiku | ~$0.02–0.05 |
| **Total** | | **~$0.05–0.12 per run** |

The ranker is the cost ceiling. The HyDE prefilter's purpose is to reduce ranker inputs from ~120 to ~25, cutting ranker cost by ~5×. Before HyDE, scoring 120 jobs with Sonnet was the original design — this was replaced because the per-run cost was ~$0.50–1.00 and latency was 4–5 minutes.

---

## Configuration

All tunable parameters live in `core/config/llm_config.yaml`. Key sections:

```yaml
hyde_prefilter:
  hyde_min_floor: 0.45        # absolute similarity floor (below = dropped)
  fallback_floor: 0.40        # retry floor when S1 is empty
  s1_max_jobs: 40             # cap on Section 1 jobs passed to ranker
  s2_max_jobs: 20             # cap on Section 2 jobs passed to ranker

ranker:
  weights_with_education:
    experience: 0.40
    skill: 0.30
    domain: 0.20
    education: 0.10
  weights_without_education:
    experience: 0.50
    skill: 0.30
    domain: 0.20
  min_experience_score_same: 0.50   # exp_filter threshold for same_level
  min_experience_score_step_up: 0.30  # exp_filter threshold for step_up
  min_title_relevance: 0.40          # title_filter threshold (S1 only)
  min_fit_score: 0.45               # final noise floor
  semaphore_size: 4                  # max concurrent Haiku calls
```

---

## Adding a new job source

The job search agent uses a registry pattern. To add a new source:

1. Create `agents/job_search/sources/my_source.py` implementing the `fetch(profile, location, pages) -> list[RawJob]` interface
2. Register it in `agents/job_search/registry.py`
3. Add an entry to `llm_config.yaml` under `job_search.sources` with `enabled: true` and `pages: N`

No other files need to change. The orchestrator discovers all registered sources automatically.

---

## Known limitations and future work

**S1-first ordering:** Final results sort S1 jobs above S2 by section, not by absolute fit score. A high-fit S2 job ranks below a lower-fit S1 job. Fix: global fit_score sort with section label as metadata.

**india_accessible false negatives:** The LLM occasionally drops India-located jobs that contain US work authorisation boilerplate (copied from standard JD templates). Fix: pass extracted job location into the scoring prompt and instruct the model to discount US auth language when the role is explicitly India-located.

**S2 cap quality for generalist profiles:** For broad profiles (e.g. "Data Analyst"), most jobs land in S2. The 20-slot S2 cap gets filled by senior/engineering-tilted roles with higher RRF scores, crowding out entry-level matches. Fix: per-seniority sub-limit within the S2 pool, or RRF penalty for senior-titled roles when the candidate is early-career.

**Per-company S2 pollution:** Bulk postings from a single company (e.g. 8+ near-identical consulting roles) can monopolise S2 cap slots. Fix: per-company cap of 2–3 slots within the S2 pool before section-level ranking.
