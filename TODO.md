# Autonomous Job Finder — Master TODO List

_Generated after full pipeline audit (May 2026). Items ordered within each priority by dependency._

---

## ✅ DONE — Category A: Tier 1+2 Parameterization

All hardcoded constants moved to `llm_config.yaml`. No further action needed; requires Docker rebuild to deploy.

| Task | File(s) Changed | What |
|------|----------------|------|
| A1 | `llm_config.yaml`, `config_loader.py`, `ranker_agent.py` | `BATCH_SIZE`, `BATCH_DELAY` now from config |
| A2 | `llm_config.yaml`, `config_loader.py`, `normalizer.py` | `MIN_JD_CHARS`, `MIN_JD_WORDS`, `MAX_JD_CHARS` now from config |
| A3 | `llm_config.yaml`, `config_loader.py`, `resume_parser.py` | `MIN_TEXT_LENGTH` now from config; new `ResumeParserConfig` dataclass |

---

## P0 — Compliance: Enforce What the User Actually Asked For

These are the highest priority because they represent a direct gap between what the user inputs and what the system delivers. Post-rank filters are a thin addition (no LLM cost, no latency) with high user-facing correctness payoff.

### E1 — Remove salary fields from state entirely
**Files:** `core/state/session_state.py`, `agents/job_search/normalizer.py`, any UI output code that references salary  
**Impact:** Eliminates dead fields that store unreliable API data and create a false impression that salary filtering is in effect. Cleaner state, no risk of future code accidentally acting on junk data.  
**Effort:** Low — field removal + grep for all references  
**Risk:** None. Salary enforcement is acknowledged as out of scope due to API data quality.  
**Remove from:** `UserPreferences.salary_min/max/currency`, `RawJob.salary_min/max`, `RankedJob.salary_min/max`

---

### E2 — Add post-rank work_type filter
**Files:** `agents/ranker/ranker_agent.py` (end of `run_ranker_agent`) or new thin node in `core/graph.py`  
**Impact:** Directly enforces `UserPreferences.work_type`. A user who asked for remote jobs will no longer see on-site listings that slipped through unreliable API-level filters. Drop any job where `job.work_type` does not match `preferences.work_type`, unless `preferences.work_type == "any"`.  
**Effort:** Low — ~15 lines  
**Risk:** Low. If normalizer has mapped work_type inconsistently, some valid jobs may drop. Mitigate by mapping `"on-site"` and `"office"` as equivalent before comparing.  
**Note:** Log dropped count for observability.

---

### E3 — Add post-rank location filter
**Files:** Same location as E2  
**Impact:** Enforces `UserPreferences.location`. Drops jobs from cities the user did not ask for. API-level location filters are unreliable (especially for broad searches like "India"); this is the authoritative enforcement layer.  
**Effort:** Low — ~15 lines  
**Risk:** Medium. Location strings from APIs are free-text and inconsistent ("Bengaluru" vs "Bangalore", "Delhi" vs "Delhi NCR"). Must use fuzzy/substring matching, not exact equality. Use the same `_resolve_city` alias map that already exists in `techmap.py` as a reference.  
**Note:** If `preferences.location` is blank or "anywhere", bypass filter entirely.

---

## P1 — API Filter Cleanup: Stop Silently Dropping Valid Jobs

These filters are applied before any job enters the pipeline. A relevant job dropped here is invisible to every downstream agent — the ranker never gets a chance to score it. Removing them costs nothing (the URL pruner and ranker handle quality downstream) and materially increases recall.

### B1 — Remove `employment_types=FULLTIME` from JSearch
**File:** `agents/job_search/sources/jsearch.py`  
**Impact:** JSearch currently hard-filters to full-time only at the API level. Contract, part-time, and mislabelled full-time roles (common for Indian postings) are silently dropped. The ranker's title relevance and experience filters handle eligibility — this API filter is redundant and harmful.  
**Effort:** Trivial — remove one param key  
**Risk:** Slightly more jobs enter the pipeline (more LLM calls in ranker). Acceptable given batch caps.

---

### B2 — Remove `job_type="fulltime"` from Jobs Search API
**File:** `agents/job_search/sources/jobs_search_api.py`  
**Impact:** Same reason as B1. Different source, same problem.  
**Effort:** Trivial  
**Risk:** Same as B1.

---

### B3 — Make `hours_old` configurable in Jobs Search API
**File:** `agents/job_search/sources/jobs_search_api.py`, `core/config/llm_config.yaml`  
**Impact:** Currently hardcoded as 72h for remote jobs and 168h/336h for others. This silently excludes jobs posted earlier in the week depending on when the session runs. Making it a config value allows easy tuning without code deploy.  
**Effort:** Low — add `hours_old: 168` to yaml, wire into source adapter  
**Risk:** Older postings may already be filled. But ranker's recency scoring deprioritizes them anyway — better to have them scored than to have them absent.

---

### B4 — Unquote `title_filter` in Active Jobs DB and LinkedIn Jobs API
**Files:** `agents/job_search/sources/active_jobs_db.py`, `agents/job_search/sources/linkedin_jobs.py`  
**Impact:** Both currently pass `title_filter='"ML Engineer"'` (with embedded quotes), requesting exact-phrase match. This drops all jobs with titles like "Senior ML Engineer", "ML Engineer II", "Applied ML Engineer". Removing the inner quotes switches to broad match and dramatically improves recall.  
**Effort:** Trivial — remove `f'"..."'` wrapping, leave as `profile_title` directly  
**Risk:** Slightly lower precision (more non-matching titles enter pipeline). The ranker's title relevance filter (≥0.5 threshold) handles this correctly.

---

## P2 — URL Pruner Improvements: Consistent Quality Gating

### C1 — Extend URL Pruner to all job sources
**File:** `agents/pruner/url_pruner.py`  
**Impact:** Currently only JSearch jobs go through the URL quality LLM classifier. All other sources (Techmap, Active Jobs DB, LinkedIn Jobs, RemoteOK, Jobs Search API) bypass it entirely. As we add more sources, this asymmetry grows. Extending to all sources ensures aggregator URLs and low-quality listings from any source are caught.  
**Effort:** Low — remove the `source == "jsearch"` bifurcation; send all jobs to `_classify_urls`  
**Risk:** Slightly higher Haiku cost per session (more URLs per call). Still one LLM call total, ~$0.003 per session at current job volumes. Acceptable.  
**Note:** JSearch already has its own Signal 1/2/3 source filter at fetch time, so JSearch jobs are double-quality-checked. That's fine — the pruner is a cheap second pass.

---

### C2 — Document JSearch source filter as a fetch optimisation, not a quality gate
**File:** `agents/job_search/sources/jsearch.py` (inline comments)  
**Impact:** The Signal 1/2/3 filter in jsearch.py is currently framed as a quality gate. Reframing it as "fetch optimisation that reduces upstream noise before the URL pruner does the real classification" prevents future developers from removing the URL pruner thinking JSearch is already clean. No code change — comment/docstring update only.  
**Effort:** Trivial  
**Risk:** None.

---

## P3 — New Job Sources: Increase Coverage

Both sources are free-tier RapidAPI products. Deduplication in the ranker makes adding sources safe — overlapping jobs are collapsed before LLM scoring. Confirm API keys before implementing.

### D1 — Add jaypat87 Job Search API (job-search15) adapter
**File:** `agents/job_search/sources/jobs_search_api_v2.py` (new file)  
**Impact:** Additional coverage for LinkedIn/Indeed/Glassdoor/ATS-sourced jobs. Broad search with no exact-title matching. Good complement to Active Jobs DB which is ATS-heavy.  
**Effort:** Medium — new source adapter + wire into `job_search_agent.py`  
**Risk:** Low. Dedup handles overlap. Watch monthly request caps.

---

### D2 — Add Jobisite Job Search API (job-search38) adapter
**File:** `agents/job_search/sources/jobisite.py` (new file)  
**Impact:** Additional coverage; Jobisite aggregates from direct company career pages, which are high-quality sources the URL pruner would retain anyway.  
**Effort:** Medium — same as D1  
**Risk:** Same as D1.

---

## Known Limitations (No Action)

| Limitation | Reason not tackled |
|------------|--------------------|
| Salary min/max enforcement | API-returned salary data is too unreliable to act on. Removed from state entirely (E1). Revisit only if a source with verified salary data is added. |
| `seniority_target` from recommender not used in ranker | Ranker uses `seniority_preference` from UserPreferences only. Recommender's `is_stretch=True` flag is not wired to a tighter experience gate. Low priority — seniority_preference already captures intent. |
| Work type from APIs is unreliable pre-filter | Addressed by removing API-level work_type filters (B1, B2) and adding post-rank enforcement (E2). |

---

## Implementation Order

```
E1 → E2 → E3   (compliance — highest user-facing impact, low effort)
B4 → B1 → B2 → B3   (API filter cleanup — increases recall, no cost)
C1 → C2   (pruner — quality consistency)
D1 → D2   (new sources — needs API keys confirmed first)
```

Docker rebuild required once any Category A changes are deployed.
