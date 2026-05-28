# Test Scripts

Live integration tests for the Autonomous Job Finder pipeline. Each script exercises one agent end-to-end against real APIs — they are not unit tests with mocks.

**All scripts require valid API keys in `.env` at the project root.**

---

## Prerequisites

```bash
# From project root — activate the virtual environment
.venv\Scripts\activate        # Windows
source .venv/bin/activate      # macOS / Linux

# Confirm keys are set
python -c "import dotenv; dotenv.load_dotenv(); import os; print(os.getenv('ANTHROPIC_API_KEY')[:8])"
```

Required keys:

| Key | Used by |
|---|---|
| `ANTHROPIC_API_KEY` | All LLM agents (parser, recommender, HyDE, ranker) |
| `VOYAGE_API_KEY` | HyDE prefilter (embedding) |
| `RAPIDAPI_KEY` | Job search (JSearch, Active Jobs DB, Techmap, Jobs Search API) |

---

## Scripts and what they validate

### 1. `test_resume_parser.py`
**Agent tested:** Resume Parser (Agent 1, Claude Sonnet)

Parses a hardcoded resume text and prints the full `CandidateProfile` output:
- Structured work history with role types and durations
- Years of full-time vs. other experience
- Skills (technical, tools, soft)
- Domain expertise and career trajectory
- ATS summary (the narrative paragraph used by the ranker)
- Token usage and latency

**What to check:** ATS summary quality, correct experience year calculation, domain expertise extraction.

---

### 2. `test_profile_recommender.py`
**Agent tested:** Profile Recommender (Agent 2, Claude Haiku)

Runs the recommender against a hardcoded `CandidateProfile` and `UserPreferences` and prints:
- 2–3 suggested profiles with title, seniority target, confidence, match reason
- `is_stretch` flag for reach profiles
- `search_variants` list (used to widen job search queries)
- Token usage and latency

**What to check:** Seniority calibration (are suggested profiles appropriate for the candidate's experience?), search variant quality and breadth.

---

### 3. `test_job_search.py`
**Agent tested:** Job Search (Agent 3, multi-source)

Searches live job APIs for two hardcoded confirmed profiles and prints:
- Total jobs retrieved per source (JSearch, Active Jobs DB, Techmap, Jobs Search API, RemoteOK)
- Total jobs per profile
- Sample of retrieved job titles, companies, and locations

**What to check:** Source health (any sources returning 0 results may indicate API key or subscription issues), geographic coverage, title relevance of raw results.

---

### 4. `test_url_validator.py`
**Agent tested:** URL Validator (Agent 4, HTTP-based)

Runs a fixed set of 8 test URLs covering known cases — live pages, 404, 410, aggregator redirects, 403 — and prints pass/drop status with expected vs. actual comparison.

**What to check:** All expected outcomes match actual. A mismatch indicates a change in the validator's domain list or HTTP status handling.

---

### 5. `test_hyde_prefilter.py`
**Agent tested:** HyDE Prefilter (Agent 5b, Claude Sonnet + Voyage AI)

Runs a live job search → deduplication → HyDE prefilter and prints:
- JD1 and JD2 previews (first 300 chars of each hypothetical JD)
- S1 and S2 job counts after partition
- HyDE floor used (normal or fallback)
- Whether the fallback was activated
- Top 5 jobs per section by RRF score with their `jd1_emb_score` and `jd2_emb_score`

**What to check:** JD1 should read as domain-specific (industry vocabulary, domain tools). JD2 should read as domain-neutral (transferable skills, functional language). S1/S2 partition should reflect the content of each section — domain-matched jobs in S1, transferable-skill matches in S2. If S1 is empty and fallback activated, check whether the candidate profile is being framed too narrowly.

---

### 6. `test_ranker.py`
**Agent tested:** Full chain — Job Search → URL Validator → Dedup → HyDE → Ranker (Agent 6)

The most comprehensive test. Runs the complete post-confirmation pipeline and prints:
- Top 20 ranked jobs with fit score, sub-scores (experience, skill, domain, education if applicable), and section (S1/S2)
- Skill gaps for each job
- `sparse_jd` and `overqualified` flags where set
- Per-source discard rates: jobs retrieved vs. jobs surviving to final output, per API source

**What to check:** Fit score distribution (top results should be 0.65+), S1/S2 balance, experience filter drops (if too many, consider switching to `step_up` preference), per-source discard rates (sources with <10% survival rate may not be worth the API cost).

---

## Recommended run order

Run sequentially — each script validates one stage and the output helps diagnose issues in downstream stages:

```bash
python test_scripts/test_resume_parser.py
python test_scripts/test_profile_recommender.py
python test_scripts/test_job_search.py
python test_scripts/test_url_validator.py
python test_scripts/test_hyde_prefilter.py
python test_scripts/test_ranker.py
```

`test_ranker.py` subsumes `test_hyde_prefilter.py` and `test_job_search.py` — run it last as a full end-to-end sanity check.

---

## Typical run times

| Script | Typical duration |
|---|---|
| `test_resume_parser.py` | 15–25s |
| `test_profile_recommender.py` | 5–10s |
| `test_job_search.py` | 20–40s |
| `test_url_validator.py` | 10–20s |
| `test_hyde_prefilter.py` | 45–90s |
| `test_ranker.py` | 90–180s |

---

## Notes

- These are **live integration tests** — they consume real API quota on every run. Expect small costs (~$0.05–0.15 total for a full sequence).
- Job search results vary by run — the job market is live. Fit scores and rankings will differ between runs even for the same candidate profile.
- If `test_url_validator.py` shows unexpected mismatches, check whether a previously-dead URL has come back online or a previously-live URL has gone down — expected outcomes in the script may need updating.
- `test_ranker.py` prints `source_discard_rates` at the end. A source consistently contributing 0 final jobs across multiple runs is a signal to check its subscription status or disable it in `llm_config.yaml`.
