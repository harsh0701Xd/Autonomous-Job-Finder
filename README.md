# Autonomous Job Finder

An end-to-end multi-agent AI pipeline that reads a candidate's resume, understands their profile, and returns a ranked, scored list of real open job listings — all in under 3 minutes.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)
![Claude](https://img.shields.io/badge/Claude-Sonnet%20%2F%20Haiku-blueviolet)
![Voyage AI](https://img.shields.io/badge/Voyage%20AI-voyage--3-6A5ACD)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-EC2%20%2B%20ECR-FF9900?logo=amazonaws&logoColor=white)

---

## What it does

Upload a resume. The system parses it, suggests matching job profiles, lets you confirm one, then searches multiple live job APIs, semantically filters and scores every listing against your resume, and returns a ranked, sectioned results page — all without touching a single search box.

```
Resume PDF  →  Parse  →  Recommend profiles  →  [Human confirms]
            →  Search 5 job APIs  →  Validate URLs  →  Deduplicate
            →  HyDE semantic filter  →  LLM scoring  →  Ranked results
```

---

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full deep-dive.

**At a glance:**

| Stage | Agent | Model | What it does |
|---|---|---|---|
| Resume parsing | Parser | Claude Sonnet | Extracts structured profile + ATS summary |
| Profile recommendation | Recommender | Claude Haiku | Suggests 2–3 target job titles with search variants |
| Human-in-the-loop | — | — | LangGraph interrupt; user confirms one profile |
| Job search | Job Search | APIs only | Queries 5 sources in parallel (~100–150 raw listings) |
| URL validation | URL Pruner | HTTP only | Drops dead links, 404s, aggregator redirects |
| Deduplication | Dedup | In-memory | Removes same-company/title duplicates before embedding |
| Semantic filter | HyDE Prefilter | Claude Sonnet + Voyage AI | Generates two hypothetical JDs, embeds all listings, partitions into two sections |
| Scoring | Ranker | Claude Haiku | Scores each job on experience, skills, domain, education fit |
| Observability | Finalise | — | Aggregates metrics, logs to MLflow |

---

## Key design decisions

**HyDE dual-JD architecture** — Instead of keyword matching, the system generates two hypothetical job descriptions from the candidate's resume: one domain-anchored (JD1) and one transferable-skills-focused (JD2). Every real job listing is embedded alongside both and partitioned by cosine similarity. Jobs closer to JD1 become *"Roles in your domain"*; jobs closer to JD2 become *"Broader opportunities"*. This produces semantic relevance without any hard-coded keyword rules.

**Resume-agnostic embedding** — Only `voyage-3` (general-purpose) is used for embeddings. No domain-specific models (`voyage-finance-2`, etc.). Domain specificity comes from the HyDE content itself, not the model choice — making the system portable across any profession.

**LangGraph with Postgres checkpointer** — State is persisted at every node. The human-in-the-loop confirmation gate is a native LangGraph `interrupt()` — the graph pauses, state survives a server restart, and resumes cleanly when the user confirms.

**Fit score formula:**
```
With education requirement:    0.40×exp + 0.30×skill + 0.20×domain + 0.10×edu
Without education requirement: 0.50×exp + 0.30×skill + 0.20×domain
```

**Seniority preference** controls both the HyDE JD generation framing (aspirational vs. current profile) and the downstream experience filter threshold (0.3 for *step_up*, 0.5 for *same_level*).

---

## Tech stack

| Layer | Technology |
|---|---|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM | [Anthropic Claude](https://www.anthropic.com/) (Sonnet for parsing/HyDE, Haiku for ranker) |
| Embeddings | [Voyage AI](https://www.voyageai.com/) (`voyage-3`) |
| Job APIs | RapidAPI (JSearch, Active Jobs DB, Techmap, Jobs Search API), RemoteOK |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Database | PostgreSQL (via LangGraph Postgres checkpointer) |
| Observability | MLflow (metrics + per-job audit artifact) |
| Containerisation | Docker (multi-stage build, single Dockerfile) |
| CI/CD | GitHub Actions → Amazon ECR → EC2 |

---

## Project structure

```
autonomous-job-finder/
├── agents/
│   ├── hyde/              # Agent 5b: HyDE prefilter (Sonnet + Voyage AI)
│   ├── job_search/        # Agent 3: multi-source job search (registry pattern)
│   │   └── sources/       # plug-and-play source adapters
│   ├── parser/            # Agent 1: resume parser (Claude Sonnet)
│   ├── pruner/            # URL validator (HTTP-based, no LLM)
│   ├── ranker/            # Agent 6: LLM scorer (Claude Haiku)
│   └── recommender/       # Agent 2: profile recommender (Claude Haiku)
├── api/                   # FastAPI app, routes, models, dependencies
├── core/
│   ├── config/            # llm_config.yaml loader + all tunable thresholds
│   ├── graph.py           # LangGraph pipeline definition
│   ├── observability/     # MLflow logger + SessionMetrics
│   ├── prompts/           # All LLM prompt builders (one file per agent)
│   ├── state/             # SessionState Pydantic model (shared pipeline state)
│   └── url_validator.py   # HTTP dead-link checker
├── docker/                # nginx config, Postgres init SQL
├── frontend/              # Streamlit app (single file)
├── test_scripts/          # Live integration tests (one per agent)
├── docker-compose.yml     # Local development
├── docker-compose.prod.yml# Production (EC2)
├── Dockerfile             # Multi-stage: base → api-runtime + streamlit-runtime
└── docs/
    └── architecture.md    # Full pipeline design document
```

---

## Quickstart (local)

**Prerequisites:** Docker Desktop, an Anthropic API key, a Voyage AI API key, and at least one RapidAPI job source subscription.

**1. Clone and configure**
```bash
git clone https://github.com/harsh0701Xd/Autonomous-Job-Finder.git
cd autonomous-job-finder
cp .env.example .env
# Fill in your API keys in .env
```

**2. Start everything**
```bash
docker compose up --build
```

**3. Open the app**
- Streamlit UI: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs

**4. Upload a resume and run**

Upload a PDF or DOCX resume, select your location and seniority preference, confirm a profile, and wait ~2 minutes for results.

---

## Environment variables

All keys are documented in [`.env.example`](.env.example). Required keys:

| Variable | Where to get it |
|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |
| `VOYAGE_API_KEY` | [dash.voyageai.com](https://dash.voyageai.com) |
| `JSEARCH_API_KEY` | [rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch](https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch) |
| `ACTIVE_JOBS_DB_API_KEY` | [rapidapi.com](https://rapidapi.com/bebity/api/active-jobs-db) (optional — enables Active Jobs DB source) |
| `JOBS_SEARCH_API_KEY` | [rapidapi.com](https://rapidapi.com/fantastic-jobs/api/jobs-search-api) (optional — enables Jobs Search API source) |
| `TECHMAP_API_KEY` | [rapidapi.com](https://rapidapi.com/techmap-io-techmap-io-default/api/daily-international-job-postings) (optional — enables Techmap source) |
| `LINKEDIN_JOBS_API_KEY` | [rapidapi.com](https://rapidapi.com/fantastic-jobs/api/linkedin-job-search-api) (optional — enables LinkedIn Jobs source) |
| `DATABASE_URL` | Auto-set by Docker Compose from `POSTGRES_*` vars |
| `LANGSMITH_API_KEY` | [smith.langchain.com](https://smith.langchain.com) (optional, for tracing) |

---

## Running tests

Each test script exercises one agent live (requires API keys in `.env`):

```bash
python test_scripts/test_resume_parser.py
python test_scripts/test_profile_recommender.py
python test_scripts/test_job_search.py
python test_scripts/test_url_validator.py
python test_scripts/test_hyde_prefilter.py
python test_scripts/test_ranker.py   # full chain: search → validate → dedup → hyde → rank
```

See [`test_scripts/README.md`](test_scripts/README.md) for the recommended run order and what each script validates.

---

## Deployment (AWS)

Deployment is automated via GitHub Actions on every push to `main`:

1. Builds `api-runtime` and `streamlit-runtime` Docker images from the shared multi-stage `Dockerfile`
2. Pushes both to Amazon ECR
3. SSHs into EC2, pulls the new images, and restarts with `docker-compose.prod.yml`

Required GitHub Secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `ECR_REGISTRY`, `EC2_HOST`, `EC2_USER`, `EC2_SSH_KEY`.

---

## Observability

Every pipeline run logs to MLflow:
- Per-agent latency, token usage, and estimated cost (INR)
- Full pipeline funnel (jobs at each gate: search → URL pruner → dedup → HyDE → ranker filters → final)
- `job_audit.json` artifact: complete per-job record including fit scores, sub-scores, HyDE section, and drop reason for every job that entered the pipeline

MLflow UI is available at http://localhost:5001 when running locally.

---

## License

MIT
