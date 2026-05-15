# Autonomous Job Finder - Project Context

## Overview

**Autonomous Job Finder** is a sophisticated LangGraph-based multi-agent AI system designed to find relevant open job positions based on an uploaded user resume. The system intelligently parses resumes, infers candidate profiles, matches them with job opportunities, and ranks results by relevance and fit.

**Core Purpose**: Automate the job discovery process by combining resume analysis, profile matching, intelligent job searching, and relevance ranking.

---

## Project Directory Structure

```
autonomous-job-finder/
├── 📁 agents/                          # AI Agent Modules
│   ├── 📁 job_search/
│   │   ├── job_search_agent.py         # Agent 3: Job Search (JSearch API)
│   │   ├── normalizer.py               # Job data normalization
│   │   └── 📁 sources/                 # Job source configurations
│   ├── 📁 parser/
│   │   └── resume_parser.py            # Agent 1: Resume Parser (Claude Sonnet)
│   ├── 📁 pruner/
│   │   └── url_pruner.py               # Agent 4: URL Pruner (Claude Haiku)
│   ├── 📁 ranker/
│   │   └── ranker_agent.py             # Agent 5: Ranker (Claude Haiku ×60)
│   └── 📁 recommender/
│       └── profile_recommender.py      # Agent 2: Profile Recommender (Claude Haiku)
│
├── 📁 api/                             # REST API Layer
│   ├── app.py                          # FastAPI application entry point
│   ├── dependencies.py                 # Session management, DB, graph deps
│   ├── models.py                       # Pydantic request/response schemas
│   └── routes.py                       # All endpoint handlers
│
├── 📁 core/                            # Core Orchestration & State
│   ├── graph.py                        # LangGraph pipeline orchestration
│   ├── 📁 config/
│   │   ├── config_loader.py            # Configuration management
│   │   └── llm_config.yaml             # LLM provider configurations
│   ├── 📁 observability/
│   │   ├── metrics.py                  # Performance metrics
│   │   └── mlflow_logger.py            # MLflow experiment tracking
│   ├── 📁 prompts/                     # LLM Prompt Templates
│   │   ├── parser_prompts.py           # Resume parser prompts
│   │   ├── pruner_prompts.py           # URL pruner prompts
│   │   ├── ranker_prompts.py           # Job ranker prompts
│   │   └── recommender_prompts.py      # Profile recommender prompts
│   ├── 📁 schemas/                     # Data validation schemas
│   └── 📁 state/
│       └── session_state.py            # SessionState Pydantic model
│
├── 📁 docker/                          # Docker Configuration
│   └── init.sql                        # PostgreSQL initialization script
│
├── 📁 frontend/                        # User Interface
│   ├── streamlit_app.py                # Streamlit web application
│   └── mock_test.py                    # Local testing without live APIs
│
├── 📁 mlruns/                          # MLflow Experiment Tracking
│   └── 📁 {experiment_id}/
│       └── 📁 {run_id}/
│           ├── artifacts/              # Stored artifacts (resumes, profiles)
│           └── metrics/                # Performance metrics
│
├── 📁 scripts/                         # Utility Scripts
│   ├── test_job_search_agent_live.py   # Live job search testing
│   ├── test_jsearch.py                 # JSearch API testing
│   ├── test_ranker_agent_live.py       # Live ranker testing
│   └── test_signals_agent_live.py      # Signals agent testing
│
├── 📁 tests/                           # Test Suite
│   └── 📁 unit/
│       └── test_api_routes.py          # API route unit tests
│
├── 📄 docker-compose.yml               # Multi-service Docker setup
├── 📄 Dockerfile                       # Container build configuration
├── 📄 README.md                        # Project documentation
├── 📄 requirements.txt                 # Python dependencies
└── 📄 PROJECT_CONTEXT.md               # This file - Project architecture docs
```

### Directory Purpose Legend

| Directory | Purpose | Key Components |
|-----------|---------|----------------|
| `agents/` | AI agent implementations | 5 specialized agents (Parser, Recommender, Search, Pruner, Ranker) |
| `api/` | REST API layer | FastAPI app, routes, models, dependencies |
| `core/` | Orchestration & state management | LangGraph pipeline, session state, configuration |
| `docker/` | Containerization | PostgreSQL init, Docker setup |
| `frontend/` | User interface | Streamlit app for resume upload & results |
| `mlruns/` | Experiment tracking | MLflow artifacts, metrics, run history |
| `scripts/` | Testing & utilities | Live integration tests, API validation |
| `tests/` | Unit test suite | Automated testing for core functionality |

### File Type Legend

- 📁 **Directory** - Contains subdirectories/files
- 📄 **Configuration/Documentation** - YAML, MD, TXT, SQL files
- 🐍 **Python Module** - Core application code
- 🔧 **Script** - Utility/testing scripts

---

## Architecture Overview

### High-Level Data Flow

```
User Resume (PDF/DOCX)
        ↓
[Agent 1: Resume Parser] → Structured Candidate Profile
        ↓
[Agent 2: Profile Recommender] → Suggested Job Profiles
        ↓
[INTERRUPT: User Confirmation] → Confirmation + Preferences
        ↓
[Agent 3: Job Search] → Raw Job Listings (100-200 jobs)
        ↓
[Agent 4: URL Pruner] → Filtered, Valid URLs (50-100 jobs)
        ↓
[Agent 5: Ranker] → Ranked Job Results (Top-60 Jobs by Fit)
        ↓
Final Results to User
```

---

## System Components

### 1. **API Layer** (`api/`)

**Purpose**: RESTful interface for clients to interact with the pipeline.

#### Endpoints

| Method | Endpoint | Purpose | Input | Output |
|--------|----------|---------|-------|--------|
| `POST` | `/sessions` | Create new session | User preferences | `session_id`, initial state |
| `POST` | `/sessions/{session_id}/resume` | Upload resume & trigger pipeline | Resume file (PDF/DOCX) | `status`, `awaiting_confirmation` flag |
| `POST` | `/sessions/{session_id}/confirm` | Submit profile confirmation | Selected profiles + criteria | Pipeline resumes, `searching` status |
| `GET` | `/sessions/{session_id}/status` | Poll pipeline progress | None | Current `pipeline_status` |
| `GET` | `/sessions/{session_id}/results` | Fetch final results | None | Ranked job list (top 60) |
| `GET` | `/health` | Health check | None | `{"status": "healthy"}` |

#### Key Files

- **`app.py`**: FastAPI application setup, CORS, logging, lifespan management
- **`routes.py`**: All endpoint handlers, state↔response model conversion
- **`models.py`**: Pydantic schemas for all request/response payloads
- **`dependencies.py`**: Session management, database, graph instantiation

---

### 2. **LangGraph Orchestration Layer** (`core/graph.py`)

**Purpose**: Choreographs the multi-agent pipeline with human-in-the-loop interrupt points.

#### Pipeline Nodes & Flow

| Node | Agent | LLM | Input | Output | Notes |
|------|-------|-----|-------|--------|-------|
| Parse Resume | Resume Parser | Claude Sonnet | Resume raw text | `CandidateProfile` | Structured candidate extraction |
| Recommend Profiles | Profile Recommender | Claude Haiku | `CandidateProfile` | List of `SuggestedProfile` | 3-5 target job titles + seniorities |
| **[INTERRUPT]** | User Confirmation | N/A | User selection | `confirmed_profiles` array | Pipeline pauses; resumes when user confirms |
| Search Jobs | Job Search (JSearch API) | N/A (external API) | Confirmed profiles | ~100-200 raw jobs | External API data source |
| Prune URLs | URL Pruner | Claude Haiku | Raw jobs | Deduplicated valid jobs | Validates URLs, removes spam/duplicates |
| Rank Results | Ranker | Claude Haiku (×60 parallel) | Pruned jobs + candidate profile | Top 60 ranked jobs | Parallel scoring using LLM |

#### Checkpoint Persistence

- **Checkpointer**: PostgreSQL via `langgraph-checkpoint-postgres`
- **State Storage**: Every node's output is persisted to DB
- **Resume Capability**: Graph can be paused/resumed at any node with full state recovery

---

### 3. **Core State Management** (`core/state/session_state.py`)

**Purpose**: Single shared state object flowing through entire pipeline.

#### SessionState Structure

```python
class SessionState(BaseModel):
    # Identity & Metadata
    session_id: str
    created_at: datetime
    user_preferences: UserPreferences
    
    # Pipeline Flow
    resume_raw_text: Optional[str]           # Raw text from uploaded file
    
    # Agent 1 (Parser) Outputs
    candidate_profile: Optional[CandidateProfile]
    parse_failed: bool = False
    parse_error: Optional[str]
    
    # Agent 2 (Recommender) Outputs
    suggested_profiles: list[SuggestedProfile]
    
    # Human-in-the-Loop
    awaiting_confirmation: bool = False
    confirmed_profiles: list[SuggestedProfile]
    
    # Agent 3 (Job Search) Outputs
    raw_jobs: list[RawJob]
    
    # Agent 4 (Pruner) Outputs
    pruned_jobs: list[RawJob]
    pruned_count: int
    
    # Agent 5 (Ranker) Outputs
    ranked_jobs: list[RankedJob]
    
    # Pipeline Status
    pipeline_complete: bool = False
    results_ready: bool = False
    error: Optional[str]
    last_updated: datetime
```

#### Supported Sub-Schemas

- **`CandidateProfile`**: Parsed resume data (skills, experience, education, seniority)
- **`SuggestedProfile`**: Recommended target job profiles (title, seniority, confidence, reason)
- **`RawJob`**: Job listing from JSearch API (title, company, location, JD text, apply URL)
- **`RankedJob`**: Job with AI-generated match score (includes reasoning)

---

### 4. **Agent Modules** (`agents/`)

#### Agent 1: Resume Parser (`agents/parser/resume_parser.py`)

| Aspect | Detail |
|--------|--------|
| **LLM** | Claude Sonnet (most capable for complex extraction) |
| **Input** | Raw resume text (extracted from PDF/DOCX) |
| **Processing** | Claude prompt → structured JSON → Pydantic validation |
| **Output** | `CandidateProfile` with: skills, experience, education, seniority, career gaps, domain expertise |
| **Error Handling** | Returns `parse_failed=true` + error message if parsing fails |

#### Agent 2: Profile Recommender (`agents/recommender/profile_recommender.py`)

| Aspect | Detail |
|--------|--------|
| **LLM** | Claude Haiku (fast, cost-efficient) |
| **Input** | `CandidateProfile` |
| **Processing** | Claude prompt → 3-5 target job title + seniority combinations |
| **Output** | List of `SuggestedProfile` with confidence levels and match reasoning |
| **User Interaction** | Graph pauses; user selects preferred profiles from suggestions |

#### Agent 3: Job Search (`agents/job_search/job_search_agent.py`)

| Aspect | Detail |
|--------|--------|
| **Data Source** | JSearch API (external, requires API key) |
| **Input** | `confirmed_profiles` (user-selected job titles/seniorities) |
| **Processing** | Calls JSearch for each profile → aggregates results |
| **Output** | 100-200 raw job listings from multiple sources |
| **Deduplication** | Job search may return overlaps; pruner handles cleanup |

#### Agent 4: URL Pruner (`agents/pruner/url_pruner.py`)

| Aspect | Detail |
|--------|--------|
| **LLM** | Claude Haiku (validates & filters) |
| **Input** | Raw jobs from JSearch |
| **Processing** | URL validation, duplicate detection, spam filtering |
| **Output** | Cleaned job list (50-100 high-quality jobs) |
| **Purpose** | Removes invalid URLs, duplicate listings, suspicious content |

#### Agent 5: Ranker (`agents/ranker/ranker_agent.py`)

| Aspect | Detail |
|--------|--------|
| **LLM** | Claude Haiku (parallelized ×60 concurrent calls) |
| **Input** | Pruned jobs + `CandidateProfile` (ATS summary) |
| **Processing** | LLM scores each job on: fit, seniority match, skills alignment |
| **Output** | Top 60 ranked jobs with match scores + reasoning |
| **Optimization** | Parallel execution for 60-job batches; reduces latency |

#### Disabled: Signals Agent (`agents/signals/`)

- **Status**: Code preserved but not active in graph
- **Reason**: Low signal reliability for Indian job market; sparse NewsAPI coverage
- **Reinstatement**: Add node + edge when alternate data source available

---

### 5. **Frontend** (`frontend/`)

#### Streamlit Application (`frontend/streamlit_app.py`)

- **Purpose**: User-friendly interactive UI for resume upload, confirmation, and result browsing
- **Flow**:
  1. User uploads resume
  2. Displays recommended profiles (from Agent 2)
  3. User selects profiles → clicks "Confirm"
  4. Shows job search progress (polling `/status`)
  5. Displays ranked results in table format
- **Features**: Real-time status polling, sortable tables, apply links

#### Mock Test (`frontend/mock_test.py`)

- Local testing without live API/LLM calls

---

### 6. **Configuration & Observability** (`core/`)

#### LLM Configuration (`core/config/llm_config.yaml`)

```yaml
llm_providers:
  sonnet: "Claude 3.5 Sonnet"
  haiku: "Claude 3.5 Haiku"
models:
  parser: "claude-3-5-sonnet-20241022"
  recommender: "claude-3-5-haiku-20241022"
  pruner: "claude-3-5-haiku-20241022"
  ranker: "claude-3-5-haiku-20241022"
temperatures:
  parser: 0.0  # Deterministic extraction
  ranker: 0.0  # Consistent scoring
```

#### MLflow Integration (`core/observability/mlflow_logger.py`)

- **Logging**: Tracks each agent's input/output, latency, token usage
- **Artifact Storage**: Resume text, profiles, job listings stored in MLflow
- **Tracking**: All experiments logged under project name (dev/prod-aware)

#### Metrics (`core/observability/metrics.py`)

- Token usage per agent
- Latency per node
- Job deduplication rate
- Ranking score distribution

---

## Data Flow & Context

### 📥 Input Context

| Component | Input Type | Format | Example |
|-----------|-----------|--------|---------|
| User | Resume file | PDF or DOCX | `resume.pdf` (2-3 pages) |
| User | Job preferences | UI selection | "Senior Software Engineer", "Mid-level Product Manager" |
| External API | Job market | JSearch | ~50 job postings returned per query |

### 📤 Output Context

| Component | Output Type | Format | Example |
|-----------|-------------|--------|---------|
| Agent 1 | Structured profile | JSON (SessionState) | `{"seniority_level": "mid", "skills": {...}}` |
| Agent 2 | Job suggestions | List of profiles | 3-5 recommended roles with confidence |
| Agent 5 | Final results | Ranked jobs | Top 60 jobs with match %, apply links |
| API | HTTP response | JSON | `{"results": [...], "status": "complete"}` |

---

## Technology Stack

### Core Orchestration
- **LangGraph** ≥0.2.0 — Multi-agent graph orchestration
- **LangChain** — Prompting framework
- **LangSmith** — Tracing & debugging (dev-aware project isolation)

### LLMs & APIs
- **Claude API** (Anthropic) ≥0.30.0 — LLM backbone
- **JSearch API** — Job data source (external)

### Backend Infrastructure
- **FastAPI** ≥0.111.0 — REST API framework
- **Uvicorn** ≥0.30.0 — ASGI server
- **PostgreSQL** + **psycopg3** ≥3.1.0 — State persistence, checkpointing

### Resume Processing
- **PyMuPDF** ≥1.24.0 — PDF text extraction
- **python-docx** ≥1.1.0 — DOCX parsing

### Observability & Monitoring
- **MLflow** ≥2.14.0 — Experiment tracking, artifact storage
- Structured logging (Python `logging` module)

### Frontend
- **Streamlit** ≥1.35.0 — Interactive UI

### Testing & Development
- **pytest** ≥8.2.0 — Unit testing
- **pytest-asyncio** ≥0.23.0 — Async test support

---

## Deployment

### Docker Setup

#### `Dockerfile`
- Multi-stage build for FastAPI + Streamlit
- Installs dependencies from `requirements.txt`
- Exposes ports 8000 (API) and 8501 (Streamlit)

#### `docker-compose.yml`
- **Services**:
  - `app` — FastAPI (port 8000)
  - `frontend` — Streamlit (port 8501)
  - `postgres` — PostgreSQL (port 5432, persistence volume)
- **Database Init**: `docker/init.sql` sets up schema

### Environment Configuration

Requires `.env` file:
```env
ANTHROPIC_API_KEY=sk-ant-...
JSEARCH_API_KEY=...
DATABASE_URL=postgresql://user:pass@localhost:5432/jobfinder
LOG_LEVEL=INFO
APP_ENV=production
LANGCHAIN_PROJECT=autonomous-job-finder
```

---

## Session Lifecycle

```
1. POST /sessions
   → Creates session_id, stores user preferences in DB

2. POST /sessions/{session_id}/resume
   → Uploads resume → Triggers graph.invoke()
   → Graph pauses at "User Confirmation" step
   → Returns awaiting_confirmation=true

3. GET /sessions/{session_id}/status
   → Client polls to detect when user confirmation is needed

4. POST /sessions/{session_id}/confirm
   → User submits profile selections
   → Graph resumes from interrupt point
   → Job search, pruning, ranking proceed
   → Returns status as each step completes

5. GET /sessions/{session_id}/results
   → After pipeline_complete=true
   → Returns top 60 ranked jobs with scores & apply links

6. Session Persistence
   → All state checkpointed to PostgreSQL after each node
   → Can recover from failures, resume interrupted pipeline
```

---

## Error Handling & Recovery

### Parse Failures
- If Resume Parser fails → sets `parse_failed=true`
- Pipeline skips subsequent agents, returns error to user
- User can re-upload resume or contact support

### Job Search Dry Spell
- If JSearch returns <10 results → warning logged
- Pruner & Ranker proceed with available jobs (no minimum threshold)

### Database/Checkpoint Failures
- Tenacity retry logic (3 retries with backoff)
- If persist fails → error logged, state held in memory temporarily

### API Rate Limiting
- Claude API: Standard rate limits apply
- JSearch: Backoff strategy in place
- Parallel ranker: Throttled to 60 concurrent calls

---

## Key Design Decisions

### 1. **Human-in-the-Loop Interrupt**
- Graph pauses after profile recommendation
- Users review AI suggestions before costly job search begins
- Reduces API costs, improves relevance

### 2. **Parallel Ranking with Claude Haiku**
- Haiku more cost-efficient than Sonnet for scoring
- 60 parallel requests reduce latency vs. sequential
- High accuracy sufficient for ranking task

### 3. **PostgreSQL Checkpointing**
- Multi-session concurrency support
- Resume interrupted pipelines across server restarts
- Audit trail of all state transitions

### 4. **Deduplicated Job Results**
- Pruner removes duplicates from multiple sources
- JSearch may return same listing from different boards
- Final list is unique job opportunities

### 5. **Disabled Signals Agent**
- Preserved for future reactivation
- Requires better data source (NewsAPI coverage insufficient)
- Domain-specific decision for Indian job market

---

## Testing

### Unit Tests (`tests/unit/`)

- **`test_api_routes.py`** — Route handlers, session lifecycle
- **Parser tests** — Resume extraction accuracy
- **Ranker tests** — Score consistency

### Live Integration Tests (`scripts/`)

- **`test_job_search_agent_live.py`** — JSearch API integration
- **`test_ranker_agent_live.py`** — Ranking pipeline end-to-end
- **`test_signals_agent_live.py`** — Historical signals processing

### MLflow Experiment Tracking

- Each test run logged as experiment
- Artifacts (resumes, profiles, results) stored
- Compare metrics across runs (latency, token usage, rankings)

---

## Observability & Monitoring

### Logging

- Structured format: `TIMESTAMP | LEVEL | MODULE | MESSAGE`
- Configurable via `LOG_LEVEL` env var (DEBUG, INFO, WARNING, ERROR)
- All agent transitions logged

### LangSmith Tracing

- **Dev**: Per-day project isolation (`dev-{YYYY-MM-DD}`)
- **Production**: Unified project (`autonomous-job-finder`)
- Traces include: prompts, LLM calls, state transitions

### MLflow Artifacts

```
mlruns/
  {experiment_id}/
    {run_id}/
      artifacts/
        resume_text.txt
        candidate_profile.json
        suggested_profiles.json
        ranked_jobs.json
      metrics/
        parser_latency_ms
        total_jobs_found
        final_ranking_time_ms
```

---

## Future Enhancements

1. **Job Alerts**: Monitor new postings matching preferred profiles
2. **Upskilling Recommendations**: Suggest courses to increase match scores
3. **Negotiation Guidance**: LLM-powered salary/benefit guidance
4. **Application Timeline**: Track which jobs user applied to, outcomes
5. **Profile Evolution**: Learn from user feedback, refine recommendations
6. **Signals Reactivation**: Integrate better hiring signals data source
7. **Candidate-Sourcing Inversion**: Help recruiters find matching candidates

---

## Contact & Documentation

- **API Docs**: Swagger at `http://localhost:8000/docs` (when running)
- **Project**: `autonomous-job-finder` (GitHub)
- **Team**: Job search automation workstream
