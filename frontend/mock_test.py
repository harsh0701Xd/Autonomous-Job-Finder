"""
frontend/mock_test.py

Synthetic test runner for the Streamlit UI.
Run with: streamlit run frontend/mock_test.py

Zero API calls -- all data is hardcoded below.
Covers every UI state:
  - High / moderate / low fit scores
  - Education tile present (5-column grid) vs absent (4-column grid)
  - Education gap in "What's missing?"
  - Experience gap, skill gaps, domain gap
  - "What's missing?" absent (no gaps at all)
  - Relative posted dates (today, days ago, weeks ago, months ago)
  - Multiple matched profiles (Senior DS, ML Engineer, Data Engineer)
  - Work type label formatting
"""

import sys
from datetime import datetime, timedelta, timezone

import streamlit as st

sys.path.insert(0, ".")

from streamlit_app import _init_state, _show_step_indicator, show_results

# -- Helpers -------------------------------------------------------------------

def _days_ago(n: int) -> str:
    """Return ISO timestamp string for n days ago."""
    dt = datetime.now(timezone.utc) - timedelta(days=n)
    return dt.isoformat()


# -- Synthetic job data --------------------------------------------------------
# Each job exercises a different combination of UI states.

MOCK_JOBS = [
    # ── Job 1: High fit, no education requirement, no gaps ─────────────────────
    # Tests: green score, 4-tile grid, no "What's missing?" expander
    {
        "job_id":             "mock_001",
        "title":              "Senior Data Scientist – NLP & GenAI",
        "company":            "Fractal",
        "location":           "Bangalore, Karnataka",
        "work_type":          "office",
        "posted_date":        _days_ago(3),
        "apply_url":          "https://fractal.ai/careers",
        "source":             "adzuna",
        "matched_via":        ["Senior Data Scientist"],
        "matched_profile":    "Senior Data Scientist",
        "fit_score":          0.836,
        "experience_score":   0.80,
        "skill_score":        0.90,
        "domain_score":       0.80,
        "recency_score":      0.711,
        "education_required": False,
        "education_score":    None,
        "scoring_notes":      (
            "Strong hands-on expertise in ML, LLMs, and GenAI directly matches role "
            "requirements. AML/fintech background is adjacent to enterprise AI applications. "
            "Minor gap: no explicit mention of production LLM deployment at scale, "
            "though RAG chatbot and anomaly detection work suggest relevant experience."
        ),
        "experience_gap":     None,
        "skill_gaps":         [],
        "domain_gap":         None,
        "education_gap":      None,
    },

    # ── Job 2: High fit, education required, meets requirement ──────────────────
    # Tests: green score, 5-tile grid, education tile green, no education gap
    {
        "job_id":             "mock_002",
        "title":              "Machine Learning Engineer, VP",
        "company":            "Natwest Group",
        "location":           "Bangalore, Karnataka",
        "work_type":          "office",
        "posted_date":        _days_ago(1),
        "apply_url":          "https://natwestgroup.com/careers",
        "source":             "adzuna",
        "matched_via":        ["Machine Learning Engineer"],
        "matched_profile":    "Machine Learning Engineer",
        "fit_score":          0.849,
        "experience_score":   0.80,
        "skill_score":        0.90,
        "domain_score":       0.80,
        "recency_score":      0.978,
        "education_required": True,
        "education_score":    0.80,
        "scoring_notes":      (
            "Strong technical foundation across ML, data engineering, and production systems. "
            "Experience with model deployment, monitoring, and automation aligns well with "
            "role requirements. IIT Kharagpur B.Tech meets the degree requirement; "
            "non-CS field partially compensated by institution prestige. "
            "Minor gaps in explicit MLOps/monitoring tools."
        ),
        "experience_gap":     None,
        "skill_gaps":         ["MLflow", "CI/CD pipelines"],
        "domain_gap":         None,
        "education_gap":      None,
    },

    # ── Job 3: High fit, education required, below requirement ──────────────────
    # Tests: green score, 5-tile grid, education tile amber, education gap shown
    {
        "job_id":             "mock_003",
        "title":              "Data Scientist Associate Senior",
        "company":            "JPMorganChase",
        "location":           "Bangalore, Karnataka",
        "work_type":          "office",
        "posted_date":        _days_ago(65),
        "apply_url":          "https://jpmorgan.com/careers",
        "source":             "adzuna",
        "matched_via":        ["Senior Data Scientist"],
        "matched_profile":    "Senior Data Scientist",
        "fit_score":          0.75,
        "experience_score":   0.60,
        "skill_score":        0.90,
        "domain_score":       1.00,
        "recency_score":      0.00,
        "education_required": True,
        "education_score":    0.60,
        "scoring_notes":      (
            "Strong technical ML and data engineering skills with direct fintech/AML "
            "domain expertise. Role implies Associate Senior level typically requiring "
            "4-6 years; candidate has 1.6 years full-time experience. JD requires "
            "Master's degree in a quantitative field; candidate holds B.Tech."
        ),
        "experience_gap":     "Role implies 4-6yr; candidate has 1.6yr full-time",
        "skill_gaps":         [],
        "domain_gap":         None,
        "education_gap":      "Requires Master's in quantitative field; candidate has B.Tech",
    },

    # ── Job 4: Moderate fit, no education requirement, domain gap ───────────────
    # Tests: amber score, 4-tile grid, domain gap in "What's missing?"
    {
        "job_id":             "mock_004",
        "title":              "Senior Data Scientist – Demand Forecasting",
        "company":            "o9 Solutions",
        "location":           "Bangalore, Karnataka",
        "work_type":          "office",
        "posted_date":        _days_ago(14),
        "apply_url":          "https://o9solutions.com/careers",
        "source":             "adzuna",
        "matched_via":        ["Senior Data Scientist"],
        "matched_profile":    "Senior Data Scientist",
        "fit_score":          0.63,
        "experience_score":   0.60,
        "skill_score":        0.80,
        "domain_score":       0.30,
        "recency_score":      0.689,
        "education_required": False,
        "education_score":    None,
        "scoring_notes":      (
            "Strong time series forecasting skills (SARIMA, LSTM, ARIMA) directly match "
            "demand forecasting requirements. Python and SQL stack aligned. Domain shift "
            "from fintech/AML to supply chain/enterprise planning requires industry "
            "context learning."
        ),
        "experience_gap":     "Requires 3-5yr; candidate has 1.6yr full-time",
        "skill_gaps":         ["Supply chain domain", "Prophet"],
        "domain_gap":         "Role in supply chain planning; candidate background is fintech/AML",
        "education_gap":      None,
    },

    # ── Job 5: Moderate fit, education required, hard mismatch ─────────────────
    # Tests: amber score, 5-tile grid, education tile red (0.3), all gap types
    {
        "job_id":             "mock_005",
        "title":              "Senior Data Scientist – AI & Healthcare Analytics",
        "company":            "Fractal",
        "location":           "Bangalore, Karnataka",
        "work_type":          "office",
        "posted_date":        _days_ago(27),
        "apply_url":          "https://fractal.ai/careers",
        "source":             "adzuna",
        "matched_via":        ["Senior Data Scientist"],
        "matched_profile":    "Senior Data Scientist",
        "fit_score":          0.55,
        "experience_score":   0.40,
        "skill_score":        0.80,
        "domain_score":       0.30,
        "recency_score":      0.40,
        "education_required": True,
        "education_score":    0.30,
        "scoring_notes":      (
            "Strong ML skills but significant gaps in seniority, domain, and education. "
            "Role requires 5-7 years at Senior level (EL3 grade); candidate has 1.6 years. "
            "Healthcare analytics requires domain-specific compliance knowledge absent "
            "from candidate profile. JD requires Master's or PhD in quantitative field."
        ),
        "experience_gap":     "Role requires 5-7yr Senior level; candidate has 1.6yr junior",
        "skill_gaps":         ["HIPAA compliance", "Clinical data standards", "SAS"],
        "domain_gap":         "Role in healthcare analytics; candidate background is fintech/AML",
        "education_gap":      "Requires Master's/PhD; candidate has B.Tech",
    },

    # ── Job 6: Low fit, no education, all gaps ──────────────────────────────────
    # Tests: gray score, 4-tile grid, multiple gaps
    {
        "job_id":             "mock_006",
        "title":              "Senior ML Engineer – Computer Vision",
        "company":            "Netradyne",
        "location":           "Bangalore, Karnataka",
        "work_type":          "office",
        "posted_date":        _days_ago(2),
        "apply_url":          "https://netradyne.com/careers",
        "source":             "adzuna",
        "matched_via":        ["Machine Learning Engineer"],
        "matched_profile":    "Machine Learning Engineer",
        "fit_score":          0.45,
        "experience_score":   0.60,
        "skill_score":        0.40,
        "domain_score":       0.30,
        "recency_score":      0.956,
        "education_required": False,
        "education_score":    None,
        "scoring_notes":      (
            "Strong ML fundamentals but significant domain and skill mismatch. "
            "Role requires Computer Vision and edge computing expertise for fleet safety; "
            "candidate background is entirely fintech/AML with no CV experience. "
            "Python and deep learning frameworks present but CV-specific stack absent."
        ),
        "experience_gap":     None,
        "skill_gaps":         ["Computer Vision", "Edge computing", "Real-time video processing", "OpenCV"],
        "domain_gap":         "Role in fleet safety/transportation; candidate background is fintech/AML",
        "education_gap":      None,
    },

    # ── Job 7: High fit, posted today, multi-profile match ──────────────────────
    # Tests: posted_label="Today", multiple matched_via, no gaps
    {
        "job_id":             "mock_007",
        "title":              "Senior Data Scientist",
        "company":            "Protium",
        "location":           "Bangalore, Karnataka",
        "work_type":          "office",
        "posted_date":        _days_ago(0),
        "apply_url":          "https://protium.co.in/careers",
        "source":             "adzuna",
        "matched_via":        ["Senior Data Scientist", "Data Scientist"],
        "matched_profile":    "Senior Data Scientist | Data Scientist",
        "fit_score":          0.82,
        "experience_score":   0.80,
        "skill_score":        0.90,
        "domain_score":       0.80,
        "recency_score":      0.40,
        "education_required": False,
        "education_score":    None,
        "scoring_notes":      (
            "Fintech/lending risk domain maps directly from AML background. "
            "Anomaly detection, ETL pipelines, and risk feature engineering highly "
            "transferable to MSME credit risk assessment. Technical skills comprehensive "
            "and well-aligned. Posted today — apply immediately."
        ),
        "experience_gap":     None,
        "skill_gaps":         [],
        "domain_gap":         None,
        "education_gap":      None,
    },

    # ── Job 8: Moderate fit, education required, exactly meets requirement ──────
    # Tests: 5-tile grid, education tile green (1.0), skill gaps only
    {
        "job_id":             "mock_008",
        "title":              "Data Engineer – AWS & ML Platform",
        "company":            "CrowdStrike",
        "location":           "Bangalore, Karnataka",
        "work_type":          "office",
        "posted_date":        _days_ago(6),
        "apply_url":          "https://crowdstrike.com/careers",
        "source":             "adzuna",
        "matched_via":        ["Data Engineer"],
        "matched_profile":    "Data Engineer",
        "fit_score":          0.66,
        "experience_score":   0.60,
        "skill_score":        0.70,
        "domain_score":       0.30,
        "recency_score":      0.867,
        "education_required": True,
        "education_score":    1.00,
        "scoring_notes":      (
            "Strong data engineering fundamentals with BigQuery, Airflow, PySpark, Kafka. "
            "Role is AWS-native (Glue, Lambda, Step Functions) and candidate's cloud "
            "exposure is primarily GCP/BigQuery. Cybersecurity domain is a significant "
            "shift from fintech/AML. B.Tech from IIT Kharagpur meets and exceeds "
            "the stated engineering degree requirement."
        ),
        "experience_gap":     "Requires 3-5yr; candidate has 1.6yr full-time",
        "skill_gaps":         ["AWS Glue", "AWS Lambda", "Cybersecurity domain"],
        "domain_gap":         "Role in cybersecurity; candidate background is fintech/AML",
        "education_gap":      None,
    },
]


# -- Render --------------------------------------------------------------------

_init_state()
st.session_state.step = 3
st.session_state.results = {"jobs": MOCK_JOBS}

_show_step_indicator()
show_results()
