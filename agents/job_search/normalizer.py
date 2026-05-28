"""
agents/job_search/normalizer.py  [RETIRED]

This module has been retired as part of the plug-and-play source registry
refactor (Task #48).

What moved where:
  - normalize_jsearch()       -> agents/job_search/sources/jsearch.py          normalize()
  - normalize_remoteok()      -> agents/job_search/sources/remoteok.py         normalize()
  - normalize_active_jobs_db()-> agents/job_search/sources/active_jobs_db.py   normalize()
  - normalize_linkedin_jobs() -> agents/job_search/sources/linkedin_jobs.py    normalize()
  - normalize_techmap()       -> agents/job_search/sources/techmap.py          normalize()
  - normalize_jobs_search_api()-> agents/job_search/sources/jobs_search_api.py normalize()
  - Shared helpers (_make_job_id, _parse_date, _clean_text, _is_jd_sufficient,
    _is_english, _is_expired) -> agents/job_search/quality_gates.py

The dispatcher (normalize_jobs) and NORMALIZERS dict are now internal to
job_search_agent.py via the registry pattern (_NORMALIZERS + _normalize_batch).

This file is kept as a tombstone to preserve git history legibility.
Do not import from this module.
"""

raise ImportError(
    "agents/job_search/normalizer.py has been retired. "
    "Each source now owns its own normalize() function. "
    "See agents/job_search/registry.py and agents/job_search/sources/."
)
