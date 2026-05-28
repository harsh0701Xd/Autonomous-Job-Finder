"""
agents/job_search/registry.py

Plug-and-play source registry for the job search pipeline.

To add a new job source:
  1. Create agents/job_search/sources/<name>.py with:
       - SOURCE_SPEC dict (name, env_key, always_on, requires_location)
       - async search(profile_title, location, pages, **kwargs) -> list[dict]
       - normalize(raw, matched_profile) -> Optional[RawJob]
  2. Import the module and add it to the SOURCES list below.
  3. Add a matching entry to llm_config.yaml [job_search.sources].
  4. Set the env var named in SOURCE_SPEC["env_key"] in your .env file.
  No other code changes needed.

Registry contract:
  - SOURCES is an ordered list of source modules (order = search priority).
  - Each module exposes SOURCE_SPEC, search(), and normalize().
  - job_search_agent.py uses SOURCES to drive all searches; it never
    imports individual source modules directly.
  - validate_source_keys() is called at startup to warn about enabled
    sources with missing API keys.
"""

from __future__ import annotations

import logging
import os

from agents.job_search.sources import (
    active_jobs_db,
    jobs_search_api,
    jsearch,
    linkedin_jobs,
    remoteok,
    techmap,
)
from core.config.config_loader import cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SOURCES — ordered list of all registered source modules.
# The job_search_agent iterates this list; sources disabled in llm_config.yaml
# are skipped. Add new sources here and in llm_config.yaml [job_search.sources].
# ---------------------------------------------------------------------------
SOURCES = [
    jsearch,
    active_jobs_db,
    linkedin_jobs,
    jobs_search_api,
    techmap,
    remoteok,
]


def validate_source_keys() -> None:
    """
    Warn at startup if any enabled source is missing its required API key.

    Called once by job_search_agent.py on first import. Does not raise —
    missing keys produce a WARNING so the pipeline continues with the
    remaining sources. The source's own search() function will also log
    and return [] when the key is absent at call time.
    """
    for source_module in SOURCES:
        spec       = source_module.SOURCE_SPEC
        name       = spec["name"]
        env_key    = spec.get("env_key")
        always_on  = spec.get("always_on", False)
        source_cfg = cfg.job_search.source(name)

        is_active = always_on or source_cfg.enabled
        if not is_active:
            continue   # disabled — no key check needed

        if env_key and not os.getenv(env_key):
            logger.warning(
                f"[registry] Source '{name}' is enabled but {env_key} is not set. "
                f"This source will return 0 jobs. Set {env_key} in your .env file."
            )
