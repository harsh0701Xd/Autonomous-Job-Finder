# This file has been intentionally removed (2026-05-27).
#
# The LLM-based URL Pruner (Agent 5, Claude Haiku) has been replaced by
# core/url_validator.py — a lightweight async HTTP validator that:
#   - Issues a HEAD request per URL (async, concurrent)
#   - Drops 404/410 dead links
#   - Drops URLs that redirect to a different domain (aggregator re-links)
#   - Keeps everything else (fail-open)
#
# The replacement has no hardcoded domain lists and touches no candidate data.
# See core/url_validator.py for the full implementation.
