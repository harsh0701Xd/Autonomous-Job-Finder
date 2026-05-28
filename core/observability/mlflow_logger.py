"""
core/observability/mlflow_logger.py

Logs SessionMetrics to MLflow using file-based tracking (no server required).
Creates ./mlruns relative to the app working directory (volume-mounted to
project root on host when running in Docker).
Run `python -m mlflow ui --backend-store-uri ./mlruns --port 5001` locally to view.

Fail-open: MLflow errors never block the pipeline.

Environment variables:
  MLFLOW_TRACKING_URI : override tracking URI (default: auto-resolved ./mlruns)
  MLFLOW_EXPERIMENT   : override experiment name (default: autonomous-job-finder)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.observability.metrics import SessionMetrics

logger = logging.getLogger(__name__)

# Silence the GitPython warning that fires when git is not on PATH inside Docker
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "autonomous-job-finder")

# Resolve tracking URI: use env var if set, otherwise build an absolute
# file:// URI from the current working directory so artifacts are always
# written to a path that is accessible both inside Docker (cwd=/app) and
# locally (cwd=project root). This avoids the hard-coded /app/mlruns path
# that broke artifact display in the MLflow UI when viewed from the host.
_default_mlruns = Path(os.getcwd()) / "mlruns"
_TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI", _default_mlruns.as_uri())


def log_session_metrics(
    metrics: "SessionMetrics",
    job_audit: list | None = None,
) -> None:
    """
    Log a completed SessionMetrics to MLflow.

    Metrics logged:
      total_latency_secs, total_cost_inr, total_llm_calls,
      total_input/output_tokens, per-agent latency + cost + tokens,
      quality metrics (raw_jobs_fetched, ranked_jobs, mean_fit_score, etc.)

    Artifacts (stored in the run's artifact directory under mlruns/):
      job_audit.json -- per-job filter decisions and LLM scores (if provided)

    Safe if mlflow not installed -- logs a warning and returns.
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("[mlflow] not installed -- skipping. Add mlflow>=2.14.0 to requirements.txt")
        return

    logger.info(f"[mlflow] Writing to tracking URI: {_TRACKING_URI}")

    try:
        # Ensure mlruns directory exists and is writable
        if _TRACKING_URI.startswith("file:///"):
            mlruns_path = Path(_TRACKING_URI[7:])   # strip file:// -> /app/mlruns
            mlruns_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"[mlflow] mlruns dir: {mlruns_path} (writable={os.access(mlruns_path, os.W_OK)})")

        mlflow.set_tracking_uri(_TRACKING_URI)
        mlflow.set_experiment(_EXPERIMENT_NAME)

        payload = metrics.to_dict()

        with mlflow.start_run(run_name=metrics.session_id[:12]):

            mlflow.log_param("session_id", metrics.session_id)
            mlflow.log_param("started_at", payload["started_at"])

            # Pipeline totals
            mlflow.log_metric("total_latency_secs",  payload["total_latency_secs"])
            mlflow.log_metric("total_cost_inr",       payload["total_cost_inr"])
            mlflow.log_metric("total_llm_calls",      payload["total_llm_calls"])
            mlflow.log_metric("total_input_tokens",   payload["total_input_tokens"])
            mlflow.log_metric("total_output_tokens",  payload["total_output_tokens"])

            # Per-agent latency
            for agent, latency in payload.get("agent_latencies", {}).items():
                mlflow.log_metric(f"latency_{agent}", latency)

            # Per-agent token usage + cost
            for agent, usage in payload.get("token_usage", {}).items():
                mlflow.log_metric(f"cost_{agent}",       usage.get("cost_inr", 0))
                mlflow.log_metric(f"tokens_in_{agent}",  usage.get("input_tokens", 0))
                mlflow.log_metric(f"tokens_out_{agent}", usage.get("output_tokens", 0))
                mlflow.log_metric(f"llm_calls_{agent}",  usage.get("llm_calls", 0))

            # Quality metrics (scalar values only -- lists/dicts excluded)
            for key, value in payload.get("quality", {}).items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"quality_{key}", value)

            # Per-job audit artifact -- logged as a dict via the MLflow REST API.
            # log_dict() uploads through the tracking server's HTTP endpoint,
            # avoiding direct filesystem writes to /mlflow/artifacts which is
            # only mounted inside the MLflow server container (not the API container).
            # View in MLflow UI: run page -> Artifacts tab -> job_audit.json
            if job_audit:
                try:
                    mlflow.log_dict(
                        {"jobs": job_audit},
                        artifact_file="job_audit.json",
                    )
                    logger.info(f"[mlflow] Logged job_audit.json ({len(job_audit)} entries)")
                except Exception as artifact_err:
                    logger.warning(f"[mlflow] Failed to log job_audit artifact: {artifact_err}")

        logger.info(
            f"[mlflow] Logged session {metrics.session_id[:8]} -- "
            f"experiment '{_EXPERIMENT_NAME}' at {_TRACKING_URI}"
        )

    except Exception as e:
        logger.error(
            f"[mlflow] Failed to log session metrics: {type(e).__name__}: {e}",
            exc_info=True,
        )
