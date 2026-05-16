"""
core/observability/mlflow_logger.py

Logs SessionMetrics to MLflow using file-based tracking (no server required).
Creates /app/mlruns inside Docker (volume-mounted to project root on host).
Run `mlflow ui --backend-store-uri ./mlruns --port 5001` locally to view.

Fail-open: MLflow errors never block the pipeline.

Environment variables:
  MLFLOW_TRACKING_URI : override tracking URI (default: file:///app/mlruns)
  MLFLOW_EXPERIMENT   : override experiment name (default: autonomous-job-finder)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.observability.metrics import SessionMetrics

logger = logging.getLogger(__name__)

_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT",    "autonomous-job-finder")
_TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI",  "file:///app/mlruns")


def log_session_metrics(metrics: "SessionMetrics") -> None:
    """
    Log a completed SessionMetrics to MLflow.

    Metrics logged:
      total_latency_secs, total_cost_inr, total_llm_calls,
      total_input/output_tokens, per-agent latency + cost + tokens,
      quality metrics (raw_jobs_fetched, ranked_jobs, mean_fit_score, etc.)

    Artifact:
      session_metrics.json -- full to_dict() payload

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

            # Quality metrics
            for key, value in payload.get("quality", {}).items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"quality_{key}", value)

            # Full JSON artifact -- use log_dict() so it's sent over HTTP
            # to the MLflow server instead of writing to local filesystem
            # (log_artifact() fails when artifact root is a remote path)
            mlflow.log_dict(payload, "session_metrics/session_metrics.json")

        logger.info(
            f"[mlflow] Logged session {metrics.session_id[:8]} -- "
            f"experiment '{_EXPERIMENT_NAME}' at {_TRACKING_URI}"
        )

    except Exception as e:
        logger.error(
            f"[mlflow] Failed to log session metrics: {type(e).__name__}: {e}",
            exc_info=True,
        )
