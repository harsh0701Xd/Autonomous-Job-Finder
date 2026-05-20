"""
core/observability/metrics.py

Session-level observability metrics for the Autonomous Job Finder pipeline.

Captures per-agent latency, token usage, cost, and quality metrics.
Data flows to Postgres (JSONB) and MLflow (file-based tracking).

Token pricing (April 2026):
  claude-sonnet-4-6:         $3.00/MTok in,  $15.00/MTok out
  claude-haiku-4-5-20251001: $0.80/MTok in,   $4.00/MTok out

Usage:
    metrics = SessionMetrics(session_id="abc-123")
    metrics.start_agent("resume_parser")
    metrics.record_llm_call("resume_parser", input_tokens=3200, output_tokens=900,
                            model="claude-sonnet-4-6")
    metrics.end_agent("resume_parser")
    metrics.finish(quality_data={...})
    payload = metrics.to_dict()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

#  Pricing constants 

_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {
        "input":  3.00,
        "output": 15.00,
    },
    "claude-haiku-4-5-20251001": {
        "input":  0.80,
        "output": 4.00,
    },
}

_USD_TO_INR      = 83.5
_DEFAULT_PRICING = {"input": 3.00, "output": 15.00}


def _tokens_to_inr(input_tokens: int, output_tokens: int, model: str) -> float:
    pricing = _PRICING.get(model, _DEFAULT_PRICING)
    usd = (
        input_tokens  / 1_000_000 * pricing["input"] +
        output_tokens / 1_000_000 * pricing["output"]
    )
    return round(usd * _USD_TO_INR, 4)


#  Per-agent accumulator 

@dataclass
class AgentTokens:
    model:         str   = ""
    input_tokens:  int   = 0
    output_tokens: int   = 0
    llm_calls:     int   = 0
    cost_inr:      float = 0.0

    def add(self, input_tokens: int, output_tokens: int, model: str) -> None:
        self.model         = model
        self.input_tokens  += input_tokens
        self.output_tokens += output_tokens
        self.llm_calls     += 1
        self.cost_inr       = round(
            self.cost_inr + _tokens_to_inr(input_tokens, output_tokens, model), 4
        )

    def to_dict(self) -> dict:
        return {
            "model":         self.model,
            "input_tokens":  self.input_tokens,
            "output_tokens": self.output_tokens,
            "llm_calls":     self.llm_calls,
            "cost_inr":      self.cost_inr,
        }


#  SessionMetrics 

@dataclass
class SessionMetrics:
    """
    Collects and aggregates observability metrics for one pipeline session.

    Thread-safe for concurrent asyncio tasks  dict updates are GIL-safe.
    All methods are fail-open  errors logged, never raised.
    """

    session_id:   str
    started_at:   datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    _agent_start:    dict[str, float]       = field(default_factory=dict, repr=False)
    agent_latencies: dict[str, float]       = field(default_factory=dict)
    _tokens:         dict[str, AgentTokens] = field(default_factory=dict, repr=False)

    total_latency_secs:  float = 0.0
    total_input_tokens:  int   = 0
    total_output_tokens: int   = 0
    total_cost_inr:      float = 0.0
    total_llm_calls:     int   = 0
    quality:             dict  = field(default_factory=dict)
    _finished:           bool  = field(default=False, repr=False)

    def start_agent(self, agent: str) -> None:
        self._agent_start[agent] = time.perf_counter()
        if agent not in self._tokens:
            self._tokens[agent] = AgentTokens()

    def end_agent(self, agent: str) -> float:
        if agent not in self._agent_start:
            return 0.0
        elapsed = round(time.perf_counter() - self._agent_start[agent], 2)
        self.agent_latencies[agent] = elapsed
        logger.debug(f"[metrics] {agent} completed in {elapsed}s")
        return elapsed

    def record_llm_call(
        self,
        agent:         str,
        input_tokens:  int,
        output_tokens: int,
        model:         str,
    ) -> None:
        if agent not in self._tokens:
            self._tokens[agent] = AgentTokens()
        try:
            self._tokens[agent].add(input_tokens, output_tokens, model)
        except Exception as e:
            logger.warning(f"[metrics] record_llm_call failed for {agent}: {e}")

    def finish(
        self,
        quality_data:           Optional[dict]  = None,
        total_latency_override: Optional[float] = None,
    ) -> None:
        """
        Aggregate per-agent metrics into session totals.

        Args:
            quality_data: Optional quality summary (ranked job counts, etc.)
            total_latency_override: If provided, used as total_latency_secs
                instead of wall-clock (now - started_at). Use this when
                reconstructing metrics from already-recorded per-agent data
                (e.g. in graph.node_finalise where started_at is the time
                of reconstruction, not the time the session actually began).
        """
        if self._finished:
            return

        if total_latency_override is not None:
            self.total_latency_secs = round(total_latency_override, 2)
        else:
            self.total_latency_secs = round(
                (datetime.now(timezone.utc) - self.started_at).total_seconds(), 2
            )

        for at in self._tokens.values():
            self.total_input_tokens  += at.input_tokens
            self.total_output_tokens += at.output_tokens
            self.total_cost_inr       = round(self.total_cost_inr + at.cost_inr, 4)
            self.total_llm_calls     += at.llm_calls
        self.quality   = quality_data or {}
        self._finished = True
        logger.info(
            f"[metrics] Session {self.session_id[:8]}  "
            f"latency={self.total_latency_secs}s | "
            f"in={self.total_input_tokens} out={self.total_output_tokens} | "
            f"cost=INR {self.total_cost_inr} | "
            f"calls={self.total_llm_calls}"
        )

    def to_dict(self) -> dict:
        return {
            "session_id":          self.session_id,
            "started_at":          self.started_at.isoformat(),
            "total_latency_secs":  self.total_latency_secs,
            "total_input_tokens":  self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_inr":      self.total_cost_inr,
            "total_llm_calls":     self.total_llm_calls,
            "agent_latencies":     self.agent_latencies,
            "token_usage": {
                agent: tokens.to_dict()
                for agent, tokens in self._tokens.items()
            },
            "quality": self.quality,
        }