"""
Metrics collection and tracking for Radiant Agentic RAG.

Provides step-by-step timing, success/failure tracking, and aggregated statistics.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StepMetric:
    """Metrics for a single pipeline step."""

    name: str
    started_at: float
    ended_at: Optional[float] = None
    ok: bool = True
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> Optional[float]:
        """Calculate step latency in milliseconds."""
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at) * 1000.0

    @property
    def is_complete(self) -> bool:
        """Check if step has been completed."""
        return self.ended_at is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "latency_ms": self.latency_ms,
            "ok": self.ok,
            "error": self.error,
            "extra": self.extra,
        }


@dataclass
class RunMetrics:
    """Aggregated metrics for a complete pipeline run."""

    run_id: str
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    steps: List[StepMetric] = field(default_factory=list)
    pipeline_extra: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    degraded_features: List[str] = field(default_factory=list)

    def start_step(self, name: str, **extra: Any) -> StepMetric:
        """
        Start a new step and add it to the metrics.

        Args:
            name: Step name (e.g., "RetrievalAgent")
            **extra: Additional metadata for the step

        Returns:
            StepMetric instance to track the step
        """
        step = StepMetric(name=name, started_at=time.time(), extra=dict(extra))
        self.steps.append(step)
        logger.debug(f"Started step: {name}")
        return step

    def end_step(
        self,
        step: StepMetric,
        ok: bool = True,
        error: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """
        Mark a step as complete.

        Args:
            step: The StepMetric to complete
            ok: Whether the step succeeded
            error: Error message if step failed
            **extra: Additional metadata to add
        """
        step.ended_at = time.time()
        step.ok = ok
        step.error = error
        step.extra.update(extra)

        if not ok:
            logger.warning(f"Step failed: {step.name} - {error}")
        else:
            logger.debug(f"Completed step: {step.name} ({step.latency_ms:.1f}ms)")

    @contextmanager
    def track_step(self, name: str, **extra: Any) -> Generator[StepMetric, None, None]:
        """
        Context manager for tracking a step.

        Usage:
            with metrics.track_step("MyStep") as step:
                # do work
                step.extra["items_processed"] = 42

        Automatically handles timing and error capture.
        """
        step = self.start_step(name, **extra)
        try:
            yield step
            self.end_step(step, ok=True)
        except Exception as e:
            self.end_step(step, ok=False, error=str(e))
            raise

    def add_warning(self, message: str) -> None:
        """Add a warning message to the run."""
        self.warnings.append(message)
        logger.warning(f"Run warning: {message}")

    def mark_degraded(self, feature: str, reason: str) -> None:
        """Mark a feature as degraded (fallback used)."""
        self.degraded_features.append(f"{feature}: {reason}")
        logger.warning(f"Degraded feature: {feature} - {reason}")

    def finish(self, **pipeline_extra: Any) -> None:
        """
        Mark the run as complete.

        Args:
            **pipeline_extra: Additional pipeline-level metadata
        """
        self.ended_at = time.time()
        self.pipeline_extra.update(pipeline_extra)
        logger.info(f"Run completed: {self.run_id} ({self.total_latency_ms:.1f}ms)")

    @property
    def total_latency_ms(self) -> Optional[float]:
        """Calculate total run latency in milliseconds."""
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at) * 1000.0

    @property
    def is_complete(self) -> bool:
        """Check if run has been completed."""
        return self.ended_at is not None

    @property
    def success_rate(self) -> float:
        """Calculate success rate of steps."""
        if not self.steps:
            return 1.0
        successful = sum(1 for s in self.steps if s.ok)
        return successful / len(self.steps)

    @property
    def failed_steps(self) -> List[StepMetric]:
        """Get list of failed steps."""
        return [s for s in self.steps if not s.ok]

    def get_step(self, name: str) -> Optional[StepMetric]:
        """Get a step by name (returns first match)."""
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total_latency_ms": self.total_latency_ms,
            "success_rate": self.success_rate,
            "steps": [s.to_dict() for s in self.steps],
            "pipeline_extra": self.pipeline_extra,
            "warnings": self.warnings,
            "degraded_features": self.degraded_features,
        }

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Run ID: {self.run_id}",
            f"Total time: {self.total_latency_ms:.1f}ms" if self.total_latency_ms else "In progress",
            f"Steps: {len(self.steps)} ({self.success_rate:.0%} success)",
        ]

        if self.failed_steps:
            lines.append("Failed steps:")
            for step in self.failed_steps:
                lines.append(f"  - {step.name}: {step.error}")

        if self.degraded_features:
            lines.append("Degraded features:")
            for feature in self.degraded_features:
                lines.append(f"  - {feature}")

        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        return "\n".join(lines)


class MetricsCollector:
    """
    Collects and aggregates metrics across multiple runs.

    Useful for monitoring and performance analysis.
    """

    def __init__(self, max_history: int = 100) -> None:
        self._history: List[RunMetrics] = []
        self._max_history = max_history

    def record(self, metrics: RunMetrics) -> None:
        """Record a completed run's metrics."""
        self._history.append(metrics)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    @property
    def run_count(self) -> int:
        """Total number of recorded runs."""
        return len(self._history)

    @property
    def average_latency_ms(self) -> Optional[float]:
        """Average latency across all runs."""
        latencies = [m.total_latency_ms for m in self._history if m.total_latency_ms is not None]
        if not latencies:
            return None
        return sum(latencies) / len(latencies)

    @property
    def average_success_rate(self) -> float:
        """Average success rate across all runs."""
        if not self._history:
            return 1.0
        return sum(m.success_rate for m in self._history) / len(self._history)

    def step_stats(self, step_name: str) -> Dict[str, Any]:
        """Get statistics for a specific step across all runs."""
        step_metrics = []
        for run in self._history:
            step = run.get_step(step_name)
            if step and step.is_complete:
                step_metrics.append(step)

        if not step_metrics:
            return {"count": 0}

        latencies = [s.latency_ms for s in step_metrics if s.latency_ms is not None]
        success_count = sum(1 for s in step_metrics if s.ok)

        return {
            "count": len(step_metrics),
            "success_count": success_count,
            "success_rate": success_count / len(step_metrics),
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None,
            "min_latency_ms": min(latencies) if latencies else None,
            "max_latency_ms": max(latencies) if latencies else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert collector stats to dictionary."""
        return {
            "run_count": self.run_count,
            "average_latency_ms": self.average_latency_ms,
            "average_success_rate": self.average_success_rate,
            "history": [m.to_dict() for m in self._history[-10:]],  # Last 10 runs
        }
