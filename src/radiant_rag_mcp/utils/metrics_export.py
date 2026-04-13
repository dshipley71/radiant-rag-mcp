"""
Metrics export utilities for Prometheus and OpenTelemetry.

Provides integration between the BaseAgent metrics collection
and external monitoring systems.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from radiant_rag_mcp.agents.base_agent import AgentResult, BaseAgent

logger = logging.getLogger(__name__)


# =============================================================================
# Prometheus Integration
# =============================================================================

class PrometheusMetricsExporter:
    """
    Exports agent metrics to Prometheus format.
    
    Uses the prometheus_client library if available, otherwise
    provides a no-op implementation.
    
    Example:
        from radiant_rag_mcp.utils.metrics_export import PrometheusMetricsExporter
        
        exporter = PrometheusMetricsExporter(namespace="radiant_rag")
        exporter.register_agent(planning_agent)
        
        # After each agent run
        exporter.record_execution(agent_result)
        
        # Get metrics endpoint (for Flask/FastAPI)
        from prometheus_client import generate_latest
        metrics_output = generate_latest()
    """
    
    def __init__(
        self,
        namespace: str = "radiant",
        subsystem: str = "agent",
        enable_histograms: bool = True,
        histogram_buckets: Optional[List[float]] = None,
    ) -> None:
        """
        Initialize the Prometheus exporter.
        
        Args:
            namespace: Prometheus metric namespace
            subsystem: Prometheus metric subsystem
            enable_histograms: Enable duration histograms
            histogram_buckets: Custom histogram buckets (ms)
        """
        self._namespace = namespace
        self._subsystem = subsystem
        self._enable_histograms = enable_histograms
        self._buckets = histogram_buckets or [
            10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000
        ]
        
        self._prometheus_available = False
        self._counters: Dict[str, Any] = {}
        self._gauges: Dict[str, Any] = {}
        self._histograms: Dict[str, Any] = {}
        
        self._try_import_prometheus()
    
    def _try_import_prometheus(self) -> None:
        """Try to import prometheus_client."""
        try:
            from prometheus_client import Counter, Gauge, Histogram, REGISTRY
            
            self._Counter = Counter
            self._Gauge = Gauge
            self._Histogram = Histogram
            self._REGISTRY = REGISTRY
            self._prometheus_available = True
            
            self._init_metrics()
            logger.info("Prometheus metrics exporter initialized")
            
        except ImportError:
            logger.warning(
                "prometheus_client not installed. "
                "Install with: pip install prometheus_client"
            )
    
    def _init_metrics(self) -> None:
        """Initialize Prometheus metric objects."""
        if not self._prometheus_available:
            return
        
        # Counter for total executions
        self._counters["executions_total"] = self._Counter(
            f"{self._namespace}_{self._subsystem}_executions_total",
            "Total number of agent executions",
            ["agent_name", "agent_category", "status"],
        )
        
        # Counter for errors
        self._counters["errors_total"] = self._Counter(
            f"{self._namespace}_{self._subsystem}_errors_total",
            "Total number of agent errors",
            ["agent_name", "agent_category", "error_type"],
        )
        
        # Gauge for active executions
        self._gauges["active_executions"] = self._Gauge(
            f"{self._namespace}_{self._subsystem}_active_executions",
            "Number of currently active agent executions",
            ["agent_name"],
        )
        
        # Gauge for success rate
        self._gauges["success_rate"] = self._Gauge(
            f"{self._namespace}_{self._subsystem}_success_rate",
            "Agent success rate (0-1)",
            ["agent_name"],
        )
        
        # Histogram for duration
        if self._enable_histograms:
            self._histograms["duration_ms"] = self._Histogram(
                f"{self._namespace}_{self._subsystem}_duration_ms",
                "Agent execution duration in milliseconds",
                ["agent_name", "agent_category"],
                buckets=self._buckets,
            )
        
        # Gauges for agent-specific metrics
        self._gauges["items_processed"] = self._Gauge(
            f"{self._namespace}_{self._subsystem}_items_processed",
            "Number of items processed",
            ["agent_name"],
        )
        
        self._gauges["confidence"] = self._Gauge(
            f"{self._namespace}_{self._subsystem}_confidence",
            "Agent confidence score (0-1)",
            ["agent_name"],
        )
    
    @property
    def available(self) -> bool:
        """Check if Prometheus is available."""
        return self._prometheus_available
    
    def record_execution(self, result: "AgentResult") -> None:
        """
        Record metrics from an agent execution.
        
        Args:
            result: AgentResult from agent.run()
        """
        if not self._prometheus_available or not result.metrics:
            return
        
        metrics = result.metrics
        labels = metrics.to_prometheus_labels()
        
        # Record execution counter
        self._counters["executions_total"].labels(
            agent_name=labels["agent_name"],
            agent_category=labels["agent_category"],
            status=labels["status"],
        ).inc()
        
        # Record duration histogram
        if self._enable_histograms and metrics.duration_ms:
            self._histograms["duration_ms"].labels(
                agent_name=labels["agent_name"],
                agent_category=labels["agent_category"],
            ).observe(metrics.duration_ms)
        
        # Record items processed
        if metrics.items_processed:
            self._gauges["items_processed"].labels(
                agent_name=labels["agent_name"],
            ).set(metrics.items_processed)
        
        # Record confidence
        if metrics.confidence is not None:
            self._gauges["confidence"].labels(
                agent_name=labels["agent_name"],
            ).set(metrics.confidence)
        
        # Record error if failed
        if not result.success and result.error:
            error_type = type(result.error).__name__ if isinstance(result.error, Exception) else "Error"
            self._counters["errors_total"].labels(
                agent_name=labels["agent_name"],
                agent_category=labels["agent_category"],
                error_type=error_type,
            ).inc()
    
    def record_agent_stats(self, agent: "BaseAgent") -> None:
        """
        Record aggregate statistics from an agent.
        
        Args:
            agent: Agent instance to record stats from
        """
        if not self._prometheus_available:
            return
        
        stats = agent.get_statistics()
        
        self._gauges["success_rate"].labels(
            agent_name=agent.name,
        ).set(stats["success_rate"])
    
    def get_metrics_output(self) -> str:
        """
        Get Prometheus metrics in text format.
        
        Returns:
            Prometheus text format metrics string
        """
        if not self._prometheus_available:
            return "# Prometheus not available\n"
        
        from prometheus_client import generate_latest
        return generate_latest(self._REGISTRY).decode("utf-8")


# =============================================================================
# OpenTelemetry Integration
# =============================================================================

class OpenTelemetryExporter:
    """
    Exports agent metrics and traces to OpenTelemetry.
    
    Creates spans for agent executions and records metrics
    using OpenTelemetry APIs.
    
    Example:
        from radiant_rag_mcp.utils.metrics_export import OpenTelemetryExporter
        
        exporter = OpenTelemetryExporter(
            service_name="radiant-rag",
            endpoint="http://localhost:4317",
        )
        
        # Use as context manager for tracing
        with exporter.trace_agent(agent, query="test"):
            result = agent.run(query="test")
            exporter.record_result(result)
    """
    
    def __init__(
        self,
        service_name: str = "radiant-rag",
        endpoint: Optional[str] = None,
        enable_traces: bool = True,
        enable_metrics: bool = True,
    ) -> None:
        """
        Initialize the OpenTelemetry exporter.
        
        Args:
            service_name: Service name for traces/metrics
            endpoint: OTLP endpoint (optional)
            enable_traces: Enable distributed tracing
            enable_metrics: Enable metrics collection
        """
        self._service_name = service_name
        self._endpoint = endpoint
        self._enable_traces = enable_traces
        self._enable_metrics = enable_metrics
        
        self._otel_available = False
        self._tracer = None
        self._meter = None
        
        self._try_import_otel()
    
    def _try_import_otel(self) -> None:
        """Try to import OpenTelemetry libraries."""
        try:
            from opentelemetry import trace, metrics
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.resources import Resource
            
            self._trace = trace
            self._metrics = metrics
            self._TracerProvider = TracerProvider
            self._MeterProvider = MeterProvider
            self._Resource = Resource
            
            self._otel_available = True
            self._init_otel()
            logger.info("OpenTelemetry exporter initialized")
            
        except ImportError:
            logger.warning(
                "OpenTelemetry not installed. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk"
            )
    
    def _init_otel(self) -> None:
        """Initialize OpenTelemetry providers."""
        if not self._otel_available:
            return
        
        resource = self._Resource.create({
            "service.name": self._service_name,
            "service.version": "1.0.0",
        })
        
        if self._enable_traces:
            provider = self._TracerProvider(resource=resource)
            self._trace.set_tracer_provider(provider)
            self._tracer = self._trace.get_tracer(__name__)
            
            # Configure OTLP exporter if endpoint provided
            if self._endpoint:
                try:
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                    from opentelemetry.sdk.trace.export import BatchSpanProcessor
                    
                    exporter = OTLPSpanExporter(endpoint=self._endpoint)
                    provider.add_span_processor(BatchSpanProcessor(exporter))
                except ImportError:
                    logger.warning("OTLP exporter not available")
        
        if self._enable_metrics:
            meter_provider = self._MeterProvider(resource=resource)
            self._metrics.set_meter_provider(meter_provider)
            self._meter = self._metrics.get_meter(__name__)
            
            # Create meters
            self._execution_counter = self._meter.create_counter(
                "agent.executions",
                description="Number of agent executions",
            )
            self._duration_histogram = self._meter.create_histogram(
                "agent.duration",
                unit="ms",
                description="Agent execution duration",
            )
            self._error_counter = self._meter.create_counter(
                "agent.errors",
                description="Number of agent errors",
            )
    
    @property
    def available(self) -> bool:
        """Check if OpenTelemetry is available."""
        return self._otel_available
    
    def trace_agent(
        self,
        agent: "BaseAgent",
        **attributes: Any,
    ):
        """
        Create a tracing context for agent execution.
        
        Args:
            agent: Agent being traced
            **attributes: Additional span attributes
            
        Returns:
            Context manager for the span
        """
        if not self._otel_available or not self._tracer:
            return _NoOpContextManager()
        
        span_name = f"{agent.category.value}/{agent.name}"
        span_attributes = {
            "agent.name": agent.name,
            "agent.category": agent.category.value,
            **{f"agent.input.{k}": str(v)[:100] for k, v in attributes.items()},
        }
        
        return self._tracer.start_as_current_span(
            span_name,
            attributes=span_attributes,
        )
    
    def record_result(self, result: "AgentResult") -> None:
        """
        Record metrics from an agent execution.
        
        Args:
            result: AgentResult from agent.run()
        """
        if not self._otel_available or not result.metrics:
            return
        
        metrics = result.metrics
        attributes = metrics.to_otel_attributes()
        
        # Record execution counter
        if self._execution_counter:
            self._execution_counter.add(1, attributes)
        
        # Record duration
        if self._duration_histogram and metrics.duration_ms:
            self._duration_histogram.record(metrics.duration_ms, attributes)
        
        # Record error
        if not result.success and self._error_counter:
            self._error_counter.add(1, attributes)
        
        # Add result to current span
        current_span = self._trace.get_current_span() if self._trace else None
        if current_span:
            current_span.set_attribute("agent.success", result.success)
            current_span.set_attribute("agent.status", result.status.value if result.status else "unknown")
            if metrics.duration_ms:
                current_span.set_attribute("agent.duration_ms", metrics.duration_ms)
            if result.error:
                current_span.set_attribute("agent.error", str(result.error))


class _NoOpContextManager:
    """No-op context manager for when tracing is disabled."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


# =============================================================================
# Unified Metrics Collector
# =============================================================================

@dataclass
class MetricsCollector:
    """
    Unified metrics collector that supports both Prometheus and OpenTelemetry.
    
    Example:
        collector = MetricsCollector.create(
            prometheus_enabled=True,
            otel_enabled=True,
            otel_endpoint="http://localhost:4317",
        )
        
        # Record all agent executions
        result = agent.run(query="test")
        collector.record(result)
        
        # Get Prometheus output
        print(collector.prometheus_output())
    """
    
    prometheus: Optional[PrometheusMetricsExporter] = None
    opentelemetry: Optional[OpenTelemetryExporter] = None
    _agents: List["BaseAgent"] = field(default_factory=list)
    
    @classmethod
    def create(
        cls,
        prometheus_enabled: bool = True,
        prometheus_namespace: str = "radiant",
        otel_enabled: bool = False,
        otel_service_name: str = "radiant-rag",
        otel_endpoint: Optional[str] = None,
    ) -> "MetricsCollector":
        """
        Create a metrics collector with specified backends.
        
        Args:
            prometheus_enabled: Enable Prometheus metrics
            prometheus_namespace: Prometheus metric namespace
            otel_enabled: Enable OpenTelemetry
            otel_service_name: OpenTelemetry service name
            otel_endpoint: OTLP endpoint for OpenTelemetry
            
        Returns:
            Configured MetricsCollector
        """
        prometheus = None
        opentelemetry = None
        
        if prometheus_enabled:
            prometheus = PrometheusMetricsExporter(namespace=prometheus_namespace)
        
        if otel_enabled:
            opentelemetry = OpenTelemetryExporter(
                service_name=otel_service_name,
                endpoint=otel_endpoint,
            )
        
        return cls(prometheus=prometheus, opentelemetry=opentelemetry)
    
    def register_agent(self, agent: "BaseAgent") -> None:
        """Register an agent for metrics collection."""
        self._agents.append(agent)
    
    def record(self, result: "AgentResult") -> None:
        """
        Record metrics from an agent execution.
        
        Args:
            result: AgentResult from agent.run()
        """
        if self.prometheus:
            self.prometheus.record_execution(result)
        
        if self.opentelemetry:
            self.opentelemetry.record_result(result)
    
    def record_all_stats(self) -> None:
        """Record aggregate stats from all registered agents."""
        for agent in self._agents:
            if self.prometheus:
                self.prometheus.record_agent_stats(agent)
    
    def prometheus_output(self) -> str:
        """Get Prometheus metrics output."""
        if self.prometheus:
            return self.prometheus.get_metrics_output()
        return "# Prometheus not enabled\n"
    
    def trace_agent(self, agent: "BaseAgent", **kwargs):
        """Create a tracing context for agent execution."""
        if self.opentelemetry:
            return self.opentelemetry.trace_agent(agent, **kwargs)
        return _NoOpContextManager()


# =============================================================================
# Convenience Functions
# =============================================================================

_default_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the default metrics collector."""
    global _default_collector
    if _default_collector is None:
        _default_collector = MetricsCollector.create()
    return _default_collector


def configure_metrics(
    prometheus_enabled: bool = True,
    otel_enabled: bool = False,
    **kwargs: Any,
) -> MetricsCollector:
    """
    Configure the global metrics collector.
    
    Args:
        prometheus_enabled: Enable Prometheus metrics
        otel_enabled: Enable OpenTelemetry
        **kwargs: Additional configuration options
        
    Returns:
        Configured MetricsCollector
    """
    global _default_collector
    _default_collector = MetricsCollector.create(
        prometheus_enabled=prometheus_enabled,
        otel_enabled=otel_enabled,
        **kwargs,
    )
    return _default_collector


def record_agent_execution(result: "AgentResult") -> None:
    """Record an agent execution using the default collector."""
    collector = get_metrics_collector()
    collector.record(result)
