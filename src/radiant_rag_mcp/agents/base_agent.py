"""
Abstract Base Agent for Radiant RAG Pipeline.

Provides a formal interface that all agents must implement, ensuring
consistent behavior, logging, metrics collection, and error handling
across the entire agent ecosystem.
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generic, Iterator, List, Optional, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from radiant_rag_mcp.llm.client import LLMClient, LocalNLPModels
    from radiant_rag_mcp.storage.base import BaseVectorStore

logger = logging.getLogger(__name__)

# Type variable for generic result types
T = TypeVar("T")


class AgentCategory(Enum):
    """Categories of agents in the RAG pipeline."""
    
    PLANNING = "planning"
    QUERY_PROCESSING = "query_processing"
    RETRIEVAL = "retrieval"
    POST_RETRIEVAL = "post_retrieval"
    GENERATION = "generation"
    EVALUATION = "evaluation"
    TOOL = "tool"
    UTILITY = "utility"


class AgentStatus(Enum):
    """Execution status of an agent."""
    
    SUCCESS = "success"
    PARTIAL = "partial"  # Completed with warnings
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class AgentMetrics:
    """
    Metrics collected during agent execution.
    
    Supports Prometheus/OpenTelemetry integration via to_prometheus_labels()
    and to_otel_attributes() methods.
    """
    
    # Identification
    agent_name: str
    agent_category: str
    run_id: str
    correlation_id: str
    
    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    
    # Status
    status: AgentStatus = AgentStatus.SUCCESS
    error_message: Optional[str] = None
    
    # Counters
    items_processed: int = 0
    items_returned: int = 0
    llm_calls: int = 0
    retrieval_calls: int = 0
    
    # Quality metrics
    confidence: float = 0.0
    
    # Custom metrics
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "agent_category": self.agent_category,
            "run_id": self.run_id,
            "correlation_id": self.correlation_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "error_message": self.error_message,
            "items_processed": self.items_processed,
            "items_returned": self.items_returned,
            "llm_calls": self.llm_calls,
            "retrieval_calls": self.retrieval_calls,
            "confidence": self.confidence,
            "custom": self.custom,
        }
    
    def to_prometheus_labels(self) -> Dict[str, str]:
        """
        Convert to Prometheus metric labels.
        
        Returns:
            Dictionary suitable for Prometheus label values
        """
        return {
            "agent_name": self.agent_name,
            "agent_category": self.agent_category,
            "status": self.status.value,
        }
    
    def to_otel_attributes(self) -> Dict[str, Any]:
        """
        Convert to OpenTelemetry span attributes.
        
        Returns:
            Dictionary suitable for OTel span attributes
        """
        return {
            "agent.name": self.agent_name,
            "agent.category": self.agent_category,
            "agent.run_id": self.run_id,
            "agent.correlation_id": self.correlation_id,
            "agent.duration_ms": self.duration_ms,
            "agent.status": self.status.value,
            "agent.items_processed": self.items_processed,
            "agent.items_returned": self.items_returned,
            "agent.llm_calls": self.llm_calls,
            "agent.retrieval_calls": self.retrieval_calls,
            "agent.confidence": self.confidence,
        }


@dataclass
class AgentResult(Generic[T]):
    """
    Generic result wrapper for agent execution.
    
    Provides consistent result structure across all agents with
    status, metrics, and the actual output data.
    """
    
    # The actual result data
    data: T
    
    # Execution metadata
    success: bool = True
    status: AgentStatus = AgentStatus.SUCCESS
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Metrics collected during execution
    metrics: Optional[AgentMetrics] = None
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
        if self.status == AgentStatus.SUCCESS:
            self.status = AgentStatus.PARTIAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data_dict = self.data if isinstance(self.data, dict) else None
        if hasattr(self.data, "to_dict"):
            data_dict = self.data.to_dict()
        
        return {
            "data": data_dict,
            "success": self.success,
            "status": self.status.value,
            "error": self.error,
            "warnings": self.warnings,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }


class StructuredLogger:
    """
    Structured logger with correlation ID support.
    
    Provides consistent log formatting with contextual information
    for distributed tracing and debugging.
    """
    
    def __init__(
        self,
        name: str,
        correlation_id: Optional[str] = None,
    ) -> None:
        """
        Initialize structured logger.
        
        Args:
            name: Logger name (typically agent name)
            correlation_id: Optional correlation ID for request tracing
        """
        self._logger = logging.getLogger(name)
        self._correlation_id = correlation_id or str(uuid.uuid4())[:8]
        self._context: Dict[str, Any] = {}
    
    @property
    def correlation_id(self) -> str:
        """Get the correlation ID."""
        return self._correlation_id
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set a new correlation ID."""
        self._correlation_id = correlation_id
    
    def add_context(self, **kwargs: Any) -> None:
        """Add persistent context fields to all log messages."""
        self._context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear persistent context."""
        self._context.clear()
    
    def _format_message(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format message with correlation ID and context."""
        parts = [f"[{self._correlation_id}]"]
        
        # Add context fields
        all_extra = {**self._context, **(extra or {})}
        if all_extra:
            context_str = " ".join(f"{k}={v}" for k, v in all_extra.items())
            parts.append(f"[{context_str}]")
        
        parts.append(message)
        return " ".join(parts)
    
    def debug(self, message: str, **extra: Any) -> None:
        """Log debug message."""
        self._logger.debug(self._format_message(message, extra))
    
    def info(self, message: str, **extra: Any) -> None:
        """Log info message."""
        self._logger.info(self._format_message(message, extra))
    
    def warning(self, message: str, **extra: Any) -> None:
        """Log warning message."""
        self._logger.warning(self._format_message(message, extra))
    
    def error(self, message: str, **extra: Any) -> None:
        """Log error message."""
        self._logger.error(self._format_message(message, extra))
    
    def exception(self, message: str, **extra: Any) -> None:
        """Log exception with traceback."""
        self._logger.exception(self._format_message(message, extra))


class BaseAgent(ABC):
    """
    Abstract base class for all RAG pipeline agents.
    
    Provides:
    - Consistent initialization pattern
    - Structured logging with correlation IDs
    - Automatic metrics collection
    - Error handling and recovery
    - Lifecycle hooks (before/after execution)
    
    All agents in the pipeline should inherit from this class
    and implement the abstract methods.
    
    Example:
        class MyCustomAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "MyCustomAgent"
            
            @property
            def category(self) -> AgentCategory:
                return AgentCategory.UTILITY
            
            def _execute(self, **kwargs) -> Any:
                # Implementation here
                return result
    """
    
    def __init__(
        self,
        llm: Optional["LLMClient"] = None,
        store: Optional["BaseVectorStore"] = None,
        local_models: Optional["LocalNLPModels"] = None,
        enabled: bool = True,
    ) -> None:
        """
        Initialize the base agent.
        
        Args:
            llm: Optional LLM client for reasoning tasks
            store: Optional vector store for retrieval tasks
            local_models: Optional local models for embeddings
            enabled: Whether the agent is enabled
        """
        self._llm = llm
        self._store = store
        self._local_models = local_models
        self._enabled = enabled
        
        # Initialize structured logger
        self._log = StructuredLogger(self.name)
        
        # Metrics tracking
        self._total_executions = 0
        self._total_successes = 0
        self._total_failures = 0
        self._total_duration_ms = 0.0
        
        self._log.info(f"Initialized {self.name}", enabled=enabled)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the agent's unique name.
        
        This name is used for logging, metrics, and registration.
        """
        pass
    
    @property
    @abstractmethod
    def category(self) -> AgentCategory:
        """
        Return the agent's category.
        
        Used for organizing agents and filtering in metrics.
        """
        pass
    
    @property
    def description(self) -> str:
        """
        Return a human-readable description of the agent.
        
        Override this to provide a more detailed description.
        """
        return self.__class__.__doc__ or f"{self.name} agent"
    
    @property
    def enabled(self) -> bool:
        """Check if the agent is enabled."""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable the agent."""
        self._enabled = value
        self._log.info(f"Agent {'enabled' if value else 'disabled'}")
    
    @property
    def logger(self) -> StructuredLogger:
        """Get the agent's structured logger."""
        return self._log
    
    @abstractmethod
    def _execute(self, **kwargs: Any) -> Any:
        """
        Execute the agent's core logic.
        
        This method must be implemented by all subclasses.
        It contains the actual processing logic of the agent.
        
        Args:
            **kwargs: Agent-specific arguments
            
        Returns:
            Agent-specific result type
        """
        pass
    
    def _before_execute(self, **kwargs: Any) -> None:
        """
        Hook called before execution.
        
        Override this to add pre-processing logic.
        """
        pass
    
    def _after_execute(
        self,
        result: Any,
        metrics: AgentMetrics,
        **kwargs: Any,
    ) -> Any:
        """
        Hook called after successful execution.
        
        Override this to add post-processing logic.
        
        Args:
            result: The result from _execute()
            metrics: Collected metrics
            **kwargs: Original kwargs passed to run()
            
        Returns:
            Optionally modified result
        """
        return result
    
    def _on_error(
        self,
        error: Exception,
        metrics: AgentMetrics,
        **kwargs: Any,
    ) -> Optional[Any]:
        """
        Hook called when an error occurs.
        
        Override this to add custom error handling or recovery logic.
        
        Args:
            error: The exception that occurred
            metrics: Collected metrics
            **kwargs: Original kwargs passed to run()
            
        Returns:
            Optional fallback result, or None to re-raise the error
        """
        return None
    
    def execute(
        self,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute the agent and return raw data (backward compatible).
        
        This is a convenience method for code that doesn't need
        the full AgentResult wrapper. It calls run() internally
        and extracts the data.
        
        Args:
            correlation_id: Optional correlation ID for tracing
            **kwargs: Agent-specific arguments
            
        Returns:
            The raw result data from _execute()
            
        Raises:
            RuntimeError: If execution fails and no fallback is available
        """
        result = self.run(correlation_id=correlation_id, **kwargs)
        
        if not result.success and result.data is None:
            error_msg = result.error or "Unknown error"
            raise RuntimeError(f"{self.name} failed: {error_msg}")
        
        return result.data

    def run(
        self,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentResult[Any]:
        """
        Execute the agent with full lifecycle management.
        
        This method handles:
        - Correlation ID propagation
        - Metrics collection
        - Before/after hooks
        - Error handling
        - Structured logging
        
        Args:
            correlation_id: Optional correlation ID for tracing
            **kwargs: Agent-specific arguments
            
        Returns:
            AgentResult wrapping the execution result
        """
        # Check if enabled
        if not self._enabled:
            self._log.debug("Agent is disabled, skipping execution")
            return AgentResult(
                data=None,
                success=True,
                status=AgentStatus.SKIPPED,
            )
        
        # Set up correlation ID
        run_id = str(uuid.uuid4())
        corr_id = correlation_id or str(uuid.uuid4())[:8]
        self._log.set_correlation_id(corr_id)
        
        # Initialize metrics
        metrics = AgentMetrics(
            agent_name=self.name,
            agent_category=self.category.value,
            run_id=run_id,
            correlation_id=corr_id,
            start_time=time.time(),
        )
        
        self._log.debug("Starting execution", run_id=run_id)
        
        try:
            # Before hook
            self._before_execute(**kwargs)
            
            # Execute core logic
            result = self._execute(**kwargs)
            
            # Record success metrics
            metrics.end_time = time.time()
            metrics.duration_ms = (metrics.end_time - metrics.start_time) * 1000
            metrics.status = AgentStatus.SUCCESS
            
            # After hook
            result = self._after_execute(result, metrics, **kwargs)
            
            # Update totals
            self._total_executions += 1
            self._total_successes += 1
            self._total_duration_ms += metrics.duration_ms
            
            self._log.info(
                "Execution completed",
                duration_ms=f"{metrics.duration_ms:.2f}",
                status="success",
            )
            
            return AgentResult(
                data=result,
                success=True,
                status=AgentStatus.SUCCESS,
                metrics=metrics,
            )
            
        except Exception as e:
            # Record failure metrics
            metrics.end_time = time.time()
            metrics.duration_ms = (metrics.end_time - metrics.start_time) * 1000
            metrics.status = AgentStatus.FAILED
            metrics.error_message = str(e)
            
            self._log.error(
                f"Execution failed: {e}",
                duration_ms=f"{metrics.duration_ms:.2f}",
                error_type=type(e).__name__,
            )
            
            # Update totals
            self._total_executions += 1
            self._total_failures += 1
            self._total_duration_ms += metrics.duration_ms
            
            # Try error recovery
            fallback = self._on_error(e, metrics, **kwargs)
            if fallback is not None:
                self._log.info("Error recovery provided fallback result")
                return AgentResult(
                    data=fallback,
                    success=True,
                    status=AgentStatus.PARTIAL,
                    warnings=[f"Recovered from error: {e}"],
                    metrics=metrics,
                )
            
            return AgentResult(
                data=None,
                success=False,
                status=AgentStatus.FAILED,
                error=str(e),
                metrics=metrics,
            )
    
    @contextmanager
    def track_llm_call(self, metrics: AgentMetrics) -> Iterator[None]:
        """
        Context manager to track LLM calls.
        
        Usage:
            with self.track_llm_call(metrics):
                response = self._llm.chat(...)
        """
        metrics.llm_calls += 1
        yield
    
    @contextmanager
    def track_retrieval_call(self, metrics: AgentMetrics) -> Iterator[None]:
        """
        Context manager to track retrieval calls.
        
        Usage:
            with self.track_retrieval_call(metrics):
                results = self._store.retrieve_by_embedding(...)
        """
        metrics.retrieval_calls += 1
        yield
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        avg_duration = (
            self._total_duration_ms / self._total_executions
            if self._total_executions > 0
            else 0.0
        )
        success_rate = (
            self._total_successes / self._total_executions
            if self._total_executions > 0
            else 0.0
        )
        
        return {
            "name": self.name,
            "category": self.category.value,
            "enabled": self._enabled,
            "total_executions": self._total_executions,
            "total_successes": self._total_successes,
            "total_failures": self._total_failures,
            "success_rate": success_rate,
            "total_duration_ms": self._total_duration_ms,
            "avg_duration_ms": avg_duration,
        }
    
    def reset_statistics(self) -> None:
        """Reset execution statistics."""
        self._total_executions = 0
        self._total_successes = 0
        self._total_failures = 0
        self._total_duration_ms = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert agent metadata to dictionary.
        
        Useful for registration and discovery.
        """
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "enabled": self._enabled,
            "has_llm": self._llm is not None,
            "has_store": self._store is not None,
            "has_local_models": self._local_models is not None,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, enabled={self._enabled})"


class LLMAgent(BaseAgent):
    """
    Base class for agents that primarily use LLM for processing.
    
    Provides convenience methods for LLM interactions including
    JSON parsing and structured prompting.
    """
    
    def __init__(
        self,
        llm: "LLMClient",
        **kwargs: Any,
    ) -> None:
        """
        Initialize LLM agent.
        
        Args:
            llm: LLM client (required)
            **kwargs: Additional arguments for BaseAgent
        """
        if llm is None:
            raise ValueError(f"{self.__class__.__name__} requires an LLM client")
        super().__init__(llm=llm, **kwargs)
    
    @property
    def category(self) -> AgentCategory:
        """Default category for LLM agents."""
        return AgentCategory.GENERATION
    
    def _chat(
        self,
        system: str,
        user: str,
        metrics: Optional[AgentMetrics] = None,
    ) -> str:
        """
        Send a chat request to the LLM.
        
        Args:
            system: System prompt
            user: User message
            metrics: Optional metrics to track the call
            
        Returns:
            LLM response content
            
        Raises:
            RuntimeError: If LLM call fails
        """
        if metrics:
            metrics.llm_calls += 1
        
        response = self._llm.chat(
            self._llm.create_messages(system, user),
            retry_on_error=True,
        )
        
        if not response.success:
            raise RuntimeError(f"LLM call failed: {response.error}")
        
        return response.content.strip()
    
    def _chat_json(
        self,
        system: str,
        user: str,
        default: Any,
        expected_type: type,
        metrics: Optional[AgentMetrics] = None,
    ) -> Any:
        """
        Send a chat request expecting JSON response.
        
        Args:
            system: System prompt
            user: User message
            default: Default value if parsing fails
            expected_type: Expected type (dict or list)
            metrics: Optional metrics to track the call
            
        Returns:
            Parsed JSON data
        """
        if metrics:
            metrics.llm_calls += 1
        
        result, response = self._llm.chat_json(
            system=system,
            user=user,
            default=default,
            expected_type=expected_type,
        )
        
        return result


class RetrievalAgent(BaseAgent):
    """
    Base class for agents that perform document retrieval.
    
    Provides convenience methods for embedding and retrieval operations.
    """
    
    def __init__(
        self,
        store: "BaseVectorStore",
        local_models: "LocalNLPModels",
        **kwargs: Any,
    ) -> None:
        """
        Initialize retrieval agent.
        
        Args:
            store: Vector store (required)
            local_models: Local NLP models for embeddings (required)
            **kwargs: Additional arguments for BaseAgent
        """
        if store is None:
            raise ValueError(f"{self.__class__.__name__} requires a vector store")
        if local_models is None:
            raise ValueError(f"{self.__class__.__name__} requires local models")
        super().__init__(store=store, local_models=local_models, **kwargs)
    
    @property
    def category(self) -> AgentCategory:
        """Default category for retrieval agents."""
        return AgentCategory.RETRIEVAL
    
    def _embed(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self._local_models.embed_single(text)
    
    def _retrieve(
        self,
        query_embedding: List[float],
        top_k: int,
        min_similarity: float = 0.0,
        doc_level_filter: Optional[str] = None,
        metrics: Optional[AgentMetrics] = None,
    ) -> List[Any]:
        """
        Retrieve documents by embedding similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Maximum results
            min_similarity: Minimum similarity threshold
            doc_level_filter: Optional document level filter
            metrics: Optional metrics to track the call
            
        Returns:
            List of (document, score) tuples
        """
        if metrics:
            metrics.retrieval_calls += 1
        
        return self._store.retrieve_by_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
            min_similarity=min_similarity,
            doc_level_filter=doc_level_filter,
        )


# Export all public classes
__all__ = [
    "AgentCategory",
    "AgentStatus",
    "AgentMetrics",
    "AgentResult",
    "StructuredLogger",
    "BaseAgent",
    "LLMAgent",
    "RetrievalAgent",
]
