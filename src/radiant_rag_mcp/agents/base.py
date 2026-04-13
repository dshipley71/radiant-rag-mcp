"""
Base classes and context for RAG pipeline agents.

Provides the AgentContext dataclass that accumulates results
as the query flows through each pipeline stage.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """
    Context object passed through the RAG pipeline.

    Accumulates results from each agent stage.
    """

    run_id: str
    original_query: str
    conversation_id: Optional[str] = None

    # Query processing results
    plan: Dict[str, Any] = field(default_factory=dict)
    decomposed_queries: List[str] = field(default_factory=list)
    rewrites: List[Tuple[str, str]] = field(default_factory=list)
    expansions: List[str] = field(default_factory=list)

    # Retrieval results - using Any to avoid import issues
    dense_retrieved: List[Tuple[Any, float]] = field(default_factory=list)
    bm25_retrieved: List[Tuple[Any, float]] = field(default_factory=list)
    web_search_retrieved: List[Tuple[Any, float]] = field(default_factory=list)
    fused: List[Tuple[Any, float]] = field(default_factory=list)
    auto_merged: List[Tuple[Any, float]] = field(default_factory=list)
    reranked: List[Tuple[Any, float]] = field(default_factory=list)

    # Generation results
    final_answer: Optional[str] = None
    critic_notes: List[Dict[str, Any]] = field(default_factory=list)

    # Conversation history
    conversation_history: str = ""

    # Metadata
    warnings: List[str] = field(default_factory=list)
    
    # === Agentic enhancements ===
    
    # Confidence tracking
    answer_confidence: float = 0.0
    retrieval_confidence: float = 0.0
    
    # Retrieval mode used (can be dynamically selected)
    retrieval_mode: str = "hybrid"
    
    # Retry tracking
    retry_count: int = 0
    retry_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Tool usage
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Whether we fell back to "I don't know"
    low_confidence_response: bool = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
        logger.warning(f"[{self.run_id}] {message}")
    
    def record_retry(
        self,
        reason: str,
        previous_confidence: float,
        modifications: Dict[str, Any],
    ) -> None:
        """Record a retry attempt."""
        self.retry_count += 1
        self.retry_history.append({
            "attempt": self.retry_count,
            "reason": reason,
            "previous_confidence": previous_confidence,
            "modifications": modifications,
        })
        logger.info(f"[{self.run_id}] Retry #{self.retry_count}: {reason}")
    
    def add_tool_result(self, result: Dict[str, Any]) -> None:
        """Add a tool execution result."""
        self.tool_results.append(result)
    
    def get_confidence_summary(self) -> Dict[str, Any]:
        """Get a summary of confidence metrics."""
        return {
            "answer_confidence": self.answer_confidence,
            "retrieval_confidence": self.retrieval_confidence,
            "low_confidence_response": self.low_confidence_response,
            "retry_count": self.retry_count,
        }


def new_agent_context(
    query: str,
    conversation_id: Optional[str] = None,
) -> AgentContext:
    """
    Create a new agent context for a query.

    Args:
        query: User query
        conversation_id: Optional conversation ID

    Returns:
        New AgentContext instance
    """
    return AgentContext(
        run_id=str(uuid.uuid4()),
        original_query=query,
        conversation_id=conversation_id,
    )
