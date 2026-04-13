"""
BM25 retrieval agent for RAG pipeline.

Uses sparse keyword-based retrieval.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

from radiant_rag_mcp.agents.base_agent import (
    AgentCategory,
    AgentMetrics,
    BaseAgent,
)

if TYPE_CHECKING:
    from radiant_rag_mcp.config import RetrievalConfig
    from radiant_rag_mcp.storage.bm25_index import PersistentBM25Index

logger = logging.getLogger(__name__)


class BM25RetrievalAgent(BaseAgent):
    """
    Sparse keyword-based retrieval using BM25.
    """

    def __init__(
        self,
        bm25_index: "PersistentBM25Index",
        config: "RetrievalConfig",
        enabled: bool = True,
    ) -> None:
        """
        Initialize the BM25 retrieval agent.
        
        Args:
            bm25_index: BM25 index for retrieval
            config: Retrieval configuration
            enabled: Whether the agent is enabled
        """
        super().__init__(enabled=enabled)
        self._index = bm25_index
        self._config = config

    @property
    def name(self) -> str:
        """Return the agent's unique name."""
        return "BM25RetrievalAgent"

    @property
    def category(self) -> AgentCategory:
        """Return the agent's category."""
        return AgentCategory.RETRIEVAL

    @property
    def description(self) -> str:
        """Return a human-readable description."""
        return "Sparse keyword-based retrieval using BM25"

    def _execute(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Tuple[Any, float]]:
        """
        Retrieve documents using BM25.

        Args:
            query: Query text
            top_k: Maximum results (defaults to config)

        Returns:
            List of (document, score) tuples
        """
        k = top_k or self._config.bm25_top_k
        results = self._index.search(query, top_k=k)

        self.logger.info(
            "BM25 retrieval completed",
            query_length=len(query),
            num_results=len(results),
            top_k=k,
        )

        return results

    def _on_error(
        self,
        error: Exception,
        metrics: AgentMetrics,
        **kwargs: Any,
    ) -> Optional[List[Tuple[Any, float]]]:
        """
        Return empty list on error.
        """
        self.logger.warning(f"BM25 retrieval failed: {error}")
        return []
