"""
Dense retrieval agent for RAG pipeline.

Uses embedding-based vector similarity search.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

from radiant_rag_mcp.agents.base_agent import (
    AgentCategory,
    AgentMetrics,
    RetrievalAgent,
)

if TYPE_CHECKING:
    from radiant_rag_mcp.config import RetrievalConfig
    from radiant_rag_mcp.llm.client import LocalNLPModels
    from radiant_rag_mcp.storage.base import BaseVectorStore

logger = logging.getLogger(__name__)


class DenseRetrievalAgent(RetrievalAgent):
    """
    Dense embedding-based retrieval using vector similarity search.
    
    Supports searching leaves (child chunks), parents, or both based on
    the search_scope configuration.
    """

    def __init__(
        self,
        store: "BaseVectorStore",
        local: "LocalNLPModels",
        config: "RetrievalConfig",
        enabled: bool = True,
    ) -> None:
        """
        Initialize the dense retrieval agent.
        
        Args:
            store: Vector store for retrieval
            local: Local NLP models for embeddings
            config: Retrieval configuration
            enabled: Whether the agent is enabled
        """
        super().__init__(store=store, local_models=local, enabled=enabled)
        self._config = config

    @property
    def name(self) -> str:
        """Return the agent's unique name."""
        return "DenseRetrievalAgent"

    @property
    def category(self) -> AgentCategory:
        """Return the agent's category."""
        return AgentCategory.RETRIEVAL

    @property
    def description(self) -> str:
        """Return a human-readable description."""
        return "Dense embedding-based retrieval using vector similarity search"

    def _get_doc_level_filter(self, search_scope: Optional[str] = None) -> Optional[str]:
        """
        Convert search_scope to doc_level_filter.
        
        Args:
            search_scope: Override for config search_scope
            
        Returns:
            doc_level_filter value or None for no filtering
        """
        scope = search_scope or self._config.search_scope
        
        if scope == "leaves":
            return "child"
        elif scope == "parents":
            return "parent"
        elif scope == "all":
            return None
        else:
            # Default to leaves for backward compatibility
            return "child"

    def _execute(
        self,
        query: str,
        top_k: Optional[int] = None,
        search_scope: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Any, float]]:
        """
        Retrieve documents by embedding similarity.

        Args:
            query: Query text
            top_k: Maximum results (defaults to config)
            search_scope: Override search scope ("leaves", "parents", "all")

        Returns:
            List of (document, score) tuples
        """
        k = top_k or self._config.dense_top_k
        doc_level_filter = self._get_doc_level_filter(search_scope)

        # Generate query embedding
        query_vec = self._embed(query)

        # Search vector store
        results = self._retrieve(
            query_embedding=query_vec,
            top_k=k,
            min_similarity=self._config.min_similarity,
            doc_level_filter=doc_level_filter,
        )

        self.logger.info(
            "Dense retrieval completed",
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
        self.logger.warning(f"Dense retrieval failed: {error}")
        return []
