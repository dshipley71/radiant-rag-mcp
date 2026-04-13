"""
Cross-encoder reranking agent for RAG pipeline.

Provides more accurate relevance scoring using cross-encoder models.
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
    from radiant_rag_mcp.config import RerankConfig
    from radiant_rag_mcp.llm.client import LocalNLPModels

logger = logging.getLogger(__name__)


class CrossEncoderRerankingAgent(BaseAgent):
    """
    Reranks documents using a cross-encoder model.

    Provides more accurate relevance scoring than bi-encoder similarity.
    """

    def __init__(
        self,
        local: "LocalNLPModels",
        config: "RerankConfig",
        enabled: bool = True,
    ) -> None:
        """
        Initialize the reranking agent.
        
        Args:
            local: Local NLP models for reranking
            config: Reranking configuration
            enabled: Whether the agent is enabled
        """
        super().__init__(local_models=local, enabled=enabled)
        self._config = config

    @property
    def name(self) -> str:
        """Return the agent's unique name."""
        return "CrossEncoderRerankingAgent"

    @property
    def category(self) -> AgentCategory:
        """Return the agent's category."""
        return AgentCategory.POST_RETRIEVAL

    @property
    def description(self) -> str:
        """Return a human-readable description."""
        return "Reranks documents using cross-encoder for more accurate relevance"

    def _execute(
        self,
        query: str,
        docs: List[Tuple[Any, float]],
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Tuple[Any, float]]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: Query text
            docs: Documents to rerank
            top_k: Maximum results

        Returns:
            Reranked documents
        """
        if not docs:
            return []

        k = top_k or self._config.top_k
        max_doc_chars = self._config.max_doc_chars

        # Determine candidate count
        num_candidates = max(
            k * self._config.candidate_multiplier,
            self._config.min_candidates,
        )
        candidates = docs[:num_candidates]

        # Prepare texts for reranking
        doc_texts = [
            doc.content[:max_doc_chars] if len(doc.content) > max_doc_chars else doc.content
            for doc, _ in candidates
        ]

        # Get reranking scores
        rerank_scores = self._local_models.rerank(query, doc_texts, top_k=k)

        # Map back to documents
        result = [
            (candidates[idx][0], score)
            for idx, score in rerank_scores
        ]

        self.logger.info(
            "Reranking completed",
            input_docs=len(docs),
            candidates_scored=len(candidates),
            output_docs=len(result),
        )

        return result

    def _on_error(
        self,
        error: Exception,
        metrics: AgentMetrics,
        **kwargs: Any,
    ) -> Optional[List[Tuple[Any, float]]]:
        """
        Return original docs on error.
        """
        self.logger.warning(f"Reranking failed: {error}")
        docs = kwargs.get("docs", [])
        top_k = kwargs.get("top_k") or self._config.top_k
        return docs[:top_k] if docs else []
