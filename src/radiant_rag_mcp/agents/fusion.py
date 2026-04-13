"""
Reciprocal Rank Fusion agent for RAG pipeline.

Combines results from multiple retrieval methods using RRF.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from radiant_rag_mcp.agents.base_agent import (
    AgentCategory,
    AgentMetrics,
    BaseAgent,
)

if TYPE_CHECKING:
    from radiant_rag_mcp.config import RetrievalConfig

logger = logging.getLogger(__name__)


class RRFAgent(BaseAgent):
    """
    Reciprocal Rank Fusion agent for combining retrieval results.

    Merges results from multiple retrieval methods using the RRF formula.
    """

    def __init__(
        self,
        config: "RetrievalConfig",
        enabled: bool = True,
    ) -> None:
        """
        Initialize the RRF agent.
        
        Args:
            config: Retrieval configuration
            enabled: Whether the agent is enabled
        """
        super().__init__(enabled=enabled)
        self._config = config

    @property
    def name(self) -> str:
        """Return the agent's unique name."""
        return "RRFAgent"

    @property
    def category(self) -> AgentCategory:
        """Return the agent's category."""
        return AgentCategory.POST_RETRIEVAL

    @property
    def description(self) -> str:
        """Return a human-readable description."""
        return "Combines retrieval results using Reciprocal Rank Fusion"

    def _execute(
        self,
        runs: List[List[Tuple[Any, float]]],
        top_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Tuple[Any, float]]:
        """
        Fuse multiple retrieval runs using RRF.

        Args:
            runs: List of retrieval results from different methods
            top_k: Maximum results (defaults to config)
            rrf_k: RRF constant (defaults to config)

        Returns:
            Fused results sorted by RRF score
        """
        k = top_k or self._config.fused_top_k
        rrf_constant = rrf_k or self._config.rrf_k

        scores: Dict[str, float] = {}
        doc_map: Dict[str, Any] = {}

        for run in runs:
            for rank, (doc, _score) in enumerate(run, start=1):
                doc_map[doc.doc_id] = doc
                scores[doc.doc_id] = scores.get(doc.doc_id, 0.0) + (1.0 / (rrf_constant + rank))

        fused = [(doc_map[doc_id], score) for doc_id, score in scores.items()]
        fused.sort(key=lambda x: x[1], reverse=True)

        results = fused[:k]

        self.logger.info(
            "RRF fusion completed",
            num_runs=len(runs),
            total_docs=len(doc_map),
            fused_results=len(results),
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
        self.logger.warning(f"RRF fusion failed: {error}")
        return []
