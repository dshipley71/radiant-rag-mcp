"""
Hierarchical auto-merging agent for RAG pipeline.

Merges child chunks into parent documents when appropriate.
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
    from radiant_rag_mcp.config import AutoMergeConfig
    from radiant_rag_mcp.storage.base import BaseVectorStore

logger = logging.getLogger(__name__)


class HierarchicalAutoMergingAgent(BaseAgent):
    """
    Auto-merges child chunks into parent documents.

    When multiple children from the same parent are retrieved,
    replaces them with the parent document for better context.
    """

    def __init__(
        self,
        store: "BaseVectorStore",
        config: "AutoMergeConfig",
        enabled: bool = True,
    ) -> None:
        """
        Initialize the auto-merge agent.
        
        Args:
            store: Vector store for retrieving parent documents
            config: Auto-merge configuration
            enabled: Whether the agent is enabled
        """
        super().__init__(store=store, enabled=enabled)
        self._config = config

    @property
    def name(self) -> str:
        """Return the agent's unique name."""
        return "HierarchicalAutoMergingAgent"

    @property
    def category(self) -> AgentCategory:
        """Return the agent's category."""
        return AgentCategory.POST_RETRIEVAL

    @property
    def description(self) -> str:
        """Return a human-readable description."""
        return "Auto-merges child chunks into parent documents when appropriate"

    def _execute(
        self,
        candidates: List[Tuple[Any, float]],
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Tuple[Any, float]]:
        """
        Apply hierarchical auto-merging.

        Args:
            candidates: Retrieved documents
            top_k: Maximum results

        Returns:
            Documents with auto-merging applied
        """
        if not candidates:
            return []

        k = top_k or len(candidates)
        min_children = self._config.min_children_to_merge
        max_parent_chars = self._config.max_parent_chars

        # Group children by parent
        by_parent: Dict[str, List[Tuple[Any, float]]] = {}
        passthrough: List[Tuple[Any, float]] = []

        for doc, score in candidates:
            parent_id = str(doc.meta.get("parent_id", "")).strip()
            doc_level = str(doc.meta.get("doc_level", "child"))

            if doc_level == "child" and parent_id:
                by_parent.setdefault(parent_id, []).append((doc, score))
            else:
                passthrough.append((doc, score))

        merged: List[Tuple[Any, float]] = []
        parents_merged = 0

        for parent_id, children in by_parent.items():
            if len(children) >= min_children:
                parent = self._store.get_doc(parent_id)
                if parent is not None and len(parent.content) <= max_parent_chars:
                    best_score = max(s for _, s in children)
                    merged.append((parent, best_score))
                    parents_merged += 1
                else:
                    merged.extend(children)
            else:
                merged.extend(children)

        # Combine and deduplicate
        combined = passthrough + merged
        best_by_id: Dict[str, Tuple[Any, float]] = {}

        for doc, score in combined:
            existing = best_by_id.get(doc.doc_id)
            if existing is None or score > existing[1]:
                best_by_id[doc.doc_id] = (doc, score)

        result = list(best_by_id.values())
        result.sort(key=lambda x: x[1], reverse=True)

        final_results = result[:k]

        self.logger.info(
            "Auto-merge completed",
            input_docs=len(candidates),
            parents_merged=parents_merged,
            output_docs=len(final_results),
        )

        return final_results

    def _on_error(
        self,
        error: Exception,
        metrics: AgentMetrics,
        **kwargs: Any,
    ) -> Optional[List[Tuple[Any, float]]]:
        """
        Return original candidates on error.
        """
        self.logger.warning(f"Auto-merge failed: {error}")
        candidates = kwargs.get("candidates", [])
        return candidates if candidates else []
