"""
Query decomposition agent for RAG pipeline.

Breaks complex queries into simpler sub-queries for better retrieval.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, TYPE_CHECKING

from radiant_rag_mcp.agents.base_agent import (
    AgentCategory,
    AgentMetrics,
    LLMAgent,
)

if TYPE_CHECKING:
    from radiant_rag_mcp.config import QueryConfig
    from radiant_rag_mcp.llm.client import LLMClient

logger = logging.getLogger(__name__)


class QueryDecompositionAgent(LLMAgent):
    """
    Decomposes complex queries into simpler sub-queries.

    Useful for multi-part questions or queries requiring
    information from multiple sources.
    """

    def __init__(
        self,
        llm: "LLMClient",
        config: "QueryConfig",
        enabled: bool = True,
    ) -> None:
        """
        Initialize the decomposition agent.
        
        Args:
            llm: LLM client for reasoning
            config: Query configuration
            enabled: Whether the agent is enabled
        """
        super().__init__(llm=llm, enabled=enabled)
        self._config = config

    @property
    def name(self) -> str:
        """Return the agent's unique name."""
        return "QueryDecompositionAgent"

    @property
    def category(self) -> AgentCategory:
        """Return the agent's category."""
        return AgentCategory.QUERY_PROCESSING

    @property
    def description(self) -> str:
        """Return a human-readable description."""
        return "Decomposes complex queries into simpler sub-queries"

    def _execute(
        self,
        query: str,
        **kwargs: Any,
    ) -> List[str]:
        """
        Decompose query into sub-queries.

        Args:
            query: Original query

        Returns:
            List of sub-queries (may be single element if no decomposition needed)
        """
        system = """You are a QueryDecompositionAgent.
If the query is complex or contains multiple distinct questions, decompose it into independent sub-queries.
If the query is simple and doesn't need decomposition, return it as-is.

Return ONLY raw JSON array with no markdown formatting. Do not wrap in ```json code blocks.
Each string should be a complete, self-contained query.
Maximum sub-queries: Return at most 5 sub-queries.

Examples:
- "What is Python and how does it compare to Java?" -> ["What is Python?", "How does Python compare to Java?"]
- "Tell me about climate change" -> ["Tell me about climate change"]"""

        user = f"Query: {query}\n\nReturn raw JSON array only, no code blocks."

        result = self._chat_json(
            system=system,
            user=user,
            default=[query],
            expected_type=list,
        )

        # Validate and clean results
        if isinstance(result, list):
            queries = [str(q).strip() for q in result if isinstance(q, str) and q.strip()]
            if queries:
                final_queries = queries[: self._config.max_decomposed_queries]
                self.logger.info(
                    "Query decomposed",
                    original=query[:50],
                    num_sub_queries=len(final_queries),
                )
                return final_queries

        return [query]

    def _on_error(
        self,
        error: Exception,
        metrics: AgentMetrics,
        **kwargs: Any,
    ) -> Optional[List[str]]:
        """
        Return original query on error.
        """
        query = kwargs.get("query", "")
        self.logger.warning(f"Decomposition failed, using original: {error}")
        return [query] if query else []
