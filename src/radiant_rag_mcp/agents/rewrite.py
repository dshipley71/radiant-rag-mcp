"""
Query rewrite agent for RAG pipeline.

Transforms queries to improve retrieval effectiveness.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple, TYPE_CHECKING

from radiant_rag_mcp.agents.base_agent import (
    AgentCategory,
    AgentMetrics,
    LLMAgent,
)

if TYPE_CHECKING:
    from radiant_rag_mcp.llm.client import LLMClient

logger = logging.getLogger(__name__)


class QueryRewriteAgent(LLMAgent):
    """
    Rewrites queries to improve retrieval effectiveness.

    Transforms queries to be more specific, remove ambiguity,
    or better match document terminology.
    """

    def __init__(
        self,
        llm: "LLMClient",
        enabled: bool = True,
    ) -> None:
        """
        Initialize the rewrite agent.
        
        Args:
            llm: LLM client for reasoning
            enabled: Whether the agent is enabled
        """
        super().__init__(llm=llm, enabled=enabled)

    @property
    def name(self) -> str:
        """Return the agent's unique name."""
        return "QueryRewriteAgent"

    @property
    def category(self) -> AgentCategory:
        """Return the agent's category."""
        return AgentCategory.QUERY_PROCESSING

    @property
    def description(self) -> str:
        """Return a human-readable description."""
        return "Rewrites queries to improve retrieval effectiveness"

    def _execute(
        self,
        query: str,
        **kwargs: Any,
    ) -> Tuple[str, str]:
        """
        Rewrite query for better retrieval.

        Args:
            query: Original query

        Returns:
            Tuple of (original_query, rewritten_query)
        """
        system = """You are a QueryRewriteAgent.
Rewrite the query to maximize retrieval precision while preserving the original meaning.

Consider:
- Making implicit concepts explicit
- Using more specific terminology
- Removing filler words
- Clarifying ambiguous references

Return ONLY raw JSON object with no markdown formatting. Do not wrap in ```json code blocks.
Format: {"before": "original query", "after": "rewritten query"}"""

        user = f"Query: {query}\n\nReturn raw JSON only, no code blocks."

        result = self._chat_json(
            system=system,
            user=user,
            default={"before": query, "after": query},
            expected_type=dict,
        )

        if not result:
            return query, query

        before = str(result.get("before", query)).strip() or query
        after = str(result.get("after", query)).strip() or query

        if before != after:
            self.logger.info(
                "Query rewritten",
                original=before[:50],
                rewritten=after[:50],
            )

        return before, after

    def rewrite_batch(self, queries: list[str]) -> list[Tuple[str, str]]:
        """
        Rewrite multiple queries in a single LLM call for better performance.

        Args:
            queries: List of queries to rewrite

        Returns:
            List of tuples (original_query, rewritten_query)
        """
        if not queries:
            return []

        if len(queries) == 1:
            return [self._execute(queries[0])]

        # PERFORMANCE OPTIMIZATION: Batch rewrite in single LLM call
        system = """You are a QueryRewriteAgent.
Rewrite each query to maximize retrieval precision while preserving the original meaning.

Consider:
- Making implicit concepts explicit
- Using more specific terminology
- Removing filler words
- Clarifying ambiguous references

Return ONLY raw JSON object with no markdown formatting. Do not wrap in ```json code blocks.
Format: {"rewrites": [{"before": "original", "after": "rewritten"}, ...]}"""

        queries_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(queries)])
        user = f"Queries to rewrite:\n{queries_text}\n\nReturn raw JSON only, no code blocks."

        result = self._chat_json(
            system=system,
            user=user,
            default={"rewrites": [{"before": q, "after": q} for q in queries]},
            expected_type=dict,
        )

        if not result or "rewrites" not in result:
            return [(q, q) for q in queries]

        rewrites = result.get("rewrites", [])
        output = []

        for i, query in enumerate(queries):
            if i < len(rewrites) and isinstance(rewrites[i], dict):
                before = str(rewrites[i].get("before", query)).strip() or query
                after = str(rewrites[i].get("after", query)).strip() or query
                output.append((before, after))

                if before != after:
                    self.logger.info(
                        "Query rewritten (batch)",
                        original=before[:50],
                        rewritten=after[:50],
                    )
            else:
                output.append((query, query))

        return output

    def _on_error(
        self,
        error: Exception,
        metrics: AgentMetrics,
        **kwargs: Any,
    ) -> Optional[Tuple[str, str]]:
        """
        Return original query on error.
        """
        query = kwargs.get("query", "")
        self.logger.warning(f"Rewrite failed, using original: {error}")
        return (query, query) if query else ("", "")
