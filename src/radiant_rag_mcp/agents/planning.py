"""
Planning agent for RAG pipeline.

Analyzes queries and produces execution plans that control
which pipeline features are activated, including dynamic
retrieval mode selection and tool usage.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from radiant_rag_mcp.agents.base_agent import (
    AgentCategory,
    AgentMetrics,
    LLMAgent,
)

if TYPE_CHECKING:
    from radiant_rag_mcp.llm.client import LLMClient
    from radiant_rag_mcp.agents.strategy_memory import RetrievalStrategyMemory

logger = logging.getLogger(__name__)


class PlanningAgent(LLMAgent):
    """
    Planning agent that decides which pipeline features to use.

    Analyzes the query and produces a plan that enables/disables
    various pipeline stages based on query characteristics.
    
    Enhanced with:
    - Dynamic retrieval mode selection (dense/bm25/hybrid)
    - Tool selection for calculators, code execution
    - Strategy memory integration for adaptive behavior
    """

    def __init__(
        self,
        llm: "LLMClient",
        web_search_enabled: bool = False,
        tools_enabled: bool = True,
        available_tools: Optional[List[Dict[str, Any]]] = None,
        strategy_memory: Optional["RetrievalStrategyMemory"] = None,
        enabled: bool = True,
    ) -> None:
        """
        Initialize the planning agent.
        
        Args:
            llm: LLM client for reasoning
            web_search_enabled: Whether web search is available
            tools_enabled: Whether tools are available
            available_tools: List of available tool definitions
            strategy_memory: Optional strategy memory for adaptive behavior
            enabled: Whether the agent is enabled
        """
        super().__init__(llm=llm, enabled=enabled)
        self._web_search_enabled = web_search_enabled
        self._tools_enabled = tools_enabled
        self._available_tools = available_tools or []
        self._strategy_memory = strategy_memory

    @property
    def name(self) -> str:
        """Return the agent's unique name."""
        return "PlanningAgent"

    @property
    def category(self) -> AgentCategory:
        """Return the agent's category."""
        return AgentCategory.PLANNING

    @property
    def description(self) -> str:
        """Return a human-readable description."""
        return "Analyzes queries and produces execution plans for the RAG pipeline"

    def _execute(
        self,
        query: str,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate execution plan for query.

        Args:
            query: User query
            context: Optional additional context

        Returns:
            Plan dictionary with feature flags and retrieval mode
        """
        # Get strategy recommendation from memory if available
        recommended_mode = "hybrid"
        mode_confidence = 0.5
        
        if self._strategy_memory:
            recommended_mode, mode_confidence = self._strategy_memory.recommend_strategy(query)
            self.logger.debug(
                f"Strategy memory recommends: {recommended_mode}",
                confidence=f"{mode_confidence:.2f}",
            )

        # Build dynamic parts of prompt
        web_search_instruction = ""
        if self._web_search_enabled:
            web_search_instruction = """
- use_web_search: Search the web for current/recent information (use for queries about recent events, news, current status, or when indexed content may be outdated)"""

        tools_instruction = ""
        if self._tools_enabled and self._available_tools:
            tools_list = "\n".join([
                f"    - {t['name']}: {t['description']}"
                for t in self._available_tools
            ])
            tools_instruction = f"""
- tools_to_use: List of tools to invoke (optional). Available tools:
{tools_list}"""

        strategy_hint = ""
        if mode_confidence > 0.6:
            strategy_hint = f"""

Based on historical performance, "{recommended_mode}" retrieval mode has worked well for similar queries.
Consider this when selecting retrieval_mode, but override if query characteristics suggest otherwise."""

        system = f"""You are a PlanningAgent for an agentic RAG system.
Analyze the query and produce a JSON plan that decides which features to use.

Available features:
- use_decomposition: Break complex queries into sub-queries
- use_rewrite: Rewrite query for better retrieval
- use_expansion: Add synonyms and related terms
- use_rrf: Use Reciprocal Rank Fusion to combine retrieval methods
- use_automerge: Merge child chunks into parent documents when appropriate
- use_rerank: Use cross-encoder reranking
- use_critic: Evaluate answer quality{web_search_instruction}{tools_instruction}

Retrieval mode selection:
- retrieval_mode: One of "hybrid", "dense", or "bm25"
  - "hybrid": Best for most queries, combines semantic and keyword search
  - "dense": Best for conceptual/semantic queries ("what is the meaning of...")
  - "bm25": Best for specific terms, names, codes, exact phrases
{strategy_hint}
Consider:
- Simple factual queries may not need decomposition
- Queries with multiple parts benefit from decomposition
- Ambiguous queries benefit from rewriting and expansion
- Complex queries benefit from all features
- Queries about recent events, news, or current information benefit from web search
- Queries with words like "latest", "recent", "current", "today", "news" may need web search
- Mathematical queries should use the calculator tool
- Queries requiring data manipulation may benefit from code execution

Return ONLY raw JSON with no markdown formatting. Do not wrap in ```json code blocks."""

        user = f"Query: {query}"
        if context:
            user += f"\n\nContext: {context}"
        user += "\n\nReturn raw JSON only, no code blocks."

        result = self._chat_json(
            system=system,
            user=user,
            default={},
            expected_type=dict,
        )

        # Ensure all required keys exist with defaults
        default_plan = {
            "use_decomposition": True,
            "use_rewrite": True,
            "use_expansion": True,
            "use_rrf": True,
            "use_automerge": True,
            "use_rerank": True,
            "use_critic": True,
            "use_web_search": False,
            "retrieval_mode": recommended_mode,
            "tools_to_use": [],
        }

        if not result:
            self.logger.warning("Planning failed, using default plan")
            return default_plan

        # Merge with defaults
        for key in default_plan:
            if key not in result:
                result[key] = default_plan[key]
            elif key == "retrieval_mode":
                # Validate retrieval mode
                if result[key] not in ("hybrid", "dense", "bm25"):
                    result[key] = recommended_mode
            elif key == "tools_to_use":
                # Validate tools
                if not isinstance(result[key], list):
                    result[key] = []
                else:
                    # Filter to valid tools
                    valid_tool_names = {t["name"] for t in self._available_tools}
                    result[key] = [t for t in result[key] if t in valid_tool_names]
            else:
                result[key] = bool(result[key])

        # Only allow web search if enabled in config
        if not self._web_search_enabled:
            result["use_web_search"] = False
        
        # Only allow tools if enabled
        if not self._tools_enabled:
            result["tools_to_use"] = []

        self.logger.info(
            "Plan generated",
            retrieval_mode=result["retrieval_mode"],
            tools=result.get("tools_to_use", []),
        )
        
        return result
    
    def plan_retry(
        self,
        query: str,
        previous_plan: Dict[str, Any],
        critique: Dict[str, Any],
        retry_count: int,
    ) -> Dict[str, Any]:
        """
        Generate a modified plan for retry attempt.
        
        Args:
            query: Original query
            previous_plan: Plan used in previous attempt
            critique: Critique from previous attempt
            retry_count: Current retry count
            
        Returns:
            Modified plan for retry
        """
        retry_suggestions = critique.get("retry_suggestions", [])
        issues = critique.get("issues", [])
        
        system = f"""You are a PlanningAgent adjusting a RAG pipeline plan after a failed attempt.

Previous plan: {previous_plan}
Issues found: {issues}
Suggestions: {retry_suggestions}
Retry attempt: {retry_count}

Modify the plan to address the issues. Consider:
- Switching retrieval mode if current mode isn't working
- Enabling query rewriting if terms might be wrong
- Enabling expansion if query is too narrow
- Using web search if content might be outdated

Return ONLY raw JSON object with no markdown formatting. Do not wrap in ```json code blocks."""

        user = f"Query: {query}\n\nReturn raw JSON plan only, no code blocks."

        result = self._chat_json(
            system=system,
            user=user,
            default=previous_plan,
            expected_type=dict,
        )

        if not result or result == previous_plan:
            # Simple fallback: toggle retrieval mode
            modified = dict(previous_plan)
            current_mode = modified.get("retrieval_mode", "hybrid")
            
            # Cycle through modes
            mode_cycle = {"hybrid": "dense", "dense": "bm25", "bm25": "hybrid"}
            modified["retrieval_mode"] = mode_cycle.get(current_mode, "hybrid")
            modified["use_rewrite"] = True
            modified["use_expansion"] = True
            
            return modified

        # Merge with previous plan
        for key in previous_plan:
            if key not in result:
                result[key] = previous_plan[key]

        return result

    def _on_error(
        self,
        error: Exception,
        metrics: AgentMetrics,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Provide fallback plan on error.
        """
        self.logger.warning(f"Planning failed, using fallback: {error}")
        return {
            "use_decomposition": True,
            "use_rewrite": True,
            "use_expansion": True,
            "use_rrf": True,
            "use_automerge": True,
            "use_rerank": True,
            "use_critic": True,
            "use_web_search": False,
            "retrieval_mode": "hybrid",
            "tools_to_use": [],
        }
