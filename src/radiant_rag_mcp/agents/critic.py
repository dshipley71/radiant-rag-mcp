"""
Critic agent for RAG pipeline.

Evaluates answer quality, provides confidence scores, and suggests improvements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from radiant_rag_mcp.agents.base_agent import (
    AgentCategory,
    AgentMetrics,
    LLMAgent,
)

if TYPE_CHECKING:
    from radiant_rag_mcp.config import CriticConfig
    from radiant_rag_mcp.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class CritiqueResult:
    """Result of answer critique with confidence metrics."""
    ok: bool
    confidence: float  # Overall confidence 0-1
    relevance_score: float  # 0-10
    faithfulness_score: float  # 0-10
    coverage_score: float  # 0-10
    issues: List[str]
    suggested_improvements: List[str]
    should_retry: bool
    retry_suggestions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "confidence": self.confidence,
            "relevance_score": self.relevance_score,
            "faithfulness_score": self.faithfulness_score,
            "coverage_score": self.coverage_score,
            "issues": self.issues,
            "suggested_improvements": self.suggested_improvements,
            "should_retry": self.should_retry,
            "retry_suggestions": self.retry_suggestions,
        }


class CriticAgent(LLMAgent):
    """
    Evaluates answer quality and provides confidence scoring.

    Checks for relevance, faithfulness to context, coverage,
    and provides actionable feedback for retry decisions.
    """

    def __init__(
        self,
        llm: "LLMClient",
        config: "CriticConfig",
        enabled: bool = True,
    ) -> None:
        """
        Initialize the critic agent.
        
        Args:
            llm: LLM client for evaluation
            config: Critic configuration
            enabled: Whether the agent is enabled
        """
        super().__init__(llm=llm, enabled=enabled)
        self._config = config

    @property
    def name(self) -> str:
        """Return the agent's unique name."""
        return "CriticAgent"

    @property
    def category(self) -> AgentCategory:
        """Return the agent's category."""
        return AgentCategory.EVALUATION

    @property
    def description(self) -> str:
        """Return a human-readable description."""
        return "Evaluates answer quality and provides confidence scoring"

    def _execute(
        self,
        query: str,
        answer: str,
        context_docs: List[Any],
        is_retry: bool = False,
        retry_count: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Critique the generated answer with confidence scoring.

        Args:
            query: Original query
            answer: Generated answer
            context_docs: Context documents used
            is_retry: Whether this is a retry attempt
            retry_count: Current retry count

        Returns:
            Critique dictionary with ok flag, confidence, and issues
        """
        max_docs = self._config.max_context_docs
        max_chars = self._config.max_doc_chars

        # Format context
        context_parts = []
        for i, doc in enumerate(context_docs[:max_docs], start=1):
            content = doc.content[:max_chars] if len(doc.content) > max_chars else doc.content
            context_parts.append(f"[DOC {i}] {content}")

        context = "\n\n".join(context_parts)
        
        # Adjust prompt based on retry status
        retry_context = ""
        if is_retry:
            retry_context = f"""
This is retry attempt #{retry_count}. Be especially critical and look for:
- Whether the new answer addressed previous issues
- Any remaining gaps in the response
- If further retrieval might help
"""

        system = f"""You are a CriticAgent for RAG systems with confidence scoring.
Evaluate the answer for:
1. Relevance: Does it address the question directly? (0-10)
2. Faithfulness: Is it fully supported by the context? (0-10)
3. Coverage: Does it cover all important aspects? (0-10)
4. Accuracy: Are there any factual errors or unsupported claims?
{retry_context}
Return ONLY raw JSON object with no markdown formatting. Do not wrap in ```json code blocks.
Format:
{{
  "ok": true/false (true if answer is acceptable),
  "confidence": 0.0-1.0 (overall confidence in the answer),
  "relevance_score": 0-10,
  "faithfulness_score": 0-10,
  "coverage_score": 0-10,
  "issues": ["list of specific issues found"],
  "suggested_improvements": ["list of specific suggestions"],
  "should_retry": true/false (whether retrieval should be retried),
  "retry_suggestions": ["what to change if retrying, e.g., 'search for X', 'use different keywords'"]
}}

Confidence scoring guidelines:
- 0.9-1.0: Excellent - fully addresses question with strong evidence
- 0.7-0.9: Good - addresses question well with adequate evidence
- 0.5-0.7: Moderate - partially addresses question, some gaps
- 0.3-0.5: Weak - significant gaps or unsupported claims
- 0.0-0.3: Poor - fails to address question or contradicts evidence

Set should_retry=true if:
- Critical information is missing that might be found with different search
- Answer quality is below acceptable threshold
- Context seems incomplete for the question asked"""

        user = f"""QUERY:
{query}

CONTEXT:
{context}

ANSWER:
{answer}

Return raw JSON critique only, no code blocks."""

        result = self._chat_json(
            system=system,
            user=user,
            default=self._default_result(),
            expected_type=dict,
        )

        if not result:
            return self._default_result()

        # Ensure all required fields with proper types
        result = self._normalize_result(result)
        
        # Log confidence level
        conf = result.get("confidence", 0.5)
        self.logger.info(
            "Critique completed",
            confidence=f"{conf:.2f}",
            ok=result.get("ok", True),
        )
        
        if not result.get("ok", True):
            issues = result.get("issues", [])
            if issues:
                self.logger.warning(f"Issues found: {issues}")

        return result
    
    def evaluate_retrieval_quality(
        self,
        query: str,
        docs: List[Any],
    ) -> float:
        """
        Evaluate the quality of retrieved documents.
        
        Args:
            query: Original query
            docs: Retrieved documents
            
        Returns:
            Retrieval confidence score 0-1
        """
        if not docs:
            return 0.0
        
        # Quick heuristic based on scores if available
        scores = []
        for item in docs[:10]:
            if isinstance(item, tuple):
                doc, score = item
                if isinstance(score, (int, float)):
                    scores.append(score)
            else:
                scores.append(0.5)
        
        if scores:
            avg_score = sum(scores) / len(scores)
            # Normalize to 0-1 range (assuming scores are typically 0-1 already)
            return min(1.0, max(0.0, avg_score))
        
        return 0.5  # Default confidence
    
    def _default_result(self) -> Dict[str, Any]:
        """Return default result for failed critiques."""
        return {
            "ok": True,
            "confidence": 0.5,
            "relevance_score": 5.0,
            "faithfulness_score": 5.0,
            "coverage_score": 5.0,
            "issues": [],
            "suggested_improvements": [],
            "should_retry": False,
            "retry_suggestions": [],
        }
    
    def _normalize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and validate critique result."""
        defaults = self._default_result()
        
        # Ensure all keys exist
        for key in defaults:
            if key not in result:
                result[key] = defaults[key]
        
        # Normalize types
        result["ok"] = bool(result.get("ok", True))
        result["confidence"] = float(result.get("confidence", 0.5))
        result["relevance_score"] = float(result.get("relevance_score", 5.0))
        result["faithfulness_score"] = float(result.get("faithfulness_score", 5.0))
        result["coverage_score"] = float(result.get("coverage_score", 5.0))
        result["should_retry"] = bool(result.get("should_retry", False))
        
        # Ensure lists
        for key in ["issues", "suggested_improvements", "retry_suggestions"]:
            if not isinstance(result.get(key), list):
                result[key] = []
        
        # Clamp confidence
        result["confidence"] = max(0.0, min(1.0, result["confidence"]))
        
        # Derive confidence from scores if not provided or seems wrong
        if result["confidence"] == 0.5:  # Default value, recalculate
            score_avg = (
                result["relevance_score"] +
                result["faithfulness_score"] +
                result["coverage_score"]
            ) / 30.0  # Normalize to 0-1
            result["confidence"] = score_avg
        
        return result
    
    def should_give_up(
        self,
        critique: Dict[str, Any],
        retry_count: int,
    ) -> bool:
        """
        Determine if we should give up and return "I don't know".
        
        Args:
            critique: Critique result
            retry_count: Current retry count
            
        Returns:
            True if we should give up
        """
        confidence = critique.get("confidence", 0.5)
        max_retries = self._config.max_retries
        threshold = self._config.confidence_threshold
        
        # Give up if we've exhausted retries and still low confidence
        if retry_count >= max_retries and confidence < threshold:
            return True
        
        # Give up if confidence is extremely low and not improving
        if confidence < 0.2 and retry_count > 0:
            return True
        
        return False

    def _on_error(
        self,
        error: Exception,
        metrics: AgentMetrics,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Provide fallback critique on error.
        """
        self.logger.warning(f"Critique failed, using fallback: {error}")
        return self._default_result()
