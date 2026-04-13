"""
Context evaluation agent for RAG pipeline.

Evaluates retrieved context quality BEFORE generation to prevent
wasted LLM calls on poor retrieval results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from radiant_rag_mcp.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ContextEvaluation:
    """Result of context quality evaluation."""
    
    # Overall assessment
    sufficient: bool  # Whether context is sufficient to answer query
    confidence: float  # Confidence in the assessment (0-1)
    
    # Detailed scores (0-10)
    relevance_score: float  # How relevant is the context to the query
    coverage_score: float  # How well does context cover the query's scope
    specificity_score: float  # How specific/detailed is the information
    freshness_score: float  # Estimated freshness of information
    
    # Actionable feedback
    missing_aspects: List[str]  # What information is missing
    suggestions: List[str]  # Suggestions for improving retrieval
    
    # Decision
    recommendation: str  # "proceed", "expand_retrieval", "rewrite_query", "abort"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "confidence": self.confidence,
            "relevance_score": self.relevance_score,
            "coverage_score": self.coverage_score,
            "specificity_score": self.specificity_score,
            "freshness_score": self.freshness_score,
            "missing_aspects": self.missing_aspects,
            "suggestions": self.suggestions,
            "recommendation": self.recommendation,
        }


class ContextEvaluationAgent:
    """
    Evaluates retrieved context quality before generation.
    
    This agent serves as a quality gate between retrieval and generation,
    preventing wasted LLM calls when context is insufficient.
    
    Key functions:
    - Assess relevance of retrieved documents to query
    - Identify gaps in coverage
    - Recommend actions (proceed, expand retrieval, rewrite query)
    - Provide confidence scores for downstream decision making
    """
    
    # Thresholds for evaluation
    DEFAULT_SUFFICIENCY_THRESHOLD = 0.5
    DEFAULT_MIN_RELEVANT_DOCS = 1
    
    def __init__(
        self,
        llm: "LLMClient",
        sufficiency_threshold: float = 0.5,
        min_relevant_docs: int = 1,
        max_docs_to_evaluate: int = 8,
        max_doc_chars: int = 1000,
        use_llm_evaluation: bool = True,
    ) -> None:
        """
        Initialize the context evaluation agent.
        
        Args:
            llm: LLM client for evaluation
            sufficiency_threshold: Minimum score to consider context sufficient
            min_relevant_docs: Minimum number of relevant docs required
            max_docs_to_evaluate: Maximum docs to include in evaluation
            max_doc_chars: Maximum characters per doc for evaluation
            use_llm_evaluation: Whether to use LLM for detailed evaluation
        """
        self._llm = llm
        self._sufficiency_threshold = sufficiency_threshold
        self._min_relevant_docs = min_relevant_docs
        self._max_docs = max_docs_to_evaluate
        self._max_chars = max_doc_chars
        self._use_llm = use_llm_evaluation
    
    def evaluate(
        self,
        query: str,
        docs: List[Any],
        scores: Optional[List[float]] = None,
    ) -> ContextEvaluation:
        """
        Evaluate the quality of retrieved context for a query.
        
        Args:
            query: The user query
            docs: Retrieved documents (with .content attribute)
            scores: Optional retrieval scores for the documents
            
        Returns:
            ContextEvaluation with assessment and recommendations
        """
        # Handle empty retrieval
        if not docs:
            return ContextEvaluation(
                sufficient=False,
                confidence=1.0,
                relevance_score=0.0,
                coverage_score=0.0,
                specificity_score=0.0,
                freshness_score=5.0,  # Unknown
                missing_aspects=["No documents retrieved"],
                suggestions=["Try different search terms", "Expand the query"],
                recommendation="rewrite_query",
            )
        
        # Quick heuristic check first
        heuristic_eval = self._heuristic_evaluation(query, docs, scores)
        
        # If heuristics clearly indicate sufficient/insufficient, skip LLM
        if heuristic_eval.confidence >= 0.8:
            return heuristic_eval
        
        # Use LLM for detailed evaluation
        if self._use_llm:
            try:
                return self._llm_evaluation(query, docs, scores)
            except Exception as e:
                logger.warning(f"LLM evaluation failed, using heuristics: {e}")
                return heuristic_eval
        
        return heuristic_eval
    
    def quick_check(
        self,
        query: str,
        docs: List[Any],
        scores: Optional[List[float]] = None,
    ) -> bool:
        """
        Quick check if context is likely sufficient (no LLM call).
        
        Args:
            query: The user query
            docs: Retrieved documents
            scores: Optional retrieval scores
            
        Returns:
            True if context appears sufficient for generation
        """
        if not docs:
            return False
        
        # Check if we have minimum docs
        if len(docs) < self._min_relevant_docs:
            return False
        
        # Check scores if available
        if scores:
            high_score_docs = sum(1 for s in scores if s > 0.5)
            if high_score_docs < self._min_relevant_docs:
                return False
        
        # Check content coverage (simple keyword overlap)
        query_words = set(query.lower().split())
        
        relevant_docs = 0
        for doc in docs[:self._max_docs]:
            content = getattr(doc, 'content', str(doc)).lower()
            content_words = set(content.split())
            
            overlap = len(query_words & content_words)
            if overlap >= len(query_words) * 0.3:  # 30% keyword overlap
                relevant_docs += 1
        
        return relevant_docs >= self._min_relevant_docs
    
    def _heuristic_evaluation(
        self,
        query: str,
        docs: List[Any],
        scores: Optional[List[float]],
    ) -> ContextEvaluation:
        """Perform quick heuristic-based evaluation."""
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Calculate relevance based on keyword overlap
        relevance_scores = []
        for doc in docs[:self._max_docs]:
            content = getattr(doc, 'content', str(doc)).lower()
            content_words = set(content.split())
            
            if query_words:
                overlap = len(query_words & content_words) / len(query_words)
            else:
                overlap = 0.5
            
            relevance_scores.append(overlap)
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        # Use retrieval scores if available
        if scores:
            avg_retrieval_score = sum(scores[:self._max_docs]) / min(len(scores), self._max_docs)
            # Blend heuristic and retrieval scores
            relevance_score = (avg_relevance * 5 + avg_retrieval_score * 5)
        else:
            relevance_score = avg_relevance * 10
        
        # Estimate coverage based on number of docs with relevant content
        relevant_docs = sum(1 for r in relevance_scores if r > 0.2)
        coverage_score = min(10, relevant_docs * 3)
        
        # Estimate specificity based on content length
        total_content = sum(len(getattr(d, 'content', str(d))) for d in docs[:self._max_docs])
        if total_content > 5000:
            specificity_score = 8.0
        elif total_content > 2000:
            specificity_score = 6.0
        elif total_content > 500:
            specificity_score = 4.0
        else:
            specificity_score = 2.0
        
        # Calculate overall sufficiency
        avg_score = (relevance_score + coverage_score + specificity_score) / 3
        sufficient = avg_score >= self._sufficiency_threshold * 10
        
        # Determine recommendation
        if sufficient:
            recommendation = "proceed"
        elif relevance_score < 3:
            recommendation = "rewrite_query"
        elif coverage_score < 3:
            recommendation = "expand_retrieval"
        else:
            recommendation = "proceed"  # Marginal but try anyway
        
        # Confidence based on score variance
        scores_list = [relevance_score, coverage_score, specificity_score]
        variance = sum((s - avg_score) ** 2 for s in scores_list) / len(scores_list)
        confidence = max(0.4, min(0.8, 1.0 - variance / 50))
        
        return ContextEvaluation(
            sufficient=sufficient,
            confidence=confidence,
            relevance_score=relevance_score,
            coverage_score=coverage_score,
            specificity_score=specificity_score,
            freshness_score=5.0,  # Unknown without LLM
            missing_aspects=self._identify_missing_aspects_heuristic(query, docs),
            suggestions=self._generate_suggestions_heuristic(relevance_score, coverage_score),
            recommendation=recommendation,
        )
    
    def _llm_evaluation(
        self,
        query: str,
        docs: List[Any],
        scores: Optional[List[float]],
    ) -> ContextEvaluation:
        """Perform detailed LLM-based evaluation."""
        
        # Format context for evaluation
        context_parts = []
        for i, doc in enumerate(docs[:self._max_docs], start=1):
            content = getattr(doc, 'content', str(doc))
            truncated = content[:self._max_chars] if len(content) > self._max_chars else content
            score_info = f" (score: {scores[i-1]:.2f})" if scores and i <= len(scores) else ""
            context_parts.append(f"[DOC {i}]{score_info}\n{truncated}")
        
        context = "\n\n".join(context_parts)
        
        system = """You are a context quality evaluator for a RAG system.
Assess whether the retrieved documents provide sufficient information to answer the query.

Evaluate on these dimensions (0-10 scale):
1. Relevance: How relevant are the documents to the query?
2. Coverage: Do the documents cover all aspects of the query?
3. Specificity: How specific and detailed is the information?
4. Freshness: Does the information appear current? (estimate)

Return ONLY raw JSON object with no markdown formatting. Do not wrap in ```json code blocks.
Format:
{
  "sufficient": true/false,
  "relevance_score": 0-10,
  "coverage_score": 0-10,
  "specificity_score": 0-10,
  "freshness_score": 0-10,
  "missing_aspects": ["list of missing information"],
  "suggestions": ["list of suggestions to improve retrieval"],
  "recommendation": "proceed" | "expand_retrieval" | "rewrite_query" | "abort"
}

Guidelines:
- "proceed": Context is sufficient for a good answer
- "expand_retrieval": Need more documents on the same topic
- "rewrite_query": Query may need reformulation for better results
- "abort": Context is too poor to generate a meaningful answer"""

        user = f"""QUERY: {query}

RETRIEVED CONTEXT:
{context}

Evaluate the context quality and return raw JSON only, no code blocks."""

        result, response = self._llm.chat_json(
            system=system,
            user=user,
            default=self._default_evaluation(),
            expected_type=dict,
        )
        
        if not response.success:
            logger.warning("LLM evaluation failed, using defaults")
            result = self._default_evaluation()
        
        # Normalize and validate
        result = self._normalize_evaluation(result)
        
        # Calculate overall confidence
        scores_list = [
            result.get("relevance_score", 5),
            result.get("coverage_score", 5),
            result.get("specificity_score", 5),
        ]
        avg_score = sum(scores_list) / len(scores_list)
        
        return ContextEvaluation(
            sufficient=result.get("sufficient", avg_score >= self._sufficiency_threshold * 10),
            confidence=0.8,  # LLM evaluation has higher confidence
            relevance_score=result.get("relevance_score", 5.0),
            coverage_score=result.get("coverage_score", 5.0),
            specificity_score=result.get("specificity_score", 5.0),
            freshness_score=result.get("freshness_score", 5.0),
            missing_aspects=result.get("missing_aspects", []),
            suggestions=result.get("suggestions", []),
            recommendation=result.get("recommendation", "proceed"),
        )
    
    def _default_evaluation(self) -> Dict[str, Any]:
        """Return default evaluation for failed LLM calls."""
        return {
            "sufficient": True,
            "relevance_score": 5.0,
            "coverage_score": 5.0,
            "specificity_score": 5.0,
            "freshness_score": 5.0,
            "missing_aspects": [],
            "suggestions": [],
            "recommendation": "proceed",
        }
    
    def _normalize_evaluation(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and validate evaluation result."""
        
        # Ensure boolean
        result["sufficient"] = bool(result.get("sufficient", True))
        
        # Clamp scores to 0-10
        for key in ["relevance_score", "coverage_score", "specificity_score", "freshness_score"]:
            value = result.get(key, 5.0)
            try:
                result[key] = max(0.0, min(10.0, float(value)))
            except (TypeError, ValueError):
                result[key] = 5.0
        
        # Ensure lists
        for key in ["missing_aspects", "suggestions"]:
            if not isinstance(result.get(key), list):
                result[key] = []
            else:
                result[key] = [str(item) for item in result[key]]
        
        # Validate recommendation
        valid_recommendations = {"proceed", "expand_retrieval", "rewrite_query", "abort"}
        if result.get("recommendation") not in valid_recommendations:
            result["recommendation"] = "proceed"
        
        return result
    
    def _identify_missing_aspects_heuristic(
        self,
        query: str,
        docs: List[Any],
    ) -> List[str]:
        """Identify potentially missing aspects using heuristics."""
        
        missing = []
        query_lower = query.lower()
        
        # Check for question words that might indicate missing info
        if "how" in query_lower:
            # Check if docs contain procedural language
            has_procedural = any(
                "step" in getattr(d, 'content', str(d)).lower() or
                "first" in getattr(d, 'content', str(d)).lower()
                for d in docs
            )
            if not has_procedural:
                missing.append("Step-by-step instructions may be missing")
        
        if "why" in query_lower:
            has_explanation = any(
                "because" in getattr(d, 'content', str(d)).lower() or
                "reason" in getattr(d, 'content', str(d)).lower()
                for d in docs
            )
            if not has_explanation:
                missing.append("Explanatory content may be missing")
        
        if "compare" in query_lower or "difference" in query_lower:
            missing.append("Comparative analysis may need more sources")
        
        return missing[:3]  # Limit to top 3
    
    def _generate_suggestions_heuristic(
        self,
        relevance_score: float,
        coverage_score: float,
    ) -> List[str]:
        """Generate improvement suggestions based on scores."""
        
        suggestions = []
        
        if relevance_score < 4:
            suggestions.append("Try using more specific keywords")
            suggestions.append("Consider rephrasing the query")
        
        if coverage_score < 4:
            suggestions.append("Search for additional related documents")
            suggestions.append("Try breaking the query into sub-questions")
        
        if not suggestions:
            suggestions.append("Context appears adequate")
        
        return suggestions[:3]
    
    def should_proceed(self, evaluation: ContextEvaluation) -> bool:
        """
        Determine if generation should proceed based on evaluation.
        
        Args:
            evaluation: Context evaluation result
            
        Returns:
            True if generation should proceed
        """
        if evaluation.recommendation == "abort":
            return False
        
        if evaluation.sufficient:
            return True
        
        # Even if not sufficient, proceed if recommendation isn't abort
        # (let the critic handle quality issues post-generation)
        return evaluation.recommendation in ("proceed", "expand_retrieval")
