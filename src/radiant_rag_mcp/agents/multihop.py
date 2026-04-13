"""
Multi-hop reasoning agent for RAG pipeline.

Handles complex queries that require multiple retrieval steps and
inference chains to arrive at the final answer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from radiant_rag_mcp.agents.base_agent import (
    AgentCategory,
    AgentMetrics,
    BaseAgent,
)

if TYPE_CHECKING:
    from radiant_rag_mcp.llm.client import LLMClient
    from radiant_rag_mcp.storage.base import BaseVectorStore
    from radiant_rag_mcp.llm.local_models import LocalNLPModels

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """A single step in the multi-hop reasoning chain."""
    
    step_number: int
    sub_question: str
    retrieved_docs: List[Any]
    extracted_answer: str
    confidence: float
    entities_found: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "sub_question": self.sub_question,
            "extracted_answer": self.extracted_answer,
            "confidence": self.confidence,
            "entities_found": self.entities_found,
            "num_docs_retrieved": len(self.retrieved_docs),
            "metadata": self.metadata,
        }


@dataclass
class MultiHopResult:
    """Result of multi-hop reasoning process."""
    
    original_query: str
    requires_multihop: bool
    reasoning_chain: List[ReasoningStep]
    final_context: List[Any]  # Combined context from all hops
    intermediate_answers: List[str]
    total_hops: int
    success: bool
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "requires_multihop": self.requires_multihop,
            "reasoning_chain": [step.to_dict() for step in self.reasoning_chain],
            "intermediate_answers": self.intermediate_answers,
            "total_hops": self.total_hops,
            "success": self.success,
            "error": self.error,
        }


class MultiHopReasoningAgent(BaseAgent):
    """
    Multi-hop reasoning agent for complex queries.
    
    Handles queries that require:
    - Bridge reasoning: "Who is the CEO of the company that made X?"
    - Comparison: "Which is larger, the population of city A or city B?"
    - Temporal: "What happened after event X?"
    - Compositional: Questions requiring multiple facts to be combined
    
    Process:
    1. Analyze query to determine if multi-hop is needed
    2. Decompose into sub-questions
    3. Execute retrieval for each sub-question
    4. Extract intermediate answers
    5. Use intermediate answers to formulate next sub-question
    6. Combine all retrieved context for final synthesis
    """
    
    # Patterns that suggest multi-hop reasoning is needed
    MULTIHOP_INDICATORS = [
        # Bridge questions
        r"who (?:is|was) the .+ of the .+ (?:that|which|who)",
        r"what (?:is|was) the .+ of the .+ (?:that|which|who)",
        r"where (?:is|was) the .+ (?:that|which|who)",
        # Comparison questions
        r"which is (?:more|less|larger|smaller|better|worse)",
        r"compare .+ (?:and|with|to) .+",
        r"difference between .+ and .+",
        # Temporal chains
        r"what happened (?:after|before|when) .+",
        r"who (?:succeeded|preceded|followed) .+",
        # Compositional
        r"how many .+ (?:are|were) .+ and .+",
        r"what .+ and .+ have in common",
    ]
    
    def __init__(
        self,
        llm: "LLMClient",
        store: "BaseVectorStore",
        local_models: "LocalNLPModels",
        max_hops: int = 3,
        docs_per_hop: int = 5,
        min_confidence_to_continue: float = 0.3,
        enable_entity_extraction: bool = True,
        enabled: bool = True,
    ) -> None:
        """
        Initialize the multi-hop reasoning agent.
        
        Args:
            llm: LLM client for reasoning
            store: Vector store for retrieval
            local_models: Local models for embedding
            max_hops: Maximum reasoning hops allowed
            docs_per_hop: Documents to retrieve per hop
            min_confidence_to_continue: Minimum confidence to continue chain
            enable_entity_extraction: Extract entities for follow-up queries
            enabled: Whether the agent is enabled
        """
        super().__init__(
            llm=llm,
            store=store,
            local_models=local_models,
            enabled=enabled,
        )
        self._max_hops = max_hops
        self._docs_per_hop = docs_per_hop
        self._min_confidence = min_confidence_to_continue
        self._enable_entities = enable_entity_extraction

    @property
    def name(self) -> str:
        """Return the agent's unique name."""
        return "MultiHopReasoningAgent"

    @property
    def category(self) -> AgentCategory:
        """Return the agent's category."""
        return AgentCategory.EVALUATION

    @property
    def description(self) -> str:
        """Return a human-readable description."""
        return "Handles complex queries requiring multiple retrieval and reasoning steps"
    
    def requires_multihop(self, query: str) -> Tuple[bool, str]:
        """
        Determine if a query requires multi-hop reasoning.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (requires_multihop, reason)
        """
        query_lower = query.lower()
        
        # Check pattern indicators
        for pattern in self.MULTIHOP_INDICATORS:
            if re.search(pattern, query_lower):
                return True, f"Pattern match: {pattern[:30]}..."
        
        # Use LLM for complex cases
        system = """Analyze if this query requires multi-hop reasoning.

Multi-hop reasoning is needed when:
1. The answer requires combining information from multiple separate facts
2. The query asks about an entity that must first be identified through another fact
3. Comparison between multiple entities requiring separate lookups
4. Temporal reasoning requiring multiple events to be connected

Return ONLY raw JSON object with no markdown formatting. Do not wrap in ```json code blocks.
Format:
{
  "requires_multihop": true/false,
  "reason": "brief explanation",
  "estimated_hops": 1-3
}"""

        user = f"Query: {query}\n\nReturn raw JSON only, no code blocks."
        
        result, response = self._llm.chat_json(
            system=system,
            user=user,
            default={"requires_multihop": False, "reason": "Simple query"},
            expected_type=dict,
        )
        
        if not response.success:
            return False, "Analysis failed, assuming single-hop"
        
        return (
            bool(result.get("requires_multihop", False)),
            str(result.get("reason", "LLM analysis"))
        )

    def _execute(
        self,
        query: str,
        initial_context: Optional[List[Any]] = None,
        force_multihop: bool = False,
        **kwargs: Any,
    ) -> MultiHopResult:
        """
        Execute multi-hop reasoning for a query.
        
        Args:
            query: User query
            initial_context: Optional initial retrieved documents
            force_multihop: Force multi-hop even if not detected as needed
            
        Returns:
            MultiHopResult with reasoning chain and combined context
        """
        # Check if multi-hop is needed
        needs_multihop, reason = self.requires_multihop(query)
        
        if not needs_multihop and not force_multihop:
            self.logger.debug(f"Query does not require multi-hop: {reason}")
            return MultiHopResult(
                original_query=query,
                requires_multihop=False,
                reasoning_chain=[],
                final_context=initial_context or [],
                intermediate_answers=[],
                total_hops=0,
                success=True,
            )
        
        self.logger.info(f"Starting multi-hop reasoning for: {query[:50]}...")
        
        # Decompose into sub-questions
        sub_questions = self._decompose_query(query)
        
        if not sub_questions:
            self.logger.warning("Failed to decompose query, using single-hop")
            return MultiHopResult(
                original_query=query,
                requires_multihop=False,
                reasoning_chain=[],
                final_context=initial_context or [],
                intermediate_answers=[],
                total_hops=0,
                success=True,
            )
        
        # Execute reasoning chain
        reasoning_chain: List[ReasoningStep] = []
        all_context: List[Any] = list(initial_context or [])
        intermediate_answers: List[str] = []
        accumulated_knowledge = ""
        
        for i, sub_question in enumerate(sub_questions):
            self.logger.debug(f"Processing hop {i+1}: {sub_question[:50]}...")
            
            # Retrieve for this hop
            step_docs = self._retrieve_for_hop(sub_question)
            
            # Extract answer from retrieved docs
            answer, confidence, entities = self._extract_hop_answer(
                sub_question,
                step_docs,
                accumulated_knowledge,
            )
            
            # Create reasoning step
            step = ReasoningStep(
                step_number=i + 1,
                sub_question=sub_question,
                retrieved_docs=step_docs,
                extracted_answer=answer,
                confidence=confidence,
                entities_found=entities,
                metadata={"accumulated_knowledge_length": len(accumulated_knowledge)},
            )
            reasoning_chain.append(step)
            
            # Accumulate context and knowledge
            all_context.extend(step_docs)
            intermediate_answers.append(answer)
            
            if answer and answer.lower() not in ("unknown", "not found", ""):
                accumulated_knowledge += f" {answer}"
            
            # Check if we should continue
            if confidence < self._min_confidence:
                self.logger.info(f"Stopping at hop {i+1} due to low confidence: {confidence:.2f}")
                break
            
            # Check if we have enough information
            if self._has_sufficient_info(query, accumulated_knowledge):
                self.logger.info(f"Sufficient information gathered at hop {i+1}")
                break
        
        # Deduplicate context
        final_context = self._deduplicate_context(all_context)
        
        self.logger.info(
            f"Multi-hop complete: {len(reasoning_chain)} hops, "
            f"{len(final_context)} unique docs"
        )
        
        return MultiHopResult(
            original_query=query,
            requires_multihop=True,
            reasoning_chain=reasoning_chain,
            final_context=final_context,
            intermediate_answers=intermediate_answers,
            total_hops=len(reasoning_chain),
            success=True,
        )
    
    def _decompose_query(self, query: str) -> List[str]:
        """Decompose a complex query into sub-questions."""

        system = """Decompose this complex query into sequential sub-questions.

Each sub-question should:
1. Be answerable independently or with knowledge from previous answers
2. Build toward answering the original query
3. Be ordered so earlier answers help answer later questions

Return ONLY raw JSON object with no markdown formatting. Do not wrap in ```json code blocks.
Format:
{
  "sub_questions": ["Question 1", "Question 2", "Question 3"],
  "reasoning_type": "bridge|comparison|temporal|compositional"
}

Maximum 3 sub-questions."""

        user = f"Query: {query}\n\nReturn raw JSON only, no code blocks."
        
        result, response = self._llm.chat_json(
            system=system,
            user=user,
            default={"sub_questions": [query]},
            expected_type=dict,
        )
        
        if not response.success:
            return [query]
        
        sub_questions = result.get("sub_questions", [query])
        
        if not isinstance(sub_questions, list) or not sub_questions:
            return [query]
        
        return [str(q) for q in sub_questions[:self._max_hops]]
    
    def _retrieve_for_hop(self, query: str) -> List[Any]:
        """Retrieve documents for a single reasoning hop."""
        
        try:
            # Embed query
            query_vec = self._local_models.embed_single(query)
            
            # Retrieve - search all document levels for multihop reasoning
            # since we may need both children and parents for complex queries
            results = self._store.retrieve_by_embedding(
                query_vec,
                top_k=self._docs_per_hop,
                doc_level_filter=None,  # Search all levels for multihop
            )
            
            return [doc for doc, score in results]
            
        except Exception as e:
            self.logger.warning(f"Retrieval failed for hop: {e}")
            return []
    
    def _extract_hop_answer(
        self,
        question: str,
        docs: List[Any],
        prior_knowledge: str,
    ) -> Tuple[str, float, List[str]]:
        """
        Extract answer and entities from retrieved documents.
        
        Returns:
            Tuple of (answer, confidence, entities)
        """
        if not docs:
            return "No information found", 0.0, []
        
        # Format context
        context_parts = []
        for i, doc in enumerate(docs[:5], start=1):
            content = getattr(doc, 'content', str(doc))[:1500]
            context_parts.append(f"[{i}] {content}")
        
        context = "\n\n".join(context_parts)
        
        system = """Extract a concise answer from the context for the given question.

Also extract any named entities (people, organizations, places, dates) that might
be useful for follow-up questions.

Return ONLY raw JSON object with no markdown formatting. Do not wrap in ```json code blocks.
Format:
{
  "answer": "concise answer extracted from context",
  "confidence": 0.0-1.0,
  "entities": ["entity1", "entity2"],
  "source_doc_indices": [1, 2]
}

If the answer is not in the context, return:
{
  "answer": "Not found in context",
  "confidence": 0.0,
  "entities": [],
  "source_doc_indices": []
}"""

        prior_info = f"\nPrior knowledge: {prior_knowledge}" if prior_knowledge else ""

        user = f"""Question: {question}
{prior_info}

Context:
{context}

Return raw JSON only, no code blocks."""

        result, response = self._llm.chat_json(
            system=system,
            user=user,
            default={"answer": "Unknown", "confidence": 0.0, "entities": []},
            expected_type=dict,
        )
        
        if not response.success:
            return "Extraction failed", 0.0, []
        
        answer = str(result.get("answer", "Unknown"))
        confidence = float(result.get("confidence", 0.0))
        entities = result.get("entities", [])
        
        if not isinstance(entities, list):
            entities = []
        else:
            entities = [str(e) for e in entities]
        
        return answer, confidence, entities
    
    def _has_sufficient_info(self, original_query: str, accumulated: str) -> bool:
        """Check if we have enough information to answer the original query."""
        
        if not accumulated or len(accumulated) < 20:
            return False
        
        system = """Determine if the accumulated knowledge is sufficient to answer the query.

Return ONLY raw JSON object with no markdown formatting. Do not wrap in ```json code blocks.
Format:
{
  "sufficient": true/false,
  "missing": "what's still needed (if not sufficient)"
}"""

        user = f"""Original query: {original_query}

Accumulated knowledge: {accumulated}

Return raw JSON only, no code blocks."""

        result, response = self._llm.chat_json(
            system=system,
            user=user,
            default={"sufficient": False},
            expected_type=dict,
        )
        
        return bool(result.get("sufficient", False))
    
    def _deduplicate_context(self, docs: List[Any]) -> List[Any]:
        """Remove duplicate documents from context."""
        
        seen_ids = set()
        unique_docs = []
        
        for doc in docs:
            doc_id = getattr(doc, 'doc_id', id(doc))
            
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)
        
        return unique_docs
    
    def get_reasoning_summary(self, result: MultiHopResult) -> str:
        """Generate a human-readable summary of the reasoning chain."""
        
        if not result.requires_multihop or not result.reasoning_chain:
            return "Single-hop reasoning (no chain)"
        
        lines = [f"Multi-hop reasoning ({result.total_hops} hops):"]
        
        for step in result.reasoning_chain:
            lines.append(
                f"  Step {step.step_number}: {step.sub_question}"
                f"\n    → Answer: {step.extracted_answer[:100]}..."
                f"\n    → Confidence: {step.confidence:.2f}"
            )
        
        return "\n".join(lines)

    def _on_error(
        self,
        error: Exception,
        metrics: AgentMetrics,
        **kwargs: Any,
    ) -> Optional[MultiHopResult]:
        """
        Provide fallback result on error.
        """
        query = kwargs.get("query", "")
        initial_context = kwargs.get("initial_context", [])
        
        self.logger.warning(f"Multi-hop reasoning failed: {error}")
        
        return MultiHopResult(
            original_query=query,
            requires_multihop=True,
            reasoning_chain=[],
            final_context=initial_context,
            intermediate_answers=[],
            total_hops=0,
            success=False,
            error=str(error),
        )
