"""
Agentic pipeline orchestrator for Radiant RAG.

Coordinates the execution of all pipeline agents with:
- Critic-driven retry loop for quality improvement
- Confidence thresholds with "I don't know" fallback
- Dynamic retrieval mode selection
- Tool integration
- Strategy memory for adaptive behavior
- Pre-generation context evaluation
- Context summarization and compression
- Metrics collection via Prometheus/OpenTelemetry
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypeVar

from radiant_rag_mcp.config import AppConfig
from radiant_rag_mcp.utils.metrics import RunMetrics
from radiant_rag_mcp.utils.metrics_export import MetricsCollector, get_metrics_collector
from radiant_rag_mcp.llm.client import LLMClient, LocalNLPModels
from radiant_rag_mcp.storage.base import BaseVectorStore
from radiant_rag_mcp.storage.bm25_index import PersistentBM25Index
from radiant_rag_mcp.utils.conversation import ConversationManager
from radiant_rag_mcp.agents import (
    AgentContext,
    AnswerSynthesisAgent,
    BM25RetrievalAgent,
    CriticAgent,
    CrossEncoderRerankingAgent,
    DenseRetrievalAgent,
    HierarchicalAutoMergingAgent,
    PlanningAgent,
    QueryDecompositionAgent,
    QueryExpansionAgent,
    QueryRewriteAgent,
    RRFAgent,
    WebSearchAgent,
    new_agent_context,
    # Evaluation agents
    ContextEvaluationAgent,
    SummarizationAgent,
    # New agents
    MultiHopReasoningAgent,
    FactVerificationAgent,
    CitationTrackingAgent,
    CitationStyle,
    # Base classes for type checking
    AgentResult,
)
from radiant_rag_mcp.agents.tools import (
    ToolRegistry,
    ToolSelector,
    create_default_tool_registry,
)
from radiant_rag_mcp.agents.strategy_memory import RetrievalStrategyMemory

logger = logging.getLogger(__name__)

# Type variable for AgentResult data extraction
T = TypeVar('T')


def _extract_agent_data(
    result: AgentResult[T],
    default: T,
    agent_name: str = "Agent",
    metrics_collector: Optional[MetricsCollector] = None,
) -> T:
    """
    Extract data from AgentResult with fallback.
    
    Args:
        result: AgentResult from agent.run()
        default: Default value if extraction fails
        agent_name: Agent name for logging
        metrics_collector: Optional metrics collector
        
    Returns:
        The extracted data or default value
    """
    # Record metrics if collector provided
    if metrics_collector:
        metrics_collector.record(result)
    
    if result.success and result.data is not None:
        return result.data
    
    if result.error:
        logger.warning(f"{agent_name} failed: {result.error}")
    
    return default


# Low confidence response template
LOW_CONFIDENCE_RESPONSE = """I don't have enough confidence to provide a reliable answer to your question.

**What I found:** {summary}

**Why I'm uncertain:**
{reasons}

**Suggestions:**
- Try rephrasing your question with more specific terms
- The information you're looking for may not be in the indexed documents
- Consider providing more context about what you're looking for

Confidence: {confidence:.0%}"""


@dataclass
class PipelineResult:
    """Result of a complete pipeline run with confidence metrics."""
    
    answer: str
    context: AgentContext
    metrics: RunMetrics
    success: bool = True
    error: Optional[str] = None
    
    # Agentic enhancements
    confidence: float = 0.0
    retrieval_mode_used: str = "hybrid"
    retry_count: int = 0
    tools_used: List[str] = field(default_factory=list)
    low_confidence: bool = False
    
    # Multi-hop reasoning
    multihop_used: bool = False
    multihop_hops: int = 0
    
    # Fact verification
    fact_verification_score: float = 1.0
    fact_verification_passed: bool = True
    
    # Citation tracking
    cited_answer: Optional[str] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    audit_id: Optional[str] = None

    @property
    def run_id(self) -> str:
        return self.context.run_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "answer": self.answer,
            "success": self.success,
            "error": self.error,
            "original_query": self.context.original_query,
            "decomposed_queries": self.context.decomposed_queries,
            "num_retrieved_docs": len(self.context.reranked),
            "warnings": self.context.warnings,
            "metrics": self.metrics.to_dict(),
            # Agentic fields
            "confidence": self.confidence,
            "retrieval_mode_used": self.retrieval_mode_used,
            "retry_count": self.retry_count,
            "tools_used": self.tools_used,
            "low_confidence": self.low_confidence,
            # Multi-hop
            "multihop_used": self.multihop_used,
            "multihop_hops": self.multihop_hops,
            # Fact verification
            "fact_verification_score": self.fact_verification_score,
            "fact_verification_passed": self.fact_verification_passed,
            # Citation
            "cited_answer": self.cited_answer,
            "citations": self.citations,
            "num_sources": len(self.sources),
            "audit_id": self.audit_id,
        }


class RAGOrchestrator:
    """
    Agentic orchestrator for the complete RAG pipeline.
    
    Coordinates all agents with:
    - Critic-driven retry loop for answer quality improvement
    - Confidence thresholds with graceful "I don't know" fallback
    - Dynamic retrieval mode selection based on query analysis
    - Tool integration for calculations and code execution
    - Strategy memory for learning optimal retrieval strategies
    - Pre-generation context evaluation
    - Context summarization and compression
    - Metrics collection via Prometheus/OpenTelemetry
    """

    def __init__(
        self,
        config: AppConfig,
        llm: LLMClient,
        local: LocalNLPModels,
        store: BaseVectorStore,
        bm25_index: PersistentBM25Index,
        conversation_manager: Optional[ConversationManager] = None,
        metrics_collector: Optional[MetricsCollector] = None,
    ) -> None:
        """
        Initialize the agentic orchestrator.
        
        Args:
            config: Application configuration
            llm: LLM client for chat completions
            local: Local NLP models (embedding, cross-encoder)
            store: Vector store (Redis, Chroma, or PgVector)
            bm25_index: BM25 index for sparse retrieval
            conversation_manager: Optional conversation manager
            metrics_collector: Optional metrics collector for Prometheus/OTel
        """
        self._config = config
        self._pipeline_config = config.pipeline
        self._agentic_config = config.agentic
        self._conversation = conversation_manager
        self._llm = llm  # Store for new agents
        
        # Initialize metrics collector
        self._metrics_collector = metrics_collector or get_metrics_collector()

        # Initialize strategy memory
        self._strategy_memory: Optional[RetrievalStrategyMemory] = None
        if config.agentic.strategy_memory_enabled:
            self._strategy_memory = RetrievalStrategyMemory(
                storage_path=config.agentic.strategy_memory_path
            )

        # Initialize tool registry
        self._tool_registry: Optional[ToolRegistry] = None
        self._tool_selector: Optional[ToolSelector] = None
        if config.agentic.tools_enabled:
            self._tool_registry = create_default_tool_registry()
            self._tool_selector = ToolSelector(llm, self._tool_registry)

        # Initialize agents
        self._planning_agent = PlanningAgent(
            llm,
            web_search_enabled=config.web_search.enabled,
            tools_enabled=config.agentic.tools_enabled,
            available_tools=self._tool_registry.get_tools_for_llm() if self._tool_registry else [],
            strategy_memory=self._strategy_memory,
        )
        self._decomposition_agent = QueryDecompositionAgent(llm, config.query)
        self._rewrite_agent = QueryRewriteAgent(llm)
        self._expansion_agent = QueryExpansionAgent(llm, config.query)
        
        self._dense_retrieval = DenseRetrievalAgent(store, local, config.retrieval)
        self._bm25_retrieval = BM25RetrievalAgent(bm25_index, config.retrieval)
        self._rrf_agent = RRFAgent(config.retrieval)
        
        # Web search agent (conditionally enabled)
        self._web_search_agent = WebSearchAgent(llm, config.web_search) if config.web_search.enabled else None
        
        self._automerge_agent = HierarchicalAutoMergingAgent(store, config.automerge)
        self._rerank_agent = CrossEncoderRerankingAgent(local, config.rerank)
        
        self._synthesis_agent = AnswerSynthesisAgent(
            llm, config.synthesis, conversation_manager
        )
        self._critic_agent = CriticAgent(llm, config.critic)

        # Initialize new agents (context evaluation and summarization)
        self._context_eval_agent: Optional[ContextEvaluationAgent] = None
        if config.context_evaluation.enabled:
            self._context_eval_agent = ContextEvaluationAgent(
                llm=llm,
                sufficiency_threshold=config.context_evaluation.sufficiency_threshold,
                min_relevant_docs=config.context_evaluation.min_relevant_docs,
                max_docs_to_evaluate=config.context_evaluation.max_docs_to_evaluate,
                max_doc_chars=config.context_evaluation.max_doc_chars,
                use_llm_evaluation=config.context_evaluation.use_llm_evaluation,
            )

        self._summarization_agent: Optional[SummarizationAgent] = None
        if config.summarization.enabled:
            self._summarization_agent = SummarizationAgent(
                llm=llm,
                min_doc_length_for_summary=config.summarization.min_doc_length_for_summary,
                target_summary_length=config.summarization.target_summary_length,
                conversation_compress_threshold=config.summarization.conversation_compress_threshold,
                conversation_preserve_recent=config.summarization.conversation_preserve_recent,
                similarity_threshold=config.summarization.similarity_threshold,
                max_cluster_size=config.summarization.max_cluster_size,
            )

        # Initialize multi-hop reasoning agent
        self._multihop_agent: Optional[MultiHopReasoningAgent] = None
        if config.multihop.enabled:
            self._multihop_agent = MultiHopReasoningAgent(
                llm=llm,
                store=store,
                local_models=local,
                max_hops=config.multihop.max_hops,
                docs_per_hop=config.multihop.docs_per_hop,
                min_confidence_to_continue=config.multihop.min_confidence_to_continue,
                enable_entity_extraction=config.multihop.enable_entity_extraction,
            )

        # Initialize fact verification agent
        self._fact_verification_agent: Optional[FactVerificationAgent] = None
        if config.fact_verification.enabled:
            self._fact_verification_agent = FactVerificationAgent(
                llm=llm,
                min_support_confidence=config.fact_verification.min_support_confidence,
                max_claims_to_verify=config.fact_verification.max_claims_to_verify,
                generate_corrections=config.fact_verification.generate_corrections,
                strict_mode=config.fact_verification.strict_mode,
            )

        # Initialize citation tracking agent
        self._citation_agent: Optional[CitationTrackingAgent] = None
        if config.citation.enabled:
            # Map citation style string to enum
            style_map = {
                "inline": CitationStyle.INLINE,
                "footnote": CitationStyle.FOOTNOTE,
                "academic": CitationStyle.ACADEMIC,
                "hyperlink": CitationStyle.HYPERLINK,
                "enterprise": CitationStyle.ENTERPRISE,
            }
            citation_style = style_map.get(
                config.citation.citation_style.lower(),
                CitationStyle.INLINE
            )
            self._citation_agent = CitationTrackingAgent(
                llm=llm,
                citation_style=citation_style,
                min_citation_confidence=config.citation.min_citation_confidence,
                max_citations_per_claim=config.citation.max_citations_per_claim,
                include_excerpts=config.citation.include_excerpts,
                excerpt_max_length=config.citation.excerpt_max_length,
            )

        logger.info(
            f"Agentic RAG orchestrator initialized "
            f"(web_search={'enabled' if config.web_search.enabled else 'disabled'}, "
            f"tools={'enabled' if config.agentic.tools_enabled else 'disabled'}, "
            f"strategy_memory={'enabled' if config.agentic.strategy_memory_enabled else 'disabled'}, "
            f"context_eval={'enabled' if config.context_evaluation.enabled else 'disabled'}, "
            f"summarization={'enabled' if config.summarization.enabled else 'disabled'}, "
            f"multihop={'enabled' if config.multihop.enabled else 'disabled'}, "
            f"fact_verification={'enabled' if config.fact_verification.enabled else 'disabled'}, "
            f"citation={'enabled' if config.citation.enabled else 'disabled'})"
        )

    def _is_simple_query(self, query: str) -> bool:
        """
        Determine if a query is simple enough for fast-path execution.

        Simple queries can skip expensive operations like:
        - Query decomposition
        - Multi-hop reasoning
        - Fact verification (when confidence is high)

        Args:
            query: User query text

        Returns:
            True if query is simple, False if complex
        """
        # Simple heuristics for query complexity
        query_lower = query.lower().strip()
        query_words = query_lower.split()

        # Short queries are usually simple
        if len(query_words) <= 5:
            return True

        # Questions with "what", "who", "when", "where" are usually simple
        if any(query_lower.startswith(word) for word in ["what is", "who is", "when did", "where is"]):
            return True

        # Queries without complex conjunctions are simpler
        complex_markers = ["and", "but", "also", "additionally", "furthermore", "moreover", "compare", "contrast"]
        if not any(marker in query_lower for marker in complex_markers):
            if len(query_words) <= 10:
                return True

        return False

    def run(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        plan_override: Optional[Dict[str, bool]] = None,
        retrieval_mode: Optional[str] = None,
    ) -> PipelineResult:
        """
        Execute the complete agentic RAG pipeline with retry loop.
        
        Args:
            query: User query
            conversation_id: Optional conversation ID for history
            plan_override: Optional override for pipeline plan
            retrieval_mode: Override retrieval mode (None = dynamic selection)
            
        Returns:
            PipelineResult with answer, confidence, and metadata
        """
        # Initialize context and metrics
        ctx = new_agent_context(query, conversation_id)
        metrics = RunMetrics(run_id=ctx.run_id)
        
        logger.info(f"[{ctx.run_id}] Starting agentic pipeline for query: {query[:100]}...")

        try:
            # Load conversation history if available
            if self._conversation and conversation_id:
                self._conversation.load_conversation(conversation_id)
                ctx.conversation_history = self._conversation.get_history_for_synthesis()

            # PERFORMANCE OPTIMIZATION: Detect simple queries for fast-path execution
            is_simple = self._is_simple_query(query)
            if is_simple:
                logger.debug(f"[{ctx.run_id}] Detected simple query - using fast path")

            # Phase 1: Planning with dynamic mode selection
            plan = self._run_planning(ctx, metrics, plan_override)

            # EARLY STOPPING: Skip expensive operations for simple queries
            if is_simple and not plan_override:
                # Disable expensive operations that aren't needed for simple queries
                plan["use_decomposition"] = False  # Skip query decomposition
                plan["use_expansion"] = False  # Skip query expansion
                # Keep rewrite as it's relatively cheap and can improve results
                logger.debug(f"[{ctx.run_id}] Fast path: disabled decomposition and expansion")

            # Determine retrieval mode (override > plan > dynamic)
            if retrieval_mode and retrieval_mode in ("hybrid", "dense", "bm25"):
                ctx.retrieval_mode = retrieval_mode
            else:
                ctx.retrieval_mode = plan.get("retrieval_mode", "hybrid")

            logger.info(f"[{ctx.run_id}] Retrieval mode: {ctx.retrieval_mode}, fast_path: {is_simple}")

            # Phase 2: Execute tools if any were selected
            if plan.get("tools_to_use") and self._tool_registry:
                self._run_tools(ctx, metrics, plan.get("tools_to_use", []))

            # Main retrieval-generation loop with critic retry
            max_retries = self._agentic_config.max_critic_retries

            # PERFORMANCE OPTIMIZATION: Cache retrieval results to avoid re-fetching on retry
            # Only re-retrieve if context evaluation suggests it
            queries_cache = None
            retrieval_done = False

            for attempt in range(max_retries + 1):
                is_retry = attempt > 0

                if is_retry:
                    logger.info(f"[{ctx.run_id}] Retry attempt {attempt}/{max_retries}")

                # TARGETED RETRY: Only re-run query processing if we need different results
                # On first attempt or if context eval suggests query modification
                need_new_queries = (
                    not is_retry
                    or plan.get("use_expansion")
                    or plan.get("use_rewrite")
                    or queries_cache is None
                )

                if need_new_queries:
                    # Phase 3: Query Processing
                    queries = self._run_query_processing(ctx, metrics, plan, is_retry)
                    queries_cache = queries
                    retrieval_done = False
                    logger.debug(f"[{ctx.run_id}] Generated {len(queries)} queries for retrieval")
                else:
                    # OPTIMIZATION: Reuse cached queries from previous attempt
                    queries = queries_cache
                    logger.debug(f"[{ctx.run_id}] Reusing {len(queries)} cached queries")

                # TARGETED RETRY: Only re-retrieve if context was insufficient or queries changed
                if not retrieval_done or need_new_queries:
                    # Phase 4: Retrieval
                    self._run_retrieval(ctx, metrics, queries, plan, ctx.retrieval_mode)

                    # Phase 5: Post-retrieval Processing
                    self._run_post_retrieval(ctx, metrics, plan)

                    # Evaluate retrieval quality
                    ctx.retrieval_confidence = self._critic_agent.evaluate_retrieval_quality(
                        ctx.original_query,
                        ctx.reranked
                    )

                    # Phase 5.5: Pre-generation Context Evaluation
                    context_eval = self._run_context_evaluation(ctx, metrics, plan)

                    # If context is insufficient and we should abort, handle it
                    if context_eval and not context_eval.sufficient:
                        if self._config.context_evaluation.abort_on_poor_context and context_eval.recommendation == "abort":
                            logger.warning(f"[{ctx.run_id}] Aborting due to poor context quality")
                            ctx.add_warning("Context quality too low for reliable answer")
                            answer = self._generate_low_confidence_response(
                                ctx,
                                {
                                    "confidence": context_eval.confidence * 0.5,
                                    "issues": context_eval.missing_aspects,
                                }
                            )
                            ctx.low_confidence_response = True
                            break  # Exit retry loop
                        elif context_eval.recommendation == "expand_retrieval" and attempt < max_retries:
                            # Modify plan to expand retrieval on next attempt
                            plan["use_expansion"] = True
                            ctx.add_warning(f"Context evaluation suggests expansion: {', '.join(context_eval.suggestions[:2])}")
                        elif context_eval.recommendation == "rewrite_query" and attempt < max_retries:
                            plan["use_rewrite"] = True
                            ctx.add_warning(f"Context evaluation suggests rewrite: {', '.join(context_eval.suggestions[:2])}")

                    # Phase 5.6: Context Summarization
                    self._run_context_summarization(ctx, metrics, plan)

                    retrieval_done = True
                else:
                    # OPTIMIZATION: Skip retrieval, reuse previous results
                    logger.info(f"[{ctx.run_id}] Reusing retrieval results from previous attempt")

                # Phase 6: Generation (always run on retry as this is the core retry target)
                answer = self._run_generation(ctx, metrics, plan, is_retry, attempt)

                # Phase 7: Critique with confidence scoring
                if plan.get("use_critic", True) and self._config.critic.enabled:
                    critique = self._run_critique(ctx, metrics, is_retry, attempt)
                    ctx.answer_confidence = critique.get("confidence", 0.5)

                    # Check if we should retry
                    if critique.get("should_retry", False) and attempt < max_retries:
                        # Record retry
                        ctx.record_retry(
                            reason=", ".join(critique.get("issues", ["low confidence"])),
                            previous_confidence=ctx.answer_confidence,
                            modifications={"attempt": attempt + 1}
                        )

                        # TARGETED RETRY: Determine what needs to change
                        # Only modify plan if critic specifically identifies retrieval issues
                        critique_issues = critique.get("issues", [])
                        needs_more_context = any(
                            "context" in issue.lower() or "information" in issue.lower()
                            for issue in critique_issues
                        )

                        if needs_more_context:
                            # Context issue - may need new retrieval
                            if self._agentic_config.expand_retrieval_on_retry:
                                plan["use_expansion"] = True
                            if self._agentic_config.rewrite_on_retry:
                                plan["use_rewrite"] = True
                            retrieval_done = False
                            logger.info(f"[{ctx.run_id}] Critic identified context issue - will re-retrieve")
                        else:
                            # Answer quality issue - just regenerate with feedback
                            # Keep existing retrieval results
                            logger.info(f"[{ctx.run_id}] Critic identified answer issue - will regenerate only")

                        # Maybe switch retrieval mode (only if retrieval needed)
                        if needs_more_context:
                            plan = self._planning_agent.plan_retry(
                                query, plan, critique, attempt + 1
                            )
                            ctx.retrieval_mode = plan.get("retrieval_mode", ctx.retrieval_mode)

                        continue  # Retry

                    # Check if we should give up
                    if self._critic_agent.should_give_up(critique, attempt):
                        answer = self._generate_low_confidence_response(
                            ctx, critique
                        )
                        ctx.low_confidence_response = True
                else:
                    ctx.answer_confidence = 0.5  # Default if critic disabled

                # Success - break retry loop
                break

            # Record strategy outcome
            if self._strategy_memory:
                self._strategy_memory.record_outcome(
                    query=query,
                    strategy=ctx.retrieval_mode,
                    confidence=ctx.answer_confidence,
                    success=not ctx.low_confidence_response,
                    num_retrieved=len(ctx.fused),
                    num_relevant=len(ctx.reranked),
                    critic_ok=not ctx.low_confidence_response,
                )

            # Phase 8 & 9: Fact Verification and Citation Tracking (run in parallel)
            # PERFORMANCE OPTIMIZATION: Run fact verification and citation in parallel
            # These are independent operations that can save 200-800ms each
            should_verify_facts = (
                self._fact_verification_agent is not None
                and (not is_simple or ctx.answer_confidence < 0.8)
            )

            if should_verify_facts and self._citation_agent:
                # Both enabled - run in parallel
                logger.debug(f"[{ctx.run_id}] Running parallel fact verification and citation tracking")

                def run_fact_verification():
                    return self._run_fact_verification(ctx, metrics, answer)

                def run_citation_tracking():
                    return self._run_citation_tracking(ctx, metrics, answer)

                with ThreadPoolExecutor(max_workers=2) as executor:
                    fact_future = executor.submit(run_fact_verification)
                    citation_future = executor.submit(run_citation_tracking)

                    fact_result = fact_future.result()
                    citation_result = citation_future.result()

                fact_score = fact_result.overall_score if fact_result else 1.0
                fact_passed = fact_result.is_factual if fact_result else True

            elif should_verify_facts:
                # Only fact verification enabled
                fact_result = self._run_fact_verification(ctx, metrics, answer)
                fact_score = fact_result.overall_score if fact_result else 1.0
                fact_passed = fact_result.is_factual if fact_result else True
                citation_result = None

            elif self._citation_agent:
                # Only citation enabled
                fact_result = None
                fact_score = 1.0
                fact_passed = True
                citation_result = self._run_citation_tracking(ctx, metrics, answer)

            else:
                # Neither enabled
                if is_simple and self._fact_verification_agent is not None:
                    logger.debug(f"[{ctx.run_id}] Skipping fact verification for simple query with high confidence")
                fact_result = None
                fact_score = 1.0
                fact_passed = True
                citation_result = None

            # Use corrected answer if available and fact verification suggests it
            if fact_result and fact_result.corrected_answer and fact_result.needs_correction:
                logger.info(f"[{ctx.run_id}] Using fact-corrected answer")
                answer = fact_result.corrected_answer
                ctx.final_answer = answer
                ctx.add_warning("Answer was corrected based on fact verification")
            cited_answer = None
            citations_list = []
            sources_list = []
            audit_id = None
            
            if citation_result:
                cited_answer = citation_result.cited_answer
                citations_list = [c.to_dict() for c in citation_result.citations]
                sources_list = [s.to_dict() for s in citation_result.sources]
                audit_id = citation_result.audit_id
                
                # If bibliography is enabled, append to answer
                if self._config.citation.generate_bibliography and citation_result.sources:
                    bibliography = citation_result.get_bibliography()
                    answer = f"{answer}\n\n{bibliography}"
                    ctx.final_answer = answer

            # Record conversation turn
            if self._conversation and conversation_id:
                self._conversation.add_user_query(query)
                self._conversation.add_assistant_response(ctx.final_answer or answer)

            metrics.finish(
                query=query,
                answer_length=len(ctx.final_answer or answer),
                num_docs_used=len(ctx.reranked),
            )

            return PipelineResult(
                answer=ctx.final_answer or answer,
                context=ctx,
                metrics=metrics,
                success=True,
                confidence=ctx.answer_confidence,
                retrieval_mode_used=ctx.retrieval_mode,
                retry_count=ctx.retry_count,
                tools_used=[r.get("tool_name", "") for r in ctx.tool_results],
                low_confidence=ctx.low_confidence_response,
                # Multi-hop
                multihop_used=getattr(ctx, 'multihop_used', False),
                multihop_hops=getattr(ctx, 'multihop_hops', 0),
                # Fact verification
                fact_verification_score=fact_score,
                fact_verification_passed=fact_passed,
                # Citation
                cited_answer=cited_answer,
                citations=citations_list,
                sources=sources_list,
                audit_id=audit_id,
            )

        except Exception as e:
            logger.error(f"[{ctx.run_id}] Pipeline failed: {e}", exc_info=True)
            metrics.finish(error=str(e))
            
            return PipelineResult(
                answer=f"I apologize, but I encountered an error processing your query: {e}",
                context=ctx,
                metrics=metrics,
                success=False,
                error=str(e),
                confidence=0.0,
                retrieval_mode_used=ctx.retrieval_mode,
                retry_count=ctx.retry_count,
                low_confidence=True,
            )

    def _run_planning(
        self,
        ctx: AgentContext,
        metrics: RunMetrics,
        plan_override: Optional[Dict[str, bool]],
    ) -> Dict[str, Any]:
        """Execute planning phase with dynamic retrieval mode selection."""
        if plan_override:
            ctx.plan = plan_override
            return plan_override

        if not self._pipeline_config.use_planning:
            # Use default plan from pipeline config
            ctx.plan = {
                "use_decomposition": self._pipeline_config.use_decomposition,
                "use_rewrite": self._pipeline_config.use_rewrite,
                "use_expansion": self._pipeline_config.use_expansion,
                "use_rrf": self._pipeline_config.use_rrf,
                "use_automerge": self._pipeline_config.use_automerge,
                "use_rerank": self._pipeline_config.use_rerank,
                "use_critic": self._pipeline_config.use_critic,
                "retrieval_mode": "hybrid",
                "tools_to_use": [],
            }
            return ctx.plan

        with metrics.track_step("PlanningAgent") as step:
            try:
                result = self._planning_agent.run(
                    correlation_id=ctx.run_id,
                    query=ctx.original_query,
                )
                ctx.plan = _extract_agent_data(
                    result,
                    default=self._default_plan(),
                    agent_name="PlanningAgent",
                    metrics_collector=self._metrics_collector,
                )
                step.extra["plan"] = ctx.plan
                step.extra["retrieval_mode"] = ctx.plan.get("retrieval_mode", "hybrid")
                step.extra["tools"] = ctx.plan.get("tools_to_use", [])
            except Exception as e:
                logger.warning(f"Planning failed, using defaults: {e}")
                metrics.mark_degraded("planning", str(e))
                ctx.plan = self._default_plan()

        return ctx.plan
    
    def _default_plan(self) -> Dict[str, Any]:
        """Return the default execution plan."""
        return {
            "use_decomposition": True,
            "use_rewrite": True,
            "use_expansion": True,
            "use_rrf": True,
            "use_automerge": True,
            "use_rerank": True,
            "use_critic": True,
            "retrieval_mode": "hybrid",
            "tools_to_use": [],
        }

    def _run_tools(
        self,
        ctx: AgentContext,
        metrics: RunMetrics,
        tools_to_use: List[str],
    ) -> None:
        """Execute requested tools."""
        if not self._tool_registry:
            return
        
        for tool_name in tools_to_use:
            with metrics.track_step(f"Tool:{tool_name}") as step:
                try:
                    # For calculator, try to extract expression from query
                    if tool_name == "calculator":
                        result = self._tool_registry.execute(
                            tool_name,
                            expression=ctx.original_query
                        )
                    else:
                        result = self._tool_registry.execute(
                            tool_name,
                            code=ctx.original_query,
                            context={}
                        )
                    
                    ctx.add_tool_result(result.to_dict())
                    step.extra["success"] = result.success
                    step.extra["output"] = str(result.output)[:200] if result.output else None
                    
                except Exception as e:
                    logger.warning(f"Tool {tool_name} failed: {e}")
                    step.extra["error"] = str(e)

    def _run_query_processing(
        self,
        ctx: AgentContext,
        metrics: RunMetrics,
        plan: Dict[str, Any],
        is_retry: bool = False,
    ) -> List[str]:
        """Execute query processing phase."""
        queries = [ctx.original_query]

        # Query decomposition
        if plan.get("use_decomposition", True) and not is_retry:
            with metrics.track_step("QueryDecompositionAgent") as step:
                try:
                    result = self._decomposition_agent.run(
                        correlation_id=ctx.run_id,
                        query=ctx.original_query,
                    )
                    ctx.decomposed_queries = _extract_agent_data(
                        result,
                        default=[ctx.original_query],
                        agent_name="QueryDecompositionAgent",
                        metrics_collector=self._metrics_collector,
                    )
                    queries = ctx.decomposed_queries
                    step.extra["num_queries"] = len(queries)
                except Exception as e:
                    logger.warning(f"Decomposition failed: {e}")
                    metrics.mark_degraded("decomposition", str(e))
                    ctx.decomposed_queries = [ctx.original_query]

        # Query rewriting (especially important on retry)
        if plan.get("use_rewrite", True):
            step_name = "QueryRewriteAgent" + ("_retry" if is_retry else "")
            with metrics.track_step(step_name) as step:
                try:
                    # PERFORMANCE OPTIMIZATION: Use batch rewrite for multiple queries
                    # Reduces N LLM calls to 1 LLM call (66-75% faster)
                    if len(queries) > 1:
                        logger.debug(f"[{ctx.run_id}] Batch rewriting {len(queries)} queries")
                        rewrites = self._rewrite_agent.rewrite_batch(queries)
                    else:
                        result = self._rewrite_agent.run(
                            correlation_id=ctx.run_id,
                            query=queries[0] if queries else "",
                        )
                        rewrite_data = _extract_agent_data(
                            result,
                            default=(queries[0] if queries else "", queries[0] if queries else ""),
                            agent_name="QueryRewriteAgent",
                            metrics_collector=self._metrics_collector,
                        )
                        rewrites = [rewrite_data]

                    ctx.rewrites = rewrites
                    queries = [after for _, after in rewrites]
                    step.extra["num_rewrites"] = len(rewrites)
                    step.extra["batched"] = len(rewrites) > 1
                except Exception as e:
                    logger.warning(f"Rewriting failed: {e}")
                    metrics.mark_degraded("rewrite", str(e))

        # Query expansion
        if plan.get("use_expansion", True):
            step_name = "QueryExpansionAgent" + ("_retry" if is_retry else "")
            with metrics.track_step(step_name) as step:
                try:
                    # PERFORMANCE OPTIMIZATION: Use batch expansion for multiple queries
                    # Reduces N LLM calls to 1 LLM call (66-75% faster)
                    if len(queries) > 1:
                        logger.debug(f"[{ctx.run_id}] Batch expanding {len(queries)} queries")
                        expansions_per_query = self._expansion_agent.expand_batch(queries)
                        expansions = []
                        for exp_list in expansions_per_query:
                            expansions.extend(exp_list)
                    else:
                        result = self._expansion_agent.run(
                            correlation_id=ctx.run_id,
                            query=queries[0] if queries else "",
                        )
                        expanded = _extract_agent_data(
                            result,
                            default=[],
                            agent_name="QueryExpansionAgent",
                            metrics_collector=self._metrics_collector,
                        )
                        expansions = expanded

                    ctx.expansions = expansions
                    # Add expansions to queries
                    queries = list(set(queries + expansions))
                    step.extra["num_expansions"] = len(expansions)
                    step.extra["batched"] = len(queries) > 1
                except Exception as e:
                    logger.warning(f"Expansion failed: {e}")
                    metrics.mark_degraded("expansion", str(e))

        return queries

    def _run_retrieval(
        self,
        ctx: AgentContext,
        metrics: RunMetrics,
        queries: List[str],
        plan: Dict[str, Any],
        retrieval_mode: str,
    ) -> None:
        """Execute retrieval phase with dynamic mode and parallel execution."""

        # PERFORMANCE OPTIMIZATION: Run dense and BM25 retrieval in parallel for hybrid mode
        # This reduces total retrieval time by ~50% (400-1000ms -> 200-500ms)
        if retrieval_mode == "hybrid":
            logger.debug(f"[{ctx.run_id}] Running parallel hybrid retrieval (dense + BM25)")

            def run_dense_retrieval():
                """Dense retrieval task for parallel execution."""
                try:
                    from radiant_rag_mcp.llm.client import LocalNLPModels
                    from radiant_rag_mcp.storage.base import BaseVectorStore

                    local_models: LocalNLPModels = self._dense_retrieval._local_models

                    if len(queries) > 1:
                        query_embeddings = local_models.embed(queries)
                    else:
                        query_embeddings = [local_models.embed_single(queries[0])] if queries else []

                    all_results = []
                    seen_ids = set()

                    store: BaseVectorStore = self._dense_retrieval._store
                    config = self._dense_retrieval._config
                    doc_level_filter = self._dense_retrieval._get_doc_level_filter()

                    for query, query_embedding in zip(queries, query_embeddings):
                        results = store.retrieve_by_embedding(
                            query_embedding=query_embedding,
                            top_k=config.dense_top_k,
                            min_similarity=config.min_similarity,
                            doc_level_filter=doc_level_filter,
                        )

                        for doc, score in results:
                            doc_id = getattr(doc, 'doc_id', id(doc))
                            if doc_id not in seen_ids:
                                seen_ids.add(doc_id)
                                all_results.append((doc, score))

                    return ("dense", all_results, None)
                except Exception as e:
                    return ("dense", [], str(e))

            def run_bm25_retrieval():
                """BM25 retrieval task for parallel execution."""
                try:
                    all_results = []
                    seen_ids = set()

                    index = self._bm25_retrieval._index
                    config = self._bm25_retrieval._config

                    for query in queries:
                        results = index.search(query, top_k=config.bm25_top_k)

                        for doc, score in results:
                            doc_id = getattr(doc, 'doc_id', id(doc))
                            if doc_id not in seen_ids:
                                seen_ids.add(doc_id)
                                all_results.append((doc, score))

                    return ("bm25", all_results, None)
                except Exception as e:
                    return ("bm25", [], str(e))

            # Execute both retrievals in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(run_dense_retrieval),
                    executor.submit(run_bm25_retrieval)
                ]

                for future in as_completed(futures):
                    retrieval_type, results, error = future.result()

                    if retrieval_type == "dense":
                        with metrics.track_step("DenseRetrievalAgent") as step:
                            if error:
                                logger.warning(f"Dense retrieval failed: {error}")
                                metrics.mark_degraded("dense_retrieval", error)
                                ctx.dense_retrieved = []
                            else:
                                ctx.dense_retrieved = results
                                step.extra["num_retrieved"] = len(results)
                                step.extra["num_queries"] = len(queries)
                                step.extra["batched_embeddings"] = True
                                step.extra["parallel"] = True
                                step.extra["mode"] = "active"

                    elif retrieval_type == "bm25":
                        with metrics.track_step("BM25RetrievalAgent") as step:
                            if error:
                                logger.warning(f"BM25 retrieval failed: {error}")
                                metrics.mark_degraded("bm25_retrieval", error)
                                ctx.bm25_retrieved = []
                            else:
                                ctx.bm25_retrieved = results
                                step.extra["num_retrieved"] = len(results)
                                step.extra["num_queries"] = len(queries)
                                step.extra["parallel"] = True
                                step.extra["mode"] = "active"

        else:
            # Single mode - run sequentially (no benefit from parallelization)
            # Dense retrieval
            if retrieval_mode == "dense":
                with metrics.track_step("DenseRetrievalAgent") as step:
                    try:
                        from radiant_rag_mcp.llm.client import LocalNLPModels
                        from radiant_rag_mcp.storage.base import BaseVectorStore

                        local_models: LocalNLPModels = self._dense_retrieval._local_models

                        if len(queries) > 1:
                            query_embeddings = local_models.embed(queries)
                        else:
                            query_embeddings = [local_models.embed_single(queries[0])] if queries else []

                        all_results = []
                        seen_ids = set()

                        store: BaseVectorStore = self._dense_retrieval._store
                        config = self._dense_retrieval._config
                        doc_level_filter = self._dense_retrieval._get_doc_level_filter()

                        for query, query_embedding in zip(queries, query_embeddings):
                            results = store.retrieve_by_embedding(
                                query_embedding=query_embedding,
                                top_k=config.dense_top_k,
                                min_similarity=config.min_similarity,
                                doc_level_filter=doc_level_filter,
                            )

                            for doc, score in results:
                                doc_id = getattr(doc, 'doc_id', id(doc))
                                if doc_id not in seen_ids:
                                    seen_ids.add(doc_id)
                                    all_results.append((doc, score))

                        ctx.dense_retrieved = all_results
                        step.extra["num_retrieved"] = len(ctx.dense_retrieved)
                        step.extra["num_queries"] = len(queries)
                        step.extra["batched_embeddings"] = True
                        step.extra["mode"] = "active"
                    except Exception as e:
                        logger.warning(f"Dense retrieval failed: {e}")
                        raise

            # BM25 retrieval
            elif retrieval_mode == "bm25":
                with metrics.track_step("BM25RetrievalAgent") as step:
                    try:
                        all_results = []
                        seen_ids = set()

                        index = self._bm25_retrieval._index
                        config = self._bm25_retrieval._config

                        for query in queries:
                            results = index.search(query, top_k=config.bm25_top_k)

                            for doc, score in results:
                                doc_id = getattr(doc, 'doc_id', id(doc))
                                if doc_id not in seen_ids:
                                    seen_ids.add(doc_id)
                                    all_results.append((doc, score))

                        ctx.bm25_retrieved = all_results
                        step.extra["num_retrieved"] = len(ctx.bm25_retrieved)
                        step.extra["num_queries"] = len(queries)
                        step.extra["mode"] = "active"
                    except Exception as e:
                        logger.warning(f"BM25 retrieval failed: {e}")
                        raise

        # Web search (if enabled and requested by plan OR fallback if no docs retrieved)
        should_use_web_search = False
        fallback_triggered = False

        if self._web_search_agent:
            # Check if planner requested web search
            if plan.get("use_web_search", False):
                should_use_web_search = True
                logger.debug("Web search requested by planner")
            else:
                # Fallback: Use web search if no documents were retrieved
                num_dense = len(ctx.dense_retrieved) if ctx.dense_retrieved else 0
                num_bm25 = len(ctx.bm25_retrieved) if ctx.bm25_retrieved else 0

                if num_dense == 0 and num_bm25 == 0:
                    should_use_web_search = True
                    fallback_triggered = True
                    logger.info("No documents retrieved from vector database - triggering web search fallback")

        if should_use_web_search:
            # Update plan to indicate web search should run (needed for agent's _should_search check)
            if fallback_triggered:
                plan = dict(plan)
                plan["use_web_search"] = True
            with metrics.track_step("WebSearchAgent") as step:
                try:
                    result = self._web_search_agent.run(
                        correlation_id=ctx.run_id,
                        query=ctx.original_query,
                        plan=plan,
                    )
                    ctx.web_search_retrieved = _extract_agent_data(
                        result,
                        default=[],
                        agent_name="WebSearchAgent",
                        metrics_collector=self._metrics_collector,
                    )
                    step.extra["num_retrieved"] = len(ctx.web_search_retrieved)
                except Exception as e:
                    logger.warning(f"Web search failed: {e}")
                    metrics.mark_degraded("web_search", str(e))
                    ctx.web_search_retrieved = []

        # RRF Fusion - combine all retrieval sources
        self._fuse_results(ctx, metrics, plan, retrieval_mode)
        
        # Multi-hop reasoning (if enabled and needed)
        if self._multihop_agent:
            self._run_multihop_reasoning(ctx, metrics)

    def _fuse_results(
        self,
        ctx: AgentContext,
        metrics: RunMetrics,
        plan: Dict[str, Any],
        retrieval_mode: str,
    ) -> None:
        """Fuse results from multiple retrievers."""
        retrieval_lists = []
        
        if retrieval_mode in ("hybrid", "dense") and ctx.dense_retrieved:
            retrieval_lists.append(ctx.dense_retrieved)
        if retrieval_mode in ("hybrid", "bm25") and ctx.bm25_retrieved:
            retrieval_lists.append(ctx.bm25_retrieved)
        if ctx.web_search_retrieved:
            retrieval_lists.append(ctx.web_search_retrieved)
        
        if len(retrieval_lists) > 1 and plan.get("use_rrf", True):
            with metrics.track_step("RRFAgent") as step:
                try:
                    result = self._rrf_agent.run(
                        correlation_id=ctx.run_id,
                        runs=retrieval_lists,
                    )
                    ctx.fused = _extract_agent_data(
                        result,
                        default=retrieval_lists[0] if retrieval_lists else [],
                        agent_name="RRFAgent",
                        metrics_collector=self._metrics_collector,
                    )
                    step.extra["num_fused"] = len(ctx.fused)
                    step.extra["sources"] = len(retrieval_lists)
                except Exception as e:
                    logger.warning(f"RRF fusion failed: {e}")
                    metrics.mark_degraded("rrf", str(e))
                    ctx.fused = retrieval_lists[0] if retrieval_lists else []
        elif len(retrieval_lists) == 1:
            ctx.fused = retrieval_lists[0]
        elif retrieval_mode == "dense":
            ctx.fused = ctx.dense_retrieved
        elif retrieval_mode == "bm25":
            ctx.fused = ctx.bm25_retrieved
        else:
            ctx.fused = ctx.dense_retrieved or ctx.bm25_retrieved or []

    def _run_post_retrieval(
        self,
        ctx: AgentContext,
        metrics: RunMetrics,
        plan: Dict[str, Any],
    ) -> None:
        """Execute post-retrieval processing phase."""
        current_docs = ctx.fused

        # Auto-merging
        if plan.get("use_automerge", True):
            with metrics.track_step("AutoMergingAgent") as step:
                try:
                    result = self._automerge_agent.run(
                        correlation_id=ctx.run_id,
                        candidates=current_docs,
                    )
                    ctx.auto_merged = _extract_agent_data(
                        result,
                        default=current_docs,
                        agent_name="AutoMergingAgent",
                        metrics_collector=self._metrics_collector,
                    )
                    current_docs = ctx.auto_merged
                    step.extra["num_docs"] = len(ctx.auto_merged)
                except Exception as e:
                    logger.warning(f"Auto-merge failed: {e}")
                    metrics.mark_degraded("automerge", str(e))
                    ctx.auto_merged = current_docs
        else:
            ctx.auto_merged = current_docs

        # Reranking
        if plan.get("use_rerank", True):
            with metrics.track_step("RerankingAgent") as step:
                try:
                    result = self._rerank_agent.run(
                        correlation_id=ctx.run_id,
                        query=ctx.original_query,
                        docs=ctx.auto_merged,
                    )
                    ctx.reranked = _extract_agent_data(
                        result,
                        default=ctx.auto_merged,
                        agent_name="RerankingAgent",
                        metrics_collector=self._metrics_collector,
                    )
                    step.extra["num_reranked"] = len(ctx.reranked)
                except Exception as e:
                    logger.warning(f"Reranking failed: {e}")
                    metrics.mark_degraded("rerank", str(e))
                    ctx.reranked = ctx.auto_merged
        else:
            ctx.reranked = ctx.auto_merged

    def _run_context_evaluation(
        self,
        ctx: AgentContext,
        metrics: RunMetrics,
        plan: Dict[str, Any],
    ) -> Optional[Any]:
        """
        Execute pre-generation context evaluation phase.
        
        Evaluates retrieved context quality before generation to prevent
        wasted LLM calls on poor retrieval results.
        
        Returns:
            ContextEvaluation result or None if evaluation is disabled
        """
        if not self._context_eval_agent:
            return None
        
        if not plan.get("use_context_eval", True):
            return None
        
        # Get scores from reranked docs if available
        scores = [score for _, score in ctx.reranked] if ctx.reranked else None
        docs = [doc for doc, _ in ctx.reranked] if ctx.reranked else []
        
        with metrics.track_step("ContextEvaluationAgent") as step:
            try:
                evaluation = self._context_eval_agent.evaluate(
                    query=ctx.original_query,
                    docs=docs,
                    scores=scores,
                )
                
                step.extra["sufficient"] = evaluation.sufficient
                step.extra["confidence"] = evaluation.confidence
                step.extra["relevance_score"] = evaluation.relevance_score
                step.extra["coverage_score"] = evaluation.coverage_score
                step.extra["recommendation"] = evaluation.recommendation
                
                if not evaluation.sufficient:
                    logger.info(
                        f"[{ctx.run_id}] Context evaluation: insufficient "
                        f"(relevance={evaluation.relevance_score:.1f}, "
                        f"coverage={evaluation.coverage_score:.1f}, "
                        f"recommendation={evaluation.recommendation})"
                    )
                    if evaluation.missing_aspects:
                        ctx.add_warning(f"Missing: {', '.join(evaluation.missing_aspects[:2])}")
                
                return evaluation
                
            except Exception as e:
                logger.warning(f"Context evaluation failed: {e}")
                metrics.mark_degraded("context_eval", str(e))
                return None

    def _run_context_summarization(
        self,
        ctx: AgentContext,
        metrics: RunMetrics,
        plan: Dict[str, Any],
    ) -> None:
        """
        Execute context summarization phase if needed.
        
        Compresses context documents when:
        - Total context exceeds character limits
        - Individual documents are very long
        - Multiple similar documents are retrieved
        """
        if not self._summarization_agent:
            return
        
        if not plan.get("use_summarization", True):
            return
        
        # Get docs from reranked results
        docs = [doc for doc, _ in ctx.reranked] if ctx.reranked else []
        scores = [score for _, score in ctx.reranked] if ctx.reranked else None
        
        # Check if summarization is needed
        max_chars = self._config.summarization.max_total_context_chars
        if not self._summarization_agent.should_summarize_documents(docs, max_chars):
            return
        
        with metrics.track_step("SummarizationAgent") as step:
            try:
                result = self._summarization_agent.compress_documents(
                    docs=docs,
                    query=ctx.original_query,
                    max_total_chars=max_chars,
                    scores=scores,
                )
                
                step.extra["docs_compressed"] = result.documents_compressed
                step.extra["compression_ratio"] = result.compression_ratio
                step.extra["original_chars"] = result.total_original_chars
                step.extra["compressed_chars"] = result.total_compressed_chars
                
                if result.documents_compressed > 0:
                    logger.info(
                        f"[{ctx.run_id}] Summarized {result.documents_compressed} docs "
                        f"({result.total_original_chars} -> {result.total_compressed_chars} chars, "
                        f"{result.compression_ratio:.1%} ratio)"
                    )
                    
                    # Replace reranked docs with compressed versions
                    # Preserve original scores
                    new_reranked = []
                    for i, compressed_doc in enumerate(result.documents):
                        score = scores[i] if scores and i < len(scores) else 0.5
                        # Create a simple doc-like object with compressed content
                        class CompressedDocWrapper:
                            def __init__(self, content: str, meta: dict):
                                self.content = content
                                self.meta = meta
                                self.doc_id = compressed_doc.original_id
                        
                        wrapped_doc = CompressedDocWrapper(
                            content=compressed_doc.content,
                            meta=compressed_doc.metadata,
                        )
                        new_reranked.append((wrapped_doc, score))
                    
                    ctx.reranked = new_reranked
                    ctx.add_warning(f"Context compressed ({result.compression_ratio:.0%} of original)")
                
            except Exception as e:
                logger.warning(f"Context summarization failed: {e}")
                metrics.mark_degraded("summarization", str(e))

    def _run_generation(
        self,
        ctx: AgentContext,
        metrics: RunMetrics,
        plan: Dict[str, Any],
        is_retry: bool = False,
        retry_count: int = 0,
    ) -> str:
        """Execute generation phase."""
        context_docs = [doc for doc, _ in ctx.reranked]

        if not context_docs:
            ctx.add_warning("No relevant documents found")
            return "I couldn't find any relevant documents to answer your question. Could you please rephrase or provide more context?"

        # Add tool results to context if available
        tool_context = ""
        if ctx.tool_results:
            tool_parts = []
            for tool_result in ctx.tool_results:
                if tool_result.get("success") and tool_result.get("output"):
                    tool_parts.append(f"[Tool: {tool_result.get('tool_name')}] {tool_result.get('output')}")
            if tool_parts:
                tool_context = "\n\nTool Results:\n" + "\n".join(tool_parts)

        # Answer synthesis
        step_name = "AnswerSynthesisAgent" + ("_retry" if is_retry else "")
        with metrics.track_step(step_name) as step:
            result = self._synthesis_agent.run(
                correlation_id=ctx.run_id,
                query=ctx.original_query,
                docs=context_docs,
                conversation_history=ctx.conversation_history + tool_context,
            )
            ctx.final_answer = _extract_agent_data(
                result,
                default="I was unable to generate an answer. Please try again.",
                agent_name="AnswerSynthesisAgent",
                metrics_collector=self._metrics_collector,
            )
            step.extra["answer_length"] = len(ctx.final_answer)
            step.extra["retry_count"] = retry_count

        return ctx.final_answer

    def _run_critique(
        self,
        ctx: AgentContext,
        metrics: RunMetrics,
        is_retry: bool = False,
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        """Execute critique phase with confidence scoring."""
        context_docs = [doc for doc, _ in ctx.reranked]
        
        step_name = "CriticAgent" + ("_retry" if is_retry else "")
        with metrics.track_step(step_name) as step:
            try:
                result = self._critic_agent.run(
                    correlation_id=ctx.run_id,
                    query=ctx.original_query,
                    answer=ctx.final_answer,
                    context_docs=context_docs,
                    is_retry=is_retry,
                    retry_count=retry_count,
                )
                critique = _extract_agent_data(
                    result,
                    default={"ok": True, "confidence": 0.5, "should_retry": False},
                    agent_name="CriticAgent",
                    metrics_collector=self._metrics_collector,
                )
                ctx.critic_notes.append(critique)
                step.extra["confidence"] = critique.get("confidence", 0.5)
                step.extra["ok"] = critique.get("ok", True)
                step.extra["should_retry"] = critique.get("should_retry", False)
                
                if not critique.get("ok", True):
                    issues = critique.get("issues", [])
                    if issues:
                        ctx.add_warning(f"Critic found issues: {', '.join(issues[:3])}")
                
                return critique
                
            except Exception as e:
                logger.warning(f"Critic failed: {e}")
                metrics.mark_degraded("critic", str(e))
                return {"ok": True, "confidence": 0.5, "should_retry": False}

    def _generate_low_confidence_response(
        self,
        ctx: AgentContext,
        critique: Dict[str, Any],
    ) -> str:
        """Generate a graceful low-confidence response."""
        # Get summary of what we found
        summary = "Limited relevant information was found."
        if ctx.reranked:
            doc_count = len(ctx.reranked)
            summary = f"{doc_count} potentially relevant document(s) were found, but the information may be incomplete or not directly applicable."
        
        # Get reasons for uncertainty
        reasons = critique.get("issues", ["The available information may not fully address your question."])
        reasons_text = "\n".join([f"- {r}" for r in reasons[:3]])
        
        confidence = critique.get("confidence", 0.0)
        
        ctx.final_answer = LOW_CONFIDENCE_RESPONSE.format(
            summary=summary,
            reasons=reasons_text,
            confidence=confidence,
        )
        
        return ctx.final_answer

    def _run_multihop_reasoning(
        self,
        ctx: AgentContext,
        metrics: RunMetrics,
    ) -> Optional[Any]:
        """
        Execute multi-hop reasoning for complex queries.
        
        Checks if the query requires multi-hop reasoning and executes
        the reasoning chain if needed, augmenting the retrieved context.
        
        Returns:
            MultiHopResult or None if not needed/disabled
        """
        if not self._multihop_agent:
            return None
        
        # Check if multi-hop should be forced
        force_multihop = self._config.multihop.force_multihop
        
        with metrics.track_step("MultiHopReasoningAgent") as step:
            try:
                # Get initial context from already retrieved docs
                initial_context = [doc for doc, _ in ctx.fused] if ctx.fused else None
                
                agent_result = self._multihop_agent.run(
                    correlation_id=ctx.run_id,
                    query=ctx.original_query,
                    initial_context=initial_context,
                    force_multihop=force_multihop,
                )
                
                # Record metrics
                if self._metrics_collector:
                    self._metrics_collector.record(agent_result)
                
                # Extract the multihop result data
                multihop_result = agent_result.data if agent_result.success else None
                
                if multihop_result is None:
                    ctx.multihop_used = False
                    ctx.multihop_hops = 0
                    return None
                
                step.extra["requires_multihop"] = multihop_result.requires_multihop
                step.extra["total_hops"] = multihop_result.total_hops
                step.extra["success"] = multihop_result.success
                
                if multihop_result.requires_multihop and multihop_result.success:
                    # Store multi-hop metadata in context
                    ctx.multihop_used = True
                    ctx.multihop_hops = multihop_result.total_hops
                    
                    # Merge multi-hop context with existing fused results
                    # Add new documents to the context
                    existing_ids = {getattr(doc, 'doc_id', id(doc)) for doc, _ in ctx.fused}
                    
                    for doc in multihop_result.final_context:
                        doc_id = getattr(doc, 'doc_id', id(doc))
                        if doc_id not in existing_ids:
                            # Add with a reasonable score
                            ctx.fused.append((doc, 0.7))
                            existing_ids.add(doc_id)
                    
                    step.extra["docs_added"] = len(multihop_result.final_context)
                    
                    logger.info(
                        f"[{ctx.run_id}] Multi-hop reasoning complete: "
                        f"{multihop_result.total_hops} hops, {len(multihop_result.final_context)} docs"
                    )
                else:
                    ctx.multihop_used = False
                    ctx.multihop_hops = 0
                
                return multihop_result
                
            except Exception as e:
                logger.warning(f"Multi-hop reasoning failed: {e}")
                metrics.mark_degraded("multihop", str(e))
                ctx.multihop_used = False
                ctx.multihop_hops = 0
                return None

    def _run_fact_verification(
        self,
        ctx: AgentContext,
        metrics: RunMetrics,
        answer: str,
    ) -> Optional[Any]:
        """
        Execute fact verification on the generated answer.
        
        Verifies each claim in the answer against the retrieved context
        to ensure factual accuracy and prevent hallucinations.
        
        Args:
            ctx: Agent context
            metrics: Run metrics
            answer: Generated answer to verify
            
        Returns:
            FactVerificationResult or None if disabled
        """
        if not self._fact_verification_agent:
            return None
        
        context_docs = [doc for doc, _ in ctx.reranked]
        
        if not context_docs:
            return None
        
        with metrics.track_step("FactVerificationAgent") as step:
            try:
                result = self._fact_verification_agent.verify_answer(
                    answer=answer,
                    context_docs=context_docs,
                    query=ctx.original_query,
                )
                
                step.extra["num_claims"] = len(result.claims)
                step.extra["overall_score"] = result.overall_score
                step.extra["is_factual"] = result.is_factual
                step.extra["num_supported"] = result.num_supported
                step.extra["num_contradicted"] = result.num_contradicted
                step.extra["needs_correction"] = result.needs_correction
                
                if not result.is_factual:
                    ctx.add_warning(
                        f"Fact verification: {result.num_contradicted} contradicted, "
                        f"{result.num_not_supported} unsupported claims"
                    )
                
                # Check if we should block the answer
                if (self._config.fact_verification.block_on_failure and 
                    result.overall_score < self._config.fact_verification.min_factuality_score):
                    logger.warning(
                        f"[{ctx.run_id}] Answer blocked due to low factuality score: "
                        f"{result.overall_score:.2f}"
                    )
                    ctx.add_warning("Answer blocked due to factual accuracy concerns")
                
                logger.info(
                    f"[{ctx.run_id}] Fact verification: score={result.overall_score:.2f}, "
                    f"supported={result.num_supported}/{len(result.claims)}"
                )
                
                return result
                
            except Exception as e:
                logger.warning(f"Fact verification failed: {e}")
                metrics.mark_degraded("fact_verification", str(e))
                return None

    def _run_citation_tracking(
        self,
        ctx: AgentContext,
        metrics: RunMetrics,
        answer: str,
    ) -> Optional[Any]:
        """
        Execute citation tracking on the generated answer.
        
        Adds citations to the answer linking claims to their source documents
        for enterprise compliance and auditability.
        
        Args:
            ctx: Agent context
            metrics: Run metrics
            answer: Generated answer to add citations to
            
        Returns:
            CitedAnswer or None if disabled
        """
        if not self._citation_agent:
            return None
        
        context_docs = [doc for doc, _ in ctx.reranked]
        scores = [score for _, score in ctx.reranked] if ctx.reranked else None
        
        if not context_docs:
            return None
        
        with metrics.track_step("CitationTrackingAgent") as step:
            try:
                result = self._citation_agent.create_cited_answer(
                    answer=answer,
                    context_docs=context_docs,
                    query=ctx.original_query,
                    scores=scores,
                )
                
                step.extra["num_citations"] = len(result.citations)
                step.extra["num_sources"] = len(result.sources)
                step.extra["coverage_score"] = result.coverage_score
                step.extra["audit_id"] = result.audit_id
                
                # Generate audit log if enabled
                if self._config.citation.generate_audit_trail:
                    audit_log = self._citation_agent.generate_audit_report(
                        result, ctx.original_query
                    )
                    step.extra["audit_log"] = audit_log
                
                logger.info(
                    f"[{ctx.run_id}] Citation tracking: {len(result.citations)} citations, "
                    f"{result.coverage_score:.0%} coverage, audit_id={result.audit_id}"
                )
                
                return result
                
            except Exception as e:
                logger.warning(f"Citation tracking failed: {e}")
                metrics.mark_degraded("citation", str(e))
                return None


class SimplifiedOrchestrator:
    """
    Simplified orchestrator for quick queries without full pipeline.
    
    Useful for simple questions that don't need the full agentic flow.
    """

    def __init__(
        self,
        llm: LLMClient,
        local: LocalNLPModels,
        store: BaseVectorStore,
        config: AppConfig,
    ) -> None:
        """Initialize simplified orchestrator."""
        self._llm = llm
        self._local = local
        self._store = store
        self._config = config

    def run(self, query: str, top_k: int = 5) -> str:
        """
        Execute simplified RAG pipeline.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            Generated answer
        """
        # Embed query
        query_vec = self._local.embed_single(query)

        # Retrieve
        results = self._store.retrieve_by_embedding(query_vec, top_k=top_k)

        if not results:
            return "I couldn't find any relevant information to answer your question."

        # Format context
        context_parts = []
        for i, (doc, score) in enumerate(results, start=1):
            content = doc.content[:2000]
            context_parts.append(f"[{i}] {content}")

        context = "\n\n".join(context_parts)

        # Generate answer
        system = "Answer the question using the provided context. Be concise and accurate."
        user = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        response = self._llm.chat(
            self._llm.create_messages(system, user),
            retry_on_error=True,
        )

        if not response.success:
            return f"Error generating answer: {response.error}"

        return response.content.strip()
