"""
RAG pipeline agents package.

Provides specialized agents for each stage of the RAG pipeline:
    - Query processing (planning, decomposition, rewriting, expansion)
    - Retrieval (dense, sparse, web search)
    - Fusion (RRF)
    - Post-retrieval (auto-merging, reranking)
    - Generation (answer synthesis, critique)
    - Tools (calculator, code execution)
    - Strategy memory (adaptive retrieval)
    - Intelligent chunking (semantic document chunking)
    - Summarization (context compression)
    - Context evaluation (pre-generation quality gate)
    - Multi-hop reasoning (complex inference chains)
    - Fact verification (per-claim grounding)
    - Citation tracking (enterprise compliance)
    - Language detection (multilingual support)
    - Translation (canonical language indexing)
"""

# Base agent classes
from radiant_rag_mcp.agents.base_agent import (
    AgentCategory,
    AgentStatus,
    AgentMetrics,
    AgentResult,
    StructuredLogger,
    BaseAgent,
    LLMAgent,
    RetrievalAgent,
)

# Base context
from radiant_rag_mcp.agents.base import AgentContext, new_agent_context

# Planning
from radiant_rag_mcp.agents.planning import PlanningAgent

# Query processing
from radiant_rag_mcp.agents.decomposition import QueryDecompositionAgent
from radiant_rag_mcp.agents.rewrite import QueryRewriteAgent
from radiant_rag_mcp.agents.expansion import QueryExpansionAgent

# Retrieval
from radiant_rag_mcp.agents.dense import DenseRetrievalAgent
from radiant_rag_mcp.agents.bm25 import BM25RetrievalAgent
from radiant_rag_mcp.agents.web_search import WebSearchAgent

# Fusion
from radiant_rag_mcp.agents.fusion import RRFAgent

# Post-retrieval
from radiant_rag_mcp.agents.automerge import HierarchicalAutoMergingAgent
from radiant_rag_mcp.agents.rerank import CrossEncoderRerankingAgent

# Generation
from radiant_rag_mcp.agents.synthesis import AnswerSynthesisAgent
from radiant_rag_mcp.agents.critic import CriticAgent

# Tools (agentic enhancements)
from radiant_rag_mcp.agents.tools import (
    BaseTool,
    ToolType,
    ToolResult,
    ToolRegistry,
    ToolSelector,
    CalculatorTool,
    CodeExecutionTool,
    create_default_tool_registry,
)

# Strategy memory (agentic enhancements)
from radiant_rag_mcp.agents.strategy_memory import (
    RetrievalStrategyMemory,
    StrategyOutcome,
    QueryPattern,
    QueryPatternExtractor,
)

# Intelligent chunking (new)
from radiant_rag_mcp.agents.chunking import (
    IntelligentChunkingAgent,
    SemanticChunk,
    ChunkingResult,
)

# Summarization (new)
from radiant_rag_mcp.agents.summarization import (
    SummarizationAgent,
    CompressedDocument,
    SummarizationResult,
)

# Context evaluation (new)
from radiant_rag_mcp.agents.context_eval import (
    ContextEvaluationAgent,
    ContextEvaluation,
)

# Multi-hop reasoning (new)
from radiant_rag_mcp.agents.multihop import (
    MultiHopReasoningAgent,
    MultiHopResult,
    ReasoningStep,
)

# Fact verification (new)
from radiant_rag_mcp.agents.fact_verification import (
    FactVerificationAgent,
    FactVerificationResult,
    Claim,
    ClaimVerification,
    VerificationStatus,
)

# Citation tracking (new)
from radiant_rag_mcp.agents.citation import (
    CitationTrackingAgent,
    CitedAnswer,
    Citation,
    SourceDocument,
    CitationStyle,
)

# Language detection (new)
from radiant_rag_mcp.agents.language_detection import (
    LanguageDetectionAgent,
    LanguageDetection,
)

# Translation (new)
from radiant_rag_mcp.agents.translation import (
    TranslationAgent,
    TranslationResult,
    TranslationError,
)

# Video summarization (new)
from radiant_rag_mcp.agents.video_summarization import (
    VideoSummarizationAgent,
    VideoSummaryResult,
    VideoChapter,
)

__all__ = [
    # Base agent classes
    "AgentCategory",
    "AgentStatus",
    "AgentMetrics",
    "AgentResult",
    "StructuredLogger",
    "BaseAgent",
    "LLMAgent",
    "RetrievalAgent",
    # Base context
    "AgentContext",
    "new_agent_context",
    # Planning
    "PlanningAgent",
    # Query processing
    "QueryDecompositionAgent",
    "QueryRewriteAgent",
    "QueryExpansionAgent",
    # Retrieval
    "DenseRetrievalAgent",
    "BM25RetrievalAgent",
    "WebSearchAgent",
    # Fusion
    "RRFAgent",
    # Post-retrieval
    "HierarchicalAutoMergingAgent",
    "CrossEncoderRerankingAgent",
    # Generation
    "AnswerSynthesisAgent",
    "CriticAgent",
    # Tools
    "BaseTool",
    "ToolType",
    "ToolResult",
    "ToolRegistry",
    "ToolSelector",
    "CalculatorTool",
    "CodeExecutionTool",
    "create_default_tool_registry",
    # Strategy memory
    "RetrievalStrategyMemory",
    "StrategyOutcome",
    "QueryPattern",
    "QueryPatternExtractor",
    # Intelligent chunking
    "IntelligentChunkingAgent",
    "SemanticChunk",
    "ChunkingResult",
    # Summarization
    "SummarizationAgent",
    "CompressedDocument",
    "SummarizationResult",
    # Context evaluation
    "ContextEvaluationAgent",
    "ContextEvaluation",
    # Multi-hop reasoning
    "MultiHopReasoningAgent",
    "MultiHopResult",
    "ReasoningStep",
    # Fact verification
    "FactVerificationAgent",
    "FactVerificationResult",
    "Claim",
    "ClaimVerification",
    "VerificationStatus",
    # Citation tracking
    "CitationTrackingAgent",
    "CitedAnswer",
    "Citation",
    "SourceDocument",
    "CitationStyle",
    # Language detection
    "LanguageDetectionAgent",
    "LanguageDetection",
    # Translation
    "TranslationAgent",
    "TranslationResult",
    "TranslationError",
    # Video summarization
    "VideoSummarizationAgent",
    "VideoSummaryResult",
    "VideoChapter",
]
