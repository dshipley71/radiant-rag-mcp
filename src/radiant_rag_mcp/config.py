"""
Configuration management for Radiant Agentic RAG.

Loads configuration from YAML file with environment variable overrides.
Environment variables use the pattern: RADIANT_<SECTION>_<KEY>
Example: RADIANT_OLLAMA_CHAT_MODEL overrides ollama.chat_model
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)

# Default config file locations (searched in order)
DEFAULT_CONFIG_PATHS = [
    Path("./config.yaml"),
    Path("./config.yml"),
    Path("./radiant.yaml"),
    Path("./radiant.yml"),
    Path.home() / ".radiant" / "config.yaml",
    Path("/etc/radiant/config.yaml"),
]


def _get_env_override(section: str, key: str) -> Optional[str]:
    """Get environment variable override for a config key."""
    env_key = f"RADIANT_{section.upper()}_{key.upper()}"
    return os.environ.get(env_key)


def _parse_bool(value: Union[str, bool]) -> bool:
    """Parse boolean from string or bool."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_int(value: Union[str, int], default: int) -> int:
    """Parse integer from string or int."""
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return default


def _parse_float(value: Union[str, float], default: float) -> float:
    """Parse float from string or float."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return default


def _get_config_value(
    data: Dict[str, Any],
    section: str,
    key: str,
    default: Any,
    parser: Optional[callable] = None,
) -> Any:
    """Get config value with environment override support."""
    # Check environment override first
    env_value = _get_env_override(section, key)
    if env_value is not None:
        if parser:
            return parser(env_value) if parser != _parse_int and parser != _parse_float else parser(env_value, default)
        return env_value

    # Get from config data
    section_data = data.get(section, {})
    if isinstance(section_data, dict):
        value = section_data.get(key, default)
    else:
        value = default

    if parser and value is not None:
        if parser in (_parse_int, _parse_float):
            return parser(value, default)
        return parser(value)

    return value


def _get_nested_config_value(
    data: Dict[str, Any],
    section: str,
    subsection: str,
    key: str,
    default: Any,
    parser: Optional[callable] = None,
) -> Any:
    """Get nested config value with environment override support."""
    # Environment override: RADIANT_SECTION_SUBSECTION_KEY
    env_key = f"RADIANT_{section.upper()}_{subsection.upper()}_{key.upper()}"
    env_value = os.environ.get(env_key)
    if env_value is not None:
        if parser:
            if parser in (_parse_int, _parse_float):
                return parser(env_value, default)
            return parser(env_value)
        return env_value

    # Get from config data
    section_data = data.get(section, {})
    if isinstance(section_data, dict):
        subsection_data = section_data.get(subsection, {})
        if isinstance(subsection_data, dict):
            value = subsection_data.get(key, default)
        else:
            value = default
    else:
        value = default

    if parser and value is not None:
        if parser in (_parse_int, _parse_float):
            return parser(value, default)
        return parser(value)

    return value


@dataclass(frozen=True)
class OllamaConfig:
    """Ollama Cloud (OpenAI-compatible) configuration."""
    openai_base_url: str
    openai_api_key: str
    chat_model: str = "qwen2.5:latest"
    timeout: int = 90
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass(frozen=True)
class VLMCaptionerConfig:
    """Vision Language Model configuration for image captioning."""

    # Whether VLM captioning is enabled
    enabled: bool = True

    # HuggingFace model name
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"

    # Device: "auto", "cuda", "cpu", "mps"
    device: str = "auto"

    # Quantization for memory efficiency
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.2

    # Cache directory for model downloads (None = default HF cache)
    cache_dir: Optional[str] = None

    # Fallback to Ollama if HuggingFace not available
    ollama_fallback_url: str = "http://localhost:11434"
    ollama_fallback_model: str = "llava"


@dataclass(frozen=True)
class LocalModelsConfig:
    """Local HuggingFace/sentence-transformers configuration."""
    embed_model_name: str = "sentence-transformers/all-MiniLM-L12-v2"
    cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L12-v2"
    device: str = "auto"
    embedding_dimension: int = 384


@dataclass(frozen=True)
class LLMBackendConfig:
    """
    LLM backend configuration.

    Supports multiple backend types:
    - ollama: Ollama API (default)
    - vllm: vLLM OpenAI-compatible API
    - openai: OpenAI API
    - local: Local HuggingFace Transformers models
    """
    # Backend type
    backend_type: str = "ollama"

    # OpenAI-compatible settings (for ollama, vllm, openai)
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None

    # HuggingFace settings (for local backend)
    model_name: Optional[str] = None
    device: str = "auto"
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # Common settings
    timeout: int = 90
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass(frozen=True)
class EmbeddingBackendConfig:
    """
    Embedding backend configuration.

    Supports multiple backend types:
    - local: Local HuggingFace/sentence-transformers models (default)
    - ollama: Ollama embedding API
    - vllm: vLLM embedding API
    - openai: OpenAI embedding API
    """
    # Backend type
    backend_type: str = "local"

    # Model settings
    model_name: Optional[str] = "sentence-transformers/all-MiniLM-L12-v2"
    device: str = "auto"
    embedding_dimension: int = 384

    # OpenAI-compatible settings (for ollama, vllm, openai)
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None

    # Performance
    cache_size: int = 10000


@dataclass(frozen=True)
class RerankingBackendConfig:
    """
    Reranking backend configuration.

    Supports multiple backend types:
    - local: Local cross-encoder models (default)
    - ollama: Use Ollama LLM for reranking
    - vllm: Use vLLM for reranking
    - openai: Use OpenAI for reranking
    """
    # Backend type
    backend_type: str = "local"

    # Cross-encoder settings (for local backend)
    model_name: Optional[str] = "cross-encoder/ms-marco-MiniLM-L12-v2"
    device: str = "auto"

    # API settings (for ollama, vllm, openai)
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None


@dataclass(frozen=True)
class VectorIndexConfig:
    """Redis vector index configuration."""
    name: str = "radiant_vectors"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_runtime: int = 100
    distance_metric: str = "COSINE"


@dataclass(frozen=True)
class QuantizationConfig:
    """Binary quantization configuration for embeddings."""
    # Whether quantization is enabled
    enabled: bool = False
    
    # Precision for corpus embeddings: "binary", "int8", or "both"
    precision: str = "both"
    
    # Rescoring multiplier: retrieve N×top_k candidates for rescoring
    rescore_multiplier: float = 4.0
    
    # Whether to use rescoring step (improves accuracy)
    use_rescoring: bool = True
    
    # Path to int8 calibration ranges file (.npy format)
    # Required for int8 quantization precision
    int8_ranges_file: Optional[str] = None
    
    # Whether to keep int8 embeddings only on disk (saves memory)
    int8_on_disk_only: bool = True


@dataclass(frozen=True)
class ChromaConfig:
    """Chroma vector database configuration."""
    # Persistence directory for Chroma DB
    persist_directory: str = "./data/chroma_db"
    
    # Collection name for documents
    collection_name: str = "radiant_docs"
    
    # Distance function: "l2", "ip" (inner product), "cosine"
    distance_fn: str = "cosine"
    
    # Embedding dimension (must match the embedding model)
    embedding_dimension: int = 384
    
    # Maximum content characters (for truncation)
    max_content_chars: int = 200_000
    
    # Binary quantization configuration
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)



@dataclass(frozen=True)
class StorageConfig:
    """Storage backend configuration."""
    # Backend type: "redis", "chroma"
    backend: str = "redis"


@dataclass(frozen=True)
class RedisConfig:
    """Redis connection and storage configuration."""
    url: str = "redis://localhost:6379/0"
    key_prefix: str = "radiant"
    doc_ns: str = "doc"
    embed_ns: str = "emb"
    meta_ns: str = "meta"
    conversation_ns: str = "conv"
    max_content_chars: int = 200_000
    vector_index: VectorIndexConfig = field(default_factory=VectorIndexConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)


@dataclass(frozen=True)
class BM25Config:
    """BM25 index configuration."""
    # Base path for index file (without extension)
    # Stored as .json.gz (compressed JSON) for security
    index_path: str = "./data/bm25_index"
    max_documents: int = 100_000
    auto_save_threshold: int = 100
    k1: float = 1.5
    b: float = 0.75


@dataclass(frozen=True)
class IngestionConfig:
    """Ingestion and batch processing configuration."""
    # Batch size for embedding generation (larger = faster but more memory)
    embedding_batch_size: int = 32
    # Batch size for Redis pipeline operations
    redis_batch_size: int = 100
    # Enable batch processing (recommended for large corpora)
    batch_enabled: bool = True
    # Child chunk size for hierarchical storage
    child_chunk_size: int = 512
    # Child chunk overlap for hierarchical storage
    child_chunk_overlap: int = 50
    # Show progress bar during ingestion
    show_progress: bool = True
    # Embed parent documents (enables retrieval from parents)
    # When False (default), only leaf chunks are embedded and searchable
    # When True, parent documents are also embedded and can be retrieved
    embed_parents: bool = False


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval configuration."""
    dense_top_k: int = 10
    bm25_top_k: int = 10
    fused_top_k: int = 15
    rrf_k: int = 60
    min_similarity: float = 0.0
    # Search scope for vector retrieval
    # "leaves" - only search leaf chunks (default, original behavior)
    # "parents" - only search parent documents (requires embed_parents=True)
    # "all" - search both leaves and parents (requires embed_parents=True)
    search_scope: str = "leaves"


@dataclass(frozen=True)
class RerankConfig:
    """Reranking configuration."""
    top_k: int = 8
    max_doc_chars: int = 3000
    candidate_multiplier: int = 4
    min_candidates: int = 16


@dataclass(frozen=True)
class AutoMergeConfig:
    """Hierarchical auto-merging configuration."""
    min_children_to_merge: int = 2
    max_parent_chars: int = 50_000


@dataclass(frozen=True)
class SynthesisConfig:
    """Answer synthesis configuration."""
    max_context_docs: int = 8
    max_doc_chars: int = 4000
    include_history: bool = True
    max_history_turns: int = 5


@dataclass(frozen=True)
class CriticConfig:
    """Critic agent configuration."""
    enabled: bool = True
    max_context_docs: int = 8
    max_doc_chars: int = 1200
    retry_on_issues: bool = True  # Enable critic-driven retry by default
    max_retries: int = 2
    # Confidence threshold - below this returns "I don't know"
    confidence_threshold: float = 0.4
    # Minimum score to consider retrieval successful
    min_retrieval_confidence: float = 0.3


@dataclass(frozen=True)
class AgenticConfig:
    """Agentic behavior configuration."""
    # Enable dynamic retrieval mode selection
    dynamic_retrieval_mode: bool = True
    
    # Enable tool usage (calculator, code execution)
    tools_enabled: bool = True
    
    # Enable strategy memory for adaptive retrieval
    strategy_memory_enabled: bool = True
    
    # Path to store strategy memory (relative to data dir)
    strategy_memory_path: str = "./data/strategy_memory.json.gz"
    
    # Maximum retry attempts when critic finds issues
    max_critic_retries: int = 2
    
    # Confidence threshold for "I don't know" response
    confidence_threshold: float = 0.4
    
    # Enable query rewriting on retry
    rewrite_on_retry: bool = True
    
    # Expand retrieval on retry (fetch more documents)
    expand_retrieval_on_retry: bool = True
    
    # Retrieval expansion factor on retry
    retry_expansion_factor: float = 1.5


@dataclass(frozen=True)
class ChunkingConfig:
    """Intelligent chunking configuration."""
    
    # Enable intelligent (LLM-based) chunking
    enabled: bool = True
    
    # Use LLM for chunking decisions (vs rule-based only)
    use_llm_chunking: bool = True
    
    # Document length threshold for LLM chunking (shorter docs use rule-based)
    llm_chunk_threshold: int = 3000
    
    # Chunk size parameters
    min_chunk_size: int = 200
    max_chunk_size: int = 1500
    target_chunk_size: int = 800
    overlap_size: int = 100


@dataclass(frozen=True)
class SummarizationConfig:
    """Summarization agent configuration."""
    
    # Enable summarization agent
    enabled: bool = True
    
    # Minimum document length to trigger summarization
    min_doc_length_for_summary: int = 2000
    
    # Target summary length
    target_summary_length: int = 500
    
    # Conversation compression settings
    conversation_compress_threshold: int = 6
    conversation_preserve_recent: int = 2
    
    # Document deduplication settings
    similarity_threshold: float = 0.85
    max_cluster_size: int = 3
    
    # Maximum total context characters (triggers compression if exceeded)
    max_total_context_chars: int = 8000


@dataclass(frozen=True)
class ContextEvaluationConfig:
    """Context evaluation agent configuration."""
    
    # Enable pre-generation context evaluation
    enabled: bool = True
    
    # Use LLM for detailed evaluation (vs heuristics only)
    use_llm_evaluation: bool = True
    
    # Minimum score to consider context sufficient (0-1)
    sufficiency_threshold: float = 0.5
    
    # Minimum number of relevant documents required
    min_relevant_docs: int = 1
    
    # Maximum docs to include in evaluation
    max_docs_to_evaluate: int = 8
    
    # Maximum characters per doc for evaluation
    max_doc_chars: int = 1000
    
    # Skip generation if context evaluation recommends abort
    abort_on_poor_context: bool = False


@dataclass(frozen=True)
class MultiHopConfig:
    """Multi-hop reasoning agent configuration."""
    
    # Enable multi-hop reasoning for complex queries
    enabled: bool = True
    
    # Maximum reasoning hops
    max_hops: int = 3
    
    # Documents to retrieve per hop
    docs_per_hop: int = 5
    
    # Minimum confidence to continue chain
    min_confidence_to_continue: float = 0.3
    
    # Enable entity extraction for follow-up queries
    enable_entity_extraction: bool = True
    
    # Force multi-hop for all queries (for testing)
    force_multihop: bool = False


@dataclass(frozen=True)
class FactVerificationConfig:
    """Fact verification agent configuration."""
    
    # Enable fact verification
    enabled: bool = True
    
    # Minimum confidence to consider a claim supported
    min_support_confidence: float = 0.6
    
    # Maximum claims to verify (for efficiency)
    max_claims_to_verify: int = 20
    
    # Generate corrected answers when issues found
    generate_corrections: bool = True
    
    # Strict mode: require explicit support (vs inference)
    strict_mode: bool = False
    
    # Minimum overall score to accept answer (0-1)
    min_factuality_score: float = 0.5
    
    # Block answers that fail verification
    block_on_failure: bool = False


@dataclass(frozen=True)
class CitationConfig:
    """Citation tracking agent configuration."""
    
    # Enable citation tracking
    enabled: bool = True
    
    # Citation style: inline, footnote, academic, hyperlink, enterprise
    citation_style: str = "inline"
    
    # Minimum confidence for citations
    min_citation_confidence: float = 0.5
    
    # Maximum citations per claim
    max_citations_per_claim: int = 3
    
    # Include supporting excerpts in citations
    include_excerpts: bool = True
    
    # Maximum excerpt length
    excerpt_max_length: int = 200
    
    # Generate bibliography/references section
    generate_bibliography: bool = True
    
    # Generate audit trail
    generate_audit_trail: bool = True


@dataclass(frozen=True)
class LanguageDetectionConfig:
    """Language detection agent configuration."""

    # Enable language detection
    enabled: bool = True

    # Detection method: "fast" (fasttext), "llm", "auto"
    method: str = "fast"

    # Minimum confidence threshold for fasttext
    min_confidence: float = 0.7

    # Use LLM fallback for low-confidence detections
    use_llm_fallback: bool = True

    # Default language if detection fails
    fallback_language: str = "en"

    # FastText model configuration
    model_path: str = "./data/models/fasttext/lid.176.ftz"
    model_url: str = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
    auto_download: bool = True
    verify_checksum: bool = False  # Set to True to verify model integrity (requires checksum)


@dataclass(frozen=True)
class TranslationConfig:
    """Translation agent configuration."""
    
    # Enable translation
    enabled: bool = True
    
    # Translation method: "llm", "google", "deepl"
    method: str = "llm"
    
    # Canonical language for indexing (target language)
    canonical_language: str = "en"
    
    # Maximum characters per LLM translation call
    max_chars_per_llm_call: int = 4000
    
    # Translate documents at ingestion time
    translate_at_ingestion: bool = True
    
    # Translate retrieved docs at query time (fallback)
    translate_at_query: bool = False
    
    # Preserve original text in metadata
    preserve_original: bool = True
    
    # API keys for external services (if method != "llm")
    google_api_key: str = ""
    deepl_api_key: str = ""


@dataclass(frozen=True)
class QueryConfig:
    """Query processing configuration."""
    max_decomposed_queries: int = 5
    max_expansions: int = 12
    cache_enabled: bool = False
    cache_ttl: int = 3600


@dataclass(frozen=True)
class ConversationConfig:
    """Conversation history configuration."""
    enabled: bool = True
    max_turns: int = 50
    ttl: int = 86400
    use_history_for_retrieval: bool = True
    history_turns_for_context: int = 3


@dataclass(frozen=True)
class ParsingConfig:
    """LLM response parsing configuration."""
    max_retries: int = 2
    retry_delay: float = 0.5
    strict_json: bool = False
    log_failures: bool = True


@dataclass(frozen=True)
class UnstructuredCleaningConfig:
    """Unstructured document cleaning configuration."""
    enabled: bool = True
    bullets: bool = False
    extra_whitespace: bool = True
    dashes: bool = False
    trailing_punctuation: bool = False
    lowercase: bool = False
    preview_enabled: bool = False
    preview_max_items: int = 12
    preview_max_chars: int = 800


@dataclass(frozen=True)
class JSONParsingConfig:
    """JSON and JSONL document parsing configuration."""
    # Enable JSON/JSONL parsing
    enabled: bool = True

    # Parsing strategy: "auto", "flatten", "records", "semantic", "logs"
    default_strategy: str = "auto"

    # For "records" strategy: minimum array size to split into separate documents
    min_array_size_for_splitting: int = 3

    # For "semantic" strategy: fields to prioritize for text content
    text_fields: List[str] = field(default_factory=lambda: [
        "content", "body", "text", "description", "message", "summary", "details", "value"
    ])
    title_fields: List[str] = field(default_factory=lambda: [
        "title", "name", "subject", "heading", "label", "key"
    ])

    # For "flatten" strategy: maximum nesting depth to prevent infinite recursion
    max_nesting_depth: int = 10
    flatten_separator: str = "."

    # JSONL batch processing size
    jsonl_batch_size: int = 1000

    # Fields to always preserve in metadata (not just content)
    preserve_fields: List[str] = field(default_factory=lambda: [
        "id", "timestamp", "date", "created_at", "updated_at", "type", "category", "level", "status"
    ])


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = ""
    json_logging: bool = False
    quiet_third_party: bool = True  # Suppress noisy third-party library logs
    colorize: bool = True  # Enable colorized console output


@dataclass(frozen=True)
class MetricsConfig:
    """Metrics and observability configuration."""
    enabled: bool = True
    detailed_timing: bool = True
    store_history: bool = False
    history_retention: int = 100


@dataclass(frozen=True)
class PerformanceConfig:
    """Performance optimization configuration."""
    # Embedding cache settings
    embedding_cache_enabled: bool = True
    embedding_cache_size: int = 10000  # ~15MB RAM at 384-dim

    # Query result cache settings (for LLM operations)
    query_cache_enabled: bool = True
    query_cache_size: int = 1000  # Cache decomposition, rewrite, expansion results

    # Parallel execution settings
    parallel_retrieval_enabled: bool = True  # Run dense + BM25 in parallel
    parallel_postprocessing_enabled: bool = True  # Run fact verification + citation in parallel

    # Early stopping settings
    early_stopping_enabled: bool = True  # Skip expensive ops for simple queries
    simple_query_max_words: int = 10  # Max words for query to be considered simple

    # Retry optimization settings
    cache_retrieval_on_retry: bool = True  # Reuse retrieval results across retries
    targeted_retry_enabled: bool = True  # Only retry what failed, not full pipeline


@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline feature flags."""
    use_planning: bool = True
    use_decomposition: bool = True
    use_rewrite: bool = True
    use_expansion: bool = True
    use_rrf: bool = True
    use_automerge: bool = True
    use_rerank: bool = True
    use_critic: bool = True


@dataclass(frozen=True)
class WebCrawlerConfig:
    """Web crawler configuration for URL ingestion."""

    # Crawl depth (0 = seed URLs only, 1 = seed + direct links, etc.)
    max_depth: int = 2

    # Maximum total pages to crawl per session
    max_pages: int = 100

    # Only crawl pages from the same domain as seed URLs
    same_domain_only: bool = True

    # URL patterns (regex) - URLs must match at least one include pattern
    include_patterns: List[str] = field(default_factory=list)

    # URL patterns (regex) - URLs matching any exclude pattern are skipped
    exclude_patterns: List[str] = field(default_factory=list)

    # Request timeout in seconds
    timeout: int = 30

    # Delay between requests in seconds (rate limiting)
    delay: float = 0.5

    # User agent string for requests
    user_agent: str = "AgenticRAG-Crawler/1.0"

    # Basic authentication credentials (if required)
    basic_auth_user: str = ""
    basic_auth_password: str = ""

    # SSL certificate verification
    verify_ssl: bool = True

    # Temporary directory for downloaded files (None = system temp)
    temp_dir: Optional[str] = None

    # Whether to follow redirects
    follow_redirects: bool = True

    # Maximum file size to download (bytes, 0 = unlimited)
    max_file_size: int = 50_000_000  # 50 MB

    # Respect robots.txt (future enhancement)
    respect_robots_txt: bool = True


@dataclass(frozen=True)
class WebSearchConfig:
    """Real-time web search agent configuration."""

    # Enable/disable web search during queries
    enabled: bool = False

    # Maximum number of URLs to fetch per query
    max_results: int = 5

    # Maximum pages to fetch (may be less than max_results if pages fail)
    max_pages: int = 3

    # Request timeout in seconds
    timeout: int = 15

    # User agent for requests
    user_agent: str = "AgenticRAG-WebSearch/1.0"

    # Whether to include web results in the answer synthesis
    include_in_synthesis: bool = True

    # Minimum relevance score to include web result (0.0-1.0)
    min_relevance: float = 0.3

    # Search engine to use (for future expansion)
    # Currently only "direct" is supported (direct URL fetching based on query analysis)
    search_mode: str = "direct"

    # Whether to cache web search results
    cache_enabled: bool = True

    # Cache TTL in seconds (default 1 hour)
    cache_ttl: int = 3600

    # Keywords that trigger web search (if empty, rely on planner)
    trigger_keywords: List[str] = field(default_factory=lambda: [
        "latest", "recent", "current", "today", "news",
        "update", "new", "2024", "2025", "now",
    ])

    # Domains to prefer for searches (optional)
    preferred_domains: List[str] = field(default_factory=list)

    # Domains to block
    blocked_domains: List[str] = field(default_factory=lambda: [
        "facebook.com", "twitter.com", "instagram.com",
        "tiktok.com", "pinterest.com",
    ])


@dataclass(frozen=True)
class AppConfig:
    """Main application configuration."""
    ollama: OllamaConfig
    local_models: LocalModelsConfig
    llm_backend: LLMBackendConfig
    embedding_backend: EmbeddingBackendConfig
    reranking_backend: RerankingBackendConfig
    storage: StorageConfig
    redis: RedisConfig
    chroma: ChromaConfig
    bm25: BM25Config
    ingestion: IngestionConfig
    retrieval: RetrievalConfig
    rerank: RerankConfig
    automerge: AutoMergeConfig
    synthesis: SynthesisConfig
    critic: CriticConfig
    agentic: AgenticConfig
    chunking: ChunkingConfig
    summarization: SummarizationConfig
    context_evaluation: ContextEvaluationConfig
    multihop: MultiHopConfig
    fact_verification: FactVerificationConfig
    citation: CitationConfig
    language_detection: LanguageDetectionConfig
    translation: TranslationConfig
    query: QueryConfig
    conversation: ConversationConfig
    parsing: ParsingConfig
    unstructured_cleaning: UnstructuredCleaningConfig
    json_parsing: JSONParsingConfig
    logging: LoggingConfig
    metrics: MetricsConfig
    performance: PerformanceConfig
    pipeline: PipelineConfig
    vlm: VLMCaptionerConfig
    web_crawler: WebCrawlerConfig
    web_search: WebSearchConfig


def find_config_file(config_path: Optional[str] = None) -> Optional[Path]:
    """Find configuration file from explicit path or default locations."""
    if config_path:
        path = Path(config_path)
        if path.exists():
            return path
        logger.warning(f"Config file not found at specified path: {config_path}")
        return None

    for path in DEFAULT_CONFIG_PATHS:
        if path.exists():
            logger.info(f"Found config file: {path}")
            return path

    return None


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.error(f"Failed to load config file {path}: {e}")
        return {}


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load application configuration.

    Priority (highest to lowest):
    1. Environment variables (RADIANT_<SECTION>_<KEY>)
    2. Config file values
    3. Default values

    Args:
        config_path: Optional explicit path to config file

    Returns:
        AppConfig instance

    Raises:
        ValueError: If required configuration is missing
    """
    # Load config file if available
    config_file = find_config_file(config_path)
    data = load_yaml_config(config_file) if config_file else {}

    # Build configuration with environment overrides

    # Check if new backend configurations are present
    has_new_backend_config = (
        "llm_backend" in data or
        "embedding_backend" in data or
        "reranking_backend" in data
    )

    # Ollama config (optional if using new backend system)
    ollama_base_url = _get_config_value(data, "ollama", "openai_base_url", "")
    ollama_api_key = _get_config_value(data, "ollama", "openai_api_key", "")

    # Also check legacy environment variables
    if not ollama_base_url:
        ollama_base_url = os.environ.get("OLLAMA_OPENAI_BASE_URL", "")
    if not ollama_api_key:
        ollama_api_key = os.environ.get("OLLAMA_OPENAI_API_KEY", "")

    # Only require ollama config if NOT using new backend system
    if not has_new_backend_config:
        if not ollama_base_url:
            raise ValueError(
                "Missing Ollama base URL. Set RADIANT_OLLAMA_OPENAI_BASE_URL or "
                "OLLAMA_OPENAI_BASE_URL environment variable, or set ollama.openai_base_url in config file."
            )
        if not ollama_api_key:
            raise ValueError(
                "Missing Ollama API key. Set RADIANT_OLLAMA_OPENAI_API_KEY or "
                "OLLAMA_OPENAI_API_KEY environment variable, or set ollama.openai_api_key in config file."
            )

    # Create ollama config (use defaults if not present)
    ollama = OllamaConfig(
        openai_base_url=(ollama_base_url.rstrip("/") if ollama_base_url else "http://localhost:11434/v1"),
        openai_api_key=(ollama_api_key if ollama_api_key else "ollama"),
        chat_model=_get_config_value(data, "ollama", "chat_model", "qwen2.5:latest"),
        timeout=_get_config_value(data, "ollama", "timeout", 90, _parse_int),
        max_retries=_get_config_value(data, "ollama", "max_retries", 3, _parse_int),
        retry_delay=_get_config_value(data, "ollama", "retry_delay", 1.0, _parse_float),
    )

    vlm = VLMCaptionerConfig(
        enabled=_get_config_value(data, "vlm", "enabled", True, _parse_bool),
        model_name=_get_config_value(data, "vlm", "model_name", "Qwen/Qwen2-VL-2B-Instruct"),
        device=_get_config_value(data, "vlm", "device", "auto"),
        load_in_4bit=_get_config_value(data, "vlm", "load_in_4bit", False, _parse_bool),
        load_in_8bit=_get_config_value(data, "vlm", "load_in_8bit", False, _parse_bool),
        max_new_tokens=_get_config_value(data, "vlm", "max_new_tokens", 512, _parse_int),
        temperature=_get_config_value(data, "vlm", "temperature", 0.2, _parse_float),
        cache_dir=_get_config_value(data, "vlm", "cache_dir", None) or None,
        ollama_fallback_url=_get_config_value(data, "vlm", "ollama_fallback_url", "http://localhost:11434"),
        ollama_fallback_model=_get_config_value(data, "vlm", "ollama_fallback_model", "llava"),
    )

    local_models = LocalModelsConfig(
        embed_model_name=_get_config_value(data, "local_models", "embed_model_name", "sentence-transformers/all-MiniLM-L12-v2"),
        cross_encoder_name=_get_config_value(data, "local_models", "cross_encoder_name", "cross-encoder/ms-marco-MiniLM-L12-v2"),
        device=_get_config_value(data, "local_models", "device", "auto"),
        embedding_dimension=_get_config_value(data, "local_models", "embedding_dimension", 384, _parse_int),
    )

    # New backend configurations (with fallback to old config for backward compatibility)
    # Check if new llm_backend section exists
    has_llm_backend = "llm_backend" in data and data["llm_backend"] is not None

    if has_llm_backend:
        # Use new configuration format
        llm_backend = LLMBackendConfig(
            backend_type=_get_config_value(data, "llm_backend", "backend_type", "ollama"),
            base_url=_get_config_value(data, "llm_backend", "base_url", None) or None,
            api_key=_get_config_value(data, "llm_backend", "api_key", None) or None,
            model=_get_config_value(data, "llm_backend", "model", None) or None,
            model_name=_get_config_value(data, "llm_backend", "model_name", None) or None,
            device=_get_config_value(data, "llm_backend", "device", "auto"),
            load_in_4bit=_get_config_value(data, "llm_backend", "load_in_4bit", False, _parse_bool),
            load_in_8bit=_get_config_value(data, "llm_backend", "load_in_8bit", False, _parse_bool),
            timeout=_get_config_value(data, "llm_backend", "timeout", 90, _parse_int),
            max_retries=_get_config_value(data, "llm_backend", "max_retries", 3, _parse_int),
            retry_delay=_get_config_value(data, "llm_backend", "retry_delay", 1.0, _parse_float),
        )
    else:
        # Fallback to old ollama config
        llm_backend = LLMBackendConfig(
            backend_type="ollama",
            base_url=ollama.openai_base_url,
            api_key=ollama.openai_api_key,
            model=ollama.chat_model,
            timeout=ollama.timeout,
            max_retries=ollama.max_retries,
            retry_delay=ollama.retry_delay,
        )

    # Check if new embedding_backend section exists
    has_embedding_backend = "embedding_backend" in data and data["embedding_backend"] is not None

    if has_embedding_backend:
        # Use new configuration format
        embedding_backend = EmbeddingBackendConfig(
            backend_type=_get_config_value(data, "embedding_backend", "backend_type", "local"),
            model_name=_get_config_value(data, "embedding_backend", "model_name", "sentence-transformers/all-MiniLM-L12-v2"),
            device=_get_config_value(data, "embedding_backend", "device", "auto"),
            embedding_dimension=_get_config_value(data, "embedding_backend", "embedding_dimension", 384, _parse_int),
            base_url=_get_config_value(data, "embedding_backend", "base_url", None) or None,
            api_key=_get_config_value(data, "embedding_backend", "api_key", None) or None,
            model=_get_config_value(data, "embedding_backend", "model", None) or None,
            cache_size=_get_config_value(data, "embedding_backend", "cache_size", 10000, _parse_int),
        )
    else:
        # Fallback to old local_models config
        embedding_backend = EmbeddingBackendConfig(
            backend_type="local",
            model_name=local_models.embed_model_name,
            device=local_models.device,
            embedding_dimension=local_models.embedding_dimension,
            cache_size=10000,
        )

    # Check if new reranking_backend section exists
    has_reranking_backend = "reranking_backend" in data and data["reranking_backend"] is not None

    if has_reranking_backend:
        # Use new configuration format
        reranking_backend = RerankingBackendConfig(
            backend_type=_get_config_value(data, "reranking_backend", "backend_type", "local"),
            model_name=_get_config_value(data, "reranking_backend", "model_name", "cross-encoder/ms-marco-MiniLM-L12-v2"),
            device=_get_config_value(data, "reranking_backend", "device", "auto"),
            base_url=_get_config_value(data, "reranking_backend", "base_url", None) or None,
            api_key=_get_config_value(data, "reranking_backend", "api_key", None) or None,
            model=_get_config_value(data, "reranking_backend", "model", None) or None,
        )
    else:
        # Fallback to old local_models config
        reranking_backend = RerankingBackendConfig(
            backend_type="local",
            model_name=local_models.cross_encoder_name,
            device=local_models.device,
        )

    vector_index = VectorIndexConfig(
        name=_get_nested_config_value(data, "redis", "vector_index", "name", "radiant_vectors"),
        hnsw_m=_get_nested_config_value(data, "redis", "vector_index", "hnsw_m", 16, _parse_int),
        hnsw_ef_construction=_get_nested_config_value(data, "redis", "vector_index", "hnsw_ef_construction", 200, _parse_int),
        hnsw_ef_runtime=_get_nested_config_value(data, "redis", "vector_index", "hnsw_ef_runtime", 100, _parse_int),
        distance_metric=_get_nested_config_value(data, "redis", "vector_index", "distance_metric", "COSINE"),
    )

    redis = RedisConfig(
        url=_get_config_value(data, "redis", "url", "redis://localhost:6379/0"),
        key_prefix=_get_config_value(data, "redis", "key_prefix", "radiant"),
        doc_ns=_get_config_value(data, "redis", "doc_ns", "doc"),
        embed_ns=_get_config_value(data, "redis", "embed_ns", "emb"),
        meta_ns=_get_config_value(data, "redis", "meta_ns", "meta"),
        conversation_ns=_get_config_value(data, "redis", "conversation_ns", "conv"),
        max_content_chars=_get_config_value(data, "redis", "max_content_chars", 200_000, _parse_int),
        vector_index=vector_index,
    )

    # Storage backend selection (redis is default)
    storage = StorageConfig(
        backend=_get_config_value(data, "storage", "backend", "redis"),
    )

    # Chroma configuration
    chroma = ChromaConfig(
        persist_directory=_get_config_value(data, "chroma", "persist_directory", "./data/chroma_db"),
        collection_name=_get_config_value(data, "chroma", "collection_name", "radiant_docs"),
        distance_fn=_get_config_value(data, "chroma", "distance_fn", "cosine"),
        embedding_dimension=_get_config_value(data, "chroma", "embedding_dimension", 384, _parse_int),
        max_content_chars=_get_config_value(data, "chroma", "max_content_chars", 200_000, _parse_int),
    )


    bm25 = BM25Config(
        index_path=_get_config_value(data, "bm25", "index_path", "./data/bm25_index.pkl"),
        max_documents=_get_config_value(data, "bm25", "max_documents", 100_000, _parse_int),
        auto_save_threshold=_get_config_value(data, "bm25", "auto_save_threshold", 100, _parse_int),
        k1=_get_config_value(data, "bm25", "k1", 1.5, _parse_float),
        b=_get_config_value(data, "bm25", "b", 0.75, _parse_float),
    )

    ingestion = IngestionConfig(
        embedding_batch_size=_get_config_value(data, "ingestion", "embedding_batch_size", 32, _parse_int),
        redis_batch_size=_get_config_value(data, "ingestion", "redis_batch_size", 100, _parse_int),
        batch_enabled=_get_config_value(data, "ingestion", "batch_enabled", True, _parse_bool),
        child_chunk_size=_get_config_value(data, "ingestion", "child_chunk_size", 512, _parse_int),
        child_chunk_overlap=_get_config_value(data, "ingestion", "child_chunk_overlap", 50, _parse_int),
        show_progress=_get_config_value(data, "ingestion", "show_progress", True, _parse_bool),
        embed_parents=_get_config_value(data, "ingestion", "embed_parents", False, _parse_bool),
    )

    retrieval = RetrievalConfig(
        dense_top_k=_get_config_value(data, "retrieval", "dense_top_k", 10, _parse_int),
        bm25_top_k=_get_config_value(data, "retrieval", "bm25_top_k", 10, _parse_int),
        fused_top_k=_get_config_value(data, "retrieval", "fused_top_k", 15, _parse_int),
        rrf_k=_get_config_value(data, "retrieval", "rrf_k", 60, _parse_int),
        min_similarity=_get_config_value(data, "retrieval", "min_similarity", 0.0, _parse_float),
        search_scope=_get_config_value(data, "retrieval", "search_scope", "leaves"),
    )

    rerank = RerankConfig(
        top_k=_get_config_value(data, "rerank", "top_k", 8, _parse_int),
        max_doc_chars=_get_config_value(data, "rerank", "max_doc_chars", 3000, _parse_int),
        candidate_multiplier=_get_config_value(data, "rerank", "candidate_multiplier", 4, _parse_int),
        min_candidates=_get_config_value(data, "rerank", "min_candidates", 16, _parse_int),
    )

    automerge = AutoMergeConfig(
        min_children_to_merge=_get_config_value(data, "automerge", "min_children_to_merge", 2, _parse_int),
        max_parent_chars=_get_config_value(data, "automerge", "max_parent_chars", 50_000, _parse_int),
    )

    synthesis = SynthesisConfig(
        max_context_docs=_get_config_value(data, "synthesis", "max_context_docs", 8, _parse_int),
        max_doc_chars=_get_config_value(data, "synthesis", "max_doc_chars", 4000, _parse_int),
        include_history=_get_config_value(data, "synthesis", "include_history", True, _parse_bool),
        max_history_turns=_get_config_value(data, "synthesis", "max_history_turns", 5, _parse_int),
    )

    critic = CriticConfig(
        enabled=_get_config_value(data, "critic", "enabled", True, _parse_bool),
        max_context_docs=_get_config_value(data, "critic", "max_context_docs", 8, _parse_int),
        max_doc_chars=_get_config_value(data, "critic", "max_doc_chars", 1200, _parse_int),
        retry_on_issues=_get_config_value(data, "critic", "retry_on_issues", True, _parse_bool),
        max_retries=_get_config_value(data, "critic", "max_retries", 2, _parse_int),
        confidence_threshold=_get_config_value(data, "critic", "confidence_threshold", 0.4, _parse_float),
        min_retrieval_confidence=_get_config_value(data, "critic", "min_retrieval_confidence", 0.3, _parse_float),
    )

    agentic = AgenticConfig(
        dynamic_retrieval_mode=_get_config_value(data, "agentic", "dynamic_retrieval_mode", True, _parse_bool),
        tools_enabled=_get_config_value(data, "agentic", "tools_enabled", True, _parse_bool),
        strategy_memory_enabled=_get_config_value(data, "agentic", "strategy_memory_enabled", True, _parse_bool),
        strategy_memory_path=_get_config_value(data, "agentic", "strategy_memory_path", "./data/strategy_memory.json.gz"),
        max_critic_retries=_get_config_value(data, "agentic", "max_critic_retries", 2, _parse_int),
        confidence_threshold=_get_config_value(data, "agentic", "confidence_threshold", 0.4, _parse_float),
        rewrite_on_retry=_get_config_value(data, "agentic", "rewrite_on_retry", True, _parse_bool),
        expand_retrieval_on_retry=_get_config_value(data, "agentic", "expand_retrieval_on_retry", True, _parse_bool),
        retry_expansion_factor=_get_config_value(data, "agentic", "retry_expansion_factor", 1.5, _parse_float),
    )

    chunking = ChunkingConfig(
        enabled=_get_config_value(data, "chunking", "enabled", True, _parse_bool),
        use_llm_chunking=_get_config_value(data, "chunking", "use_llm_chunking", True, _parse_bool),
        llm_chunk_threshold=_get_config_value(data, "chunking", "llm_chunk_threshold", 3000, _parse_int),
        min_chunk_size=_get_config_value(data, "chunking", "min_chunk_size", 200, _parse_int),
        max_chunk_size=_get_config_value(data, "chunking", "max_chunk_size", 1500, _parse_int),
        target_chunk_size=_get_config_value(data, "chunking", "target_chunk_size", 800, _parse_int),
        overlap_size=_get_config_value(data, "chunking", "overlap_size", 100, _parse_int),
    )

    summarization = SummarizationConfig(
        enabled=_get_config_value(data, "summarization", "enabled", True, _parse_bool),
        min_doc_length_for_summary=_get_config_value(data, "summarization", "min_doc_length_for_summary", 2000, _parse_int),
        target_summary_length=_get_config_value(data, "summarization", "target_summary_length", 500, _parse_int),
        conversation_compress_threshold=_get_config_value(data, "summarization", "conversation_compress_threshold", 6, _parse_int),
        conversation_preserve_recent=_get_config_value(data, "summarization", "conversation_preserve_recent", 2, _parse_int),
        similarity_threshold=_get_config_value(data, "summarization", "similarity_threshold", 0.85, _parse_float),
        max_cluster_size=_get_config_value(data, "summarization", "max_cluster_size", 3, _parse_int),
        max_total_context_chars=_get_config_value(data, "summarization", "max_total_context_chars", 8000, _parse_int),
    )

    context_evaluation = ContextEvaluationConfig(
        enabled=_get_config_value(data, "context_evaluation", "enabled", True, _parse_bool),
        use_llm_evaluation=_get_config_value(data, "context_evaluation", "use_llm_evaluation", True, _parse_bool),
        sufficiency_threshold=_get_config_value(data, "context_evaluation", "sufficiency_threshold", 0.5, _parse_float),
        min_relevant_docs=_get_config_value(data, "context_evaluation", "min_relevant_docs", 1, _parse_int),
        max_docs_to_evaluate=_get_config_value(data, "context_evaluation", "max_docs_to_evaluate", 8, _parse_int),
        max_doc_chars=_get_config_value(data, "context_evaluation", "max_doc_chars", 1000, _parse_int),
        abort_on_poor_context=_get_config_value(data, "context_evaluation", "abort_on_poor_context", False, _parse_bool),
    )

    multihop = MultiHopConfig(
        enabled=_get_config_value(data, "multihop", "enabled", True, _parse_bool),
        max_hops=_get_config_value(data, "multihop", "max_hops", 3, _parse_int),
        docs_per_hop=_get_config_value(data, "multihop", "docs_per_hop", 5, _parse_int),
        min_confidence_to_continue=_get_config_value(data, "multihop", "min_confidence_to_continue", 0.3, _parse_float),
        enable_entity_extraction=_get_config_value(data, "multihop", "enable_entity_extraction", True, _parse_bool),
        force_multihop=_get_config_value(data, "multihop", "force_multihop", False, _parse_bool),
    )

    fact_verification = FactVerificationConfig(
        enabled=_get_config_value(data, "fact_verification", "enabled", True, _parse_bool),
        min_support_confidence=_get_config_value(data, "fact_verification", "min_support_confidence", 0.6, _parse_float),
        max_claims_to_verify=_get_config_value(data, "fact_verification", "max_claims_to_verify", 20, _parse_int),
        generate_corrections=_get_config_value(data, "fact_verification", "generate_corrections", True, _parse_bool),
        strict_mode=_get_config_value(data, "fact_verification", "strict_mode", False, _parse_bool),
        min_factuality_score=_get_config_value(data, "fact_verification", "min_factuality_score", 0.5, _parse_float),
        block_on_failure=_get_config_value(data, "fact_verification", "block_on_failure", False, _parse_bool),
    )

    citation = CitationConfig(
        enabled=_get_config_value(data, "citation", "enabled", True, _parse_bool),
        citation_style=_get_config_value(data, "citation", "citation_style", "inline"),
        min_citation_confidence=_get_config_value(data, "citation", "min_citation_confidence", 0.5, _parse_float),
        max_citations_per_claim=_get_config_value(data, "citation", "max_citations_per_claim", 3, _parse_int),
        include_excerpts=_get_config_value(data, "citation", "include_excerpts", True, _parse_bool),
        excerpt_max_length=_get_config_value(data, "citation", "excerpt_max_length", 200, _parse_int),
        generate_bibliography=_get_config_value(data, "citation", "generate_bibliography", True, _parse_bool),
        generate_audit_trail=_get_config_value(data, "citation", "generate_audit_trail", True, _parse_bool),
    )

    language_detection = LanguageDetectionConfig(
        enabled=_get_config_value(data, "language_detection", "enabled", True, _parse_bool),
        method=_get_config_value(data, "language_detection", "method", "fast"),
        min_confidence=_get_config_value(data, "language_detection", "min_confidence", 0.7, _parse_float),
        use_llm_fallback=_get_config_value(data, "language_detection", "use_llm_fallback", True, _parse_bool),
        fallback_language=_get_config_value(data, "language_detection", "fallback_language", "en"),
    )

    translation = TranslationConfig(
        enabled=_get_config_value(data, "translation", "enabled", True, _parse_bool),
        method=_get_config_value(data, "translation", "method", "llm"),
        canonical_language=_get_config_value(data, "translation", "canonical_language", "en"),
        max_chars_per_llm_call=_get_config_value(data, "translation", "max_chars_per_llm_call", 4000, _parse_int),
        translate_at_ingestion=_get_config_value(data, "translation", "translate_at_ingestion", True, _parse_bool),
        translate_at_query=_get_config_value(data, "translation", "translate_at_query", False, _parse_bool),
        preserve_original=_get_config_value(data, "translation", "preserve_original", True, _parse_bool),
        google_api_key=_get_config_value(data, "translation", "google_api_key", ""),
        deepl_api_key=_get_config_value(data, "translation", "deepl_api_key", ""),
    )

    query = QueryConfig(
        max_decomposed_queries=_get_config_value(data, "query", "max_decomposed_queries", 5, _parse_int),
        max_expansions=_get_config_value(data, "query", "max_expansions", 12, _parse_int),
        cache_enabled=_get_config_value(data, "query", "cache_enabled", False, _parse_bool),
        cache_ttl=_get_config_value(data, "query", "cache_ttl", 3600, _parse_int),
    )

    conversation = ConversationConfig(
        enabled=_get_config_value(data, "conversation", "enabled", True, _parse_bool),
        max_turns=_get_config_value(data, "conversation", "max_turns", 50, _parse_int),
        ttl=_get_config_value(data, "conversation", "ttl", 86400, _parse_int),
        use_history_for_retrieval=_get_config_value(data, "conversation", "use_history_for_retrieval", True, _parse_bool),
        history_turns_for_context=_get_config_value(data, "conversation", "history_turns_for_context", 3, _parse_int),
    )

    parsing = ParsingConfig(
        max_retries=_get_config_value(data, "parsing", "max_retries", 2, _parse_int),
        retry_delay=_get_config_value(data, "parsing", "retry_delay", 0.5, _parse_float),
        strict_json=_get_config_value(data, "parsing", "strict_json", False, _parse_bool),
        log_failures=_get_config_value(data, "parsing", "log_failures", True, _parse_bool),
    )

    unstructured_cleaning = UnstructuredCleaningConfig(
        enabled=_get_config_value(data, "unstructured_cleaning", "enabled", True, _parse_bool),
        bullets=_get_config_value(data, "unstructured_cleaning", "bullets", False, _parse_bool),
        extra_whitespace=_get_config_value(data, "unstructured_cleaning", "extra_whitespace", True, _parse_bool),
        dashes=_get_config_value(data, "unstructured_cleaning", "dashes", False, _parse_bool),
        trailing_punctuation=_get_config_value(data, "unstructured_cleaning", "trailing_punctuation", False, _parse_bool),
        lowercase=_get_config_value(data, "unstructured_cleaning", "lowercase", False, _parse_bool),
        preview_enabled=_get_config_value(data, "unstructured_cleaning", "preview_enabled", False, _parse_bool),
        preview_max_items=_get_config_value(data, "unstructured_cleaning", "preview_max_items", 12, _parse_int),
        preview_max_chars=_get_config_value(data, "unstructured_cleaning", "preview_max_chars", 800, _parse_int),
    )

    # JSON parsing configuration
    json_text_fields = _get_config_value(data, "json_parsing", "text_fields", [])
    json_title_fields = _get_config_value(data, "json_parsing", "title_fields", [])
    json_preserve_fields = _get_config_value(data, "json_parsing", "preserve_fields", [])

    json_parsing = JSONParsingConfig(
        enabled=_get_config_value(data, "json_parsing", "enabled", True, _parse_bool),
        default_strategy=_get_config_value(data, "json_parsing", "default_strategy", "auto"),
        min_array_size_for_splitting=_get_config_value(data, "json_parsing", "min_array_size_for_splitting", 3, _parse_int),
        text_fields=json_text_fields if json_text_fields else [
            "content", "body", "text", "description", "message", "summary", "details", "value"
        ],
        title_fields=json_title_fields if json_title_fields else [
            "title", "name", "subject", "heading", "label", "key"
        ],
        max_nesting_depth=_get_config_value(data, "json_parsing", "max_nesting_depth", 10, _parse_int),
        flatten_separator=_get_config_value(data, "json_parsing", "flatten_separator", "."),
        jsonl_batch_size=_get_config_value(data, "json_parsing", "jsonl_batch_size", 1000, _parse_int),
        preserve_fields=json_preserve_fields if json_preserve_fields else [
            "id", "timestamp", "date", "created_at", "updated_at", "type", "category", "level", "status"
        ],
    )

    logging_config = LoggingConfig(
        level=_get_config_value(data, "logging", "level", "INFO"),
        format=_get_config_value(data, "logging", "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        file=_get_config_value(data, "logging", "file", ""),
        json_logging=_get_config_value(data, "logging", "json_logging", False, _parse_bool),
        quiet_third_party=_get_config_value(data, "logging", "quiet_third_party", True, _parse_bool),
        colorize=_get_config_value(data, "logging", "colorize", True, _parse_bool),
    )

    metrics = MetricsConfig(
        enabled=_get_config_value(data, "metrics", "enabled", True, _parse_bool),
        detailed_timing=_get_config_value(data, "metrics", "detailed_timing", True, _parse_bool),
        store_history=_get_config_value(data, "metrics", "store_history", False, _parse_bool),
        history_retention=_get_config_value(data, "metrics", "history_retention", 100, _parse_int),
    )

    performance = PerformanceConfig(
        embedding_cache_enabled=_get_config_value(data, "performance", "embedding_cache_enabled", True, _parse_bool),
        embedding_cache_size=_get_config_value(data, "performance", "embedding_cache_size", 10000, _parse_int),
        query_cache_enabled=_get_config_value(data, "performance", "query_cache_enabled", True, _parse_bool),
        query_cache_size=_get_config_value(data, "performance", "query_cache_size", 1000, _parse_int),
        parallel_retrieval_enabled=_get_config_value(data, "performance", "parallel_retrieval_enabled", True, _parse_bool),
        parallel_postprocessing_enabled=_get_config_value(data, "performance", "parallel_postprocessing_enabled", True, _parse_bool),
        early_stopping_enabled=_get_config_value(data, "performance", "early_stopping_enabled", True, _parse_bool),
        simple_query_max_words=_get_config_value(data, "performance", "simple_query_max_words", 10, _parse_int),
        cache_retrieval_on_retry=_get_config_value(data, "performance", "cache_retrieval_on_retry", True, _parse_bool),
        targeted_retry_enabled=_get_config_value(data, "performance", "targeted_retry_enabled", True, _parse_bool),
    )

    pipeline = PipelineConfig(
        use_planning=_get_config_value(data, "pipeline", "use_planning", True, _parse_bool),
        use_decomposition=_get_config_value(data, "pipeline", "use_decomposition", True, _parse_bool),
        use_rewrite=_get_config_value(data, "pipeline", "use_rewrite", True, _parse_bool),
        use_expansion=_get_config_value(data, "pipeline", "use_expansion", True, _parse_bool),
        use_rrf=_get_config_value(data, "pipeline", "use_rrf", True, _parse_bool),
        use_automerge=_get_config_value(data, "pipeline", "use_automerge", True, _parse_bool),
        use_rerank=_get_config_value(data, "pipeline", "use_rerank", True, _parse_bool),
        use_critic=_get_config_value(data, "pipeline", "use_critic", True, _parse_bool),
    )

    # Web crawler configuration
    web_crawler_include = _get_config_value(data, "web_crawler", "include_patterns", [])
    web_crawler_exclude = _get_config_value(data, "web_crawler", "exclude_patterns", [])
    
    # Ensure patterns are lists
    if isinstance(web_crawler_include, str):
        web_crawler_include = [web_crawler_include] if web_crawler_include else []
    if isinstance(web_crawler_exclude, str):
        web_crawler_exclude = [web_crawler_exclude] if web_crawler_exclude else []

    web_crawler = WebCrawlerConfig(
        max_depth=_get_config_value(data, "web_crawler", "max_depth", 2, _parse_int),
        max_pages=_get_config_value(data, "web_crawler", "max_pages", 100, _parse_int),
        same_domain_only=_get_config_value(data, "web_crawler", "same_domain_only", True, _parse_bool),
        include_patterns=web_crawler_include,
        exclude_patterns=web_crawler_exclude,
        timeout=_get_config_value(data, "web_crawler", "timeout", 30, _parse_int),
        delay=_get_config_value(data, "web_crawler", "delay", 0.5, _parse_float),
        user_agent=_get_config_value(data, "web_crawler", "user_agent", "AgenticRAG-Crawler/1.0"),
        basic_auth_user=_get_config_value(data, "web_crawler", "basic_auth_user", ""),
        basic_auth_password=_get_config_value(data, "web_crawler", "basic_auth_password", ""),
        verify_ssl=_get_config_value(data, "web_crawler", "verify_ssl", True, _parse_bool),
        temp_dir=_get_config_value(data, "web_crawler", "temp_dir", None) or None,
        follow_redirects=_get_config_value(data, "web_crawler", "follow_redirects", True, _parse_bool),
        max_file_size=_get_config_value(data, "web_crawler", "max_file_size", 50_000_000, _parse_int),
        respect_robots_txt=_get_config_value(data, "web_crawler", "respect_robots_txt", True, _parse_bool),
    )

    # Web search configuration (real-time during queries)
    web_search_triggers = _get_config_value(data, "web_search", "trigger_keywords", [])
    web_search_preferred = _get_config_value(data, "web_search", "preferred_domains", [])
    web_search_blocked = _get_config_value(data, "web_search", "blocked_domains", [])
    
    # Ensure lists
    if isinstance(web_search_triggers, str):
        web_search_triggers = [web_search_triggers] if web_search_triggers else []
    if isinstance(web_search_preferred, str):
        web_search_preferred = [web_search_preferred] if web_search_preferred else []
    if isinstance(web_search_blocked, str):
        web_search_blocked = [web_search_blocked] if web_search_blocked else []

    web_search = WebSearchConfig(
        enabled=_get_config_value(data, "web_search", "enabled", False, _parse_bool),
        max_results=_get_config_value(data, "web_search", "max_results", 5, _parse_int),
        max_pages=_get_config_value(data, "web_search", "max_pages", 3, _parse_int),
        timeout=_get_config_value(data, "web_search", "timeout", 15, _parse_int),
        user_agent=_get_config_value(data, "web_search", "user_agent", "AgenticRAG-WebSearch/1.0"),
        include_in_synthesis=_get_config_value(data, "web_search", "include_in_synthesis", True, _parse_bool),
        min_relevance=_get_config_value(data, "web_search", "min_relevance", 0.3, _parse_float),
        search_mode=_get_config_value(data, "web_search", "search_mode", "direct"),
        cache_enabled=_get_config_value(data, "web_search", "cache_enabled", True, _parse_bool),
        cache_ttl=_get_config_value(data, "web_search", "cache_ttl", 3600, _parse_int),
        trigger_keywords=web_search_triggers if web_search_triggers else [
            "latest", "recent", "current", "today", "news",
            "update", "new", "2024", "2025", "now",
        ],
        preferred_domains=web_search_preferred,
        blocked_domains=web_search_blocked if web_search_blocked else [
            "facebook.com", "twitter.com", "instagram.com",
            "tiktok.com", "pinterest.com",
        ],
    )

    return AppConfig(
        ollama=ollama,
        local_models=local_models,
        llm_backend=llm_backend,
        embedding_backend=embedding_backend,
        reranking_backend=reranking_backend,
        storage=storage,
        redis=redis,
        chroma=chroma,
        bm25=bm25,
        ingestion=ingestion,
        retrieval=retrieval,
        rerank=rerank,
        automerge=automerge,
        synthesis=synthesis,
        critic=critic,
        agentic=agentic,
        chunking=chunking,
        summarization=summarization,
        context_evaluation=context_evaluation,
        multihop=multihop,
        fact_verification=fact_verification,
        citation=citation,
        language_detection=language_detection,
        translation=translation,
        query=query,
        conversation=conversation,
        parsing=parsing,
        unstructured_cleaning=unstructured_cleaning,
        json_parsing=json_parsing,
        logging=logging_config,
        metrics=metrics,
        performance=performance,
        pipeline=pipeline,
        vlm=vlm,
        web_crawler=web_crawler,
        web_search=web_search,
    )


class ColorFormatter(logging.Formatter):
    """
    Custom logging formatter with ANSI color codes for different log levels.
    
    Colors:
    - DEBUG: Cyan
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Bold Red with white background
    """
    
    # ANSI escape codes for colors
    COLORS = {
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
        'DIM': '\033[2m',
        
        # Foreground colors
        'BLACK': '\033[30m',
        'RED': '\033[31m',
        'GREEN': '\033[32m',
        'YELLOW': '\033[33m',
        'BLUE': '\033[34m',
        'MAGENTA': '\033[35m',
        'CYAN': '\033[36m',
        'WHITE': '\033[37m',
        
        # Bright foreground colors
        'BRIGHT_RED': '\033[91m',
        'BRIGHT_GREEN': '\033[92m',
        'BRIGHT_YELLOW': '\033[93m',
        'BRIGHT_BLUE': '\033[94m',
        'BRIGHT_MAGENTA': '\033[95m',
        'BRIGHT_CYAN': '\033[96m',
        'BRIGHT_WHITE': '\033[97m',
        
        # Background colors
        'BG_RED': '\033[41m',
        'BG_WHITE': '\033[47m',
    }
    
    # Log level to color mapping
    LEVEL_COLORS = {
        logging.DEBUG: 'CYAN',
        logging.INFO: 'GREEN',
        logging.WARNING: 'YELLOW',
        logging.ERROR: 'RED',
        logging.CRITICAL: 'BRIGHT_RED',
    }
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, 
                 style: str = '%', use_colors: bool = True):
        super().__init__(fmt, datefmt, style)
        self.use_colors = use_colors
        self._detect_color_support()
    
    def _detect_color_support(self) -> None:
        """Detect if the terminal supports colors."""
        import sys
        
        # Check if output is a TTY
        if not hasattr(sys.stderr, 'isatty') or not sys.stderr.isatty():
            self.use_colors = False
            return
        
        # Check for NO_COLOR environment variable (standard)
        if os.environ.get('NO_COLOR'):
            self.use_colors = False
            return
        
        # Check TERM environment variable
        term = os.environ.get('TERM', '')
        if term == 'dumb':
            self.use_colors = False
            return
    
    def _colorize(self, text: str, color_name: str) -> str:
        """Apply color to text."""
        if not self.use_colors:
            return text
        color = self.COLORS.get(color_name, '')
        reset = self.COLORS['RESET']
        return f"{color}{text}{reset}"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Save original values
        original_levelname = record.levelname
        original_name = record.name
        original_msg = record.msg
        
        if self.use_colors:
            # Get color for this level
            color_name = self.LEVEL_COLORS.get(record.levelno, 'WHITE')
            
            # Colorize level name with appropriate color
            level_color = self.COLORS.get(color_name, '')
            reset = self.COLORS['RESET']
            bold = self.COLORS['BOLD']
            dim = self.COLORS['DIM']
            
            # Apply colors
            record.levelname = f"{bold}{level_color}{record.levelname}{reset}"
            record.name = f"{dim}{self.COLORS['BLUE']}{record.name}{reset}"
            
            # Colorize message based on level
            if record.levelno >= logging.ERROR:
                record.msg = f"{level_color}{record.msg}{reset}"
            elif record.levelno == logging.WARNING:
                record.msg = f"{level_color}{record.msg}{reset}"
        
        # Format the record
        result = super().format(record)
        
        # Restore original values
        record.levelname = original_levelname
        record.name = original_name
        record.msg = original_msg
        
        return result


def setup_logging(config: LoggingConfig) -> None:
    """Configure application logging based on config."""
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    if config.colorize:
        console_handler.setFormatter(ColorFormatter(config.format, use_colors=True))
    else:
        console_handler.setFormatter(logging.Formatter(config.format))
    handlers.append(console_handler)

    # File handler if configured (never use colors for files)
    if config.file:
        file_path = Path(config.file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(config.file)
        file_handler.setFormatter(logging.Formatter(config.format))
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )

    # Quiet noisy third-party loggers if enabled
    if config.quiet_third_party:
        # These libraries produce verbose or misleading log messages
        noisy_loggers = {
            # HuggingFace - suppress retry warnings (they're expected with slow connections)
            "huggingface_hub": logging.ERROR,
            "huggingface_hub.utils._http": logging.ERROR,

            # Transformers - reduce verbosity
            "transformers": logging.WARNING,
            "transformers.modeling_utils": logging.WARNING,

            # Accelerate - suppress sharding and memory allocation warnings
            "accelerate": logging.ERROR,
            "accelerate.utils.modeling": logging.ERROR,

            # Safetensors - suppress "not sharded" warnings for single-file models
            "safetensors": logging.ERROR,

            # Unstructured - suppress misleading "text extraction failed" (it's expected for OCR)
            "unstructured": logging.WARNING,

            # pikepdf - suppress initialization messages
            "pikepdf": logging.WARNING,
            "pikepdf._core": logging.WARNING,

            # Other common noisy libraries
            "urllib3": logging.WARNING,
            "httpx": logging.WARNING,
            "httpcore": logging.WARNING,
            "filelock": logging.WARNING,
            "PIL": logging.WARNING,
            "torch": logging.WARNING,
            "sentence_transformers": logging.WARNING,
        }

        for logger_name, level in noisy_loggers.items():
            logging.getLogger(logger_name).setLevel(level)


def config_to_dict(config: AppConfig) -> Dict[str, Any]:
    """Convert AppConfig to dictionary (for serialization/logging)."""
    from dataclasses import asdict

    result = asdict(config)
    # Redact sensitive values
    if "ollama" in result and "openai_api_key" in result["ollama"]:
        result["ollama"]["openai_api_key"] = "***REDACTED***"
    return result
