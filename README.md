# Radiant Agentic RAG

A production-quality Agentic Retrieval-Augmented Generation (RAG) system with multi-agent architecture, hybrid search, and professional reporting.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Storage Backends](#storage-backends)
- [Binary Quantization](#binary-quantization)
- [Agent Pipeline](#agent-pipeline)
- [Ingestion Pipeline](#ingestion-pipeline)
- [Query Pipeline](#query-pipeline)
- [GitHub Repository Ingestion](#github-repository-ingestion)
- [Code-Aware Chunking](#code-aware-chunking)
- [Multilingual Support](#multilingual-support)
- [Performance Optimizations](#performance-optimizations)
- [Metrics & Monitoring](#metrics--monitoring)
- [Advanced Features](#advanced-features)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

Radiant RAG is an enterprise-grade retrieval-augmented generation system that combines:

- **Multi-agent orchestration** for intelligent query processing
- **Hybrid search** combining dense embeddings and BM25 sparse retrieval
- **Performance optimized** with 60-93% latency reduction through intelligent caching, parallel execution, and batching
- **Multiple storage backends** - Redis, ChromaDB, and PostgreSQL with pgvector
- **Binary quantization** for 10-20x faster retrieval with 3.5x memory reduction
- **GitHub repository ingestion** with code-aware chunking
- **Multilingual support** with automatic language detection and translation
- **Professional reporting** in multiple formats
- **Metrics export** with Prometheus and OpenTelemetry support

### Key Features

| Category | Features |
|----------|----------|
| **Retrieval** | Dense (HNSW), BM25, Hybrid (RRF fusion), Web Search |
| **Performance** | 60-93% faster with intelligent caching, parallel execution, batching, early stopping |
| **Storage** | Redis (default), ChromaDB, PostgreSQL with pgvector |
| **Quantization** | Binary and Int8 quantization for faster retrieval |
| **Agents** | 20+ specialized agents for planning, retrieval, post-processing |
| **Ingestion** | Files, URLs, GitHub repos, with code-aware chunking |
| **Languages** | 176 languages detected, LLM-based translation |
| **Output** | Markdown, HTML, JSON, Text reports |
| **Interfaces** | CLI, TUI (Textual), Python API |
| **Monitoring** | Prometheus metrics, OpenTelemetry tracing |

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RADIANT RAG SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   CLI/TUI   │    │  Python API │    │   Config    │    │   Reports   │   │
│  │  Interface  │    │   Access    │    │   (YAML)    │    │  Generator  │   │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘   │
│         │                  │                  │                  │          │
│         └──────────────────┼──────────────────┼──────────────────┘          │
│                            ▼                  ▼                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        RADIANT RAG APPLICATION                      │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │                      AGENTIC ORCHESTRATOR                     │  │    │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │  │    │
│  │  │  │Planning │→│  Query  │→│Retrieval│→│  Post-  │→│Generate │  │  │    │
│  │  │  │  Stage  │ │  Proc.  │ │  Stage  │ │Retrieval│ │  Stage  │  │  │    │
│  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘  │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                            │                  │                             │
│         ┌──────────────────┼──────────────────┼──────────────────┐          │
│         ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │    LLM      │    │   Vector    │    │    BM25     │    │   Local     │   │
│  │   Client    │    │   Store     │    │    Index    │    │   Models    │   │
│  │  (Ollama)   │    │(Redis/Chroma│    │ (Persistent)│    │(Embeddings) │   │
│  │             │    │  /PgVector) │    │             │    │             │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Agent Hierarchy

```
BaseAgent (Abstract)
├── LLMAgent (requires LLM client)
│   ├── PlanningAgent
│   ├── AnswerSynthesisAgent
│   ├── CriticAgent
│   ├── QueryDecompositionAgent
│   ├── QueryRewriteAgent
│   ├── QueryExpansionAgent
│   ├── WebSearchAgent
│   ├── SummarizationAgent
│   ├── ContextEvaluationAgent
│   ├── FactVerificationAgent
│   ├── CitationTrackingAgent
│   ├── LanguageDetectionAgent
│   ├── TranslationAgent
│   └── IntelligentChunkingAgent
│
├── RetrievalAgent (requires vector store)
│   └── DenseRetrievalAgent
│
└── BaseAgent (direct inheritance)
    ├── BM25RetrievalAgent
    ├── RRFAgent
    ├── HierarchicalAutoMergingAgent
    ├── CrossEncoderRerankingAgent
    └── MultiHopReasoningAgent
```

### Data Flow Diagram

```
                              USER QUERY
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            QUERY PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐                                                            │
│  │  PLANNING   │  Analyze query complexity, select retrieval strategy       │
│  │    AGENT    │  Outputs: mode (hybrid/dense/bm25), decompose?, expand?    │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐                                                            │
│  │   QUERY     │  Decompose complex queries into sub-queries                │
│  │DECOMPOSITION│  Example: "Compare X and Y" → ["What is X?", "What is Y?"] │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐  ┌─────────────┐                                           │
│  │   QUERY     │  │   QUERY     │  Rewrite for clarity, expand with         │
│  │  REWRITE    │→ │  EXPANSION  │  synonyms and related terms               │
│  └──────┬──────┘  └──────┬──────┘                                           │
│         │                │                                                  │
│         └───────┬────────┘                                                  │
│                 ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         RETRIEVAL STAGE                              │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐               │   │
│  │  │   DENSE     │    │    BM25     │    │ WEB SEARCH  │               │   │
│  │  │ RETRIEVAL   │    │ RETRIEVAL   │    │   (opt.)    │               │   │
│  │  │ (Embeddings)│    │  (Keywords) │    │             │               │   │
│  │  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘               │   │
│  │         │                  │                  │                      │   │
│  │         └──────────────────┼──────────────────┘                      │   │
│  │                            ▼                                         │   │
│  │                     ┌─────────────┐                                  │   │
│  │                     │  RRF FUSION │  Reciprocal Rank Fusion          │   │
│  │                     └──────┬──────┘                                  │   │
│  └─────────────────────────────┼────────────────────────────────────────┘   │
│                                ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      POST-RETRIEVAL STAGE                            │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐               │   │
│  │  │ AUTO-MERGE  │ →  │  RERANKING  │ →  │  CONTEXT    │               │   │
│  │  │Hierarchical │    │ CrossEncoder│    │ EVALUATION  │               │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘               │   │
│  │         │                  │                  │                      │   │
│  │         ▼                  ▼                  ▼                      │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐               │   │
│  │  │SUMMARIZATION│    │  MULTI-HOP  │    │    FACT     │               │   │
│  │  │   (Long)    │    │  REASONING  │    │VERIFICATION │               │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                │                                            │
│                                ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        GENERATION STAGE                              │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐               │   │
│  │  │   ANSWER    │ →  │  CITATION   │ →  │   CRITIC    │               │   │
│  │  │  SYNTHESIS  │    │  TRACKING   │    │ EVALUATION  │               │   │
│  │  └─────────────┘    └─────────────┘    └─────────────┘               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                │                                            │
└────────────────────────────────┼────────────────────────────────────────────┘
                                 ▼
                          FINAL RESPONSE
                    (Answer + Citations + Score)
```

---

## Installation

### Prerequisites

- Python 3.10+
- Redis Stack (Redis + RediSearch module) - default backend
- CUDA-capable GPU (optional, for faster inference)

### Step 1: Install Redis Stack

```bash
# Using Docker (recommended)
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack-server

# Or install locally (Ubuntu)
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt-get update
sudo apt-get install redis-stack-server
```

### Step 2: Install Radiant RAG

```bash
# Clone or extract the package
cd radiant-rag

# Install as package (recommended)
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Step 3: Configure Environment

```bash
# Required: LLM endpoint (Ollama or compatible)
export RADIANT_OLLAMA_OPENAI_BASE_URL="https://your-ollama-host/v1"
export RADIANT_OLLAMA_OPENAI_API_KEY="your-api-key"

# Optional: GitHub token for higher rate limits
export GITHUB_TOKEN="ghp_your_token_here"

# Optional: Redis connection (defaults to localhost:6379)
export RADIANT_REDIS_HOST="localhost"
export RADIANT_REDIS_PORT="6379"
```

---

## Quick Start

```bash
# 1. Ingest local documents
python -m radiant ingest ./documents/

# 2. Ingest from GitHub repository
python -m radiant ingest --url "https://github.com/owner/repo"

# 3. Query the system
python -m radiant query "What is the main topic of these documents?"

# 4. Interactive mode
python -m radiant interactive

# 5. Interactive TUI mode
python -m radiant interactive --tui
```

---

## CLI Reference

### Command Overview

```
python -m radiant <command> [options]

Commands:
  ingest       Ingest documents from files, directories, or URLs
  query        Query the RAG system with full pipeline
  search       Search documents (retrieval only, no LLM)
  interactive  Start interactive query mode
  stats        Display system statistics
  health       Check system health
  clear        Clear all indexed documents
  rebuild-bm25 Rebuild BM25 index from store
```

### ingest

Ingest documents from files, directories, or URLs.

```bash
python -m radiant ingest [paths...] [options]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--url URL` | `-u` | - | URL to ingest (repeatable) |
| `--flat` | - | false | Use flat storage (no hierarchy) |
| `--crawl-depth N` | - | config | Crawl depth for URLs |
| `--max-pages N` | - | config | Maximum pages to crawl |
| `--no-crawl` | - | false | Don't crawl, fetch single URL |
| `--auth USER:PASS` | - | - | Basic auth for URL ingestion |
| `--config PATH` | `-c` | config.yaml | Config file path |

**Examples:**

```bash
# Ingest local directory
python -m radiant ingest ./docs/

# Ingest GitHub repository (auto-detected)
python -m radiant ingest --url "https://github.com/owner/repo"

# Ingest website with crawling
python -m radiant ingest --url "https://docs.example.com" --crawl-depth 3

# Ingest multiple sources
python -m radiant ingest ./local/ --url "https://github.com/org/repo1" --url "https://github.com/org/repo2"

# Ingest with authentication
python -m radiant ingest --url "https://private.example.com" --auth "user:password"
```

### query

Query the RAG system with full agentic pipeline.

```bash
python -m radiant query "<question>" [options]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--mode MODE` | `-m` | hybrid | Retrieval mode: hybrid, dense, bm25 |
| `--conversation ID` | `-conv` | - | Continue conversation by ID |
| `--save PATH` | `-s` | - | Save report (.md, .html, .json, .txt) |
| `--compact` | - | false | Compact display format |
| `--simple` | - | false | Skip advanced agents (faster) |

**Examples:**

```bash
# Basic query
python -m radiant query "What is RAG?"

# Semantic search only
python -m radiant query "meaning of retrieval augmentation" --mode dense

# Keyword search only  
python -m radiant query "BM25 algorithm" --mode bm25

# Save report
python -m radiant query "Summarize the architecture" --save report.md

# Continue conversation
python -m radiant query "Tell me more about that" --conv abc123
```

### search

Search documents without LLM generation (retrieval only).

```bash
python -m radiant search "<query>" [options]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--mode MODE` | `-m` | hybrid | Retrieval mode |
| `--top-k N` | `-k` | 10 | Number of results |
| `--save PATH` | `-s` | - | Save results to file |

### clear

Clear all indexed documents.

```bash
python -m radiant clear [options]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--confirm` | false | Skip confirmation prompt |
| `--keep-bm25` | false | Keep BM25 index |

---

## Configuration

### Configuration File (config.yaml)

The system is configured via `config.yaml`. All settings can be overridden with environment variables prefixed with `RADIANT_`.

### Core Settings

```yaml
# LLM Configuration (Ollama OpenAI-compatible)
ollama:
  openai_base_url: "https://your-ollama-host/v1"
  openai_api_key: "your-api-key"
  chat_model: "qwen2.5:latest"
  timeout: 90
  max_retries: 3

# Local Models (HuggingFace / sentence-transformers)
local_models:
  embed_model_name: "sentence-transformers/all-MiniLM-L12-v2"
  cross_encoder_name: "cross-encoder/ms-marco-MiniLM-L12-v2"
  device: "auto"
  embedding_dimension: 384
```

### Environment Variable Overrides

All configuration values can be overridden with environment variables:

```bash
# Pattern: RADIANT_<SECTION>_<KEY>
export RADIANT_OLLAMA_CHAT_MODEL="llama3:70b"
export RADIANT_REDIS_URL="redis://redis-server:6379/0"
export RADIANT_RETRIEVAL_DENSE_TOP_K="20"
```

---

## Storage Backends

Radiant RAG supports multiple vector storage backends. Choose the one that best fits your deployment needs:

| Backend | Use Case | Pros | Cons |
|---------|----------|------|------|
| **Redis** (default) | Production, low-latency | Fast, feature-rich, real-time | Requires Redis Stack |
| **Chroma** | Development, testing | Easy setup, embedded | Less scalable |
| **PgVector** | Enterprise, PostgreSQL shops | Mature, ACID, integrates with existing DB | More setup required |

### Redis Configuration (Default)

```yaml
storage:
  backend: redis

redis:
  url: "redis://localhost:6379/0"
  key_prefix: "radiant"
  vector_index:
    name: "radiant_vectors"
    hnsw_m: 16
    hnsw_ef_construction: 200
    hnsw_ef_runtime: 100
    distance_metric: "COSINE"
```

### Chroma Configuration

```yaml
storage:
  backend: chroma

chroma:
  persist_directory: "./data/chroma_db"
  collection_name: "radiant_docs"
  distance_fn: "cosine"
  embedding_dimension: 384
```

To use Chroma, install the optional dependency:
```bash
pip install chromadb
```

### PgVector Configuration

```yaml
storage:
  backend: pgvector

pgvector:
  # Use PG_CONN_STR env var or set here
  connection_string: "postgresql://user:pass@localhost:5432/radiant"
  leaf_table_name: "haystack_leaves"
  parent_table_name: "haystack_parents"
  vector_function: "cosine_similarity"
  search_strategy: "hnsw"
```

To use PgVector, install PostgreSQL with the pgvector extension and the Python driver:
```bash
pip install psycopg2-binary
```

---

## Binary Quantization

Binary quantization provides significant performance improvements for large-scale deployments:

- ⚡ **10-20x faster** retrieval
- 💾 **3.5x less** memory usage  
- 🎯 **95-96%** accuracy retention
- 🔧 **Zero breaking changes** - disabled by default

### Quick Setup

```bash
# Step 1: Install dependencies
pip install sentence-transformers>=3.2.0 numpy>=1.26.0

# Step 2: Calibrate (only for int8/both precision)
python tools/calibrate_int8_ranges.py \
    --sample-size 100000 \
    --output data/int8_ranges.npy

# Step 3: Enable in config.yaml
```

### Configuration

```yaml
redis:  # or chroma, or pgvector
  quantization:
    enabled: true
    precision: "both"  # Options: "binary", "int8", "both"
    rescore_multiplier: 4.0
    use_rescoring: true
    int8_ranges_file: "data/int8_ranges.npy"
```

### Performance Comparison (1M Documents, 384-dim)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Memory | 1,536 MB | 432 MB | **3.5x less** |
| Retrieval Speed | 50-100ms | 5-10ms | **10-20x faster** |
| Accuracy | 100% | 95-96% | **-4%** |

For detailed documentation, see `BINARY_QUANTIZATION_README.md`.

---

## Agent Pipeline

### Agent Categories

| Category | Agents | Purpose |
|----------|--------|---------|
| Planning | PlanningAgent | Analyze query, select retrieval strategy |
| Query Processing | Decomposition, Rewrite, Expansion | Optimize queries |
| Retrieval | Dense, BM25, Web Search | Fetch documents |
| Fusion | RRFAgent | Combine retrieval results |
| Post-Retrieval | AutoMerge, Rerank, ContextEval, Summarization, MultiHop | Refine context |
| Generation | Synthesis, Critic | Generate and evaluate answers |
| Verification | FactVerification, Citation | Ensure accuracy |
| Multilingual | LanguageDetection, Translation | Cross-language support |
| Tools | Calculator, CodeExecution | Extended capabilities |

### AgentResult Pattern

All agents return results wrapped in `AgentResult`:

```python
from radiant.agents import AgentResult, AgentStatus

result = agent.run(query="test query")

if result.success:
    data = result.data
    print(f"Duration: {result.metrics.duration_ms}ms")
    print(f"Status: {result.status}")  # SUCCESS, PARTIAL, FAILED, SKIPPED
else:
    print(f"Error: {result.error}")
```

---

## Ingestion Pipeline

### Supported Formats

| Format | Extensions | Notes |
|--------|------------|-------|
| PDF | .pdf | Text extraction with fallback OCR |
| Word | .docx, .doc | Full support via unstructured |
| Text | .txt | UTF-8 encoding |
| Markdown | .md | Preserves structure |
| HTML | .html | Strips tags, extracts text |
| Images | .png, .jpg | VLM captioning with Qwen2-VL |
| Code | .py, .js, .ts, etc. | Code-aware chunking |

### Hierarchical Storage

Documents are stored with parent/child relationships:

```
Parent Document (full text)
├── Child Chunk 1 (embedded, searchable)
├── Child Chunk 2 (embedded, searchable)
└── Child Chunk 3 (embedded, searchable)
```

---

## Query Pipeline

### Retrieval Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `hybrid` | Dense + BM25 with RRF fusion | General queries |
| `dense` | Semantic similarity only | Conceptual queries |
| `bm25` | Keyword matching only | Technical terms, exact phrases |

### Configuration

```yaml
retrieval:
  dense_top_k: 10
  bm25_top_k: 10
  fused_top_k: 15
  rrf_k: 60
  min_similarity: 0.0
  search_scope: "leaves"  # "leaves", "parents", or "all"
```

---

## GitHub Repository Ingestion

GitHub URLs are automatically detected and handled with specialized crawling:

```bash
python -m radiant ingest --url "https://github.com/owner/repo"
```

### Features

- Raw markdown extraction (not HTML)
- Follow links in README to find all documentation
- Code-aware chunking for source files
- Metadata preservation (path, URL, repo name)

### Configuration

```yaml
github_crawler:
  max_files: 200
  delay: 0.5
  include_extensions:
    - ".md"
    - ".py"
    - ".js"
```

---

## Code-Aware Chunking

Source code files are chunked intelligently:

- Preserves function/class boundaries
- Includes import context
- Maintains semantic coherence
- Extracts metadata (language, block type, line numbers)

### Supported Languages

Python, JavaScript, TypeScript, Java, Go, Rust, C/C++, Ruby, PHP, SQL, and more.

---

## Multilingual Support

### Language Detection

```yaml
language_detection:
  enabled: true
  method: "fast"  # "fast" or "llm"
  min_confidence: 0.7
  use_llm_fallback: true
  fallback_language: "en"
```

### Translation

```yaml
translation:
  enabled: true
  method: "llm"
  canonical_language: "en"
  translate_at_ingestion: true
  preserve_original: true
```

---

## Performance Optimizations

Radiant RAG has been extensively optimized for production performance with **60-93% latency reduction** across different query types.

### Performance Results

| Query Type | Improvement | Benefit |
|------------|-------------|---------|
| Simple queries | 39-44% faster | Early stopping, batching |
| Complex queries | 49-54% faster | Parallel execution, batched LLM calls |
| Retry scenarios | 70% faster | Targeted retries, cached retrieval |
| Repeated queries | 93% faster | Intelligent caching (cache hits) |
| Document ingestion | 5-10× faster | Always batched |

### Key Optimizations

#### 1. Intelligent Caching (Phase 3)

**Embedding Cache**: Content-based deduplication using SHA-256 hashing
- **Hit rate**: 30-50% expected
- **Memory**: ~15MB for 10K cache (configurable)
- **LRU eviction**: True LRU using OrderedDict with move_to_end()

**Query Cache**: Full query result caching
- **Hit rate**: 20-40% expected
- **Memory**: ~5MB for 1K cache (configurable)
- **LRU eviction**: True LRU for optimal performance

```python
# Access cache statistics
from radiant.utils.cache import get_all_cache_stats

stats = get_all_cache_stats()
print(f"Embedding cache hit rate: {stats['embedding']['hit_rate']:.1%}")
print(f"Query cache hit rate: {stats['query']['hit_rate']:.1%}")
```

#### 2. Parallel Execution (Phase 2)

- **Hybrid retrieval**: Dense and BM25 run concurrently using ThreadPoolExecutor
- **Post-processing**: Fact verification and citation generation in parallel
- **Impact**: ~50% faster for hybrid retrieval mode

#### 3. Batched Operations (Phase 1 & 2)

- **Embedding generation**: Always batched (removed legacy single-item code)
- **LLM calls**: Multiple queries processed in single API call
- **Impact**: 66-75% reduction in API overhead

#### 4. Early Stopping (Phase 1)

- **Simple query detection**: Heuristic-based (≤10 words, no complex terms)
- **Skip unnecessary steps**: Decomposition, expansion, fact verification for simple queries
- **Impact**: 30-40% faster for simple queries

#### 5. Targeted Retries (Phase 1)

- **Cache retrieval results**: Reuse on retry instead of re-fetching
- **Only regenerate**: LLM generation step, skip retrieval/processing
- **Impact**: 70-90% less redundant work on retries

### Configuration

All performance optimizations are **enabled by default** and can be tuned:

```yaml
performance:
  # Embedding cache settings
  embedding_cache_enabled: true
  embedding_cache_size: 10000  # ~15MB, adjust based on RAM

  # Query cache settings
  query_cache_enabled: true
  query_cache_size: 1000  # ~5MB

  # Parallel execution settings
  parallel_retrieval_enabled: true
  parallel_postprocessing_enabled: true

  # Early stopping settings
  early_stopping_enabled: true
  simple_query_max_words: 10

  # Retry optimization settings
  cache_retrieval_on_retry: true
  targeted_retry_enabled: true
```

### Cache Tuning Guidelines

| Deployment Size | embedding_cache_size | Memory | Hit Rate |
|-----------------|---------------------|--------|----------|
| Small | 5,000 | ~7.5MB | 25-35% |
| Medium (recommended) | 10,000 | ~15MB | 30-50% |
| Large | 20,000 | ~30MB | 40-60% |

**Note**: Diminishing returns beyond 20K cache size.

### Performance Monitoring

```python
# Monitor cache effectiveness
from radiant.utils.cache import get_all_cache_stats

stats = get_all_cache_stats()
if stats['embedding']['hit_rate'] < 0.20:
    print("Consider increasing cache size")
```

For complete performance documentation, see:
- `PERFORMANCE_ANALYSIS.md` - Initial analysis
- `PERFORMANCE_IMPROVEMENTS_IMPLEMENTED.md` - Implementation details
- `POST_OPTIMIZATION_ANALYSIS.md` - Verification and results

---

## Metrics & Monitoring

### Prometheus Integration

```python
from radiant.utils.metrics_export import PrometheusMetricsExporter

exporter = PrometheusMetricsExporter(namespace="radiant_rag")
exporter.register_agent(planning_agent)

# After each agent run
result = agent.run(query="test")
exporter.record_execution(result)

# Get metrics for /metrics endpoint
metrics_output = exporter.get_metrics_output()
```

### OpenTelemetry Integration

```python
from radiant.utils.metrics_export import OpenTelemetryExporter

exporter = OpenTelemetryExporter(
    service_name="radiant-rag",
    endpoint="http://localhost:4317",
)

# Trace agent execution
with exporter.trace_agent(agent, query="test"):
    result = agent.run(query="test")
    exporter.record_result(result)
```

### Unified Collector

```python
from radiant.utils.metrics_export import MetricsCollector

collector = MetricsCollector.create(
    prometheus_enabled=True,
    otel_enabled=True,
    otel_endpoint="http://localhost:4317",
)

result = agent.run(query="test")
collector.record(result)
```

---

## Advanced Features

### Strategy Memory

Learns from successful retrieval patterns:

```yaml
agentic:
  strategy_memory_enabled: true
  strategy_memory_path: "./data/strategy_memory.json.gz"
```

### Citation Tracking

```yaml
citation:
  enabled: true
  citation_style: "inline"  # inline, footnote, academic, enterprise
  generate_bibliography: true
  generate_audit_trail: true
```

### Web Search Augmentation

```yaml
web_search:
  enabled: false
  provider: "duckduckgo"
  max_results: 5
```

---

## API Reference

### Python API

```python
from radiant.app import RadiantRAG, create_app

# Create application
app = create_app("config.yaml")  # Or RadiantRAG()

# Ingest documents
app.ingest_documents(["./docs/"], use_hierarchical=True)

# Ingest URLs (auto-detects GitHub)
app.ingest_urls(["https://github.com/owner/repo"])

# Query with full pipeline
result = app.query("What is RAG?", mode="hybrid")
print(result.answer)
print(result.confidence)

# Search only (no LLM generation)
results = app.search("BM25 algorithm", mode="hybrid", top_k=10)

# Simple query (minimal pipeline)
answer = app.simple_query("What is RAG?", top_k=5)

# Conversation support
conversation_id = app.start_conversation()
result1 = app.query("What is RAG?", conversation_id=conversation_id)
result2 = app.query("Tell me more", conversation_id=conversation_id)

# System management
app.clear_index()
health = app.check_health()
stats = app.get_stats()
```

### PipelineResult Object

```python
@dataclass
class PipelineResult:
    answer: str                    # Generated answer
    context: AgentContext          # Pipeline context
    metrics: RunMetrics            # Performance metrics
    success: bool                  # Execution status
    confidence: float              # Critic score (0-1)
    retrieval_mode_used: str       # Actual mode used
    retry_count: int               # Number of retries
    tools_used: List[str]          # Tools invoked
    
    # Multi-hop reasoning
    multihop_used: bool
    multihop_hops: int
    
    # Fact verification
    fact_verification_score: float
    fact_verification_passed: bool
    
    # Citations
    cited_answer: Optional[str]
    citations: List[Dict]
    sources: List[Dict]
    audit_id: Optional[str]
```

---

## Troubleshooting

### Common Issues

**Redis connection failed**
```bash
# Check Redis is running
docker ps | grep redis-stack

# Start Redis if needed
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack-server
```

**No documents found after ingestion**
```bash
# Check index status
python -m radiant stats

# Clear and re-ingest
python -m radiant clear --confirm
python -m radiant ingest ./docs/
```

**GitHub rate limit exceeded**
```bash
# Set GitHub token
export GITHUB_TOKEN="ghp_your_token"
```

**Slow ingestion**
```yaml
# Increase batch sizes in config.yaml
ingestion:
  embedding_batch_size: 64  # Increase if GPU has memory
  redis_batch_size: 200
```

### Diagnostic Tools

```bash
# Check Redis connectivity and stats
python tools/check_redis.py

# Inspect index contents
python tools/inspect_index.py

# Validate quantization implementation
python tools/validate_quantization.py

# Calibrate int8 quantization ranges
python tools/calibrate_int8_ranges.py --sample-size 100000 --output data/int8_ranges.npy

# View system health
python -m radiant health

# View statistics
python -m radiant stats
```

---

## File Structure

```
radiant-rag/
├── config.yaml                 # Configuration file
├── config_quantization_example.yaml  # Quantization config example
├── README.md                   # This file
├── BINARY_QUANTIZATION_README.md  # Quantization documentation
├── PERFORMANCE_ANALYSIS.md     # Performance analysis
├── PERFORMANCE_IMPROVEMENTS_IMPLEMENTED.md  # Implementation details
├── POST_OPTIMIZATION_ANALYSIS.md  # Optimization verification
├── CHANGES_SUMMARY.md          # Code changes summary
├── AGENTS.md                   # Agent development guide
├── NEW_CORPUS_GUIDE.md         # Adding new document sources
├── requirements.txt            # Python dependencies
├── requirements-dev.txt        # Development dependencies
├── pyproject.toml              # Package configuration
│
├── radiant/                    # Main package
│   ├── app.py                  # RadiantRAG application
│   ├── orchestrator.py         # Agent pipeline orchestration
│   ├── config.py               # Configuration loading
│   │
│   ├── agents/                 # Pipeline agents
│   │   ├── base_agent.py       # BaseAgent ABC with metrics
│   │   ├── agent_template.py   # Template for new agents
│   │   ├── registry.py         # Agent registration
│   │   ├── planning.py         # Query planning
│   │   ├── decomposition.py    # Query decomposition
│   │   ├── rewrite.py          # Query rewriting
│   │   ├── expansion.py        # Query expansion
│   │   ├── dense.py            # Dense retrieval
│   │   ├── bm25.py             # BM25 retrieval
│   │   ├── fusion.py           # RRF fusion
│   │   ├── automerge.py        # Hierarchical merging
│   │   ├── rerank.py           # Cross-encoder reranking
│   │   ├── synthesis.py        # Answer generation
│   │   ├── citation.py         # Citation tracking
│   │   ├── critic.py           # Answer evaluation
│   │   ├── context_eval.py     # Context evaluation
│   │   ├── summarization.py    # Context summarization
│   │   ├── multihop.py         # Multi-hop reasoning
│   │   ├── fact_verification.py # Fact checking
│   │   ├── language_detection.py # Language detection
│   │   ├── translation.py      # Translation
│   │   ├── chunking.py         # Intelligent chunking
│   │   ├── strategy_memory.py  # Strategy learning
│   │   ├── web_search.py       # Web search
│   │   ├── tools.py            # Calculator, code execution
│   │   └── AGENTS.md           # Agent development guide
│   │
│   ├── ingestion/              # Document processing
│   │   ├── processor.py        # Document processor
│   │   ├── github_crawler.py   # GitHub repository crawler
│   │   ├── web_crawler.py      # Web page crawler
│   │   ├── code_chunker.py     # Code-aware chunking
│   │   └── image_captioner.py  # VLM image captioning
│   │
│   ├── storage/                # Storage backends
│   │   ├── base.py             # BaseVectorStore ABC
│   │   ├── factory.py          # Storage backend factory
│   │   ├── redis_store.py      # Redis vector store
│   │   ├── chroma_store.py     # ChromaDB vector store
│   │   ├── pgvector_store.py   # PostgreSQL pgvector store
│   │   ├── bm25_index.py       # Persistent BM25 index
│   │   └── quantization.py     # Binary quantization utilities
│   │
│   ├── llm/                    # LLM clients
│   │   ├── client.py           # LLM API client
│   │   └── local_models.py     # Local embedding/reranking
│   │
│   ├── utils/                  # Utilities
│   │   ├── cache.py            # Intelligent LRU caching
│   │   ├── metrics.py          # Metrics collection
│   │   ├── metrics_export.py   # Prometheus/OTel export
│   │   └── conversation.py     # Conversation management
│   │
│   └── ui/                     # User interfaces
│       ├── display.py          # Console output
│       ├── tui.py              # Textual TUI
│       └── reports/            # Report generation
│
├── tools/                      # Diagnostic tools
│   ├── check_redis.py          # Redis connectivity check
│   ├── inspect_index.py        # Index inspection
│   ├── validate_quantization.py # Quantization validation
│   ├── validate_bugfix.py      # Bugfix validation
│   └── calibrate_int8_ranges.py # Int8 calibration
│
├── docs/                       # Documentation
│   ├── USER_MANUAL.md          # Full user manual
│   ├── AGENT_ARCHITECTURE.md   # Agent architecture docs
│   ├── AGENTS_MD_USAGE.md      # AGENTS.md usage guide
│   └── CHANGES_ORCHESTRATOR_UPDATE.md
│
└── tests/                      # Test suite
    ├── test_all.py             # Comprehensive tests
    ├── test_base_agent_lifecycle.py  # Agent lifecycle tests
    ├── test_agents/            # Agent-specific tests
    ├── test_storage/           # Storage tests
    ├── test_ingestion/         # Ingestion tests
    └── test_ui/                # UI tests
```

---

## License

MIT License

## Contributing

Contributions welcome! Please read the contributing guidelines first.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-12 | Initial release |
| 1.1.0 | 2024-12 | Added GitHub crawler, code-aware chunking |
| 1.2.0 | 2024-12 | Added multilingual support, fact verification |
| 1.3.0 | 2025-01 | Added binary quantization, multiple storage backends |
| 1.4.0 | 2025-01 | Added BaseAgent ABC, AgentResult pattern, metrics export |
| 1.5.0 | 2025-01 | Major performance optimization: 60-93% latency reduction with intelligent caching, parallel execution, batching, early stopping, and targeted retries |
