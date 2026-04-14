# radiant-rag-mcp

**Radiant Agentic RAG exposed as a Model Context Protocol (MCP) server.**

`radiant-rag-mcp` wraps the full Radiant RAG pipeline as ten MCP tools.
Use **stdio** transport for Claude Code and Claude Desktop, or **HTTP** transport
for remote server and notebook deployments.

For MCP tool reference and Claude Code integration, see [MCP_README.md](MCP_README.md).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Storage Backends](#storage-backends)
- [Agent Pipeline](#agent-pipeline)
- [Ingestion Pipeline](#ingestion-pipeline)
- [Query Pipeline](#query-pipeline)
- [Performance](#performance)
- [Metrics & Monitoring](#metrics--monitoring)
- [API Reference](#api-reference)

---

## Overview

Radiant RAG is an enterprise-grade retrieval-augmented generation system exposed as an
MCP server. It combines:

- **Multi-agent orchestration** — 20+ specialized agents for query processing, retrieval,
  post-retrieval refinement, and answer generation
- **Hybrid search** — dense embeddings (sentence-transformers) + BM25 sparse retrieval
  with RRF fusion
- **Multiple storage backends** — Redis Stack (production) and ChromaDB (development)
- **OpenAI-compatible LLM backend** — works with Ollama, vLLM, OpenAI, and any
  compatible endpoint
- **Local models** — sentence-transformers for embeddings, cross-encoder for reranking;
  no external embedding API required
- **Hierarchical storage** — parent/child chunk relationships for auto-merging retrieval
- **Conversation support** — multi-turn session management with in-memory or
  backend-persisted history

### Key Features

| Category | Details |
|---|---|
| **Retrieval** | Dense (HNSW), BM25, Hybrid (RRF fusion) |
| **Storage** | Redis Stack, ChromaDB |
| **LLM Backends** | Ollama, vLLM, OpenAI, any OpenAI-compatible API |
| **Embedding** | sentence-transformers/all-MiniLM-L12-v2 (local, no API key) |
| **Reranking** | cross-encoder/ms-marco-MiniLM-L12-v2 (local, CPU or GPU) |
| **Agents** | 20+ specialized pipeline agents |
| **Ingestion** | Files, directories, URLs, GitHub repositories |
| **MCP Tools** | 10 tools via stdio or HTTP transport |
| **Monitoring** | Prometheus metrics, OpenTelemetry tracing |

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     RADIANT RAG MCP SERVER                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐     │
│  │  Claude Code │  │Claude Desktop│  │  HTTP Client /     │     │
│  │  (stdio MCP) │  │ (stdio MCP)  │  │  Colab Notebook    │     │
│  └──────┬───────┘  └──────┬───────┘  └─────────┬──────────┘     │
│         └─────────────────┼──────────────────--┘                │
│                           ▼                                     │
│  ┌────────────────────────────────────────────────────────┐     │
│  │              FastMCP 3.x Server (server.py)            │     │
│  │           10 tools — stdio or http transport           │     │
│  └─────────────────────────┬──────────────────────────────┘     │
│                            ▼                                    │
│  ┌────────────────────────────────────────────────────────┐     │
│  │              RadiantRAG Application (app.py)           │     │
│  │  ┌──────────────────────────────────────────────────┐  │     │
│  │  │           Agentic Orchestrator                   │  │     │
│  │  │  Plan → QueryPrep → Retrieve → Rerank → Generate │  │     │
│  │  └──────────────────────────────────────────────────┘  │     │
│  └──────┬──────────────────┬──────────────────┬───────────┘     │
│         ▼                  ▼                  ▼                 │
│  ┌─────────────┐  ┌───────────────┐  ┌─────────────────┐        │
│  │ LLM Backend │  │  Vector Store │  │  Local Models   │        │
│  │  (OpenAI-   │  │  Redis Stack  │  │  sentence-      │        │
│  │ compatible) │  │  or ChromaDB  │  │  transformers / │        │
│  └─────────────┘  └───────────────┘  │  cross-encoder  │        │
│                                      └─────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Hierarchy

```
BaseAgent (Abstract)
├── LLMAgent (requires LLM client)
│   ├── PlanningAgent               (disabled by default)
│   ├── QueryDecompositionAgent     (disabled by default)
│   ├── QueryRewriteAgent
│   ├── QueryExpansionAgent
│   ├── AnswerSynthesisAgent
│   ├── CriticAgent
│   ├── ContextEvaluationAgent
│   ├── SummarizationAgent
│   ├── FactVerificationAgent       (disabled by default)
│   ├── CitationTrackingAgent
│   ├── LanguageDetectionAgent      (disabled by default)
│   ├── TranslationAgent            (disabled by default)
│   ├── IntelligentChunkingAgent
│   └── WebSearchAgent              (disabled by default)
│
├── RetrievalAgent (requires vector store)
│   └── DenseRetrievalAgent
│
└── BaseAgent (direct inheritance)
    ├── BM25RetrievalAgent
    ├── RRFAgent
    ├── HierarchicalAutoMergingAgent
    ├── CrossEncoderRerankingAgent
    └── MultiHopReasoningAgent      (disabled by default)
```

---

## Installation

### Prerequisites

- Python 3.10+
- An OpenAI-compatible LLM endpoint (Ollama Cloud, local Ollama, vLLM, OpenAI, etc.)
- Redis Stack (optional — only needed for the Redis backend)

### Install

```bash
# ChromaDB backend (default — no external service required)
pip install "radiant-rag-mcp[chroma] @ git+https://github.com/dshipley71/radiant-rag-mcp.git"

# Redis backend
pip install "radiant-rag-mcp[redis] @ git+https://github.com/dshipley71/radiant-rag-mcp.git"
```

### Development install (editable)

```bash
git clone https://github.com/dshipley71/radiant-rag-mcp.git
cd radiant-rag-mcp
pip install -e ".[chroma,dev]"
```

### Redis Stack (if using the Redis backend)

```bash
docker run -d --name radiant-redis \
  -p 6379:6379 -p 8001:8001 \
  redis/redis-stack:latest
```

---

## Quick Start

### stdio transport (Claude Code / Claude Desktop)

```bash
export RADIANT_OLLAMA_BASE_URL="https://ollama.com/v1"
export RADIANT_OLLAMA_API_KEY="your-api-key"
radiant-mcp
```

See [MCP_README.md](MCP_README.md) for Claude Code and Claude Desktop JSON config blocks.

### HTTP transport (standalone / notebook)

```bash
RADIANT_TRANSPORT=http \
RADIANT_HOST=127.0.0.1 \
RADIANT_PORT=8000 \
RADIANT_OLLAMA_BASE_URL="https://ollama.com/v1" \
RADIANT_OLLAMA_API_KEY="your-api-key" \
radiant-mcp
```

The MCP endpoint is at `http://127.0.0.1:8000/mcp`.

### Google Colab

Use the notebook at `notebooks/radiant_rag_mcp_colab_test.ipynb`.
The install cell:

```python
!pip install -q "radiant-rag-mcp[chroma] @ git+https://github.com/dshipley71/radiant-rag-mcp.git"
!pip install -q --prefer-binary nest_asyncio httpx "fastmcp>=3.0"
```

---

## Configuration

### Config file location

```bash
radiant-mcp --config /path/to/config.yaml
# or
export RADIANT_CONFIG_PATH=/path/to/config.yaml
```

Default search order: `./config.yaml`, `./config.yml`, `./radiant.yaml`,
`~/.radiant/config.yaml`, `/etc/radiant/config.yaml`.

### Key config sections

```yaml
# LLM backend (Ollama, vLLM, OpenAI, or local HuggingFace)
llm_backend:
  backend_type: "ollama"
  base_url: "https://ollama.com/v1"
  api_key: ""               # or set RADIANT_OLLAMA_API_KEY
  model: "gemma4:31b-cloud"
  timeout: 120
  max_retries: 1

# Embedding (local sentence-transformers)
embedding_backend:
  backend_type: "local"
  model_name: "sentence-transformers/all-MiniLM-L12-v2"
  device: "auto"            # auto | cuda | cpu
  embedding_dimension: 384

# Reranking (local cross-encoder)
reranking_backend:
  backend_type: "local"
  model_name: "cross-encoder/ms-marco-MiniLM-L12-v2"
  device: "auto"

# Storage
storage:
  backend: redis            # redis | chroma
```

### Environment variable overrides

Pattern: `RADIANT_<SECTION>_<KEY>` — overrides config.yaml at runtime.

```bash
export RADIANT_LLM_BACKEND_MODEL="llama3:8b"
export RADIANT_STORAGE_BACKEND="chroma"
export RADIANT_RETRIEVAL_DENSE_TOP_K="5"
export RADIANT_PIPELINE_USE_CRITIC="false"
export RADIANT_EMBEDDING_BACKEND_DEVICE="cuda"
export RADIANT_RERANKING_BACKEND_DEVICE="cuda"
```

---

## Storage Backends

| Backend | Use Case | Setup |
|---|---|---|
| **Redis Stack** | Production, low latency, large corpora | Docker container |
| **ChromaDB** | Development, testing, Colab, small corpora | Embedded, no service |

### ChromaDB

```yaml
storage:
  backend: chroma

chroma:
  persist_directory: "./data/chroma_db"
  collection_name: "radiant_docs"
  distance_fn: "cosine"
  embedding_dimension: 384
  max_content_chars: 200000
```

### Redis Stack

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
    distance_metric: "COSINE"
```

### Binary quantization (Redis and ChromaDB)

Both backends support binary and Int8 quantization for large-scale deployments:

```yaml
redis:   # or chroma
  quantization:
    enabled: true
    precision: "binary"     # binary | int8 | both
    rescore_multiplier: 4.0
    use_rescoring: true
```

Performance impact at 1M documents (384-dim):
- Memory: 1,536 MB → 432 MB (3.5× reduction)
- Retrieval speed: 50–100 ms → 5–10 ms (10–20× faster)
- Accuracy: 95–96% retained

---

## Agent Pipeline

### Pipeline flags

| Flag | Default | Effect when `false` |
|---|---|---|
| `pipeline.use_planning` | `false` | Skip PlanningAgent (saves 12+ min on Ollama Cloud) |
| `pipeline.use_decomposition` | `false` | Skip QueryDecomposition (3× LLM multiplier) |
| `pipeline.use_rewrite` | `true` | Skip query rewriting |
| `pipeline.use_expansion` | `true` | Skip query expansion |
| `pipeline.use_rrf` | `true` | Skip RRF fusion |
| `pipeline.use_automerge` | `true` | Skip hierarchical merging |
| `pipeline.use_rerank` | `true` | Skip cross-encoder reranking |
| `pipeline.use_critic` | `true` | Skip critic evaluation (saves ~6 min if LLM is slow) |

Setting `pipeline.use_critic: false` is sufficient — you do not also need
`critic.enabled: false`. The pipeline flag is the primary gate.

---

## Ingestion Pipeline

### Supported formats

| Format | Extensions |
|---|---|
| PDF | .pdf |
| Word | .docx, .doc |
| Text | .txt |
| Markdown | .md |
| HTML | .html |
| Code | .py, .js, .ts, .go, .rs, .java, .cpp, and more |
| JSON/JSONL | .json, .jsonl |
| Images | .png, .jpg (requires VLM, disabled by default) |

### Hierarchical storage

Documents are split into parent (full text) and child (chunk) records:

```
Parent Document
├── Child Chunk 1  ← embedded, searchable
├── Child Chunk 2  ← embedded, searchable
└── Child Chunk 3  ← embedded, searchable
```

The `HierarchicalAutoMergingAgent` promotes child chunks to the full parent context
when multiple sibling chunks are retrieved for the same query.

```yaml
ingestion:
  child_chunk_size: 512
  child_chunk_overlap: 50
  embed_parents: false     # true to also embed parent documents
```

---

## Query Pipeline

### Retrieval modes

| Mode | Description | Best for |
|---|---|---|
| `hybrid` | Dense + BM25 with RRF fusion | General-purpose queries |
| `dense` | Semantic similarity only | Conceptual / paraphrase queries |
| `bm25` | Keyword matching only | Exact terms, technical identifiers |

```yaml
retrieval:
  dense_top_k: 10
  bm25_top_k: 10
  fused_top_k: 15
  search_scope: "leaves"    # leaves | parents | all
```

---

## Performance

### Typical latencies (Ollama Cloud, CPU embedding/reranking)

| Component | Time |
|---|---|
| Query rewrite + expansion | ~3 s |
| Hybrid retrieval + fusion | < 0.1 s |
| Cross-encoder reranking (CPU) | ~3 s |
| Cross-encoder reranking (GPU T4) | ~0.3 s |
| Answer synthesis | ~4 s |
| Critic (if enabled, no timeout) | ~5 s |
| Citation tracking (if enabled) | ~80 s |

### Recommended fast-query config

Add to the Colab config cell or shell environment:

```python
os.environ['RADIANT_PIPELINE_USE_CRITIC']                   = 'false'
os.environ['RADIANT_CITATION_ENABLED']                      = 'false'
os.environ['RADIANT_CONTEXT_EVALUATION_USE_LLM_EVALUATION'] = 'false'
os.environ['RADIANT_LLM_BACKEND_TIMEOUT']                   = '30'
os.environ['RADIANT_LLM_BACKEND_MAX_RETRIES']               = '0'
os.environ['RADIANT_RERANKING_BACKEND_DEVICE']              = 'cuda'
os.environ['RADIANT_EMBEDDING_BACKEND_DEVICE']              = 'cuda'
```

Expected query time with these settings: **5–8 seconds**.

### Embedding model options

| Model | Dim | Speed | Quality |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Fastest | Slightly lower |
| `all-MiniLM-L12-v2` | 384 | Fast (default) | Good |
| `all-mpnet-base-v2` | 768 | Slower | Better |

Changing the model requires updating `embedding_dimension` in both
`embedding_backend` and the storage backend config, and rebuilding the index.

---

## Metrics & Monitoring

### Prometheus

```python
from radiant_rag_mcp.utils.metrics_export import PrometheusMetricsExporter

exporter = PrometheusMetricsExporter(namespace="radiant_rag")
exporter.record_execution(result)
output = exporter.get_metrics_output()  # expose at /metrics
```

### OpenTelemetry

```python
from radiant_rag_mcp.utils.metrics_export import OpenTelemetryExporter

exporter = OpenTelemetryExporter(
    service_name="radiant-rag",
    endpoint="http://localhost:4317",
)
```

---

## API Reference

### Python API

```python
from radiant_rag_mcp.app import create_app

app = create_app("config.yaml")

# Ingest
app.ingest_documents(["./docs/"], use_hierarchical=True)
app.ingest_urls(["https://github.com/owner/repo"])

# Query (full pipeline)
result = app.query_raw("What is RAG?", retrieval_mode="hybrid")
print(result.answer)
print(result.confidence)

# Search only (no LLM)
results = app.search("BM25 algorithm", mode="hybrid", top_k=10)

# Simple query (minimal pipeline, no agents)
answer = app.simple_query("What is RAG?", top_k=5)

# Multi-turn conversation
conv_id = app.start_conversation()
r1 = app.query_raw("What is RAG?", conversation_id=conv_id)
r2 = app.query_raw("How does BM25 work?", conversation_id=conv_id)

# System management
app.clear_index()
health = app.check_health()
stats = app.get_stats()
count = app.rebuild_bm25_index()
```

### PipelineResult fields

| Field | Type | Description |
|---|---|---|
| `answer` | `str` | Generated answer |
| `success` | `bool` | Pipeline completion status |
| `confidence` | `float` | Critic score 0–1 (0.5 if critic disabled) |
| `cited_answer` | `str` | Answer with inline citations |
| `citations` | `List[Dict]` | Citation detail objects |
| `retrieval_mode_used` | `str` | Actual retrieval mode applied |
| `retry_count` | `int` | Pipeline retry count |
| `multihop_used` | `bool` | Whether multi-hop reasoning ran |
| `fact_verification_score` | `float` | Factuality score (1.0 if disabled) |
| `audit_id` | `str` | Citation audit trail ID |

---

## File Structure

```
radiant-rag-mcp/
├── config.yaml
├── pyproject.toml
├── README.md
├── MCP_README.md
├── fastmcp.json
│
├── src/
│   └── radiant_rag_mcp/
│       ├── server.py           # FastMCP server + 10 tool definitions
│       ├── app.py              # RadiantRAG application
│       ├── orchestrator.py     # Agent pipeline orchestration
│       ├── config.py           # Configuration dataclasses + loading
│       ├── agents/             # 20+ pipeline agents  (see AGENTS.md)
│       ├── ingestion/          # Document processors and crawlers
│       ├── storage/            # Redis and ChromaDB backends
│       ├── llm/                # LLM client, backends, local models
│       ├── utils/              # Cache, metrics, conversation management
│       └── ui/                 # Display helpers and report generation
│
└── notebooks/
    └── radiant_rag_mcp_colab_test.ipynb
```

---

## License

MIT License
