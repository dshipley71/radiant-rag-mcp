# APP_API.md — Python API Reference

Developer guide for using `radiant-rag-mcp` directly from Python, without
going through the MCP protocol layer.

Package: `radiant_rag_mcp`  
Module: `src/radiant_rag_mcp/app.py`

---

## Overview

`app.py` is the core engine of the system.  Everything — the MCP server,
the CLI, and any custom Python integration — runs through the same
`RadiantRAG` class defined here.

```
┌─────────────────────────────────────────────────────────────┐
│                         app.py                              │
│                    RadiantRAG class                         │
│                                                             │
│   All ingestion, retrieval, query, and conversation         │
│   logic lives here.  Nothing is duplicated elsewhere.       │
└───────────────┬─────────────────────┬───────────────────────┘
                │                     │
                ▼                     ▼
      ┌──────────────────┐   ┌────────────────────────────────┐
      │    server.py     │   │   Any Python application       │
      │   MCP shell      │   │                                │
      │  (10 MCP tools,  │   │   FastAPI  ·  Flask  ·  Django │
      │   thin wrappers  │   │   Celery   ·  Jupyter          │
      │   around app.py) │   │   scripts  ·  pipelines        │
      └──────────────────┘   └────────────────────────────────┘
```

`server.py` contributes exactly two things on top of `app.py`: MCP protocol
serialisation and `asyncio.to_thread()` to prevent the async server from
blocking.  The ten MCP tools are one-to-one mappings onto `RadiantRAG`
methods:

| MCP tool | `RadiantRAG` method |
|---|---|
| `search_knowledge` | `app.search()` |
| `query_knowledge` | `app.query_raw()` |
| `simple_query` | `app.simple_query()` |
| `start_conversation` | `app.start_conversation()` |
| `ingest_documents` | `app.ingest_documents()` |
| `ingest_url` | `app.ingest_urls()` |
| `get_index_stats` | `app.get_stats()` + `app.check_health()` |
| `clear_index` | `app.clear_index()` |
| `rebuild_bm25` | `app.rebuild_bm25_index()` |
| `set_ingest_mode` | *(server-side flag only — no app.py method)* |

---

## Quick start

```python
from radiant_rag_mcp.app import create_app

app = create_app("config.yaml")          # or create_app() for default config
```

`create_app()` reads config from `config.yaml` (or the path you supply),
wires up the vector store, BM25 index, LLM clients, local embedding and
reranking models, and the full agent orchestrator.  The call is synchronous
and safe to make at module import time.

---

## Configuration

### From a file

```python
app = create_app("/path/to/config.yaml")
```

### From environment variables

All `config.yaml` keys can be overridden at runtime without touching the file.
The pattern is `RADIANT_<SECTION>_<KEY>`:

```bash
export RADIANT_LLM_BACKEND_MODEL="llama3:8b"
export RADIANT_STORAGE_BACKEND="chroma"
export RADIANT_RETRIEVAL_DENSE_TOP_K="10"
export RADIANT_PIPELINE_USE_CRITIC="false"
export RADIANT_SUMMARIZATION_ENABLED="true"
export RADIANT_EMBEDDING_BACKEND_DEVICE="cuda"
```

```python
import os
os.environ["RADIANT_LLM_BACKEND_MODEL"] = "llama3:8b"
app = create_app()                       # picks up env overrides
```

### Accessing the config at runtime

```python
cfg = app.config                         # AppConfig dataclass (frozen)
print(cfg.llm_backend.model)
print(cfg.storage.backend)
print(cfg.summarization.enabled)
```

---

## Ingestion

### Local files and directories

```python
stats = app.ingest_documents(
    paths=["./docs/", "./reports/q4.pdf", "./notes.md"],
    use_hierarchical=True,    # parent/child chunk storage (default)
    child_chunk_size=512,     # tokens per child chunk
    child_chunk_overlap=50,   # token overlap between chunks
)

print(stats["files_processed"])   # int
print(stats["chunks_created"])    # int
print(stats["documents_stored"])  # int
print(stats["errors"])            # list[str]
```

Setting `use_hierarchical=False` stores all chunks at a single level.  Use
this for small corpora or when you want to disable auto-merging.

### URLs and GitHub repositories

`ingest_urls()` automatically detects GitHub URLs and routes them to the
GitHub crawler.  All other URLs go through the standard web crawler.

```python
# Single page — no link following
stats = app.ingest_urls(
    urls=["https://docs.example.com/api-reference"],
    crawl_depth=0,
)

# Crawl an entire documentation site
stats = app.ingest_urls(
    urls=["https://docs.example.com"],
    crawl_depth=3,
    max_pages=200,
)

# GitHub repository — all markdown files are fetched
stats = app.ingest_urls(
    urls=["https://github.com/owner/repo"],
)

# Private site with basic auth
stats = app.ingest_urls(
    urls=["https://internal.corp/wiki"],
    basic_auth=("username", "password"),
)
```

### GitHub repositories (direct)

```python
stats = app.ingest_github(
    github_url="https://github.com/owner/repo",
    use_hierarchical=True,
    fetch_all_files=True,
    follow_readme_links=True,
)
```

---

## Querying

### Full agentic pipeline — `query_raw()`

Returns a `PipelineResult` without any console output.  This is the
equivalent of the `query_knowledge` MCP tool and the right choice for
programmatic use.

```python
result = app.query_raw(
    query="What storage backends does Radiant RAG support?",
    retrieval_mode="hybrid",   # "hybrid" | "dense" | "bm25"
    conversation_id=None,      # omit for stateless single-turn
)

print(result.answer)
print(f"Confidence: {result.confidence:.0%}")
print(f"Citations : {len(result.citations)}")
```

### Full pipeline with Rich console output — `query()`

Formats and prints a report to the terminal.  Useful for scripts and the
interactive CLI.  Returns the same `PipelineResult`.

```python
result = app.query(
    query="Explain binary quantization.",
    retrieval_mode="hybrid",
    show_result=True,
    show_metrics=False,
    compact=False,
    save_path="./reports/result.json",   # optional
)
```

### Lightweight pipeline — `simple_query()`

Skips advanced agents (planning, decomposition, critic, citation tracking).
Lower latency; suitable for simple factual lookups.

```python
answer: str = app.simple_query(
    query="What embedding model does Radiant RAG use?",
    top_k=5,
)
print(answer)
```

### Search without LLM — `search()`

Pure retrieval: no answer synthesis, no LLM call.

```python
hits = app.search(
    query="BM25 sparse retrieval",
    mode="hybrid",   # "hybrid" | "dense" | "bm25"
    top_k=10,
    show_results=False,
)

for doc, score in hits:
    print(f"[{score:.3f}]  {doc.content[:120]}")
    print(f"        source: {doc.meta.get('source_url') or doc.meta.get('file_path')}")
```

---

## Multi-turn conversations

```python
# Start a session
conv_id = app.start_conversation()

# Turn 1
r1 = app.query_raw(
    "What is RAG?",
    conversation_id=conv_id,
)
print(r1.answer)

# Turn 2 — pipeline receives the prior context automatically
r2 = app.query_raw(
    "How does hybrid retrieval improve on dense-only approaches?",
    conversation_id=conv_id,
)
print(r2.answer)

# Retrieve formatted history
history = app.get_conversation_history(conv_id, max_turns=10)
print(history)
```

You can supply your own `conversation_id` string (e.g. a user session ID
from your web framework) instead of letting the system generate one:

```python
conv_id = app.start_conversation(conversation_id="user-session-abc123")
```

---

## `PipelineResult` reference

Returned by `query_raw()` and `query()`.  Also available as a dict via
`result.to_dict()` (which is what the MCP `query_knowledge` tool returns).

| Field | Type | Description |
|---|---|---|
| `answer` | `str` | Generated answer text |
| `success` | `bool` | `True` if the pipeline completed without fatal error |
| `confidence` | `float` | Critic score 0–1 (`0.5` when critic is disabled) |
| `cited_answer` | `str \| None` | Answer with inline citations (`[DOC N]`) |
| `citations` | `list[dict]` | Citation detail objects |
| `retrieval_mode_used` | `str` | Actual retrieval mode applied |
| `retry_count` | `int` | Number of critic-driven retries |
| `low_confidence` | `bool` | `True` if confidence fell below threshold |
| `multihop_used` | `bool` | `True` if multi-hop reasoning ran |
| `multihop_hops` | `int` | Number of reasoning hops taken |
| `audit_id` | `str \| None` | Citation audit trail identifier |
| `warnings` | `list[str]` | Pipeline warnings (e.g. context compressed) |
| `metrics` | `RunMetrics` | Per-step latency and agent execution details |

---

## System management

### Index statistics and health

```python
health = app.check_health()
print(health["vector_index"]["document_count"])
print(health["bm25_index"]["document_count"])
print(health["bm25_index"]["needs_rebuild"])

stats = app.get_stats()   # health + pipeline run metrics history
```

### Rebuild BM25 index

Required after bulk ingestion or if the BM25 index becomes out of sync with
the vector store (e.g. after a failed `clear_index`).

```python
count = app.rebuild_bm25_index(limit=0)   # 0 = rebuild all
print(f"Indexed {count} documents")
```

### Clear the index

```python
cleared = app.clear_index(keep_bm25=False)
```

Setting `keep_bm25=True` clears only the vector store, leaving the BM25 index
intact — useful when you want to swap a storage backend without losing the
sparse index.

> **Known ChromaDB behaviour:** the Chroma collection is dropped successfully
> but the post-clear re-initialisation raises
> `'ChromaVectorStore' object has no attribute '_ensure_index'`.
> The data is gone; only the reinit step fails.  Call
> `app.rebuild_bm25_index()` and then re-ingest to restore a clean state.

---

## Integration patterns

### FastAPI service

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from radiant_rag_mcp.app import create_app

@asynccontextmanager
async def lifespan(app_: FastAPI):
    app_.state.rag = create_app("config.yaml")
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/query")
async def query(question: str, mode: str = "hybrid"):
    import asyncio
    result = await asyncio.to_thread(
        app.state.rag.query_raw, question, retrieval_mode=mode
    )
    return {"answer": result.answer, "confidence": result.confidence}

@app.post("/ingest")
async def ingest(paths: list[str]):
    import asyncio
    stats = await asyncio.to_thread(
        app.state.rag.ingest_documents, paths
    )
    return stats
```

### Celery background worker

```python
from celery import Celery
from radiant_rag_mcp.app import create_app

celery = Celery("tasks", broker="redis://localhost:6379/1")
_app = None

def get_app():
    global _app
    if _app is None:
        _app = create_app("config.yaml")
    return _app

@celery.task
def ingest_document(path: str) -> dict:
    return get_app().ingest_documents([path])

@celery.task
def answer_question(question: str) -> str:
    result = get_app().query_raw(question, retrieval_mode="hybrid")
    return result.answer
```

### Jupyter / Colab notebook

```python
from radiant_rag_mcp.app import create_app

app = create_app("config.yaml")

# Ingest your research corpus
app.ingest_documents(["./papers/", "./notes/"])
app.ingest_urls(["https://arxiv.org/abs/2312.10997"])

# Query interactively
result = app.query_raw("What methods improve RAG retrieval quality?")
print(result.answer)
```

### Accessing internals for custom agents

When you need to wire up an agent that isn't part of the standard pipeline
(e.g. `SummarizationAgent` called directly), you can reach into the
orchestrator to get the shared LLM and embedding clients:

```python
from radiant_rag_mcp.app import create_app
from radiant_rag_mcp.agents.summarization import SummarizationAgent

app = create_app("config.yaml")
llm = app._orchestrator._llm         # LLMClient

agent = SummarizationAgent(
    llm=llm,
    min_doc_length_for_summary=800,
    target_summary_length=300,
)

summary = agent.summarize_for_query(long_text, query="What are the key findings?")
```

---

## Standalone CLI

`app.py` contains a full `argparse` CLI that is accessible without the MCP
server.  It is not registered as a `pyproject.toml` entry point by default
(only `radiant-mcp` pointing at `server.py` is registered), so invoke it as
a module:

```bash
# Ingest files
python -m radiant_rag_mcp.app ingest ./docs/ ./reports/
python -m radiant_rag_mcp.app ingest --url https://docs.example.com --crawl-depth 2
python -m radiant_rag_mcp.app ingest --url https://github.com/owner/repo

# Query
python -m radiant_rag_mcp.app query "What is hybrid retrieval?"
python -m radiant_rag_mcp.app query "Summarise the latest report" --mode bm25 --save result.json
python -m radiant_rag_mcp.app query "Quick answer" --simple

# Search (no LLM)
python -m radiant_rag_mcp.app search "binary quantization" --top-k 5

# Interactive REPL
python -m radiant_rag_mcp.app interactive

# Maintenance
python -m radiant_rag_mcp.app stats
python -m radiant_rag_mcp.app health
python -m radiant_rag_mcp.app rebuild-bm25
python -m radiant_rag_mcp.app clear --confirm
```

To register it as a first-class entry point alongside `radiant-mcp`, add to
`pyproject.toml`:

```toml
[project.scripts]
radiant-mcp = "radiant_rag_mcp.server:main"
radiant-rag = "radiant_rag_mcp.app:main"    # ← add this line
```

After `pip install -e .` the `radiant-rag` command will be available on your
`PATH`:

```bash
radiant-rag ingest ./docs/
radiant-rag query "What is RAG?" --mode hybrid
```

---

## File reference

| File | Role |
|---|---|
| `app.py` | `RadiantRAG` class — all ingestion, query, and management logic |
| `server.py` | FastMCP server — 10 MCP tools, each a thin wrapper around `app.py` |
| `orchestrator.py` | Agent pipeline execution — called by `app.query_raw()` |
| `config.py` | Configuration dataclasses and `load_config()` |
| `ingestion/processor.py` | Document parsing, chunking, and cleaning |
| `ingestion/web_crawler.py` | URL crawling |
| `ingestion/github_crawler.py` | GitHub repository crawling |
| `storage/` | Redis Stack and ChromaDB backends |
| `agents/` | 20+ pipeline agents (see `AGENTS.md`) |
| `llm/client.py` | LLM and embedding client wrappers |
| `utils/conversation.py` | Multi-turn session management |
