# radiant-rag-mcp

**Radiant Agentic RAG as an MCP server.**

`radiant-rag-mcp` wraps the full Radiant RAG pipeline as ten
[Model Context Protocol (MCP)](https://modelcontextprotocol.io) tools.
Transport is runtime-selectable: use **stdio** for Claude Code and Claude
Desktop, or **SSE** / **Streamable HTTP** for remote server deployments.

---

## Install

### Default backend — ChromaDB (no external service required)

```bash
pip install radiant-rag-mcp
# or explicitly:
pip install "radiant-rag-mcp[chroma]"
```

ChromaDB runs embedded in-process.  No Docker, no database server.

### Redis Stack backend

```bash
pip install "radiant-rag-mcp[redis]"
```

Requires a running Redis Stack container (see [Docker commands](#docker-commands) below).

### PostgreSQL + pgvector backend

```bash
pip install "radiant-rag-mcp[pgvector]"
```

Requires a running pgvector container (see [Docker commands](#docker-commands) below).

---

## Docker commands

### Redis Stack

```bash
docker run -d \
  --name radiant-redis \
  -p 6379:6379 \
  -p 8001:8001 \
  redis/redis-stack:latest
```

The server is reachable at `redis://localhost:6379`.  RedisInsight is
available at `http://localhost:8001`.

Set the backend at runtime:

```bash
export RADIANT_STORAGE_BACKEND=redis
```

### PostgreSQL + pgvector

```bash
docker run -d \
  --name radiant-pgvector \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=radiant \
  -e POSTGRES_DB=radiant \
  pgvector/pgvector:pg16
```

Default connection string: `postgresql://postgres:radiant@localhost:5432/radiant`.
Override with:

```bash
export PG_CONN_STR="postgresql://user:password@host:5432/dbname"
export RADIANT_STORAGE_BACKEND=pgvector
```

---

## Configuration

### Claude Code / Claude Desktop config blocks

#### stdio (default — Claude Code / Claude Desktop)

```json
{
  "mcpServers": {
    "radiant-rag": {
      "type": "stdio",
      "command": "radiant-mcp",
      "env": {
        "RADIANT_OLLAMA_BASE_URL": "http://ollama.com",
        "RADIANT_OLLAMA_API_KEY": "your-api-key"
      }
    }
  }
}
```

#### SSE (remote server)

```json
{
  "mcpServers": {
    "radiant-rag": {
      "type": "sse",
      "url": "http://127.0.0.1:8000/sse"
    }
  }
}
```

#### Streamable HTTP (remote server, modern)

```json
{
  "mcpServers": {
    "radiant-rag": {
      "type": "streamable-http",
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

### Starting the server for SSE or HTTP transport

Set `RADIANT_TRANSPORT` before starting `radiant-mcp`.  The server binds to
`RADIANT_HOST:RADIANT_PORT` (defaults: `127.0.0.1:8000`):

```bash
# SSE transport
RADIANT_TRANSPORT=sse \
  RADIANT_HOST=127.0.0.1 \
  RADIANT_PORT=8000 \
  RADIANT_OLLAMA_BASE_URL=http://ollama.com \
  RADIANT_OLLAMA_API_KEY=your-api-key \
  radiant-mcp

# Streamable HTTP transport
RADIANT_TRANSPORT=http \
  RADIANT_HOST=127.0.0.1 \
  RADIANT_PORT=8000 \
  RADIANT_OLLAMA_BASE_URL=http://ollama.com \
  RADIANT_OLLAMA_API_KEY=your-api-key \
  radiant-mcp
```

### With a custom config file

```bash
radiant-mcp --config /path/to/config.yaml
# or
RADIANT_CONFIG_PATH=/path/to/config.yaml radiant-mcp
```

---

## Environment variable reference

| Variable | Description | Default |
|---|---|---|
| `RADIANT_OLLAMA_BASE_URL` | Ollama / OpenAI-compatible LLM endpoint base URL | `http://ollama.com` |
| `RADIANT_OLLAMA_API_KEY` | API key for the LLM endpoint | _(none)_ |
| `RADIANT_TRANSPORT` | MCP transport: `stdio`, `sse`, or `http` | `stdio` |
| `RADIANT_HOST` | Bind address for SSE and HTTP transports | `127.0.0.1` |
| `RADIANT_PORT` | Bind port for SSE and HTTP transports | `8000` |
| `RADIANT_CONFIG_PATH` | Path to a custom `config.yaml` | _(repo root)_ |
| `RADIANT_STORAGE_BACKEND` | Override vector store backend: `chroma`, `redis`, or `pgvector` | `chroma` |
| `PG_CONN_STR` | PostgreSQL connection string (pgvector backend only) | `postgresql://postgres:radiant@localhost:5432/radiant` |
| `RADIANT_LLM_BACKEND_MODEL` | Override the LLM model name | `gemma4:31b-cloud` |
| `RADIANT_LLM_BACKEND_BASE_URL` | Low-level alias for `RADIANT_OLLAMA_BASE_URL` | — |
| `RADIANT_LLM_BACKEND_API_KEY` | Low-level alias for `RADIANT_OLLAMA_API_KEY` | — |

All `config.yaml` keys can also be overridden via `RADIANT_<SECTION>_<KEY>`
(e.g. `RADIANT_RETRIEVAL_TOP_K=20`).

---

## Exposed MCP tools

| Tool | Description |
|---|---|
| `search_knowledge` | Hybrid/dense/BM25 retrieval without LLM generation |
| `query_knowledge` | Full agentic pipeline with synthesis and citations |
| `simple_query` | Lightweight query skipping advanced agents |
| `start_conversation` | Create a conversation ID for multi-turn sessions |
| `ingest_documents` | Index local file paths or directories |
| `ingest_url` | Index a URL or GitHub repository |
| `get_index_stats` | Document counts and system health |
| `clear_index` | Clear all indexed documents (requires `confirm=True`) |
| `rebuild_bm25` | Rebuild BM25 index from vector store contents |
| `set_ingest_mode` | Set hierarchical vs flat storage for subsequent ingestion |

---

## Tool usage examples

Each example shows both the natural language prompt you would give Claude Code
and the equivalent MCP Inspector JSON call for direct testing.

---

### `search_knowledge`

**Claude Code prompt:**
```
Search my knowledge base for documents about hybrid retrieval and BM25 scoring.
```

**MCP Inspector call:**
```json
{
  "tool": "search_knowledge",
  "arguments": {
    "query": "hybrid retrieval and BM25 scoring",
    "mode": "hybrid",
    "top_k": 5
  }
}
```

---

### `query_knowledge`

**Claude Code prompt (single-turn):**
```
Query my knowledge base: what are the performance trade-offs between Redis and ChromaDB for large document collections?
```

**Claude Code prompt (multi-turn — requires a conversation ID from `start_conversation` first):**
```
Query my knowledge base using conversation ID "conv_abc123": based on the previous answer, which backend is better for low-latency production use?
```

**MCP Inspector call:**
```json
{
  "tool": "query_knowledge",
  "arguments": {
    "question": "What are the performance trade-offs between Redis and ChromaDB?",
    "mode": "hybrid",
    "conversation_id": null
  }
}
```

---

### `simple_query`

**Claude Code prompt:**
```
Do a quick lookup in my knowledge base: what chunk size does Radiant RAG use by default?
```

**MCP Inspector call:**
```json
{
  "tool": "simple_query",
  "arguments": {
    "question": "What chunk size does Radiant RAG use by default?",
    "top_k": 5
  }
}
```

---

### `start_conversation`

**Claude Code prompt:**
```
Start a new conversation session in my knowledge base so I can ask follow-up questions.
```

**MCP Inspector call:**
```json
{
  "tool": "start_conversation",
  "arguments": {}
}
```

**Example response:**
```json
{
  "conversation_id": "conv_abc123"
}
```

Pass the returned `conversation_id` to subsequent `query_knowledge` calls to maintain context across turns.

---

### `ingest_documents`

**Claude Code prompt (hierarchical, default):**
```
Index the documents in /home/user/docs/reports into my knowledge base.
```

**Claude Code prompt (flat storage):**
```
Index /home/user/docs/glossary into my knowledge base using flat storage.
```

**MCP Inspector call:**
```json
{
  "tool": "ingest_documents",
  "arguments": {
    "paths": ["/home/user/docs/reports"],
    "hierarchical": true
  }
}
```

---

### `ingest_url`

**Claude Code prompt (single page):**
```
Index https://docs.example.com/api-reference into my knowledge base without crawling.
```

**Claude Code prompt (GitHub repo):**
```
Index the GitHub repository https://github.com/owner/repo into my knowledge base.
```

**Claude Code prompt (crawled site):**
```
Crawl and index https://docs.example.com up to 3 levels deep, maximum 50 pages.
```

**MCP Inspector call:**
```json
{
  "tool": "ingest_url",
  "arguments": {
    "url": "https://docs.example.com/api-reference",
    "no_crawl": true
  }
}
```

**MCP Inspector call (crawled):**
```json
{
  "tool": "ingest_url",
  "arguments": {
    "url": "https://docs.example.com",
    "crawl_depth": 3,
    "max_pages": 50,
    "no_crawl": false
  }
}
```

---

### `get_index_stats`

**Claude Code prompt:**
```
Show me the current state of my knowledge base — document counts, health, and storage backend.
```

**MCP Inspector call:**
```json
{
  "tool": "get_index_stats",
  "arguments": {}
}
```

**Example response:**
```json
{
  "total_documents": 1482,
  "total_chunks": 9847,
  "storage_backend": "chroma",
  "index_healthy": true,
  "bm25_index_size": 9847,
  "embedding_model": "sentence-transformers/all-MiniLM-L12-v2"
}
```

---

### `clear_index`

**Claude Code prompt:**
```
Clear all documents from my knowledge base. I confirm this action.
```

**MCP Inspector call:**
```json
{
  "tool": "clear_index",
  "arguments": {
    "confirm": true
  }
}
```

**Without confirmation (safe default):**
```json
{
  "tool": "clear_index",
  "arguments": {
    "confirm": false
  }
}
```

Returns: `{"status": "cancelled", "reason": "pass confirm=True to proceed"}` — no data is deleted.

---

### `rebuild_bm25`

**Claude Code prompt:**
```
My BM25 search results seem stale. Rebuild the BM25 index from the current vector store.
```

**MCP Inspector call:**
```json
{
  "tool": "rebuild_bm25",
  "arguments": {}
}
```

**Example response:**
```json
{
  "status": "ok",
  "message": "BM25 index rebuilt from 9847 documents."
}
```

---

### `set_ingest_mode`

**Claude Code prompt (switch to flat):**
```
Switch my knowledge base ingestion to flat storage mode for this session.
```

**Claude Code prompt (switch back to hierarchical):**
```
Switch ingestion back to hierarchical storage mode.
```

**MCP Inspector call:**
```json
{
  "tool": "set_ingest_mode",
  "arguments": {
    "hierarchical": false
  }
}
```

**Example response:**
```json
{
  "hierarchical": false,
  "message": "Ingestion mode set to flat storage for this session."
}
```

---

## LLM model

The default LLM model is `gemma4:31b-cloud` and the default base URL is
`http://ollama.com` (configured in `config.yaml` under `llm_backend`).
Override at runtime:

```bash
export RADIANT_OLLAMA_BASE_URL="http://localhost:11434/v1"   # local Ollama
export RADIANT_LLM_BACKEND_MODEL="your-model-name"
```

---

## Development

```bash
git clone https://github.com/dshipley71/radiant-rag
cd radiant-rag
pip install -e ".[chroma,dev]"
radiant-mcp --config config.yaml
```

Run tests:

```bash
pytest tests/
```
