# radiant-rag-mcp

**Radiant Agentic RAG as an MCP server.**

`radiant-rag-mcp` wraps the full Radiant RAG pipeline as ten
[Model Context Protocol (MCP)](https://modelcontextprotocol.io) tools.
Transport is runtime-selectable: use **stdio** for Claude Code and Claude Desktop,
or **HTTP** for remote server and notebook deployments.

---

## Install

### Default backend — ChromaDB (no external service required)

```bash
pip install "radiant-rag-mcp[chroma] @ git+https://github.com/dshipley71/radiant-rag-mcp.git"
```

ChromaDB runs embedded in-process. No Docker, no database server.

### Redis Stack backend

```bash
pip install "radiant-rag-mcp[redis] @ git+https://github.com/dshipley71/radiant-rag-mcp.git"
```

Requires a running Redis Stack container (see [Docker commands](#docker-commands) below).

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

The server is reachable at `redis://localhost:6379`. RedisInsight is
available at `http://localhost:8001`.

Set the backend at runtime:

```bash
export RADIANT_STORAGE_BACKEND=redis
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
        "RADIANT_OLLAMA_BASE_URL": "https://ollama.com/v1",
        "RADIANT_OLLAMA_API_KEY": "your-api-key"
      }
    }
  }
}
```

#### HTTP (remote server, modern Streamable HTTP transport)

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

#### SSE (remote server, legacy)

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

### Starting the server for HTTP or SSE transport

Set `RADIANT_TRANSPORT` before starting `radiant-mcp`. The server binds to
`RADIANT_HOST:RADIANT_PORT` (defaults: `127.0.0.1:8000`):

```bash
# HTTP transport (Streamable HTTP — recommended)
RADIANT_TRANSPORT=http \
  RADIANT_HOST=127.0.0.1 \
  RADIANT_PORT=8000 \
  RADIANT_OLLAMA_BASE_URL=https://ollama.com/v1 \
  RADIANT_OLLAMA_API_KEY=your-api-key \
  radiant-mcp

# SSE transport (legacy)
RADIANT_TRANSPORT=sse \
  RADIANT_HOST=127.0.0.1 \
  RADIANT_PORT=8000 \
  RADIANT_OLLAMA_BASE_URL=https://ollama.com/v1 \
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
| `RADIANT_OLLAMA_BASE_URL` | LLM endpoint base URL (must include `/v1` for Ollama) | _(required)_ |
| `RADIANT_OLLAMA_API_KEY` | API key for the LLM endpoint | _(none)_ |
| `RADIANT_TRANSPORT` | MCP transport: `stdio`, `sse`, or `http` | `stdio` |
| `RADIANT_HOST` | Bind address for SSE and HTTP transports | `127.0.0.1` |
| `RADIANT_PORT` | Bind port for SSE and HTTP transports | `8000` |
| `RADIANT_CONFIG_PATH` | Path to a custom `config.yaml` | _(repo root)_ |
| `RADIANT_STORAGE_BACKEND` | Override vector store backend: `chroma` or `redis` | `chroma` |
| `RADIANT_LLM_BACKEND_MODEL` | Override the LLM model name | `gemma4:31b-cloud` |
| `RADIANT_LLM_BACKEND_BASE_URL` | Low-level alias for `RADIANT_OLLAMA_BASE_URL` | — |
| `RADIANT_LLM_BACKEND_API_KEY` | Low-level alias for `RADIANT_OLLAMA_API_KEY` | — |

All `config.yaml` keys can be overridden via `RADIANT_<SECTION>_<KEY>`
(e.g. `RADIANT_RETRIEVAL_DENSE_TOP_K=5`, `RADIANT_PIPELINE_USE_CRITIC=false`).

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
Query my knowledge base: what are the performance trade-offs between Redis and ChromaDB
for large document collections?
```

**Claude Code prompt (multi-turn):**
```
Query my knowledge base using conversation ID "conv_abc123": based on the previous
answer, which backend is better for low-latency production use?
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
  "conversation_id": "a3d68495-88d8-474a-9646-a49ee162bea3"
}
```

Pass the returned `conversation_id` to subsequent `query_knowledge` calls to maintain
context across turns.

---

### `ingest_documents`

**Claude Code prompt:**
```
Index the documents in /home/user/docs/reports into my knowledge base.
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
    "max_pages": 50
  }
}
```

---

### `get_index_stats`

**Claude Code prompt:**
```
Show me the current state of my knowledge base — document counts and health.
```

**MCP Inspector call:**
```json
{
  "tool": "get_index_stats",
  "arguments": {}
}
```

---

### `clear_index`

**MCP Inspector call (confirmed):**
```json
{
  "tool": "clear_index",
  "arguments": {
    "confirm": true
  }
}
```

**Without confirmation (safe default — no data deleted):**
```json
{
  "tool": "clear_index",
  "arguments": {
    "confirm": false
  }
}
```

---

### `rebuild_bm25`

**Claude Code prompt:**
```
Rebuild the BM25 index from the current vector store.
```

**MCP Inspector call:**
```json
{
  "tool": "rebuild_bm25",
  "arguments": {}
}
```

---

### `set_ingest_mode`

**MCP Inspector call (switch to flat):**
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
  "message": "Ingest mode set to 'flat'. All subsequent ingest_documents calls will use flat storage by default."
}
```

---

## LLM model

The default model is `gemma4:31b-cloud` via `https://ollama.com/v1` (configured
in `config.yaml` under `llm_backend`). Override at runtime:

```bash
export RADIANT_OLLAMA_BASE_URL="http://localhost:11434/v1"  # local Ollama
export RADIANT_LLM_BACKEND_MODEL="your-model-name"
```

Note: the Ollama base URL must end in `/v1` for the OpenAI-compatible endpoint.

---

## Development

```bash
git clone https://github.com/dshipley71/radiant-rag-mcp.git
cd radiant-rag-mcp
pip install -e ".[chroma,dev]"
radiant-mcp --config config.yaml
```
