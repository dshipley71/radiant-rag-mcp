# Radiant RAG MCP — Audit & Optimization Report

**Status as of `main` branch, April 2026.**  
Items marked ✅ are completed. Items marked 🔲 remain open.

---

## 1. Completed Changes

The following items from the original audit have been resolved:

| Item | Status |
|---|---|
| pgvector storage removed from codebase | ✅ `storage/pgvector_store.py` deleted; factory, `__init__`, `config.py`, `config.yaml`, `pyproject.toml` all cleaned |
| `haystack-ai` dependency replaced | ✅ Replaced with `openai>=1.0.0`; `ChatMessage` is now a plain dataclass in `llm/client.py` |
| `fast-langdetect` removed from core deps | ✅ Removed from `pyproject.toml`; language detection disabled by default |
| Language detection disabled | ✅ `language_detection.enabled: false` in `config.yaml` |
| `OllamaConfig` / `llm_backend` wired correctly | ✅ `llm_backend`, `embedding_backend`, `reranking_backend` sections added to `config.yaml` and read by `load_config()` |
| Missing `performance` section in `config.yaml` | ✅ Added with all 10 keys documented |
| `language_detection` model_path/model_url/auto_download not read by `load_config()` | ✅ Fixed — all four keys now read |
| FastMCP transport string mismatch | ✅ `server.py` `main()` updated: `http` transport now calls `mcp.run(transport="http", ...)` (was `streamable-http`) |
| FastMCP pinned to working version | ✅ `fastmcp>=3.0` in `pyproject.toml` and notebook install cell |
| Notebook: separate `!pip install fasttext` cell | ✅ Removed; fasttext not installed (language detection disabled) |
| Notebook: `REPO_DIR` sys.path manipulation | ✅ Removed; package installs to site-packages |
| Notebook: haystack in import checks | ✅ Removed; openai added |
| Notebook: models pre-cached before server startup | ✅ sentence-transformers and cross-encoder pre-downloaded in install cell |
| Notebook: port kill before server thread | ✅ `fuser -k {port}/tcp` added to prevent port conflicts on re-run |
| Notebook: API key via Colab Secrets | ✅ Uses `userdata.get('OLLAMA_API_KEY')` |
| Docs out of sync (README, MCP_README, AGENTS.md) | ✅ All updated — see below |

---

## 2. Open Items

### 2a. Heavy dependencies in core install

The following packages are in `[project.dependencies]` but are only needed for
optional VLM captioning (`vlm.enabled: false`) and local HuggingFace LLM backends
(both disabled by default):

```toml
# Should move to [project.optional-dependencies.local] or [extras.vlm]
"torch>=2.0.0",
"transformers>=4.35.0",
"tokenizers>=0.20.0",
"accelerate>=0.24.0",
"safetensors>=0.4.5",
"qwen-vl-utils>=0.0.8",
"huggingface-hub>=0.24.6",
"pillow>=10.0.0",
```

These add approximately 4–6 GB of install weight to every deployment, including
Colab sessions that only use the local embedding/reranking models.
`sentence-transformers` already pulls in `torch` transitively, so the explicit
`torch` dependency is doubly unnecessary for the default deployment.

**Recommendation:** Move to `pip install "radiant-rag-mcp[local]"` or
`"radiant-rag-mcp[vlm]"` extras.

### 2b. Dual LLM config paths (`OllamaConfig` vs `LLMBackendConfig`)

`config.py` maintains two parallel LLM configuration paths:
- **Legacy**: `OllamaConfig` — used internally by `LLMClient.__init__` as a fallback
- **Current**: `LLMBackendConfig` + `EmbeddingBackendConfig` + `RerankingBackendConfig`

When `llm_backend` is present in `config.yaml` (which it is), `LLMClients.build()`
uses the new backend system. `OllamaConfig` is populated from the YAML but only
used if the `llm_backend` section is absent.

The `OllamaConfig` path can be retired:
1. Remove `OllamaConfig` from `config.py`
2. Remove `ollama` from `AppConfig`
3. Update `LLMClient` to accept `LLMBackendConfig` directly
4. Remove the legacy fallback branch from `LLMClients.build()`

This eliminates a confusing dual-path where `RADIANT_OLLAMA_CHAT_MODEL` and
`RADIANT_LLM_BACKEND_MODEL` both exist but only one takes effect.

### 2c. Strategy memory path is ephemeral in Colab

```yaml
agentic:
  strategy_memory_path: "./data/strategy_memory.json.gz"
```

`./data/` is wiped on every Colab runtime reset. The strategy memory file
accumulates learned retrieval patterns across queries — it is worthless if
reset every session.

**Fix:** Override in the Colab config cell:
```python
os.environ['RADIANT_AGENTIC_STRATEGY_MEMORY_PATH'] = \
    '/content/drive/MyDrive/radiant/strategy_memory.json.gz'
```

Or mount Google Drive and point the path there.

### 2d. LLM-based chunking still enabled by default

```yaml
chunking:
  use_llm_chunking: true
```

This fires one LLM call per document over `llm_chunk_threshold` (3000 chars)
during ingestion. For bulk ingestion of large corpora this is the dominant
cost driver. The rule-based chunker produces quality within 5–10% for
well-structured technical documents.

**Recommendation:** Default to `false`; let users opt in for high-value documents.

### 2e. `agent_template.py` not imported anywhere

`src/radiant_rag_mcp/agents/agent_template.py` is never imported at runtime.
It is referenced only in `AGENTS.md` as a starter template for new agents.
Safe to keep as developer scaffolding, but worth noting it does not affect
the running system.

---

## 3. Performance Recommendations

### 3a. Pipeline feature flags (highest-impact levers)

| Flag | Default | Cost when enabled | Recommendation |
|---|---|---|---|
| `pipeline.use_planning` | `false` | 12+ min on Ollama Cloud | Keep `false` |
| `pipeline.use_decomposition` | `false` | 3× LLM call multiplier | Keep `false` |
| `pipeline.use_critic` | `true` | ~6 min if LLM times out | Set `false` for development |
| `citation.enabled` | `true` | ~80 s | Set `false` for development |
| `context_evaluation.use_llm_evaluation` | `true` | ~4 s per query | Set `false` (use heuristics) |
| `multihop.enabled` | `false` | 2 extra LLM calls per query | Keep `false` |
| `fact_verification.enabled` | `false` | ~52 s; redundant with critic | Keep `false` |
| `web_search.enabled` | `false` | Network round-trip + LLM call | Keep `false` unless needed |
| `language_detection.enabled` | `false` | 131 MB model + per-doc inference | Keep `false` for English corpora |

**Setting `pipeline.use_critic: false` is sufficient** — `critic.enabled` is a
secondary guard inside the agent and only matters if the pipeline gate is `true`.

### 3b. LLM timeout and retries

```yaml
llm_backend:
  timeout: 30     # was 120 — fail fast
  max_retries: 0  # was 1 — no retry cascade
```

With `timeout: 120` and `max_retries: 1`, a single timed-out LLM call costs
4+ minutes (2 × 120 s + retry delay). At `timeout: 30, max_retries: 0` the
failure surfaces in 30 seconds.

### 3c. GPU for local models

The cross-encoder reranking model runs on CPU by default (~3 s per query on
Colab T4). With GPU it runs in ~0.3 s. Override in the config cell:

```python
os.environ['RADIANT_RERANKING_BACKEND_DEVICE'] = 'cuda'
os.environ['RADIANT_EMBEDDING_BACKEND_DEVICE']  = 'cuda'
```

`device: "auto"` in the YAML should detect a GPU automatically, but explicit
override is more reliable in the Colab environment.

### 3d. Retrieval top-k sizing

Current defaults retrieve 10 dense + 10 BM25 = up to 20 unique docs, fuse to 15,
then rerank all 15. Tighter funneling reduces reranker load with negligible recall loss:

```yaml
retrieval:
  dense_top_k: 5
  bm25_top_k: 5
  fused_top_k: 10

rerank:
  top_k: 6
```

### 3e. Summarization threshold

```yaml
summarization:
  min_doc_length_for_summary: 2000  # raise to 3500
```

At 2000 chars, many mid-size chunks trigger a summarization LLM call unnecessarily.

### 3f. Recommended fast-query config block (Colab)

```python
# Add to the configuration cell — env vars override config.yaml at runtime
os.environ['RADIANT_PIPELINE_USE_CRITIC']                   = 'false'
os.environ['RADIANT_CITATION_ENABLED']                      = 'false'
os.environ['RADIANT_CONTEXT_EVALUATION_USE_LLM_EVALUATION'] = 'false'
os.environ['RADIANT_LLM_BACKEND_TIMEOUT']                   = '30'
os.environ['RADIANT_LLM_BACKEND_MAX_RETRIES']               = '0'
os.environ['RADIANT_RERANKING_BACKEND_DEVICE']              = 'cuda'
os.environ['RADIANT_EMBEDDING_BACKEND_DEVICE']              = 'cuda'
os.environ['RADIANT_CHUNKING_USE_LLM_CHUNKING']             = 'false'
```

Expected query time with these settings: **5–8 seconds**.

---

## 4. Future Improvements

### 4a. HyDE agent (Hypothetical Document Embedding)

The highest-ROI missing agent. Between `QueryExpansionAgent` and `DenseRetrievalAgent`:

```
Query → Rewrite → Expand → [HyDE: generate hypothetical answer, embed it]
                                   ↓
                       Dense retrieval using hypothetical embedding
```

One LLM call generates a plausible answer. That answer is embedded and used as the
dense query vector. Because hypothetical answers are semantically closer to
answer-bearing passages than the raw question, retrieval precision on short or
ambiguous queries improves significantly (+10–20% NDCG in published benchmarks).
Fits cleanly into the existing `LLMAgent` / `BaseAgent` framework in ~150 lines.

### 4b. Config.yaml reduction

The active `config.yaml` is ~560 lines. Many entries are rarely-tuned defaults that
exist in `config.py` dataclasses. A slimmed-down reference config could reduce it
to ~150 lines while keeping all tunable parameters visible. The full list of
removable per-section keys was documented in the original audit (still valid).

### 4c. `unstructured[all-docs]` install weight

`unstructured[all-docs]` pulls in OCR libraries, PDF parsers, image processors,
and more. For deployments that only ingest Markdown and plain text, a lighter
`unstructured` variant (or explicit optional extras) would significantly reduce
install time and container size.
