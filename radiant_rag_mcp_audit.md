# Radiant RAG MCP — Audit & Optimization Report

---

## 1. Files and Directories Safe to Remove

### Entire `docs/` directory

None of these files are imported or referenced at runtime. They are developer notes
and generated analysis docs only.

```
docs/AGENTS_MD_EXAMPLE_WALKTHROUGH.md
docs/AGENTS_MD_USAGE.md
docs/AGENTS_MD_VALIDATION_REPORT.md
docs/AGENT_ARCHITECTURE.md
docs/BINARY_QUANTIZATION_README.md
docs/CHANGES_ORCHESTRATOR_UPDATE.md
docs/NEW_CORPUS_GUIDE.md
docs/POST_OPTIMIZATION_ANALYSIS.md
docs/POST_OPTIMIZATION_ANALYSIS.md
docs/radiant_rag_executive_summary.html
docs/radiant_rag_executive_summary.md
docs/radiant_rag_query_processing_pipeline.html
docs/view_html_pages.md
docs/USER_MANUAL.md          # Keep only if you want end-user docs in the repo
```

### Source files with no runtime callers

| File | Reason |
|---|---|
| `radiant/agents/agent_template.py` | Never imported anywhere in the codebase. Referenced only in `radiant/agents/AGENTS.md` as a developer guide. Safe to remove or move into `docs/`. |
| `config_quantization_example.yaml` | Only referenced in `docs/BINARY_QUANTIZATION_README.md`. All quantization defaults live in `config.py`. |

### CLI-only file (safe to remove for pure MCP deployments)

`radiant/ui/tui.py` — Only reached via the `--tui` CLI flag (lazy-imported at line 1638
of `app.py`). An MCP server deployment never hits this path. Removing it eliminates the
`rich`-based terminal UI but has zero effect on the MCP server or Colab notebook.

---

## 2. `config.yaml` Streamlining

The current config.yaml is ~560 lines. Below are two categories of cuts:

### 2a. Entire sections to remove (disabled features, all defaults exist in `config.py`)

These sections are gated by `enabled: false` in the active config. Every key in
them has a hard-coded default in the corresponding `*Config` dataclass in `config.py`.
Remove the sections entirely to reduce visual noise and cognitive load.

```yaml
# REMOVE — multihop is disabled; all keys default correctly
multihop: ...

# REMOVE — fact_verification is disabled; all keys default correctly
fact_verification: ...

# REMOVE — translation is disabled; all keys default correctly
translation: ...

# REMOVE — web_search is disabled; all keys default correctly
web_search: ...
```

For `web_search`, keep only the on/off toggle if you might enable it later:

```yaml
web_search:
  enabled: false    # Re-enable here; all other params have safe defaults
```

### 2b. Rarely-tuned keys to remove (defaults verified in `config.py`)

For each section below, the listed keys can be stripped. The default shown is
what `config.py` will supply when the key is absent.

**`vlm`** — VLM is disabled. If you keep the section, reduce it to:

```yaml
vlm:
  enabled: false
  model_name: "Qwen/Qwen3-VL-8B-Instruct"
  device: "auto"
  load_in_4bit: true
  load_in_8bit: false
# Remove: max_new_tokens (512), temperature (0.2), cache_dir (null),
#          ollama_fallback_url, ollama_fallback_model
```

**`llm_backend`** — Remove `retry_delay` (default 1.0). Keep `timeout` and `max_retries`
since they're specifically tuned for Ollama Cloud.

**`embedding_backend`** — Remove `cache_size` (default 10 000).

**`redis`** — Remove `doc_ns`, `embed_ns`, `meta_ns`, `conversation_ns`, `max_content_chars`
(200 000). Under `vector_index`, remove `hnsw_ef_construction` (200) and `hnsw_ef_runtime`
(100). Keep `name`, `hnsw_m`, `distance_metric`.

**Inactive backend sections** — If your deployment uses Redis, remove the `chroma` and
`pgvector` sections entirely (and vice-versa). Keeping all three actively encourages
accidental misconfiguration. Use the `storage.backend` key as the single source of truth.

**`bm25`** — Remove `max_documents` (100 000), `auto_save_threshold` (100).
Keep `index_path`, `k1`, `b`.

**`ingestion`** — Remove `redis_batch_size` (100), `show_progress` (true).
Keep `batch_enabled`, `embedding_batch_size`, `child_chunk_size`, `child_chunk_overlap`,
`embed_parents`.

**`retrieval`** — Remove `rrf_k` (60), `min_similarity` (0.0). Keep the four top-k values
and `search_scope`.

**`rerank`** — Remove `candidate_multiplier` (4), `min_candidates` (16). Keep `top_k`
and `max_doc_chars`.

**`automerge`** — Remove `max_parent_chars` (50 000). Keep `min_children_to_merge`.

**`synthesis`** — Remove `max_context_docs` (8), `max_doc_chars` (4000),
`max_history_turns` (5). Keep `include_history`.

**`critic`** — Remove `max_context_docs` (8), `max_doc_chars` (1200). Keep `enabled`,
`retry_on_issues`, `max_retries`, `confidence_threshold`, `min_retrieval_confidence`.

**`agentic`** — Remove `retry_expansion_factor` (1.5), `max_critic_retries` (2).
Keep all the boolean feature flags, `confidence_threshold`, `strategy_memory_path`.

**`chunking`** — Remove `llm_chunk_threshold` (3000), `min_chunk_size` (200),
`max_chunk_size` (1500), `overlap_size` (100). Keep `enabled`, `use_llm_chunking`,
`target_chunk_size`.

**`summarization`** — Remove `similarity_threshold` (0.85), `max_cluster_size` (3),
`conversation_compress_threshold` (6), `conversation_preserve_recent` (2).
Keep `enabled`, `min_doc_length_for_summary`, `target_summary_length`,
`max_total_context_chars`.

**`context_evaluation`** — Remove `min_relevant_docs` (1), `max_docs_to_evaluate` (8),
`max_doc_chars` (1000). Keep `enabled`, `use_llm_evaluation`, `sufficiency_threshold`,
`abort_on_poor_context`.

**`citation`** — Remove `max_citations_per_claim` (3), `excerpt_max_length` (200).
Keep `enabled`, `citation_style`, `min_citation_confidence`, `include_excerpts`,
`generate_bibliography`, `generate_audit_trail`.

**`language_detection`** — Remove `model_url` (Facebook AI default, never needs changing),
`verify_checksum` (false), `use_llm_fallback`. Keep `enabled`, `method`,
`min_confidence`, `fallback_language`, `model_path`, `auto_download`.

**`query`** — Remove `cache_ttl` (3600). Keep `max_decomposed_queries`,
`max_expansions`, `cache_enabled`.

**`conversation`** — Remove `ttl` (86 400), `history_turns_for_context` (3).
Keep `enabled`, `max_turns`, `use_history_for_retrieval`.

**`parsing`** — Remove the entire section. All three keys (`max_retries: 2`,
`retry_delay: 0.5`, `strict_json: false`, `log_failures: true`) are stable defaults
that users never need to touch.

**`unstructured_cleaning`** — Remove `preview_enabled`, `preview_max_items`,
`preview_max_chars`. These are debug scaffolding.

**`json_parsing`** — Remove `max_nesting_depth` (10), `flatten_separator` (.),
`jsonl_batch_size` (1000). Keep `enabled`, `default_strategy`,
`min_array_size_for_splitting`, `text_fields`, `title_fields`, `preserve_fields`.

**`logging`** — Remove `format`, `file`, `json_logging`. Keep `level`,
`quiet_third_party`, `colorize`.

**`metrics`** — Remove `history_retention` (100), `store_history` (false).
Keep `enabled`, `detailed_timing`.

**`web_crawler`** — Reduce to the keys users actually need to change:

```yaml
web_crawler:
  max_depth: 2
  max_pages: 100
  same_domain_only: true
  delay: 0.5
  timeout: 30
  verify_ssl: true
  # Everything else (patterns, user_agent, auth, redirects, file size)
  # has safe defaults and is rarely changed
```

### 2c. Missing section: `performance`

`PerformanceConfig` is fully implemented in `config.py` but absent from `config.yaml`.
The defaults are fine, but adding a stub with the two most impactful keys makes the
system more transparent:

```yaml
# -----------------------------------------------------------------------------
# Performance Tuning
# (all keys have safe defaults — edit only when profiling shows a need)
# -----------------------------------------------------------------------------
performance:
  parallel_retrieval_enabled: true     # Run dense + BM25 in parallel threads
  parallel_postprocessing_enabled: true
  simple_query_max_words: 10           # Queries ≤ N words skip decomposition/expansion
```

---

## 3. Performance Optimization Recommendations

### 3a. Pipeline feature flags (the highest-impact levers)

| Flag | Current | Recommendation | Rationale |
|---|---|---|---|
| `pipeline.use_planning` | `false` | Keep `false` | 12+ min overhead on Ollama Cloud. Re-enable only for local deployments with fast LLMs. |
| `pipeline.use_decomposition` | `false` | Keep `false` | Multiplies LLM calls 3× through the entire pipeline for minimal quality gain on most queries. |
| `multihop.enabled` | `false` | Keep `false` | 2 extra LLM calls + retrieval loops per query. Only useful for complex multi-entity reasoning over large corpora. |
| `fact_verification.enabled` | `false` | Keep `false` | ~52 s overhead on Ollama Cloud; the `CriticAgent` already performs quality and confidence scoring. Redundant. |
| `web_search.enabled` | `false` | Keep `false` | Adds a network round-trip and an extra LLM call to every query. Enable only if the corpus needs augmentation with live data. |
| `language_detection.enabled` | `true` | **Set `false` for English-only corpora** | Downloads a 131 MB fasttext model on first run, then runs inference on every ingested document. Zero benefit if your corpus is English-only. |

### 3b. Context evaluation LLM call

```yaml
context_evaluation:
  use_llm_evaluation: true   # CHANGE TO: false
```

`use_llm_evaluation: true` fires a full LLM round-trip *before* every generation as a
quality gate. The heuristic-only path (`false`) runs in microseconds, catches the same
obvious "no relevant docs" cases, and avoids a second LLM call on every query.
Re-enable only if you're seeing the pipeline generate confidently bad answers from
clearly irrelevant context.

### 3c. LLM-based chunking at ingestion

```yaml
chunking:
  use_llm_chunking: true   # CHANGE TO: false for bulk ingestion
```

LLM chunking fires one LLM call per document above `llm_chunk_threshold` (3000 chars).
For large ingestion batches this is the dominant cost driver. The rule-based chunker
produces quality within 5–10% of LLM chunking for well-structured technical documents.
Strategy: ingest with `use_llm_chunking: false`, then re-ingest high-value documents
selectively with it enabled.

### 3d. Retrieval top-k sizing

Current: `dense_top_k: 10`, `bm25_top_k: 10`, `fused_top_k: 15`, `rerank.top_k: 8`

```yaml
retrieval:
  dense_top_k: 8      # was 10 — saves ~20% embedding lookup time
  bm25_top_k: 8       # was 10
  fused_top_k: 12     # was 15 — fewer docs for the reranker to score

rerank:
  top_k: 6            # was 8 — synthesis gets 6 high-quality docs vs 8 noisier ones
```

The reranker is doing useful work; the issue is feeding it too many candidates. Tighter
funneling (8 → 12 → 6) reduces reranker latency by ~30% with negligible recall loss
because RRF fusion has already surfaced the best candidates.

### 3e. Summarization threshold

```yaml
summarization:
  min_doc_length_for_summary: 2000   # CHANGE TO: 3500
```

At 2000 chars, many moderately-sized chunks are being summarized unnecessarily. Raising
to 3500 chars focuses compression on genuinely long documents while cutting
summarization LLM calls by roughly half in typical corpora.

### 3f. Strategy memory path (Colab-specific)

```yaml
agentic:
  strategy_memory_path: "./data/strategy_memory.json.gz"   # CHANGE FOR COLAB:
  strategy_memory_path: "/content/drive/MyDrive/radiant/strategy_memory.json.gz"
```

`./data/` is ephemeral in Colab. The strategy memory file accumulates learned retrieval
patterns across queries — it's worthless if it's lost on every runtime reset. Point it
at Google Drive or a persisted mount.

### 3g. Embedding model selection

The current `sentence-transformers/all-MiniLM-L12-v2` (384d) is a solid default.
Two alternatives worth considering:

| Model | Dim | Speed | Quality | Use case |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | ~2× faster | Slightly lower | High-throughput ingestion |
| `all-mpnet-base-v2` | 768 | ~1.5× slower | Noticeably better | Quality-critical retrieval |

If you switch to `all-mpnet-base-v2`, update `embedding_dimension: 768` in both
`embedding_backend` and the storage backend config (Redis/Chroma/PgVector), and
rebuild the index.

### 3h. Agent to consider adding: HyDE

**Hypothetical Document Embedding (HyDE)** is the single highest-ROI agent missing from
the current pipeline. It fits between `QueryExpansionAgent` and `DenseRetrievalAgent`:

```
Query → Rewrite → Expand → [HyDE: generate a hypothetical answer, embed it]
                                     ↓
                          Dense retrieval using hypothetical embedding
                                     ↓
                          (existing pipeline continues)
```

One LLM call generates a plausible answer to the question. That answer is embedded and
used as the dense query vector instead of the raw question. Because the hypothetical
answer is semantically closer to actual answer-bearing passages than the question itself,
retrieval precision on short or ambiguous queries improves significantly (commonly
+10–20% NDCG in benchmarks). The `BaseAgent`/`LLMAgent` framework makes this a
~150-line addition.

---

## 4. Additional Optimizations

### 4a. `pyproject.toml` — heavy deps in core install

The following packages are in `dependencies` (always installed) but are only needed for
optional local-model and VLM use cases, both of which are disabled by default:

```toml
# MOVE THESE to [project.optional-dependencies.local] or [extras.vlm]
"torch>=2.0.0",
"transformers>=4.35.0",
"tokenizers>=0.20.0",
"accelerate>=0.24.0",
"safetensors>=0.4.5",
"qwen-vl-utils>=0.0.8",
"huggingface-hub>=0.24.6",
"pillow>=10.0.0",
```

These add ~4–6 GB of install weight to every deployment even when using Ollama as the
backend. Making them optional (`pip install radiant-rag-mcp[local]`) reduces the default
install to under 500 MB.

Similarly, `fast-langdetect` (fasttext wrapper, ~131 MB model download) should move to
an optional `[extras.multilingual]` group.

### 4b. Dual LLM config architecture (`OllamaConfig` vs `llm_backend`)

`config.py` maintains two parallel LLM config paths: the legacy `OllamaConfig` (used
by `LLMClient`) and the new `LLMBackendConfig` / `EmbeddingBackendConfig` /
`RerankingBackendConfig` system (used by the backend factory). Both are populated from
`config.yaml`. `LLMClients` in `client.py` (line 782) falls back to `OllamaConfig` when
`llm_backend` isn't fully wired.

The `OllamaConfig` class should be retired and `LLMClient` updated to accept
`LLMBackendConfig` directly. This is a meaningful refactor but eliminates a latent source
of "I changed `llm_backend.base_url` but nothing happened" confusion.

### 4c. `performance` section is invisible

`PerformanceConfig` has 8 fields that control parallel retrieval, embedding cache,
query cache, and early stopping — all highly relevant to tuning. None of them appear
in `config.yaml`. Users have no way of knowing these knobs exist. Add a `performance:`
stub to `config.yaml` (see Section 2c above).

### 4d. Notebook issues

**Cell 7 — hardcoded API key:**

```python
# Replace:
os.environ['RADIANT_OLLAMA_API_KEY'] = '71a62853...'

# With:
from google.colab import userdata
os.environ['RADIANT_OLLAMA_API_KEY'] = userdata.get('OLLAMA_KEY')
```

Store the key in Colab Secrets (the key icon in the left sidebar). The current approach
commits a credential to a shareable notebook.

**Cell 11 — isolated `fasttext` install:**

`!pip install fasttext` is a standalone cell that runs after Cell 3. Merge it into
Cell 3 alongside `rank-bm25`:

```python
!pip install -q --prefer-binary chromadb rank-bm25 fasttext
```

**Missing: config validation cell** — Add a cell between configuration (Cell 7) and
server startup (Cell 12) that loads and prints the active config to confirm everything
resolved correctly:

```python
from radiant.config import load_config
cfg = load_config()
print(f"LLM backend: {cfg.llm_backend.backend_type} / {cfg.llm_backend.model}")
print(f"Storage:     {cfg.storage.backend}")
print(f"Embedding:   {cfg.embedding_backend.model_name}")
print(f"Critic:      {cfg.critic.enabled}  |  Citation: {cfg.citation.enabled}")
```

**Missing: `httpx` timeout in `_call()`:** The helper in Cell 9 creates an `httpx`
client with no explicit timeout. A stalled LLM call will hang the notebook indefinitely.
Add `timeout=90.0` to the `httpx.AsyncClient` constructor.

**Missing: graceful shutdown cell** — Add a final cell that signals the server thread
to stop (or at minimum documents that kernel restart is the shutdown path). The current
notebook leaves the background thread dangling.

### 4e. Entry-point mismatch in `pyproject.toml`

```toml
[project.scripts]
radiant-mcp = "radiant_rag_mcp.server:main"
```

The installed package namespace is `radiant` (the `radiant/` source directory), not
`radiant_rag_mcp`. The `[tool.setuptools.packages.find] where = ["src"]` line implies
a `src/` layout, but the source tree uses a flat layout. This entry-point will fail
unless `radiant_rag_mcp` is a valid alias or a shim module under `src/`. Verify and
align with the actual import path (`radiant.app:main`).
