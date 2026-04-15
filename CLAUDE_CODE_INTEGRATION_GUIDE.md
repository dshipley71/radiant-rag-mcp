# Integrating radiant-rag-mcp into Claude Code
## And Using the Video RAG Prompt to Generate New Code

---

## Overview

This guide has two parts:

**Part 1** — Wire `radiant-rag-mcp` into Claude Code as a persistent MCP server so
all ten RAG tools (`ingest_documents`, `search_knowledge`, `query_knowledge`, etc.)
are available in every Claude Code session.

**Part 2** — Use `CLAUDE_CODE_VIDEO_RAG_PROMPT.md` to drive Claude Code through the
implementation of the video ingestion, summarization, and retrieval extension.

---

# Part 1 — Integrating radiant-rag-mcp into Claude Code

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | `python --version` |
| Node.js | 18+ | Required by Claude Code CLI |
| Claude Code | latest | `npm install -g @anthropic-ai/claude-code` |
| Ollama API key | — | Get one at https://ollama.com/settings/keys |
| Git | any | For pip install from GitHub |

---

## Step 1 — Install Claude Code

If you haven't already:

```bash
npm install -g @anthropic-ai/claude-code
```

Verify the installation:

```bash
claude --version
```

---

## Step 2 — Install radiant-rag-mcp

Install into the Python environment that will run the server.
**Use the same Python that Claude Code will find on your PATH.**

```bash
# ChromaDB backend (no external service — recommended for getting started)
pip install "radiant-rag-mcp[chroma] @ git+https://github.com/dshipley71/radiant-rag-mcp.git"

# Also install the transport helper used by Claude Code
pip install "fastmcp>=3.0"
```

Verify the entry point was installed:

```bash
which radiant-mcp
radiant-mcp --help
```

> **Virtual environments:** If you use a venv or conda environment, you must
> either activate it before running Claude Code, or use the full absolute path
> to `radiant-mcp` in the MCP configuration (see Step 4).

---

## Step 3 — Download the config file

The server needs `config.yaml` to know which LLM backend, embedding model,
and storage backend to use.

```bash
# Save to a permanent location — the server reads this on every startup
mkdir -p ~/.radiant
wget -O ~/.radiant/config.yaml \
  "https://raw.githubusercontent.com/dshipley71/radiant-rag-mcp/main/config.yaml"
```

You can also keep it at the root of any project and point to it with
`RADIANT_CONFIG_PATH`.

---

## Step 4 — Configure the MCP server in Claude Code

Claude Code reads MCP server configuration from JSON settings files.
There are three scopes — choose the one that fits your use case:

| Scope | File | Use when |
|---|---|---|
| **Global** (recommended) | `~/.claude/settings.json` | You want `radiant-rag` in every project |
| **Project** (checked in) | `.claude/settings.json` | Shared config for a team |
| **Project local** (gitignored) | `.claude/settings.local.json` | Personal overrides, API keys |

### Option A — Global configuration (recommended)

Create or edit `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "radiant-rag": {
      "type": "stdio",
      "command": "radiant-mcp",
      "env": {
        "RADIANT_OLLAMA_BASE_URL": "https://ollama.com/v1",
        "RADIANT_OLLAMA_API_KEY": "your-ollama-api-key",
        "RADIANT_STORAGE_BACKEND": "chroma",
        "RADIANT_CONFIG_PATH": "/Users/yourname/.radiant/config.yaml",
        "RADIANT_PIPELINE_USE_CRITIC": "false",
        "RADIANT_CITATION_ENABLED": "false",
        "RADIANT_LLM_BACKEND_TIMEOUT": "60",
        "RADIANT_LLM_BACKEND_MAX_RETRIES": "0"
      }
    }
  }
}
```

Replace:
- `your-ollama-api-key` with your key from https://ollama.com/settings/keys
- `/Users/yourname/.radiant/config.yaml` with the actual path from Step 3

### Option B — Add via the Claude Code CLI

```bash
claude mcp add radiant-rag \
  --command radiant-mcp \
  --env RADIANT_OLLAMA_BASE_URL=https://ollama.com/v1 \
  --env RADIANT_OLLAMA_API_KEY=your-key \
  --env RADIANT_STORAGE_BACKEND=chroma \
  --env RADIANT_CONFIG_PATH=/Users/yourname/.radiant/config.yaml
```

### Option C — Project-level configuration

Create `.claude/settings.json` at the root of your project:

```json
{
  "mcpServers": {
    "radiant-rag": {
      "type": "stdio",
      "command": "radiant-mcp",
      "env": {
        "RADIANT_OLLAMA_BASE_URL": "https://ollama.com/v1",
        "RADIANT_OLLAMA_API_KEY": "${RADIANT_OLLAMA_API_KEY}",
        "RADIANT_STORAGE_BACKEND": "chroma",
        "RADIANT_CONFIG_PATH": "${PWD}/config.yaml"
      }
    }
  }
}
```

Keep API keys out of the checked-in file by using environment variable
references (`${VAR_NAME}`) and storing the actual values in
`.claude/settings.local.json` (add that file to `.gitignore`).

### If radiant-mcp is not on PATH (virtual environment case)

Use the full absolute path:

```json
{
  "mcpServers": {
    "radiant-rag": {
      "type": "stdio",
      "command": "/Users/yourname/venvs/rag/bin/radiant-mcp",
      "env": { ... }
    }
  }
}
```

Find the path with:

```bash
# With your venv active:
which radiant-mcp
```

---

## Step 5 — Verify the MCP server starts

Check that Claude Code can see the server without starting a full session:

```bash
claude mcp list
```

You should see `radiant-rag` listed as a configured server. If it shows an
error, check:
- Is `radiant-mcp` on the PATH that Claude Code uses?
- Is the `RADIANT_CONFIG_PATH` pointing to a file that exists?
- Does the Python environment with `radiant-rag-mcp` installed match the
  Python that `radiant-mcp` resolves to?

---

## Step 6 — Start a Claude Code session and verify tools

Open a terminal in your project directory and start Claude Code:

```bash
claude
```

Once inside the session, ask Claude to list available tools:

```
/mcp
```

or:

```
What MCP tools do you have available?
```

You should see all ten tools listed:

```
radiant-rag:search_knowledge
radiant-rag:query_knowledge
radiant-rag:simple_query
radiant-rag:start_conversation
radiant-rag:ingest_documents
radiant-rag:ingest_url
radiant-rag:get_index_stats
radiant-rag:clear_index
radiant-rag:rebuild_bm25
radiant-rag:set_ingest_mode
```

---

## Step 7 — Ingest your first documents

From inside a Claude Code session:

```
Ingest the documents in ./docs/ and ./README.md into the knowledge base.
```

Claude Code will call `ingest_documents` automatically. You can also be explicit:

```
Use the ingest_documents tool to index ./src/ with hierarchical=true.
```

To ingest a URL or GitHub repository:

```
Index the GitHub repository https://github.com/dshipley71/radiant-rag-mcp
into the knowledge base.
```

---

## Step 8 — Use the knowledge base in your work session

Once documents are ingested, every question you ask Claude Code can be grounded
in your indexed content:

```
Search the knowledge base for how VideoProcessor handles audio detection.
```

```
Query the knowledge base: what storage backends does Radiant RAG support
and what are the performance trade-offs?
```

```
Start a conversation session, then ask: what agents run during a query_knowledge
call and in what order?
```

For code tasks, Claude Code will retrieve relevant context automatically
when it determines retrieval would improve the answer. You can also make it
explicit:

```
Before writing the implementation, search the knowledge base for how
ImageCaptioner handles lazy model loading and temp file cleanup.
```

---

## Step 9 — Persistent index across sessions

The ChromaDB index persists to `./data/chroma_db/` (relative to wherever
`radiant-mcp` runs from, which is the working directory of Claude Code).
The BM25 index persists to `./data/bm25_index.json.gz`.

You only need to ingest once. On subsequent Claude Code sessions, the index
is already populated and all search/query tools work immediately without
re-ingesting.

To check what is currently in the index:

```
Show me the current knowledge base statistics.
```

---

## Performance Tips for Claude Code Sessions

| Setting | Recommended value | Effect |
|---|---|---|
| `RADIANT_PIPELINE_USE_CRITIC` | `false` | Saves ~5 s per query |
| `RADIANT_CITATION_ENABLED` | `false` | Saves ~80 s per query |
| `RADIANT_LLM_BACKEND_TIMEOUT` | `60` | Prevents hung tool calls |
| `RADIANT_LLM_BACKEND_MAX_RETRIES` | `0` | Fast failure on LLM errors |
| `RADIANT_RETRIEVAL_DENSE_TOP_K` | `5` | Smaller context for faster synthesis |

These are already set in the configuration blocks above. Adjust
`RADIANT_CITATION_ENABLED=true` when you specifically need source citations.

---

---

# Part 2 — Using the Video RAG Prompt to Generate New Code

## Overview

`CLAUDE_CODE_VIDEO_RAG_PROMPT.md` is a detailed implementation specification
that tells Claude Code exactly what files to create and modify, what patterns
to follow, and what the implementations must satisfy. This part walks through
using it to generate the full video RAG extension for `radiant-rag-mcp`.

---

## Step 1 — Clone the repository and open it in Claude Code

```bash
git clone https://github.com/dshipley71/radiant-rag-mcp.git
cd radiant-rag-mcp
```

Start Claude Code in this directory:

```bash
claude
```

Claude Code now has read/write access to the entire repository.

---

## Step 2 — Ingest the codebase and specification into the knowledge base

This is where Part 1 pays off. Before touching a single line of code, give
Claude Code a searchable index of everything it needs to reference:

```
Use the ingest_documents tool to index ./src/ with hierarchical=true.
```

Then ingest the specification:

```
Use the ingest_documents tool to index ./CLAUDE_CODE_VIDEO_RAG_PROMPT.md.
```

If you have the other reference docs:

```
Use the ingest_documents tool to index ./AGENTS.md, ./APP_API.md, and ./MCP_README.md.
```

Verify the index:

```
Show me the knowledge base statistics.
```

You should see at least 200–300 documents indexed. Every implementation
question Claude Code asks itself will now retrieve an answer from the actual
source rather than training knowledge.

---

## Step 3 — Present the specification to Claude Code

Copy the full contents of `CLAUDE_CODE_VIDEO_RAG_PROMPT.md` into the Claude
Code session, or reference it directly:

```
Read the file CLAUDE_CODE_VIDEO_RAG_PROMPT.md and tell me what it's asking
you to implement. Summarise the new files, the modified files, and the key
design decisions before we start writing any code.
```

Claude Code will read the spec and give you a summary like:

```
The spec asks me to implement:

New files:
  - src/radiant_rag_mcp/ingestion/video_processor.py
  - src/radiant_rag_mcp/agents/video_summarization.py
  - notebooks/radiant_rag_mcp_video_test.ipynb

Modified files:
  - src/radiant_rag_mcp/config.py
  - src/radiant_rag_mcp/ingestion/processor.py
  - src/radiant_rag_mcp/ingestion/__init__.py
  - src/radiant_rag_mcp/agents/__init__.py
  - src/radiant_rag_mcp/app.py
  - src/radiant_rag_mcp/server.py
  - pyproject.toml

Key design decisions:
  - VideoProcessor detects audio presence and routes to Whisper or VLM
  - Silent videos use a sliding frame-window approach with filmstrip tiling
  - Three verbosity presets (brief/standard/detailed) for summaries
  ...
```

If the summary looks correct, proceed. If anything is misunderstood, clarify
before generating any code.

---

## Step 4 — Implement section by section

Work through the nine implementation sections of the spec one at a time.
Do **not** ask Claude Code to implement everything at once — the spec covers
nine files, and implementing them in dependency order avoids having to fix
forward-reference errors later.

### Recommended order

#### Section 1 — `pyproject.toml`

```
Implement Section 1 of the spec: add the video optional-dependency group
to pyproject.toml and add yt-dlp and faster-whisper to base dependencies.
Show me the diff before applying it.
```

Review the diff. Apply it.

---

#### Section 2 — `config.py`

```
Implement Section 2 of the spec: add VideoProcessorConfig and
VideoSummarizationConfig to config.py, wire them into AppConfig, and
add their load_config() blocks. Before you start, search the knowledge
base for how VLMCaptionerConfig is defined and wired as a reference.
```

> **What to look for in the output:**
> - Both dataclasses are `@dataclass(frozen=True)`
> - Both appear as fields in `AppConfig`
> - Both have `_get_config_value(data, "video", ...)` blocks in `load_config()`
> - Both are included in the `return AppConfig(...)` call

Verify with a quick import check:

```
Run: python -c "from radiant_rag_mcp.config import VideoProcessorConfig,
VideoSummarizationConfig; print('ok')"
```

---

#### Section 3 — `video_processor.py`

This is the largest file. Split it into sub-tasks:

**3a — Availability guards and dataclasses:**
```
Implement Section 3a and 3b of the spec: write the module-level
availability guards and the VideoSegment, FrameWindow, and VideoMetadata
dataclasses in a new file src/radiant_rag_mcp/ingestion/video_processor.py.
```

**3b — Audio detection:**
```
Add the _has_audio() and _probe_streams() methods to VideoProcessor.
Search the knowledge base for how ffprobe JSON output is structured
before writing the ffmpeg-python branch.
```

**3c — Audio (transcript) path:**
```
Add the _process_audio_video(), _load_whisper(), _transcribe(), and
_chunks_from_segments() methods. The chunk meta dict must match exactly
what is specified in Section 3e of the spec.
```

**3d — Silent video path:**
```
Add the silent video methods: _process_silent_video(), _detect_scene_changes(),
_build_windows(), _extract_window_frames(), _tile_filmstrip(),
_save_filmstrip_to_temp(), _caption_window(), _caption_single_frame(),
_analyse_windows(), and _chunks_from_frame_windows().

Important constraints from the spec:
- _save_filmstrip_to_temp() must return a file path (str), not a PIL.Image,
  because ImageCaptioner.caption_image() takes a file path
- _caption_window() must call _save_filmstrip_to_temp() and delete the
  temp file after captioning
Search the knowledge base for how ImageCaptioner.caption_image() is
called to confirm the signature before writing the captioning code.
```

**3e — yt-dlp helpers:**
```
Add _download_youtube(), _extract_local_metadata(), is_youtube_url(),
is_video_file(), and the public process_video(), process_youtube(),
process_local_video() methods.
```

Run the import smoke test after each sub-task:

```
Run: python -c "from radiant_rag_mcp.ingestion.video_processor import
VideoProcessor; print('ok')"
```

---

#### Section 4 — `video_summarization.py`

```
Implement Section 4 of the spec: create
src/radiant_rag_mcp/agents/video_summarization.py with the VideoChapter,
VideoSummaryResult dataclasses and the VideoSummarizationAgent class.

Use the exact prompt templates from Section 4c of the spec for:
- Chapter summary (transcript)
- Chapter summary (frame_window_captions)
- Overall summary
- Key topics extraction

Apply the preset-to-field mapping table from Section 2b at the start of
summarize_video() to override chapter_paragraphs_* and overall_paragraphs_*
when summary_detail is set.
```

Verify:

```
Run: python -c "from radiant_rag_mcp.agents.video_summarization import
VideoSummarizationAgent; print('ok')"
```

---

#### Section 5 — `ingestion/processor.py`

```
Implement Section 5 of the spec: add VIDEO_EXTENSIONS to processor.py,
add video_config parameter to DocumentProcessor.__init__, and add the
video routing branch in process_file() before the JSON branch.
```

---

#### Section 6 — `ingestion/__init__.py` and `agents/__init__.py`

```
Implement Sections 8 and 9 of the spec: add the VideoProcessor exports
to ingestion/__init__.py and the VideoSummarizationAgent exports to
agents/__init__.py.
```

---

#### Section 7 — `app.py`

```
Implement Section 6 of the spec: add ingest_videos() to app.py.

Critical: the LLM client must be accessed as self._llm_clients.chat,
not self._llm_client. Search the knowledge base for how query_raw() in
app.py accesses the LLM client to confirm this before writing the method.
```

Verify there are no references to `self._llm_client` (the wrong name):

```
Run: grep -n "_llm_client\b" src/radiant_rag_mcp/app.py
```

The only matches should be `self._llm_clients` (with the `s`).

---

#### Section 8 — `server.py`

```
Implement Section 7 of the spec: add the ingest_video MCP tool to
server.py immediately after the ingest_url tool definition. Update
the module docstring to say eleven tools.
```

Verify the tool is registered:

```
Run: python -c "
from radiant_rag_mcp.server import mcp
tools = [t.name for t in mcp._tool_manager.list_tools()]
print('ingest_video' in tools, tools)
"
```

---

#### Section 9 — Test notebook

```
Create notebooks/radiant_rag_mcp_video_test.ipynb following the structure
described in Section 10 of the spec. The notebook must follow the same
install/server-startup/helpers pattern as
notebooks/radiant_rag_mcp_search_query.ipynb — read that file first to
confirm the exact cell structure before writing the new notebook.
```

---

## Step 5 — Run the full validation checklist

After all sections are implemented, run through the spec's Validation Checklist
as a single Claude Code task:

```
Run all the validation checklist items from CLAUDE_CODE_VIDEO_RAG_PROMPT.md.
For each item, execute the command, report the result, and mark it as
passed or failed. Stop and fix any failure before continuing to the next item.
```

The checklist covers seventeen items including import smoke tests,
unit-level checks for `_has_audio()`, `_build_windows()`, `_tile_filmstrip()`,
`_save_filmstrip_to_temp()`, and end-to-end `ingest_video` path verification.

A passing checklist means the implementation is safe to test in Colab.

---

## Step 6 — Fix any checklist failures

For each failure, use the RAG knowledge base to retrieve relevant context
before asking Claude Code to fix it:

```
The validation for _has_audio() is failing with [paste error].
Search the knowledge base for how ffprobe JSON output is structured and
what fields to expect. Then fix the _has_audio() implementation.
```

```
The _chunks_from_frame_windows() meta dict is missing the is_silent field.
Search the knowledge base for how _chunks_from_segments() builds its meta
dict and apply the same pattern.
```

The value of having the codebase indexed is that Claude Code can look up
the correct pattern instead of guessing.

---

## Step 7 — Commit the implementation

Once the checklist passes:

```bash
git add src/radiant_rag_mcp/ingestion/video_processor.py
git add src/radiant_rag_mcp/agents/video_summarization.py
git add src/radiant_rag_mcp/config.py
git add src/radiant_rag_mcp/ingestion/processor.py
git add src/radiant_rag_mcp/ingestion/__init__.py
git add src/radiant_rag_mcp/agents/__init__.py
git add src/radiant_rag_mcp/app.py
git add src/radiant_rag_mcp/server.py
git add pyproject.toml
git add notebooks/radiant_rag_mcp_video_test.ipynb
git commit -m "feat: add video ingestion, summarization, and RAG support

- VideoProcessor: auto-detects audio; routes to Whisper transcription
  (audio videos) or VLM frame-window analysis (silent videos)
- VideoSummarizationAgent: three detail presets (brief/standard/detailed)
  with content-type-aware chapter and overall summary prompts
- ingest_video MCP tool: supports local files and yt-dlp URLs
- Silent video: sliding frame windows, scene change detection, filmstrip
  tiling, Qwen2-VL captioning
- Test notebook: notebooks/radiant_rag_mcp_video_test.ipynb"
```

---

## Step 8 — Test in Google Colab

Open `notebooks/radiant_rag_mcp_video_test.ipynb` in Google Colab with a T4 GPU.

Run the cells in order. Key things to watch for:

| Section | Expected outcome |
|---|---|
| Install (§1) | All packages install without conflicts |
| Import check (§2) | All imports resolve including `video_processor` and `video_summarization` |
| Audio detection (§8) | `has_audio=True` for audio video, `has_audio=False` for silent |
| Audio ingest (§9) | `audio_sources=1`, `silent_sources=0`, chunks created |
| Silent ingest (§10) | `silent_sources=1`, `audio_sources=0`, chunks created with `content_type=frame_window_captions` |
| Search (§11) | Results include both `transcript` and `frame_window_captions` content types |
| Brief summary (§13) | Overall summary is 1 paragraph |
| Standard summary (§14) | Overall summary is 2–3 paragraphs |
| Detailed summary (§15) | Overall summary is 3–4 structured paragraphs |

If any section fails, paste the error back into Claude Code with the
relevant knowledge base context and iterate.

---

## Workflow Summary

```
Part 1: Wire the MCP server
  1. Install Claude Code, radiant-rag-mcp, fastmcp
  2. Download config.yaml to ~/.radiant/
  3. Add MCP config to ~/.claude/settings.json
  4. Verify: claude mcp list
  5. Start session, run /mcp to confirm 10 tools
  6. Ingest your first documents
  7. Query and search from within Claude Code

Part 2: Generate the video extension
  1. Clone repo, open in Claude Code
  2. Ingest src/, spec, and doc files into RAG index
  3. Present spec — ask for summary before coding
  4. Implement section by section (9 stages, dependency order)
     pyproject.toml → config.py → video_processor.py → video_summarization.py
     → processor.py → __init__ files → app.py → server.py → notebook
  5. Run validation checklist after all sections complete
  6. Fix failures using RAG-retrieved context
  7. Commit
  8. Test in Colab with T4 GPU
```

---

## Troubleshooting Reference

### `radiant-mcp: command not found`
The entry point is not on the PATH that Claude Code uses. Use the full
absolute path in the `command` field:
```bash
which radiant-mcp   # copy this path into settings.json
```

### `ModuleNotFoundError: No module named 'radiant_rag_mcp'`
Claude Code is using a different Python than the one where you installed the
package. Set `command` to the full path of `radiant-mcp` inside your venv.

### Server times out at startup
The server pre-loads the embedding and reranking models on first start.
This can take 30–60 s on first run while model weights download. Subsequent
starts are instant. Set a generous `startup_timeout` if your Claude Code
version supports it.

### `Tool error: ValidationError: top_k — Unexpected keyword argument`
`query_knowledge` does not accept `top_k`. Only `search_knowledge` does.
Remove `top_k` from any `query_knowledge` calls.

### ChromaDB `_ensure_index` error on `clear_index`
Known library bug. The collection data IS deleted; only the re-initialisation
step fails. Call `rebuild_bm25` after clearing to restore a clean state.

### `ingest_video` tool not found after server.py update
Claude Code cached the old server. Restart the MCP server:
```
/mcp restart radiant-rag
```
or exit and restart the Claude Code session.

### Silent video produces zero chunks
The `ImageCaptioner` (VLM) was not initialised — either `vlm.enabled=false`
in config or the `_image_captioner` attribute is None. Check:
```bash
RADIANT_VLM_ENABLED=true radiant-mcp
```
and confirm the VLM model downloads on first start.
