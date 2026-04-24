"""
Radiant RAG MCP server.

Exposes eleven tools via the Model Context Protocol (MCP) using FastMCP 2.x with
stdio transport, compatible with Claude Code and Claude Desktop.

STDOUT PROHIBITION: The MCP stdio protocol uses stdout as the JSON-RPC channel.
Nothing in this file or any module it imports may write to stdout.  All logging
is directed to stderr.  The Rich console used by the underlying display helpers
is patched to stderr before the application is initialised.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Logging — MUST target stderr before any other import that might emit output
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastMCP imports
# ---------------------------------------------------------------------------
from fastmcp import FastMCP, Context  # noqa: E402

# ---------------------------------------------------------------------------
# Application import — deferred until after logging is configured
# ---------------------------------------------------------------------------
from radiant_rag_mcp.app import create_app  # noqa: E402



# ---------------------------------------------------------------------------
# Lifespan — all blocking I/O happens here, not at module import time
# ---------------------------------------------------------------------------
@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """
    Initialise RadiantRAG once at server start-up and tear it down on exit.

    The yielded dict is available in every tool via ``ctx.lifespan_context``.
    The ``hierarchical`` flag is mutable so that ``set_ingest_mode`` can
    update it for the lifetime of the server process.
    """
    # Redirect the Rich console used by display helpers to stderr so that
    # no stray writes corrupt the MCP stdio channel.
    try:
        import radiant_rag_mcp.ui.display as _display
        from rich.console import Console as _Console
        _display.console = _Console(file=sys.stderr, force_terminal=False)
    except Exception:
        pass  # non-fatal; display helpers are always called with show_* flags off

    logger.info("radiant-mcp: initialising RadiantRAG…")
    config_path = os.environ.get("RADIANT_CONFIG_PATH")
    app_instance = await asyncio.to_thread(create_app, config_path)
    logger.info("radiant-mcp: RadiantRAG ready")

    try:
        yield {
            "app": app_instance,
            # Mutable default for set_ingest_mode / ingest_documents
            "hierarchical": True,
        }
    finally:
        logger.info("radiant-mcp: shutting down")


# ---------------------------------------------------------------------------
# FastMCP server instance
# ---------------------------------------------------------------------------
mcp = FastMCP("radiant-rag", lifespan=app_lifespan)


# ---------------------------------------------------------------------------
# Helper — serialise a list of (StoredDoc, score) tuples
# ---------------------------------------------------------------------------
def _serialise_search_results(
    results: List[Any],
) -> List[Dict[str, Any]]:
    out = []
    for item in results:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            doc, score = item
        else:
            doc, score = item, 0.0
        out.append(
            {
                "doc_id": getattr(doc, "doc_id", str(doc)),
                "content": getattr(doc, "content", ""),
                "meta": getattr(doc, "meta", {}),
                "score": float(score),
            }
        )
    return out


# ===========================================================================
# Tool 1 — search_knowledge
# ===========================================================================
@mcp.tool()
async def search_knowledge(
    query: str,
    mode: str = "hybrid",
    top_k: int = 10,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Retrieve documents from the knowledge base without LLM generation.

    Performs pure vector/BM25/hybrid retrieval and returns ranked document
    chunks with scores.  Use this when you need raw context rather than a
    synthesised answer.

    Args:
        query: Search query text.
        mode:  Retrieval strategy — ``hybrid`` (default), ``dense``, or
               ``bm25``.
        top_k: Maximum number of results to return (default 10).

    Returns:
        ``{"query": str, "mode": str, "results": [{"doc_id", "content",
        "meta", "score"}, ...]}``
    """
    app = ctx.lifespan_context["app"]
    raw = await asyncio.to_thread(
        app.search,
        query,
        mode=mode,
        top_k=top_k,
        show_results=False,
    )
    return {
        "query": query,
        "mode": mode,
        "results": _serialise_search_results(raw[:top_k]),
    }


# ===========================================================================
# Tool 2 — query_knowledge
# ===========================================================================
@mcp.tool()
async def query_knowledge(
    question: str,
    mode: str = "hybrid",
    conversation_id: Optional[str] = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Answer a question using the full agentic RAG pipeline.

    Runs query planning, decomposition, dense + BM25 retrieval, RRF fusion,
    cross-encoder reranking, answer synthesis with citations, and optional
    critic-driven refinement.  Returns a structured result with the answer,
    confidence score, citations, and pipeline metrics.

    Args:
        question:        The question to answer.
        mode:            Retrieval strategy — ``hybrid`` (default), ``dense``,
                         or ``bm25``.
        conversation_id: Pass a conversation ID returned by
                         ``start_conversation`` to continue a multi-turn
                         session; omit for a stateless single-turn query.

    Returns:
        ``PipelineResult.to_dict()`` — includes ``answer``, ``confidence``,
        ``cited_answer``, ``citations``, ``metrics``, and more.
    """
    app = ctx.lifespan_context["app"]
    result = await asyncio.to_thread(
        app.query_raw,
        question,
        conversation_id=conversation_id,
        retrieval_mode=mode,
    )
    return result.to_dict()


# ===========================================================================
# Tool 3 — simple_query
# ===========================================================================
@mcp.tool()
async def simple_query(
    question: str,
    top_k: int = 5,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Answer a question using a lightweight retrieval-and-synthesis pipeline.

    Skips the full agentic pipeline (no planning, decomposition, reranking,
    or critic loop) for lower latency.  Suitable for straightforward factual
    queries where speed matters more than deep reasoning.

    This tool is stateless — there is no ``conversation_id`` parameter.

    Args:
        question: The question to answer.
        top_k:    Number of document chunks to retrieve (default 5).

    Returns:
        ``{"question": str, "answer": str}``
    """
    app = ctx.lifespan_context["app"]
    answer = await asyncio.to_thread(app.simple_query, question, top_k=top_k)
    return {"question": question, "answer": answer}


# ===========================================================================
# Tool 4 — start_conversation
# ===========================================================================
@mcp.tool()
async def start_conversation(ctx: Context = None) -> Dict[str, Any]:
    """
    Create a new conversation ID for multi-turn query sessions.

    Pass the returned ``conversation_id`` to subsequent ``query_knowledge``
    calls to maintain context across turns.  Each call creates an independent
    conversation; call this once per session.

    Returns:
        ``{"conversation_id": str}``
    """
    app = ctx.lifespan_context["app"]
    conversation_id = await asyncio.to_thread(app.start_conversation)
    return {"conversation_id": conversation_id}


# ===========================================================================
# Tool 5 — ingest_documents
# ===========================================================================
@mcp.tool()
async def ingest_documents(
    paths: List[str],
    hierarchical: Optional[bool] = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Index local files or directories into the knowledge base.

    Supports all document types handled by Unstructured (PDF, DOCX, HTML,
    Markdown, plain text, images with VLM captioning, code files, etc.).
    Directories are traversed recursively.

    Hierarchical storage splits each document into parent (full) and child
    (smaller) chunks, enabling auto-merging retrieval for better context
    windows.  Flat storage uses a single chunk size.

    Args:
        paths:        One or more file paths or directory paths to index.
        hierarchical: ``True`` (default) for hierarchical parent/child chunk
                      storage; ``False`` for flat storage.  If omitted, uses
                      the mode set by ``set_ingest_mode`` for this session.

    Returns:
        Ingestion statistics dict with ``files_processed``, ``chunks_created``,
        ``documents_stored``, ``files_failed``, and ``errors``.
    """
    lc = ctx.lifespan_context
    app = lc["app"]
    use_hierarchical = hierarchical if hierarchical is not None else lc["hierarchical"]
    stats = await asyncio.to_thread(
        app.ingest_documents,
        paths,
        show_progress=False,
        use_hierarchical=use_hierarchical,
    )
    return stats


# ===========================================================================
# Tool 6 — ingest_url
# ===========================================================================
@mcp.tool()
async def ingest_url(
    url: str,
    crawl_depth: Optional[int] = None,
    max_pages: Optional[int] = None,
    no_crawl: bool = False,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Index a URL or GitHub repository into the knowledge base.

    Automatically detects GitHub repository URLs and uses the specialised
    GitHub crawler (fetches all Markdown and code files via the API).  For
    regular web URLs the crawler follows internal links up to ``crawl_depth``
    levels deep.

    Args:
        url:         The URL to index.  Supports https:// web pages and
                     https://github.com/<owner>/<repo> repositories.
        crawl_depth: Maximum link-follow depth for web URLs.  ``None`` uses
                     the ``config.yaml`` default.  Set to ``0`` to disable
                     crawling (equivalent to ``no_crawl=True``).
        max_pages:   Maximum pages to crawl.  ``None`` uses the config
                     default.
        no_crawl:    If ``True``, fetch only the single URL without following
                     any links.

    Returns:
        Ingestion statistics dict with ``urls_crawled``, ``files_processed``,
        ``chunks_created``, ``documents_stored``, and ``errors``.
    """
    app = ctx.lifespan_context["app"]
    effective_depth = 0 if no_crawl else crawl_depth
    stats = await asyncio.to_thread(
        app.ingest_urls,
        [url],
        show_progress=False,
        crawl_depth=effective_depth,
        max_pages=max_pages,
    )
    return stats


# ===========================================================================
# Tool 7 — get_index_stats
# ===========================================================================
@mcp.tool()
async def get_index_stats(ctx: Context = None) -> Dict[str, Any]:
    """
    Return document counts and system health for the knowledge base.

    Merges the output of the stats collector (query metrics, latency
    histograms) and the health check (vector index info, BM25 index info,
    conversation manager status).

    Returns:
        Combined dict with ``health`` (vector_index, bm25_index, redis,
        conversation) and ``metrics`` (query counts, latency histograms) keys.
    """
    app = ctx.lifespan_context["app"]
    stats = await asyncio.to_thread(app.get_stats)
    health = await asyncio.to_thread(app.check_health)
    # get_stats already calls check_health internally; merge for completeness
    return {**stats, "health": health}


# ===========================================================================
# Tool 8 — clear_index
# ===========================================================================
@mcp.tool()
async def clear_index(
    confirm: bool = False,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Delete all indexed documents from the knowledge base.

    This is a destructive, irreversible operation.  You must pass
    ``confirm=True`` explicitly; the tool returns a cancellation message
    if ``confirm`` is ``False`` or omitted.

    Args:
        confirm: Must be ``True`` to proceed with deletion.

    Returns:
        ``{"cleared": bool, "message": str}``
    """
    if not confirm:
        return {
            "cleared": False,
            "message": (
                "Index was NOT cleared.  Pass confirm=True to proceed.  "
                "This operation deletes all indexed documents and cannot be undone."
            ),
        }
    app = ctx.lifespan_context["app"]
    success = await asyncio.to_thread(app.clear_index)
    return {
        "cleared": success,
        "message": "Index cleared successfully." if success else "Index clear failed; check server logs.",
    }


# ===========================================================================
# Tool 9 — rebuild_bm25
# ===========================================================================
@mcp.tool()
async def rebuild_bm25(ctx: Context = None) -> Dict[str, Any]:
    """
    Rebuild the BM25 sparse index from the current vector store contents.

    Use this after bulk ingestion without BM25 enabled, after an index
    corruption, or any time hybrid/BM25 retrieval returns unexpected results.
    Equivalent to the ``radiant rebuild-bm25`` CLI command.

    Returns:
        ``{"status": "ok", "message": str, "documents_indexed": int}``
    """
    app = ctx.lifespan_context["app"]
    count = await asyncio.to_thread(app.rebuild_bm25_index)
    return {
        "status": "ok",
        "message": f"BM25 index rebuilt successfully from vector store.",
        "documents_indexed": count,
    }


# ===========================================================================
# Tool 10 — set_ingest_mode
# ===========================================================================
@mcp.tool()
async def set_ingest_mode(
    hierarchical: bool,
    ctx: Context = None,
) -> Dict[str, Any]:
    """
    Set the default ingestion storage mode for this server session.

    Controls whether ``ingest_documents`` uses hierarchical (parent/child
    chunks, recommended) or flat storage by default.  The preference persists
    for the lifetime of the server process without requiring a config file
    change.  Individual ``ingest_documents`` calls can override this per-call
    by passing the ``hierarchical`` parameter explicitly.

    Args:
        hierarchical: ``True`` for hierarchical parent/child chunk storage
                      (recommended for large documents and multi-hop queries);
                      ``False`` for flat single-level chunk storage.

    Returns:
        ``{"hierarchical": bool, "message": str}``
    """
    ctx.lifespan_context["hierarchical"] = hierarchical
    mode_name = "hierarchical" if hierarchical else "flat"
    return {
        "hierarchical": hierarchical,
        "message": f"Ingest mode set to {mode_name!r}.  All subsequent ingest_documents calls will use {mode_name} storage by default.",
    }


# ===========================================================================
# Tool 11 — ingest_video
# ===========================================================================
@mcp.tool()
async def ingest_video(
    ctx: Context,
    sources: List[str],
    hierarchical: bool = True,
    child_chunk_size: int = 512,
    child_chunk_overlap: int = 50,
    enable_frame_captioning: bool = False,
    force_frame_analysis: bool = False,
    summarize: bool = False,
) -> Dict[str, Any]:
    """
    Ingest one or more video files or remote video URLs into the RAG knowledge base.

    Supported local video formats:
      .mp4, .mkv, .webm, .mov, .avi, .m4v, .flv, .wmv, .ts

    Supported remote URLs (via yt-dlp):
      YouTube, Twitter/X, TikTok, Instagram, Vimeo, Twitch, Reddit, and 1 000+ others

    Processing path is chosen automatically:
      - Video with audio:          Whisper transcription → transcript chunks
      - Silent video:              VLM frame-window analysis → caption chunks
      - force_frame_analysis=True: always use VLM regardless of audio presence

    For local audio-only files (.mp3, .wav, .m4a, .flac, etc.) use ingest_audio instead.

    Args:
        sources:               One or more video file paths or URLs.
        hierarchical:          ``True`` (default) for hierarchical parent/child
                               chunk storage; ``False`` for flat storage.
        child_chunk_size:      Size of child chunks (default 512).
        child_chunk_overlap:   Overlap between child chunks (default 50).
        enable_frame_captioning: Enable per-frame VLM captioning in addition
                               to window-level captions.
        force_frame_analysis:  Always use VLM frame analysis even when audio
                               is present.
        summarize:             Generate a VideoSummaryResult for each source
                               and include it in the returned stats.

    Returns:
        Ingestion statistics dict with ``sources_processed``, ``sources_failed``,
        ``chunks_created``, ``documents_stored``, ``silent_sources``,
        ``audio_sources``, ``summaries``, and ``errors``.
    """
    app = ctx.lifespan_context["app"]
    result = await asyncio.to_thread(
        app.ingest_videos,
        sources,
        show_progress=False,
        use_hierarchical=hierarchical,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
        enable_frame_captioning=enable_frame_captioning,
        force_frame_analysis=force_frame_analysis,
        summarize=summarize,
    )
    return result


# ===========================================================================
# Tool 12 — ingest_audio
# ===========================================================================
@mcp.tool()
async def ingest_audio(
    ctx: Context,
    sources: List[str],
    hierarchical: bool = True,
    child_chunk_size: int = 512,
    child_chunk_overlap: int = 50,
    summarize: bool = False,
) -> Dict[str, Any]:
    """
    Ingest one or more local audio-only files into the RAG knowledge base
    via Whisper transcription.

    Supported formats:
      .mp3, .wav, .m4a, .flac, .ogg, .aac, .opus, .wma, .aiff

    Every source is transcribed with Whisper and stored as transcript chunks.
    Frame analysis is never applied (audio-only files have no video frames).

    For video files or remote URLs use ingest_video instead.

    Args:
        sources:             One or more local audio file paths.
        hierarchical:        ``True`` (default) for hierarchical parent/child
                             chunk storage; ``False`` for flat storage.
        child_chunk_size:    Size of child chunks (default 512).
        child_chunk_overlap: Overlap between child chunks (default 50).
        summarize:           Generate a VideoSummaryResult (title, chapters,
                             key topics) for each source and include it in
                             the returned stats.

    Returns:
        Ingestion statistics dict with ``sources_processed``, ``sources_failed``,
        ``chunks_created``, ``documents_stored``, ``audio_sources``,
        ``summaries``, and ``errors``.
    """
    app = ctx.lifespan_context["app"]
    result = await asyncio.to_thread(
        app.ingest_audio,
        sources,
        use_hierarchical=hierarchical,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
        summarize=summarize,
    )
    return result


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> None:
    """
    CLI entry point for the radiant-mcp command.

    Transport is selected at runtime via ``RADIANT_TRANSPORT`` (default
    ``stdio``).  The available values are:

    * ``stdio``  — JSON-RPC over stdin/stdout; required for Claude Code and
      Claude Desktop.
    * ``sse``    — Server-Sent Events HTTP server; for remote SSE clients.
    * ``http``   — Streamable HTTP server; for modern remote HTTP clients.

    ``RADIANT_HOST`` (default ``127.0.0.1``) and ``RADIANT_PORT`` (default
    ``8000``) set the bind address for ``sse`` and ``http`` transports.

    An optional ``--config`` / ``-c`` argument (or ``RADIANT_CONFIG_PATH``
    env var) specifies the path to a ``config.yaml`` file.
    """
    parser = argparse.ArgumentParser(
        description="Radiant RAG MCP server — supports stdio, SSE, and HTTP transports",
    )
    parser.add_argument(
        "--config", "-c",
        metavar="PATH",
        help="Path to config.yaml (overrides RADIANT_CONFIG_PATH env var)",
    )
    args, _ = parser.parse_known_args()

    if args.config:
        os.environ.setdefault("RADIANT_CONFIG_PATH", args.config)

    # Map canonical MCP env vars to the internal RADIANT_<SECTION>_<KEY> names
    # that config.py expects.  This allows the Claude Code / Desktop config
    # block to use the documented names without knowing internal config layout.
    _env_shims = {
        "RADIANT_OLLAMA_BASE_URL": "RADIANT_LLM_BACKEND_BASE_URL",
        "RADIANT_OLLAMA_API_KEY": "RADIANT_LLM_BACKEND_API_KEY",
    }
    for src, dst in _env_shims.items():
        if src in os.environ:
            os.environ.setdefault(dst, os.environ[src])

    transport = os.environ.get("RADIANT_TRANSPORT", "stdio")
    host = os.environ.get("RADIANT_HOST", "127.0.0.1")
    port = int(os.environ.get("RADIANT_PORT", "8000"))

    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "sse":
        mcp.run(transport="sse", host=host, port=port)
    elif transport == "http":
        mcp.run(transport="http", host=host, port=port)
    else:
        raise ValueError(
            f"Unknown transport {transport!r}.  "
            "Set RADIANT_TRANSPORT to 'stdio', 'sse', or 'http'."
        )


if __name__ == "__main__":
    main()
