"""
Video summarization agent for Radiant Agentic RAG.

Produces structured chapter summaries and an overall summary from the
IngestedChunk objects emitted by VideoProcessor (both transcript and
silent-video frame-window paths).  Uses a map-reduce LLM strategy:

  1. Group chunks into chapters by time gap and max duration.
  2. Summarize each chapter independently (chapter-level map).
  3. Reduce chapter summaries into a single overall summary.
  4. Extract key topics from the overall summary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from radiant_rag_mcp.agents.base_agent import AgentCategory, LLMAgent
from radiant_rag_mcp.config import VideoSummarizationConfig

if TYPE_CHECKING:
    from radiant_rag_mcp.llm.client import LLMClient
    from radiant_rag_mcp.ingestion.processor import IngestedChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Preset paragraph-count profiles
# (chapter_min, chapter_max, overall_min, overall_max)
# ---------------------------------------------------------------------------
_PRESET_SHAPES: Dict[str, tuple] = {
    "brief":    (1, 1, 1, 1),
    "standard": (1, 2, 2, 3),
    "detailed": (2, 3, 3, 4),
}
# Matches VideoSummarizationConfig field defaults — used to detect explicit
# user overrides (any value differing from these was set intentionally).
_STANDARD_SHAPE = (1, 2, 2, 3)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VideoChapter:
    """Summary of a single temporal chapter within a video."""

    index: int          # 0-based chapter index
    start: float        # Chapter start time in seconds
    end: float          # Chapter end time in seconds
    title: str          # Short chapter title (empty string when not generated)
    summary: str        # LLM-generated chapter summary text
    source_type: str    # "transcript" or "frame_window_captions"


@dataclass
class VideoSummaryResult:
    """Full summarization result for a video source."""

    source: str                      # Original URL or file path
    title: str                       # Video title
    duration_seconds: float          # Total duration in seconds
    language: str                    # Detected language code (empty for silent)
    is_silent: bool                  # True when sourced from frame captions
    summary: str                     # Overall summary text
    key_topics: List[str]            # 5-8 key topics / themes
    chapters: List[VideoChapter]     # Ordered chapter summaries
    total_chunks: int                # Number of input chunks processed
    model_used: str                  # LLM model name, or "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class VideoSummarizationAgent(LLMAgent):
    """
    Summarize video content into chapters and an overall summary.

    Supports both spoken-audio transcripts and silent-video frame-window
    caption sequences.  Uses a map-reduce strategy: each chapter is
    summarized independently, then all chapter summaries are reduced into
    a single overall summary and key-topic list.
    """

    def __init__(
        self,
        llm: "LLMClient",
        config: Optional[VideoSummarizationConfig] = None,
    ) -> None:
        """
        Initialize the VideoSummarizationAgent.

        Args:
            llm: LLM client used for all generation calls.
            config: Optional VideoSummarizationConfig; falls back to the
                standard-preset defaults when not provided.
        """
        super().__init__(llm=llm)
        self._cfg = config or VideoSummarizationConfig()

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the agent's unique name."""
        return "VideoSummarizationAgent"

    @property
    def category(self) -> AgentCategory:
        """Return the agent's category."""
        return AgentCategory.GENERATION

    @property
    def description(self) -> str:
        """Return a human-readable description of the agent."""
        return (
            "Summarizes video transcripts and silent-video frame captions "
            "into structured chapters and an overall summary"
        )

    def _execute(
        self,
        query: str,
        context: List["IngestedChunk"],
        **kwargs: Any,
    ) -> VideoSummaryResult:
        """
        Execute the agent via the BaseAgent run() lifecycle.

        Args:
            query: Video source URL or file path.
            context: IngestedChunk objects produced by VideoProcessor.
            **kwargs: Forwarded to summarize_video (e.g. chapter_gap_seconds).

        Returns:
            VideoSummaryResult.
        """
        return self.summarize_video(query, context, **kwargs)

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def summarize_video(
        self,
        source: str,
        chunks: List["IngestedChunk"],
        chapter_gap_seconds: Optional[float] = None,
        max_summary_tokens: int = 2048,
    ) -> VideoSummaryResult:
        """
        Summarize a video from its ingested chunks.

        Args:
            source: Original video URL or file path.
            chunks: IngestedChunk objects from VideoProcessor.  May be a mix
                of transcript or frame_window_captions chunks; the first
                chunk's ``content_type`` meta key determines which prompting
                strategy is used.
            chapter_gap_seconds: Override for the chapter-gap threshold.
                Defaults to ``config.chapter_gap_seconds``.
            max_summary_tokens: Token-budget hint (informational; not enforced
                by the agent itself).

        Returns:
            VideoSummaryResult with per-chapter and overall summaries,
            key topics, and provenance metadata.
        """
        cfg = self._cfg

        # ------------------------------------------------------------------
        # Step 1: Resolve effective paragraph counts
        # Preset values provide the base; explicit config overrides take
        # precedence when they differ from the VideoSummarizationConfig defaults.
        # ------------------------------------------------------------------
        preset_c_min, preset_c_max, preset_o_min, preset_o_max = _PRESET_SHAPES.get(
            cfg.summary_detail, _STANDARD_SHAPE
        )
        std_c_min, std_c_max, std_o_min, std_o_max = _STANDARD_SHAPE

        c_min = cfg.chapter_paragraphs_min if cfg.chapter_paragraphs_min != std_c_min else preset_c_min
        c_max = cfg.chapter_paragraphs_max if cfg.chapter_paragraphs_max != std_c_max else preset_c_max
        o_min = cfg.overall_paragraphs_min if cfg.overall_paragraphs_min != std_o_min else preset_o_min
        o_max = cfg.overall_paragraphs_max if cfg.overall_paragraphs_max != std_o_max else preset_o_max

        # ------------------------------------------------------------------
        # Step 2: Sort chunks by start_time
        # ------------------------------------------------------------------
        sorted_chunks = sorted(
            chunks, key=lambda c: float(c.meta.get("start_time", 0.0))
        )

        if not sorted_chunks:
            logger.warning(
                "VideoSummarizationAgent: no chunks provided for source=%s", source
            )
            return VideoSummaryResult(
                source=source,
                title="",
                duration_seconds=0.0,
                language="",
                is_silent=False,
                summary="",
                key_topics=[],
                chapters=[],
                total_chunks=0,
                model_used=self._get_model_name(),
            )

        # ------------------------------------------------------------------
        # Step 3: Detect source_type from first chunk meta content_type
        # ------------------------------------------------------------------
        first_meta = sorted_chunks[0].meta
        content_type = first_meta.get("content_type", "transcript")
        source_type = (
            "frame_window_captions"
            if content_type == "frame_window_captions"
            else "transcript"
        )
        is_silent = source_type == "frame_window_captions"

        title = str(first_meta.get("title", source))
        duration_seconds = float(first_meta.get("duration", 0.0))
        language = "" if is_silent else str(first_meta.get("language", ""))
        hint = cfg.content_type_hint

        # ------------------------------------------------------------------
        # Step 4: Chapter grouping
        # ------------------------------------------------------------------
        gap = chapter_gap_seconds if chapter_gap_seconds is not None else cfg.chapter_gap_seconds
        max_dur = cfg.max_chapter_duration_seconds

        chapter_groups: List[List["IngestedChunk"]] = []
        current: List["IngestedChunk"] = []
        chapter_start = 0.0
        prev_end = 0.0

        for chunk in sorted_chunks:
            start = float(chunk.meta.get("start_time", 0.0))
            end = float(chunk.meta.get("end_time", start))

            if not current:
                current = [chunk]
                chapter_start = start
            else:
                time_gap = start - prev_end
                projected_dur = end - chapter_start
                if time_gap > gap or projected_dur > max_dur:
                    chapter_groups.append(current)
                    current = [chunk]
                    chapter_start = start
                else:
                    current.append(chunk)

            prev_end = end

        if current:
            chapter_groups.append(current)

        logger.info(
            "VideoSummarizationAgent: %d chunks → %d chapters (source=%s, type=%s)",
            len(sorted_chunks), len(chapter_groups), source, source_type,
        )

        # ------------------------------------------------------------------
        # Step 5: Summarise each chapter
        # ------------------------------------------------------------------
        chapters: List[VideoChapter] = []
        total_chapters = len(chapter_groups)

        for idx, group in enumerate(chapter_groups):
            ch_start = float(group[0].meta.get("start_time", 0.0))
            ch_end = float(group[-1].meta.get("end_time", ch_start))

            if source_type == "transcript":
                joined = "\n".join(
                    f"[{c.meta.get('start_time', 0):.0f}s-"
                    f"{c.meta.get('end_time', 0):.0f}s] {c.content}"
                    for c in group
                )
                sys_prompt = (
                    f"You are a video content analyst. Summarise the following "
                    f"transcript segments from one chapter of a video. Write your "
                    f"summary in {c_min}-{c_max} paragraph(s). "
                    f"Paragraph 1: describe the main events, topics, and information "
                    f"in sequence. "
                    f"Paragraph 2 (if more than 1): what is established, demonstrated, "
                    f"or concluded by the end of this chapter. "
                    f"Include specific facts and details. Be specific: preserve names, "
                    f"quantities, technical terms, and key claims."
                )
                user_prompt = (
                    f"Chapter {idx + 1} of {total_chapters} "
                    f"[{ch_start:.0f}s - {ch_end:.0f}s]\n"
                    f"Source type: spoken transcript\n"
                    f"Content type hint: {hint}\n\n"
                    f"{joined}\n\n"
                    f"Write the chapter summary now."
                )
            else:
                joined = "\n\n".join(
                    f"[{c.meta.get('start_time', 0):.0f}s-"
                    f"{c.meta.get('end_time', 0):.0f}s] {c.content}"
                    for c in group
                )
                sys_prompt = (
                    f"You are a video content analyst summarising a series of visual "
                    f"scene descriptions extracted from a silent video. Each description "
                    f"covers a short time window. Your task is to synthesise them into a "
                    f"coherent narrative in {c_min}-{c_max} paragraph(s). "
                    f"Paragraph 1: describe what is happening visually across this chapter "
                    f"in sequence — subjects, actions, setting, transitions. "
                    f"Paragraph 2 (if more than 1): what changes or is established visually "
                    f"by the end of this chapter. "
                    f"Note any data, text, or indicators visible. Write in present tense. "
                    f"Be specific."
                )
                user_prompt = (
                    f"Chapter {idx + 1} of {total_chapters} "
                    f"[{ch_start:.0f}s - {ch_end:.0f}s]\n"
                    f"Source type: silent video frame descriptions\n"
                    f"Content type hint: {hint}\n\n"
                    f"{joined}\n\n"
                    f"Write the chapter summary now."
                )

            try:
                chapter_summary = self._chat(sys_prompt, user_prompt)
            except Exception as exc:
                logger.warning(
                    "Chapter %d/%d summarization failed: %s",
                    idx + 1, total_chapters, exc,
                )
                chapter_summary = " ".join(c.content for c in group)[:500]

            chapters.append(VideoChapter(
                index=idx,
                start=ch_start,
                end=ch_end,
                title="",
                summary=chapter_summary,
                source_type=source_type,
            ))

        # ------------------------------------------------------------------
        # Step 6: Overall summary (map-reduce over chapter summaries)
        # ------------------------------------------------------------------
        numbered_chapters = "\n\n".join(
            f"Chapter {ch.index + 1} [{ch.start:.0f}s-{ch.end:.0f}s]:\n{ch.summary}"
            for ch in chapters
        )

        overall_sys = (
            f"You are a video content analyst. Write a structured overall summary "
            f"of a video based on the chapter summaries provided. Write in "
            f"{o_min}-{o_max} structured paragraph(s): "
            f"Paragraph 1 (Overview — always): what this video is, its topic, "
            f"duration, and primary subject. "
            f"Paragraph 2 (Content — always): key events, techniques, information, "
            f"or visual content covered in approximate chronological order. "
            f"Paragraph 3 (Detail — include if {o_max}>=3 and video > 5 min): "
            f"noteworthy sequences, turning points, specific timestamps, or dense "
            f"content sections. "
            f"Paragraph 4 (Takeaway — include if {o_max}==4 and hint is "
            f"instructional, scientific, or documentary): what a viewer would learn "
            f"or conclude. "
            f"Omit paragraphs 3 and 4 if they do not apply."
        )
        overall_user = (
            f"Video: {title}\n"
            f"Duration: {duration_seconds:.0f}s\n"
            f"Source type: {source_type}\n"
            f"Content type hint: {hint}\n\n"
            f"Chapter summaries:\n{numbered_chapters}\n\n"
            f"Write the overall summary now."
        )

        try:
            overall_summary = self._chat(overall_sys, overall_user)
        except Exception as exc:
            logger.warning("Overall summary generation failed: %s", exc)
            overall_summary = " ".join(ch.summary for ch in chapters)[:1000]

        # ------------------------------------------------------------------
        # Step 7: Key topics extraction
        # ------------------------------------------------------------------
        topics_sys = (
            "Extract 5-8 key topics or themes from the video summary below. "
            "Return a JSON array of short strings (2-5 words each). "
            "Return ONLY the JSON array, no preamble."
        )
        key_topics: List[str] = []
        topics_raw, topics_resp = self._llm.chat_json(
            system=topics_sys,
            user=overall_summary,
            default=None,
            expected_type=list,
        )
        if isinstance(topics_raw, list):
            key_topics = [str(t) for t in topics_raw if t]
        else:
            # Fallback: split the raw response text on newlines
            raw_text = topics_resp.content if topics_resp else ""
            key_topics = [
                line.strip().lstrip("-•*0123456789. ").strip()
                for line in raw_text.splitlines()
                if line.strip()
            ]
            logger.debug(
                "Key-topics JSON parse failed; extracted %d topics via newline split.",
                len(key_topics),
            )

        # ------------------------------------------------------------------
        # Step 8: Return result
        # ------------------------------------------------------------------
        logger.info(
            "VideoSummarizationAgent: summarized '%s' → %d chapters, "
            "%d topics, model=%s",
            title, len(chapters), len(key_topics), self._get_model_name(),
        )
        return VideoSummaryResult(
            source=source,
            title=title,
            duration_seconds=duration_seconds,
            language=language,
            is_silent=is_silent,
            summary=overall_summary,
            key_topics=key_topics,
            chapters=chapters,
            total_chunks=len(chunks),
            model_used=self._get_model_name(),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model_name(self) -> str:
        """
        Return the model name from the underlying LLM client.

        Returns:
            Model name string, or "unknown" if not determinable.
        """
        return (
            getattr(getattr(self._llm, "_ollama_config", None), "chat_model", None)
            or "unknown"
        )
