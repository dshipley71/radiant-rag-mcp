# Claude Code Task: Add Video Summarization & RAG to radiant-rag-mcp

## Goal

Extend `radiant-rag-mcp` to support **video ingestion, summarization, and retrieval** for:
- Local video files (`.mp4`, `.mkv`, `.webm`, `.mov`, `.avi`, `.m4v`)
- YouTube URLs (and any `yt-dlp`-compatible source: Vimeo, Twitch clips, etc.)
- **Silent videos** (no audio stream) — fully handled via VLM frame-window analysis

The pipeline selects its processing path automatically:

```
process_video(source)
    |
    +-- has audio? --YES--> Whisper transcription --> transcript chunks
    |                                                 content_type="transcript"
    |
    +-- no audio  ---------> frame-window VLM analysis --> caption chunks
                              (also: forced via force_frame_analysis=True)
                              content_type="frame_window_captions"
```

Summaries are produced by a new `VideoSummarizationAgent` with three verbosity
presets — `brief`, `standard`, `detailed` — applied independently at the window,
chapter, and overall levels.

---

## Architecture Overview

Follow the **exact same patterns** used by the existing `ImageCaptioner` /
`WebCrawler` pipeline. New work touches these areas:

| File | Action |
|---|---|
| `src/radiant_rag_mcp/ingestion/video_processor.py` | **NEW** |
| `src/radiant_rag_mcp/agents/video_summarization.py` | **NEW** |
| `src/radiant_rag_mcp/config.py` | **MODIFY** — add `VideoProcessorConfig`, `VideoSummarizationConfig` |
| `src/radiant_rag_mcp/ingestion/processor.py` | **MODIFY** — route video extensions |
| `src/radiant_rag_mcp/ingestion/__init__.py` | **MODIFY** — export new symbols |
| `src/radiant_rag_mcp/agents/__init__.py` | **MODIFY** — export new agent |
| `src/radiant_rag_mcp/app.py` | **MODIFY** — add `ingest_videos()` |
| `src/radiant_rag_mcp/server.py` | **MODIFY** — add `ingest_video` MCP tool |
| `pyproject.toml` | **MODIFY** — add `video` optional-dependency group |
| `notebooks/radiant_rag_mcp_video_test.ipynb` | **NEW** — Colab test notebook |

---

## Detailed Instructions

### 1 · `pyproject.toml` — add `[project.optional-dependencies] video`

```toml
[project.optional-dependencies]
video = [
    "yt-dlp>=2024.1.0",
    "faster-whisper>=1.0.0",
    "openai-whisper>=20231117",
    "opencv-python-headless>=4.8.0",
    "Pillow>=10.0.0",
    "ffmpeg-python>=0.2.0",
    "numpy>=1.24.0",
    "scenedetect[opencv]>=0.6.3",
]
```

Also add `"yt-dlp>=2024.1.0"` and `"faster-whisper>=1.0.0"` to the base
`dependencies` list.

---

### 2 · `src/radiant_rag_mcp/config.py` — add two new dataclasses

#### 2a · `VideoProcessorConfig`

Insert after `VLMCaptionerConfig` (around line 175).
The config section name in `config.yaml` is `video:`.
The env-var prefix is `RADIANT_VIDEO_*`.

```python
@dataclass(frozen=True)
class VideoProcessorConfig:
    """Configuration for video ingestion (transcription + frame-window analysis)."""

    # -- Transcription (audio videos) ----------------------------------------
    # Whisper model: "tiny", "base", "small", "medium", "large-v3"
    whisper_model: str = "base"
    whisper_device: str = "auto"
    whisper_compute_type: str = "int8"
    whisper_language: str = "auto"

    # -- YouTube / yt-dlp ----------------------------------------------------
    max_duration_seconds: int = 3600
    ytdlp_format: str = "bestaudio/best"
    ytdlp_format_video: str = "bestvideo[ext=mp4]/bestvideo/best"
    download_dir: Optional[str] = None
    cleanup_after_ingest: bool = True

    # -- Frame-window analysis -----------------------------------------------
    enable_silent_video_analysis: bool = True
    enable_frame_captioning: bool = False

    # -- Sliding window -------------------------------------------------------
    window_duration_seconds: float = 10.0
    window_overlap_seconds: float = 2.0
    frames_per_window: int = 3
    max_windows: int = 0

    # -- Scene change detection ----------------------------------------------
    enable_scene_change_detection: bool = True
    scene_change_threshold: float = 0.25
    scene_change_min_gap_seconds: float = 2.0

    # -- Filmstrip tiling ----------------------------------------------------
    filmstrip_tile_width: int = 480
    filmstrip_tile_height: int = 270

    # -- Transcript chunking (audio path) ------------------------------------
    chunk_duration_seconds: int = 60
    chunk_overlap_seconds: int = 10
```

#### 2b · `VideoSummarizationConfig`

Insert immediately after `VideoProcessorConfig`.
The config section name is `video_summarization:`.
The env-var prefix is `RADIANT_VIDEO_SUMMARIZATION_*`.

```python
@dataclass(frozen=True)
class VideoSummarizationConfig:
    """
    Controls verbosity at three independent levels:

    - Window captions   : text embedded in the RAG index; kept concise for
                          retrieval precision.  Length is sentence-count only.
    - Chapter summaries : human-readable; paragraph count scales with preset.
    - Overall summary   : human-readable; follows a fixed schema by preset.

    Presets:
      brief    -- fast, compact; good for cataloguing large libraries.
      standard -- balanced; default for most use cases.
      detailed -- maximum fidelity; best for research / analysis.
    """

    # brief | standard | detailed
    summary_detail: str = "standard"

    # Window caption target (sentences).
    # Retrieval precision degrades above ~5 sentences per chunk.
    window_caption_sentences: int = 4

    # Chapter summary targets (paragraphs):
    #   brief:    1 paragraph  (min=1, max=1)
    #   standard: 1-2 paragraphs
    #   detailed: 2-3 paragraphs
    chapter_paragraphs_min: int = 1
    chapter_paragraphs_max: int = 2

    # Overall summary targets (paragraphs):
    #   brief:    1 paragraph
    #   standard: 2-3 paragraphs
    #   detailed: 3-4 structured paragraphs
    overall_paragraphs_min: int = 2
    overall_paragraphs_max: int = 3

    # Hint to the LLM about the nature of the content.
    # Shapes what details to emphasise in summaries.
    # Values: "general" | "instructional" | "surveillance" | "scientific" | "documentary"
    content_type_hint: str = "general"

    # Chapter gap: a new chapter begins when the gap between chunk
    # end_time and the next start_time exceeds this value (seconds).
    chapter_gap_seconds: float = 120.0

    # Maximum chapter duration (seconds) before a forced chapter break.
    max_chapter_duration_seconds: float = 300.0
```

**Preset-to-field mapping** (apply this logic in `VideoSummarizationAgent.summarize_video()`):

| Preset | chapter_paragraphs_min | chapter_paragraphs_max | overall_paragraphs_min | overall_paragraphs_max |
|---|---|---|---|---|
| `brief` | 1 | 1 | 1 | 1 |
| `standard` | 1 | 2 | 2 | 3 |
| `detailed` | 2 | 3 | 3 | 4 |

If `summary_detail` is set, override `chapter_paragraphs_*` and `overall_paragraphs_*`
from the table above at the start of `summarize_video()` — explicit field values always
take precedence over the preset, so callers can fine-tune without inventing a new preset.

#### 2c · Wire both configs into `AppConfig` and `load_config()`

**`AppConfig` class** — add two new fields (class is `@dataclass(frozen=True)`):

```python
    video: VideoProcessorConfig
    video_summarization: VideoSummarizationConfig
```

**`load_config()` function** — add two loading blocks using the exact same
`_get_config_value(data, section, key, default, parser)` pattern used for `vlm:`:

```python
    video = VideoProcessorConfig(
        whisper_model=_get_config_value(data, "video", "whisper_model", "base"),
        whisper_device=_get_config_value(data, "video", "whisper_device", "auto"),
        whisper_compute_type=_get_config_value(data, "video", "whisper_compute_type", "int8"),
        whisper_language=_get_config_value(data, "video", "whisper_language", "auto"),
        max_duration_seconds=_get_config_value(data, "video", "max_duration_seconds", 3600, _parse_int),
        ytdlp_format=_get_config_value(data, "video", "ytdlp_format", "bestaudio/best"),
        ytdlp_format_video=_get_config_value(data, "video", "ytdlp_format_video", "bestvideo[ext=mp4]/bestvideo/best"),
        download_dir=_get_config_value(data, "video", "download_dir", None) or None,
        cleanup_after_ingest=_get_config_value(data, "video", "cleanup_after_ingest", True, _parse_bool),
        enable_silent_video_analysis=_get_config_value(data, "video", "enable_silent_video_analysis", True, _parse_bool),
        enable_frame_captioning=_get_config_value(data, "video", "enable_frame_captioning", False, _parse_bool),
        window_duration_seconds=_get_config_value(data, "video", "window_duration_seconds", 10.0, _parse_float),
        window_overlap_seconds=_get_config_value(data, "video", "window_overlap_seconds", 2.0, _parse_float),
        frames_per_window=_get_config_value(data, "video", "frames_per_window", 3, _parse_int),
        max_windows=_get_config_value(data, "video", "max_windows", 0, _parse_int),
        enable_scene_change_detection=_get_config_value(data, "video", "enable_scene_change_detection", True, _parse_bool),
        scene_change_threshold=_get_config_value(data, "video", "scene_change_threshold", 0.25, _parse_float),
        scene_change_min_gap_seconds=_get_config_value(data, "video", "scene_change_min_gap_seconds", 2.0, _parse_float),
        filmstrip_tile_width=_get_config_value(data, "video", "filmstrip_tile_width", 480, _parse_int),
        filmstrip_tile_height=_get_config_value(data, "video", "filmstrip_tile_height", 270, _parse_int),
        chunk_duration_seconds=_get_config_value(data, "video", "chunk_duration_seconds", 60, _parse_int),
        chunk_overlap_seconds=_get_config_value(data, "video", "chunk_overlap_seconds", 10, _parse_int),
    )

    video_summarization = VideoSummarizationConfig(
        summary_detail=_get_config_value(data, "video_summarization", "summary_detail", "standard"),
        window_caption_sentences=_get_config_value(data, "video_summarization", "window_caption_sentences", 4, _parse_int),
        chapter_paragraphs_min=_get_config_value(data, "video_summarization", "chapter_paragraphs_min", 1, _parse_int),
        chapter_paragraphs_max=_get_config_value(data, "video_summarization", "chapter_paragraphs_max", 2, _parse_int),
        overall_paragraphs_min=_get_config_value(data, "video_summarization", "overall_paragraphs_min", 2, _parse_int),
        overall_paragraphs_max=_get_config_value(data, "video_summarization", "overall_paragraphs_max", 3, _parse_int),
        content_type_hint=_get_config_value(data, "video_summarization", "content_type_hint", "general"),
        chapter_gap_seconds=_get_config_value(data, "video_summarization", "chapter_gap_seconds", 120.0, _parse_float),
        max_chapter_duration_seconds=_get_config_value(data, "video_summarization", "max_chapter_duration_seconds", 300.0, _parse_float),
    )
```

Add both variables to the `return AppConfig(...)` call at the end of `load_config()`.

---

### 3 · `src/radiant_rag_mcp/ingestion/video_processor.py` — NEW FILE

#### 3a · Module-level availability guards

```python
from __future__ import annotations
import logging, tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from radiant_rag_mcp.config import VideoProcessorConfig
    from radiant_rag_mcp.ingestion.image_captioner import ImageCaptioner
    from radiant_rag_mcp.ingestion.processor import IngestedChunk

logger = logging.getLogger(__name__)

FASTER_WHISPER_AVAILABLE = False
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError: pass

OPENAI_WHISPER_AVAILABLE = False
try:
    import whisper as openai_whisper
    OPENAI_WHISPER_AVAILABLE = True
except ImportError: pass

YTDLP_AVAILABLE = False
try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError: pass

CV2_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError: pass

NUMPY_AVAILABLE = False
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError: pass

PIL_AVAILABLE = False
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError: pass

SCENEDETECT_AVAILABLE = False
try:
    from scenedetect import detect, ContentDetector
    SCENEDETECT_AVAILABLE = True
except ImportError: pass

FFMPEG_AVAILABLE = False
try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError: pass

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".flv", ".wmv", ".ts"}
```

#### 3b · Dataclasses

```python
@dataclass
class VideoSegment:
    start: float; end: float; text: str
    language: str = "en"; confidence: float = 1.0


@dataclass
class FrameWindow:
    window_index: int
    start: float           # seconds
    end: float             # seconds
    frame_timestamps: List[float]
    caption: str           # VLM output for the whole window (filmstrip)
    per_frame_captions: List[str]
    is_scene_boundary: bool = False


@dataclass
class VideoMetadata:
    source: str; title: str; duration: float
    language: str = "und"; has_audio: bool = True
    thumbnail_url: str = ""; uploader: str = ""
    upload_date: str = ""; description: str = ""
    is_youtube: bool = False; video_id: str = ""
```

#### 3c · `VideoProcessor` class — public API

```python
class VideoProcessor:
    """
    Downloads (if URL), detects audio, then routes to:
      - Audio present  --> Whisper --> transcript IngestedChunks
      - No audio       --> VLM frame-window analysis --> caption IngestedChunks
    """

    def __init__(
        self, config: "VideoProcessorConfig",
        image_captioner: Optional["ImageCaptioner"] = None,
    ) -> None:
        self._cfg = config
        self._captioner = image_captioner
        self._whisper_model = None

    def process_video(
        self, source: str,
        force_frame_analysis: bool = False,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> List["IngestedChunk"]: ...

    def process_youtube(self, url: str, force_frame_analysis: bool = False,
                        extra_meta: Optional[Dict[str, Any]] = None) -> List["IngestedChunk"]: ...

    def process_local_video(self, path: str, force_frame_analysis: bool = False,
                            extra_meta: Optional[Dict[str, Any]] = None) -> List["IngestedChunk"]: ...

    def is_youtube_url(self, url: str) -> bool: ...
    def is_video_file(self, path: str) -> bool: ...
```

#### 3d · Audio detection

```python
    def _has_audio(self, video_path: str) -> bool:
        """
        Return True if the file contains at least one audio stream with duration > 0.

        Priority:
        1. ffprobe via ffmpeg-python: parse JSON streams,
           any(s['codec_type']=='audio' and float(s.get('duration',0))>0)
        2. ffprobe via subprocess (same JSON).
        3. cv2.VideoCapture audio stream count.
        4. Default True (conservative).
        """

    def _probe_streams(self, video_path: str) -> List[Dict[str, Any]]:
        """Return stream dicts from ffprobe.  Returns [] on any error."""
```

#### 3e · Audio (transcript) path

```python
    def _process_audio_video(self, audio_path, metadata, extra_meta=None): ...
    def _load_whisper(self) -> None: ...
    def _transcribe(self, audio_path) -> Tuple[List[VideoSegment], str]: ...
    def _chunks_from_segments(self, segments, metadata, extra_meta=None) -> List["IngestedChunk"]:
        """
        Chunk meta dict must include:
          source, title, start_time, end_time, language, duration,
          video_id, is_youtube, is_silent=False, chunk_index, total_chunks,
          file_type="video", content_type="transcript"
        """
```

#### 3f · Silent video (frame-window) path

```python
    def _process_silent_video(self, video_path, metadata, extra_meta=None):
        """
        Steps:
        1. Guard: raise RuntimeError if no ImageCaptioner.
        2. _detect_scene_changes()
        3. _build_windows()
        4. _analyse_windows()
        5. _chunks_from_frame_windows()
        """
        if not self._captioner:
            raise RuntimeError(
                f"Silent video ({metadata.source!r}) requires an ImageCaptioner. "
                "Set vlm.enabled=true in config.")

    def _detect_scene_changes(self, video_path: str) -> List[float]:
        """
        1. PySceneDetect ContentDetector if SCENEDETECT_AVAILABLE
           (threshold = cfg.scene_change_threshold * 100).
        2. Numpy fallback: sample 1 fps, grayscale MAD normalised 0-1.
        3. Returns [] if unavailable.
        """

    def _build_windows(self, duration: float, scene_changes: List[float]) -> List[Tuple[float, float]]:
        """
        1. Fixed grid: step = window_duration - window_overlap
        2. Split windows at scene changes if sc is not within overlap_seconds of edge.
        3. Clip to [0, duration], drop windows < overlap_seconds.
        4. Apply max_windows cap if > 0.
        """

    def _extract_window_frames(self, video_path, window_start, window_end) -> List[Tuple[float, Any]]:
        """
        Extract cfg.frames_per_window frames uniformly spaced across the window.
        cv2.VideoCapture: set POS_MSEC, read, cvtColor BGR->RGB, PIL.Image.fromarray.
        Resize each frame to (filmstrip_tile_width, filmstrip_tile_height) LANCZOS.
        Returns [(timestamp_seconds, PIL.Image), ...].
        """

    def _tile_filmstrip(self, frames: List[Any]) -> Any:  # returns PIL.Image
        """
        Tile frames horizontally: total_width = N * tile_width, height = tile_height.
        Draw 2px dark (40,40,40) separator between tiles.
        Return RGB PIL.Image.
        """

    def _save_filmstrip_to_temp(self, filmstrip: Any) -> str:
        """
        Save the filmstrip PIL.Image to a temp PNG file.
        IMPORTANT: ImageCaptioner.caption_image() takes a FILE PATH, not a PIL.Image.
        Use tempfile.NamedTemporaryFile(suffix='.png', delete=False).
        The caller is responsible for deleting the file after use.
        """

    def _caption_window(self, frames, window_start, window_end, metadata) -> Tuple[str, List[str]]:
        """
        1. Tile frames into filmstrip via _tile_filmstrip().
        2. Save filmstrip to temp file via _save_filmstrip_to_temp().
        3. Build VLM prompt (see VLM Prompt Templates section).
        4. Call self._captioner.caption_image(filmstrip_path, prompt=window_prompt).
        5. Caption each frame individually via _caption_single_frame() -> per_frame_captions.
        6. Delete temp file.
        7. Return (window_caption, per_frame_captions).
        NOTE: caption_image() takes a file path (str), not a PIL.Image.
        """

    def _caption_single_frame(self, frame: Any, timestamp: float) -> str:
        """Save frame to temp PNG, call caption_image(), delete temp file."""

    def _analyse_windows(self, video_path, windows, scene_changes, metadata) -> List[FrameWindow]:
        """
        Iterate windows, caption each, return FrameWindow list.
        Log: logger.info("VideoProcessor: window %d/%d [%.1fs-%.1fs]", i+1, total, start, end)
        On per-window exception: log warning and continue.
        """

    def _chunks_from_frame_windows(self, windows, metadata, extra_meta=None) -> List["IngestedChunk"]:
        """
        Chunk text: f"[{w.start:.1f}s-{w.end:.1f}s] {w.caption}"
        Chunk meta must include:
          source, title, duration, is_silent=True,
          window_index, start_time, end_time, frame_timestamps,
          is_scene_boundary, file_type="video",
          content_type="frame_window_captions",
          chunk_index, total_chunks, video_id, is_youtube
        """
```

#### 3g · yt-dlp download helpers

```python
    def _download_youtube(self, url, force_frame_analysis=False) -> Tuple[str, VideoMetadata]:
        """
        Use cfg.ytdlp_format for audio videos, cfg.ytdlp_format_video for silent/forced.
        Determine has_audio from info.get('acodec') not in (None, 'none').
        """

    def _extract_local_metadata(self, path: str) -> VideoMetadata:
        """source, title=Path(path).stem, duration from ffprobe/cv2, has_audio=_has_audio()."""
```

---

### 4 · `src/radiant_rag_mcp/agents/video_summarization.py` — NEW FILE

#### 4a · Dataclasses

```python
@dataclass
class VideoChapter:
    index: int; start: float; end: float
    title: str; summary: str
    source_type: str  # "transcript" or "frame_window_captions"


@dataclass
class VideoSummaryResult:
    source: str; title: str; duration_seconds: float
    language: str; is_silent: bool
    summary: str; key_topics: List[str]
    chapters: List[VideoChapter]
    total_chunks: int; model_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### 4b · `VideoSummarizationAgent`

```python
class VideoSummarizationAgent(LLMAgent):

    def __init__(self, llm, config: Optional[VideoSummarizationConfig] = None):
        super().__init__(llm=llm)
        self._cfg = config or VideoSummarizationConfig()

    def summarize_video(
        self, source: str, chunks: List["IngestedChunk"],
        chapter_gap_seconds: Optional[float] = None,
        max_summary_tokens: int = 2048,
    ) -> VideoSummaryResult:
        """
        Build a structured summary from all chunks belonging to source.

        Steps:
        1. Apply preset overrides: if cfg.summary_detail is set, override
           chapter_paragraphs_* and overall_paragraphs_* from the preset table.
        2. Sort chunks by meta['start_time'].
        3. Detect source_type from first chunk meta.
        4. Group into chapters (see chapter grouping logic below).
        5. Summarise each chapter (LLM call with content-type prompt).
        6. Synthesise overall summary (map-reduce, LLM call with schema prompt).
        7. Extract key_topics from overall summary.
        8. Return VideoSummaryResult.
        """

    def _execute(self, query, context, **kwargs):
        return self.summarize_video(query, context)
```

**Chapter grouping logic:**
```
gap = cfg.chapter_gap_seconds (or constructor arg)
max_dur = cfg.max_chapter_duration_seconds
Start a new chapter when:
  (chunk.start_time - prev_chunk.end_time) > gap
  OR (chunk.end_time - chapter_start) > max_dur
```

#### 4c · LLM Prompt Templates

Use these exact prompt templates.  Insert actual values for `{...}` placeholders.

**Window caption prompt (used by `VideoProcessor._caption_window()`):**

```
system:
  You are a video content analyst. You will be shown a filmstrip of {N} frames
  sampled at equal intervals from a {window_duration:.1f}-second window of a
  silent video (no spoken audio). Frames are arranged left-to-right in
  chronological order, separated by thin vertical lines.

  Describe what is happening across these frames in {sentences} focused sentences.
  Include:
  - The main subjects and their actions or states
  - Any changes or transitions visible between the first and last frame
  - The setting or environment
  - Any text, labels, symbols, or data visible on screen

  Be specific: use proper nouns, quantities, technical terms, and spatial
  references where possible.  Do not speculate about audio.

user:
  Video: {title}
  Time window: {start:.1f}s to {end:.1f}s  (total duration: {total_duration:.0f}s)

  [filmstrip image]
```

Where `{sentences}` = `cfg.window_caption_sentences` (default 4).

---

**Chapter summary prompt — TRANSCRIPT content:**

```
system:
  You are a video content analyst. Summarise the following transcript segments
  from one chapter of a video.

  Write your summary in {min_para}-{max_para} paragraph(s).
  Paragraph 1: describe the main events, topics, and information in sequence.
  Paragraph 2 (if more than 1): what is established, demonstrated, or
    concluded by the end of this chapter.  Include specific facts and details.

  Be specific: preserve names, quantities, technical terms, and key claims.

user:
  Chapter {i} of {total}  [{start:.0f}s - {end:.0f}s]
  Source type: spoken transcript
  Content type hint: {content_type_hint}

  {joined_transcript_segments}

  Write the chapter summary now.
```

---

**Chapter summary prompt — FRAME_WINDOW_CAPTIONS content:**

```
system:
  You are a video content analyst summarising a series of visual scene
  descriptions extracted from a silent video.

  Each description covers a short time window.  Your task is to synthesise
  them into a coherent narrative in {min_para}-{max_para} paragraph(s).
  Paragraph 1: describe what is happening visually across this chapter in
    sequence — subjects, actions, setting, transitions.
  Paragraph 2 (if more than 1): what changes or is established visually
    by the end of this chapter.  Note any data, text, or indicators visible.

  Write in present tense.  Be specific about visual details.

user:
  Chapter {i} of {total}  [{start:.0f}s - {end:.0f}s]
  Source type: silent video frame descriptions
  Content type hint: {content_type_hint}

  {joined_window_captions}

  Write the chapter summary now.
```

---

**Overall summary prompt (map-reduce, both content types):**

```
system:
  You are a video content analyst. Write a structured overall summary of a
  video based on the chapter summaries provided.

  Write the summary in {min_para}-{max_para} structured paragraph(s):
  Paragraph 1 (Overview — always): what this video is, its topic, duration,
    and primary subject.
  Paragraph 2 (Content — always): key events, techniques, information, or
    visual content covered, in approximate chronological order.
  Paragraph 3 (Detail — include if {total_paragraphs} >= 3 and video > 5 min):
    noteworthy sequences, turning points, specific timestamps, or particularly
    dense content sections.
  Paragraph 4 (Takeaway — include if {total_paragraphs} == 4 and content_type_hint
    is instructional, scientific, or documentary):
    what a viewer would learn, observe, or conclude from the video.

  Omit Paragraphs 3 and 4 if they do not apply to this content.

user:
  Video: {title}
  Duration: {duration:.0f}s
  Source type: {source_type}
  Content type hint: {content_type_hint}

  Chapter summaries:
  {numbered_chapter_summaries}

  Write the overall summary now.
```

Where `{total_paragraphs}` = `cfg.overall_paragraphs_max`.

---

**Key topics extraction prompt:**

```
system:
  Extract 5-8 key topics or themes from the video summary below.
  Return a JSON array of short strings (2-5 words each).
  Return ONLY the JSON array, no preamble.

user:
  {overall_summary}
```

Parse the response as JSON; fall back to splitting on newlines if parsing fails.

---

### 5 · `src/radiant_rag_mcp/ingestion/processor.py` — MODIFY

Add at module level alongside `IMAGE_EXTENSIONS`:

```python
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".flv", ".wmv", ".ts"
}
```

`DocumentProcessor.__init__` — accept `video_config: Optional[VideoProcessorConfig] = None`.
Store as `self._video_cfg`.

In `process_file()`, before the JSON branch:

```python
if path.suffix.lower() in VIDEO_EXTENSIONS:
    try:
        from radiant_rag_mcp.ingestion.video_processor import VideoProcessor
        from radiant_rag_mcp.config import VideoProcessorConfig
        vp = VideoProcessor(config=self._video_cfg or VideoProcessorConfig())
        return vp.process_video(str(path))
    except Exception as e:
        logger.warning(f"Video processing failed for {path}: {e}")
        return []
```

---

### 6 · `src/radiant_rag_mcp/app.py` — add `ingest_videos()`

```python
def ingest_videos(
    self,
    sources: Sequence[str],
    show_progress: bool = True,
    use_hierarchical: bool = True,
    child_chunk_size: int = 512,
    child_chunk_overlap: int = 50,
    enable_frame_captioning: bool = False,
    force_frame_analysis: bool = False,
    summarize: bool = False,
) -> Dict[str, Any]:
    """
    Ingest videos into the RAG system.

    IMPORTANT: Access the LLM client as self._llm_clients.chat (not self._llm_client).
    The image captioner is at self._image_captioner.

    Returns:
        sources_processed, sources_failed, chunks_created, documents_stored,
        silent_sources, audio_sources, summaries, errors
    """
    from radiant_rag_mcp.ingestion.video_processor import VideoProcessor
    from radiant_rag_mcp.config import VideoSummarizationConfig
    from dataclasses import replace

    video_cfg = self._config.video
    if enable_frame_captioning != video_cfg.enable_frame_captioning:
        video_cfg = replace(video_cfg, enable_frame_captioning=enable_frame_captioning)

    captioner = self._image_captioner if (
        enable_frame_captioning or video_cfg.enable_silent_video_analysis
    ) else None

    processor = VideoProcessor(config=video_cfg, image_captioner=captioner)

    stats = {
        "sources_processed": 0, "sources_failed": 0,
        "chunks_created": 0, "documents_stored": 0,
        "silent_sources": 0, "audio_sources": 0,
        "summaries": {}, "errors": [],
    }

    for source in sources:
        try:
            chunks = processor.process_video(
                source, force_frame_analysis=force_frame_analysis)
            if not chunks:
                stats["sources_failed"] += 1
                stats["errors"].append(f"No chunks from: {source}")
                continue

            content_types = {c.meta.get("content_type") for c in chunks}
            if "frame_window_captions" in content_types:
                stats["silent_sources"] += 1
            if "transcript" in content_types:
                stats["audio_sources"] += 1

            stats["sources_processed"] += 1
            stats["chunks_created"] += len(chunks)
            stored = (
                self._ingest_hierarchical(source, chunks, child_chunk_size, child_chunk_overlap)
                if use_hierarchical else self._ingest_flat(chunks)
            )
            stats["documents_stored"] += stored

            if summarize:
                from radiant_rag_mcp.agents.video_summarization import VideoSummarizationAgent
                summ_cfg = self._config.video_summarization
                agent = VideoSummarizationAgent(llm=self._llm_clients.chat, config=summ_cfg)
                result = agent.summarize_video(source, chunks)
                stats["summaries"][source] = result.__dict__

        except Exception as e:
            logger.error(f"Failed to ingest video {source}: {e}")
            stats["sources_failed"] += 1
            stats["errors"].append(f"{source}: {e}")

    self._bm25_index.sync_with_store()
    self._bm25_index.save()
    return stats
```

---

### 7 · `src/radiant_rag_mcp/server.py` — add `ingest_video` MCP tool

```python
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
    Ingest one or more videos into the RAG knowledge base.

    Sources may be local file paths (.mp4, .mkv, .webm, .mov, .avi, etc.)
    or any URL supported by yt-dlp (YouTube, Vimeo, Twitch clips, etc.).

    Processing path chosen automatically:
    - Videos with audio:   Whisper transcription -> transcript chunks
    - Silent videos:       VLM frame-window analysis -> caption chunks
    - force_frame_analysis: always use VLM regardless of audio presence

    Args:
        sources:                File paths or video URLs.
        hierarchical:           Use hierarchical (parent/child) storage.
        child_chunk_size:       Token size for child chunks.
        child_chunk_overlap:    Overlap tokens between chunks.
        enable_frame_captioning: Also run VLM on audio videos.
        force_frame_analysis:   Skip audio detection; always use VLM.
        summarize:              Run VideoSummarizationAgent; include summaries.

    Returns:
        sources_processed, chunks_created, documents_stored, silent_sources,
        audio_sources, errors, summaries.
    """
    app: RadiantRAG = ctx.lifespan_context["app"]
    result = await asyncio.to_thread(
        app.ingest_videos, sources,
        show_progress=False,
        use_hierarchical=hierarchical,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
        enable_frame_captioning=enable_frame_captioning,
        force_frame_analysis=force_frame_analysis,
        summarize=summarize,
    )
    return result
```

Update the module docstring to say 'eleven tools'.

---

### 8 · `src/radiant_rag_mcp/ingestion/__init__.py` — MODIFY

```python
from radiant_rag_mcp.ingestion.video_processor import (
    VideoProcessor, VideoSegment, FrameWindow, VideoMetadata,
    VIDEO_EXTENSIONS, YTDLP_AVAILABLE, FASTER_WHISPER_AVAILABLE,
    OPENAI_WHISPER_AVAILABLE, CV2_AVAILABLE, SCENEDETECT_AVAILABLE,
)
```

---

### 9 · `src/radiant_rag_mcp/agents/__init__.py` — MODIFY

```python
from radiant_rag_mcp.agents.video_summarization import (
    VideoSummarizationAgent, VideoSummaryResult, VideoChapter,
)
```

---

### 10 · `notebooks/radiant_rag_mcp_video_test.ipynb` — NEW FILE

Create this file at `notebooks/radiant_rag_mcp_video_test.ipynb`.
The content is provided as a separate deliverable alongside this specification.
The notebook follows the same install / server-startup / helper / verify
structure as `notebooks/radiant_rag_mcp_search_query.ipynb` and covers:

| Section | What is tested |
|---|---|
| 1–6 | Install, import check, config, helpers, server, tool verification |
| 7 | Create two synthetic test videos (audio + silent) using cv2 + ffmpeg |
| 8 | `_has_audio()` smoke test — True for audio, False for silent |
| 9 | `ingest_video` audio path (Whisper) |
| 10 | `ingest_video` silent path (VLM frame windows) |
| 11 | `search_knowledge` across both transcript and frame-caption chunks |
| 12 | `query_knowledge` grounded in video content `[LLM]` |
| 13–15 | `VideoSummarizationAgent` direct API — brief / standard / detailed `[LLM]` |
| 16 | Side-by-side word-count comparison across detail levels `[LLM]` |
| 17 | YouTube URL ingestion via yt-dlp `[LLM]` |
| 18 | Cleanup |

---

## Silent Video Processing — Design Reference

### Audio detection decision tree

```
_has_audio(path)
    |
    +-- ffprobe via ffmpeg-python?
    |       YES --> streams = probe['streams']
    |              return any(s['codec_type']=='audio'
    |                        and float(s.get('duration',0))>0
    |                        for s in streams)
    +-- ffprobe via subprocess?
    |       YES --> same JSON, subprocess.run(['ffprobe',...])
    +-- cv2 available?
    |       YES --> cap.get(cv2.CAP_PROP_AUDIO_TOTAL_STREAMS) > 0
    +-- default: True (conservative)
```

### Sliding window construction

Given: `duration=90s`, `window_duration=10s`, `window_overlap=2s`,
scene changes at `t=15s` and `t=42s`:

```
step = 10 - 2 = 8s
Fixed grid:  [0-10] [8-18] [16-26] [24-34] [32-42] [40-50] [48-58] [56-66] [64-74] [72-82] [80-90]
Scene t=15s in [8-18]:  split -> [8-17] and [13-18]
Scene t=42s on boundary: no split needed
Final: [0-10] [8-17] [13-18] [16-26] [24-34] [32-42] [40-50] [48-58] [56-66] [64-74] [72-82] [80-90]
```

### Filmstrip layout

```
+-------------+-------------+-------------+
|  t=0.5s     |  t=5.0s     |  t=9.5s     |   tile_width=480, tile_height=270
|  (480x270)  |  (480x270)  |  (480x270)  |   total: 1440 x 270
+-------------+-------------+-------------+
```

**The filmstrip is saved to a temp PNG file and passed as a file path to**
**`ImageCaptioner.caption_image(filmstrip_path, prompt=window_prompt)`.**
The temp file is deleted after captioning.

---

## Summary Detail Levels Reference

| Level | Window caption | Chapter summary | Overall summary |
|---|---|---|---|
| `brief` | 3-5 sentences; specific subjects/actions | 3-5 sentences (1 paragraph) | 1 paragraph |
| `standard` | 3-5 sentences; specificity-focused | 1-2 paragraphs (5-10 sentences) | 2-3 paragraphs |
| `detailed` | 3-5 sentences; specificity-focused | 2-3 paragraphs (8-15 sentences) | 3-4 structured paragraphs |

**Key principle:** Window captions are retrieval units — retrieval precision
degrades with length.  Chapter and overall summaries are for human consumption
and map-reduce synthesis, so they scale with the preset.  Window caption length
is controlled by `window_caption_sentences` and is independent of the preset.

**Content-type-aware emphasis by `content_type_hint`:**

| Hint | What to emphasise in summaries |
|---|---|
| `general` | Events, subjects, actions in sequence |
| `instructional` | Steps, procedures, techniques; numbered where possible |
| `surveillance` | Entities present, movements, timestamps, changes in scene |
| `scientific` | Observations, measurements, data visible on screen, instruments |
| `documentary` | Narrative arc, factual claims, interviewees, visual evidence |

---

## Coding Standards

Follow the existing codebase conventions exactly:

- `from __future__ import annotations` at the top of every new file.
- All config types use `@dataclass(frozen=True)`.
- All heavy imports guarded with `try/except ImportError` and boolean
  availability flags at module level.
- Logging via `logger = logging.getLogger(__name__)` — never `print()`.
- Google-style docstrings on all public methods.
- Type hints on all public methods; use `Optional[X]` not `X | None`.
- `ImageCaptioner.caption_image()` takes a **file path** (`str`), not a PIL.Image.
  Always save filmstrips and individual frames to temp files before captioning.
- LLM client accessed as `self._llm_clients.chat` in `app.py` (not `self._llm_client`).
- Must run in Google Colab on T4 GPU and CPU-only mode.
- Temp files: `tempfile.mkdtemp()` if `cfg.download_dir` is None.
  Delete if `cfg.cleanup_after_ingest` is True.

---

## Validation Checklist

- [ ] `python -c "from radiant_rag_mcp.ingestion.video_processor import VideoProcessor"` succeeds with no optional deps.
- [ ] `python -c "from radiant_rag_mcp.agents.video_summarization import VideoSummarizationAgent"` succeeds.
- [ ] `python -c "from radiant_rag_mcp import app"` succeeds.
- [ ] `VideoProcessorConfig()` and `VideoSummarizationConfig()` are accessible from `radiant_rag_mcp.config`.
- [ ] `VideoProcessor(config=VideoProcessorConfig()).is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")` returns `True`.
- [ ] `VideoProcessor(config=VideoProcessorConfig()).is_video_file("clip.mp4")` returns `True`.
- [ ] `_has_audio()` returns `True` for a file with an audio stream.
- [ ] `_has_audio()` returns `False` for a file with no audio stream.
- [ ] `_build_windows(duration=90, scene_changes=[15.0, 42.0])` returns windows with boundaries near those timestamps.
- [ ] `_tile_filmstrip([f1, f2, f3])` returns a PIL.Image of width `3 * cfg.filmstrip_tile_width`.
- [ ] `_save_filmstrip_to_temp(filmstrip)` returns a string path to an existing `.png` file.
- [ ] Processing a silent `.mp4` with an ImageCaptioner produces chunks with `meta["content_type"] == "frame_window_captions"` and `meta["is_silent"] == True`.
- [ ] Processing an audio `.mp4` produces chunks with `meta["content_type"] == "transcript"` and `meta["is_silent"] == False`.
- [ ] `ingest_video(sources=[silent_mp4])` returns `silent_sources=1, audio_sources=0`.
- [ ] `ingest_video(sources=[audio_mp4], force_frame_analysis=True)` returns `silent_sources=1, audio_sources=0`.
- [ ] `VideoSummarizationAgent` with `summary_detail='brief'` produces a shorter overall summary than `standard` or `detailed`.
- [ ] `VideoSummarizationAgent` with `summary_detail='detailed'` produces 3-4 overall paragraphs.
- [ ] The `ingest_video` tool appears in `await client.list_tools()` output.
- [ ] `notebooks/radiant_rag_mcp_video_test.ipynb` runs end-to-end in Colab.

---

## Do NOT Change

- `storage/` — no changes to vector or BM25 store implementations.
- `agents/orchestrator.py` — no changes to the main RAG query pipeline.
- `llm/` — no changes to LLM backends or client.
- Existing MCP tool names or signatures.
- `config.yaml` defaults — only add new `video:` and `video_summarization:` sections.