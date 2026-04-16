"""
Video processing for Radiant Agentic RAG.

Handles YouTube URLs and local video files.  Provides:
  - Audio transcription via faster-whisper (preferred) or openai-whisper
  - yt-dlp download helper for YouTube
  - Time-windowed IngestedChunk output compatible with the RAG pipeline
  - Silent-video / frame-captioning path (stubs — not yet implemented)

Usage::

    from radiant_rag_mcp.config import VideoProcessorConfig
    from radiant_rag_mcp.ingestion.video_processor import VideoProcessor

    vp = VideoProcessor(VideoProcessorConfig())
    chunks = vp.process_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

LLM access (when needed by future paths): use ``self._llm_clients.chat``
not ``self._llm_client``.
"""

from __future__ import annotations

import json as _json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from radiant_rag_mcp.config import VideoProcessorConfig
    from radiant_rag_mcp.ingestion.processor import IngestedChunk
    from radiant_rag_mcp.ingestion.image_captioner import ImageCaptioner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency availability guards
# ---------------------------------------------------------------------------

try:
    from faster_whisper import WhisperModel as _FasterWhisperModel  # noqa: F401
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    _FasterWhisperModel = None  # type: ignore[assignment,misc]
    FASTER_WHISPER_AVAILABLE = False

try:
    import whisper as _openai_whisper  # noqa: F401
    OPENAI_WHISPER_AVAILABLE = True
except ImportError:
    _openai_whisper = None  # type: ignore[assignment]
    OPENAI_WHISPER_AVAILABLE = False

try:
    import yt_dlp as _yt_dlp  # noqa: F401
    YTDLP_AVAILABLE = True
except ImportError:
    _yt_dlp = None  # type: ignore[assignment]
    YTDLP_AVAILABLE = False

try:
    import cv2 as _cv2  # noqa: F401
    CV2_AVAILABLE = True
except ImportError:
    _cv2 = None  # type: ignore[assignment]
    CV2_AVAILABLE = False

try:
    import numpy as _numpy  # noqa: F401
    NUMPY_AVAILABLE = True
except ImportError:
    _numpy = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False

try:
    from PIL import Image as _PIL_Image  # noqa: F401
    PIL_AVAILABLE = True
except ImportError:
    _PIL_Image = None  # type: ignore[assignment]
    PIL_AVAILABLE = False

try:
    import scenedetect as _scenedetect  # noqa: F401
    SCENEDETECT_AVAILABLE = True
except ImportError:
    _scenedetect = None  # type: ignore[assignment]
    SCENEDETECT_AVAILABLE = False

try:
    import ffmpeg as _ffmpeg  # noqa: F401
    FFMPEG_AVAILABLE = True
except ImportError:
    _ffmpeg = None  # type: ignore[assignment]
    FFMPEG_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".webm", ".mov", ".avi",
    ".m4v", ".flv", ".wmv", ".ts",
}

# Compiled patterns that identify YouTube URLs
_YOUTUBE_RE: List[re.Pattern[str]] = [
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/watch\?.*v=[\w-]+"),
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/shorts/[\w-]+"),
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/embed/[\w-]+"),
    re.compile(r"(?:https?://)?youtu\.be/[\w-]+"),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VideoSegment:
    """A single transcribed segment produced by Whisper."""

    start: float           # Segment start time in seconds
    end: float             # Segment end time in seconds
    text: str              # Transcribed text content
    language: str = ""     # ISO 639-1 language code (e.g. "en")
    confidence: float = 1.0  # Average word-level confidence (if available)


@dataclass
class FrameWindow:
    """A temporal window of video frames used for silent-video analysis."""

    start: float                           # Window start time in seconds
    end: float                             # Window end time in seconds
    frame_paths: List[str] = field(default_factory=list)
    caption: Optional[str] = None          # VLM caption for the window


@dataclass
class VideoMetadata:
    """Metadata describing a video source (YouTube or local file)."""

    source: str                           # Original URL or file path
    title: str                            # Human-readable title
    duration: float = 0.0                 # Total duration in seconds
    video_id: Optional[str] = None        # YouTube video ID, or None
    is_youtube: bool = False
    is_silent: bool = False               # True when no audio stream detected
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    uploader: Optional[str] = None
    upload_date: Optional[str] = None
    description: Optional[str] = None
    ext: str = ""                         # File extension without leading dot


# ---------------------------------------------------------------------------
# VideoProcessor
# ---------------------------------------------------------------------------

class VideoProcessor:
    """
    Ingest video sources (YouTube URLs and local files) for RAG pipelines.

    Audio path (fully implemented):
        Videos with audio are transcribed with faster-whisper (preferred) or
        openai-whisper and split into time-windowed IngestedChunk objects.

    Silent-video / frame-captioning path (stubs):
        All frame-analysis methods raise NotImplementedError and will be
        implemented in a future release.
    """

    def __init__(
        self,
        config: "VideoProcessorConfig",
        image_captioner: Optional["ImageCaptioner"] = None,
    ) -> None:
        """
        Initialise the VideoProcessor.

        Args:
            config: VideoProcessorConfig instance controlling all behaviour.
            image_captioner: Optional VLM captioner injected for frame
                captions (used by the silent-video path once implemented).
        """
        self._config = config
        self._image_captioner = image_captioner

        # Whisper model — loaded lazily via _load_whisper()
        self._whisper_model: Optional[Any] = None
        self._whisper_backend: Optional[str] = None  # "faster" | "openai"

    # ------------------------------------------------------------------
    # Public routing methods
    # ------------------------------------------------------------------

    def is_youtube_url(self, url: str) -> bool:
        """
        Check whether *url* is a recognised YouTube URL.

        Args:
            url: Candidate URL string.

        Returns:
            True if any YouTube URL pattern matches.
        """
        for pattern in _YOUTUBE_RE:
            if pattern.search(url):
                return True
        return False

    def is_video_file(self, path: str) -> bool:
        """
        Check whether *path* has a recognised video file extension.

        Args:
            path: File path string.

        Returns:
            True if the file suffix is listed in VIDEO_EXTENSIONS.
        """
        return Path(path).suffix.lower() in VIDEO_EXTENSIONS

    def process_video(
        self,
        source: str,
        force_frame_analysis: bool = False,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> List["IngestedChunk"]:
        """
        Route a video source to the appropriate processing path.

        Args:
            source: YouTube URL or absolute/relative path to a local video.
            force_frame_analysis: When True, skip audio transcription even
                if audio is present and run the (currently stubbed) silent-
                video frame-analysis path instead.
            extra_meta: Extra key/value pairs merged into every chunk's
                ``meta`` dict.

        Returns:
            List of IngestedChunk objects ready for RAG indexing.
        """
        if self.is_youtube_url(source):
            return self.process_youtube(source, force_frame_analysis, extra_meta)
        return self.process_local_video(source, force_frame_analysis, extra_meta)

    def process_youtube(
        self,
        url: str,
        force_frame_analysis: bool = False,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> List["IngestedChunk"]:
        """
        Download and ingest a YouTube video.

        Args:
            url: YouTube watch, short, or embed URL.
            force_frame_analysis: Force the frame-analysis path regardless
                of whether audio is detected.
            extra_meta: Extra metadata merged into every chunk.

        Returns:
            List of IngestedChunk objects.

        Raises:
            RuntimeError: If yt-dlp is not installed.
        """
        if not YTDLP_AVAILABLE:
            raise RuntimeError(
                "yt-dlp is required for YouTube ingestion. "
                "Install with: pip install yt-dlp"
            )

        video_path, metadata = self._download_youtube(url, force_frame_analysis)
        try:
            if not force_frame_analysis and not metadata.is_silent:
                return self._process_audio_video(video_path, metadata, extra_meta)
            return self._process_silent_video(video_path, metadata, extra_meta)
        finally:
            if self._config.cleanup_after_ingest and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    logger.debug("Removed temporary download: %s", video_path)
                except OSError as exc:
                    logger.warning(
                        "Could not remove temporary file %s: %s", video_path, exc
                    )

    def process_local_video(
        self,
        path: str,
        force_frame_analysis: bool = False,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> List["IngestedChunk"]:
        """
        Ingest a local video file.

        Args:
            path: Absolute or relative path to a video file.
            force_frame_analysis: Force the frame-analysis path.
            extra_meta: Extra metadata merged into every chunk.

        Returns:
            List of IngestedChunk objects.
        """
        metadata = self._extract_local_metadata(path)
        if not force_frame_analysis and not metadata.is_silent:
            return self._process_audio_video(path, metadata, extra_meta)
        return self._process_silent_video(path, metadata, extra_meta)

    # ------------------------------------------------------------------
    # Audio / stream detection
    # ------------------------------------------------------------------

    def _probe_streams(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Return stream descriptors from ffprobe.

        Args:
            video_path: Path to the video file.

        Returns:
            List of stream dicts from ffprobe, or an empty list on failure.
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-print_format", "json",
                    "-show_streams",
                    video_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return []
            data = _json.loads(result.stdout)
            return data.get("streams", [])
        except Exception as exc:
            logger.debug("ffprobe probe failed for %s: %s", video_path, exc)
            return []

    def _has_audio(self, video_path: str) -> bool:
        """
        Determine whether *video_path* contains an audio stream.

        Strategy (in order):
          1. ffprobe via subprocess — definitive when available.
          2. OpenCV ``CAP_PROP_AUDIO_STREAM_COUNT`` — lightweight fallback.
          3. Assume True — transcription will fail gracefully if wrong.

        Args:
            video_path: Path to the video file.

        Returns:
            True if an audio stream is found (or conservatively assumed).
        """
        # 1. ffprobe
        streams = self._probe_streams(video_path)
        if streams:
            for stream in streams:
                if stream.get("codec_type") == "audio":
                    return True
            # ffprobe ran successfully but found no audio stream
            return False

        # 2. OpenCV
        if CV2_AVAILABLE:
            try:
                cap = _cv2.VideoCapture(video_path)
                audio_count = cap.get(_cv2.CAP_PROP_AUDIO_STREAM_COUNT)
                cap.release()
                if audio_count is not None:
                    return int(audio_count) > 0
                return False
            except Exception as exc:
                logger.debug("OpenCV audio check failed for %s: %s", video_path, exc)

        # 3. Conservative fallback
        logger.debug(
            "Cannot determine audio presence for %s; assuming audio exists.",
            video_path,
        )
        return True

    # ------------------------------------------------------------------
    # Whisper loading and transcription
    # ------------------------------------------------------------------

    def _load_whisper(self) -> bool:
        """
        Lazily load a Whisper model.

        Tries faster-whisper first; falls back to openai-whisper.

        Returns:
            True if a model is successfully loaded.
        """
        if self._whisper_model is not None:
            return True

        model_size = self._config.whisper_model
        device = self._config.whisper_device
        compute_type = self._config.whisper_compute_type

        # Resolve "auto" device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        # 1. faster-whisper (preferred)
        if FASTER_WHISPER_AVAILABLE:
            try:
                logger.info(
                    "Loading faster-whisper model '%s' on %s (compute_type=%s)",
                    model_size, device, compute_type,
                )
                self._whisper_model = _FasterWhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type,
                )
                self._whisper_backend = "faster"
                logger.info("faster-whisper model ready.")
                return True
            except Exception as exc:
                logger.warning("faster-whisper load failed: %s", exc)
                self._whisper_model = None

        # 2. openai-whisper fallback
        if OPENAI_WHISPER_AVAILABLE:
            try:
                logger.info("Loading openai-whisper model '%s'", model_size)
                self._whisper_model = _openai_whisper.load_model(
                    model_size,
                    device=device if device != "auto" else None,
                )
                self._whisper_backend = "openai"
                logger.info("openai-whisper model ready.")
                return True
            except Exception as exc:
                logger.warning("openai-whisper load failed: %s", exc)
                self._whisper_model = None

        logger.error(
            "No Whisper backend available. "
            "Install faster-whisper>=1.0.0 or openai-whisper>=20231117."
        )
        return False

    def _transcribe(
        self,
        audio_path: str,
    ) -> Tuple[List[VideoSegment], str]:
        """
        Transcribe an audio or video file with the loaded Whisper model.

        Args:
            audio_path: Path to an audio or video file.

        Returns:
            Tuple of (segments, language_code). Returns ([], "") if
            transcription is unavailable or fails.
        """
        if not self._load_whisper():
            logger.error(
                "Whisper unavailable; cannot transcribe %s.", audio_path
            )
            return [], ""

        language_hint: Optional[str] = self._config.whisper_language
        if language_hint == "auto":
            language_hint = None  # Whisper auto-detect

        segments: List[VideoSegment] = []
        language = ""

        try:
            if self._whisper_backend == "faster":
                raw_segs, info = self._whisper_model.transcribe(
                    audio_path,
                    language=language_hint,
                    beam_size=5,
                )
                language = info.language if info else ""
                for seg in raw_segs:
                    segments.append(VideoSegment(
                        start=float(seg.start),
                        end=float(seg.end),
                        text=seg.text.strip(),
                        language=language,
                        confidence=float(getattr(seg, "avg_logprob", 1.0)),
                    ))

            elif self._whisper_backend == "openai":
                result = self._whisper_model.transcribe(
                    audio_path,
                    language=language_hint,
                    verbose=False,
                )
                language = result.get("language", "")
                for seg in result.get("segments", []):
                    segments.append(VideoSegment(
                        start=float(seg["start"]),
                        end=float(seg["end"]),
                        text=seg["text"].strip(),
                        language=language,
                        confidence=float(seg.get("avg_logprob", 0.0)),
                    ))

        except Exception as exc:
            logger.error("Transcription failed for %s: %s", audio_path, exc)

        logger.info(
            "Transcribed %s: %d segments, language=%s",
            audio_path, len(segments), language or "unknown",
        )
        return segments, language

    def _process_audio_video(
        self,
        audio_path: str,
        metadata: VideoMetadata,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> List["IngestedChunk"]:
        """
        Transcribe *audio_path* and convert the result to IngestedChunk objects.

        Args:
            audio_path: Path to the audio/video file.
            metadata: VideoMetadata for the source.
            extra_meta: Extra key/value pairs merged into every chunk.

        Returns:
            List of IngestedChunk objects (empty if transcription yields
            nothing).
        """
        segments, _lang = self._transcribe(audio_path)
        if not segments:
            logger.warning(
                "No transcription segments for %s; returning empty result.",
                audio_path,
            )
            return []
        return self._chunks_from_segments(segments, metadata, extra_meta)

    def _chunks_from_segments(
        self,
        segments: List[VideoSegment],
        metadata: VideoMetadata,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> List["IngestedChunk"]:
        """
        Group transcription segments into fixed-duration IngestedChunk objects.

        Windows are sized by ``config.chunk_duration_seconds`` with
        ``config.chunk_overlap_seconds`` overlap.  Each chunk carries the
        meta keys required by the retrieval pipeline:
        ``source``, ``title``, ``start_time``, ``end_time``, ``language``,
        ``duration``, ``video_id``, ``is_youtube``, ``is_silent``,
        ``chunk_index``, ``total_chunks``, ``file_type``, ``content_type``.

        Args:
            segments: Ordered list of VideoSegment objects.
            metadata: VideoMetadata for the source.
            extra_meta: Extra key/value pairs merged into every chunk.

        Returns:
            List of IngestedChunk objects.
        """
        # Local import prevents circular dependency (processor → video_processor)
        from radiant_rag_mcp.ingestion.processor import IngestedChunk

        if not segments:
            return []

        chunk_dur = float(self._config.chunk_duration_seconds)
        overlap = float(self._config.chunk_overlap_seconds)
        step = max(chunk_dur - overlap, 1.0)

        total_duration = metadata.duration or (
            segments[-1].end if segments else 0.0
        )

        # Build time-window boundaries
        windows: List[Tuple[float, float]] = []
        cursor = 0.0
        while True:
            win_end = cursor + chunk_dur
            windows.append((cursor, win_end))
            if win_end >= total_duration:
                break
            cursor += step

        total_chunks = len(windows)
        chunks: List[IngestedChunk] = []

        for idx, (win_start, win_end) in enumerate(windows):
            # Include segments that overlap this window
            window_segs = [
                s for s in segments
                if s.end > win_start and s.start < win_end
            ]
            if not window_segs:
                continue

            text = " ".join(s.text for s in window_segs).strip()
            if not text:
                continue

            # Majority-vote language across window segments
            lang_counts: Dict[str, int] = {}
            for s in window_segs:
                lang_counts[s.language] = lang_counts.get(s.language, 0) + 1
            language = max(lang_counts, key=lambda k: lang_counts[k]) if lang_counts else ""

            meta: Dict[str, Any] = {
                "source": metadata.source,
                "title": metadata.title,
                "start_time": win_start,
                "end_time": min(win_end, total_duration),
                "language": language,
                "duration": total_duration,
                "video_id": metadata.video_id,
                "is_youtube": metadata.is_youtube,
                "is_silent": False,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "file_type": "video",
                "content_type": "transcript",
            }
            if extra_meta:
                meta.update(extra_meta)

            chunks.append(IngestedChunk(content=text, meta=meta))

        logger.info(
            "Produced %d transcript chunks from %d segments (source=%s)",
            len(chunks), len(segments), metadata.source,
        )
        return chunks

    # ------------------------------------------------------------------
    # yt-dlp helpers
    # ------------------------------------------------------------------

    def _download_youtube(
        self,
        url: str,
        force_frame_analysis: bool = False,
    ) -> Tuple[str, VideoMetadata]:
        """
        Download a YouTube video using yt-dlp.

        Downloads audio-only by default (faster); switches to full video
        when *force_frame_analysis* is True.

        Args:
            url: YouTube URL.
            force_frame_analysis: Download full video stream when True.

        Returns:
            Tuple of (local_file_path, VideoMetadata).

        Raises:
            RuntimeError: If yt-dlp is unavailable or the download fails.
        """
        if not YTDLP_AVAILABLE:
            raise RuntimeError(
                "yt-dlp is required but not installed. "
                "Install with: pip install yt-dlp"
            )

        download_dir = self._config.download_dir or tempfile.mkdtemp(
            prefix="radiant_video_"
        )
        os.makedirs(download_dir, exist_ok=True)

        fmt = (
            self._config.ytdlp_format_video
            if force_frame_analysis
            else self._config.ytdlp_format
        )

        ydl_opts: Dict[str, Any] = {
            "format": fmt,
            "outtmpl": os.path.join(download_dir, "%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }

        info: Optional[Dict[str, Any]] = None
        with _yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
            except Exception as exc:
                raise RuntimeError(
                    f"yt-dlp download failed for {url}: {exc}"
                ) from exc

        if info is None:
            raise RuntimeError(f"yt-dlp returned no metadata for {url}")

        video_id: str = info.get("id", "")
        ext: str = info.get("ext", "mp4")
        downloaded_path = os.path.join(download_dir, f"{video_id}.{ext}")

        # Some yt-dlp versions write a merged/post-processed filename
        if not os.path.exists(downloaded_path):
            candidates = sorted(
                f for f in os.listdir(download_dir)
                if f.startswith(video_id)
            )
            if candidates:
                downloaded_path = os.path.join(download_dir, candidates[0])
                ext = Path(downloaded_path).suffix.lstrip(".")
            else:
                raise RuntimeError(
                    f"Downloaded file not found in {download_dir} for {url}"
                )

        duration = float(info.get("duration") or 0.0)
        is_silent = not self._has_audio(downloaded_path)
        raw_desc: Optional[str] = info.get("description")

        metadata = VideoMetadata(
            source=url,
            title=info.get("title", url),
            duration=duration,
            video_id=video_id,
            is_youtube=True,
            is_silent=is_silent,
            width=info.get("width"),
            height=info.get("height"),
            fps=info.get("fps"),
            uploader=info.get("uploader"),
            upload_date=info.get("upload_date"),
            description=raw_desc[:500] if raw_desc else None,
            ext=ext,
        )

        logger.info(
            "YouTube download complete: '%s' (%s, %.0fs, silent=%s)",
            metadata.title, video_id, duration, is_silent,
        )
        return downloaded_path, metadata

    def _extract_local_metadata(self, path: str) -> VideoMetadata:
        """
        Extract metadata from a local video file.

        Tries ffprobe first; falls back to OpenCV; then uses bare minimum.

        Args:
            path: Absolute or relative path to the video file.

        Returns:
            VideoMetadata describing the file.
        """
        p = Path(path)
        title = p.stem
        ext = p.suffix.lstrip(".")
        duration = 0.0
        width: Optional[int] = None
        height: Optional[int] = None
        fps: Optional[float] = None

        # 1. ffprobe
        streams = self._probe_streams(path)
        if streams:
            for stream in streams:
                codec_type = stream.get("codec_type", "")
                if codec_type == "video" and width is None:
                    width = stream.get("width")
                    height = stream.get("height")
                    rfr = stream.get("r_frame_rate", "")
                    if "/" in rfr:
                        try:
                            num, den = rfr.split("/")
                            denom = float(den)
                            fps = float(num) / denom if denom else None
                        except (ValueError, ZeroDivisionError):
                            fps = None
                raw_dur = stream.get("duration")
                if raw_dur:
                    try:
                        duration = max(duration, float(raw_dur))
                    except (ValueError, TypeError):
                        pass

        # 2. OpenCV fallback
        if duration == 0.0 and CV2_AVAILABLE:
            try:
                cap = _cv2.VideoCapture(path)
                fps_cv = cap.get(_cv2.CAP_PROP_FPS)
                frame_count = cap.get(_cv2.CAP_PROP_FRAME_COUNT)
                if fps_cv and fps_cv > 0 and frame_count > 0:
                    duration = frame_count / fps_cv
                if width is None:
                    w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
                    width = w or None
                    height = h or None
                if fps is None and fps_cv:
                    fps = float(fps_cv)
                cap.release()
            except Exception as exc:
                logger.debug(
                    "OpenCV metadata extraction failed for %s: %s", path, exc
                )

        is_silent = not self._has_audio(path)

        return VideoMetadata(
            source=path,
            title=title,
            duration=duration,
            video_id=None,
            is_youtube=False,
            is_silent=is_silent,
            width=width,
            height=height,
            fps=fps,
            ext=ext,
        )

    # ------------------------------------------------------------------
    # Silent-video path — stubs (not yet implemented)
    # ------------------------------------------------------------------

    def _process_silent_video(
        self,
        video_path: str,
        metadata: VideoMetadata,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> List["IngestedChunk"]:
        """Stub: full silent-video frame-analysis path not yet implemented."""
        raise NotImplementedError(
            "_process_silent_video is not yet implemented."
        )

    def _detect_scene_changes(self, video_path: str) -> List[float]:
        """Stub: scene-change detection not yet implemented."""
        raise NotImplementedError(
            "_detect_scene_changes is not yet implemented."
        )

    def _build_windows(
        self,
        duration: float,
        scene_changes: Optional[List[float]] = None,
    ) -> List[FrameWindow]:
        """Stub: window-building not yet implemented."""
        raise NotImplementedError("_build_windows is not yet implemented.")

    def _extract_window_frames(
        self,
        video_path: str,
        window: FrameWindow,
    ) -> FrameWindow:
        """Stub: frame extraction not yet implemented."""
        raise NotImplementedError(
            "_extract_window_frames is not yet implemented."
        )

    def _tile_filmstrip(self, frame_paths: List[str]) -> Any:
        """Stub: filmstrip tiling not yet implemented."""
        raise NotImplementedError("_tile_filmstrip is not yet implemented.")

    def _save_filmstrip_to_temp(self, filmstrip: Any) -> str:
        """Stub: filmstrip saving not yet implemented."""
        raise NotImplementedError(
            "_save_filmstrip_to_temp is not yet implemented."
        )

    def _caption_window(self, window: FrameWindow) -> Optional[str]:
        """Stub: window captioning not yet implemented."""
        raise NotImplementedError("_caption_window is not yet implemented.")

    def _caption_single_frame(self, frame_path: str) -> Optional[str]:
        """Stub: single-frame captioning not yet implemented."""
        raise NotImplementedError(
            "_caption_single_frame is not yet implemented."
        )

    def _analyse_windows(
        self,
        windows: List[FrameWindow],
        video_path: str,
    ) -> List[FrameWindow]:
        """Stub: window analysis not yet implemented."""
        raise NotImplementedError("_analyse_windows is not yet implemented.")

    def _chunks_from_frame_windows(
        self,
        windows: List[FrameWindow],
        metadata: VideoMetadata,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> List["IngestedChunk"]:
        """Stub: frame-window chunking not yet implemented."""
        raise NotImplementedError(
            "_chunks_from_frame_windows is not yet implemented."
        )
