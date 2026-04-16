"""
Video processing for Radiant Agentic RAG.

Handles YouTube URLs and local video files.  Provides:
  - Audio transcription via faster-whisper (preferred) or openai-whisper
  - yt-dlp download helper for YouTube
  - Time-windowed IngestedChunk output compatible with the RAG pipeline
  - Silent-video / frame-captioning path (frame extraction, filmstrip tiling, VLM captioning)

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

    start: float                                    # Window start time in seconds
    end: float                                      # Window end time in seconds
    frame_paths: List[str] = field(default_factory=list)
    caption: Optional[str] = None                   # VLM caption for the window
    frame_timestamps: List[float] = field(default_factory=list)   # Timestamps of extracted frames
    per_frame_captions: List[str] = field(default_factory=list)   # Per-frame VLM captions
    is_scene_boundary: bool = False                 # True if window starts at a scene change


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

    Silent-video / frame-captioning path (fully implemented):
        Silent videos (no audio) are processed by sampling frames into
        time windows, optionally splitting at scene changes, tiling frames
        into filmstrips, and captioning via the injected VLM captioner.
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
            image_captioner: Optional VLM captioner for frame captions.
                Required for the silent-video analysis path.
        """
        self._config = config
        self._captioner = image_captioner

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
    # Silent-video path — frame analysis and captioning
    # ------------------------------------------------------------------

    def _process_silent_video(
        self,
        video_path: str,
        metadata: VideoMetadata,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> List["IngestedChunk"]:
        """
        Run the full silent-video frame-analysis pipeline.

        Args:
            video_path: Path to the video file.
            metadata: VideoMetadata for the source.
            extra_meta: Extra key/value pairs merged into every chunk.

        Returns:
            List of IngestedChunk objects.

        Raises:
            RuntimeError: If no image captioner is configured.
        """
        if not self._captioner:
            raise RuntimeError(
                "An image_captioner is required for silent video processing. "
                "Provide a VLM captioner (e.g. HuggingFaceVLMCaptioner) when "
                "constructing VideoProcessor(config, image_captioner=<captioner>)."
            )

        scene_changes = self._detect_scene_changes(video_path)
        windows = self._build_windows(metadata.duration, scene_changes)

        if not windows:
            logger.warning(
                "No windows built for %s (duration=%.1f); returning empty result.",
                video_path, metadata.duration,
            )
            return []

        frame_windows = self._analyse_windows(
            video_path, windows, scene_changes, metadata
        )
        return self._chunks_from_frame_windows(frame_windows, metadata, extra_meta)

    def _detect_scene_changes(self, video_path: str) -> List[float]:
        """
        Detect scene-change timestamps in *video_path*.

        Tries PySceneDetect first; falls back to a per-second grayscale MAD
        comparison using OpenCV + NumPy; returns [] if neither is available.

        Args:
            video_path: Path to the video file.

        Returns:
            Sorted list of scene-change timestamps in seconds.
        """
        cfg = self._config
        threshold = cfg.scene_change_threshold    # 0–1
        min_gap = cfg.scene_change_min_gap_seconds

        # 1. PySceneDetect (ContentDetector threshold is 0–100)
        if SCENEDETECT_AVAILABLE:
            try:
                from scenedetect import detect, ContentDetector  # type: ignore[import]
                scene_list = detect(
                    video_path,
                    ContentDetector(threshold=threshold * 100.0),
                )
                # Each element is (start_timecode, end_timecode); skip the
                # first scene which begins at t=0.
                raw: List[float] = [
                    scene[0].get_seconds() for scene in scene_list[1:]
                ]
                filtered: List[float] = []
                last_t = -(min_gap + 1.0)
                for t in sorted(raw):
                    if t - last_t >= min_gap:
                        filtered.append(t)
                        last_t = t
                logger.debug(
                    "PySceneDetect: %d scene changes in %s", len(filtered), video_path
                )
                return filtered
            except Exception as exc:
                logger.warning(
                    "PySceneDetect failed for %s (%s); falling back to numpy.",
                    video_path, exc,
                )

        # 2. NumPy + OpenCV fallback: sample at ~1 fps, normalised grayscale MAD
        if not (CV2_AVAILABLE and NUMPY_AVAILABLE):
            logger.debug(
                "OpenCV+NumPy unavailable; returning empty scene changes for %s.",
                video_path,
            )
            return []

        try:
            cap = _cv2.VideoCapture(video_path)
            fps = cap.get(_cv2.CAP_PROP_FPS) or 25.0
            sample_interval = max(1, int(round(fps)))  # one frame per second

            prev_gray = None
            frame_idx = 0
            diffs: List[float] = []
            sample_times: List[float] = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % sample_interval == 0:
                    gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
                    t = frame_idx / fps
                    if prev_gray is not None:
                        mad = float(
                            _numpy.mean(
                                _numpy.abs(
                                    gray.astype(_numpy.float32)
                                    - prev_gray.astype(_numpy.float32)
                                )
                            )
                        ) / 255.0  # normalise to [0, 1]
                        diffs.append(mad)
                        sample_times.append(t)
                    prev_gray = gray
                frame_idx += 1

            cap.release()

            if not diffs:
                return []

            changes: List[float] = []
            last_t = -(min_gap + 1.0)
            for mad, t in zip(diffs, sample_times):
                if mad >= threshold and t - last_t >= min_gap:
                    changes.append(t)
                    last_t = t

            logger.debug(
                "NumPy fallback: %d scene changes in %s", len(changes), video_path
            )
            return changes

        except Exception as exc:
            logger.warning(
                "NumPy scene detection failed for %s: %s", video_path, exc
            )
            return []

    def _build_windows(
        self,
        duration: float,
        scene_changes: Optional[List[float]] = None,
    ) -> List[Tuple[float, float]]:
        """
        Build temporal windows over *duration*, splitting at scene changes.

        A fixed grid is produced first (step = window_duration − overlap).
        Each scene change that does NOT land within *overlap* seconds of a
        window edge causes that window to be split: the first sub-window ends
        at the scene change and the second starts at ``sc − overlap`` to
        preserve context.

        Args:
            duration: Total video duration in seconds.
            scene_changes: Optional list of scene-change timestamps.

        Returns:
            Ordered list of (start, end) tuples clipped to [0, duration]
            with windows shorter than *overlap_seconds* removed.
        """
        cfg = self._config
        window_dur = cfg.window_duration_seconds
        overlap = cfg.window_overlap_seconds
        max_wins = cfg.max_windows

        step = window_dur - overlap
        if step <= 0:
            step = window_dur

        # --- Build fixed grid ---
        windows: List[Tuple[float, float]] = []
        cursor = 0.0
        while cursor < duration:
            win_end = cursor + window_dur
            windows.append((cursor, win_end))
            if win_end >= duration:
                break
            cursor += step

        # --- Apply scene-change splits ---
        if scene_changes:
            for sc in sorted(scene_changes):
                new_windows: List[Tuple[float, float]] = []
                for a, b in windows:
                    if a < sc < b:
                        near_start = (sc - a) <= overlap
                        near_end = (b - sc) <= overlap
                        if not near_start and not near_end:
                            # Split: first part [a, sc], second [sc-overlap, b]
                            second_start = max(sc - overlap, a)
                            new_windows.append((a, sc))
                            new_windows.append((second_start, b))
                        else:
                            new_windows.append((a, b))
                    else:
                        new_windows.append((a, b))
                windows = new_windows

        # --- Clip to [0, duration], drop short windows ---
        result: List[Tuple[float, float]] = []
        for a, b in windows:
            a_c = max(0.0, a)
            b_c = min(b, duration)
            if b_c - a_c >= overlap:
                result.append((a_c, b_c))

        # --- Cap total number of windows ---
        if max_wins > 0 and len(result) > max_wins:
            result = result[:max_wins]

        logger.debug(
            "Built %d windows for duration=%.1f (scene_changes=%s)",
            len(result), duration, len(scene_changes) if scene_changes else 0,
        )
        return result

    def _extract_window_frames(
        self,
        video_path: str,
        start: float,
        end: float,
    ) -> List[Tuple[float, Any]]:
        """
        Extract uniformly-spaced frames from a window of a video file.

        Args:
            video_path: Path to the video file.
            start: Window start time in seconds.
            end: Window end time in seconds.

        Returns:
            List of (timestamp, PIL.Image) tuples.  The images are resized
            to (filmstrip_tile_width × filmstrip_tile_height).

        Raises:
            RuntimeError: If OpenCV or Pillow is unavailable.
        """
        if not CV2_AVAILABLE:
            raise RuntimeError(
                "OpenCV (cv2) is required for frame extraction. "
                "Install with: pip install opencv-python-headless"
            )
        if not PIL_AVAILABLE:
            raise RuntimeError(
                "Pillow is required for frame extraction. "
                "Install with: pip install Pillow"
            )

        cfg = self._config
        n = cfg.frames_per_window
        tile_w = cfg.filmstrip_tile_width
        tile_h = cfg.filmstrip_tile_height
        duration = end - start

        # Compute uniformly-spaced seek timestamps
        if n <= 1:
            timestamps = [start + duration / 2.0]
        else:
            timestamps = [
                start + i * duration / (n - 1)
                for i in range(n)
            ]

        results: List[Tuple[float, Any]] = []
        cap = _cv2.VideoCapture(video_path)
        try:
            for ts in timestamps:
                # Clamp slightly inside the window to avoid seeking past EOF
                seek_ms = min(ts, end - 0.033) * 1000.0
                cap.set(_cv2.CAP_PROP_POS_MSEC, seek_ms)
                ret, frame = cap.read()
                if not ret:
                    logger.debug(
                        "Frame read failed at ts=%.3fs in %s", ts, video_path
                    )
                    continue
                rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
                img = _PIL_Image.fromarray(rgb)
                img = img.resize((tile_w, tile_h), _PIL_Image.LANCZOS)
                results.append((ts, img))
        finally:
            cap.release()

        return results

    def _tile_filmstrip(self, frames: List[Any]) -> Any:
        """
        Tile *frames* horizontally into a single PIL.Image filmstrip.

        A 2-pixel separator in colour (40, 40, 40) is drawn between tiles.

        Args:
            frames: List of PIL.Image objects, each pre-sized to
                (filmstrip_tile_width × filmstrip_tile_height).

        Returns:
            RGB PIL.Image of size (N*tile_w + (N-1)*2, tile_h).

        Raises:
            RuntimeError: If Pillow is unavailable.
        """
        if not PIL_AVAILABLE:
            raise RuntimeError(
                "Pillow is required for filmstrip tiling. "
                "Install with: pip install Pillow"
            )

        cfg = self._config
        tile_w = cfg.filmstrip_tile_width
        tile_h = cfg.filmstrip_tile_height
        n = len(frames)

        if n == 0:
            return _PIL_Image.new("RGB", (tile_w, tile_h), (0, 0, 0))

        separator_w = 2
        total_width = n * tile_w + (n - 1) * separator_w
        filmstrip = _PIL_Image.new("RGB", (total_width, tile_h), (0, 0, 0))

        x = 0
        for i, frame in enumerate(frames):
            filmstrip.paste(frame.convert("RGB"), (x, 0))
            x += tile_w
            if i < n - 1:
                sep = _PIL_Image.new("RGB", (separator_w, tile_h), (40, 40, 40))
                filmstrip.paste(sep, (x, 0))
                x += separator_w

        return filmstrip

    def _save_filmstrip_to_temp(self, filmstrip: Any) -> str:
        """
        Save *filmstrip* as a temporary PNG file.

        The caller is responsible for deleting the file when done.

        Args:
            filmstrip: PIL.Image to save.

        Returns:
            Absolute path to the temporary PNG file.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        filmstrip.save(tmp.name, format="PNG")
        return tmp.name

    def _caption_single_frame(self, frame: Any, timestamp: float) -> str:
        """
        Caption a single PIL.Image frame via the VLM captioner.

        Saves the frame to a temporary PNG, calls
        ``self._captioner.caption_image(path)``, then deletes the file.

        Args:
            frame: PIL.Image of the frame.
            timestamp: Frame timestamp in seconds (used for logging).

        Returns:
            Caption string, or "" if captioning fails or captioner is absent.
        """
        if self._captioner is None:
            return ""

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        try:
            frame.save(tmp.name, format="PNG")
            caption = self._captioner.caption_image(tmp.name) or ""
        except Exception as exc:
            logger.warning(
                "Single-frame caption failed at ts=%.2fs: %s", timestamp, exc
            )
            caption = ""
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

        return caption

    def _caption_window(
        self,
        frames: List[Any],
        window_start: float,
        window_end: float,
        metadata: VideoMetadata,
    ) -> Tuple[str, List[str]]:
        """
        Caption a window by tiling its frames into a filmstrip and querying
        the VLM captioner with a structured prompt.

        Args:
            frames: List of PIL.Image objects for the window.
            window_start: Window start time in seconds.
            window_end: Window end time in seconds.
            metadata: VideoMetadata describing the source.

        Returns:
            Tuple of (window_caption, per_frame_captions).  If captioner is
            absent or frames is empty, returns ("", []).
        """
        cfg = self._config
        n = len(frames)

        if n == 0 or self._captioner is None:
            return "", []

        filmstrip = self._tile_filmstrip(frames)
        filmstrip_path = self._save_filmstrip_to_temp(filmstrip)
        per_frame_captions: List[str] = []

        try:
            window_duration = window_end - window_start
            sentences = cfg.window_caption_sentences

            system_text = (
                f"You are a video content analyst. You will be shown a filmstrip "
                f"of {n} frames sampled at equal intervals from a "
                f"{window_duration:.1f}-second window of a silent video (no spoken "
                f"audio). Frames are arranged left-to-right in chronological order, "
                f"separated by thin vertical lines. Describe what is happening across "
                f"these frames in {sentences} focused sentences. Include: the main "
                f"subjects and their actions or states, any changes or transitions "
                f"visible, the setting or environment, any text, labels, symbols, or "
                f"data visible on screen. Be specific: use proper nouns, quantities, "
                f"technical terms, and spatial references. Do not speculate about audio."
            )
            user_text = (
                f"Video: {metadata.title}\n"
                f"Time window: {window_start:.1f}s to {window_end:.1f}s "
                f"(total duration: {metadata.duration:.0f}s)\n"
                f"[filmstrip image]"
            )
            window_prompt = f"{system_text}\n\n{user_text}"

            # caption_image() takes a FILE PATH (str), not a PIL.Image
            caption = (
                self._captioner.caption_image(filmstrip_path, prompt=window_prompt)
                or ""
            )

            # Optional per-frame captions
            if cfg.enable_frame_captioning:
                if n > 1:
                    frame_ts = [
                        window_start + i * (window_end - window_start) / (n - 1)
                        for i in range(n)
                    ]
                else:
                    frame_ts = [window_start + (window_end - window_start) / 2.0]
                for frame, ts in zip(frames, frame_ts):
                    per_frame_captions.append(self._caption_single_frame(frame, ts))

        finally:
            try:
                os.unlink(filmstrip_path)
            except OSError:
                pass

        return caption, per_frame_captions

    def _analyse_windows(
        self,
        video_path: str,
        windows: List[Tuple[float, float]],
        scene_changes: Optional[List[float]],
        metadata: VideoMetadata,
    ) -> List[FrameWindow]:
        """
        Extract frames and caption every window in *windows*.

        Args:
            video_path: Path to the video file.
            windows: Ordered list of (start, end) tuples from _build_windows.
            scene_changes: Scene-change timestamps for boundary detection.
            metadata: VideoMetadata describing the source.

        Returns:
            List of populated FrameWindow objects (failures are skipped with
            a warning log).
        """
        result: List[FrameWindow] = []
        total = len(windows)
        half_gap = max(self._config.scene_change_min_gap_seconds / 2.0, 0.5)

        for i, (start, end) in enumerate(windows):
            logger.info(
                "VideoProcessor: window %d/%d [%.1fs-%.1fs]",
                i + 1, total, start, end,
            )
            try:
                frame_results = self._extract_window_frames(video_path, start, end)

                if not frame_results:
                    logger.warning(
                        "No frames extracted for window [%.1f-%.1f]; skipping.",
                        start, end,
                    )
                    continue

                frames = [img for _ts, img in frame_results]
                timestamps = [ts for ts, _img in frame_results]

                caption, per_frame_captions = self._caption_window(
                    frames, start, end, metadata
                )

                is_scene_boundary = any(
                    abs(start - sc) <= half_gap for sc in (scene_changes or [])
                )

                result.append(FrameWindow(
                    start=start,
                    end=end,
                    frame_paths=[],
                    caption=caption,
                    frame_timestamps=timestamps,
                    per_frame_captions=per_frame_captions,
                    is_scene_boundary=is_scene_boundary,
                ))

            except Exception as exc:
                logger.warning(
                    "Window %d/%d [%.1fs-%.1fs] processing failed: %s",
                    i + 1, total, start, end, exc,
                )

        return result

    def _chunks_from_frame_windows(
        self,
        windows: List[FrameWindow],
        metadata: VideoMetadata,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> List["IngestedChunk"]:
        """
        Convert captioned FrameWindow objects to IngestedChunk objects.

        Each chunk's content is ``"[start-end] caption"``; its meta carries
        all keys required by the retrieval pipeline.

        Args:
            windows: List of captioned FrameWindow objects.
            metadata: VideoMetadata describing the source.
            extra_meta: Extra key/value pairs merged into every chunk.

        Returns:
            List of IngestedChunk objects.
        """
        from radiant_rag_mcp.ingestion.processor import IngestedChunk

        if not windows:
            return []

        total_chunks = len(windows)
        chunks: List[IngestedChunk] = []

        for idx, w in enumerate(windows):
            caption = w.caption or ""
            text = f"[{w.start:.1f}s-{w.end:.1f}s] {caption}".strip()
            if not text:
                continue

            meta: Dict[str, Any] = {
                "source": metadata.source,
                "title": metadata.title,
                "duration": metadata.duration,
                "is_silent": True,
                "window_index": idx,
                "start_time": w.start,
                "end_time": w.end,
                "frame_timestamps": w.frame_timestamps,
                "is_scene_boundary": w.is_scene_boundary,
                "file_type": "video",
                "content_type": "frame_window_captions",
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "video_id": metadata.video_id,
                "is_youtube": metadata.is_youtube,
            }
            if extra_meta:
                meta.update(extra_meta)

            chunks.append(IngestedChunk(content=text, meta=meta))

        logger.info(
            "Produced %d frame-window chunks from %d windows (source=%s)",
            len(chunks), len(windows), metadata.source,
        )
        return chunks
