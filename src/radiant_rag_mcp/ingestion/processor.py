"""
Document ingestion for Radiant Agentic RAG.

Provides document parsing, cleaning, chunking, and image captioning
using the Unstructured library and local VLM models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from radiant_rag_mcp.config import UnstructuredCleaningConfig, JSONParsingConfig, VideoProcessorConfig

if TYPE_CHECKING:
    from radiant_rag_mcp.ingestion.image_captioner import ImageCaptioner

logger = logging.getLogger(__name__)

# Try to import unstructured - it's optional
try:
    from unstructured.cleaners.core import clean as unstructured_clean
    from unstructured.partition.auto import partition
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logger.warning(
        "Unstructured library not available. "
        "Install with: pip install unstructured"
    )

# Image extensions for captioning
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".flv", ".wmv", ".ts"}

# Try to import JSON parser
try:
    from radiant_rag_mcp.ingestion.json_parser import (
        JSONParser,
        JSONParsingStrategy,
        ParsedJSONChunk,
    )
    JSON_PARSER_AVAILABLE = True
except ImportError:
    JSON_PARSER_AVAILABLE = False
    logger.warning("JSON parser not available")


@dataclass(frozen=True)
class IngestedChunk:
    """A single chunk from a parsed document."""

    content: str
    meta: Dict[str, Any]


@dataclass(frozen=True)
class CleaningOptions:
    """Options for text cleaning."""

    enabled: bool = True
    bullets: bool = False
    extra_whitespace: bool = True
    dashes: bool = False
    trailing_punctuation: bool = False
    lowercase: bool = False

    @staticmethod
    def from_config(config: UnstructuredCleaningConfig) -> "CleaningOptions":
        """Create from configuration."""
        return CleaningOptions(
            enabled=config.enabled,
            bullets=config.bullets,
            extra_whitespace=config.extra_whitespace,
            dashes=config.dashes,
            trailing_punctuation=config.trailing_punctuation,
            lowercase=config.lowercase,
        )


@dataclass(frozen=True)
class CleaningPreview:
    """Preview sample of cleaning results."""

    source_path: str
    chunk_index: int
    before: str
    after: str
    cleaning_flags: Dict[str, bool]


def iter_input_files(paths: Sequence[str]) -> List[Path]:
    """
    Iterate over input paths, expanding directories.

    Args:
        paths: List of file or directory paths

    Returns:
        List of file paths
    """
    out: List[Path] = []

    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            for child in pp.rglob("*"):
                if child.is_file() and not child.name.startswith("."):
                    out.append(child)
        elif pp.is_file():
            out.append(pp)
        else:
            logger.warning(f"Path does not exist: {p}")

    return out


def _apply_cleaning(text: str, opts: CleaningOptions) -> str:
    """
    Apply text cleaning using Unstructured library.

    Args:
        text: Raw text to clean
        opts: Cleaning options

    Returns:
        Cleaned text
    """
    if not opts.enabled:
        return text

    if not UNSTRUCTURED_AVAILABLE:
        # Basic fallback cleaning
        if opts.extra_whitespace:
            import re
            text = re.sub(r"\s+", " ", text)
        if opts.lowercase:
            text = text.lower()
        return text.strip()

    return unstructured_clean(
        text,
        bullets=opts.bullets,
        extra_whitespace=opts.extra_whitespace,
        dashes=opts.dashes,
        trailing_punctuation=opts.trailing_punctuation,
        lowercase=opts.lowercase,
    )


def parse_image_with_caption(
    file_path: str,
    captioner: Optional["ImageCaptioner"] = None,
    cleaning: Optional[CleaningOptions] = None,
) -> List[IngestedChunk]:
    """
    Parse an image file by generating a VLM caption.

    Args:
        file_path: Path to image file
        captioner: ImageCaptioner instance for generating captions
        cleaning: Text cleaning options

    Returns:
        List containing a single IngestedChunk with the image caption
    """
    path = Path(file_path)
    cleaning = cleaning or CleaningOptions()

    # Generate caption using VLM
    caption = None
    if captioner is not None:
        caption = captioner.caption_image(file_path)

    if caption:
        # Apply cleaning to caption
        content = _apply_cleaning(caption, cleaning)
        logger.info(f"Generated VLM caption for {path.name}: {content[:100]}...")
    else:
        # Fallback: just describe as an image
        content = f"[Image: {path.name}]"
        logger.warning(f"Could not generate caption for {path.name}, using placeholder")

    return [
        IngestedChunk(
            content=content,
            meta={
                "source_path": file_path,
                "element_type": "Image",
                "has_vlm_caption": caption is not None,
                "file_type": path.suffix.lower(),
            }
        )
    ]


def _is_image_file(file_path: str) -> bool:
    """Check if file is an image based on extension."""
    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS


def parse_document(
    file_path: str,
    strategy: Optional[str] = None,
    cleaning: Optional[CleaningOptions] = None,
    preview_sink: Optional[List[CleaningPreview]] = None,
    preview_max_items: int = 12,
    preview_max_chars: int = 800,
) -> List[IngestedChunk]:
    """
    Parse a document into chunks.

    Uses Unstructured's partition function with configurable strategy.

    Args:
        file_path: Path to document file
        strategy: Partition strategy ("auto", "fast", "hi_res", "ocr_only")
                  Default is "auto" which selects based on file type
        cleaning: Text cleaning options
        preview_sink: Optional list to collect cleaning preview samples
        preview_max_items: Maximum preview samples to collect
        preview_max_chars: Maximum characters per preview sample

    Returns:
        List of IngestedChunk objects
    """
    if not UNSTRUCTURED_AVAILABLE:
        raise RuntimeError(
            "Unstructured library required for document parsing. "
            "Install with: pip install unstructured"
        )

    cleaning = cleaning or CleaningOptions()

    # Determine partition strategy based on file type
    partition_kwargs: Dict[str, Any] = {"filename": file_path}

    # Valid strategies: auto, fast, hi_res, ocr_only
    file_ext = Path(file_path).suffix.lower()

    if strategy and strategy in ("auto", "fast", "hi_res", "ocr_only"):
        partition_kwargs["strategy"] = strategy
    elif file_ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"):
        # For images, try OCR but fall back to auto if it fails
        partition_kwargs["strategy"] = "auto"
    elif file_ext == ".pdf":
        # For PDFs, try fast first (most compatible)
        partition_kwargs["strategy"] = "fast"
    # else: use default "auto" strategy

    try:
        elements = partition(**partition_kwargs)

        # If PDF returned no elements, try with ocr_only strategy
        if not elements and file_ext == ".pdf":
            logger.debug(f"PDF has no extractable text, using OCR for {file_path}")
            partition_kwargs["strategy"] = "ocr_only"
            elements = partition(**partition_kwargs)

    except Exception as e:
        # If strategy failed, try with "auto" as fallback
        if partition_kwargs.get("strategy") and partition_kwargs["strategy"] != "auto":
            logger.warning(f"Strategy '{partition_kwargs['strategy']}' failed for {file_path}, trying 'auto': {e}")
            partition_kwargs["strategy"] = "auto"
            try:
                elements = partition(**partition_kwargs)
            except Exception as e2:
                logger.error(f"Failed to parse {file_path} with auto strategy: {e2}")
                raise
        else:
            logger.error(f"Failed to parse {file_path}: {e}")
            raise

    chunks: List[IngestedChunk] = []

    for el in elements:
        text = getattr(el, "text", None)
        if not text:
            continue

        before = str(text).strip()
        after = _apply_cleaning(before, cleaning).strip()

        if not after:
            continue

        # Extract metadata
        meta: Dict[str, Any] = {"source_path": file_path}
        if hasattr(el, "metadata") and el.metadata is not None:
            if hasattr(el.metadata, "to_dict"):
                md = el.metadata.to_dict()
            else:
                md = dict(el.metadata) if isinstance(el.metadata, dict) else {}
            meta.update(md)

        # Record cleaning flags in metadata
        meta["cleaning"] = {
            "enabled": cleaning.enabled,
            "bullets": cleaning.bullets,
            "extra_whitespace": cleaning.extra_whitespace,
            "dashes": cleaning.dashes,
            "trailing_punctuation": cleaning.trailing_punctuation,
            "lowercase": cleaning.lowercase,
        }

        chunk_index = len(chunks)

        # Capture preview sample if requested
        if preview_sink is not None and len(preview_sink) < preview_max_items:
            preview_sink.append(
                CleaningPreview(
                    source_path=file_path,
                    chunk_index=chunk_index,
                    before=before[:preview_max_chars] + ("…" if len(before) > preview_max_chars else ""),
                    after=after[:preview_max_chars] + ("…" if len(after) > preview_max_chars else ""),
                    cleaning_flags=meta["cleaning"],
                )
            )

        chunks.append(IngestedChunk(content=after, meta=meta))

    logger.debug(f"Parsed {file_path}: {len(chunks)} chunks")
    return chunks


def parse_text_file(
    file_path: str,
    cleaning: Optional[CleaningOptions] = None,
) -> List[IngestedChunk]:
    """
    Simple text file parsing (without Unstructured).

    Args:
        file_path: Path to text file
        cleaning: Text cleaning options

    Returns:
        List containing single IngestedChunk
    """
    cleaning = cleaning or CleaningOptions()

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    cleaned = _apply_cleaning(content, cleaning).strip()

    if not cleaned:
        return []

    return [
        IngestedChunk(
            content=cleaned,
            meta={
                "source_path": file_path,
                "cleaning": {
                    "enabled": cleaning.enabled,
                    "bullets": cleaning.bullets,
                    "extra_whitespace": cleaning.extra_whitespace,
                    "dashes": cleaning.dashes,
                    "trailing_punctuation": cleaning.trailing_punctuation,
                    "lowercase": cleaning.lowercase,
                },
            },
        )
    ]


class ChunkSplitter:
    """
    Split large chunks into smaller pieces.

    Useful for ensuring chunks fit within embedding model limits
    or for creating child documents for hierarchical retrieval.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = " ",
    ) -> None:
        """
        Initialize chunk splitter.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            separator: Text separator for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def split(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: Text to split

        Returns:
            List of chunk strings
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        prev_start = -1

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at a separator
            if end < len(text):
                # Look for separator within the chunk
                sep_pos = text.rfind(self.separator, start, end)
                if sep_pos > start:
                    end = sep_pos + len(self.separator)

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Calculate next start with overlap
            next_start = end - self.chunk_overlap

            # Avoid infinite loop - ensure we make progress
            if next_start <= prev_start:
                next_start = end

            prev_start = start
            start = next_start

        return chunks

    def split_chunk(
        self,
        chunk: IngestedChunk,
    ) -> List[IngestedChunk]:
        """
        Split an IngestedChunk into smaller chunks.

        Args:
            chunk: Chunk to split

        Returns:
            List of smaller chunks
        """
        texts = self.split(chunk.content)

        return [
            IngestedChunk(
                content=text,
                meta={**chunk.meta, "split_index": i, "split_total": len(texts)},
            )
            for i, text in enumerate(texts)
        ]


class DocumentProcessor:
    """
    High-level document processing pipeline.

    Combines parsing, cleaning, splitting, and image captioning.
    """

    def __init__(
        self,
        cleaning_config: UnstructuredCleaningConfig,
        strategy: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        image_captioner: Optional["ImageCaptioner"] = None,
        json_config: Optional[JSONParsingConfig] = None,
        video_config: Optional[VideoProcessorConfig] = None,
    ) -> None:
        """
        Initialize document processor.

        Args:
            cleaning_config: Cleaning configuration
            strategy: Partition strategy ("auto", "fast", "hi_res", "ocr_only")
            chunk_size: Target chunk size
            chunk_overlap: Chunk overlap
            image_captioner: Optional VLM captioner for image files
            json_config: JSON/JSONL parsing configuration
            video_config: Optional VideoProcessorConfig for video files
        """
        self._cleaning = CleaningOptions.from_config(cleaning_config)
        self._strategy = strategy
        self._splitter = ChunkSplitter(chunk_size, chunk_overlap)
        self._preview_config = cleaning_config
        self._captioner = image_captioner
        self._video_cfg = video_config

        # Initialize JSON parser if available and enabled
        self._json_parser = None
        if JSON_PARSER_AVAILABLE and json_config and json_config.enabled:
            from radiant_rag_mcp.ingestion.json_parser import JSONParsingConfig as ParserConfig
            # Convert config dataclass to parser config
            parser_config = ParserConfig(
                default_strategy=JSONParsingStrategy(json_config.default_strategy),
                min_array_size_for_splitting=json_config.min_array_size_for_splitting,
                text_fields=json_config.text_fields,
                title_fields=json_config.title_fields,
                max_nesting_depth=json_config.max_nesting_depth,
                flatten_separator=json_config.flatten_separator,
                jsonl_batch_size=json_config.jsonl_batch_size,
                preserve_fields=json_config.preserve_fields,
            )
            self._json_parser = JSONParser(parser_config)

    def process_file(
        self,
        file_path: str,
        split_large_chunks: bool = False,
        max_chunk_size: int = 2000,
        preview_sink: Optional[List[CleaningPreview]] = None,
    ) -> List[IngestedChunk]:
        """
        Process a single file.

        Args:
            file_path: Path to file
            split_large_chunks: Whether to split large chunks
            max_chunk_size: Maximum chunk size before splitting
            preview_sink: Optional list for cleaning previews

        Returns:
            List of processed chunks
        """
        # Determine parsing method
        path = Path(file_path)

        if path.suffix.lower() in VIDEO_EXTENSIONS:
            try:
                from radiant_rag_mcp.ingestion.video_processor import VideoProcessor
                vp = VideoProcessor(config=self._video_cfg or VideoProcessorConfig())
                return vp.process_video(str(path))
            except Exception as e:
                logger.warning(f"Video processing failed for {path}: {e}")
                return []

        if path.suffix.lower() in (".json", ".jsonl") and self._json_parser:
            # JSON/JSONL files - use JSON parser
            try:
                if path.suffix.lower() == ".jsonl":
                    parsed_chunks = self._json_parser.parse_jsonl_file(file_path)
                else:
                    parsed_chunks = self._json_parser.parse_json_file(file_path)

                # Convert ParsedJSONChunk to IngestedChunk
                chunks = [
                    IngestedChunk(content=pc.content, meta=pc.meta)
                    for pc in parsed_chunks
                ]
            except Exception as e:
                logger.error(f"JSON parsing failed for {file_path}: {e}")
                # Fallback to text parsing
                chunks = parse_text_file(file_path, self._cleaning)

        elif path.suffix.lower() in (".txt", ".md", ".rst"):
            # Simple text files
            chunks = parse_text_file(file_path, self._cleaning)
        elif _is_image_file(file_path):
            # Image files - use VLM captioning
            chunks = parse_image_with_caption(
                file_path,
                captioner=self._captioner,
                cleaning=self._cleaning,
            )
        elif UNSTRUCTURED_AVAILABLE:
            # Use Unstructured for complex documents
            chunks = parse_document(
                file_path,
                strategy=self._strategy,
                cleaning=self._cleaning,
                preview_sink=preview_sink,
                preview_max_items=self._preview_config.preview_max_items,
                preview_max_chars=self._preview_config.preview_max_chars,
            )
        else:
            # Fallback to text parsing
            chunks = parse_text_file(file_path, self._cleaning)

        # Optionally split large chunks
        if split_large_chunks:
            result: List[IngestedChunk] = []
            for chunk in chunks:
                if len(chunk.content) > max_chunk_size:
                    result.extend(self._splitter.split_chunk(chunk))
                else:
                    result.append(chunk)
            return result

        return chunks

    def process_paths(
        self,
        paths: Sequence[str],
        split_large_chunks: bool = False,
        max_chunk_size: int = 2000,
    ) -> Dict[str, List[IngestedChunk]]:
        """
        Process multiple paths.

        Args:
            paths: File or directory paths
            split_large_chunks: Whether to split large chunks
            max_chunk_size: Maximum chunk size before splitting

        Returns:
            Dictionary mapping file paths to their chunks
        """
        files = iter_input_files(paths)
        results: Dict[str, List[IngestedChunk]] = {}
        preview_sink: List[CleaningPreview] = []

        for fp in files:
            file_path = str(fp)
            try:
                chunks = self.process_file(
                    file_path,
                    split_large_chunks=split_large_chunks,
                    max_chunk_size=max_chunk_size,
                    preview_sink=preview_sink if self._preview_config.preview_enabled else None,
                )
                results[file_path] = chunks
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results[file_path] = []

        # Log preview samples if enabled
        if self._preview_config.preview_enabled and preview_sink:
            logger.info(f"Cleaning preview ({len(preview_sink)} samples):")
            for sample in preview_sink[:5]:  # Log first 5
                logger.info(
                    f"  {sample.source_path}[{sample.chunk_index}]: "
                    f"'{sample.before[:50]}...' -> '{sample.after[:50]}...'"
                )

        return results


class IntelligentDocumentProcessor(DocumentProcessor):
    """
    Enhanced document processor with intelligent LLM-based chunking.
    
    Extends DocumentProcessor with semantic chunking capabilities that
    preserve logical document boundaries for better retrieval quality.
    """
    
    def __init__(
        self,
        cleaning_config: UnstructuredCleaningConfig,
        strategy: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        image_captioner: Optional["ImageCaptioner"] = None,
        chunking_agent: Optional[Any] = None,
        use_intelligent_chunking: bool = True,
        json_config: Optional[JSONParsingConfig] = None,
    ) -> None:
        """
        Initialize intelligent document processor.

        Args:
            cleaning_config: Cleaning configuration
            strategy: Partition strategy ("auto", "fast", "hi_res", "ocr_only")
            chunk_size: Target chunk size (for fallback)
            chunk_overlap: Chunk overlap (for fallback)
            image_captioner: Optional VLM captioner for image files
            chunking_agent: IntelligentChunkingAgent instance
            use_intelligent_chunking: Whether to use intelligent chunking
            json_config: JSON/JSONL parsing configuration
        """
        super().__init__(
            cleaning_config=cleaning_config,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            image_captioner=image_captioner,
            json_config=json_config,
        )
        self._chunking_agent = chunking_agent
        self._use_intelligent_chunking = use_intelligent_chunking and chunking_agent is not None
    
    def process_file(
        self,
        file_path: str,
        split_large_chunks: bool = False,
        max_chunk_size: int = 2000,
        preview_sink: Optional[List[CleaningPreview]] = None,
    ) -> List[IngestedChunk]:
        """
        Process a single file with intelligent chunking.
        
        Args:
            file_path: Path to file
            split_large_chunks: Whether to split large chunks (fallback)
            max_chunk_size: Maximum chunk size before splitting (fallback)
            preview_sink: Optional list for cleaning previews
            
        Returns:
            List of processed chunks
        """
        path = Path(file_path)
        
        # Use parent class for images (no chunking needed)
        if _is_image_file(file_path):
            return super().process_file(
                file_path, split_large_chunks, max_chunk_size, preview_sink
            )
        
        # Get raw content first
        if path.suffix.lower() in (".txt", ".md", ".rst"):
            raw_chunks = parse_text_file(file_path, self._cleaning)
        elif UNSTRUCTURED_AVAILABLE:
            raw_chunks = parse_document(
                file_path,
                strategy=self._strategy,
                cleaning=self._cleaning,
                preview_sink=preview_sink,
                preview_max_items=self._preview_config.preview_max_items,
                preview_max_chars=self._preview_config.preview_max_chars,
            )
        else:
            raw_chunks = parse_text_file(file_path, self._cleaning)
        
        # Combine all chunks into single content for intelligent chunking
        if not raw_chunks:
            return []
        
        # If only one small chunk or intelligent chunking disabled, use fallback
        total_content = "\n\n".join(c.content for c in raw_chunks)
        
        if not self._use_intelligent_chunking or len(total_content) < 500:
            if split_large_chunks:
                result: List[IngestedChunk] = []
                for chunk in raw_chunks:
                    if len(chunk.content) > max_chunk_size:
                        result.extend(self._splitter.split_chunk(chunk))
                    else:
                        result.append(chunk)
                return result
            return raw_chunks
        
        # Use intelligent chunking
        try:
            # Detect document type from file extension
            doc_type = self._detect_doc_type(path)
            
            # Get metadata from first chunk
            base_meta = raw_chunks[0].meta if raw_chunks else {}
            
            result = self._chunking_agent.chunk_document(
                content=total_content,
                doc_type=doc_type,
                metadata={"source_path": file_path, **base_meta},
            )
            
            # Convert SemanticChunks to IngestedChunks
            ingested_chunks = []
            for semantic_chunk in result.chunks:
                ingested_chunks.append(IngestedChunk(
                    content=semantic_chunk.content,
                    meta={
                        "source_path": file_path,
                        "chunk_index": semantic_chunk.chunk_index,
                        "chunk_type": semantic_chunk.chunk_type,
                        "chunking_method": result.chunking_method,
                        **semantic_chunk.metadata,
                    },
                ))
            
            logger.info(
                f"Intelligent chunking: {file_path} -> {len(ingested_chunks)} chunks "
                f"(method={result.chunking_method})"
            )
            
            return ingested_chunks
            
        except Exception as e:
            logger.warning(
                f"Intelligent chunking failed for {file_path}, "
                f"falling back to basic chunking: {e}"
            )
            # Fallback to parent implementation
            return super().process_file(
                file_path, split_large_chunks, max_chunk_size, preview_sink
            )
    
    def _detect_doc_type(self, path: Path) -> Optional[str]:
        """Detect document type from file extension."""
        ext = path.suffix.lower()
        
        if ext in (".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"):
            return "code"
        elif ext in (".md", ".markdown"):
            return "markdown"
        elif ext in (".html", ".htm", ".xml"):
            return "structured"
        elif ext == ".pdf":
            return None  # Let chunking agent detect
        else:
            return "prose"


class TranslatingDocumentProcessor:
    """
    Document processor with language detection and translation support.
    
    Wraps an existing document processor and adds:
    - Language detection for each document
    - Translation to canonical language (e.g., English) for indexing
    - Preservation of original text in metadata
    
    This enables multilingual document corpora to be indexed and searched
    in a single canonical language while retaining original content.
    """
    
    def __init__(
        self,
        base_processor: DocumentProcessor,
        language_detection_agent: Optional[Any] = None,
        translation_agent: Optional[Any] = None,
        canonical_language: str = "en",
        translate_at_ingestion: bool = True,
        preserve_original: bool = True,
    ) -> None:
        """
        Initialize translating document processor.
        
        Args:
            base_processor: Base document processor (DocumentProcessor or IntelligentDocumentProcessor)
            language_detection_agent: LanguageDetectionAgent instance
            translation_agent: TranslationAgent instance
            canonical_language: Target language for indexing (default: "en")
            translate_at_ingestion: Whether to translate during ingestion
            preserve_original: Whether to store original text in metadata
        """
        self._base_processor = base_processor
        self._lang_detector = language_detection_agent
        self._translator = translation_agent
        self._canonical_language = canonical_language
        self._translate_at_ingestion = translate_at_ingestion
        self._preserve_original = preserve_original
        
        # Track if translation is available
        self._translation_enabled = (
            translate_at_ingestion and 
            language_detection_agent is not None and 
            translation_agent is not None
        )
        
        if self._translation_enabled:
            logger.info(
                f"Translation enabled: documents will be translated to "
                f"'{canonical_language}' for indexing"
            )
    
    def process_file(
        self,
        file_path: str,
        split_large_chunks: bool = False,
        max_chunk_size: int = 2000,
        preview_sink: Optional[List[CleaningPreview]] = None,
    ) -> List[IngestedChunk]:
        """
        Process a single file with optional translation.
        
        Args:
            file_path: Path to file
            split_large_chunks: Whether to split large chunks
            max_chunk_size: Maximum chunk size before splitting
            preview_sink: Optional list for cleaning previews
            
        Returns:
            List of processed chunks with language metadata
        """
        # Get chunks from base processor
        chunks = self._base_processor.process_file(
            file_path, split_large_chunks, max_chunk_size, preview_sink
        )
        
        if not chunks:
            return chunks
        
        # If translation not enabled, return chunks as-is
        if not self._translation_enabled:
            return chunks
        
        # Detect document language from combined content
        # Use first few chunks to determine document language
        sample_content = " ".join(
            chunk.content[:1000] for chunk in chunks[:3]
        )
        
        doc_language = self._detect_document_language(sample_content)
        
        # Check if translation is needed
        if self._is_canonical_language(doc_language.language_code):
            # Document is already in canonical language
            logger.debug(
                f"Document {file_path} is in canonical language "
                f"({doc_language.language_code}), no translation needed"
            )
            return self._add_language_metadata(chunks, doc_language, translated=False)
        
        # Translate chunks
        logger.info(
            f"Translating document {file_path} from {doc_language.language_name} "
            f"to {self._canonical_language}"
        )
        
        return self._translate_chunks(chunks, doc_language)
    
    def _detect_document_language(self, content: str) -> Any:
        """Detect language of document content."""
        return self._lang_detector.detect_document(content)
    
    def _is_canonical_language(self, language_code: str) -> bool:
        """Check if language matches canonical language."""
        return language_code.lower() == self._canonical_language.lower()
    
    def _add_language_metadata(
        self,
        chunks: List[IngestedChunk],
        detection: Any,
        translated: bool = False,
    ) -> List[IngestedChunk]:
        """Add language metadata to chunks without translation."""
        result = []
        for chunk in chunks:
            meta = {
                **chunk.meta,
                "language_code": detection.language_code,
                "language_name": detection.language_name,
                "language_confidence": detection.confidence,
                "language_detection_method": detection.method,
                "was_translated": translated,
                "canonical_language": self._canonical_language,
            }
            result.append(IngestedChunk(content=chunk.content, meta=meta))
        return result
    
    def _translate_chunks(
        self,
        chunks: List[IngestedChunk],
        source_language: Any,
    ) -> List[IngestedChunk]:
        """Translate chunks to canonical language."""
        result = []
        
        for chunk in chunks:
            try:
                # Translate chunk content
                translation_result = self._translator.translate(
                    text=chunk.content,
                    target_language=self._canonical_language,
                    source_language=source_language.language_code,
                )
                
                # Build metadata with language information
                meta = {
                    **chunk.meta,
                    # Language detection info
                    "language_code": source_language.language_code,
                    "language_name": source_language.language_name,
                    "language_confidence": source_language.confidence,
                    "language_detection_method": source_language.method,
                    # Translation info
                    "was_translated": translation_result.was_translated,
                    "canonical_language": self._canonical_language,
                    "translation_method": translation_result.method,
                }
                
                # Preserve original text if configured
                if self._preserve_original and translation_result.was_translated:
                    meta["original_content"] = chunk.content
                    meta["original_language"] = source_language.language_code
                
                # Use translated content
                result.append(IngestedChunk(
                    content=translation_result.translated_text,
                    meta=meta,
                ))
                
            except Exception as e:
                logger.warning(
                    f"Translation failed for chunk, keeping original: {e}"
                )
                # Keep original on failure with metadata
                meta = {
                    **chunk.meta,
                    "language_code": source_language.language_code,
                    "language_name": source_language.language_name,
                    "was_translated": False,
                    "translation_error": str(e),
                    "canonical_language": self._canonical_language,
                }
                result.append(IngestedChunk(content=chunk.content, meta=meta))
        
        translated_count = sum(
            1 for c in result if c.meta.get("was_translated", False)
        )
        logger.info(
            f"Translated {translated_count}/{len(chunks)} chunks from "
            f"{source_language.language_name} to {self._canonical_language}"
        )
        
        return result
    
    def process_batch(
        self,
        file_paths: List[str],
        split_large_chunks: bool = False,
        max_chunk_size: int = 2000,
    ) -> List[IngestedChunk]:
        """
        Process multiple files with translation.
        
        Args:
            file_paths: List of file paths
            split_large_chunks: Whether to split large chunks
            max_chunk_size: Maximum chunk size before splitting
            
        Returns:
            List of all processed chunks from all files
        """
        all_chunks: List[IngestedChunk] = []
        
        for file_path in file_paths:
            try:
                chunks = self.process_file(
                    file_path, split_large_chunks, max_chunk_size
                )
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
        
        return all_chunks

    def process_paths(
        self,
        paths: Sequence[str],
        split_large_chunks: bool = False,
        max_chunk_size: int = 2000,
    ) -> Dict[str, List[IngestedChunk]]:
        """
        Process multiple paths with translation.

        Args:
            paths: File or directory paths
            split_large_chunks: Whether to split large chunks
            max_chunk_size: Maximum chunk size before splitting

        Returns:
            Dictionary mapping file paths to their chunks
        """
        files = iter_input_files(paths)
        results: Dict[str, List[IngestedChunk]] = {}

        for fp in files:
            file_path = str(fp)
            try:
                chunks = self.process_file(
                    file_path,
                    split_large_chunks=split_large_chunks,
                    max_chunk_size=max_chunk_size,
                )
                results[file_path] = chunks
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results[file_path] = []

        return results
    
    @property
    def translation_enabled(self) -> bool:
        """Check if translation is enabled."""
        return self._translation_enabled
    
    @property
    def canonical_language(self) -> str:
        """Get the canonical language code."""
        return self._canonical_language

