"""
Translation agent for RAG pipeline.

Translates text between languages using LLM for high-quality translations.
Supports document-level translation with chunking for long texts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from radiant_rag_mcp.llm.client import LLMClient
    from radiant_rag_mcp.agents.language_detection import LanguageDetection

logger = logging.getLogger(__name__)


# Language code to name mapping for prompts
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "es": "Spanish",
    "ar": "Arabic",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "de": "German",
    "fr": "French",
    "ko": "Korean",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "cs": "Czech",
    "sv": "Swedish",
    "ro": "Romanian",
    "el": "Greek",
    "hu": "Hungarian",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "ur": "Urdu",
    "fa": "Persian",
    "he": "Hebrew",
    "ms": "Malay",
    "sw": "Swahili",
    "zu": "Zulu",
    "af": "Afrikaans",
}


@dataclass
class TranslationResult:
    """Result of a translation operation."""
    
    original_text: str
    translated_text: str
    source_language: str  # ISO 639-1 code
    target_language: str  # ISO 639-1 code
    source_language_name: str
    target_language_name: str
    method: str  # "llm", "llm_chunked", "skip_same_language", "skip_empty"
    confidence: float = 1.0
    chunks_translated: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_text": self.original_text,
            "translated_text": self.translated_text,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "source_language_name": self.source_language_name,
            "target_language_name": self.target_language_name,
            "method": self.method,
            "confidence": self.confidence,
            "chunks_translated": self.chunks_translated,
        }
    
    @property
    def was_translated(self) -> bool:
        """Check if actual translation occurred."""
        return self.method in ("llm", "llm_chunked")
    
    def get_language_metadata(self) -> Dict[str, Any]:
        """Get language metadata for storage."""
        return {
            "original_language": self.source_language,
            "original_language_name": self.source_language_name,
            "translated_to": self.target_language,
            "translated_to_name": self.target_language_name,
            "translation_method": self.method,
            "was_translated": self.was_translated,
        }


class TranslationError(Exception):
    """Exception raised when translation fails."""
    pass


class TranslationAgent:
    """
    Translation agent using LLM for high-quality translations.
    
    Features:
    - LLM-based translation for high quality
    - Automatic chunking for long documents
    - Preserves formatting (paragraphs, lists, etc.)
    - Handles technical terms and code appropriately
    - Returns both original and translated text with metadata
    """
    
    def __init__(
        self,
        llm: "LLMClient",
        canonical_language: str = "en",
        max_chars_per_call: int = 4000,
        preserve_original: bool = True,
    ) -> None:
        """
        Initialize translation agent.
        
        Args:
            llm: LLM client for translation
            canonical_language: Target language for indexing (default: English)
            max_chars_per_call: Maximum characters per LLM call (for chunking)
            preserve_original: Whether to preserve original text in metadata
        """
        self._llm = llm
        self._canonical_language = canonical_language
        self._max_chars = max_chars_per_call
        self._preserve_original = preserve_original
    
    def translate(
        self,
        text: str,
        target_language: str = "en",
        source_language: Optional[str] = None,
    ) -> TranslationResult:
        """
        Translate text to target language.
        
        Args:
            text: Text to translate
            target_language: Target language code (default: "en")
            source_language: Source language code (auto-detect if None)
            
        Returns:
            TranslationResult with original and translated text
        """
        # Handle empty text
        if not text or len(text.strip()) < 1:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_language or "unknown",
                target_language=target_language,
                source_language_name=self._get_language_name(source_language or "unknown"),
                target_language_name=self._get_language_name(target_language),
                method="skip_empty",
                confidence=1.0,
            )
        
        # If source language matches target, no translation needed
        if source_language and source_language.lower() == target_language.lower():
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_language,
                target_language=target_language,
                source_language_name=self._get_language_name(source_language),
                target_language_name=self._get_language_name(target_language),
                method="skip_same_language",
                confidence=1.0,
            )
        
        # Check if chunking is needed
        if len(text) > self._max_chars:
            return self._translate_long_text(text, target_language, source_language)
        
        # Single translation call
        return self._translate_single(text, target_language, source_language)
    
    def _translate_single(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str],
    ) -> TranslationResult:
        """Translate a single chunk of text."""
        
        target_name = self._get_language_name(target_language)
        
        source_hint = ""
        if source_language:
            source_name = self._get_language_name(source_language)
            source_hint = f"\nSource language: {source_name}"
        else:
            source_name = "Auto-detected"
        
        system = f"""You are a professional translator. Translate the following text to {target_name}.

Rules:
- Return ONLY the translated text, no explanations or preamble
- Preserve all formatting (paragraphs, bullet points, numbered lists, etc.)
- Keep technical terms, code snippets, URLs, and proper nouns as appropriate
- Maintain the original tone and style
- For ambiguous terms, prefer the translation that fits the technical/professional context
- Do not add any notes, explanations, or commentary"""

        user = f"Translate the following text to {target_name}:{source_hint}\n\n{text}"
        
        response = self._llm.chat(
            self._llm.create_messages(system, user),
            retry_on_error=True,
        )
        
        if not response.success:
            logger.error(f"Translation failed: {response.error}")
            raise TranslationError(f"Translation failed: {response.error}")
        
        translated = response.content.strip()
        
        # Basic validation - translation shouldn't be empty
        if not translated:
            logger.warning("LLM returned empty translation, using original text")
            translated = text
        
        return TranslationResult(
            original_text=text,
            translated_text=translated,
            source_language=source_language or "auto",
            target_language=target_language,
            source_language_name=source_name,
            target_language_name=target_name,
            method="llm",
            confidence=0.95,  # LLM translations are generally high quality
            chunks_translated=1,
        )
    
    def _translate_long_text(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str],
    ) -> TranslationResult:
        """Translate long text by chunking at paragraph boundaries."""
        
        logger.info(
            f"Chunking long text ({len(text)} chars) for translation "
            f"(max {self._max_chars} per chunk)"
        )
        
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        translated_parts: List[str] = []
        chunks_count = 0
        
        current_chunk = ""
        current_chunk_paras: List[str] = []
        
        for para in paragraphs:
            # Check if adding this paragraph would exceed limit
            if len(current_chunk) + len(para) + 2 > self._max_chars:
                # Translate current chunk if not empty
                if current_chunk.strip():
                    result = self._translate_single(
                        current_chunk.strip(),
                        target_language,
                        source_language,
                    )
                    translated_parts.append(result.translated_text)
                    chunks_count += 1
                
                # Handle case where single paragraph exceeds limit
                if len(para) > self._max_chars:
                    # Split paragraph into sentences
                    sub_chunks = self._split_paragraph(para)
                    for sub_chunk in sub_chunks:
                        result = self._translate_single(
                            sub_chunk,
                            target_language,
                            source_language,
                        )
                        translated_parts.append(result.translated_text)
                        chunks_count += 1
                    current_chunk = ""
                    current_chunk_paras = []
                else:
                    current_chunk = para
                    current_chunk_paras = [para]
            else:
                current_chunk += ("\n\n" if current_chunk else "") + para
                current_chunk_paras.append(para)
        
        # Translate final chunk
        if current_chunk.strip():
            result = self._translate_single(
                current_chunk.strip(),
                target_language,
                source_language,
            )
            translated_parts.append(result.translated_text)
            chunks_count += 1
        
        # Join translated parts
        translated_text = "\n\n".join(translated_parts)
        
        source_name = self._get_language_name(source_language) if source_language else "Auto-detected"
        target_name = self._get_language_name(target_language)
        
        logger.info(f"Translated {chunks_count} chunks")
        
        return TranslationResult(
            original_text=text,
            translated_text=translated_text,
            source_language=source_language or "auto",
            target_language=target_language,
            source_language_name=source_name,
            target_language_name=target_name,
            method="llm_chunked",
            confidence=0.9,  # Slightly lower confidence for chunked translation
            chunks_translated=chunks_count,
        )
    
    def _split_paragraph(self, paragraph: str) -> List[str]:
        """Split a long paragraph into smaller chunks at sentence boundaries."""
        
        import re
        
        # Split at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        chunks: List[str] = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > self._max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If single sentence exceeds limit, split by words
                if len(sentence) > self._max_chars:
                    words = sentence.split()
                    word_chunk = ""
                    for word in words:
                        if len(word_chunk) + len(word) + 1 > self._max_chars:
                            if word_chunk:
                                chunks.append(word_chunk.strip())
                            word_chunk = word
                        else:
                            word_chunk += (" " if word_chunk else "") + word
                    if word_chunk:
                        current_chunk = word_chunk
                else:
                    current_chunk = sentence
            else:
                current_chunk += (" " if current_chunk else "") + sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def translate_to_canonical(
        self,
        text: str,
        source_language: Optional[str] = None,
    ) -> TranslationResult:
        """
        Translate text to the canonical (indexing) language.
        
        Args:
            text: Text to translate
            source_language: Source language code
            
        Returns:
            TranslationResult with translation to canonical language
        """
        return self.translate(
            text=text,
            target_language=self._canonical_language,
            source_language=source_language,
        )
    
    def translate_with_detection(
        self,
        text: str,
        detection: "LanguageDetection",
        target_language: Optional[str] = None,
    ) -> TranslationResult:
        """
        Translate text using pre-detected language information.
        
        Args:
            text: Text to translate
            detection: Language detection result
            target_language: Target language (default: canonical language)
            
        Returns:
            TranslationResult
        """
        target = target_language or self._canonical_language
        
        return self.translate(
            text=text,
            target_language=target,
            source_language=detection.language_code,
        )
    
    def translate_batch(
        self,
        texts: List[str],
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
    ) -> List[TranslationResult]:
        """
        Translate multiple texts.
        
        Args:
            texts: List of texts to translate
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            List of TranslationResult objects
        """
        target = target_language or self._canonical_language
        
        results = []
        for text in texts:
            try:
                result = self.translate(text, target, source_language)
                results.append(result)
            except TranslationError as e:
                logger.error(f"Batch translation failed for text: {e}")
                # Return original on failure
                results.append(TranslationResult(
                    original_text=text,
                    translated_text=text,
                    source_language=source_language or "unknown",
                    target_language=target,
                    source_language_name=self._get_language_name(source_language or "unknown"),
                    target_language_name=self._get_language_name(target),
                    method="error",
                    confidence=0.0,
                    metadata={"error": str(e)},
                ))
        
        return results
    
    def needs_translation(
        self,
        source_language: str,
        target_language: Optional[str] = None,
    ) -> bool:
        """
        Check if translation is needed between languages.
        
        Args:
            source_language: Source language code
            target_language: Target language code (default: canonical)
            
        Returns:
            True if translation is needed
        """
        target = target_language or self._canonical_language
        return source_language.lower() != target.lower()
    
    def _get_language_name(self, code: str) -> str:
        """Get language name from ISO 639-1 code."""
        if not code or code == "unknown":
            return "Unknown"
        return LANGUAGE_NAMES.get(code.lower(), code.upper())
    
    @property
    def canonical_language(self) -> str:
        """Get the canonical language code."""
        return self._canonical_language
    
    @property
    def canonical_language_name(self) -> str:
        """Get the canonical language name."""
        return self._get_language_name(self._canonical_language)
