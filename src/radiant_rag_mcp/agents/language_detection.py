"""
Language detection agent for RAG pipeline.

Detects the language of text using FastText with optional LLM fallback
for low-confidence or ambiguous cases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from radiant_rag_mcp.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Try to import fasttext
try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False
    logger.warning(
        "fasttext not available. Install with: pip install fasttext"
    )

# Try to import fast-langdetect as fallback
try:
    from fast_langdetect import detect as fast_detect
    FAST_LANGDETECT_AVAILABLE = True
except ImportError:
    FAST_LANGDETECT_AVAILABLE = False

# Import model manager
try:
    from radiant_rag_mcp.utils.model_manager import ModelManager, ModelDownloadError
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    MODEL_MANAGER_AVAILABLE = False
    logger.warning("ModelManager not available")


# ISO 639-1 language code to name mapping (common languages)
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
    "sk": "Slovak",
    "uk": "Ukrainian",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "lt": "Lithuanian",
    "sl": "Slovenian",
    "lv": "Latvian",
    "et": "Estonian",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
    "fa": "Persian",
    "he": "Hebrew",
    "ms": "Malay",
    "tl": "Tagalog",
    "sw": "Swahili",
    "zu": "Zulu",
    "af": "Afrikaans",
    "ca": "Catalan",
    "eu": "Basque",
    "gl": "Galician",
    "cy": "Welsh",
    "ga": "Irish",
    "mt": "Maltese",
    "is": "Icelandic",
    "mk": "Macedonian",
    "sq": "Albanian",
    "sr": "Serbian",
    "bs": "Bosnian",
    "az": "Azerbaijani",
    "ka": "Georgian",
    "hy": "Armenian",
    "kk": "Kazakh",
    "uz": "Uzbek",
    "mn": "Mongolian",
    "ne": "Nepali",
    "si": "Sinhala",
    "km": "Khmer",
    "lo": "Lao",
    "my": "Burmese",
}


@dataclass
class LanguageDetection:
    """Result of language detection."""
    
    language_code: str  # ISO 639-1 code (e.g., "en", "zh", "fr")
    language_name: str  # Human readable name
    confidence: float   # 0-1 confidence score
    method: str         # Detection method used: "fast", "llm", "fallback"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "language_code": self.language_code,
            "language_name": self.language_name,
            "confidence": self.confidence,
            "method": self.method,
        }
    
    @property
    def is_english(self) -> bool:
        """Check if detected language is English."""
        return self.language_code.lower() == "en"


class LanguageDetectionAgent:
    """
    Language detection agent using FastText with LLM fallback.

    FastText language detection model provides:
    - Very fast detection (~0.1ms per text)
    - Support for 176 languages
    - Good accuracy even on short text

    For low-confidence detections, falls back to LLM-based detection
    for more accurate results on ambiguous content.
    """

    def __init__(
        self,
        llm: Optional["LLMClient"] = None,
        method: str = "fast",
        min_confidence: float = 0.7,
        use_llm_fallback: bool = True,
        fallback_language: str = "en",
        model_path: str = "./data/models/fasttext/lid.176.ftz",
        model_url: str = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
        auto_download: bool = True,
        verify_checksum: bool = False,
    ) -> None:
        """
        Initialize language detection agent.

        Args:
            llm: LLM client for fallback detection
            method: Detection method ("fast", "llm", "auto")
            min_confidence: Minimum confidence for fasttext
            use_llm_fallback: Whether to use LLM for low-confidence cases
            fallback_language: Default language if detection fails
            model_path: Path to FastText model file
            model_url: URL to download model from if not found
            auto_download: Whether to auto-download model if missing
            verify_checksum: Whether to verify model checksum
        """
        self._llm = llm
        self._method = method
        self._min_confidence = min_confidence
        self._use_llm_fallback = use_llm_fallback
        self._fallback_language = fallback_language
        self._model_path = model_path
        self._model_url = model_url
        self._auto_download = auto_download
        self._verify_checksum = verify_checksum
        self._fasttext_model = None

        # Try to load FastText model if using fast method
        if method == "fast" or method == "auto":
            self._load_fasttext_model()

        # Check if we have a working detection method
        if method == "fast" and self._fasttext_model is None:
            if FAST_LANGDETECT_AVAILABLE:
                logger.info("Using fast-langdetect as fallback")
            elif self._llm and use_llm_fallback:
                logger.warning(
                    "FastText model not available, falling back to LLM-only detection"
                )
                self._method = "llm"
            else:
                logger.warning(
                    "No detection method available! Install fasttext or fast-langdetect, "
                    "or provide an LLM client"
                )

    def _load_fasttext_model(self) -> None:
        """Load FastText model from configured path."""

        if not FASTTEXT_AVAILABLE:
            logger.debug("fasttext library not available, skipping model load")
            return

        if not MODEL_MANAGER_AVAILABLE:
            logger.warning("ModelManager not available, cannot download model")
            # Try to load from path if it exists
            if Path(self._model_path).exists():
                try:
                    # Suppress fasttext warnings
                    fasttext.FastText.eprint = lambda *args, **kwargs: None
                    self._fasttext_model = fasttext.load_model(self._model_path)
                    logger.info(f"Loaded FastText model from {self._model_path}")
                except Exception as e:
                    logger.error(f"Failed to load FastText model: {e}")
            return

        try:
            # Ensure model exists (download if needed)
            model_path = ModelManager.ensure_model(
                model_path=self._model_path,
                download_url=self._model_url if self._auto_download else None,
                auto_download=self._auto_download,
                checksum=None if not self._verify_checksum else None,  # Add checksum if needed
            )

            if model_path:
                # Suppress fasttext warnings
                fasttext.FastText.eprint = lambda *args, **kwargs: None
                self._fasttext_model = fasttext.load_model(str(model_path))
                logger.info(f"Loaded FastText model from {model_path}")
            else:
                logger.warning(f"FastText model not available at {self._model_path}")

        except ModelDownloadError as e:
            logger.error(f"Failed to download FastText model: {e}")
        except Exception as e:
            logger.error(f"Failed to load FastText model: {e}")
    
    def detect(self, text: str) -> LanguageDetection:
        """
        Detect the language of the given text.
        
        Args:
            text: Text to detect language for
            
        Returns:
            LanguageDetection result
        """
        # Handle empty or very short text
        if not text or len(text.strip()) < 3:
            return LanguageDetection(
                language_code=self._fallback_language,
                language_name=self._get_language_name(self._fallback_language),
                confidence=0.0,
                method="too_short",
            )
        
        # Clean text for detection (remove excess whitespace)
        clean_text = " ".join(text.split())
        
        # Route based on method
        if self._method == "llm":
            return self._detect_with_llm(clean_text)
        elif self._method == "fast":
            return self._detect_with_fast(clean_text)
        else:  # auto
            return self._detect_auto(clean_text)
    
    def _detect_with_fast(self, text: str) -> LanguageDetection:
        """Detect language using FastText model."""

        # Try fasttext model first (preferred)
        if self._fasttext_model is not None:
            try:
                # Sample longer texts to improve accuracy
                sample = text[:5000] if len(text) > 5000 else text

                # FastText predict returns: (labels, probabilities)
                # labels is a tuple like ('__label__en',)
                # probabilities is a tuple like (0.9999,)
                predictions = self._fasttext_model.predict(sample.replace('\n', ' '), k=1)

                if predictions and len(predictions) == 2:
                    labels, probabilities = predictions

                    if len(labels) > 0 and len(probabilities) > 0:
                        # Extract language code from label (remove __label__ prefix)
                        label = str(labels[0])
                        lang_code = label.replace('__label__', '').strip()
                        confidence = float(probabilities[0])

                        # Check if confidence meets threshold
                        if confidence >= self._min_confidence:
                            return LanguageDetection(
                                language_code=lang_code,
                                language_name=self._get_language_name(lang_code),
                                confidence=confidence,
                                method="fasttext",
                            )

                        # Low confidence - try LLM fallback if enabled
                        if self._use_llm_fallback and self._llm:
                            logger.debug(
                                f"Low confidence ({confidence:.2f}) from FastText, "
                                f"trying LLM fallback"
                            )
                            return self._detect_with_llm(text)

                        # Return low-confidence result
                        return LanguageDetection(
                            language_code=lang_code,
                            language_name=self._get_language_name(lang_code),
                            confidence=confidence,
                            method="fasttext_low_confidence",
                        )

            except Exception as e:
                logger.warning(f"FastText detection failed: {e}")
                # Fall through to fast-langdetect fallback

        # Fallback to fast-langdetect if available
        if FAST_LANGDETECT_AVAILABLE:
            try:
                sample = text[:5000] if len(text) > 5000 else text
                result = fast_detect(sample)

                # Handle different return formats from fast-langdetect
                if isinstance(result, dict):
                    lang_code = result.get("lang", self._fallback_language)
                    confidence = float(result.get("score", 0.0))
                elif isinstance(result, (list, tuple)):
                    if len(result) > 0:
                        first_result = result[0] if isinstance(result[0], (list, tuple)) else result
                        lang_code = str(first_result[0]) if len(first_result) > 0 else self._fallback_language
                        confidence = float(first_result[1]) if len(first_result) > 1 else 0.0
                    else:
                        lang_code = self._fallback_language
                        confidence = 0.0
                elif hasattr(result, 'lang') and hasattr(result, 'score'):
                    lang_code = str(result.lang)
                    confidence = float(result.score)
                else:
                    lang_code = str(result)[:2].lower() if result else self._fallback_language
                    confidence = 0.5

                lang_code = lang_code.lower().strip()

                if confidence >= self._min_confidence:
                    return LanguageDetection(
                        language_code=lang_code,
                        language_name=self._get_language_name(lang_code),
                        confidence=confidence,
                        method="fast_langdetect",
                    )

                if self._use_llm_fallback and self._llm:
                    return self._detect_with_llm(text)

                return LanguageDetection(
                    language_code=lang_code,
                    language_name=self._get_language_name(lang_code),
                    confidence=confidence,
                    method="fast_langdetect_low_confidence",
                )

            except Exception as e:
                logger.warning(f"fast-langdetect failed: {e}")

        # Last resort - try LLM or return fallback
        if self._use_llm_fallback and self._llm:
            return self._detect_with_llm(text)

        return LanguageDetection(
            language_code=self._fallback_language,
            language_name=self._get_language_name(self._fallback_language),
            confidence=0.5,
            method="fallback_no_detector",
        )
    
    def _detect_with_llm(self, text: str) -> LanguageDetection:
        """Detect language using LLM."""
        
        if not self._llm:
            return LanguageDetection(
                language_code=self._fallback_language,
                language_name=self._get_language_name(self._fallback_language),
                confidence=0.5,
                method="fallback_no_llm",
            )
        
        system = """You are a language detection expert. Identify the primary language of the given text.

Return JSON only:
{
  "language_code": "ISO 639-1 two-letter code (e.g., en, zh, fr, de, ja, ko, es)",
  "language_name": "Full language name in English",
  "confidence": 0.0-1.0
}

Rules:
- Use ISO 639-1 two-letter codes
- For Chinese, use "zh" (covers Simplified and Traditional)
- For mixed-language text, identify the dominant language
- Confidence should reflect how certain you are about the detection"""

        # Use a sample for very long texts
        sample = text[:2000] if len(text) > 2000 else text
        
        user = f"Detect the language of this text:\n\n{sample}\n\nReturn JSON only."
        
        result, response = self._llm.chat_json(
            system=system,
            user=user,
            default={
                "language_code": self._fallback_language,
                "language_name": self._get_language_name(self._fallback_language),
                "confidence": 0.5,
            },
            expected_type=dict,
        )
        
        if not response.success:
            logger.warning(f"LLM language detection failed: {response.error}")
            return LanguageDetection(
                language_code=self._fallback_language,
                language_name=self._get_language_name(self._fallback_language),
                confidence=0.5,
                method="fallback_llm_error",
            )
        
        lang_code = str(result.get("language_code", self._fallback_language)).lower()
        lang_name = str(result.get("language_name", self._get_language_name(lang_code)))
        confidence = float(result.get("confidence", 0.8))
        
        return LanguageDetection(
            language_code=lang_code,
            language_name=lang_name,
            confidence=confidence,
            method="llm",
        )
    
    def _detect_auto(self, text: str) -> LanguageDetection:
        """Auto-detect using best available method."""
        
        if FAST_LANGDETECT_AVAILABLE:
            return self._detect_with_fast(text)
        elif self._llm:
            return self._detect_with_llm(text)
        else:
            return LanguageDetection(
                language_code=self._fallback_language,
                language_name=self._get_language_name(self._fallback_language),
                confidence=0.5,
                method="fallback_no_detector",
            )
    
    def detect_batch(self, texts: List[str]) -> List[LanguageDetection]:
        """
        Detect language for multiple texts.
        
        Args:
            texts: List of texts to detect
            
        Returns:
            List of LanguageDetection results
        """
        return [self.detect(text) for text in texts]
    
    def detect_document(
        self,
        content: str,
        sample_size: int = 5000,
    ) -> LanguageDetection:
        """
        Detect language of a document.
        
        Uses a sample from the beginning of the document for efficiency.
        
        Args:
            content: Full document content
            sample_size: Number of characters to sample
            
        Returns:
            LanguageDetection result
        """
        # Sample from beginning (usually more representative)
        sample = content[:sample_size] if len(content) > sample_size else content
        return self.detect(sample)
    
    def detect_with_context(
        self,
        text: str,
        document_language: Optional[str] = None,
    ) -> LanguageDetection:
        """
        Detect language with document context hint.
        
        When processing chunks, the document language can be used as a hint
        to improve detection accuracy for short chunks.
        
        Args:
            text: Text to detect
            document_language: Optional language code of parent document
            
        Returns:
            LanguageDetection result
        """
        # First, try normal detection
        result = self.detect(text)
        
        # If low confidence and document language is provided, use it
        if result.confidence < self._min_confidence and document_language:
            logger.debug(
                f"Using document language hint: {document_language} "
                f"(detection confidence was {result.confidence:.2f})"
            )
            return LanguageDetection(
                language_code=document_language,
                language_name=self._get_language_name(document_language),
                confidence=result.confidence,
                method=f"{result.method}_with_context_hint",
            )
        
        return result
    
    def _get_language_name(self, code: str) -> str:
        """Get language name from ISO 639-1 code."""
        return LANGUAGE_NAMES.get(code.lower(), code.upper())
    
    def is_canonical_language(
        self,
        detection: LanguageDetection,
        canonical: str = "en",
    ) -> bool:
        """
        Check if detected language matches canonical language.
        
        Args:
            detection: Language detection result
            canonical: Canonical language code
            
        Returns:
            True if languages match
        """
        return detection.language_code.lower() == canonical.lower()
