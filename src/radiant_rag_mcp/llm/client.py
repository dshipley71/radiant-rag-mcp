"""
LLM client with retry logic and robust response handling.

Provides:
    - JSONParser: Robust JSON extraction from LLM responses
    - LLMResponse: Structured response objects
    - LLMClient: Chat client with retries
    - LLMClients: Container for all LLM dependencies
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, TYPE_CHECKING

import openai as _openai

@dataclass
class ChatMessage:
    """Simple chat message — replaces haystack.dataclasses.ChatMessage."""
    role: str
    text: str

    @classmethod
    def from_system(cls, text: str) -> "ChatMessage":
        return cls(role="system", text=text)

    @classmethod
    def from_user(cls, text: str) -> "ChatMessage":
        return cls(role="user", text=text)

    @classmethod
    def from_assistant(cls, text: str) -> "ChatMessage":
        return cls(role="assistant", text=text)

from radiant_rag_mcp.llm.local_models import LocalNLPModels

if TYPE_CHECKING:
    from radiant_rag_mcp.config import (
        LocalModelsConfig,
        OllamaConfig,
        ParsingConfig,
        LLMBackendConfig,
        EmbeddingBackendConfig,
        RerankingBackendConfig,
    )
    from radiant_rag_mcp.llm.backends.base import BaseLLMBackend, BaseEmbeddingBackend, BaseRerankingBackend

logger = logging.getLogger(__name__)

T = TypeVar("T")

# HTTP status codes that indicate a non-retryable client error.
# Retrying these wastes time since the request itself is invalid.
_NON_RETRYABLE_STATUS_CODES = (400, 401, 403, 404, 422)


def _is_non_retryable(error: Exception) -> bool:
    """Check if an error is a non-retryable client error (4xx)."""
    err_str = str(error)
    for code in _NON_RETRYABLE_STATUS_CODES:
        if f"Error code: {code}" in err_str or f"status_code: {code}" in err_str:
            return True
    # Also check for openai-specific exception attributes
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if isinstance(status, int) and 400 <= status < 500:
        return True
    return False


class JSONParser:
    """
    Robust JSON parser for LLM responses.

    Handles common issues:
        - Markdown code blocks
        - Leading/trailing text
        - Missing quotes on keys
        - Trailing commas
    """

    # Patterns for extracting JSON from responses
    JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
    JSON_OBJECT_PATTERN = re.compile(r"\{[\s\S]*\}")
    JSON_ARRAY_PATTERN = re.compile(r"\[[\s\S]*\]")

    @classmethod
    def extract_json_string(cls, text: str) -> Optional[str]:
        """
        Extract JSON string from text that may contain markdown or other content.

        Args:
            text: Raw text potentially containing JSON

        Returns:
            Extracted JSON string, or None if not found
        """
        text = text.strip()

        # Try to extract from markdown code block first
        match = cls.JSON_BLOCK_PATTERN.search(text)
        if match:
            return match.group(1).strip()

        # Try to find raw JSON object
        match = cls.JSON_OBJECT_PATTERN.search(text)
        if match:
            return match.group(0)

        # Try to find raw JSON array
        match = cls.JSON_ARRAY_PATTERN.search(text)
        if match:
            return match.group(0)

        return None

    @classmethod
    def clean_json_string(cls, json_str: str) -> str:
        """
        Clean common JSON formatting issues.

        Args:
            json_str: Raw JSON string

        Returns:
            Cleaned JSON string
        """
        # Remove trailing commas before } or ]
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

        # Remove comment-only lines (// style).
        # Only strip lines that begin with optional whitespace then //
        # to avoid corrupting URLs (https://...) inside JSON strings.
        json_str = re.sub(r"^\s*//[^\n]*$", "", json_str, flags=re.MULTILINE)

        return json_str

    @classmethod
    def repair_truncated_json(cls, json_str: str) -> Optional[str]:
        """
        Attempt to repair JSON truncated by LLM max_tokens limits.

        Walks the string tracking open/close brackets and quotes,
        then appends whatever closing characters are needed.

        Args:
            json_str: Potentially truncated JSON string

        Returns:
            Repaired JSON string, or None if repair is not feasible
        """
        if not json_str:
            return None

        # Strip trailing whitespace and incomplete trailing tokens
        # (e.g. a key name that was cut off mid-word)
        repaired = json_str.rstrip()

        # Track nesting state
        stack: list[str] = []  # expected closing chars
        in_string = False
        escape_next = False

        for ch in repaired:
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                if in_string:
                    escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in ("}", "]"):
                if stack and stack[-1] == ch:
                    stack.pop()

        # If nothing to close, repair won't help
        if not stack and not in_string:
            return None

        # Close open string if needed
        if in_string:
            repaired += '"'

        # Remove a possible trailing comma before we close containers
        repaired = re.sub(r",\s*$", "", repaired)

        # Close all open containers in reverse order
        repaired += "".join(reversed(stack))

        return repaired

    @classmethod
    def parse(
        cls,
        text: str,
        default: Optional[T] = None,
        expected_type: Optional[Type[T]] = None,
    ) -> Union[Dict[str, Any], List[Any], T, None]:
        """
        Parse JSON from LLM response text.

        Args:
            text: Raw LLM response
            default: Default value if parsing fails
            expected_type: Expected return type (dict or list)

        Returns:
            Parsed JSON data, or default if parsing fails
        """
        if not text:
            return default

        # Extract JSON string
        json_str = cls.extract_json_string(text)
        if not json_str:
            # Try parsing the whole text
            json_str = text.strip()

        # Clean the JSON
        json_str = cls.clean_json_string(json_str)

        # Attempt parsing
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError:
            # Try repairing truncated JSON (common with LLM max_tokens limits)
            repaired = cls.repair_truncated_json(json_str)
            if repaired:
                try:
                    result = json.loads(repaired)
                    logger.debug("Parsed JSON after truncation repair")
                except json.JSONDecodeError as e2:
                    logger.debug(f"JSON parse error after repair attempt: {e2}")
                    return default
            else:
                return default

        # Validate type if specified
        if expected_type is not None:
            if expected_type == dict and not isinstance(result, dict):
                logger.warning(f"Expected dict but got {type(result).__name__}")
                return default
            if expected_type == list and not isinstance(result, list):
                logger.warning(f"Expected list but got {type(result).__name__}")
                return default

        return result


@dataclass
class LLMResponse:
    """Structured LLM response with metadata."""

    content: str
    raw_response: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None
    retries: int = 0
    latency_ms: float = 0.0


class LLMClient:
    """
    LLM client with retry logic and robust response handling.

    Wraps an OpenAI-compatible API (Ollama, vLLM, OpenAI) with additional features:
        - Automatic retry on failures
        - JSON response parsing
        - Structured response objects
    """

    def __init__(
        self,
        ollama_config: "OllamaConfig",
        parsing_config: "ParsingConfig",
    ) -> None:
        """
        Initialize LLM client.

        Args:
            ollama_config: Ollama/LLM configuration
            parsing_config: Response parsing configuration
        """
        self._ollama_config = ollama_config
        self._parsing_config = parsing_config

        # Lazy-initialized openai client (no network call at construction time)
        self._openai_client: Optional[_openai.OpenAI] = None

        logger.info(f"Configured LLM client: model={ollama_config.chat_model}")

    def _get_client(self) -> "_openai.OpenAI":
        """Return the OpenAI client, creating it lazily on first call."""
        if self._openai_client is None:
            self._openai_client = _openai.OpenAI(
                api_key=self._ollama_config.openai_api_key or "ollama",
                base_url=self._ollama_config.openai_base_url,
                timeout=self._ollama_config.timeout,
            )
        return self._openai_client

    @staticmethod
    def create_messages(system: str, user: str) -> List[ChatMessage]:
        """
        Create chat messages for a request.

        Args:
            system: System prompt
            user: User message

        Returns:
            List of ChatMessage objects
        """
        return [
            ChatMessage.from_system(system),
            ChatMessage.from_user(user),
        ]

    def chat(
        self,
        messages: List[ChatMessage],
        retry_on_error: bool = True,
    ) -> LLMResponse:
        """
        Send chat request to LLM.

        Args:
            messages: Chat messages
            retry_on_error: Whether to retry on failures

        Returns:
            LLMResponse object
        """
        max_retries = self._ollama_config.max_retries if retry_on_error else 1
        retry_delay = self._ollama_config.retry_delay
        last_error = None

        for attempt in range(max_retries):
            start_time = time.time()

            try:
                result = self._get_client().chat.completions.create(
                    model=self._ollama_config.chat_model,
                    messages=[{"role": m.role, "content": m.text} for m in messages],
                )
                latency = (time.time() - start_time) * 1000

                content = result.choices[0].message.content or ""

                return LLMResponse(
                    content=content,
                    raw_response={},
                    success=True,
                    retries=attempt,
                    latency_ms=latency,
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(f"LLM request failed (attempt {attempt + 1}/{max_retries}): {e}")

                # Don't retry on non-retryable client errors (400, 401, etc.)
                if _is_non_retryable(e):
                    logger.debug("Non-retryable error, skipping remaining attempts")
                    break

                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff

        return LLMResponse(
            content="",
            raw_response={},
            success=False,
            error=last_error,
            retries=max_retries,
            latency_ms=0.0,
        )

    def chat_json(
        self,
        system: str,
        user: str,
        default: Optional[T] = None,
        expected_type: Optional[Type[T]] = None,
        retry_on_parse_error: bool = True,
    ) -> tuple[Union[Dict[str, Any], List[Any], T, None], LLMResponse]:
        """
        Send chat request expecting JSON response.

        If parsing fails, can optionally retry with a clarification message.

        Args:
            system: System prompt
            user: User message
            default: Default value if parsing fails
            expected_type: Expected JSON type (dict or list)
            retry_on_parse_error: Whether to retry on JSON parse failures

        Returns:
            Tuple of (parsed_data, LLMResponse)
        """
        messages = self.create_messages(system, user)
        response = self.chat(messages)

        if not response.success:
            return default, response

        # Try to parse JSON
        parsed = JSONParser.parse(
            response.content,
            default=None,
            expected_type=expected_type,
        )

        if parsed is not None:
            return parsed, response

        # Parsing failed - optionally retry with clarification
        if retry_on_parse_error and self._parsing_config.max_retries > 0:
            logger.debug("JSON parse failed, retrying with clarification...")

            # Add clarification message
            clarification = (
                "Your previous response could not be parsed as valid JSON. "
                "Please respond with ONLY valid JSON, no markdown code blocks, "
                "no explanatory text, just the raw JSON data."
            )
            retry_messages = messages + [
                ChatMessage.from_assistant(response.content),
                ChatMessage.from_user(clarification),
            ]

            for retry_attempt in range(self._parsing_config.max_retries):
                time.sleep(self._parsing_config.retry_delay)

                retry_response = self.chat(retry_messages, retry_on_error=False)
                if not retry_response.success:
                    continue

                parsed = JSONParser.parse(
                    retry_response.content,
                    default=None,
                    expected_type=expected_type,
                )

                if parsed is not None:
                    retry_response.retries = response.retries + retry_attempt + 1
                    return parsed, retry_response

                # Update messages for next retry
                retry_messages = retry_messages + [
                    ChatMessage.from_assistant(retry_response.content),
                    ChatMessage.from_user("That still wasn't valid JSON. Please try again with just the JSON."),
                ]

        # Log failure if configured
        if self._parsing_config.log_failures:
            logger.warning(
                f"Failed to parse JSON from LLM response. "
                f"Raw content: {response.content[:500]}..."
            )

        return default, response


class LLMClientBackendAdapter:
    """
    Adapter that wraps a BaseLLMBackend to provide the LLMClient interface.

    Maintains backward compatibility with existing code.
    """

    def __init__(
        self,
        backend: "BaseLLMBackend",
        parsing_config: "ParsingConfig",
    ) -> None:
        """
        Initialize adapter.

        Args:
            backend: LLM backend implementation
            parsing_config: Response parsing configuration
        """
        self._backend = backend
        self._parsing_config = parsing_config

    def _get_client(self) -> "_openai.OpenAI":
        """Return the OpenAI client, creating it lazily on first call."""
        if self._openai_client is None:
            self._openai_client = _openai.OpenAI(
                api_key=self._ollama_config.openai_api_key or "ollama",
                base_url=self._ollama_config.openai_base_url,
                timeout=self._ollama_config.timeout,
            )
        return self._openai_client

    @staticmethod
    def create_messages(system: str, user: str) -> List[ChatMessage]:
        """
        Create chat messages for a request.

        Args:
            system: System prompt
            user: User message

        Returns:
            List of ChatMessage objects
        """
        return [
            ChatMessage.from_system(system),
            ChatMessage.from_user(user),
        ]

    def chat(
        self,
        messages: List[ChatMessage],
        retry_on_error: bool = True,
    ) -> LLMResponse:
        """
        Send chat request to LLM.

        Args:
            messages: Chat messages
            retry_on_error: Whether to retry on failures (ignored for backends)

        Returns:
            LLMResponse object
        """
        # Convert Haystack ChatMessages to dict format
        message_dicts = []
        for msg in messages:
            # Extract role from meta if available
            role = getattr(msg, 'role', None)
            if role is None:
                role = msg.meta.get('role', 'user') if hasattr(msg, 'meta') and msg.meta else 'user'

            # Extract content
            if hasattr(msg, 'text'):
                content = msg.text
            elif hasattr(msg, 'content'):
                content = msg.content
            else:
                content = str(msg)

            message_dicts.append({"role": role.value if hasattr(role, 'value') else str(role), "content": content})

        try:
            # Call backend
            backend_response = self._backend.chat(message_dicts)

            # Convert backend response to LLMResponse
            return LLMResponse(
                content=backend_response.content,
                raw_response=backend_response.meta,
                success=True,
                retries=0,
                latency_ms=backend_response.meta.get("latency_ms", 0.0),
            )

        except Exception as e:
            logger.error(f"LLM backend request failed: {e}")
            return LLMResponse(
                content="",
                raw_response={},
                success=False,
                error=str(e),
                retries=0,
                latency_ms=0.0,
            )

    def chat_json(
        self,
        system: str,
        user: str,
        default: Optional[T] = None,
        expected_type: Optional[Type[T]] = None,
        retry_on_parse_error: bool = True,
    ) -> tuple[Union[Dict[str, Any], List[Any], T, None], LLMResponse]:
        """
        Send chat request expecting JSON response.

        Args:
            system: System prompt
            user: User message
            default: Default value if parsing fails
            expected_type: Expected JSON type (dict or list)
            retry_on_parse_error: Whether to retry on JSON parse failures

        Returns:
            Tuple of (parsed_data, LLMResponse)
        """
        messages = self.create_messages(system, user)
        response = self.chat(messages)

        if not response.success:
            return default, response

        # Try to parse JSON
        parsed = JSONParser.parse(
            response.content,
            default=None,
            expected_type=expected_type,
        )

        if parsed is not None:
            return parsed, response

        # Parsing failed
        if self._parsing_config.log_failures:
            logger.warning(
                f"Failed to parse JSON from LLM response. "
                f"Raw content: {response.content[:500]}..."
            )

        return default, response


class LocalNLPModelsBackendAdapter:
    """
    Adapter that wraps embedding and reranking backends to provide the LocalNLPModels interface.

    Maintains backward compatibility with existing code.
    """

    def __init__(
        self,
        embedding_backend: "BaseEmbeddingBackend",
        reranking_backend: "BaseRerankingBackend",
        embedding_dim: int,
    ) -> None:
        """
        Initialize adapter.

        Args:
            embedding_backend: Embedding backend implementation
            reranking_backend: Reranking backend implementation
            embedding_dim: Embedding dimension
        """
        self._embedding_backend = embedding_backend
        self._reranking_backend = reranking_backend
        self.embedding_dim = embedding_dim

    def embed(
        self,
        texts: List[str],
        normalize: bool = True,
        use_cache: bool = True,
    ) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            normalize: Whether to L2-normalize
            use_cache: Whether to use embedding cache (always True for backends with built-in caching)

        Returns:
            List of embedding vectors
        """
        return self._embedding_backend.embed(
            texts=texts,
            normalize=normalize,
            show_progress=False,
            batch_size=32,
        )

    def embed_single(
        self,
        text: str,
        normalize: bool = True,
        use_cache: bool = True,
    ) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            normalize: Whether to L2-normalize
            use_cache: Whether to use embedding cache (always True for backends with built-in caching)

        Returns:
            Embedding vector
        """
        return self._embedding_backend.embed_single(
            text=text,
            normalize=normalize,
        )

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[tuple[int, float]]:
        """
        Rerank documents by relevance to query.

        Args:
            query: Query text
            documents: List of document texts
            top_k: Optional limit on results

        Returns:
            List of (document_index, score) tuples, sorted by score descending
        """
        return self._reranking_backend.rerank(
            query=query,
            documents=documents,
            top_k=top_k,
        )


@dataclass(frozen=True)
class LLMClients:
    """
    Container for all LLM-related clients.

    Convenience class for passing around LLM dependencies.
    """

    chat: LLMClient
    local: LocalNLPModels

    @staticmethod
    def build(
        ollama_config: "OllamaConfig",
        local_config: "LocalModelsConfig",
        parsing_config: "ParsingConfig",
        embedding_cache_size: int = 10000,
        # New backend configurations (optional)
        llm_backend_config: Optional["LLMBackendConfig"] = None,
        embedding_backend_config: Optional["EmbeddingBackendConfig"] = None,
        reranking_backend_config: Optional["RerankingBackendConfig"] = None,
    ) -> "LLMClients":
        """
        Build all LLM clients from configuration.

        Supports both old configuration format (ollama_config + local_config) and
        new backend configuration format (llm_backend_config + embedding_backend_config + reranking_backend_config).

        Args:
            ollama_config: Ollama/LLM configuration (old format, used as fallback)
            local_config: Local models configuration (old format, used as fallback)
            parsing_config: Response parsing configuration
            embedding_cache_size: Size of embedding cache (default 10000)
            llm_backend_config: Optional LLM backend configuration (new format)
            embedding_backend_config: Optional embedding backend configuration (new format)
            reranking_backend_config: Optional reranking backend configuration (new format)

        Returns:
            LLMClients instance
        """
        # Check if new backend configurations are provided
        use_new_backends = (
            llm_backend_config is not None
            or embedding_backend_config is not None
            or reranking_backend_config is not None
        )

        if use_new_backends:
            # Use new backend system
            logger.info("Using new backend configuration system")

            from radiant_rag_mcp.llm.backends.factory import (
                create_llm_backend,
                create_embedding_backend,
                create_reranking_backend,
            )

            # Create LLM backend (or fallback to ollama)
            if llm_backend_config is not None:
                llm_backend = create_llm_backend(llm_backend_config)
                chat_client = LLMClientBackendAdapter(llm_backend, parsing_config)
            else:
                # Fallback to old config
                logger.info("LLM backend: using ollama config (fallback)")
                chat_client = LLMClient(ollama_config, parsing_config)

            # Create embedding backend (or fallback to local)
            if embedding_backend_config is not None:
                embedding_backend = create_embedding_backend(embedding_backend_config)
            else:
                # Fallback to old config
                logger.info("Embedding backend: using local_config (fallback)")
                from radiant_rag_mcp.llm.backends.embedding_backends import SentenceTransformersEmbeddingBackend
                embedding_backend = SentenceTransformersEmbeddingBackend(
                    model_name=local_config.embed_model_name,
                    device=local_config.device,
                    cache_size=embedding_cache_size,
                )

            # Create reranking backend (or fallback to local)
            if reranking_backend_config is not None:
                # If reranking uses LLM, pass the LLM backend
                if reranking_backend_config.backend_type in ["ollama", "vllm", "openai"]:
                    # Get the LLM backend
                    if llm_backend_config is not None:
                        llm_backend_for_rerank = create_llm_backend(llm_backend_config)
                    else:
                        # Need to create an LLM backend from ollama config
                        from radiant_rag_mcp.config import LLMBackendConfig
                        temp_llm_config = LLMBackendConfig(
                            backend_type="ollama",
                            base_url=ollama_config.openai_base_url,
                            api_key=ollama_config.openai_api_key,
                            model=ollama_config.chat_model,
                        )
                        llm_backend_for_rerank = create_llm_backend(temp_llm_config)

                    reranking_backend = create_reranking_backend(
                        reranking_backend_config,
                        llm_backend=llm_backend_for_rerank,
                    )
                else:
                    reranking_backend = create_reranking_backend(reranking_backend_config)
            else:
                # Fallback to old config
                logger.info("Reranking backend: using local_config (fallback)")
                from radiant_rag_mcp.llm.backends.reranking_backends import CrossEncoderRerankingBackend
                reranking_backend = CrossEncoderRerankingBackend(
                    model_name=local_config.cross_encoder_name,
                    device=local_config.device,
                )

            # Create LocalNLPModels wrapper using backends
            local_models = LocalNLPModelsBackendAdapter(
                embedding_backend=embedding_backend,
                reranking_backend=reranking_backend,
                embedding_dim=embedding_backend.embedding_dimension,
            )

            return LLMClients(
                chat=chat_client,
                local=local_models,
            )

        else:
            # Use old configuration system (backward compatibility)
            logger.info("Using legacy ollama/local_models configuration")
            return LLMClients(
                chat=LLMClient(ollama_config, parsing_config),
                local=LocalNLPModels.build(local_config, cache_size=embedding_cache_size),
            )
