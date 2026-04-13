"""
Factory functions for creating LLM, embedding, and reranking backends.

Provides simple API for instantiating backends based on configuration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from radiant_rag_mcp.llm.backends.base import (
    BaseLLMBackend,
    BaseEmbeddingBackend,
    BaseRerankingBackend,
)
from radiant_rag_mcp.llm.backends.llm_backends import (
    OpenAICompatibleLLMBackend,
    LocalHuggingFaceLLMBackend,
)
from radiant_rag_mcp.llm.backends.embedding_backends import (
    SentenceTransformersEmbeddingBackend,
    OpenAICompatibleEmbeddingBackend,
    HuggingFaceTransformersEmbeddingBackend,
)
from radiant_rag_mcp.llm.backends.reranking_backends import (
    CrossEncoderRerankingBackend,
    LLMRerankingBackend,
    OpenAICompatibleRerankingBackend,
)

if TYPE_CHECKING:
    from radiant_rag_mcp.config import LLMBackendConfig, EmbeddingBackendConfig, RerankingBackendConfig

logger = logging.getLogger(__name__)


def create_llm_backend(config: "LLMBackendConfig") -> BaseLLMBackend:
    """
    Create LLM backend based on configuration.

    Args:
        config: LLM backend configuration

    Returns:
        LLM backend instance

    Raises:
        ValueError: If backend type is unknown or configuration is invalid
    """
    backend_type = config.backend_type.lower()

    logger.info(f"Creating LLM backend: type={backend_type}")

    if backend_type in ["ollama", "vllm", "openai"]:
        # OpenAI-compatible API
        if not config.base_url or not config.model:
            raise ValueError(
                f"base_url and model are required for {backend_type} backend"
            )

        return OpenAICompatibleLLMBackend(
            base_url=config.base_url,
            api_key=config.api_key or "",
            model=config.model,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    elif backend_type == "local":
        # Local HuggingFace model
        if not config.model_name:
            raise ValueError("model_name is required for local backend")

        return LocalHuggingFaceLLMBackend(
            model_name=config.model_name,
            device=config.device,
            load_in_4bit=config.load_in_4bit,
            load_in_8bit=config.load_in_8bit,
        )

    else:
        raise ValueError(
            f"Unknown LLM backend type: {backend_type}. "
            f"Supported types: ollama, vllm, openai, local"
        )


def create_embedding_backend(config: "EmbeddingBackendConfig") -> BaseEmbeddingBackend:
    """
    Create embedding backend based on configuration.

    Args:
        config: Embedding backend configuration

    Returns:
        Embedding backend instance

    Raises:
        ValueError: If backend type is unknown or configuration is invalid
    """
    backend_type = config.backend_type.lower()

    logger.info(f"Creating embedding backend: type={backend_type}")

    if backend_type == "local":
        # Local sentence-transformers model (default)
        if not config.model_name:
            raise ValueError("model_name is required for local embedding backend")

        # Check if it's a sentence-transformers model or raw transformers
        # Heuristic: sentence-transformers models usually have "sentence-transformers/" prefix
        # or are in the sentence-transformers organization
        if "sentence-transformers" in config.model_name.lower() or "/" not in config.model_name:
            return SentenceTransformersEmbeddingBackend(
                model_name=config.model_name,
                device=config.device,
                cache_size=config.cache_size,
            )
        else:
            # Use direct HuggingFace Transformers
            return HuggingFaceTransformersEmbeddingBackend(
                model_name=config.model_name,
                device=config.device,
            )

    elif backend_type in ["ollama", "vllm", "openai"]:
        # OpenAI-compatible API
        if not config.base_url or not config.model:
            raise ValueError(
                f"base_url and model are required for {backend_type} embedding backend"
            )

        return OpenAICompatibleEmbeddingBackend(
            base_url=config.base_url,
            api_key=config.api_key or "",
            model=config.model,
            embedding_dimension=config.embedding_dimension,
        )

    else:
        raise ValueError(
            f"Unknown embedding backend type: {backend_type}. "
            f"Supported types: local, ollama, vllm, openai"
        )


def create_reranking_backend(
    config: "RerankingBackendConfig",
    llm_backend: BaseLLMBackend | None = None,
) -> BaseRerankingBackend:
    """
    Create reranking backend based on configuration.

    Args:
        config: Reranking backend configuration
        llm_backend: Optional LLM backend for LLM-based reranking

    Returns:
        Reranking backend instance

    Raises:
        ValueError: If backend type is unknown or configuration is invalid
    """
    backend_type = config.backend_type.lower()

    logger.info(f"Creating reranking backend: type={backend_type}")

    if backend_type == "local":
        # Local cross-encoder model (default)
        if not config.model_name:
            raise ValueError("model_name is required for local reranking backend")

        return CrossEncoderRerankingBackend(
            model_name=config.model_name,
            device=config.device,
        )

    elif backend_type in ["ollama", "vllm", "openai"]:
        # Use LLM for reranking
        if llm_backend is None:
            raise ValueError(
                f"llm_backend is required for {backend_type} reranking backend"
            )

        return OpenAICompatibleRerankingBackend(
            llm_backend=llm_backend,
        )

    else:
        raise ValueError(
            f"Unknown reranking backend type: {backend_type}. "
            f"Supported types: local, ollama, vllm, openai"
        )
