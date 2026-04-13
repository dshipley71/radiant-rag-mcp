"""
Abstract base classes for LLM, embedding, and reranking backends.

Defines the interfaces that all backend implementations must follow.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM backend."""

    content: str
    """The generated text content."""

    meta: Dict[str, Any]
    """Metadata about the response (model, tokens, etc.)."""

    def __str__(self) -> str:
        return self.content


class BaseLLMBackend(ABC):
    """
    Abstract base class for LLM chat/generation backends.

    All LLM backends (OpenAI-compatible, HuggingFace) must implement this interface.
    """

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Send chat request to LLM.

        Args:
            messages: List of chat messages with 'role' and 'content' keys
            **kwargs: Additional backend-specific parameters

        Returns:
            LLMResponse with generated content and metadata

        Raises:
            Exception: If the request fails
        """
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The input prompt
            **kwargs: Additional backend-specific parameters

        Returns:
            Generated text string

        Raises:
            Exception: If the request fails
        """
        pass


class BaseEmbeddingBackend(ABC):
    """
    Abstract base class for embedding backends.

    All embedding backends (OpenAI-compatible, HuggingFace) must implement this interface.
    """

    @abstractmethod
    def embed(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = False,
        batch_size: int = 32,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of text strings to embed
            normalize: Whether to normalize embeddings to unit length
            show_progress: Whether to show progress bar
            batch_size: Batch size for processing

        Returns:
            List of embedding vectors (one per input text)

        Raises:
            Exception: If embedding generation fails
        """
        pass

    @abstractmethod
    def embed_single(
        self,
        text: str,
        normalize: bool = True,
    ) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed
            normalize: Whether to normalize embedding to unit length

        Returns:
            Embedding vector

        Raises:
            Exception: If embedding generation fails
        """
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """
        Get the embedding dimension for this backend.

        Returns:
            Dimension of embedding vectors
        """
        pass


class BaseRerankingBackend(ABC):
    """
    Abstract base class for reranking backends.

    All reranking backends (cross-encoder, LLM-based, API) must implement this interface.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents by relevance to query.

        Args:
            query: The query text
            documents: List of document texts to rerank
            top_k: Return only top K results (None = return all)

        Returns:
            List of (index, score) tuples sorted by relevance (highest first)
            Index refers to position in input documents list

        Raises:
            Exception: If reranking fails
        """
        pass
