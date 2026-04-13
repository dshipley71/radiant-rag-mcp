"""
Reranking backend implementations.

Supports:
- Local cross-encoder models (sentence-transformers)
- OpenAI-compatible API (using LLM for scoring)
- LLM-based reranking
"""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from typing import List, Optional, Tuple, TYPE_CHECKING

from radiant_rag_mcp.llm.backends.base import BaseRerankingBackend

if TYPE_CHECKING:
    from radiant_rag_mcp.llm.backends.base import BaseLLMBackend

logger = logging.getLogger(__name__)

# Loggers that produce noisy output during model loading
_NOISY_LOADING_LOGGERS = (
    "sentence_transformers",
    "safetensors",
    "accelerate",
    "accelerate.utils.modeling",
    "transformers.modeling_utils",
)


@contextmanager
def _quiet_model_loading():
    """
    Suppress noisy warnings during model loading.

    Temporarily raises log levels for chatty libraries and filters warnings.
    """
    prev_levels = {}
    for name in _NOISY_LOADING_LOGGERS:
        lgr = logging.getLogger(name)
        prev_levels[name] = lgr.level
        lgr.setLevel(logging.ERROR)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not sharded.*")
        warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
        try:
            yield
        finally:
            for name, level in prev_levels.items():
                logging.getLogger(name).setLevel(level)


class CrossEncoderRerankingBackend(BaseRerankingBackend):
    """
    Reranking backend using sentence-transformers CrossEncoder.

    This is the default backend for local reranking.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        **kwargs,
    ) -> None:
        """
        Initialize cross-encoder reranking backend.

        Args:
            model_name: Model name (e.g., "cross-encoder/ms-marco-MiniLM-L12-v2")
            device: Device to load model on ("auto", "cuda", "cpu")
            **kwargs: Additional arguments
        """
        self._model_name = model_name

        logger.info(f"Initializing cross-encoder reranking backend: {model_name}")

        try:
            from sentence_transformers import CrossEncoder
            import torch
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking backend. "
                "Install with: pip install sentence-transformers"
            ) from e

        # Resolve device
        device_resolved = self._resolve_device(device)

        # Load model with suppressed warnings
        with _quiet_model_loading():
            self._model = CrossEncoder(model_name, device=device_resolved)

        logger.info(f"Loaded cross-encoder reranking model: {model_name} (device={device_resolved})")

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
        """
        if not documents:
            return []

        # Create query-document pairs
        pairs = [(query, doc) for doc in documents]

        # Predict scores
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Create indexed scores and sort
        indexed_scores = [(i, float(score)) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores

    @staticmethod
    def _resolve_device(pref: str) -> str:
        """Resolve device preference to actual device."""
        import torch
        pref = (pref or "auto").lower()
        if pref == "cpu":
            return "cpu"
        if pref == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"


class LLMRerankingBackend(BaseRerankingBackend):
    """
    Reranking backend using LLM for scoring.

    Uses an LLM to score document relevance. Slower but more flexible than cross-encoders.
    """

    def __init__(
        self,
        llm_backend: BaseLLMBackend,
        **kwargs,
    ) -> None:
        """
        Initialize LLM-based reranking backend.

        Args:
            llm_backend: LLM backend to use for scoring
            **kwargs: Additional arguments
        """
        self._llm = llm_backend

        logger.info("Initialized LLM-based reranking backend")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents by relevance to query using LLM scoring.

        Args:
            query: The query text
            documents: List of document texts to rerank
            top_k: Return only top K results (None = return all)

        Returns:
            List of (index, score) tuples sorted by relevance (highest first)
        """
        if not documents:
            return []

        indexed_scores = []

        for i, doc in enumerate(documents):
            # Truncate long documents
            doc_truncated = doc[:500] if len(doc) > 500 else doc

            # Create scoring prompt
            prompt = f"""Rate the relevance of the following document to the query on a scale of 0-100.
Consider how well the document answers or relates to the query.

Query: {query}

Document: {doc_truncated}

Respond with ONLY a number between 0 and 100 (0=completely irrelevant, 100=perfectly relevant).
Do not provide any explanation, just the number."""

            try:
                response = self._llm.generate(prompt)
                # Extract numeric score
                score_str = response.strip()
                # Try to parse the first number found
                import re
                numbers = re.findall(r'\d+', score_str)
                if numbers:
                    score = float(numbers[0])
                    # Normalize to 0-1 range
                    score = score / 100.0
                else:
                    logger.warning(f"Failed to parse score from LLM response: {score_str}")
                    score = 0.0

                indexed_scores.append((i, score))

            except Exception as e:
                logger.warning(f"LLM reranking failed for document {i}: {e}")
                indexed_scores.append((i, 0.0))

        # Sort by score descending
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores


class OpenAICompatibleRerankingBackend(BaseRerankingBackend):
    """
    Reranking backend using OpenAI-compatible reranking API.

    Note: Most OpenAI-compatible APIs don't have dedicated reranking endpoints,
    so this falls back to using the LLM for scoring.
    """

    def __init__(
        self,
        llm_backend: BaseLLMBackend,
        **kwargs,
    ) -> None:
        """
        Initialize OpenAI-compatible reranking backend.

        Args:
            llm_backend: LLM backend to use for scoring
            **kwargs: Additional arguments
        """
        # For now, delegate to LLM-based reranking since most APIs don't have
        # dedicated reranking endpoints
        self._delegate = LLMRerankingBackend(llm_backend, **kwargs)

        logger.info("Initialized OpenAI-compatible reranking backend (using LLM scoring)")

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
        """
        return self._delegate.rerank(query, documents, top_k)
