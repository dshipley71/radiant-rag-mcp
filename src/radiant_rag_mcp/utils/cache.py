"""
Caching utilities for performance optimization.

Provides LRU caches for expensive operations like embeddings and LLM responses.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    True LRU cache for text embeddings to avoid redundant computation.

    Uses OrderedDict with move_to_end() for proper LRU eviction.
    Thread-safe through Python's GIL for dictionary operations.
    """

    def __init__(self, max_size: int = 10000) -> None:
        """
        Initialize embedding cache with true LRU eviction.

        Args:
            max_size: Maximum number of embeddings to cache (default 10000)
                     At ~1.5KB per embedding (384-dim float32), this uses ~15MB RAM
        """
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        logger.info(f"Initialized EmbeddingCache with max_size={max_size} (true LRU)")

    def _hash_text(self, text: str) -> str:
        """Generate cache key from text content."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """
        Retrieve cached embedding for text with LRU tracking.

        Args:
            text: Input text

        Returns:
            Cached embedding or None if not found
        """
        key = self._hash_text(text)
        embedding = self._cache.get(key)

        if embedding is not None:
            self._hits += 1
            # TRUE LRU: Move to end (most recently used)
            self._cache.move_to_end(key)
        else:
            self._misses += 1

        return embedding

    def put(self, text: str, embedding: List[float]) -> None:
        """
        Store embedding in cache with LRU eviction.

        Args:
            text: Input text
            embedding: Computed embedding vector
        """
        key = self._hash_text(text)

        # If key exists, remove it (will be re-added at end)
        if key in self._cache:
            del self._cache[key]

        # TRUE LRU eviction: Remove least recently used (first item)
        if len(self._cache) >= self._max_size:
            # Remove first item (least recently used in OrderedDict)
            lru_key = next(iter(self._cache))
            del self._cache[lru_key]
            logger.debug(f"Evicted LRU embedding from cache (size={len(self._cache)})")

        # Add to end (most recently used)
        self._cache[key] = embedding

    def get_batch(self, texts: List[str]) -> Tuple[List[Optional[List[float]]], List[int]]:
        """
        Retrieve multiple embeddings, returning cache hits and miss indices.

        Args:
            texts: List of input texts

        Returns:
            Tuple of (embeddings_or_none, miss_indices)
            - embeddings_or_none: List with cached embeddings or None for misses
            - miss_indices: Indices of texts that need to be computed
        """
        embeddings = []
        miss_indices = []

        for i, text in enumerate(texts):
            embedding = self.get(text)
            embeddings.append(embedding)
            if embedding is None:
                miss_indices.append(i)

        return embeddings, miss_indices

    def put_batch(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """
        Store multiple embeddings in cache.

        Args:
            texts: List of input texts
            embeddings: List of computed embedding vectors
        """
        for text, embedding in zip(texts, embeddings):
            self.put(text, embedding)

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("Cleared embedding cache")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
        }


class QueryCache:
    """
    True LRU cache for query results to avoid redundant LLM calls.

    Caches query processing results like decomposition, rewriting, expansion.
    Uses OrderedDict with move_to_end() for proper LRU eviction.
    """

    def __init__(self, max_size: int = 1000) -> None:
        """
        Initialize query cache with true LRU eviction.

        Args:
            max_size: Maximum number of query results to cache
        """
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        logger.info(f"Initialized QueryCache with max_size={max_size} (true LRU)")

    def _make_key(self, operation: str, query: str, **kwargs: Any) -> str:
        """Generate cache key from operation and query."""
        # Include operation type and query content
        key_parts = [operation, query]

        # Add any additional parameters that affect the result
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")

        key_str = "|".join(key_parts)
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()

    def get(self, operation: str, query: str, **kwargs: Any) -> Optional[Any]:
        """
        Retrieve cached result for query operation with LRU tracking.

        Args:
            operation: Operation name (e.g., "rewrite", "expand", "decompose")
            query: Input query text
            **kwargs: Additional parameters affecting the result

        Returns:
            Cached result or None if not found
        """
        key = self._make_key(operation, query, **kwargs)
        result = self._cache.get(key)

        if result is not None:
            self._hits += 1
            # TRUE LRU: Move to end (most recently used)
            self._cache.move_to_end(key)
            logger.debug(f"Cache HIT for {operation}: {query[:50]}")
        else:
            self._misses += 1

        return result

    def put(self, operation: str, query: str, result: Any, **kwargs: Any) -> None:
        """
        Store operation result in cache with LRU eviction.

        Args:
            operation: Operation name
            query: Input query text
            result: Operation result
            **kwargs: Additional parameters affecting the result
        """
        key = self._make_key(operation, query, **kwargs)

        # If key exists, remove it (will be re-added at end)
        if key in self._cache:
            del self._cache[key]

        # TRUE LRU eviction: Remove least recently used (first item)
        if len(self._cache) >= self._max_size:
            lru_key = next(iter(self._cache))
            del self._cache[lru_key]
            logger.debug(f"Evicted LRU query result from cache (size={len(self._cache)})")

        # Add to end (most recently used)
        self._cache[key] = result
        logger.debug(f"Cached {operation} result: {query[:50]}")

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("Cleared query cache")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
        }


# Global cache instances (can be configured via environment)
_embedding_cache: Optional[EmbeddingCache] = None
_query_cache: Optional[QueryCache] = None


def get_embedding_cache(max_size: int = 10000) -> EmbeddingCache:
    """
    Get or create global embedding cache.

    Args:
        max_size: Maximum cache size (only used on first call)

    Returns:
        Global EmbeddingCache instance
    """
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache(max_size=max_size)
    return _embedding_cache


def get_query_cache(max_size: int = 1000) -> QueryCache:
    """
    Get or create global query cache.

    Args:
        max_size: Maximum cache size (only used on first call)

    Returns:
        Global QueryCache instance
    """
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCache(max_size=max_size)
    return _query_cache


def clear_all_caches() -> None:
    """Clear all global caches."""
    global _embedding_cache, _query_cache

    if _embedding_cache is not None:
        _embedding_cache.clear()

    if _query_cache is not None:
        _query_cache.clear()

    logger.info("Cleared all global caches")


def get_all_cache_stats() -> Dict[str, Dict[str, Any]]:
    """
    Get statistics from all caches.

    Returns:
        Dictionary with stats for each cache type
    """
    stats = {}

    if _embedding_cache is not None:
        stats["embedding"] = _embedding_cache.get_stats()

    if _query_cache is not None:
        stats["query"] = _query_cache.get_stats()

    return stats
