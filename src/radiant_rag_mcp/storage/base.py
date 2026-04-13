"""
Abstract base class for vector storage backends.

Defines the interface that all storage backends must implement.
This allows the application to switch between Redis, Chroma, and PgVector
without changing the business logic.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _sha256_hex(text: str) -> str:
    """Generate SHA-256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class StoredDoc:
    """Represents a stored document."""
    
    doc_id: str
    content: str
    meta: Dict[str, Any]

    def __hash__(self) -> int:
        return hash(self.doc_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StoredDoc):
            return False
        return self.doc_id == other.doc_id


class BaseVectorStore(ABC):
    """
    Abstract base class for vector storage backends.
    
    All storage backends (Redis, Chroma, PgVector) must implement this interface.
    """

    @abstractmethod
    def ping(self) -> bool:
        """
        Test connection to the storage backend.
        
        Returns:
            True if connection is successful
        """
        pass

    @abstractmethod
    def make_doc_id(self, content: str, meta: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate deterministic document ID from content and metadata.
        
        Args:
            content: Document text
            meta: Optional metadata (included in hash)
            
        Returns:
            Unique document identifier
        """
        pass

    @abstractmethod
    def upsert(
        self,
        doc_id: str,
        content: str,
        embedding: List[float],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Insert or update a document with embedding.
        
        Args:
            doc_id: Unique document identifier
            content: Document text
            embedding: Vector embedding
            meta: Optional metadata
        """
        pass

    @abstractmethod
    def upsert_doc_only(
        self,
        doc_id: str,
        content: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store document without embedding (for parent documents).
        
        These documents are NOT indexed for vector search but can be
        retrieved by ID for auto-merging.
        
        Args:
            doc_id: Unique document identifier
            content: Document text
            meta: Optional metadata
        """
        pass

    @abstractmethod
    def upsert_batch(
        self,
        documents: List[Dict[str, Any]],
    ) -> int:
        """
        Batch insert/update documents with embeddings.
        
        Args:
            documents: List of dicts with keys: doc_id, content, embedding, meta
            
        Returns:
            Number of documents stored
        """
        pass

    @abstractmethod
    def upsert_doc_only_batch(
        self,
        documents: List[Dict[str, Any]],
    ) -> int:
        """
        Batch store documents without embeddings.
        
        Args:
            documents: List of dicts with keys: doc_id, content, meta
            
        Returns:
            Number of documents stored
        """
        pass

    @abstractmethod
    def get_doc(self, doc_id: str) -> Optional[StoredDoc]:
        """
        Retrieve a document by ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            StoredDoc if found, None otherwise
        """
        pass

    @abstractmethod
    def has_embedding(self, doc_id: str) -> bool:
        """
        Check if document has an embedding stored.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if document has embedding
        """
        pass

    @abstractmethod
    def delete_doc(self, doc_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if document was deleted
        """
        pass

    @abstractmethod
    def retrieve_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int,
        min_similarity: float = 0.0,
        ef_runtime: Optional[int] = None,
        language_filter: Optional[str] = None,
        doc_level_filter: Optional[str] = None,
    ) -> List[Tuple[StoredDoc, float]]:
        """
        Retrieve documents by vector similarity.
        
        Args:
            query_embedding: Query vector
            top_k: Maximum number of results
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            ef_runtime: Optional HNSW ef parameter for this query
            language_filter: Optional language code to filter by
            doc_level_filter: Optional filter by document level:
                - None or "all": Return both parents and children
                - "child" or "leaves": Return only leaf/child documents
                - "parent" or "parents": Return only parent documents
            
        Returns:
            List of (document, similarity_score) tuples, sorted by score descending
        """
        pass

    def retrieve_by_embedding_quantized(
        self,
        query_embedding: List[float],
        top_k: int,
        min_similarity: float = 0.0,
        rescore_multiplier: Optional[float] = None,
        use_rescoring: Optional[bool] = None,
        language_filter: Optional[str] = None,
        doc_level_filter: Optional[str] = None,
    ) -> List[Tuple[StoredDoc, float]]:
        """
        Retrieve documents using quantized embeddings (faster, less memory).
        
        Uses two-stage retrieval:
        1. Fast binary search to get top candidates
        2. Rescore with higher precision embeddings
        
        Falls back to standard retrieval if quantization is not enabled.
        
        Args:
            query_embedding: Query vector (float32)
            top_k: Maximum number of results
            min_similarity: Minimum similarity threshold
            rescore_multiplier: Retrieve N×top_k candidates for rescoring
            use_rescoring: Whether to use rescoring step
            language_filter: Optional language code to filter by
            doc_level_filter: Optional document level filter
            
        Returns:
            List of (document, similarity_score) tuples, sorted by score descending
        """
        # Default implementation: fall back to standard retrieval
        # Backends with quantization support should override this method
        return self.retrieve_by_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
            min_similarity=min_similarity,
            language_filter=language_filter,
            doc_level_filter=doc_level_filter,
        )

    @abstractmethod
    def list_doc_ids(self, pattern: str = "*", limit: int = 10_000) -> List[str]:
        """
        List document IDs matching a pattern.
        
        Args:
            pattern: Pattern to match (implementation-specific)
            limit: Maximum number of IDs to return
            
        Returns:
            List of document IDs
        """
        pass

    @abstractmethod
    def list_doc_ids_with_embeddings(self, limit: int = 10_000) -> List[str]:
        """
        List document IDs that have embeddings stored.
        
        Args:
            limit: Maximum number of IDs to return
            
        Returns:
            List of document IDs
        """
        pass

    @abstractmethod
    def get_index_info(self) -> Dict[str, Any]:
        """
        Get storage/index statistics.
        
        Returns:
            Dictionary with backend-specific statistics
        """
        pass

    @abstractmethod
    def drop_index(self, delete_documents: bool = False) -> bool:
        """
        Drop the vector index.
        
        Args:
            delete_documents: If True, also delete all indexed documents
            
        Returns:
            True if index was dropped
        """
        pass

    @abstractmethod
    def count_documents(self) -> int:
        """
        Count total documents stored.
        
        Returns:
            Number of documents
        """
        pass

    def _default_make_doc_id(
        self, content: str, meta: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Default implementation for generating document IDs.
        
        Args:
            content: Document text
            meta: Optional metadata
            
        Returns:
            SHA-256 hash as document ID
        """
        meta_part = json.dumps(meta or {}, sort_keys=True, ensure_ascii=False)
        return _sha256_hex(content + "\n" + meta_part)
