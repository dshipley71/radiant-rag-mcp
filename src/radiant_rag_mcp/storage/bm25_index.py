"""
BM25 sparse retrieval index with persistence and incremental updates.

Provides keyword-based retrieval to complement dense vector search.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from radiant_rag_mcp.config import BM25Config
from radiant_rag_mcp.storage.base import BaseVectorStore, StoredDoc

logger = logging.getLogger(__name__)

# Known index file extensions (for path normalization)
_INDEX_EXTENSIONS = {".json.gz", ".json", ".pkl", ".pickle"}


def _normalize_index_path(path: Path) -> Path:
    """
    Normalize index path by stripping known extensions.
    
    This allows users to specify paths with or without extensions
    and ensures consistent file naming.
    
    Args:
        path: Index path (may include extension)
        
    Returns:
        Path with known extensions stripped
    """
    # Handle compound extensions like .json.gz
    name = path.name
    for ext in sorted(_INDEX_EXTENSIONS, key=len, reverse=True):
        if name.endswith(ext):
            return path.parent / name[:-len(ext)]
    return path


def _tokenize(text: str) -> List[str]:
    """
    Tokenize text for BM25 indexing.

    Simple tokenization: lowercase, alphanumeric only, split on whitespace.
    """
    normalized = "".join(ch if ch.isalnum() else " " for ch in text.lower())
    tokens = [t for t in normalized.split() if t and len(t) > 1]
    return tokens


@dataclass
class BM25Index:
    """
    BM25 index data structure.

    Stores:
        - Document IDs and their token lists
        - Inverted index (term -> doc positions)
        - IDF values
        - Document lengths
    """

    # Document storage
    doc_ids: List[str] = field(default_factory=list)
    doc_tokens: List[List[str]] = field(default_factory=list)

    # Index metadata
    doc_id_set: Set[str] = field(default_factory=set)
    doc_id_to_idx: Dict[str, int] = field(default_factory=dict)

    # BM25 parameters
    k1: float = 1.5
    b: float = 0.75

    # Computed values (rebuilt on load/modification)
    avgdl: float = 0.0
    doc_lengths: List[int] = field(default_factory=list)
    idf: Dict[str, float] = field(default_factory=dict)
    term_doc_freqs: Dict[str, int] = field(default_factory=dict)

    # Index state
    dirty: bool = False
    needs_rebuild: bool = True

    def __post_init__(self) -> None:
        """Initialize computed values if we have documents."""
        if self.doc_ids and self.needs_rebuild:
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild computed index values (IDF, avgdl, etc.)."""
        if not self.doc_ids:
            self.avgdl = 0.0
            self.doc_lengths = []
            self.idf = {}
            self.term_doc_freqs = {}
            self.needs_rebuild = False
            return

        logger.debug(f"Rebuilding BM25 index for {len(self.doc_ids)} documents...")

        # Rebuild doc_id mappings
        self.doc_id_set = set(self.doc_ids)
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}

        # Calculate document lengths
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0

        # Calculate term document frequencies
        self.term_doc_freqs = {}
        for tokens in self.doc_tokens:
            seen_terms: Set[str] = set()
            for token in tokens:
                if token not in seen_terms:
                    self.term_doc_freqs[token] = self.term_doc_freqs.get(token, 0) + 1
                    seen_terms.add(token)

        # Calculate IDF values
        n = len(self.doc_ids)
        self.idf = {}
        for term, df in self.term_doc_freqs.items():
            # Standard BM25 IDF formula
            self.idf[term] = np.log((n - df + 0.5) / (df + 0.5) + 1.0)

        self.needs_rebuild = False
        logger.debug("BM25 index rebuild complete")

    def add_document(self, doc_id: str, tokens: List[str]) -> bool:
        """
        Add a document to the index.

        Args:
            doc_id: Unique document identifier
            tokens: Pre-tokenized document

        Returns:
            True if document was added (not duplicate)
        """
        if doc_id in self.doc_id_set:
            logger.debug(f"Document {doc_id} already in index, skipping")
            return False

        self.doc_ids.append(doc_id)
        self.doc_tokens.append(tokens)
        self.doc_id_set.add(doc_id)
        self.doc_id_to_idx[doc_id] = len(self.doc_ids) - 1

        # Update document length
        self.doc_lengths.append(len(tokens))

        # Update avgdl incrementally
        n = len(self.doc_ids)
        total_length = self.avgdl * (n - 1) + len(tokens)
        self.avgdl = total_length / n

        # Update term frequencies and IDF incrementally
        seen_terms: Set[str] = set()
        for token in tokens:
            if token not in seen_terms:
                old_df = self.term_doc_freqs.get(token, 0)
                new_df = old_df + 1
                self.term_doc_freqs[token] = new_df

                # Update IDF for this term
                self.idf[token] = np.log((n - new_df + 0.5) / (new_df + 0.5) + 1.0)
                seen_terms.add(token)

        self.dirty = True
        return True

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the index.

        Note: This is expensive as it requires rebuilding the index.

        Args:
            doc_id: Document identifier to remove

        Returns:
            True if document was removed
        """
        if doc_id not in self.doc_id_set:
            return False

        idx = self.doc_id_to_idx[doc_id]

        # Remove from lists
        del self.doc_ids[idx]
        del self.doc_tokens[idx]
        
        # Remove from set and dict
        self.doc_id_set.discard(doc_id)
        del self.doc_id_to_idx[doc_id]
        
        # Update indices for all documents that came after the removed one
        for other_doc_id, other_idx in self.doc_id_to_idx.items():
            if other_idx > idx:
                self.doc_id_to_idx[other_doc_id] = other_idx - 1

        # Mark for full rebuild (removal is complex to do incrementally)
        self.needs_rebuild = True
        self.dirty = True

        return True

    def search(self, query_tokens: List[str], top_k: int) -> List[Tuple[str, float]]:
        """
        Search the index with BM25 scoring.

        Args:
            query_tokens: Tokenized query
            top_k: Maximum number of results

        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        if self.needs_rebuild:
            self._rebuild_index()

        if not self.doc_ids or not query_tokens:
            return []

        scores = np.zeros(len(self.doc_ids), dtype=np.float64)

        for term in query_tokens:
            if term not in self.idf:
                continue

            term_idf = self.idf[term]

            # Score each document containing this term
            for idx, tokens in enumerate(self.doc_tokens):
                # Count term frequency in document
                tf = tokens.count(term)
                if tf == 0:
                    continue

                doc_len = self.doc_lengths[idx]

                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[idx] += term_idf * (numerator / denominator)

        # Get top-k indices
        if top_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            score = scores[idx]
            if score > 0:
                results.append((self.doc_ids[idx], float(score)))

        return results[:top_k]

    def __len__(self) -> int:
        return len(self.doc_ids)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize index to a JSON-compatible dictionary.
        
        Returns:
            Dictionary representation of the index
        """
        return {
            "version": 2,  # Serialization format version
            "doc_ids": self.doc_ids,
            "doc_tokens": self.doc_tokens,
            "k1": self.k1,
            "b": self.b,
            # Note: doc_id_set, doc_id_to_idx, avgdl, doc_lengths, idf, term_doc_freqs
            # are computed values that will be rebuilt on load
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BM25Index":
        """
        Deserialize index from a dictionary.
        
        Args:
            data: Dictionary representation of the index
            
        Returns:
            BM25Index instance
        """
        version = data.get("version", 1)
        
        if version >= 2:
            index = cls(
                doc_ids=data.get("doc_ids", []),
                doc_tokens=data.get("doc_tokens", []),
                k1=data.get("k1", 1.5),
                b=data.get("b", 0.75),
                needs_rebuild=True,  # Force rebuild of computed values
            )
        else:
            # Handle legacy format (version 1 or unversioned)
            index = cls(
                doc_ids=data.get("doc_ids", []),
                doc_tokens=data.get("doc_tokens", []),
                k1=data.get("k1", 1.5),
                b=data.get("b", 0.75),
                needs_rebuild=True,
            )
        
        # Rebuild computed values
        if index.doc_ids:
            index._rebuild_index()
        
        return index


class PersistentBM25Index:
    """
    Thread-safe, persistent BM25 index with incremental updates.

    Features:
        - Automatic persistence to disk
        - Incremental document addition
        - Thread-safe operations
        - Auto-save on threshold
    """

    def __init__(
        self,
        config: BM25Config,
        store: BaseVectorStore,
    ) -> None:
        """
        Initialize persistent BM25 index.

        Args:
            config: BM25 configuration
            store: Vector store for document retrieval (Redis, Chroma, or PgVector)
        """
        self._config = config
        self._store = store
        self._lock = threading.RLock()
        self._index: Optional[BM25Index] = None
        self._unsaved_count = 0

        # Ensure index directory exists
        index_path = Path(config.index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_or_create_index(self) -> BM25Index:
        """
        Load index from disk or create new one.
        
        Supports both new JSON format (.json.gz) and legacy pickle format (.pkl).
        """
        base_path = _normalize_index_path(Path(self._config.index_path))
        
        # Try JSON format first (new format)
        json_path = base_path.with_suffix(".json.gz")
        if json_path.exists():
            try:
                with gzip.open(json_path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
                
                index = BM25Index.from_dict(data)
                index.k1 = self._config.k1
                index.b = self._config.b
                logger.info(f"Loaded BM25 index (JSON) with {len(index)} documents")
                return index
            except Exception as e:
                logger.warning(f"Failed to load JSON BM25 index: {e}")
        
        # Try legacy pickle format for backward compatibility
        pkl_path = base_path.with_suffix(".pkl")
        if pkl_path.exists():
            try:
                import pickle
                with open(pkl_path, "rb") as f:
                    index = pickle.load(f)
                
                if isinstance(index, BM25Index):
                    index.k1 = self._config.k1
                    index.b = self._config.b
                    logger.info(f"Loaded BM25 index (legacy pickle) with {len(index)} documents")
                    
                    # Mark dirty to trigger migration to JSON on next save
                    index.dirty = True
                    logger.info("Legacy pickle format detected - will migrate to JSON on next save")
                    return index
                else:
                    logger.warning("Invalid pickle index file format")
            except Exception as e:
                logger.warning(f"Failed to load legacy pickle BM25 index: {e}")
        
        logger.info("Creating new BM25 index")
        return BM25Index(k1=self._config.k1, b=self._config.b)

    @property
    def index(self) -> BM25Index:
        """Get or load the index (lazy loading)."""
        if self._index is None:
            with self._lock:
                if self._index is None:
                    self._index = self._load_or_create_index()
        return self._index

    def save(self) -> bool:
        """
        Save index to disk using gzip-compressed JSON.
        
        Uses atomic write (temp file + rename) to prevent corruption.
        Also cleans up legacy pickle files after successful migration.

        Returns:
            True if save was successful
        """
        with self._lock:
            if self._index is None or not self._index.dirty:
                return True

            try:
                base_path = _normalize_index_path(Path(self._config.index_path))
                # Use .json.gz extension for new format
                json_path = base_path.with_suffix(".json.gz")
                temp_path = base_path.with_suffix(".tmp.gz")

                # Serialize to JSON with gzip compression
                data = self._index.to_dict()
                with gzip.open(temp_path, "wt", encoding="utf-8") as f:
                    json.dump(data, f, separators=(",", ":"))  # Compact JSON

                # Atomic rename
                os.replace(temp_path, json_path)

                self._index.dirty = False
                self._unsaved_count = 0
                logger.info(f"Saved BM25 index with {len(self._index)} documents")
                
                # Clean up legacy pickle file if it exists
                pkl_path = base_path.with_suffix(".pkl")
                if pkl_path.exists():
                    try:
                        pkl_path.unlink()
                        logger.info("Removed legacy pickle index file after migration")
                    except Exception as e:
                        logger.warning(f"Failed to remove legacy pickle file: {e}")
                
                return True
            except Exception as e:
                logger.error(f"Failed to save BM25 index: {e}")
                # Clean up temp file if it exists
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except Exception:
                    pass
                return False

    def _maybe_auto_save(self) -> None:
        """Auto-save if threshold reached."""
        if self._unsaved_count >= self._config.auto_save_threshold:
            self.save()

    def add_document(self, doc_id: str, content: str) -> bool:
        """
        Add a document to the index.

        Args:
            doc_id: Document identifier
            content: Document text

        Returns:
            True if document was added
        """
        tokens = _tokenize(content)
        if not tokens:
            return False

        with self._lock:
            added = self.index.add_document(doc_id, tokens)
            if added:
                self._unsaved_count += 1
                self._maybe_auto_save()
            return added

    def add_documents_batch(
        self,
        documents: List[Tuple[str, str]],
    ) -> int:
        """
        Add multiple documents to the index.

        Args:
            documents: List of (doc_id, content) tuples

        Returns:
            Number of documents added
        """
        added_count = 0

        with self._lock:
            for doc_id, content in documents:
                tokens = _tokenize(content)
                if tokens and self.index.add_document(doc_id, tokens):
                    added_count += 1

            if added_count > 0:
                self._unsaved_count += added_count
                self._maybe_auto_save()

        return added_count

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the index.

        Args:
            doc_id: Document identifier

        Returns:
            True if document was removed
        """
        with self._lock:
            removed = self.index.remove_document(doc_id)
            if removed:
                self._unsaved_count += 1
                self._maybe_auto_save()
            return removed

    def search(
        self,
        query: str,
        top_k: int,
    ) -> List[Tuple[StoredDoc, float]]:
        """
        Search the index and retrieve full documents.

        Args:
            query: Search query text
            top_k: Maximum number of results

        Returns:
            List of (StoredDoc, score) tuples
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        with self._lock:
            results = self.index.search(query_tokens, top_k)

        # Retrieve full documents from Redis
        docs_with_scores: List[Tuple[StoredDoc, float]] = []
        for doc_id, score in results:
            doc = self._store.get_doc(doc_id)
            if doc is not None:
                docs_with_scores.append((doc, score))

        return docs_with_scores

    def build_from_store(self, limit: int = 0) -> int:
        """
        Build index from all documents in Redis store.

        Args:
            limit: Maximum documents to index (0 = no limit, use config max)

        Returns:
            Number of documents indexed
        """
        max_docs = limit or self._config.max_documents

        logger.info(f"Building BM25 index from store (max {max_docs} documents)...")

        doc_ids = self._store.list_doc_ids_with_embeddings(limit=max_docs)

        documents: List[Tuple[str, str]] = []
        for doc_id in doc_ids:
            doc = self._store.get_doc(doc_id)
            if doc and doc.content:
                documents.append((doc_id, doc.content))

        with self._lock:
            # Create fresh index
            self._index = BM25Index(k1=self._config.k1, b=self._config.b)

            for doc_id, content in documents:
                tokens = _tokenize(content)
                if tokens:
                    self._index.add_document(doc_id, tokens)

            self._index.dirty = True
            self.save()

        logger.info(f"Built BM25 index with {len(self._index)} documents")
        return len(self._index)

    def sync_with_store(self) -> Tuple[int, int]:
        """
        Sync index with Redis store (add new, remove deleted).

        Returns:
            Tuple of (documents_added, documents_removed)
        """
        logger.info("Syncing BM25 index with store...")

        store_ids = set(self._store.list_doc_ids_with_embeddings(
            limit=self._config.max_documents
        ))

        with self._lock:
            index_ids = set(self.index.doc_ids)

            # Find documents to add
            to_add = store_ids - index_ids
            # Find documents to remove
            to_remove = index_ids - store_ids

            # Remove deleted documents
            removed = 0
            for doc_id in to_remove:
                if self.index.remove_document(doc_id):
                    removed += 1

            # Add new documents
            added = 0
            for doc_id in to_add:
                doc = self._store.get_doc(doc_id)
                if doc and doc.content:
                    tokens = _tokenize(doc.content)
                    if tokens and self.index.add_document(doc_id, tokens):
                        added += 1

            if added > 0 or removed > 0:
                self._index.dirty = True
                self.save()

        logger.info(f"Sync complete: {added} added, {removed} removed")
        return added, removed

    def clear(self) -> None:
        """Clear the index and delete all index files."""
        with self._lock:
            self._index = BM25Index(k1=self._config.k1, b=self._config.b)
            self._unsaved_count = 0

            # Delete index files (both new JSON and legacy pickle formats)
            base_path = _normalize_index_path(Path(self._config.index_path))
            
            files_to_delete = [
                base_path.with_suffix(".json.gz"),  # New JSON format
                base_path.with_suffix(".pkl"),      # Legacy pickle format
            ]
            
            for file_path in files_to_delete:
                if file_path.exists():
                    try:
                        file_path.unlink()
                        logger.debug(f"Deleted index file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete index file {file_path}: {e}")

    def __len__(self) -> int:
        """Return number of documents in index."""
        return len(self.index)

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        with self._lock:
            idx = self.index
            base_path = _normalize_index_path(Path(self._config.index_path))
            json_path = base_path.with_suffix(".json.gz")
            pkl_path = base_path.with_suffix(".pkl")
            
            # Determine actual file being used
            if json_path.exists():
                actual_path = str(json_path)
                storage_format = "json.gz"
            elif pkl_path.exists():
                actual_path = str(pkl_path)
                storage_format = "pickle (legacy)"
            else:
                actual_path = str(json_path)  # Will be created on save
                storage_format = "json.gz (pending)"
            
            return {
                "document_count": len(idx),
                "unique_terms": len(idx.idf),
                "avg_doc_length": idx.avgdl,
                "k1": idx.k1,
                "b": idx.b,
                "dirty": idx.dirty,
                "needs_rebuild": idx.needs_rebuild,
                "index_path": actual_path,
                "storage_format": storage_format,
            }
