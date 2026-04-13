"""
Chroma-based vector storage for Radiant Agentic RAG.

Uses ChromaDB for persistent vector similarity search with support for
metadata filtering and efficient approximate nearest neighbor search.

Requirements:
    - chromadb >= 0.4.0
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from radiant_rag_mcp.config import ChromaConfig
from radiant_rag_mcp.storage.base import BaseVectorStore, StoredDoc
from radiant_rag_mcp.storage.quantization import (
    quantize_embeddings,
    rescore_candidates,
    QUANTIZATION_AVAILABLE,
)

# Try to import ChromaDB
CHROMA_AVAILABLE = False
_CHROMA_IMPORT_ERROR = None
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError as e:
    _CHROMA_IMPORT_ERROR = str(e)
    chromadb = None
    Settings = None

logger = logging.getLogger(__name__)

# Log import status
if _CHROMA_IMPORT_ERROR:
    logger.debug(f"ChromaDB not available: {_CHROMA_IMPORT_ERROR}")


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB-based document and vector storage with ANN search.
    
    Storage schema:
        Collection: {collection_name}
            - id: document ID
            - documents: document text content
            - embeddings: vector embeddings
            - metadatas: JSON-compatible metadata
    
    Supports:
        - Persistent storage to disk
        - Cosine, L2, and inner product distance metrics
        - Metadata filtering
    """

    def __init__(self, config: ChromaConfig) -> None:
        """
        Initialize Chroma vector store.
        
        Args:
            config: Chroma configuration
            
        Raises:
            ImportError: If chromadb is not installed
        """
        if not CHROMA_AVAILABLE:
            raise ImportError(
                f"ChromaDB is not available. Install with: pip install chromadb\n"
                f"Original error: {_CHROMA_IMPORT_ERROR}"
            )
        
        self._config = config
        self._max_chars = config.max_content_chars
        self._embedding_dim = config.embedding_dimension
        
        # Ensure persistence directory exists
        persist_dir = Path(config.persist_directory)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma client with persistence
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        
        # Map distance function names
        distance_fn_map = {
            "cosine": "cosine",
            "l2": "l2",
            "ip": "ip",
            "inner_product": "ip",
        }
        distance_fn = distance_fn_map.get(
            config.distance_fn.lower(), "cosine"
        )
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": distance_fn},
        )
        
        # Quantization configuration
        self._quant_config = config.quantization
        
        # Load int8 calibration ranges if provided
        self._int8_ranges: Optional[np.ndarray] = None
        if self._quant_config.enabled and self._quant_config.int8_ranges_file:
            try:
                self._int8_ranges = np.load(self._quant_config.int8_ranges_file)
                logger.info(f"Loaded int8 calibration ranges from {self._quant_config.int8_ranges_file}")
            except Exception as e:
                logger.warning(f"Failed to load int8 ranges: {e}")
        
        # Create separate collections for quantized embeddings if enabled
        if self._quant_config.enabled and QUANTIZATION_AVAILABLE:
            if self._quant_config.precision in ("binary", "both"):
                try:
                    self._binary_collection = self._client.get_or_create_collection(
                        name=f"{config.collection_name}_binary",
                        metadata={"hnsw:space": "l2"},  # Use L2 for binary (hamming not directly supported)
                    )
                except Exception as e:
                    logger.warning(f"Failed to create binary collection: {e}")
                    self._binary_collection = None
            else:
                self._binary_collection = None
            
            if self._quant_config.precision in ("int8", "both"):
                try:
                    self._int8_collection = self._client.get_or_create_collection(
                        name=f"{config.collection_name}_int8",
                        metadata={"hnsw:space": distance_fn},
                    )
                except Exception as e:
                    logger.warning(f"Failed to create int8 collection: {e}")
                    self._int8_collection = None
            else:
                self._int8_collection = None
        else:
            self._binary_collection = None
            self._int8_collection = None
        
        logger.info(
            f"Initialized ChromaDB store at '{persist_dir}' "
            f"with collection '{config.collection_name}'"
        )

    def ping(self) -> bool:
        """Test Chroma connection by counting documents."""
        try:
            self._collection.count()
            return True
        except Exception as e:
            logger.error(f"Chroma connection check failed: {e}")
            return False

    def make_doc_id(self, content: str, meta: Optional[Dict[str, Any]] = None) -> str:
        """Generate deterministic document ID from content and metadata."""
        return self._default_make_doc_id(content, meta)

    def _prepare_metadata(self, meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare metadata for Chroma storage.
        
        Chroma only supports str, int, float, bool values in metadata.
        Complex objects are JSON-serialized.
        """
        if not meta:
            return {}
        
        prepared = {}
        for key, value in meta.items():
            if isinstance(value, (str, int, float, bool)):
                prepared[key] = value
            elif value is None:
                prepared[key] = ""
            else:
                # Serialize complex types to JSON string
                prepared[f"{key}_json"] = json.dumps(value, ensure_ascii=False)
        
        return prepared

    def _restore_metadata(self, meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Restore metadata from Chroma storage format."""
        if not meta:
            return {}
        
        restored = {}
        for key, value in meta.items():
            if key.endswith("_json"):
                # Restore JSON-serialized values
                original_key = key[:-5]
                try:
                    restored[original_key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    restored[original_key] = value
            else:
                restored[key] = value
        
        return restored

    def upsert(
        self,
        doc_id: str,
        content: str,
        embedding: List[float],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or update a document with embedding."""
        # Truncate content if necessary
        truncated = False
        if len(content) > self._max_chars:
            content = content[: self._max_chars]
            truncated = True
        
        # Prepare metadata
        meta = dict(meta or {})
        if truncated:
            meta["truncated"] = True
        
        # Mark as having embedding
        meta["has_embedding"] = True
        
        # Ensure doc_level is stored (default to "child" for embedded docs)
        if "doc_level" not in meta:
            meta["doc_level"] = "child"
        
        prepared_meta = self._prepare_metadata(meta)
        
        # Upsert to collection
        self._collection.upsert(
            ids=[doc_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[prepared_meta],
        )
        
        # Store quantized versions if enabled
        if self._quant_config.enabled and QUANTIZATION_AVAILABLE:
            try:
                embedding_array = np.array([embedding], dtype=np.float32)
                
                # Store binary version
                if self._quant_config.precision in ("binary", "both") and self._binary_collection is not None:
                    try:
                        binary_emb = quantize_embeddings(embedding_array, precision="ubinary")[0]
                        self._binary_collection.upsert(
                            ids=[doc_id],
                            embeddings=[binary_emb.tolist()],
                        )
                    except Exception as e:
                        logger.debug(f"Failed to store binary embedding: {e}")
                
                # Store int8 version
                if self._quant_config.precision in ("int8", "both") and self._int8_collection is not None:
                    try:
                        int8_emb = quantize_embeddings(
                            embedding_array,
                            precision="int8",
                            ranges=self._int8_ranges
                        )[0]
                        self._int8_collection.upsert(
                            ids=[doc_id],
                            embeddings=[int8_emb.tolist()],
                        )
                    except Exception as e:
                        logger.debug(f"Failed to store int8 embedding: {e}")
            except Exception as e:
                logger.warning(f"Quantization storage failed for {doc_id}: {e}")
        
        logger.debug(f"Upserted document with embedding: {doc_id}")

    def upsert_doc_only(
        self,
        doc_id: str,
        content: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store document without embedding (for parent documents)."""
        # Truncate content if necessary
        truncated = False
        if len(content) > self._max_chars:
            content = content[: self._max_chars]
            truncated = True
        
        # Prepare metadata
        meta = dict(meta or {})
        if truncated:
            meta["truncated"] = True
        
        # Mark as not having embedding
        meta["has_embedding"] = False
        
        # Ensure doc_level is stored (default to "parent" for non-embedded docs)
        if "doc_level" not in meta:
            meta["doc_level"] = "parent"
        
        prepared_meta = self._prepare_metadata(meta)
        
        # For documents without embeddings, we use a zero vector
        # This allows them to be stored but not found via similarity search
        zero_embedding = [0.0] * self._embedding_dim
        
        self._collection.upsert(
            ids=[doc_id],
            documents=[content],
            embeddings=[zero_embedding],
            metadatas=[prepared_meta],
        )
        
        logger.debug(f"Upserted document (no embedding): {doc_id}")

    def upsert_batch(
        self,
        documents: List[Dict[str, Any]],
    ) -> int:
        """Batch insert/update documents with embeddings."""
        if not documents:
            return 0
        
        ids = []
        contents = []
        embeddings = []
        metadatas = []
        
        for doc in documents:
            doc_id = doc["doc_id"]
            content = doc["content"]
            embedding = doc["embedding"]
            meta = dict(doc.get("meta") or {})
            
            # Truncate content if necessary
            if len(content) > self._max_chars:
                content = content[: self._max_chars]
                meta["truncated"] = True
            
            meta["has_embedding"] = True
            
            ids.append(doc_id)
            contents.append(content)
            embeddings.append(embedding)
            metadatas.append(self._prepare_metadata(meta))
        
        # Batch upsert
        self._collection.upsert(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        
        logger.debug(f"Batch upserted {len(documents)} documents with embeddings")
        return len(documents)

    def upsert_doc_only_batch(
        self,
        documents: List[Dict[str, Any]],
    ) -> int:
        """Batch store documents without embeddings."""
        if not documents:
            return 0
        
        ids = []
        contents = []
        embeddings = []
        metadatas = []
        
        zero_embedding = [0.0] * self._embedding_dim
        
        for doc in documents:
            doc_id = doc["doc_id"]
            content = doc["content"]
            meta = dict(doc.get("meta") or {})
            
            # Truncate content if necessary
            if len(content) > self._max_chars:
                content = content[: self._max_chars]
                meta["truncated"] = True
            
            meta["has_embedding"] = False
            
            ids.append(doc_id)
            contents.append(content)
            embeddings.append(zero_embedding)
            metadatas.append(self._prepare_metadata(meta))
        
        # Batch upsert
        self._collection.upsert(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        
        logger.debug(f"Batch upserted {len(documents)} documents (no embedding)")
        return len(documents)

    def get_doc(self, doc_id: str) -> Optional[StoredDoc]:
        """Retrieve a document by ID."""
        try:
            result = self._collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"],
            )
            
            if not result["ids"]:
                return None
            
            content = result["documents"][0] if result["documents"] else ""
            meta = self._restore_metadata(
                result["metadatas"][0] if result["metadatas"] else {}
            )
            
            return StoredDoc(doc_id=doc_id, content=content, meta=meta)
            
        except Exception as e:
            logger.debug(f"Failed to get document {doc_id}: {e}")
            return None

    def has_embedding(self, doc_id: str) -> bool:
        """Check if document has an embedding stored."""
        try:
            result = self._collection.get(
                ids=[doc_id],
                include=["metadatas"],
            )
            
            if not result["ids"]:
                return False
            
            meta = result["metadatas"][0] if result["metadatas"] else {}
            return bool(meta.get("has_embedding", False))
            
        except Exception:
            return False

    def delete_doc(self, doc_id: str) -> bool:
        """Delete a document."""
        try:
            self._collection.delete(ids=[doc_id])
            logger.debug(f"Deleted document: {doc_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete document {doc_id}: {e}")
            return False

    def _build_where_filter(
        self,
        language_filter: Optional[str] = None,
        doc_level_filter: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build where clause for Chroma filtering."""
        # Normalize doc_level_filter
        level_value = None
        if doc_level_filter:
            filter_lower = doc_level_filter.lower()
            if filter_lower in ("child", "leaves", "leaf"):
                level_value = "child"
            elif filter_lower in ("parent", "parents"):
                level_value = "parent"
        
        # Build where clause
        conditions: List[Dict[str, Any]] = [{"has_embedding": True}]
        
        if language_filter:
            conditions.append({"language_code": language_filter})
        
        if level_value:
            conditions.append({"doc_level": level_value})
        
        if len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    def retrieve_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int,
        min_similarity: float = 0.0,
        ef_runtime: Optional[int] = None,
        language_filter: Optional[str] = None,
        doc_level_filter: Optional[str] = None,
    ) -> List[Tuple[StoredDoc, float]]:
        """Retrieve documents by vector similarity."""
        # Normalize doc_level_filter
        level_value = None
        if doc_level_filter:
            filter_lower = doc_level_filter.lower()
            if filter_lower in ("child", "leaves", "leaf"):
                level_value = "child"
            elif filter_lower in ("parent", "parents"):
                level_value = "parent"
            # "all" or None means no doc_level filter
        
        # Build where clause for filtering
        conditions = [{"has_embedding": True}]
        
        if language_filter:
            conditions.append({"language_code": language_filter})
        
        if level_value:
            conditions.append({"doc_level": level_value})
        
        if len(conditions) == 1:
            where_clause = conditions[0]
        else:
            where_clause = {"$and": conditions}
        
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Chroma query failed: {e}")
            return []
        
        if not results["ids"] or not results["ids"][0]:
            return []
        
        docs: List[Tuple[StoredDoc, float]] = []
        
        for i, doc_id in enumerate(results["ids"][0]):
            content = results["documents"][0][i] if results["documents"] else ""
            meta = self._restore_metadata(
                results["metadatas"][0][i] if results["metadatas"] else {}
            )
            distance = results["distances"][0][i] if results["distances"] else 0.0
            
            # Convert distance to similarity
            # For cosine distance: similarity = 1 - distance
            # For L2: similarity = 1 / (1 + distance)
            # For IP: distance is already negative dot product, so similarity = -distance
            similarity = 1.0 - distance  # Assuming cosine by default
            
            if similarity < min_similarity:
                continue
            
            docs.append((
                StoredDoc(doc_id=doc_id, content=content, meta=meta),
                similarity,
            ))
        
        # Sort by similarity descending
        docs.sort(key=lambda x: x[1], reverse=True)
        return docs

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
        """Retrieve using binary quantization for speed with rescoring for accuracy."""
        # Fall back if quantization not enabled
        if not self._quant_config.enabled or not QUANTIZATION_AVAILABLE or self._binary_collection is None:
            logger.debug("Quantization not available, using standard retrieval")
            return self.retrieve_by_embedding(
                query_embedding, top_k, min_similarity,
                language_filter=language_filter,
                doc_level_filter=doc_level_filter
            )
        
        # Use config defaults
        rescore_mult = rescore_multiplier if rescore_multiplier is not None else self._quant_config.rescore_multiplier
        use_rescore = use_rescoring if use_rescoring is not None else self._quant_config.use_rescoring
        candidate_k = int(top_k * rescore_mult) if use_rescore else top_k
        
        # Stage 1: Binary search
        try:
            query_array = np.array([query_embedding], dtype=np.float32)
            binary_query = quantize_embeddings(query_array, precision="ubinary")[0]
        except Exception as e:
            logger.warning(f"Binary quantization failed: {e}")
            return self.retrieve_by_embedding(
                query_embedding, top_k, min_similarity,
                language_filter=language_filter,
                doc_level_filter=doc_level_filter
            )
        
        # Build where filter
        where_filter = self._build_where_filter(language_filter, doc_level_filter)
        
        # Query binary collection
        try:
            results = self._binary_collection.query(
                query_embeddings=[binary_query.tolist()],
                n_results=candidate_k,
                where=where_filter if where_filter else None,
                include=["distances"],
            )
            
            candidate_ids = results["ids"][0] if results["ids"] else []
        except Exception as e:
            logger.warning(f"Binary search failed: {e}")
            return self.retrieve_by_embedding(
                query_embedding, top_k, min_similarity,
                language_filter=language_filter,
                doc_level_filter=doc_level_filter
            )
        
        if not candidate_ids:
            return []
        
        if not use_rescore:
            # Get documents for binary results
            final_results: List[Tuple[StoredDoc, float]] = []
            for doc_id in candidate_ids[:top_k]:
                doc = self.get_doc(doc_id)
                if doc:
                    final_results.append((doc, 1.0))  # Placeholder score
            return final_results
        
        # Stage 2: Rescore with int8/float32
        query_vec = np.array(query_embedding, dtype=np.float32)
        candidate_embeddings: List[np.ndarray] = []
        candidate_docs: List[StoredDoc] = []
        
        for doc_id in candidate_ids:
            doc = self.get_doc(doc_id)
            if not doc:
                continue
            
            # Try int8 collection first
            if self._int8_collection is not None:
                try:
                    int8_result = self._int8_collection.get(
                        ids=[doc_id],
                        include=["embeddings"],
                    )
                    if int8_result["embeddings"] and int8_result["embeddings"][0]:
                        emb = np.array(int8_result["embeddings"][0], dtype=np.int8).astype(np.float32)
                        candidate_embeddings.append(emb)
                        candidate_docs.append(doc)
                        continue
                except Exception:
                    pass
            
            # Fall back to float32
            try:
                float_result = self._collection.get(
                    ids=[doc_id],
                    include=["embeddings"],
                )
                if float_result["embeddings"] and float_result["embeddings"][0]:
                    emb = np.array(float_result["embeddings"][0], dtype=np.float32)
                    candidate_embeddings.append(emb)
                    candidate_docs.append(doc)
            except Exception as e:
                logger.debug(f"Failed to load embedding for rescoring: {e}")
                continue
        
        if not candidate_embeddings:
            logger.warning("No embeddings loaded for rescoring")
            return []
        
        # Rescore
        rescored = rescore_candidates(query_vec, candidate_embeddings, [d.doc_id for d in candidate_docs])
        
        # Build results
        doc_map = {doc.doc_id: doc for doc in candidate_docs}
        results: List[Tuple[StoredDoc, float]] = []
        for doc_id, score in rescored[:top_k]:
            if score >= min_similarity and doc_id in doc_map:
                results.append((doc_map[doc_id], score))
        
        logger.debug(
            f"Quantized retrieval: {len(candidate_ids)} candidates → "
            f"{len(rescored)} rescored → {len(results)} returned"
        )
        
        return results

    def list_doc_ids(self, pattern: str = "*", limit: int = 10_000) -> List[str]:
        """List document IDs (pattern ignored for Chroma)."""
        try:
            result = self._collection.get(
                limit=limit,
                include=[],
            )
            return result["ids"] if result["ids"] else []
        except Exception as e:
            logger.warning(f"Failed to list document IDs: {e}")
            return []

    def list_doc_ids_with_embeddings(self, limit: int = 10_000) -> List[str]:
        """List document IDs that have embeddings stored."""
        try:
            result = self._collection.get(
                where={"has_embedding": True},
                limit=limit,
                include=[],
            )
            return result["ids"] if result["ids"] else []
        except Exception as e:
            logger.warning(f"Failed to list document IDs with embeddings: {e}")
            return []

    def get_index_info(self) -> Dict[str, Any]:
        """Get Chroma collection statistics."""
        try:
            count = self._collection.count()
            
            # Try to get collection metadata
            metadata = {}
            try:
                metadata = self._collection.metadata or {}
            except Exception:
                pass
            
            return {
                "name": self._config.collection_name,
                "backend": "chroma",
                "persist_directory": self._config.persist_directory,
                "document_count": count,
                "distance_function": self._config.distance_fn,
                "embedding_dimension": self._embedding_dim,
                "metadata": metadata,
            }
        except Exception as e:
            return {
                "name": self._config.collection_name,
                "backend": "chroma",
                "error": str(e),
            }

    def drop_index(self, delete_documents: bool = False) -> bool:
        """Drop the collection and optionally delete all data."""
        try:
            if delete_documents:
                # Delete the collection entirely
                self._client.delete_collection(self._config.collection_name)
                
                # Recreate empty collection
                distance_fn_map = {
                    "cosine": "cosine",
                    "l2": "l2",
                    "ip": "ip",
                    "inner_product": "ip",
                }
                distance_fn = distance_fn_map.get(
                    self._config.distance_fn.lower(), "cosine"
                )
                
                self._collection = self._client.get_or_create_collection(
                    name=self._config.collection_name,
                    metadata={"hnsw:space": distance_fn},
                )
            else:
                # Just clear all documents
                all_ids = self.list_doc_ids(limit=100_000)
                if all_ids:
                    # Delete in batches
                    batch_size = 5000
                    for i in range(0, len(all_ids), batch_size):
                        batch = all_ids[i:i + batch_size]
                        self._collection.delete(ids=batch)
            
            logger.info(f"Dropped Chroma collection '{self._config.collection_name}'")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to drop collection: {e}")
            return False

    def count_documents(self) -> int:
        """Count total documents stored."""
        try:
            return self._collection.count()
        except Exception:
            return 0
