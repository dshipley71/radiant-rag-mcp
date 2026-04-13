"""
Redis-based storage with vector search (ANN indexing) for Radiant Agentic RAG.

Uses Redis Stack with RediSearch module for efficient vector similarity search
via HNSW (Hierarchical Navigable Small World) indexing.

Requirements:
    - Redis Stack (includes RediSearch, RedisJSON)
    - redis-py >= 4.0
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import redis

from radiant_rag_mcp.config import RedisConfig
from radiant_rag_mcp.storage.base import BaseVectorStore, StoredDoc
from radiant_rag_mcp.storage.quantization import (
    quantize_embeddings,
    embedding_to_bytes as quant_embedding_to_bytes,
    bytes_to_embedding as quant_bytes_to_embedding,
    get_binary_dimension,
    rescore_candidates,
    QUANTIZATION_AVAILABLE,
)

# Try to import Redis Search components (requires redis-stack)
REDIS_SEARCH_IMPORTS_AVAILABLE = False
_REDIS_SEARCH_IMPORT_ERROR = None
try:
    from redis.commands.search.field import TagField, TextField, VectorField
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
    REDIS_SEARCH_IMPORTS_AVAILABLE = True
except ImportError as e:
    _REDIS_SEARCH_IMPORT_ERROR = str(e)
    TagField = TextField = VectorField = None
    IndexDefinition = IndexType = Query = None

logger = logging.getLogger(__name__)

# Log import status after logger is defined
if _REDIS_SEARCH_IMPORT_ERROR:
    logger.debug(f"Redis Search imports not available: {_REDIS_SEARCH_IMPORT_ERROR}")


def _sha256_hex(text: str) -> str:
    """Generate SHA-256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check_redis_search_module(client: redis.Redis) -> bool:
    """
    Check if RediSearch module is loaded in Redis server AND Python imports are available.
    
    Args:
        client: Redis client instance
        
    Returns:
        True if RediSearch module is available and usable
    """
    # First check if Python imports are available
    if not REDIS_SEARCH_IMPORTS_AVAILABLE:
        # Check if the server has Search even if we can't use it
        try:
            client.execute_command("FT._LIST")
            logger.warning(
                "Redis Search module is available on server, but Python redis.commands.search "
                "imports failed. Try upgrading redis-py: pip install --upgrade redis"
            )
        except redis.ResponseError:
            pass
        return False
    
    try:
        # Use MODULE LIST to check for search module
        modules = client.execute_command("MODULE", "LIST")
        logger.debug(f"Redis MODULE LIST returned {len(modules)} modules")
        
        # modules is a list of module info
        for module in modules:
            # Module info can be a list or dict depending on Redis version
            if isinstance(module, (list, tuple)):
                # Format: [b'name', b'search', b'ver', 20812, ...]
                for i, item in enumerate(module):
                    if item in (b'name', 'name') and i + 1 < len(module):
                        name = module[i + 1]
                        if isinstance(name, bytes):
                            name = name.decode()
                        if name.lower() in ('search', 'ft'):
                            logger.debug(f"Found Redis Search module: {name}")
                            return True
            elif isinstance(module, dict):
                name = module.get(b'name', module.get('name', ''))
                if isinstance(name, bytes):
                    name = name.decode()
                if name.lower() in ('search', 'ft'):
                    logger.debug(f"Found Redis Search module (dict): {name}")
                    return True
        
        # Fallback: try a direct FT command
        # Some Redis configurations don't list modules but still have Search
        logger.debug("Search module not in MODULE LIST, trying FT._LIST fallback")
        try:
            client.execute_command("FT._LIST")
            logger.debug("FT._LIST succeeded - Redis Search is available")
            return True
        except redis.ResponseError as e:
            logger.debug(f"FT._LIST failed: {e}")
        
        return False
        
    except redis.ResponseError as e:
        # MODULE LIST not available (older Redis or ACL restriction)
        logger.debug(f"MODULE LIST failed ({e}), trying FT._LIST fallback")
        try:
            client.execute_command("FT._LIST")
            logger.debug("FT._LIST succeeded - Redis Search is available")
            return True
        except redis.ResponseError:
            return False
    except Exception as e:
        logger.debug(f"Error checking Redis modules: {e}")
        return False


class RedisVectorStore(BaseVectorStore):
    """
    Redis-based document and vector storage with ANN search.
    
    Storage schema (using Redis Hashes):
        {prefix}:{doc_ns}:{doc_id} -> Hash with fields:
            - content: document text
            - meta: JSON-encoded metadata
            - embedding: binary vector (for indexed docs)
            - doc_level: "parent" or "child"
            - parent_id: parent doc ID (for children)
    
    Vector index:
        Uses HNSW algorithm for approximate nearest neighbor search.
        Only documents with embeddings are indexed for vector search.
    """

    def __init__(self, config: RedisConfig) -> None:
        """
        Initialize Redis vector store.
        
        Args:
            config: Redis configuration
        """
        self._config = config
        self._r = redis.Redis.from_url(config.url, decode_responses=False)
        self._r_text = redis.Redis.from_url(config.url, decode_responses=True)
        self._prefix = config.key_prefix
        self._doc_ns = config.doc_ns
        self._embed_ns = config.embed_ns
        self._vector_config = config.vector_index
        self._max_chars = config.max_content_chars
        self._index_created = False
        self._redis_search_available: Optional[bool] = None
        self._embedding_dim: Optional[int] = None
        
        # Quantization configuration
        self._quant_config = config.quantization
        self._binary_index_created = False
        self._int8_index_created = False
        
        # Load int8 calibration ranges if provided
        self._int8_ranges: Optional[np.ndarray] = None
        if self._quant_config.enabled and self._quant_config.int8_ranges_file:
            try:
                self._int8_ranges = np.load(self._quant_config.int8_ranges_file)
                logger.info(f"Loaded int8 calibration ranges from {self._quant_config.int8_ranges_file}")
            except Exception as e:
                logger.warning(f"Failed to load int8 ranges: {e}")
    
    def _check_search_available(self) -> bool:
        """
        Check if Redis Search module is available (cached after first check).
        
        Returns:
            True if RediSearch module is loaded in Redis server
        """
        if self._redis_search_available is None:
            self._redis_search_available = _check_redis_search_module(self._r)
            if self._redis_search_available:
                logger.info("Redis Search module detected - using HNSW vector index")
            else:
                logger.warning(
                    "Redis Search module NOT available. Vector search will use fallback linear scan. "
                    "For production use, install Redis Stack: docker run -p 6379:6379 redis/redis-stack"
                )
        return self._redis_search_available

    def ping(self) -> bool:
        """Test Redis connection."""
        try:
            return bool(self._r.ping())
        except redis.ConnectionError as e:
            logger.error(f"Redis connection failed: {e}")
            return False

    def _doc_key(self, doc_id: str) -> str:
        """Generate Redis key for a document."""
        return f"{self._prefix}:{self._doc_ns}:{doc_id}"

    def make_doc_id(self, content: str, meta: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate deterministic document ID from content and metadata.
        
        Args:
            content: Document text
            meta: Optional metadata (included in hash)
            
        Returns:
            SHA-256 hash as document ID
        """
        meta_part = json.dumps(meta or {}, sort_keys=True, ensure_ascii=False)
        return _sha256_hex(content + "\n" + meta_part)

    def _ensure_index(self, embedding_dim: int) -> None:
        """
        Ensure vector index exists, creating if necessary.
        
        Args:
            embedding_dim: Dimension of embedding vectors
        """
        # Store embedding dimension for later reference
        self._embedding_dim = embedding_dim
        
        if self._index_created:
            return

        if not self._check_search_available():
            # Warning already logged by _check_search_available
            self._index_created = True
            return

        index_name = self._vector_config.name
        
        try:
            # Check if index exists
            self._r.ft(index_name).info()
            logger.debug(f"Vector index '{index_name}' already exists")
            self._index_created = True
            return
        except redis.ResponseError:
            # Index doesn't exist, create it
            pass

        logger.info(f"Creating vector index '{index_name}'...")

        # Define index schema
        schema = [
            TextField("content", no_stem=True),
            TagField("doc_level"),
            TagField("parent_id"),
            TagField("language_code"),  # Language filtering support
            VectorField(
                "embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": embedding_dim,
                    "DISTANCE_METRIC": self._vector_config.distance_metric,
                    "M": self._vector_config.hnsw_m,
                    "EF_CONSTRUCTION": self._vector_config.hnsw_ef_construction,
                },
            ),
        ]

        # Create index on hash keys matching our prefix pattern
        index_def = IndexDefinition(
            prefix=[f"{self._prefix}:{self._doc_ns}:"],
            index_type=IndexType.HASH,
        )

        try:
            self._r.ft(index_name).create_index(
                schema,
                definition=index_def,
            )
            logger.info(f"Vector index '{index_name}' created successfully")
            self._index_created = True
        except redis.ResponseError as e:
            if "Index already exists" in str(e):
                self._index_created = True
            else:
                raise

    def _embedding_to_bytes(self, embedding: List[float]) -> bytes:
        """Convert embedding list to bytes for Redis storage."""
        return np.array(embedding, dtype=np.float32).tobytes()

    def _bytes_to_embedding(self, data: bytes) -> List[float]:
        """Convert bytes back to embedding list."""
        return np.frombuffer(data, dtype=np.float32).tolist()
    
    def _binary_doc_key(self, doc_id: str) -> str:
        """Generate Redis key for binary embedding."""
        return f"{self._prefix}:{self._doc_ns}_binary:{doc_id}"
    
    def _int8_doc_key(self, doc_id: str) -> str:
        """Generate Redis key for int8 embedding."""
        return f"{self._prefix}:{self._doc_ns}_int8:{doc_id}"
    
    def _store_quantized_embeddings(
        self,
        doc_id: str,
        embedding: List[float],
    ) -> None:
        """Store quantized versions of embedding (binary and/or int8)."""
        if not self._quant_config.enabled or not QUANTIZATION_AVAILABLE:
            return
        
        try:
            embedding_array = np.array([embedding], dtype=np.float32)
            
            # Store binary version
            if self._quant_config.precision in ("binary", "both"):
                try:
                    binary_emb = quantize_embeddings(embedding_array, precision="ubinary")[0]
                    self._r.hset(
                        self._binary_doc_key(doc_id),
                        "binary_embedding",
                        quant_embedding_to_bytes(binary_emb)
                    )
                except Exception as e:
                    logger.debug(f"Failed to store binary embedding for {doc_id}: {e}")
            
            # Store int8 version
            if self._quant_config.precision in ("int8", "both"):
                try:
                    int8_emb = quantize_embeddings(
                        embedding_array,
                        precision="int8",
                        ranges=self._int8_ranges
                    )[0]
                    self._r.hset(
                        self._int8_doc_key(doc_id),
                        "int8_embedding",
                        quant_embedding_to_bytes(int8_emb)
                    )
                except Exception as e:
                    logger.debug(f"Failed to store int8 embedding for {doc_id}: {e}")
        except Exception as e:
            logger.warning(f"Quantization storage failed for {doc_id}: {e}")
    
    def _load_binary_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """Load binary embedding for a document."""
        try:
            data = self._r.hget(self._binary_doc_key(doc_id), "binary_embedding")
            if data:
                binary_dim = get_binary_dimension(self._embedding_dim or 384)
                return quant_bytes_to_embedding(data, np.uint8, (binary_dim,))
            return None
        except Exception as e:
            logger.debug(f"Failed to load binary embedding for {doc_id}: {e}")
            return None
    
    def _load_int8_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """Load int8 embedding for a document."""
        try:
            data = self._r.hget(self._int8_doc_key(doc_id), "int8_embedding")
            if data:
                return quant_bytes_to_embedding(data, np.int8, (self._embedding_dim or 384,))
            return None
        except Exception as e:
            logger.debug(f"Failed to load int8 embedding for {doc_id}: {e}")
            return None

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
        # Ensure index exists
        self._ensure_index(len(embedding))

        # Truncate content if necessary
        truncated = False
        if len(content) > self._max_chars:
            content = content[: self._max_chars]
            truncated = True

        # Prepare metadata
        meta = dict(meta or {})
        if truncated:
            meta["truncated"] = True

        # Extract special fields for indexing
        doc_level = str(meta.get("doc_level", "child"))
        parent_id = str(meta.get("parent_id", ""))
        language_code = str(meta.get("language_code", "en"))  # Default to English

        # Store as hash
        key = self._doc_key(doc_id)
        mapping = {
            "content": content.encode("utf-8"),
            "meta": json.dumps(meta, ensure_ascii=False).encode("utf-8"),
            "embedding": self._embedding_to_bytes(embedding),
            "doc_level": doc_level.encode("utf-8"),
            "parent_id": parent_id.encode("utf-8"),
            "language_code": language_code.encode("utf-8"),
        }

        self._r.hset(key, mapping=mapping)
        
        # Store quantized versions if enabled
        self._store_quantized_embeddings(doc_id, embedding)
        
        logger.debug(f"Upserted document: {doc_id}")

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
        # Truncate content if necessary
        truncated = False
        if len(content) > self._max_chars:
            content = content[: self._max_chars]
            truncated = True

        # Prepare metadata
        meta = dict(meta or {})
        if truncated:
            meta["truncated"] = True

        doc_level = str(meta.get("doc_level", "parent"))
        parent_id = str(meta.get("parent_id", ""))
        language_code = str(meta.get("language_code", "en"))

        # Store as hash (without embedding field)
        key = self._doc_key(doc_id)
        mapping = {
            "content": content.encode("utf-8"),
            "meta": json.dumps(meta, ensure_ascii=False).encode("utf-8"),
            "doc_level": doc_level.encode("utf-8"),
            "parent_id": parent_id.encode("utf-8"),
            "language_code": language_code.encode("utf-8"),
        }

        self._r.hset(key, mapping=mapping)
        logger.debug(f"Upserted document (no embedding): {doc_id}")

    def upsert_batch(
        self,
        documents: List[Dict[str, Any]],
    ) -> int:
        """
        Batch insert/update documents with embeddings using Redis pipeline.
        
        Args:
            documents: List of dicts with keys: doc_id, content, embedding, meta
            
        Returns:
            Number of documents stored
        """
        if not documents:
            return 0

        # Ensure index exists (use first embedding to get dimension)
        if documents and documents[0].get("embedding"):
            self._ensure_index(len(documents[0]["embedding"]))

        pipeline = self._r.pipeline()
        count = 0

        for doc in documents:
            doc_id = doc["doc_id"]
            content = doc["content"]
            embedding = doc["embedding"]
            meta = dict(doc.get("meta") or {})

            # Truncate content if necessary
            if len(content) > self._max_chars:
                content = content[: self._max_chars]
                meta["truncated"] = True

            # Extract special fields for indexing
            doc_level = str(meta.get("doc_level", "child"))
            parent_id = str(meta.get("parent_id", ""))
            language_code = str(meta.get("language_code", "en"))

            # Store as hash
            key = self._doc_key(doc_id)
            mapping = {
                "content": content.encode("utf-8"),
                "meta": json.dumps(meta, ensure_ascii=False).encode("utf-8"),
                "embedding": self._embedding_to_bytes(embedding),
                "doc_level": doc_level.encode("utf-8"),
                "parent_id": parent_id.encode("utf-8"),
                "language_code": language_code.encode("utf-8"),
            }

            pipeline.hset(key, mapping=mapping)
            count += 1

        # Execute pipeline
        pipeline.execute()
        logger.debug(f"Batch upserted {count} documents with embeddings")
        return count

    def upsert_doc_only_batch(
        self,
        documents: List[Dict[str, Any]],
    ) -> int:
        """
        Batch store documents without embeddings using Redis pipeline.
        
        Args:
            documents: List of dicts with keys: doc_id, content, meta
            
        Returns:
            Number of documents stored
        """
        if not documents:
            return 0

        pipeline = self._r.pipeline()
        count = 0

        for doc in documents:
            doc_id = doc["doc_id"]
            content = doc["content"]
            meta = dict(doc.get("meta") or {})

            # Truncate content if necessary
            if len(content) > self._max_chars:
                content = content[: self._max_chars]
                meta["truncated"] = True

            doc_level = str(meta.get("doc_level", "parent"))
            parent_id = str(meta.get("parent_id", ""))
            language_code = str(meta.get("language_code", "en"))

            # Store as hash (without embedding field)
            key = self._doc_key(doc_id)
            mapping = {
                "content": content.encode("utf-8"),
                "meta": json.dumps(meta, ensure_ascii=False).encode("utf-8"),
                "doc_level": doc_level.encode("utf-8"),
                "parent_id": parent_id.encode("utf-8"),
                "language_code": language_code.encode("utf-8"),
            }

            pipeline.hset(key, mapping=mapping)
            count += 1

        # Execute pipeline
        pipeline.execute()
        logger.debug(f"Batch upserted {count} documents (no embedding)")
        return count

    def get_doc(self, doc_id: str) -> Optional[StoredDoc]:
        """
        Retrieve a document by ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            StoredDoc if found, None otherwise
        """
        key = self._doc_key(doc_id)
        data = self._r.hgetall(key)

        if not data:
            return None

        content = data.get(b"content", b"").decode("utf-8")
        meta_raw = data.get(b"meta", b"{}").decode("utf-8")

        try:
            meta = json.loads(meta_raw)
        except json.JSONDecodeError:
            meta = {}

        return StoredDoc(doc_id=doc_id, content=content, meta=meta)

    def has_embedding(self, doc_id: str) -> bool:
        """Check if document has an embedding stored."""
        key = self._doc_key(doc_id)
        return self._r.hexists(key, "embedding")

    def delete_doc(self, doc_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if document was deleted
        """
        key = self._doc_key(doc_id)
        deleted = self._r.delete(key)
        if deleted:
            logger.debug(f"Deleted document: {doc_id}")
        return bool(deleted)

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
        Retrieve documents by vector similarity using ANN search.
        
        Args:
            query_embedding: Query vector
            top_k: Maximum number of results
            min_similarity: Minimum similarity threshold (0.0 to 1.0 for cosine)
            ef_runtime: Optional HNSW ef parameter for this query
            language_filter: Optional ISO 639-1 language code to filter by
            doc_level_filter: Optional filter by document level:
                - None or "all": Return both parents and children
                - "child" or "leaves": Return only leaf/child documents
                - "parent" or "parents": Return only parent documents
            
        Returns:
            List of (document, similarity_score) tuples, sorted by score descending
        """
        # Ensure index exists
        self._ensure_index(len(query_embedding))

        # Fallback to linear scan if Redis Search not available
        if not self._redis_search_available:
            return self._retrieve_by_embedding_linear(
                query_embedding, top_k, min_similarity, doc_level_filter
            )

        index_name = self._vector_config.name

        # Normalize doc_level_filter
        level_value = None
        if doc_level_filter:
            filter_lower = doc_level_filter.lower()
            if filter_lower in ("child", "leaves", "leaf"):
                level_value = "child"
            elif filter_lower in ("parent", "parents"):
                level_value = "parent"
            # "all" or None means no filter

        # Build filter expression
        filters = []
        if language_filter:
            filters.append(f"@language_code:{{{language_filter}}}")
        if level_value:
            filters.append(f"@doc_level:{{{level_value}}}")

        # Build KNN query with optional filters
        # For COSINE distance in Redis, score is 1 - cosine_similarity
        # So we need to convert: similarity = 1 - score
        if filters:
            filter_str = " ".join(filters)
            query_str = f"({filter_str})=>[KNN {top_k} @embedding $vec AS score]"
        else:
            query_str = f"*=>[KNN {top_k} @embedding $vec AS score]"
        
        query = (
            Query(query_str)
            .sort_by("score", asc=True)  # Lower score = higher similarity for cosine
            .return_fields("content", "meta", "score", "language_code")
            .dialect(2)
            .paging(0, top_k)
        )

        # Set EF_RUNTIME for this query
        params = {
            "vec": self._embedding_to_bytes(query_embedding),
        }

        try:
            results = self._r.ft(index_name).search(query, query_params=params)
        except redis.ResponseError as e:
            logger.error(f"Vector search failed: {e}")
            # Fallback to linear scan
            return self._retrieve_by_embedding_linear(
                query_embedding, top_k, min_similarity
            )

        docs: List[Tuple[StoredDoc, float]] = []
        for doc in results.docs:
            # Extract doc_id from key
            key = doc.id
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            
            # Extract doc_id from key pattern: prefix:ns:doc_id
            parts = key.split(":")
            doc_id = parts[-1] if len(parts) >= 3 else key

            # Get content and meta with fallbacks
            content = getattr(doc, 'content', None) or ""
            if isinstance(content, bytes):
                content = content.decode("utf-8")

            meta_raw = getattr(doc, 'meta', None) or "{}"
            if isinstance(meta_raw, bytes):
                meta_raw = meta_raw.decode("utf-8")
            
            try:
                meta = json.loads(meta_raw) if meta_raw else {}
            except json.JSONDecodeError:
                meta = {}

            # Convert score to similarity
            # For COSINE: similarity = 1 - distance_score
            raw_score = float(getattr(doc, 'score', 0))
            similarity = 1.0 - raw_score

            # Apply minimum similarity filter
            if similarity < min_similarity:
                continue

            docs.append((StoredDoc(doc_id=doc_id, content=content, meta=meta), similarity))

        # Sort by similarity descending (should already be sorted, but ensure)
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
        """
        Retrieve documents using binary quantization for speed.
        
        Two-stage process:
        1. Retrieve candidates using standard retrieval (binary search needs index)
        2. Rescore with int8/float32 embeddings for accuracy
        
        Args:
            query_embedding: Query vector (float32)
            top_k: Maximum number of results
            min_similarity: Minimum similarity threshold
            rescore_multiplier: Retrieve N×top_k candidates for rescoring
            use_rescoring: Whether to use rescoring step
            language_filter: Optional language code filter
            doc_level_filter: Optional document level filter
            
        Returns:
            List of (document, similarity_score) tuples sorted by score descending
        """
        # Fall back to standard retrieval if quantization disabled
        if not self._quant_config.enabled or not QUANTIZATION_AVAILABLE:
            logger.debug("Quantization not enabled, using standard retrieval")
            return self.retrieve_by_embedding(
                query_embedding, top_k, min_similarity,
                language_filter=language_filter,
                doc_level_filter=doc_level_filter
            )
        
        # Use config defaults if not specified
        rescore_mult = rescore_multiplier if rescore_multiplier is not None else self._quant_config.rescore_multiplier
        use_rescore = use_rescoring if use_rescoring is not None else self._quant_config.use_rescoring
        
        # Stage 1: Get candidates (using standard retrieval for now)
        # In production, this would use a binary HNSW index for speed
        candidate_k = int(top_k * rescore_mult) if use_rescore else top_k
        
        candidates = self.retrieve_by_embedding(
            query_embedding,
            candidate_k,
            min_similarity=0.0,  # Get all candidates for rescoring
            language_filter=language_filter,
            doc_level_filter=doc_level_filter,
        )
        
        if not use_rescore or not candidates:
            return candidates[:top_k]
        
        # Stage 2: Rescore with int8/float32 embeddings
        query_vec = np.array(query_embedding, dtype=np.float32)
        candidate_embeddings: List[np.ndarray] = []
        candidate_ids: List[str] = []
        candidate_docs: List[StoredDoc] = []
        
        for doc, _ in candidates:
            # Try to load int8 embedding first (faster than float32)
            int8_emb = self._load_int8_embedding(doc.doc_id)
            
            if int8_emb is not None:
                candidate_embeddings.append(int8_emb.astype(np.float32))
                candidate_ids.append(doc.doc_id)
                candidate_docs.append(doc)
            else:
                # Fall back to float32 from stored embedding
                try:
                    key = self._doc_key(doc.doc_id)
                    emb_bytes = self._r.hget(key, "embedding")
                    if emb_bytes:
                        float_emb = np.frombuffer(emb_bytes, dtype=np.float32)
                        candidate_embeddings.append(float_emb)
                        candidate_ids.append(doc.doc_id)
                        candidate_docs.append(doc)
                except Exception as e:
                    logger.debug(f"Failed to load embedding for rescoring: {e}")
                    continue
        
        if not candidate_embeddings:
            logger.warning("No embeddings loaded for rescoring, returning original candidates")
            return candidates[:top_k]
        
        # Rescore using cosine similarity
        rescored = rescore_candidates(query_vec, candidate_embeddings, candidate_ids)
        
        # Build final results with documents
        doc_map = {doc.doc_id: doc for doc in candidate_docs}
        results: List[Tuple[StoredDoc, float]] = []
        for doc_id, score in rescored[:top_k]:
            if score >= min_similarity and doc_id in doc_map:
                results.append((doc_map[doc_id], score))
        
        logger.debug(
            f"Quantized retrieval: {len(candidates)} candidates → "
            f"{len(rescored)} rescored → {len(results)} returned"
        )
        
        return results

    def _retrieve_by_embedding_linear(
        self,
        query_embedding: List[float],
        top_k: int,
        min_similarity: float = 0.0,
        doc_level_filter: Optional[str] = None,
    ) -> List[Tuple[StoredDoc, float]]:
        """
        Fallback linear scan retrieval when Redis Search is not available.
        
        Warning: This is O(n) and not suitable for large document collections.
        """
        logger.debug("Using linear scan fallback for vector retrieval")
        
        # Normalize doc_level_filter
        level_value = None
        if doc_level_filter:
            filter_lower = doc_level_filter.lower()
            if filter_lower in ("child", "leaves", "leaf"):
                level_value = "child"
            elif filter_lower in ("parent", "parents"):
                level_value = "parent"
        
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        query_vec = query_vec / query_norm

        # Get all document IDs with embeddings
        doc_ids = self.list_doc_ids(limit=200_000)
        
        scored_docs: List[Tuple[StoredDoc, float]] = []
        
        # Batch fetch for efficiency
        batch_size = 100
        for i in range(0, len(doc_ids), batch_size):
            batch_ids = doc_ids[i:i + batch_size]
            keys = [self._doc_key(doc_id) for doc_id in batch_ids]
            
            pipeline = self._r.pipeline()
            for key in keys:
                pipeline.hgetall(key)
            
            results = pipeline.execute()
            
            for doc_id, data in zip(batch_ids, results):
                if not data or b"embedding" not in data:
                    continue
                
                # Apply doc_level filter
                if level_value:
                    doc_level = data.get(b"doc_level", b"child").decode("utf-8")
                    if doc_level != level_value:
                        continue
                
                # Get embedding
                emb_bytes = data.get(b"embedding")
                if not emb_bytes:
                    continue
                
                doc_vec = self._bytes_to_embedding(emb_bytes)
                doc_vec = np.array(doc_vec, dtype=np.float32)
                doc_norm = np.linalg.norm(doc_vec)
                if doc_norm == 0:
                    continue
                doc_vec = doc_vec / doc_norm
                
                # Cosine similarity
                similarity = float(np.dot(query_vec, doc_vec))
                
                if similarity < min_similarity:
                    continue
                
                # Get content and meta
                content = data.get(b"content", b"").decode("utf-8")
                meta_raw = data.get(b"meta", b"{}").decode("utf-8")
                try:
                    meta = json.loads(meta_raw)
                except json.JSONDecodeError:
                    meta = {}
                
                scored_docs.append((
                    StoredDoc(doc_id=doc_id, content=content, meta=meta),
                    similarity,
                ))

        # Sort and return top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]

    def list_doc_ids(self, pattern: str = "*", limit: int = 10_000) -> List[str]:
        """
        List document IDs matching a pattern.
        
        Args:
            pattern: Redis key pattern (default: all docs)
            limit: Maximum number of IDs to return
            
        Returns:
            List of document IDs
        """
        full_pattern = f"{self._prefix}:{self._doc_ns}:{pattern}"
        ids: List[str] = []
        cursor = 0

        while True:
            cursor, keys = self._r.scan(cursor=cursor, match=full_pattern, count=1000)
            for key in keys:
                if isinstance(key, bytes):
                    key = key.decode("utf-8")
                parts = key.split(":")
                if len(parts) >= 3:
                    ids.append(parts[-1])
                if len(ids) >= limit:
                    return ids
            if cursor == 0:
                break

        return ids

    def list_doc_ids_with_embeddings(self, limit: int = 10_000) -> List[str]:
        """
        List document IDs that have embeddings stored.
        
        Uses the vector index to efficiently find indexed documents.
        For large limits (>10000), uses batched pagination to work around
        Redis Search's 10000 result limit.
        
        Args:
            limit: Maximum number of IDs to return
            
        Returns:
            List of document IDs
        """
        if not self._check_search_available():
            return self._list_doc_ids_with_embeddings_scan(limit)
        
        index_name = self._vector_config.name
        
        # Redis Search has a hard limit of 10000 results per query
        REDIS_SEARCH_MAX_LIMIT = 10000
        
        try:
            ids: List[str] = []
            offset = 0
            batch_size = min(limit, REDIS_SEARCH_MAX_LIMIT)
            
            while len(ids) < limit:
                remaining = limit - len(ids)
                fetch_size = min(batch_size, remaining)
                
                # Query documents in batches using LIMIT offset, count
                query = Query("*").return_fields("__key").paging(offset, fetch_size)
                results = self._r.ft(index_name).search(query)
                
                if not results.docs:
                    # No more results
                    break
                
                for doc in results.docs:
                    key = doc.id
                    if isinstance(key, bytes):
                        key = key.decode("utf-8")
                    parts = key.split(":")
                    if len(parts) >= 3:
                        ids.append(parts[-1])
                
                # Check if we got fewer results than requested (end of data)
                if len(results.docs) < fetch_size:
                    break
                
                offset += fetch_size
                
                # Safety check to avoid infinite loops
                if offset > 1_000_000:
                    logger.warning("list_doc_ids_with_embeddings: exceeded 1M offset, stopping")
                    break
            
            return ids
            
        except redis.ResponseError as e:
            logger.warning(f"Index query failed, falling back to scan: {e}")
            # Fallback to scanning
            return self._list_doc_ids_with_embeddings_scan(limit)

    def _list_doc_ids_with_embeddings_scan(self, limit: int) -> List[str]:
        """Fallback method using SCAN for listing docs with embeddings."""
        all_ids = self.list_doc_ids(limit=limit)
        ids_with_embeddings = []
        
        for doc_id in all_ids:
            if self.has_embedding(doc_id):
                ids_with_embeddings.append(doc_id)
                if len(ids_with_embeddings) >= limit:
                    break
        
        return ids_with_embeddings

    def get_index_info(self) -> Dict[str, Any]:
        """Get vector index statistics."""
        index_name = self._vector_config.name
        
        # Count documents in store (works regardless of Redis Search availability)
        try:
            pattern = f"{self._prefix}:{self._doc_ns}:*"
            doc_count = 0
            for _ in self._r.scan_iter(match=pattern, count=1000):
                doc_count += 1
        except Exception:
            doc_count = 0
        
        if not self._check_search_available():
            result = {
                "name": index_name,
                "exists": False,
                "redis_search_available": False,
                "document_count": doc_count,
            }
            # Check if it's an import issue vs server missing module
            if not REDIS_SEARCH_IMPORTS_AVAILABLE:
                result["imports_available"] = False
                result["import_error"] = _REDIS_SEARCH_IMPORT_ERROR
                # Check if server actually has it
                try:
                    self._r.execute_command("FT._LIST")
                    result["server_has_module"] = True
                except redis.ResponseError:
                    result["server_has_module"] = False
            return result
        
        try:
            info = self._r.ft(index_name).info()
            result = {
                "name": index_name,
                "exists": True,
                "document_count": int(info.get("num_docs", 0)),
                "num_terms": info.get("num_terms", 0),
                "num_records": info.get("num_records", 0),
                "indexing": info.get("indexing", False),
                "redis_search_available": True,
            }
            if self._embedding_dim:
                result["dimension"] = self._embedding_dim
            return result
        except redis.ResponseError:
            return {
                "name": index_name,
                "exists": False,
                "redis_search_available": True,
                "document_count": doc_count,
            }

    def drop_index(self, delete_documents: bool = False) -> bool:
        """
        Drop the vector index.
        
        Args:
            delete_documents: If True, also delete all indexed documents
            
        Returns:
            True if index was dropped
        """
        if not self._check_search_available():
            logger.warning("Redis Search not available, cannot drop index")
            return False
        
        index_name = self._vector_config.name
        
        try:
            if delete_documents:
                self._r.ft(index_name).dropindex(delete_documents=True)
            else:
                self._r.ft(index_name).dropindex()
            
            self._index_created = False
            logger.info(f"Dropped vector index '{index_name}'")
            return True
        except redis.ResponseError as e:
            logger.warning(f"Failed to drop index: {e}")
            return False

    def count_documents(self) -> int:
        """Count total documents stored."""
        pattern = f"{self._prefix}:{self._doc_ns}:*"
        count = 0
        cursor = 0
        
        while True:
            cursor, keys = self._r.scan(cursor=cursor, match=pattern, count=1000)
            count += len(keys)
            if cursor == 0:
                break
        
        return count
