"""
Binary quantization utilities for vector embeddings.

Provides functions for quantizing float32 embeddings to binary and int8 formats
to reduce memory usage and improve retrieval speed while maintaining accuracy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Check for sentence-transformers quantization support
QUANTIZATION_AVAILABLE = False
_QUANTIZATION_IMPORT_ERROR = None
try:
    from sentence_transformers.quantization import quantize_embeddings as st_quantize_embeddings
    QUANTIZATION_AVAILABLE = True
except ImportError as e:
    _QUANTIZATION_IMPORT_ERROR = str(e)
    st_quantize_embeddings = None

# Log import status
if _QUANTIZATION_IMPORT_ERROR:
    logger.debug(f"sentence-transformers quantization not available: {_QUANTIZATION_IMPORT_ERROR}")


@dataclass
class QuantizationConfig:
    """Configuration for embedding quantization."""
    
    # Whether quantization is enabled
    enabled: bool = False
    
    # Precision for corpus embeddings: "binary", "int8", or "both"
    precision: str = "both"
    
    # Rescoring multiplier: retrieve N×top_k candidates for rescoring
    rescore_multiplier: float = 4.0
    
    # Whether to use rescoring step
    use_rescoring: bool = True
    
    # Calibration ranges for int8 quantization [2, embedding_dim]
    # First row: minimum values per dimension
    # Second row: maximum values per dimension
    int8_ranges: Optional[np.ndarray] = None
    
    # Whether to keep int8 embeddings only on disk (saves memory)
    int8_on_disk_only: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.enabled and not QUANTIZATION_AVAILABLE:
            logger.warning(
                "Quantization enabled but sentence-transformers quantization module not available. "
                "Install with: pip install sentence-transformers>=3.2.0"
            )
            # Disable quantization
            object.__setattr__(self, 'enabled', False)
        
        if self.precision not in ("binary", "int8", "both"):
            raise ValueError(f"Invalid precision '{self.precision}'. Must be 'binary', 'int8', or 'both'")
        
        if self.rescore_multiplier < 1.0:
            raise ValueError(f"rescore_multiplier must be >= 1.0, got {self.rescore_multiplier}")


def quantize_embeddings(
    embeddings: np.ndarray,
    precision: str = "binary",
    ranges: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Quantize float32 embeddings to binary or int8 format.
    
    Args:
        embeddings: Float32 embeddings array of shape [N, D]
        precision: Quantization precision - "binary", "ubinary", "int8", or "uint8"
        ranges: Optional calibration ranges for int8 quantization [2, D]
                First row: min values, second row: max values
    
    Returns:
        Quantized embeddings array
        
    Raises:
        ImportError: If sentence-transformers quantization not available
    """
    if not QUANTIZATION_AVAILABLE:
        raise ImportError(
            "sentence-transformers quantization module not available. "
            "Install with: pip install sentence-transformers>=3.2.0"
        )
    
    # Ensure input is numpy array
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings, dtype=np.float32)
    
    # Call sentence-transformers quantization
    if ranges is not None:
        return st_quantize_embeddings(embeddings, precision=precision, ranges=ranges)
    else:
        return st_quantize_embeddings(embeddings, precision=precision)


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """
    Convert embedding array to bytes for storage.
    
    Args:
        embedding: Numpy array (any dtype)
    
    Returns:
        Bytes representation
    """
    return embedding.tobytes()


def bytes_to_embedding(data: bytes, dtype: np.dtype, shape: tuple) -> np.ndarray:
    """
    Convert bytes back to embedding array.
    
    Args:
        data: Bytes data
        dtype: Numpy dtype (e.g., np.float32, np.uint8, np.int8)
        shape: Shape of the array
    
    Returns:
        Numpy array
    """
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def get_binary_dimension(float_dimension: int) -> int:
    """
    Calculate binary embedding dimension from float dimension.
    
    Binary embeddings are packed into bytes, so dimension is divided by 8.
    
    Args:
        float_dimension: Original float32 embedding dimension
    
    Returns:
        Binary embedding dimension (in bytes)
    """
    if float_dimension % 8 != 0:
        logger.warning(
            f"Float dimension {float_dimension} is not divisible by 8. "
            f"Binary embedding will be padded."
        )
    return (float_dimension + 7) // 8  # Ceiling division


def calculate_int8_ranges(embeddings: np.ndarray) -> np.ndarray:
    """
    Calculate calibration ranges for int8 quantization from sample embeddings.
    
    Args:
        embeddings: Sample embeddings array [N, D]
    
    Returns:
        Ranges array [2, D] with min/max per dimension
    """
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings, dtype=np.float32)
    
    ranges = np.vstack([
        np.min(embeddings, axis=0),
        np.max(embeddings, axis=0)
    ])
    
    logger.info(
        f"Calculated int8 ranges for {embeddings.shape[1]} dimensions from "
        f"{embeddings.shape[0]} samples"
    )
    
    return ranges


def rescore_candidates(
    query_embedding: np.ndarray,
    candidate_embeddings: List[np.ndarray],
    candidate_ids: List[str],
) -> List[tuple]:
    """
    Rescore candidates using higher precision embeddings.
    
    Args:
        query_embedding: Float32 query embedding
        candidate_embeddings: List of int8 or float32 candidate embeddings
        candidate_ids: List of candidate document IDs
    
    Returns:
        List of (doc_id, score) tuples sorted by score descending
    """
    if not candidate_embeddings:
        return []
    
    # Ensure query is float32
    if query_embedding.dtype != np.float32:
        query_embedding = query_embedding.astype(np.float32)
    
    # Calculate scores
    scores = []
    for emb, doc_id in zip(candidate_embeddings, candidate_ids):
        # Convert to float32 for dot product
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32)
        
        # Cosine similarity (assumes normalized embeddings)
        score = np.dot(query_embedding, emb)
        scores.append((doc_id, float(score)))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores
