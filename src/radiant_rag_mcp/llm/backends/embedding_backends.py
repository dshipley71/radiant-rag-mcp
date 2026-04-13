"""
Embedding backend implementations.

Supports:
- Local sentence-transformers models
- OpenAI-compatible API (ollama, vllm, openai)
- HuggingFace Transformers models
"""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from typing import List, TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from radiant_rag_mcp.llm.backends.base import BaseEmbeddingBackend

if TYPE_CHECKING:
    pass

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
    Suppress noisy warnings during sentence-transformer model loading.

    Temporarily raises log levels for chatty libraries and filters warnings
    like "The following layers were not sharded" and UNEXPECTED key reports.
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


class SentenceTransformersEmbeddingBackend(BaseEmbeddingBackend):
    """
    Embedding backend using sentence-transformers.

    This is the default backend for local embeddings.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        cache_size: int = 10000,
        **kwargs,
    ) -> None:
        """
        Initialize sentence-transformers embedding backend.

        Args:
            model_name: Model name (e.g., "sentence-transformers/all-MiniLM-L12-v2")
            device: Device to load model on ("auto", "cuda", "cpu")
            cache_size: Size of embedding cache
            **kwargs: Additional arguments
        """
        self._model_name = model_name
        self._cache_size = cache_size

        logger.info(f"Initializing sentence-transformers embedding backend: {model_name}")

        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for local embedding backend. "
                "Install with: pip install sentence-transformers"
            ) from e

        # Resolve device
        device_resolved = self._resolve_device(device)

        # Load model with suppressed warnings
        with _quiet_model_loading():
            self._model = SentenceTransformer(model_name, device=device_resolved)
        self._embedding_dim = self._model.get_sentence_embedding_dimension()

        # Initialize cache
        from radiant_rag_mcp.utils.cache import get_embedding_cache
        self._cache = get_embedding_cache(max_size=cache_size)

        logger.info(
            f"Loaded sentence-transformers embedding model: {model_name} "
            f"(dim={self._embedding_dim}, device={device_resolved})"
        )

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
            List of embedding vectors
        """
        if not texts:
            return []

        # Check cache for already-computed embeddings
        embeddings = []
        texts_to_compute = []
        indices_to_compute = []

        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                texts_to_compute.append(text)
                indices_to_compute.append(i)

        # Compute embeddings for cache misses
        if texts_to_compute:
            computed = self._model.encode(
                texts_to_compute,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                batch_size=batch_size,
                convert_to_numpy=True,
            )

            # Store in cache and update results
            for idx, text, emb in zip(indices_to_compute, texts_to_compute, computed):
                emb_list = emb.tolist()
                self._cache.put(text, emb_list)
                embeddings[idx] = emb_list

        return embeddings

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
        """
        # Check cache
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        # Compute embedding
        embedding = self._model.encode(
            text,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )

        emb_list = embedding.tolist()
        self._cache.put(text, emb_list)
        return emb_list

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self._embedding_dim

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


class OpenAICompatibleEmbeddingBackend(BaseEmbeddingBackend):
    """
    Embedding backend using OpenAI-compatible API.

    Works with:
    - Ollama
    - vLLM
    - OpenAI
    - Any other OpenAI-compatible API endpoint
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        embedding_dimension: int = 384,
        **kwargs,
    ) -> None:
        """
        Initialize OpenAI-compatible embedding backend.

        Args:
            base_url: API base URL
            api_key: API key for authentication
            model: Model name to use
            embedding_dimension: Expected embedding dimension
            **kwargs: Additional arguments
        """
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._embedding_dim = embedding_dimension

        logger.info(
            f"Initialized OpenAI-compatible embedding backend: model={model}, "
            f"base_url={base_url}, dim={embedding_dimension}"
        )

        try:
            import openai
            self._client = openai.OpenAI(
                base_url=base_url,
                api_key=api_key,
            )
        except ImportError as e:
            raise ImportError(
                "openai package is required for OpenAI-compatible embedding backend. "
                "Install with: pip install openai"
            ) from e

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
            List of embedding vectors
        """
        if not texts:
            return []

        # Process in batches
        all_embeddings = []
        iterator = range(0, len(texts), batch_size)

        if show_progress:
            iterator = tqdm(
                iterator,
                desc="Embedding batches",
                unit="batch",
            )

        for i in iterator:
            batch = texts[i:i + batch_size]

            try:
                response = self._client.embeddings.create(
                    input=batch,
                    model=self._model,
                )

                batch_embeddings = [item.embedding for item in response.data]

                if normalize:
                    batch_embeddings = [
                        self._normalize(emb) for emb in batch_embeddings
                    ]

                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(f"Embedding API request failed for batch {i//batch_size}: {e}")
                raise

        return all_embeddings

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
        """
        try:
            response = self._client.embeddings.create(
                input=[text],
                model=self._model,
            )

            embedding = response.data[0].embedding

            if normalize:
                embedding = self._normalize(embedding)

            return embedding

        except Exception as e:
            logger.error(f"Embedding API request failed: {e}")
            raise

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self._embedding_dim

    @staticmethod
    def _normalize(embedding: List[float]) -> List[float]:
        """Normalize embedding to unit length."""
        arr = np.array(embedding)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()


class HuggingFaceTransformersEmbeddingBackend(BaseEmbeddingBackend):
    """
    Embedding backend using HuggingFace Transformers directly.

    For models not available in sentence-transformers.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        pooling_strategy: str = "mean",
        **kwargs,
    ) -> None:
        """
        Initialize HuggingFace Transformers embedding backend.

        Args:
            model_name: HuggingFace model name
            device: Device to load model on
            pooling_strategy: Pooling strategy ("mean", "cls", "max")
            **kwargs: Additional arguments
        """
        self._model_name = model_name
        self._pooling_strategy = pooling_strategy

        logger.info(f"Initializing HuggingFace Transformers embedding backend: {model_name}")

        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for HuggingFace embedding backend. "
                "Install with: pip install transformers torch"
            ) from e

        # Resolve device
        device_resolved = self._resolve_device(device)

        # Load model and tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(device_resolved)
        self._device = device_resolved

        # Get embedding dimension from model config
        self._embedding_dim = self._model.config.hidden_size

        logger.info(
            f"Loaded HuggingFace Transformers embedding model: {model_name} "
            f"(dim={self._embedding_dim}, device={device_resolved})"
        )

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
            List of embedding vectors
        """
        if not texts:
            return []

        import torch

        all_embeddings = []
        iterator = range(0, len(texts), batch_size)

        if show_progress:
            iterator = tqdm(
                iterator,
                desc="Embedding batches",
                unit="batch",
            )

        self._model.eval()
        with torch.no_grad():
            for i in iterator:
                batch = texts[i:i + batch_size]

                # Tokenize
                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                # Forward pass
                outputs = self._model(**inputs)

                # Pool
                embeddings = self._pool(outputs, inputs["attention_mask"])

                # Normalize if requested
                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.extend(embeddings.cpu().numpy().tolist())

        return all_embeddings

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
        """
        embeddings = self.embed([text], normalize=normalize, show_progress=False, batch_size=1)
        return embeddings[0]

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self._embedding_dim

    def _pool(self, outputs, attention_mask):
        """Pool token embeddings based on strategy."""
        import torch

        last_hidden_state = outputs.last_hidden_state

        if self._pooling_strategy == "cls":
            # Use [CLS] token (first token)
            return last_hidden_state[:, 0, :]

        elif self._pooling_strategy == "max":
            # Max pooling
            return torch.max(last_hidden_state, dim=1)[0]

        else:  # mean (default)
            # Mean pooling with attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask

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
