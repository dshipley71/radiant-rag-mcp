"""
Backend implementations for LLM, embedding, and reranking models.

Supports multiple backends:
- OpenAI-compatible API (ollama, vllm, openai)
- Local HuggingFace models (transformers, sentence-transformers)
"""

from radiant_rag_mcp.llm.backends.base import (
    BaseLLMBackend,
    BaseEmbeddingBackend,
    BaseRerankingBackend,
    LLMResponse,
)
from radiant_rag_mcp.llm.backends.factory import (
    create_llm_backend,
    create_embedding_backend,
    create_reranking_backend,
)

__all__ = [
    "BaseLLMBackend",
    "BaseEmbeddingBackend",
    "BaseRerankingBackend",
    "LLMResponse",
    "create_llm_backend",
    "create_embedding_backend",
    "create_reranking_backend",
]
