"""
LLM backend implementations.

Supports:
- OpenAI-compatible API (ollama, vllm, openai)
- Local HuggingFace Transformers models
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import openai as _openai
from radiant_rag_mcp.llm.client import ChatMessage  # simple dataclass, no haystack

from radiant_rag_mcp.llm.backends.base import BaseLLMBackend, LLMResponse

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class OpenAICompatibleLLMBackend(BaseLLMBackend):
    """
    LLM backend using OpenAI-compatible API.

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
        timeout: int = 90,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        """
        Initialize OpenAI-compatible LLM backend.

        Reads RADIANT_OLLAMA_* environment variables at startup; they override
        the constructor arguments when present.

        Args:
            base_url: API base URL (e.g., https://ollama.com/v1)
            api_key: API key for authentication (fallback if env var not set)
            model: Model name to use
            timeout: Request timeout in seconds
            max_retries: Number of retries on failure
            **kwargs: Additional arguments
        """
        # ENV VAR WIRING — read at startup, override constructor args
        env_base_url = os.environ.get("RADIANT_OLLAMA_BASE_URL")
        self._base_url = env_base_url if env_base_url else base_url

        # Always read api_key from env; default to constructor arg if env var absent.
        # Authorization: Bearer <key> is always sent even when key is empty string.
        self._api_key = os.environ.get("RADIANT_OLLAMA_API_KEY", api_key)

        env_timeout = os.environ.get("RADIANT_OLLAMA_TIMEOUT")
        self._timeout = int(env_timeout) if env_timeout is not None else timeout

        env_max_retries = os.environ.get("RADIANT_OLLAMA_MAX_RETRIES")
        self._max_retries = int(env_max_retries) if env_max_retries is not None else max_retries

        self._model = model

        # LAZY INITIALIZATION — generator is created on first use, not here
        self._generator: Optional["_openai.OpenAI"] = None

        logger.info(
            f"Configured OpenAI-compatible LLM backend: model={model}, base_url={self._base_url}"
        )

    def _get_generator(self) -> "_openai.OpenAI":
        """Return the openai client, creating it on first call (lazy init)."""
        if self._generator is None:
            self._generator = _openai.OpenAI(
                api_key=self._api_key or "ollama",
                base_url=self._base_url,
                timeout=self._timeout,
            )
            logger.info(
                f"Initialized OpenAI-compatible LLM backend: model={self._model}, base_url={self._base_url}"
            )
        return self._generator

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Send chat request to LLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional arguments

        Returns:
            LLMResponse with generated content and metadata
        """
        start_time = time.time()

        try:
            result = self._get_generator().chat.completions.create(
                model=self._model,
                messages=messages,
            )
            latency = (time.time() - start_time) * 1000
            content = result.choices[0].message.content or ""

            return LLMResponse(
                content=content,
                meta={
                    "model": self._model,
                    "latency_ms": latency,
                },
            )

        except Exception as e:
            logger.error(f"OpenAI-compatible LLM request failed: {e}")
            raise

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The input prompt
            **kwargs: Additional arguments

        Returns:
            Generated text string
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.chat(messages, **kwargs)
        return response.content


class LocalHuggingFaceLLMBackend(BaseLLMBackend):
    """
    LLM backend using local HuggingFace Transformers models.

    Loads and runs models locally using HuggingFace transformers library.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> None:
        """
        Initialize local HuggingFace LLM backend.

        Args:
            model_name: HuggingFace model name (e.g., "meta-llama/Llama-2-7b-chat-hf")
            device: Device to load model on ("auto", "cuda", "cpu")
            load_in_4bit: Load model in 4-bit quantization (requires bitsandbytes)
            load_in_8bit: Load model in 8-bit quantization (requires bitsandbytes)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments
        """
        self._model_name = model_name
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

        logger.info(f"Initializing local HuggingFace LLM: model={model_name}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as e:
            raise ImportError(
                "HuggingFace transformers and torch are required for local LLM backend. "
                "Install with: pip install transformers torch"
            ) from e

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Prepare model loading kwargs
        model_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
        }

        # Add quantization if requested
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
                logger.info("Loading model with 4-bit quantization")
            except ImportError:
                logger.warning("bitsandbytes not available, loading without quantization")
        elif load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
            logger.info("Loading model with 8-bit quantization")
        elif device == "auto":
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = device

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

        logger.info(f"Loaded local HuggingFace LLM: {model_name}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Send chat request to local LLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional arguments (max_new_tokens, temperature, etc.)

        Returns:
            LLMResponse with generated content and metadata
        """
        start_time = time.time()

        # Format messages into a prompt
        # Try to use tokenizer's chat template if available
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}, using fallback")
                prompt = self._format_messages_fallback(messages)
        else:
            prompt = self._format_messages_fallback(messages)

        # Generate
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            max_new_tokens = kwargs.get("max_new_tokens", self._max_new_tokens)
            temperature = kwargs.get("temperature", self._temperature)

            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

            # Decode only the new tokens
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            content = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)

            latency = (time.time() - start_time) * 1000

            return LLMResponse(
                content=content,
                meta={
                    "model": self._model_name,
                    "latency_ms": latency,
                    "generated_tokens": len(generated_tokens),
                },
            )

        except Exception as e:
            logger.error(f"Local HuggingFace LLM generation failed: {e}")
            raise

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The input prompt
            **kwargs: Additional arguments

        Returns:
            Generated text string
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.chat(messages, **kwargs)
        return response.content

    def _format_messages_fallback(self, messages: List[Dict[str, str]]) -> str:
        """Format messages when no chat template is available."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
            else:
                formatted.append(f"User: {content}")

        formatted.append("Assistant:")
        return "\n\n".join(formatted)
