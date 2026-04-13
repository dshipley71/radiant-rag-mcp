"""
Image captioning using Vision Language Models (VLM) for Radiant Agentic RAG.

Provides image description generation using local VLMs via HuggingFace Transformers
or optionally via Ollama/cloud APIs.

Supported HuggingFace models:
    - Qwen/Qwen2-VL-2B-Instruct (default, good balance)
    - Qwen/Qwen2-VL-7B-Instruct (better quality)
    - Qwen/Qwen2.5-VL-7B-Instruct (latest)
    - llava-hf/llava-1.5-7b-hf
    - microsoft/Phi-3-vision-128k-instruct
    - Any transformers-compatible VLM
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Try to import transformers and torch
TRANSFORMERS_AVAILABLE = False
TORCH_AVAILABLE = False
try:
    import torch
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    Image = None

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoProcessor = None
    AutoModelForVision2Seq = None
    BitsAndBytesConfig = None

# For Qwen2-VL specifically
QWEN_VL_AVAILABLE = False
try:
    from transformers import Qwen2VLForConditionalGeneration
    QWEN_VL_AVAILABLE = True
except ImportError:
    Qwen2VLForConditionalGeneration = None

# For Qwen3-VL (requires transformers >= 4.57.0)
QWEN3_VL_AVAILABLE = False
try:
    from transformers import Qwen3VLForConditionalGeneration
    QWEN3_VL_AVAILABLE = True
except ImportError:
    Qwen3VLForConditionalGeneration = None

# Qwen VL utils for proper image processing (required for Qwen3-VL)
QWEN_VL_UTILS_AVAILABLE = False
try:
    from qwen_vl_utils import process_vision_info
    QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    process_vision_info = None


@dataclass
class VLMConfig:
    """Configuration for Vision Language Model."""

    # HuggingFace model name
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"

    # Device settings
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"

    # Quantization for memory efficiency
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.2
    do_sample: bool = False

    # Whether VLM captioning is enabled
    enabled: bool = True

    # Prompt for image captioning
    caption_prompt: str = (
        "Describe this image in detail. Include: "
        "1) Main subjects and objects, "
        "2) Actions or activities shown, "
        "3) Setting/location, "
        "4) Any text visible in the image, "
        "5) Notable colors, style, or composition. "
        "Be factual and descriptive."
    )

    # Cache directory for model downloads
    cache_dir: Optional[str] = None

    # Trust remote code (needed for some models like Qwen2-VL)
    trust_remote_code: bool = True


# Image extensions for captioning
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}


def _is_image_file(file_path: str) -> bool:
    """Check if file is an image based on extension."""
    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS


def _get_device(device: str) -> str:
    """Determine the best available device."""
    if device != "auto":
        return device

    if not TORCH_AVAILABLE:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def _is_qwen_vl_model(model_name: str) -> bool:
    """Check if model is a Qwen VL model."""
    name_lower = model_name.lower()
    return "qwen" in name_lower and ("vl" in name_lower or "vision" in name_lower)


def _is_qwen3_vl_model(model_name: str) -> bool:
    """Check if model is specifically a Qwen3 VL model."""
    name_lower = model_name.lower()
    return "qwen3" in name_lower and ("vl" in name_lower or "vision" in name_lower)


class HuggingFaceVLMCaptioner:
    """
    Generate captions for images using HuggingFace Transformers VLM models.

    Supports various vision-language models including Qwen2-VL, LLaVA, etc.
    """

    def __init__(self, config: VLMConfig) -> None:
        """
        Initialize the HuggingFace VLM captioner.

        Args:
            config: VLM configuration
        """
        self._config = config
        self._model = None
        self._processor = None
        self._device = None
        self._loaded = False
        self._available: Optional[bool] = None
        self._is_qwen_vl = _is_qwen_vl_model(config.model_name)
        self._is_qwen3_vl = _is_qwen3_vl_model(config.model_name)

    def _load_model(self) -> bool:
        """
        Lazy-load the model and processor.

        Returns:
            True if loading succeeded
        """
        if self._loaded:
            return self._model is not None

        self._loaded = True

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Install with: pip install torch")
            return False

        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available. Install with: pip install transformers")
            return False

        model_name = self._config.model_name
        self._device = _get_device(self._config.device)

        logger.info(f"Loading VLM model: {model_name} on {self._device}")

        # Check for qwen_vl_utils if using Qwen3 model
        if self._is_qwen3_vl and not QWEN_VL_UTILS_AVAILABLE:
            logger.error(
                "qwen_vl_utils required for Qwen3 VL models. "
                "Install with: pip install qwen-vl-utils"
            )
            return False

        # Check for correct model class
        if self._is_qwen3_vl and not QWEN3_VL_AVAILABLE:
            logger.error(
                "Qwen3VLForConditionalGeneration not available. "
                "Upgrade transformers: pip install transformers>=4.57.0"
            )
            return False

        try:
            # Prepare quantization config if requested
            quantization_config = None
            if self._config.load_in_4bit and BitsAndBytesConfig is not None:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            elif self._config.load_in_8bit and BitsAndBytesConfig is not None:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=self._config.trust_remote_code,
                cache_dir=self._config.cache_dir,
            )

            # Prepare model kwargs
            model_kwargs = {
                "trust_remote_code": self._config.trust_remote_code,
                "cache_dir": self._config.cache_dir,
            }

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            elif self._device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float32

            # Load model - use appropriate class based on model type
            if self._is_qwen3_vl and QWEN3_VL_AVAILABLE:
                self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_name,
                    **model_kwargs
                )
            elif self._is_qwen_vl and QWEN_VL_AVAILABLE:
                self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    **model_kwargs
                )
            else:
                self._model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    **model_kwargs
                )

            # Move to device if not using device_map
            if not quantization_config and self._device != "cuda":
                self._model = self._model.to(self._device)

            self._model.eval()

            logger.info(f"VLM model loaded successfully: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load VLM model {model_name}: {e}")
            self._model = None
            self._processor = None
            return False

    def is_available(self) -> bool:
        """
        Check if VLM is available.

        Returns:
            True if VLM can be used
        """
        if self._available is not None:
            return self._available

        if not self._config.enabled:
            self._available = False
            return False

        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            self._available = False
            return False

        # Don't actually load model yet, just check if it's possible
        self._available = True
        return True

    def _caption_with_qwen_vl(self, image_path: str, prompt: str) -> Optional[str]:
        """
        Generate caption using Qwen VL model with proper processing.

        Args:
            image_path: Path to image
            prompt: Caption prompt

        Returns:
            Caption string or None
        """
        # Build messages in Qwen VL format
        # For Qwen3, we need to use file:// URI or the image path directly
        if self._is_qwen3_vl:
            # Qwen3-VL requires qwen_vl_utils.process_vision_info
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Apply chat template
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Use process_vision_info to properly extract and process images
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            # Qwen2-VL can work with direct image loading
            image = Image.open(image_path).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Apply chat template
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process inputs - pass image directly to processor
            inputs = self._processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )

        # Move to device
        inputs = inputs.to(self._model.device)

        # Generate
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._config.max_new_tokens,
            )

        # Decode - skip input tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        caption = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return caption.strip() if caption else None

    def _caption_generic(self, image_path: str, prompt: str) -> Optional[str]:
        """
        Generate caption using generic VLM approach.

        Args:
            image_path: Path to image
            prompt: Caption prompt

        Returns:
            Caption string or None
        """
        image = Image.open(image_path).convert("RGB")

        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self._model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._config.max_new_tokens,
                temperature=self._config.temperature if self._config.do_sample else None,
                do_sample=self._config.do_sample,
            )

        caption = self._processor.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

        return caption.strip() if caption else None

    def caption_image(
        self,
        image_path: str,
        prompt: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate a caption for an image.

        Args:
            image_path: Path to the image file
            prompt: Optional custom prompt (uses default if not provided)

        Returns:
            Caption string, or None if captioning failed
        """
        if not self.is_available():
            return None

        # Lazy load model
        if not self._load_model():
            return None

        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None

        if not _is_image_file(image_path):
            logger.warning(f"Not an image file: {image_path}")
            return None

        prompt = prompt or self._config.caption_prompt

        try:
            if self._is_qwen_vl:
                caption = self._caption_with_qwen_vl(image_path, prompt)
            else:
                caption = self._caption_generic(image_path, prompt)

            if not caption:
                logger.warning(f"Empty caption generated for {image_path}")
                return None

            logger.debug(f"Generated caption for {image_path}: {caption[:100]}...")
            return caption

        except Exception as e:
            logger.error(f"Error captioning image {image_path}: {e}")
            return None

    def caption_images_batch(
        self,
        image_paths: List[str],
        prompt: Optional[str] = None,
    ) -> List[Tuple[str, Optional[str]]]:
        """
        Generate captions for multiple images.

        Args:
            image_paths: List of paths to image files
            prompt: Optional custom prompt

        Returns:
            List of (path, caption) tuples
        """
        results = []
        for path in image_paths:
            caption = self.caption_image(path, prompt)
            results.append((path, caption))
        return results


class OllamaVLMCaptioner:
    """
    Image captioner using Ollama's local API.

    Fallback option if HuggingFace models are not available.
    """

    def __init__(self, config: VLMConfig, ollama_url: str = "http://localhost:11434", model: str = "llava") -> None:
        """
        Initialize Ollama-based captioner.

        Args:
            config: VLM configuration
            ollama_url: Ollama API URL
            model: Ollama model name
        """
        self._config = config
        self._ollama_url = ollama_url.rstrip("/")
        self._model = model
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if Ollama VLM is available."""
        if self._available is not None:
            return self._available

        if not self._config.enabled:
            self._available = False
            return False

        try:
            import requests
            response = requests.get(f"{self._ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                self._available = False
                return False

            # Check if model exists
            tags = response.json()
            models = [m.get("name", "") for m in tags.get("models", [])]
            model_base = self._model.split(":")[0]

            self._available = any(model_base in m or self._model in m for m in models)
            if self._available:
                logger.info(f"Ollama VLM available: {self._model}")
            return self._available

        except Exception:
            self._available = False
            return False

    def caption_image(self, image_path: str, prompt: Optional[str] = None) -> Optional[str]:
        """Generate caption using Ollama."""
        if not self.is_available():
            return None

        if not os.path.exists(image_path):
            return None

        prompt = prompt or self._config.caption_prompt

        try:
            import requests

            # Encode image
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")

            response = requests.post(
                f"{self._ollama_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {
                        "temperature": self._config.temperature,
                        "num_predict": self._config.max_new_tokens,
                    }
                },
                timeout=120
            )

            if response.status_code != 200:
                return None

            result = response.json()
            caption = result.get("response", "").strip()
            return caption if caption else None

        except Exception as e:
            logger.error(f"Ollama VLM error: {e}")
            return None


# Type alias for captioner
ImageCaptioner = Union[HuggingFaceVLMCaptioner, OllamaVLMCaptioner]


def create_captioner(
    vlm_config: Optional[VLMConfig] = None,
    ollama_url: Optional[str] = None,
    ollama_model: Optional[str] = None,
) -> Optional[ImageCaptioner]:
    """
    Factory function to create appropriate image captioner.

    Tries HuggingFace Transformers first, falls back to Ollama if available.

    Args:
        vlm_config: VLM configuration
        ollama_url: Ollama URL (optional, for fallback)
        ollama_model: Ollama model name (optional, for fallback)

    Returns:
        ImageCaptioner instance, or None if no captioner available
    """
    vlm_config = vlm_config or VLMConfig()

    if not vlm_config.enabled:
        logger.info("VLM captioning disabled in configuration")
        return None

    # Try HuggingFace Transformers first
    if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
        hf_captioner = HuggingFaceVLMCaptioner(vlm_config)
        if hf_captioner.is_available():
            logger.info(f"Using HuggingFace VLM: {vlm_config.model_name}")
            return hf_captioner

    # Try Ollama as fallback
    if ollama_url and ollama_model:
        ollama_captioner = OllamaVLMCaptioner(vlm_config, ollama_url, ollama_model)
        if ollama_captioner.is_available():
            logger.info(f"Using Ollama VLM: {ollama_model}")
            return ollama_captioner

    logger.warning(
        "No VLM captioner available. Image captioning disabled. "
        "Install: pip install 'transformers>=4.57.0' torch pillow qwen-vl-utils"
    )
    return None
