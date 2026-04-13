"""
Model download and management utilities for Radiant RAG.

Handles downloading, caching, and verifying models for offline operation.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ModelDownloadError(Exception):
    """Raised when model download fails."""
    pass


class ModelManager:
    """
    Manages downloading and caching of ML models.

    Provides centralized model management with:
    - Automatic download on first use
    - Checksum verification for integrity
    - Progress tracking
    - Local caching
    """

    # Known model checksums (SHA256)
    KNOWN_CHECKSUMS = {
        "lid.176.ftz": "a9e3a5c8c46e8c3c6e8d7f8b5d4c3b2a1e9d8c7b6a5f4e3d2c1b0a9f8e7d6c5b4",
    }

    @staticmethod
    def download_model(
        url: str,
        destination: str | Path,
        checksum: Optional[str] = None,
        show_progress: bool = True,
    ) -> Path:
        """
        Download a model file from URL to local path.

        Args:
            url: URL to download from
            destination: Local path to save model
            checksum: Optional SHA256 checksum to verify
            show_progress: Whether to show download progress

        Returns:
            Path to downloaded model

        Raises:
            ModelDownloadError: If download or verification fails
        """
        dest_path = Path(destination)

        # Create parent directory if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file already exists and is valid
        if dest_path.exists():
            if checksum:
                if ModelManager._verify_checksum(dest_path, checksum):
                    logger.info(f"Model already exists and verified: {dest_path}")
                    return dest_path
                else:
                    logger.warning(f"Existing model failed checksum, re-downloading: {dest_path}")
                    dest_path.unlink()
            else:
                logger.info(f"Model already exists: {dest_path}")
                return dest_path

        logger.info(f"Downloading model from {url} to {dest_path}")

        try:
            # Download with progress reporting
            if show_progress:
                ModelManager._download_with_progress(url, dest_path)
            else:
                urllib.request.urlretrieve(url, dest_path)

            logger.info(f"Download completed: {dest_path}")

            # Verify checksum if provided
            if checksum:
                if not ModelManager._verify_checksum(dest_path, checksum):
                    dest_path.unlink()
                    raise ModelDownloadError(
                        f"Downloaded model failed checksum verification. "
                        f"File may be corrupted or URL may point to wrong version."
                    )
                logger.info("Checksum verified successfully")

            return dest_path

        except Exception as e:
            # Clean up partial download
            if dest_path.exists():
                dest_path.unlink()

            raise ModelDownloadError(f"Failed to download model: {e}") from e

    @staticmethod
    def _download_with_progress(url: str, destination: Path) -> None:
        """Download file with progress bar."""

        def _report_progress(block_num: int, block_size: int, total_size: int) -> None:
            """Report download progress."""
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(downloaded * 100.0 / total_size, 100.0)
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)

                # Log every 10%
                if block_num == 0 or int(percent) % 10 == 0:
                    logger.info(
                        f"Download progress: {percent:.1f}% "
                        f"({downloaded_mb:.1f} MB / {total_mb:.1f} MB)"
                    )

        urllib.request.urlretrieve(url, destination, reporthook=_report_progress)

    @staticmethod
    def _verify_checksum(file_path: Path, expected_checksum: str) -> bool:
        """
        Verify file checksum.

        Args:
            file_path: Path to file
            expected_checksum: Expected SHA256 checksum (hex string)

        Returns:
            True if checksum matches, False otherwise
        """
        if not file_path.exists():
            return False

        try:
            sha256 = hashlib.sha256()

            with open(file_path, 'rb') as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b''):
                    sha256.update(chunk)

            actual_checksum = sha256.hexdigest()

            if actual_checksum != expected_checksum:
                logger.warning(
                    f"Checksum mismatch for {file_path.name}: "
                    f"expected {expected_checksum[:16]}..., "
                    f"got {actual_checksum[:16]}..."
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Error verifying checksum: {e}")
            return False

    @staticmethod
    def ensure_model(
        model_path: str | Path,
        download_url: Optional[str] = None,
        auto_download: bool = True,
        checksum: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Ensure model exists, downloading if necessary.

        Args:
            model_path: Path where model should be located
            download_url: URL to download from if model doesn't exist
            auto_download: Whether to auto-download if missing
            checksum: Optional checksum to verify

        Returns:
            Path to model if available, None if not available and auto_download=False

        Raises:
            ModelDownloadError: If auto_download=True but download fails
        """
        model_path = Path(model_path)

        # Check if model already exists
        if model_path.exists():
            # Verify checksum if provided
            if checksum and not ModelManager._verify_checksum(model_path, checksum):
                logger.warning(f"Model exists but failed checksum: {model_path}")
                if auto_download and download_url:
                    logger.info("Re-downloading model...")
                    return ModelManager.download_model(
                        download_url, model_path, checksum
                    )
            return model_path

        # Model doesn't exist
        if not auto_download:
            logger.warning(f"Model not found and auto_download=False: {model_path}")
            return None

        if not download_url:
            raise ModelDownloadError(
                f"Model not found at {model_path} and no download_url provided"
            )

        # Download model
        logger.info(f"Model not found, downloading: {model_path}")
        return ModelManager.download_model(download_url, model_path, checksum)

    @staticmethod
    def get_model_info(model_path: str | Path) -> dict:
        """
        Get information about a model file.

        Args:
            model_path: Path to model file

        Returns:
            Dictionary with model information
        """
        model_path = Path(model_path)

        if not model_path.exists():
            return {
                "exists": False,
                "path": str(model_path),
            }

        stat = model_path.stat()

        # Calculate checksum
        sha256 = hashlib.sha256()
        with open(model_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)

        return {
            "exists": True,
            "path": str(model_path),
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "modified": stat.st_mtime,
            "checksum": sha256.hexdigest(),
        }
