"""
JSON and JSONL document parsing for Radiant RAG.

Provides flexible parsing strategies to convert structured JSON data
into searchable text suitable for RAG indexing:
- Flattened key-value (for configs, settings)
- Record-based (for data arrays)
- Semantic text extraction (for rich content)
- Log entry parsing (for JSONL logs)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class JSONParsingStrategy(Enum):
    """JSON parsing strategies."""
    AUTO = "auto"           # Auto-detect best strategy
    FLATTEN = "flatten"     # Convert to flat key-value text
    RECORDS = "records"     # Split arrays into separate documents
    SEMANTIC = "semantic"   # Extract text-rich fields
    LOGS = "logs"          # Parse as structured logs


@dataclass
class JSONParsingConfig:
    """Configuration for JSON parsing."""

    default_strategy: JSONParsingStrategy = JSONParsingStrategy.AUTO
    min_array_size_for_splitting: int = 3
    text_fields: List[str] = None
    title_fields: List[str] = None
    max_nesting_depth: int = 10
    flatten_separator: str = "."
    jsonl_batch_size: int = 1000
    preserve_fields: List[str] = None

    def __post_init__(self):
        if self.text_fields is None:
            self.text_fields = [
                "content", "body", "text", "description",
                "message", "summary", "details", "value"
            ]
        if self.title_fields is None:
            self.title_fields = [
                "title", "name", "subject", "heading", "label", "key"
            ]
        if self.preserve_fields is None:
            self.preserve_fields = [
                "id", "timestamp", "date", "created_at", "updated_at",
                "type", "category", "level", "status"
            ]


@dataclass
class ParsedJSONChunk:
    """A chunk parsed from JSON data."""

    content: str
    meta: Dict[str, Any]
    title: Optional[str] = None

    def to_ingested_chunk_dict(self) -> Dict[str, Any]:
        """Convert to IngestedChunk-compatible dict."""
        return {
            "content": self.content,
            "meta": self.meta,
        }


class JSONParser:
    """
    Flexible JSON parser with multiple parsing strategies.

    Converts structured JSON into RAG-friendly text documents.
    """

    def __init__(self, config: Optional[JSONParsingConfig] = None):
        """
        Initialize JSON parser.

        Args:
            config: Parsing configuration
        """
        self._config = config or JSONParsingConfig()
        self._text_fields_set = set(self._config.text_fields)
        self._title_fields_set = set(self._config.title_fields)
        self._preserve_fields_set = set(self._config.preserve_fields)

    def parse_json_file(
        self,
        file_path: str,
        strategy: Optional[JSONParsingStrategy] = None,
    ) -> List[ParsedJSONChunk]:
        """
        Parse a JSON file into chunks.

        Args:
            file_path: Path to JSON file
            strategy: Parsing strategy (None = auto-detect)

        Returns:
            List of ParsedJSONChunk objects
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return []

        # Auto-detect strategy if not specified
        if strategy is None or strategy == JSONParsingStrategy.AUTO:
            strategy = self._detect_strategy(data)

        logger.debug(f"Parsing {file_path} with strategy: {strategy.value}")

        # Route to appropriate parser
        if strategy == JSONParsingStrategy.FLATTEN:
            chunks = self._parse_flatten(data, file_path)
        elif strategy == JSONParsingStrategy.RECORDS:
            chunks = self._parse_records(data, file_path)
        elif strategy == JSONParsingStrategy.SEMANTIC:
            chunks = self._parse_semantic(data, file_path)
        elif strategy == JSONParsingStrategy.LOGS:
            chunks = self._parse_logs(data, file_path)
        else:
            # Fallback to flatten
            chunks = self._parse_flatten(data, file_path)

        logger.info(f"Parsed {file_path}: {len(chunks)} chunks (strategy={strategy.value})")
        return chunks

    def parse_jsonl_file(
        self,
        file_path: str,
        strategy: Optional[JSONParsingStrategy] = None,
    ) -> List[ParsedJSONChunk]:
        """
        Parse a JSONL (JSON Lines) file.

        Each line is a separate JSON object.

        Args:
            file_path: Path to JSONL file
            strategy: Parsing strategy (None = auto-detect from first line)

        Returns:
            List of ParsedJSONChunk objects
        """
        chunks = []
        line_count = 0
        error_count = 0

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read first line to detect strategy
                first_line = f.readline().strip()
                if not first_line:
                    logger.warning(f"Empty JSONL file: {file_path}")
                    return []

                try:
                    first_obj = json.loads(first_line)
                    if strategy is None or strategy == JSONParsingStrategy.AUTO:
                        strategy = self._detect_strategy(first_obj)

                    # Process first line
                    chunk = self._parse_single_json_object(
                        first_obj, file_path, 0, strategy
                    )
                    if chunk:
                        chunks.append(chunk)
                    line_count = 1
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON on line 1 of {file_path}: {e}")
                    error_count += 1

                # Process remaining lines
                for line_num, line in enumerate(f, start=2):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                        chunk = self._parse_single_json_object(
                            obj, file_path, line_num - 1, strategy
                        )
                        if chunk:
                            chunks.append(chunk)
                        line_count += 1
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num} of {file_path}: {e}")
                        error_count += 1
                        if error_count > 10:
                            logger.error(f"Too many errors in {file_path}, stopping")
                            break

        except Exception as e:
            logger.error(f"Failed to read JSONL file {file_path}: {e}")
            return []

        logger.info(
            f"Parsed JSONL {file_path}: {len(chunks)} chunks from {line_count} lines "
            f"(strategy={strategy.value}, errors={error_count})"
        )
        return chunks

    def _detect_strategy(self, data: Any) -> JSONParsingStrategy:
        """
        Auto-detect best parsing strategy for JSON data.

        Args:
            data: Parsed JSON data

        Returns:
            Recommended parsing strategy
        """
        # Check if it's a log entry (has timestamp + level/message)
        if isinstance(data, dict):
            keys_lower = {k.lower() for k in data.keys()}
            has_timestamp = any(
                field in keys_lower
                for field in ['timestamp', 'time', 'date', 'datetime', 'created_at']
            )
            has_log_fields = any(
                field in keys_lower
                for field in ['level', 'severity', 'message', 'msg', 'log']
            )

            if has_timestamp and has_log_fields:
                return JSONParsingStrategy.LOGS

            # Check if it has rich text content
            has_text_content = any(
                k.lower() in self._text_fields_set and isinstance(v, str) and len(v) > 100
                for k, v in data.items()
            )
            if has_text_content:
                return JSONParsingStrategy.SEMANTIC

            # Check for arrays that should be split
            for value in data.values():
                if isinstance(value, list) and len(value) >= self._config.min_array_size_for_splitting:
                    # Check if array contains objects (not primitives)
                    if value and isinstance(value[0], dict):
                        return JSONParsingStrategy.RECORDS

        # Check if data itself is an array
        elif isinstance(data, list):
            if len(data) >= self._config.min_array_size_for_splitting:
                if data and isinstance(data[0], dict):
                    return JSONParsingStrategy.RECORDS

        # Default to flatten for config-like structures
        return JSONParsingStrategy.FLATTEN

    def _parse_single_json_object(
        self,
        obj: Any,
        file_path: str,
        line_index: int,
        strategy: JSONParsingStrategy,
    ) -> Optional[ParsedJSONChunk]:
        """Parse a single JSON object (for JSONL processing)."""
        if strategy == JSONParsingStrategy.FLATTEN:
            content = self._flatten_to_text(obj)
            meta = {
                "source_path": file_path,
                "source_type": "jsonl",
                "json_parsing_strategy": strategy.value,
                "line_index": line_index,
            }
            # Extract preserve fields
            if isinstance(obj, dict):
                for field in self._preserve_fields_set:
                    if field in obj:
                        meta[field] = obj[field]

            return ParsedJSONChunk(content=content, meta=meta)

        elif strategy == JSONParsingStrategy.LOGS:
            return self._parse_log_object(obj, file_path, line_index)

        elif strategy == JSONParsingStrategy.SEMANTIC:
            return self._parse_semantic_object(obj, file_path, line_index)

        else:
            # Fallback to flatten
            content = self._flatten_to_text(obj)
            meta = {
                "source_path": file_path,
                "source_type": "jsonl",
                "json_parsing_strategy": "flatten",
                "line_index": line_index,
            }
            return ParsedJSONChunk(content=content, meta=meta)

    def _parse_flatten(self, data: Any, file_path: str) -> List[ParsedJSONChunk]:
        """Parse JSON by flattening to key-value text."""
        content = self._flatten_to_text(data)

        meta = {
            "source_path": file_path,
            "source_type": "json",
            "json_parsing_strategy": "flatten",
            "json_schema_hint": self._get_schema_hint(data),
        }

        # Extract preserve fields
        if isinstance(data, dict):
            for field in self._preserve_fields_set:
                if field in data:
                    meta[field] = data[field]

        return [ParsedJSONChunk(content=content, meta=meta)]

    def _parse_records(self, data: Any, file_path: str) -> List[ParsedJSONChunk]:
        """Parse JSON by splitting arrays into separate documents."""
        chunks = []

        if isinstance(data, list):
            # Top-level array
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    content = self._flatten_to_text(item)
                    meta = {
                        "source_path": file_path,
                        "source_type": "json",
                        "json_parsing_strategy": "records",
                        "record_index": idx,
                        "total_records": len(data),
                    }
                    # Extract preserve fields
                    for field in self._preserve_fields_set:
                        if field in item:
                            meta[field] = item[field]

                    chunks.append(ParsedJSONChunk(content=content, meta=meta))

        elif isinstance(data, dict):
            # Find arrays in the object
            for key, value in data.items():
                if isinstance(value, list) and len(value) >= self._config.min_array_size_for_splitting:
                    for idx, item in enumerate(value):
                        if isinstance(item, dict):
                            content = self._flatten_to_text(item)
                            meta = {
                                "source_path": file_path,
                                "source_type": "json",
                                "json_parsing_strategy": "records",
                                "parent_key": key,
                                "record_index": idx,
                                "total_records": len(value),
                            }
                            # Extract preserve fields
                            for field in self._preserve_fields_set:
                                if field in item:
                                    meta[field] = item[field]

                            chunks.append(ParsedJSONChunk(content=content, meta=meta))

            # If no arrays found, fall back to flatten
            if not chunks:
                return self._parse_flatten(data, file_path)

        return chunks

    def _parse_semantic(self, data: Any, file_path: str) -> List[ParsedJSONChunk]:
        """Parse JSON by extracting semantic text content."""
        if not isinstance(data, dict):
            return self._parse_flatten(data, file_path)

        # Extract title
        title = None
        for field in self._title_fields_set:
            if field in data and isinstance(data[field], str):
                title = data[field]
                break

        # Extract text content
        text_parts = []
        if title:
            text_parts.append(f"Title: {title}")

        for field in self._text_fields_set:
            if field in data and isinstance(data[field], str):
                value = data[field]
                if len(value) > 20:  # Skip very short values
                    text_parts.append(f"{field.capitalize()}: {value}")

        # If no text content found, fall back to flatten
        if not text_parts:
            return self._parse_flatten(data, file_path)

        content = "\n\n".join(text_parts)

        meta = {
            "source_path": file_path,
            "source_type": "json",
            "json_parsing_strategy": "semantic",
        }

        # Preserve metadata fields
        for field in self._preserve_fields_set:
            if field in data:
                meta[field] = data[field]

        # Store non-text fields in metadata
        for key, value in data.items():
            if key.lower() not in self._text_fields_set and key.lower() not in self._title_fields_set:
                if isinstance(value, (str, int, float, bool)):
                    meta[f"json_{key}"] = value

        return [ParsedJSONChunk(content=content, meta=meta, title=title)]

    def _parse_logs(self, data: Any, file_path: str) -> List[ParsedJSONChunk]:
        """Parse JSON as structured log entries."""
        if isinstance(data, list):
            # Array of log entries
            chunks = []
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    chunk = self._parse_log_object(item, file_path, idx)
                    if chunk:
                        chunks.append(chunk)
            return chunks
        elif isinstance(data, dict):
            # Single log entry
            chunk = self._parse_log_object(data, file_path, 0)
            return [chunk] if chunk else []
        else:
            return self._parse_flatten(data, file_path)

    def _parse_log_object(
        self,
        obj: Dict[str, Any],
        file_path: str,
        index: int
    ) -> Optional[ParsedJSONChunk]:
        """Parse a single log entry object."""
        if not isinstance(obj, dict):
            return None

        # Extract common log fields
        timestamp = None
        level = None
        message = None

        for key, value in obj.items():
            key_lower = key.lower()
            if key_lower in ['timestamp', 'time', 'date', 'datetime', 'created_at']:
                timestamp = str(value)
            elif key_lower in ['level', 'severity', 'loglevel']:
                level = str(value)
            elif key_lower in ['message', 'msg', 'log', 'text']:
                message = str(value)

        # Build log text
        parts = []
        if timestamp:
            parts.append(f"[{timestamp}]")
        if level:
            parts.append(f"[{level}]")
        if message:
            parts.append(message)

        # Add other fields
        for key, value in obj.items():
            key_lower = key.lower()
            if key_lower not in ['timestamp', 'time', 'date', 'datetime', 'created_at',
                                 'level', 'severity', 'loglevel', 'message', 'msg', 'log', 'text']:
                if isinstance(value, (str, int, float, bool)):
                    parts.append(f"{key}: {value}")
                elif isinstance(value, dict):
                    parts.append(f"{key}: {json.dumps(value)}")

        content = " | ".join(parts)

        meta = {
            "source_path": file_path,
            "source_type": "jsonl" if ".jsonl" in file_path else "json",
            "json_parsing_strategy": "logs",
            "log_index": index,
        }

        if timestamp:
            meta["timestamp"] = timestamp
        if level:
            meta["log_level"] = level

        return ParsedJSONChunk(content=content, meta=meta)

    def _flatten_to_text(
        self,
        data: Any,
        prefix: str = "",
        depth: int = 0
    ) -> str:
        """
        Recursively flatten JSON to readable text.

        Args:
            data: JSON data to flatten
            prefix: Key prefix for nested objects
            depth: Current nesting depth

        Returns:
            Flattened text representation
        """
        if depth > self._config.max_nesting_depth:
            return f"{prefix}: [max depth exceeded]"

        lines = []

        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}{self._config.flatten_separator}{key}" if prefix else key

                if isinstance(value, (dict, list)):
                    lines.append(self._flatten_to_text(value, new_prefix, depth + 1))
                else:
                    lines.append(f"{new_prefix}: {value}")

        elif isinstance(data, list):
            if not data:
                lines.append(f"{prefix}: []")
            elif all(isinstance(item, (str, int, float, bool, type(None))) for item in data):
                # Simple array - format inline
                lines.append(f"{prefix}: [{', '.join(str(item) for item in data)}]")
            else:
                # Complex array - expand
                for idx, item in enumerate(data):
                    new_prefix = f"{prefix}[{idx}]"
                    if isinstance(item, (dict, list)):
                        lines.append(self._flatten_to_text(item, new_prefix, depth + 1))
                    else:
                        lines.append(f"{new_prefix}: {item}")

        else:
            # Primitive value
            lines.append(f"{prefix}: {data}" if prefix else str(data))

        return "\n".join(lines)

    def _get_schema_hint(self, data: Any) -> str:
        """Get a hint about the JSON schema structure."""
        if isinstance(data, dict):
            return "object"
        elif isinstance(data, list):
            return "array"
        else:
            return "primitive"


def parse_json_file(
    file_path: str,
    config: Optional[JSONParsingConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to parse a JSON file.

    Args:
        file_path: Path to JSON file
        config: Parsing configuration

    Returns:
        List of chunk dictionaries (compatible with IngestedChunk)
    """
    parser = JSONParser(config)

    if file_path.endswith('.jsonl'):
        chunks = parser.parse_jsonl_file(file_path)
    else:
        chunks = parser.parse_json_file(file_path)

    return [chunk.to_ingested_chunk_dict() for chunk in chunks]
