"""
Conversation history management for Radiant Agentic RAG.

Provides persistent multi-turn conversation support with Redis storage
or an in-memory fallback when Redis is not the configured storage backend.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from radiant_rag_mcp.config import ConversationConfig, RedisConfig

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    turn_id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "turn_id": self.turn_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ConversationTurn":
        """Create from dictionary."""
        return ConversationTurn(
            turn_id=data.get("turn_id", str(uuid.uuid4())),
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Conversation:
    """A complete conversation with multiple turns."""

    conversation_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_turn(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationTurn:
        """
        Add a new turn to the conversation.

        Args:
            role: "user" or "assistant"
            content: Turn content
            metadata: Optional turn metadata

        Returns:
            The created turn
        """
        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self.turns.append(turn)
        self.updated_at = time.time()
        return turn

    def get_recent_turns(self, n: int) -> List[ConversationTurn]:
        """Get the n most recent turns."""
        return self.turns[-n:] if n > 0 else []

    def get_history_text(
        self,
        max_turns: int = 5,
        include_metadata: bool = False,
    ) -> str:
        """
        Format recent conversation history as text.

        Args:
            max_turns: Maximum number of turns to include
            include_metadata: Whether to include turn metadata

        Returns:
            Formatted conversation history
        """
        recent = self.get_recent_turns(max_turns)

        lines = []
        for turn in recent:
            role_label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{role_label}: {turn.content}")

            if include_metadata and turn.metadata:
                meta_str = json.dumps(turn.metadata, ensure_ascii=False)
                lines.append(f"  [Metadata: {meta_str}]")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "conversation_id": self.conversation_id,
            "turns": [t.to_dict() for t in self.turns],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Conversation":
        """Create from dictionary."""
        turns = [ConversationTurn.from_dict(t) for t in data.get("turns", [])]
        return Conversation(
            conversation_id=data.get("conversation_id", str(uuid.uuid4())),
            turns=turns,
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            metadata=data.get("metadata", {}),
        )

    def __len__(self) -> int:
        return len(self.turns)


class ConversationStore:
    """
    Conversation storage backed by Redis or an in-memory dict.

    Redis is used only when ``RADIANT_STORAGE_BACKEND == 'redis'`` (the
    default).  For any other backend value the store uses a plain in-memory
    dict so the rest of the application works without a running Redis
    instance.  No behaviour change occurs when the backend is redis.

    Storage schema (Redis):
        {prefix}:{conv_ns}:{conversation_id} -> JSON conversation data
        {prefix}:{conv_ns}:index -> Set of conversation IDs
    """

    def __init__(
        self,
        redis_config: RedisConfig,
        conv_config: ConversationConfig,
    ) -> None:
        """
        Initialize conversation store.

        Args:
            redis_config: Redis connection configuration
            conv_config: Conversation configuration
        """
        self._redis_config = redis_config
        self._conv_config = conv_config
        self._prefix = redis_config.key_prefix
        self._ns = redis_config.conversation_ns

        # Only connect to Redis when it is the configured storage backend.
        # RADIANT_STORAGE_BACKEND mirrors config.storage.backend and defaults
        # to "redis" to preserve existing behaviour.
        storage_backend = os.environ.get("RADIANT_STORAGE_BACKEND", "redis").lower()
        if storage_backend == "redis":
            import redis
            self._r = redis.Redis.from_url(redis_config.url, decode_responses=True)
            # In-memory fields unused when Redis is active
            self._mem: Dict[str, str] = {}
            self._mem_index: Set[str] = set()
        else:
            self._r = None
            self._mem = {}
            self._mem_index = set()
            logger.info(
                "ConversationStore: storage backend is %r — using in-memory storage",
                storage_backend,
            )

    def _conv_key(self, conversation_id: str) -> str:
        """Generate Redis key for a conversation."""
        return f"{self._prefix}:{self._ns}:{conversation_id}"

    def _index_key(self) -> str:
        """Generate Redis key for conversation index."""
        return f"{self._prefix}:{self._ns}:index"

    def create_conversation(
        self,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Conversation:
        """
        Create a new conversation.

        Args:
            conversation_id: Optional ID (generated if not provided)
            metadata: Optional conversation metadata

        Returns:
            New Conversation instance
        """
        conv_id = conversation_id or str(uuid.uuid4())
        conv = Conversation(
            conversation_id=conv_id,
            metadata=metadata or {},
        )
        self.save(conv)
        logger.debug(f"Created conversation: {conv_id}")
        return conv

    def save(self, conversation: Conversation) -> None:
        """
        Save a conversation to the store.

        Args:
            conversation: Conversation to save
        """
        # Enforce max turns limit
        if len(conversation.turns) > self._conv_config.max_turns:
            conversation.turns = conversation.turns[-self._conv_config.max_turns:]

        key = self._conv_key(conversation.conversation_id)
        data = json.dumps(conversation.to_dict(), ensure_ascii=False)

        if self._r is not None:
            if self._conv_config.ttl > 0:
                self._r.setex(key, self._conv_config.ttl, data)
            else:
                self._r.set(key, data)
            self._r.sadd(self._index_key(), conversation.conversation_id)
        else:
            self._mem[key] = data
            self._mem_index.add(conversation.conversation_id)

    def get(self, conversation_id: str) -> Optional[Conversation]:
        """
        Retrieve a conversation by ID.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Conversation if found, None otherwise
        """
        key = self._conv_key(conversation_id)

        if self._r is not None:
            data = self._r.get(key)
        else:
            data = self._mem.get(key)

        if data is None:
            return None

        try:
            return Conversation.from_dict(json.loads(data))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse conversation {conversation_id}: {e}")
            return None

    def get_or_create(
        self,
        conversation_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Conversation:
        """
        Get an existing conversation or create a new one.

        Args:
            conversation_id: Conversation identifier
            metadata: Metadata for new conversation (if created)

        Returns:
            Conversation instance
        """
        conv = self.get(conversation_id)
        if conv is not None:
            return conv
        return self.create_conversation(conversation_id, metadata)

    def add_turn(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationTurn:
        """
        Add a turn to a conversation.

        Args:
            conversation_id: Conversation identifier
            role: "user" or "assistant"
            content: Turn content
            metadata: Optional turn metadata

        Returns:
            The created turn
        """
        conv = self.get_or_create(conversation_id)
        turn = conv.add_turn(role, content, metadata)
        self.save(conv)
        return turn

    def delete(self, conversation_id: str) -> bool:
        """
        Delete a conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            True if conversation was deleted
        """
        key = self._conv_key(conversation_id)

        if self._r is not None:
            deleted = self._r.delete(key)
            self._r.srem(self._index_key(), conversation_id)
            result = bool(deleted)
        else:
            result = key in self._mem
            self._mem.pop(key, None)
            self._mem_index.discard(conversation_id)

        if result:
            logger.debug(f"Deleted conversation: {conversation_id}")

        return result

    def list_conversations(self, limit: int = 100) -> List[str]:
        """
        List conversation IDs.

        Args:
            limit: Maximum number of IDs to return

        Returns:
            List of conversation IDs
        """
        if self._r is not None:
            members = self._r.smembers(self._index_key())
        else:
            members = self._mem_index
        return list(members)[:limit]

    def cleanup_expired(self) -> int:
        """
        Remove expired conversations from the index.

        Returns:
            Number of conversations removed from index
        """
        if self._r is not None:
            conv_ids = self.list_conversations(limit=10000)
            removed = 0
            for conv_id in conv_ids:
                key = self._conv_key(conv_id)
                if not self._r.exists(key):
                    self._r.srem(self._index_key(), conv_id)
                    removed += 1
            if removed > 0:
                logger.info(f"Cleaned up {removed} expired conversations from index")
            return removed
        else:
            # In-memory storage has no TTL expiry; nothing to clean up.
            return 0


class ConversationManager:
    """
    High-level conversation management for RAG pipeline.

    Manages conversation state and provides formatted history
    for query augmentation and answer synthesis.
    """

    def __init__(
        self,
        store: ConversationStore,
        config: ConversationConfig,
    ) -> None:
        """
        Initialize conversation manager.

        Args:
            store: Conversation storage backend
            config: Conversation configuration
        """
        self._store = store
        self._config = config
        self._current_conversation: Optional[Conversation] = None
        self._current_id: Optional[str] = None

    @property
    def enabled(self) -> bool:
        """Check if conversation history is enabled."""
        return self._config.enabled

    @property
    def current_conversation(self) -> Optional[Conversation]:
        """Get the current active conversation."""
        return self._current_conversation

    @property
    def conversation_id(self) -> Optional[str]:
        """Get the current conversation ID."""
        return self._current_id

    def start_conversation(
        self,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Start or resume a conversation.

        Args:
            conversation_id: Optional ID to resume (generated if not provided)
            metadata: Optional conversation metadata

        Returns:
            Conversation ID
        """
        if not self.enabled:
            return conversation_id or str(uuid.uuid4())

        if conversation_id:
            self._current_conversation = self._store.get_or_create(
                conversation_id, metadata
            )
        else:
            self._current_conversation = self._store.create_conversation(
                metadata=metadata
            )

        self._current_id = self._current_conversation.conversation_id
        return self._current_id

    def add_user_query(
        self,
        query: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ConversationTurn]:
        """
        Record a user query in the conversation.

        Args:
            query: User query text
            metadata: Optional turn metadata

        Returns:
            Created turn, or None if history is disabled
        """
        if not self.enabled or self._current_conversation is None:
            return None

        turn = self._current_conversation.add_turn("user", query, metadata)
        self._store.save(self._current_conversation)
        return turn

    def add_assistant_response(
        self,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ConversationTurn]:
        """
        Record an assistant response in the conversation.

        Args:
            response: Assistant response text
            metadata: Optional turn metadata (e.g., sources used)

        Returns:
            Created turn, or None if history is disabled
        """
        if not self.enabled or self._current_conversation is None:
            return None

        turn = self._current_conversation.add_turn("assistant", response, metadata)
        self._store.save(self._current_conversation)
        return turn

    def get_history_for_query(self) -> str:
        """
        Get formatted history for query augmentation.

        Returns:
            Formatted conversation history string
        """
        if not self.enabled or self._current_conversation is None:
            return ""

        if not self._config.use_history_for_retrieval:
            return ""

        return self._current_conversation.get_history_text(
            max_turns=self._config.history_turns_for_context
        )

    def get_history_for_synthesis(self, max_turns: Optional[int] = None) -> str:
        """
        Get formatted history for answer synthesis.

        Args:
            max_turns: Override max turns (uses config default if None)

        Returns:
            Formatted conversation history string
        """
        if not self.enabled or self._current_conversation is None:
            return ""

        turns = max_turns if max_turns is not None else self._config.history_turns_for_context
        return self._current_conversation.get_history_text(max_turns=turns)

    def get_recent_queries(self, n: int = 3) -> List[str]:
        """
        Get recent user queries for context.

        Args:
            n: Number of recent queries to return

        Returns:
            List of recent query strings
        """
        if not self.enabled or self._current_conversation is None:
            return []

        queries = []
        for turn in reversed(self._current_conversation.turns):
            if turn.role == "user":
                queries.append(turn.content)
                if len(queries) >= n:
                    break

        return list(reversed(queries))

    def end_conversation(self) -> None:
        """End the current conversation."""
        self._current_conversation = None
        self._current_id = None

    def load_conversation(self, conversation_id: str) -> bool:
        """
        Load an existing conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            True if conversation was loaded
        """
        if not self.enabled:
            return False

        conv = self._store.get(conversation_id)
        if conv is None:
            return False

        self._current_conversation = conv
        self._current_id = conversation_id
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get conversation manager statistics."""
        stats = {
            "enabled": self.enabled,
            "current_conversation_id": self._current_id,
        }

        if self._current_conversation:
            stats["current_turns"] = len(self._current_conversation)
            stats["current_created_at"] = self._current_conversation.created_at
            stats["current_updated_at"] = self._current_conversation.updated_at

        return stats
