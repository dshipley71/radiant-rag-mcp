"""
Retrieval strategy memory for agentic RAG pipeline.

Tracks which retrieval strategies work best for different query patterns,
enabling adaptive strategy selection over time.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class StrategyOutcome:
    """Outcome of a retrieval strategy for a query."""
    query_hash: str
    query_pattern: str
    strategy: str  # "dense", "bm25", "hybrid"
    timestamp: float
    
    # Performance metrics
    num_retrieved: int = 0
    num_relevant: int = 0  # Based on reranking scores
    avg_score: float = 0.0
    
    # Answer quality (from critic)
    answer_confidence: float = 0.0
    critic_ok: bool = True
    
    # Overall success
    success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "StrategyOutcome":
        return StrategyOutcome(**data)


@dataclass
class QueryPattern:
    """Represents a query pattern with associated statistics."""
    pattern: str
    total_queries: int = 0
    strategy_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def record_outcome(self, strategy: str, confidence: float, success: bool) -> None:
        """Record an outcome for a strategy."""
        if strategy not in self.strategy_stats:
            self.strategy_stats[strategy] = {
                "count": 0,
                "total_confidence": 0.0,
                "successes": 0,
                "avg_confidence": 0.0,
                "success_rate": 0.0,
            }
        
        stats = self.strategy_stats[strategy]
        stats["count"] += 1
        stats["total_confidence"] += confidence
        if success:
            stats["successes"] += 1
        
        # Update averages
        stats["avg_confidence"] = stats["total_confidence"] / stats["count"]
        stats["success_rate"] = stats["successes"] / stats["count"]
        
        self.total_queries += 1
    
    def get_best_strategy(self) -> Optional[str]:
        """Get the best strategy based on historical performance."""
        if not self.strategy_stats:
            return None
        
        # Score = avg_confidence * success_rate
        best_strategy = None
        best_score = -1
        
        for strategy, stats in self.strategy_stats.items():
            if stats["count"] >= 2:  # Require minimum samples
                score = stats["avg_confidence"] * stats["success_rate"]
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
        
        return best_strategy
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern": self.pattern,
            "total_queries": self.total_queries,
            "strategy_stats": self.strategy_stats,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "QueryPattern":
        return QueryPattern(
            pattern=data["pattern"],
            total_queries=data.get("total_queries", 0),
            strategy_stats=data.get("strategy_stats", {}),
        )


class QueryPatternExtractor:
    """
    Extracts patterns from queries for similarity matching.
    
    Uses simple heuristics to classify queries into patterns.
    """
    
    # Query type patterns
    PATTERNS = [
        (r"^(what|who|where|when|which|how)\s+is\s+", "factual_is"),
        (r"^(what|who|where|when|which|how)\s+are\s+", "factual_are"),
        (r"^(what|who|where|when|which|how)\s+", "wh_question"),
        (r"^(can|could|would|should|will|do|does|did)\s+", "yes_no"),
        (r"^(explain|describe|tell me about)\s+", "explanation"),
        (r"^(compare|contrast|difference|vs|versus)\s+", "comparison"),
        (r"^(list|enumerate|name|give me)\s+", "listing"),
        (r"^(why|reason|cause)\s+", "causal"),
        (r"^(how to|steps to|guide|tutorial)\s+", "procedural"),
        (r"(latest|recent|current|today|news|update)\s+", "temporal"),
        (r"(code|programming|function|script|algorithm)\s+", "technical_code"),
        (r"(calculate|compute|math|number|percentage)\s+", "mathematical"),
        (r"(definition|define|meaning of)\s+", "definitional"),
        (r"(summarize|summary|tldr|brief)\s+", "summarization"),
    ]
    
    def extract_pattern(self, query: str) -> str:
        """
        Extract a pattern from a query.
        
        Args:
            query: User query
            
        Returns:
            Pattern identifier string
        """
        query_lower = query.lower().strip()
        
        for pattern, label in self.PATTERNS:
            if re.search(pattern, query_lower):
                return label
        
        # Fallback: classify by query length
        word_count = len(query_lower.split())
        if word_count <= 3:
            return "short_query"
        elif word_count <= 8:
            return "medium_query"
        else:
            return "long_query"
    
    def get_query_hash(self, query: str) -> str:
        """Generate a hash for the query."""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()[:12]


class RetrievalStrategyMemory:
    """
    Persistent memory for retrieval strategy performance.
    
    Tracks which strategies work best for different query patterns
    and provides recommendations for new queries.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_outcomes: int = 10000,
        decay_factor: float = 0.95,
    ):
        """
        Initialize strategy memory.
        
        Args:
            storage_path: Path to store memory (None = in-memory only)
            max_outcomes: Maximum outcomes to store
            decay_factor: Weight decay for older outcomes (0-1)
        """
        self._storage_path = Path(storage_path) if storage_path else None
        self._max_outcomes = max_outcomes
        self._decay_factor = decay_factor
        
        self._extractor = QueryPatternExtractor()
        self._patterns: Dict[str, QueryPattern] = {}
        self._recent_outcomes: List[StrategyOutcome] = []
        
        # Global strategy stats
        self._global_stats: Dict[str, Dict[str, float]] = {
            "dense": {"count": 0, "total_confidence": 0.0, "successes": 0},
            "bm25": {"count": 0, "total_confidence": 0.0, "successes": 0},
            "hybrid": {"count": 0, "total_confidence": 0.0, "successes": 0},
        }
        
        self._load()
    
    def record_outcome(
        self,
        query: str,
        strategy: str,
        confidence: float,
        success: bool,
        num_retrieved: int = 0,
        num_relevant: int = 0,
        avg_score: float = 0.0,
        critic_ok: bool = True,
    ) -> None:
        """
        Record the outcome of a retrieval strategy.
        
        Args:
            query: User query
            strategy: Strategy used ("dense", "bm25", "hybrid")
            confidence: Answer confidence (0-1)
            success: Whether the retrieval was successful
            num_retrieved: Number of documents retrieved
            num_relevant: Number of relevant documents (post-rerank)
            avg_score: Average retrieval score
            critic_ok: Whether critic approved the answer
        """
        pattern = self._extractor.extract_pattern(query)
        query_hash = self._extractor.get_query_hash(query)
        
        # Create outcome record
        outcome = StrategyOutcome(
            query_hash=query_hash,
            query_pattern=pattern,
            strategy=strategy,
            timestamp=time.time(),
            num_retrieved=num_retrieved,
            num_relevant=num_relevant,
            avg_score=avg_score,
            answer_confidence=confidence,
            critic_ok=critic_ok,
            success=success,
        )
        
        # Update pattern stats
        if pattern not in self._patterns:
            self._patterns[pattern] = QueryPattern(pattern=pattern)
        
        self._patterns[pattern].record_outcome(strategy, confidence, success)
        
        # Update global stats
        if strategy in self._global_stats:
            stats = self._global_stats[strategy]
            stats["count"] += 1
            stats["total_confidence"] += confidence
            if success:
                stats["successes"] += 1
        
        # Store outcome
        self._recent_outcomes.append(outcome)
        
        # Prune if needed
        if len(self._recent_outcomes) > self._max_outcomes:
            self._recent_outcomes = self._recent_outcomes[-self._max_outcomes:]
        
        # Auto-save periodically
        if len(self._recent_outcomes) % 100 == 0:
            self._save()
        
        logger.debug(
            f"Recorded outcome: pattern={pattern}, strategy={strategy}, "
            f"confidence={confidence:.2f}, success={success}"
        )
    
    def recommend_strategy(
        self,
        query: str,
        available_strategies: Optional[List[str]] = None,
    ) -> Tuple[str, float]:
        """
        Recommend a retrieval strategy for a query.
        
        Args:
            query: User query
            available_strategies: List of available strategies
            
        Returns:
            Tuple of (recommended_strategy, confidence)
        """
        if available_strategies is None:
            available_strategies = ["dense", "bm25", "hybrid"]
        
        pattern = self._extractor.extract_pattern(query)
        
        # Check pattern-specific recommendation
        if pattern in self._patterns:
            pattern_data = self._patterns[pattern]
            best = pattern_data.get_best_strategy()
            
            if best and best in available_strategies:
                stats = pattern_data.strategy_stats.get(best, {})
                confidence = stats.get("avg_confidence", 0.5)
                logger.debug(f"Pattern-based recommendation: {best} (confidence={confidence:.2f})")
                return best, confidence
        
        # Fallback to global stats
        best_strategy = "hybrid"  # Default
        best_score = 0.0
        
        for strategy in available_strategies:
            if strategy in self._global_stats:
                stats = self._global_stats[strategy]
                if stats["count"] > 0:
                    avg_conf = stats["total_confidence"] / stats["count"]
                    success_rate = stats["successes"] / stats["count"]
                    score = avg_conf * success_rate
                    
                    if score > best_score:
                        best_score = score
                        best_strategy = strategy
        
        confidence = best_score if best_score > 0 else 0.5
        logger.debug(f"Global-based recommendation: {best_strategy} (confidence={confidence:.2f})")
        
        return best_strategy, confidence
    
    def get_pattern_stats(self, query: str) -> Optional[Dict[str, Any]]:
        """Get statistics for the query's pattern."""
        pattern = self._extractor.extract_pattern(query)
        
        if pattern in self._patterns:
            return self._patterns[pattern].to_dict()
        
        return None
    
    def get_global_stats(self) -> Dict[str, Dict[str, float]]:
        """Get global strategy statistics."""
        result = {}
        for strategy, stats in self._global_stats.items():
            if stats["count"] > 0:
                result[strategy] = {
                    "count": stats["count"],
                    "avg_confidence": stats["total_confidence"] / stats["count"],
                    "success_rate": stats["successes"] / stats["count"],
                }
            else:
                result[strategy] = {"count": 0, "avg_confidence": 0.0, "success_rate": 0.0}
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the memory state."""
        return {
            "total_outcomes": len(self._recent_outcomes),
            "patterns_tracked": len(self._patterns),
            "global_stats": self.get_global_stats(),
            "top_patterns": sorted(
                [
                    {"pattern": p.pattern, "queries": p.total_queries}
                    for p in self._patterns.values()
                ],
                key=lambda x: x["queries"],
                reverse=True,
            )[:10],
        }
    
    def _save(self) -> None:
        """Save memory to disk."""
        if self._storage_path is None:
            return
        
        try:
            data = {
                "patterns": {k: v.to_dict() for k, v in self._patterns.items()},
                "global_stats": self._global_stats,
                "recent_outcomes": [o.to_dict() for o in self._recent_outcomes[-1000:]],
            }
            
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            with gzip.open(self._storage_path, 'wt', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved strategy memory to {self._storage_path}")
        except Exception as e:
            logger.warning(f"Failed to save strategy memory: {e}")
    
    def _load(self) -> None:
        """Load memory from disk."""
        if self._storage_path is None or not self._storage_path.exists():
            return
        
        try:
            with gzip.open(self._storage_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            
            self._patterns = {
                k: QueryPattern.from_dict(v)
                for k, v in data.get("patterns", {}).items()
            }
            
            self._global_stats = data.get("global_stats", self._global_stats)
            
            self._recent_outcomes = [
                StrategyOutcome.from_dict(o)
                for o in data.get("recent_outcomes", [])
            ]
            
            logger.info(
                f"Loaded strategy memory: {len(self._patterns)} patterns, "
                f"{len(self._recent_outcomes)} outcomes"
            )
        except Exception as e:
            logger.warning(f"Failed to load strategy memory: {e}")
    
    def clear(self) -> None:
        """Clear all memory."""
        self._patterns.clear()
        self._recent_outcomes.clear()
        self._global_stats = {
            "dense": {"count": 0, "total_confidence": 0.0, "successes": 0},
            "bm25": {"count": 0, "total_confidence": 0.0, "successes": 0},
            "hybrid": {"count": 0, "total_confidence": 0.0, "successes": 0},
        }
        
        if self._storage_path and self._storage_path.exists():
            self._storage_path.unlink()
        
        logger.info("Cleared strategy memory")
