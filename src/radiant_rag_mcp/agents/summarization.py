"""
Summarization agent for RAG pipeline.

Provides query-aware document summarization and context compression
to optimize context window utilization and improve answer quality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from radiant_rag_mcp.llm.client import LLMClient
    from radiant_rag_mcp.utils.conversation import ConversationTurn

logger = logging.getLogger(__name__)


@dataclass
class CompressedDocument:
    """A compressed/summarized document."""
    
    original_id: str
    content: str
    original_length: int
    compressed_length: int
    compression_ratio: float
    summary: str
    key_facts: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def was_compressed(self) -> bool:
        return self.compressed_length < self.original_length


@dataclass
class SummarizationResult:
    """Result of a summarization operation."""
    
    documents: List[CompressedDocument]
    total_original_chars: int
    total_compressed_chars: int
    compression_ratio: float
    documents_compressed: int
    clusters_merged: int = 0


class SummarizationAgent:
    """
    Query-aware document summarization for context optimization.
    
    Activates conditionally based on:
    - Document length exceeding threshold
    - Conversation history exceeding threshold
    - Multiple highly-similar documents retrieved
    
    Provides:
    - Query-focused document compression
    - Conversation history summarization
    - Similar document deduplication and merging
    """
    
    def __init__(
        self,
        llm: "LLMClient",
        min_doc_length_for_summary: int = 2000,
        target_summary_length: int = 500,
        conversation_compress_threshold: int = 6,
        conversation_preserve_recent: int = 2,
        similarity_threshold: float = 0.85,
        max_cluster_size: int = 3,
    ) -> None:
        """
        Initialize the summarization agent.
        
        Args:
            llm: LLM client for summarization
            min_doc_length_for_summary: Minimum doc length to trigger summarization
            target_summary_length: Target length for summaries
            conversation_compress_threshold: Compress after this many turns
            conversation_preserve_recent: Keep this many recent turns verbatim
            similarity_threshold: Threshold for clustering similar docs
            max_cluster_size: Maximum documents to merge in a cluster
        """
        self._llm = llm
        self._min_doc_length = min_doc_length_for_summary
        self._target_length = target_summary_length
        self._conv_threshold = conversation_compress_threshold
        self._preserve_recent = conversation_preserve_recent
        self._similarity_threshold = similarity_threshold
        self._max_cluster_size = max_cluster_size
    
    def should_summarize_documents(
        self,
        docs: List[Any],
        total_char_limit: int = 8000,
    ) -> bool:
        """
        Determine if document summarization should be activated.
        
        Args:
            docs: List of documents to potentially summarize
            total_char_limit: Total character budget for context
            
        Returns:
            True if summarization should be activated
        """
        if not docs:
            return False
        
        total_chars = sum(len(getattr(doc, 'content', str(doc))) for doc in docs)
        
        # Activate if total exceeds limit
        if total_chars > total_char_limit:
            return True
        
        # Activate if any single document is very long
        for doc in docs:
            content = getattr(doc, 'content', str(doc))
            if len(content) > self._min_doc_length:
                return True
        
        return False
    
    def should_compress_conversation(
        self,
        turns: List["ConversationTurn"],
    ) -> bool:
        """
        Determine if conversation compression should be activated.
        
        Args:
            turns: Conversation turns
            
        Returns:
            True if compression should be activated
        """
        return len(turns) > self._conv_threshold
    
    def summarize_for_query(
        self,
        content: str,
        query: str,
        max_length: Optional[int] = None,
    ) -> str:
        """
        Create a query-focused summary of content.
        
        Args:
            content: Content to summarize
            query: Query to focus summary on
            max_length: Maximum summary length (uses target_length if None)
            
        Returns:
            Query-focused summary
        """
        max_length = max_length or self._target_length

        # Don't summarize if already short enough
        if len(content) <= max_length:
            return content

        # Guard against content that would exceed LLM token limits.
        # ~4 chars/token is a conservative estimate; cap at 100K chars
        # (~25K tokens) to leave headroom for system prompt and response.
        max_input_chars = 100_000
        if len(content) > max_input_chars:
            logger.debug(
                f"Truncating content from {len(content)} to {max_input_chars} "
                f"chars before summarization to stay within token limits"
            )
            content = content[:max_input_chars]

        system = """You are a document summarization specialist for a RAG system.
Create a concise summary that preserves information relevant to answering the query.

PRESERVE:
- Key facts, figures, and statistics
- Information directly relevant to the query
- Important quotes or specific claims (paraphrased)
- Technical details if the query is technical
- Proper nouns, names, and specific references

OMIT:
- Redundant or repetitive information
- Generic introductions or conclusions
- Tangential details not relevant to the query
- Boilerplate text

Return ONLY the summary, no preamble or explanation."""

        user = f"""Query: {query}

Document ({len(content)} characters):
{content}

Create a summary of approximately {max_length} characters that preserves query-relevant information."""

        response = self._llm.chat(
            self._llm.create_messages(system, user),
            retry_on_error=True,
        )
        
        if not response.success:
            # Fall back to truncation with ellipsis
            logger.warning(f"Summarization failed: {response.error}")
            return content[:max_length - 3] + "..."
        
        return response.content.strip()
    
    def compress_documents(
        self,
        docs: List[Any],
        query: str,
        max_total_chars: int = 8000,
        scores: Optional[List[float]] = None,
    ) -> SummarizationResult:
        """
        Compress a list of documents for context optimization.
        
        Args:
            docs: Documents to compress
            query: Query for relevance-focused compression
            max_total_chars: Maximum total characters for all documents
            scores: Optional relevance scores for prioritization
            
        Returns:
            SummarizationResult with compressed documents
        """
        if not docs:
            return SummarizationResult(
                documents=[],
                total_original_chars=0,
                total_compressed_chars=0,
                compression_ratio=1.0,
                documents_compressed=0,
            )
        
        # Calculate per-document budget
        num_docs = len(docs)
        per_doc_budget = max_total_chars // num_docs
        
        # Adjust budget based on scores if available
        if scores and len(scores) == len(docs):
            total_score = sum(scores)
            if total_score > 0:
                budgets = [
                    int((score / total_score) * max_total_chars * 1.5)  # Allow some flexibility
                    for score in scores
                ]
            else:
                budgets = [per_doc_budget] * num_docs
        else:
            budgets = [per_doc_budget] * num_docs
        
        # Ensure minimum budget
        min_budget = 200
        budgets = [max(min_budget, b) for b in budgets]
        
        compressed_docs = []
        total_original = 0
        total_compressed = 0
        num_compressed = 0
        
        for i, doc in enumerate(docs):
            content = getattr(doc, 'content', str(doc))
            doc_id = getattr(doc, 'doc_id', f"doc_{i}")
            doc_meta = getattr(doc, 'meta', {})
            
            original_length = len(content)
            total_original += original_length
            budget = budgets[i]
            
            if original_length > budget:
                # Need to compress
                summary = self.summarize_for_query(content, query, budget)
                compressed_length = len(summary)
                num_compressed += 1
                
                # Extract key facts
                key_facts = self._extract_key_facts(content, query)
                
                compressed_docs.append(CompressedDocument(
                    original_id=doc_id,
                    content=summary,
                    original_length=original_length,
                    compressed_length=compressed_length,
                    compression_ratio=compressed_length / original_length,
                    summary=summary[:200] + "..." if len(summary) > 200 else summary,
                    key_facts=key_facts,
                    metadata={**doc_meta, "was_compressed": True},
                ))
                total_compressed += compressed_length
            else:
                # Keep as-is
                compressed_docs.append(CompressedDocument(
                    original_id=doc_id,
                    content=content,
                    original_length=original_length,
                    compressed_length=original_length,
                    compression_ratio=1.0,
                    summary="",
                    key_facts=[],
                    metadata={**doc_meta, "was_compressed": False},
                ))
                total_compressed += original_length
        
        return SummarizationResult(
            documents=compressed_docs,
            total_original_chars=total_original,
            total_compressed_chars=total_compressed,
            compression_ratio=total_compressed / total_original if total_original > 0 else 1.0,
            documents_compressed=num_compressed,
        )
    
    def compress_conversation(
        self,
        turns: List["ConversationTurn"],
        max_length: Optional[int] = None,
    ) -> str:
        """
        Compress conversation history while preserving recent turns.
        
        Args:
            turns: Conversation turns to compress
            max_length: Maximum length for compressed history
            
        Returns:
            Compressed conversation history string
        """
        if not turns:
            return ""
        
        if len(turns) <= self._preserve_recent:
            return self._format_turns(turns)
        
        # Split into old and recent
        old_turns = turns[:-self._preserve_recent]
        recent_turns = turns[-self._preserve_recent:]
        
        # Format old turns for summarization
        old_text = self._format_turns(old_turns)
        
        # Summarize old turns
        system = """Summarize this conversation history concisely.
Capture:
- Key topics discussed
- Important decisions or conclusions
- Relevant context for continuing the conversation
- Any specific requests or preferences mentioned

Format as a brief summary paragraph."""

        user = f"Conversation to summarize:\n{old_text}"
        
        response = self._llm.chat(
            self._llm.create_messages(system, user),
            retry_on_error=True,
        )
        
        if response.success:
            summary = response.content.strip()
        else:
            # Fall back to simple truncation
            summary = f"[Earlier: {len(old_turns)} turns discussing various topics]"
        
        # Combine summary with recent turns
        recent_text = self._format_turns(recent_turns)
        
        result = f"[Previous conversation summary: {summary}]\n\n{recent_text}"
        
        # Truncate if still too long
        if max_length and len(result) > max_length:
            result = result[:max_length - 3] + "..."
        
        return result
    
    def deduplicate_similar_documents(
        self,
        docs: List[Any],
        query: str,
        embeddings: Optional[List[List[float]]] = None,
    ) -> Tuple[List[Any], int]:
        """
        Identify and merge highly similar documents.
        
        Args:
            docs: Documents to deduplicate
            query: Query for relevance context
            embeddings: Optional pre-computed embeddings
            
        Returns:
            Tuple of (deduplicated docs, number of clusters merged)
        """
        if len(docs) <= 1:
            return docs, 0
        
        # Without embeddings, use content similarity
        if embeddings is None:
            return self._deduplicate_by_content(docs, query)
        
        # Cluster by embedding similarity
        clusters = self._cluster_by_embeddings(docs, embeddings)
        
        merged_docs = []
        clusters_merged = 0
        
        for cluster in clusters:
            if len(cluster) == 1:
                merged_docs.append(cluster[0])
            else:
                # Merge cluster into single document
                merged = self._merge_document_cluster(cluster, query)
                merged_docs.append(merged)
                clusters_merged += 1
        
        return merged_docs, clusters_merged
    
    def _extract_key_facts(self, content: str, query: str) -> List[str]:
        """Extract key facts from content relevant to query."""
        
        # For efficiency, use a simple extraction approach
        # Could be enhanced with LLM for better extraction
        
        facts = []
        
        # Look for sentences with numbers (often facts)
        sentences = content.split('.')
        for sent in sentences[:20]:  # Limit for efficiency
            sent = sent.strip()
            if any(char.isdigit() for char in sent) and len(sent) > 20:
                facts.append(sent[:100])
        
        return facts[:5]  # Return top 5 facts
    
    def _format_turns(self, turns: List["ConversationTurn"]) -> str:
        """Format conversation turns as text."""
        lines = []
        for turn in turns:
            role = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{role}: {turn.content}")
        return "\n".join(lines)
    
    def _deduplicate_by_content(
        self,
        docs: List[Any],
        query: str,
    ) -> Tuple[List[Any], int]:
        """Deduplicate documents by content overlap."""
        
        # Simple overlap-based deduplication
        unique_docs = []
        merged_count = 0
        
        for doc in docs:
            content = getattr(doc, 'content', str(doc))
            content_words = set(content.lower().split())
            
            is_duplicate = False
            for unique_doc in unique_docs:
                unique_content = getattr(unique_doc, 'content', str(unique_doc))
                unique_words = set(unique_content.lower().split())
                
                # Calculate Jaccard similarity
                if unique_words:
                    intersection = len(content_words & unique_words)
                    union = len(content_words | unique_words)
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity > self._similarity_threshold:
                        is_duplicate = True
                        merged_count += 1
                        break
            
            if not is_duplicate:
                unique_docs.append(doc)
        
        return unique_docs, merged_count
    
    def _cluster_by_embeddings(
        self,
        docs: List[Any],
        embeddings: List[List[float]],
    ) -> List[List[Any]]:
        """Cluster documents by embedding similarity."""
        
        n = len(docs)
        assigned = [False] * n
        clusters = []
        
        for i in range(n):
            if assigned[i]:
                continue
            
            cluster = [docs[i]]
            assigned[i] = True
            
            for j in range(i + 1, n):
                if assigned[j]:
                    continue
                
                # Calculate cosine similarity
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                
                if sim >= self._similarity_threshold and len(cluster) < self._max_cluster_size:
                    cluster.append(docs[j])
                    assigned[j] = True
            
            clusters.append(cluster)
        
        return clusters
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _merge_document_cluster(
        self,
        cluster: List[Any],
        query: str,
    ) -> Any:
        """Merge a cluster of similar documents into one."""
        
        # Combine content
        combined_content = "\n\n---\n\n".join(
            getattr(doc, 'content', str(doc)) for doc in cluster
        )
        
        # Summarize the combined content
        system = """Merge these similar documents into a single coherent summary.
Remove redundant information while preserving all unique facts and details.
Focus on information relevant to the query."""

        user = f"""Query: {query}

Documents to merge:
{combined_content}

Create a unified summary that preserves all unique information."""

        response = self._llm.chat(
            self._llm.create_messages(system, user),
            retry_on_error=True,
        )
        
        if response.success:
            merged_content = response.content.strip()
        else:
            # Fall back to concatenation
            merged_content = combined_content
        
        # Return a simple object with merged content
        # This works because we only access .content attribute
        class MergedDoc:
            def __init__(self, content: str, meta: Dict[str, Any]):
                self.content = content
                self.meta = meta
                self.doc_id = f"merged_{len(cluster)}_docs"
        
        return MergedDoc(
            content=merged_content,
            meta={
                "merged_from": [getattr(d, 'doc_id', 'unknown') for d in cluster],
                "merge_count": len(cluster),
            }
        )
