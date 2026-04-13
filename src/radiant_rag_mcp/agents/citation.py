"""
Citation tracking agent for RAG pipeline.

Tracks and formats source references for generated answers,
enabling enterprise compliance and auditability.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from radiant_rag_mcp.llm.client import LLMClient

logger = logging.getLogger(__name__)


class CitationStyle(Enum):
    """Citation formatting styles."""
    INLINE = "inline"  # [1], [2], etc.
    FOOTNOTE = "footnote"  # Superscript numbers
    ACADEMIC = "academic"  # Author (Year)
    HYPERLINK = "hyperlink"  # [Source](url)
    ENTERPRISE = "enterprise"  # Doc ID + timestamp


@dataclass
class SourceDocument:
    """Represents a source document for citation."""
    
    doc_id: str
    title: str
    content_preview: str  # First N chars of content
    source_type: str  # "indexed", "web_search", "uploaded"
    source_path: Optional[str] = None
    source_url: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content_preview": self.content_preview[:200],
            "source_type": self.source_type,
            "source_path": self.source_path,
            "source_url": self.source_url,
            "author": self.author,
            "date": self.date,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "relevance_score": self.relevance_score,
        }
    
    def generate_citation_key(self) -> str:
        """Generate a unique citation key."""
        components = [self.doc_id]
        if self.page_number:
            components.append(f"p{self.page_number}")
        if self.chunk_index:
            components.append(f"c{self.chunk_index}")
        return "_".join(components)


@dataclass
class Citation:
    """A single citation linking a claim to a source."""
    
    citation_id: int
    claim_text: str
    source: SourceDocument
    supporting_excerpt: str
    confidence: float
    start_char: Optional[int] = None  # Position in answer
    end_char: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "claim_text": self.claim_text,
            "source": self.source.to_dict(),
            "supporting_excerpt": self.supporting_excerpt[:300],
            "confidence": self.confidence,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


@dataclass
class CitedAnswer:
    """An answer with citation tracking."""
    
    original_answer: str
    cited_answer: str  # Answer with inline citations
    citations: List[Citation]
    sources: List[SourceDocument]
    citation_style: CitationStyle
    coverage_score: float  # % of claims with citations
    timestamp: str
    audit_id: str  # Unique ID for audit trail
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cited_answer": self.cited_answer,
            "citations": [c.to_dict() for c in self.citations],
            "sources": [s.to_dict() for s in self.sources],
            "citation_style": self.citation_style.value,
            "coverage_score": self.coverage_score,
            "timestamp": self.timestamp,
            "audit_id": self.audit_id,
            "num_citations": len(self.citations),
            "num_sources": len(self.sources),
        }
    
    def get_bibliography(self) -> str:
        """Generate a formatted bibliography."""
        lines = ["## Sources\n"]
        
        for i, source in enumerate(self.sources, start=1):
            entry = f"[{i}] "
            
            if source.title:
                entry += f"**{source.title}**"
            else:
                entry += f"Document {source.doc_id}"
            
            if source.author:
                entry += f" by {source.author}"
            
            if source.date:
                entry += f" ({source.date})"
            
            if source.source_url:
                entry += f"\n    URL: {source.source_url}"
            elif source.source_path:
                entry += f"\n    File: {source.source_path}"
            
            if source.page_number:
                entry += f"\n    Page: {source.page_number}"
            
            lines.append(entry)
        
        return "\n".join(lines)
    
    def get_audit_log(self) -> Dict[str, Any]:
        """Generate audit log entry."""
        return {
            "audit_id": self.audit_id,
            "timestamp": self.timestamp,
            "answer_length": len(self.original_answer),
            "num_citations": len(self.citations),
            "num_sources": len(self.sources),
            "coverage_score": self.coverage_score,
            "sources_used": [
                {
                    "doc_id": s.doc_id,
                    "source_type": s.source_type,
                    "relevance_score": s.relevance_score,
                }
                for s in self.sources
            ],
        }


class CitationTrackingAgent:
    """
    Tracks and formats citations for RAG answers.
    
    Provides:
    - Automatic citation insertion into answers
    - Multiple citation styles (inline, footnote, academic, etc.)
    - Source document tracking with metadata
    - Audit trail generation for compliance
    - Bibliography/references generation
    
    Essential for:
    - Enterprise compliance requirements
    - Legal document generation
    - Academic/research applications
    - Audit trails and provenance tracking
    """
    
    def __init__(
        self,
        llm: "LLMClient",
        citation_style: CitationStyle = CitationStyle.INLINE,
        min_citation_confidence: float = 0.5,
        max_citations_per_claim: int = 3,
        include_excerpts: bool = True,
        excerpt_max_length: int = 200,
    ) -> None:
        """
        Initialize the citation tracking agent.
        
        Args:
            llm: LLM client for citation matching
            citation_style: Default citation style
            min_citation_confidence: Minimum confidence for citations
            max_citations_per_claim: Maximum citations per claim
            include_excerpts: Include supporting excerpts
            excerpt_max_length: Maximum excerpt length
        """
        self._llm = llm
        self._citation_style = citation_style
        self._min_confidence = min_citation_confidence
        self._max_citations = max_citations_per_claim
        self._include_excerpts = include_excerpts
        self._excerpt_length = excerpt_max_length
    
    def extract_sources(
        self,
        context_docs: List[Any],
        scores: Optional[List[float]] = None,
    ) -> List[SourceDocument]:
        """
        Extract source document metadata from context.

        Deduplicates multiple chunks from the same parent document.

        Args:
            context_docs: Retrieved documents
            scores: Optional relevance scores

        Returns:
            List of deduplicated SourceDocument objects
        """
        # Group chunks by parent document
        parent_groups: Dict[str, List[tuple]] = {}

        for i, doc in enumerate(context_docs):
            meta = getattr(doc, 'meta', {})
            score = scores[i] if scores and i < len(scores) else 0.5

            # Determine parent document identifier
            # Priority: parent_id > source_url > source_path > doc_id
            parent_key = (
                meta.get("parent_id") or
                meta.get("source_url") or
                meta.get("source_path") or
                getattr(doc, 'doc_id', f"doc_{i}")
            )

            if parent_key not in parent_groups:
                parent_groups[parent_key] = []
            parent_groups[parent_key].append((i, doc, score))

        # Create one SourceDocument per parent
        sources = []

        for parent_key, chunks in parent_groups.items():
            # Use chunk with highest relevance score as representative
            chunks_sorted = sorted(chunks, key=lambda x: x[2], reverse=True)
            best_idx, best_doc, best_score = chunks_sorted[0]

            # Extract metadata from best chunk
            meta = getattr(best_doc, 'meta', {})
            content = getattr(best_doc, 'content', str(best_doc))
            doc_id = getattr(best_doc, 'doc_id', f"doc_{best_idx}")

            # Determine source type
            if meta.get("source_type") == "web_search":
                source_type = "web_search"
            elif meta.get("source_url"):
                source_type = "web"
            elif meta.get("source_path"):
                source_type = "indexed"
            else:
                source_type = "unknown"

            # Extract title
            title = meta.get("page_title") or meta.get("title")
            if not title:
                # Try to extract from content
                title = self._extract_title(content)
            if not title:
                title = f"Source {len(sources) + 1}"

            source = SourceDocument(
                doc_id=doc_id,
                title=title,
                content_preview=content[:500] if content else "",
                source_type=source_type,
                source_path=meta.get("source_path"),
                source_url=meta.get("source_url"),
                author=meta.get("author"),
                date=meta.get("date") or meta.get("fetched_at"),
                page_number=meta.get("page_number"),
                chunk_index=meta.get("chunk_index"),
                relevance_score=best_score,
                metadata=meta,
            )
            sources.append(source)

        # Sort by relevance score (descending)
        sources.sort(key=lambda s: s.relevance_score, reverse=True)

        return sources
    
    def _extract_title(self, content: str) -> Optional[str]:
        """Try to extract a title from content."""
        
        if not content:
            return None
        
        # Try first line if it looks like a title
        first_line = content.split('\n')[0].strip()
        
        if 10 < len(first_line) < 100:
            # Check if it's a header
            if first_line.startswith('#'):
                return first_line.lstrip('#').strip()
            # Check if it ends without punctuation (likely a title)
            if not first_line[-1] in '.!?:':
                return first_line
        
        return None
    
    def match_citations(
        self,
        answer: str,
        sources: List[SourceDocument],
        query: str,
    ) -> List[Citation]:
        """
        Match claims in the answer to source documents.
        
        Args:
            answer: Generated answer
            sources: Available source documents
            query: Original query
            
        Returns:
            List of Citation objects
        """
        # Split answer into sentences/claims
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        
        # Format sources for LLM
        sources_text = ""
        for i, source in enumerate(sources, start=1):
            sources_text += f"\n[SOURCE {i}]\nTitle: {source.title}\n"
            sources_text += f"Content: {source.content_preview[:800]}\n"
        
        system = """Match each claim/sentence to its supporting source documents.

For each sentence, identify which source(s) support it and extract the relevant excerpt.

Return JSON:
{
  "citations": [
    {
      "sentence_index": 0,
      "claim_text": "the sentence text",
      "source_indices": [1, 2],
      "excerpts": ["relevant excerpt from source 1", "excerpt from source 2"],
      "confidence": 0.0-1.0
    }
  ]
}

Rules:
- Only cite sources that directly support the claim
- Include confidence based on how well the source supports the claim
- If a claim is common knowledge or unsupported, use empty source_indices"""

        sentences_text = "\n".join([f"[{i}] {s}" for i, s in enumerate(sentences)])
        
        user = f"""Query: {query}

Answer sentences:
{sentences_text}

Available sources:
{sources_text}

Return JSON only."""

        result, response = self._llm.chat_json(
            system=system,
            user=user,
            default={"citations": []},
            expected_type=dict,
        )
        
        if not response.success:
            return self._fallback_citation_matching(sentences, sources)
        
        citations = []
        raw_citations = result.get("citations", [])
        
        for i, raw in enumerate(raw_citations):
            if not isinstance(raw, dict):
                continue
            
            sentence_idx = raw.get("sentence_index", i)
            if sentence_idx >= len(sentences):
                continue
            
            source_indices = raw.get("source_indices", [])
            if not isinstance(source_indices, list):
                continue
            
            excerpts = raw.get("excerpts", [])
            if not isinstance(excerpts, list):
                excerpts = []
            
            confidence = float(raw.get("confidence", 0.5))
            
            if confidence < self._min_confidence:
                continue
            
            # Create citation for each supporting source
            for j, src_idx in enumerate(source_indices[:self._max_citations]):
                if not isinstance(src_idx, int) or src_idx < 1 or src_idx > len(sources):
                    continue
                
                source = sources[src_idx - 1]
                excerpt = excerpts[j] if j < len(excerpts) else ""
                
                citations.append(Citation(
                    citation_id=len(citations) + 1,
                    claim_text=raw.get("claim_text", sentences[sentence_idx]),
                    source=source,
                    supporting_excerpt=excerpt[:self._excerpt_length],
                    confidence=confidence,
                ))
        
        return citations
    
    def _fallback_citation_matching(
        self,
        sentences: List[str],
        sources: List[SourceDocument],
    ) -> List[Citation]:
        """Fallback: keyword-based citation matching."""
        
        citations = []
        
        for sentence in sentences:
            if len(sentence) < 10:
                continue
            
            # Extract keywords from sentence
            words = set(re.findall(r'\b[a-zA-Z]{4,}\b', sentence.lower()))
            
            # Find best matching source
            best_source = None
            best_score = 0.0
            best_excerpt = ""
            
            for source in sources:
                content_words = set(re.findall(
                    r'\b[a-zA-Z]{4,}\b',
                    source.content_preview.lower()
                ))
                
                overlap = len(words & content_words)
                if overlap > best_score and overlap >= 2:
                    best_score = overlap
                    best_source = source
                    # Extract excerpt around matching words
                    best_excerpt = self._extract_excerpt(
                        source.content_preview, list(words)[:3]
                    )
            
            if best_source and best_score >= 2:
                citations.append(Citation(
                    citation_id=len(citations) + 1,
                    claim_text=sentence,
                    source=best_source,
                    supporting_excerpt=best_excerpt,
                    confidence=min(0.9, best_score / 10),
                ))
        
        return citations
    
    def _extract_excerpt(self, content: str, keywords: List[str]) -> str:
        """Extract an excerpt containing keywords."""
        
        content_lower = content.lower()
        
        for keyword in keywords:
            pos = content_lower.find(keyword.lower())
            if pos >= 0:
                start = max(0, pos - 50)
                end = min(len(content), pos + len(keyword) + 150)
                excerpt = content[start:end]
                if start > 0:
                    excerpt = "..." + excerpt
                if end < len(content):
                    excerpt = excerpt + "..."
                return excerpt
        
        return content[:self._excerpt_length]
    
    def add_citations_to_answer(
        self,
        answer: str,
        citations: List[Citation],
        sources: List[SourceDocument],
        style: Optional[CitationStyle] = None,
    ) -> str:
        """
        Insert citations into the answer text.
        
        Args:
            answer: Original answer
            citations: List of citations
            sources: List of source documents
            style: Citation style to use
            
        Returns:
            Answer with inline citations
        """
        style = style or self._citation_style
        
        if not citations:
            return answer
        
        # Build source index mapping
        source_indices = {}
        for i, source in enumerate(sources, start=1):
            source_indices[source.doc_id] = i
        
        # Group citations by claim
        claim_citations: Dict[str, List[Citation]] = {}
        for citation in citations:
            claim = citation.claim_text
            if claim not in claim_citations:
                claim_citations[claim] = []
            claim_citations[claim].append(citation)
        
        # Insert citations
        cited_answer = answer
        
        for claim, cites in claim_citations.items():
            if not claim or claim not in cited_answer:
                continue
            
            # Build citation marker
            indices = []
            for c in cites[:self._max_citations]:
                idx = source_indices.get(c.source.doc_id)
                if idx:
                    indices.append(idx)
            
            if not indices:
                continue
            
            marker = self._format_citation_marker(indices, style)
            
            # Insert marker after the claim
            # Find the claim and insert after the sentence
            claim_pos = cited_answer.find(claim)
            if claim_pos >= 0:
                insert_pos = claim_pos + len(claim)
                # Find sentence end
                for end_char in '.!?':
                    end_pos = cited_answer.find(end_char, claim_pos)
                    if end_pos >= 0 and end_pos < insert_pos + 50:
                        insert_pos = end_pos + 1
                        break
                
                cited_answer = (
                    cited_answer[:insert_pos] +
                    marker +
                    cited_answer[insert_pos:]
                )
        
        return cited_answer
    
    def _format_citation_marker(
        self,
        indices: List[int],
        style: CitationStyle,
    ) -> str:
        """Format citation markers based on style."""
        
        if style == CitationStyle.INLINE:
            return " [" + ", ".join(str(i) for i in indices) + "]"
        
        elif style == CitationStyle.FOOTNOTE:
            return "^[" + ",".join(str(i) for i in indices) + "]"
        
        elif style == CitationStyle.ACADEMIC:
            # Would need author info for full academic style
            return " (Source " + ", ".join(str(i) for i in indices) + ")"
        
        elif style == CitationStyle.HYPERLINK:
            # Would need URLs for hyperlinks
            return " [" + ", ".join(str(i) for i in indices) + "]"
        
        elif style == CitationStyle.ENTERPRISE:
            return " {ref:" + ",".join(str(i) for i in indices) + "}"
        
        return " [" + ", ".join(str(i) for i in indices) + "]"
    
    def create_cited_answer(
        self,
        answer: str,
        context_docs: List[Any],
        query: str,
        scores: Optional[List[float]] = None,
        style: Optional[CitationStyle] = None,
    ) -> CitedAnswer:
        """
        Create a fully cited answer with sources and audit trail.
        
        Args:
            answer: Generated answer
            context_docs: Retrieved documents
            query: Original query
            scores: Optional relevance scores
            style: Citation style
            
        Returns:
            CitedAnswer with full citation tracking
        """
        style = style or self._citation_style
        timestamp = datetime.utcnow().isoformat() + "Z"
        audit_id = self._generate_audit_id(answer, timestamp)
        
        logger.info(f"Creating cited answer (audit_id={audit_id})")
        
        # Extract source metadata
        sources = self.extract_sources(context_docs, scores)
        
        # Match citations
        citations = self.match_citations(answer, sources, query)
        
        # Add citations to answer
        cited_answer = self.add_citations_to_answer(answer, citations, sources, style)
        
        # Calculate coverage
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        cited_sentences = set(c.claim_text for c in citations)
        coverage = len(cited_sentences) / len(sentences) if sentences else 1.0
        
        logger.info(
            f"Citation complete: {len(citations)} citations, "
            f"{len(sources)} sources, {coverage:.0%} coverage"
        )
        
        return CitedAnswer(
            original_answer=answer,
            cited_answer=cited_answer,
            citations=citations,
            sources=sources,
            citation_style=style,
            coverage_score=coverage,
            timestamp=timestamp,
            audit_id=audit_id,
        )
    
    def _generate_audit_id(self, answer: str, timestamp: str) -> str:
        """Generate a unique audit ID."""
        content = f"{answer[:100]}{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def format_bibliography(
        self,
        sources: List[SourceDocument],
        style: CitationStyle = CitationStyle.INLINE,
    ) -> str:
        """
        Format a bibliography/references section.
        
        Args:
            sources: Source documents
            style: Citation style
            
        Returns:
            Formatted bibliography string
        """
        lines = ["\n---\n## References\n"]
        
        for i, source in enumerate(sources, start=1):
            if style == CitationStyle.ACADEMIC:
                entry = f"[{i}] "
                if source.author:
                    entry += f"{source.author}. "
                if source.date:
                    entry += f"({source.date}). "
                entry += f"*{source.title}*."
                if source.source_url:
                    entry += f" Retrieved from {source.source_url}"
            
            elif style == CitationStyle.ENTERPRISE:
                entry = f"[{i}] Document ID: {source.doc_id}"
                entry += f"\n    Title: {source.title}"
                entry += f"\n    Type: {source.source_type}"
                if source.source_path:
                    entry += f"\n    Path: {source.source_path}"
                if source.source_url:
                    entry += f"\n    URL: {source.source_url}"
                entry += f"\n    Relevance: {source.relevance_score:.2f}"
            
            else:  # Default inline style
                entry = f"[{i}] {source.title}"
                if source.source_url:
                    entry += f" - {source.source_url}"
                elif source.source_path:
                    entry += f" - {source.source_path}"
            
            lines.append(entry)
        
        return "\n".join(lines)
    
    def generate_audit_report(
        self,
        cited_answer: CitedAnswer,
        query: str,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive audit report.
        
        Args:
            cited_answer: The cited answer
            query: Original query
            
        Returns:
            Audit report dictionary
        """
        return {
            "audit_id": cited_answer.audit_id,
            "timestamp": cited_answer.timestamp,
            "query": query,
            "answer_stats": {
                "original_length": len(cited_answer.original_answer),
                "cited_length": len(cited_answer.cited_answer),
                "citation_style": cited_answer.citation_style.value,
            },
            "citation_stats": {
                "total_citations": len(cited_answer.citations),
                "total_sources": len(cited_answer.sources),
                "coverage_score": cited_answer.coverage_score,
                "avg_citation_confidence": (
                    sum(c.confidence for c in cited_answer.citations) /
                    len(cited_answer.citations)
                    if cited_answer.citations else 0.0
                ),
            },
            "sources": [
                {
                    "doc_id": s.doc_id,
                    "title": s.title,
                    "source_type": s.source_type,
                    "relevance_score": s.relevance_score,
                    "source_path": s.source_path,
                    "source_url": s.source_url,
                }
                for s in cited_answer.sources
            ],
            "citations_detail": [
                {
                    "citation_id": c.citation_id,
                    "claim_preview": c.claim_text[:100],
                    "source_doc_id": c.source.doc_id,
                    "confidence": c.confidence,
                    "has_excerpt": bool(c.supporting_excerpt),
                }
                for c in cited_answer.citations
            ],
        }
