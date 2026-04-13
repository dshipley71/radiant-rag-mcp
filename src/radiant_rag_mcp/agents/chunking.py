"""
Intelligent chunking agent for RAG pipeline.

Provides LLM-based semantic chunking at ingestion time to improve
retrieval quality by preserving logical document boundaries.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from radiant_rag_mcp.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class SemanticChunk:
    """A semantically coherent chunk from a document."""
    
    content: str
    chunk_index: int
    chunk_type: str  # "paragraph", "section", "code_block", "list", "table", etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.content)


@dataclass 
class ChunkingResult:
    """Result of intelligent chunking operation."""
    
    chunks: List[SemanticChunk]
    total_chars: int
    chunking_method: str  # "llm", "rule_based", "hybrid"
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntelligentChunkingAgent:
    """
    LLM-based semantic chunking for improved retrieval quality.
    
    Unlike fixed-size chunking, this agent:
    - Identifies logical document boundaries (paragraphs, sections, topics)
    - Preserves semantic coherence within chunks
    - Adapts chunking strategy based on document type
    - Maintains context across chunk boundaries
    """
    
    # Document type patterns for rule-based detection
    DOC_TYPE_PATTERNS = {
        "code": r"(?:```|def |class |import |from |function |const |let |var )",
        "markdown": r"(?:^#{1,6}\s|^\*{1,3}\s|^-\s|^\d+\.\s)",
        "structured": r"(?:^[A-Z][^.!?]*:\s*$|^\s*[-•]\s)",
        "legal": r"(?:WHEREAS|ARTICLE|Section \d|§)",
        "technical": r"(?:Figure \d|Table \d|Equation \d|Algorithm \d)",
    }
    
    # Natural break patterns
    BREAK_PATTERNS = [
        r"\n\n+",  # Double newlines (paragraph breaks)
        r"\n(?=#{1,6}\s)",  # Before markdown headers
        r"\n(?=\d+\.\s+[A-Z])",  # Before numbered sections
        r"\n(?=[A-Z][^.!?]{0,50}:?\s*\n)",  # Before section titles
        r"(?<=\.)\s+(?=[A-Z])",  # Sentence boundaries (fallback)
    ]
    
    def __init__(
        self,
        llm: Optional["LLMClient"] = None,
        min_chunk_size: int = 200,
        max_chunk_size: int = 1500,
        target_chunk_size: int = 800,
        overlap_size: int = 100,
        use_llm_chunking: bool = True,
        llm_chunk_threshold: int = 3000,
    ) -> None:
        """
        Initialize the intelligent chunking agent.
        
        Args:
            llm: LLM client for semantic chunking (optional)
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters
            target_chunk_size: Target chunk size for LLM guidance
            overlap_size: Overlap between chunks for context preservation
            use_llm_chunking: Whether to use LLM for chunking decisions
            llm_chunk_threshold: Document length threshold for LLM chunking
        """
        self._llm = llm
        self._min_chunk_size = min_chunk_size
        self._max_chunk_size = max_chunk_size
        self._target_chunk_size = target_chunk_size
        self._overlap_size = overlap_size
        self._use_llm_chunking = use_llm_chunking and llm is not None
        self._llm_chunk_threshold = llm_chunk_threshold
    
    def chunk_document(
        self,
        content: str,
        doc_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChunkingResult:
        """
        Chunk a document using intelligent semantic boundaries.
        
        Args:
            content: Document content to chunk
            doc_type: Optional document type hint
            metadata: Optional document metadata
            
        Returns:
            ChunkingResult with semantic chunks
        """
        if not content or not content.strip():
            return ChunkingResult(
                chunks=[],
                total_chars=0,
                chunking_method="empty",
            )
        
        content = content.strip()
        total_chars = len(content)
        
        # Detect document type if not provided
        if doc_type is None:
            doc_type = self._detect_document_type(content)
        
        # Choose chunking strategy
        if total_chars <= self._min_chunk_size:
            # Document is small enough to be a single chunk
            chunks = [SemanticChunk(
                content=content,
                chunk_index=0,
                chunk_type="full_document",
                metadata=metadata or {},
            )]
            return ChunkingResult(
                chunks=chunks,
                total_chars=total_chars,
                chunking_method="single",
                metadata={"doc_type": doc_type},
            )
        
        # Use LLM chunking for complex documents above threshold
        if (
            self._use_llm_chunking 
            and total_chars >= self._llm_chunk_threshold
            and doc_type not in ("code",)  # Code is better with rule-based
        ):
            try:
                result = self._llm_chunk(content, doc_type, metadata)
                if result.chunks:
                    return result
            except Exception as e:
                logger.warning(f"LLM chunking failed, falling back to rule-based: {e}")
        
        # Fall back to rule-based chunking
        return self._rule_based_chunk(content, doc_type, metadata)
    
    def _detect_document_type(self, content: str) -> str:
        """Detect the document type from content patterns."""
        content_sample = content[:2000]  # Sample for efficiency
        
        for doc_type, pattern in self.DOC_TYPE_PATTERNS.items():
            if re.search(pattern, content_sample, re.MULTILINE):
                return doc_type
        
        return "prose"
    
    def _llm_chunk(
        self,
        content: str,
        doc_type: str,
        metadata: Optional[Dict[str, Any]],
    ) -> ChunkingResult:
        """Use LLM to identify semantic chunk boundaries."""
        
        # For very long documents, we need to process in sections
        if len(content) > 15000:
            return self._llm_chunk_long_document(content, doc_type, metadata)
        
        system = f"""You are a document chunking specialist. Analyze the document and identify natural semantic boundaries for chunking.

Document type: {doc_type}
Target chunk size: ~{self._target_chunk_size} characters
Minimum chunk size: {self._min_chunk_size} characters
Maximum chunk size: {self._max_chunk_size} characters

Identify logical break points where the document naturally divides into coherent sections.
Each chunk should:
- Be self-contained and semantically coherent
- Preserve complete thoughts, paragraphs, or code blocks
- Not break mid-sentence or mid-code-block
- Include enough context to be understandable standalone

Return a JSON object with:
{{
  "chunks": [
    {{"start": 0, "end": 500, "type": "introduction", "summary": "Brief description"}},
    {{"start": 500, "end": 1200, "type": "section", "summary": "Brief description"}},
    ...
  ]
}}

The start/end values are character indices. Ensure chunks cover the entire document without gaps."""

        user = f"Document to chunk ({len(content)} characters):\n\n{content}"
        
        result, response = self._llm.chat_json(
            system=system,
            user=user,
            default={"chunks": []},
            expected_type=dict,
        )
        
        if not response.success or not result.get("chunks"):
            raise ValueError("LLM chunking returned no chunks")
        
        # Parse LLM response into SemanticChunks
        chunks = []
        raw_chunks = result.get("chunks", [])
        
        for i, chunk_info in enumerate(raw_chunks):
            start = int(chunk_info.get("start", 0))
            end = int(chunk_info.get("end", len(content)))
            
            # Clamp to document bounds
            start = max(0, min(start, len(content)))
            end = max(start, min(end, len(content)))
            
            chunk_content = content[start:end].strip()
            
            if len(chunk_content) >= self._min_chunk_size // 2:  # Allow slightly smaller
                chunks.append(SemanticChunk(
                    content=chunk_content,
                    chunk_index=i,
                    chunk_type=chunk_info.get("type", "section"),
                    metadata={
                        **(metadata or {}),
                        "summary": chunk_info.get("summary", ""),
                        "char_start": start,
                        "char_end": end,
                    },
                ))
        
        # Validate coverage - merge small gaps
        chunks = self._validate_and_fix_coverage(chunks, content)
        
        return ChunkingResult(
            chunks=chunks,
            total_chars=len(content),
            chunking_method="llm",
            metadata={"doc_type": doc_type},
        )
    
    def _llm_chunk_long_document(
        self,
        content: str,
        doc_type: str,
        metadata: Optional[Dict[str, Any]],
    ) -> ChunkingResult:
        """Handle very long documents by chunking in sections."""
        
        # First, do a coarse rule-based split
        coarse_chunks = self._rule_based_chunk(
            content, 
            doc_type, 
            metadata,
            target_size=5000,  # Larger chunks for LLM refinement
        )
        
        # Then refine each coarse chunk with LLM
        refined_chunks = []
        chunk_index = 0
        
        for coarse_chunk in coarse_chunks.chunks:
            if len(coarse_chunk.content) > self._max_chunk_size:
                # Needs further splitting
                sub_result = self._llm_chunk(
                    coarse_chunk.content,
                    doc_type,
                    metadata,
                )
                for sub_chunk in sub_result.chunks:
                    sub_chunk.chunk_index = chunk_index
                    refined_chunks.append(sub_chunk)
                    chunk_index += 1
            else:
                coarse_chunk.chunk_index = chunk_index
                refined_chunks.append(coarse_chunk)
                chunk_index += 1
        
        return ChunkingResult(
            chunks=refined_chunks,
            total_chars=len(content),
            chunking_method="hybrid",
            metadata={"doc_type": doc_type},
        )
    
    def _rule_based_chunk(
        self,
        content: str,
        doc_type: str,
        metadata: Optional[Dict[str, Any]],
        target_size: Optional[int] = None,
    ) -> ChunkingResult:
        """Rule-based chunking using natural document boundaries."""
        
        target = target_size or self._target_chunk_size
        
        # Split by document type
        if doc_type == "code":
            chunks = self._chunk_code(content, target)
        elif doc_type == "markdown":
            chunks = self._chunk_markdown(content, target)
        else:
            chunks = self._chunk_prose(content, target)
        
        # Convert to SemanticChunks
        semantic_chunks = []
        for i, chunk_content in enumerate(chunks):
            chunk_content = chunk_content.strip()
            if chunk_content:
                semantic_chunks.append(SemanticChunk(
                    content=chunk_content,
                    chunk_index=i,
                    chunk_type=self._infer_chunk_type(chunk_content, doc_type),
                    metadata=metadata or {},
                ))
        
        return ChunkingResult(
            chunks=semantic_chunks,
            total_chars=len(content),
            chunking_method="rule_based",
            metadata={"doc_type": doc_type},
        )
    
    def _chunk_prose(self, content: str, target_size: int) -> List[str]:
        """Chunk prose content by paragraph boundaries."""
        
        # Split by double newlines first
        paragraphs = re.split(r'\n\s*\n', content)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_size = len(para)
            
            # If single paragraph exceeds max, split it
            if para_size > self._max_chunk_size:
                # Flush current chunk
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    if current_size + len(sent) > target_size and current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                        # Add overlap from last chunk
                        overlap_text = current_chunk[-1] if current_chunk else ""
                        current_chunk = [overlap_text] if len(overlap_text) <= self._overlap_size else []
                        current_size = len(overlap_text) if current_chunk else 0
                    current_chunk.append(sent)
                    current_size += len(sent)
            
            # Check if adding this paragraph exceeds target
            elif current_size + para_size > target_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                # Add overlap
                overlap_text = current_chunk[-1] if current_chunk else ""
                if len(overlap_text) <= self._overlap_size:
                    current_chunk = [overlap_text, para]
                    current_size = len(overlap_text) + para_size
                else:
                    current_chunk = [para]
                    current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        # Flush remaining
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _chunk_markdown(self, content: str, target_size: int) -> List[str]:
        """Chunk markdown content by headers and sections."""
        
        # Split by headers
        header_pattern = r'^(#{1,6}\s+.+)$'
        parts = re.split(header_pattern, content, flags=re.MULTILINE)
        
        chunks = []
        current_chunk = []
        current_size = 0
        current_header = ""
        
        i = 0
        while i < len(parts):
            part = parts[i].strip()
            
            if re.match(r'^#{1,6}\s+', part):
                # This is a header
                if current_chunk and current_size > self._min_chunk_size:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                current_header = part
                current_chunk.append(part)
                current_size += len(part)
            elif part:
                # Content section
                if current_size + len(part) > target_size and current_size > self._min_chunk_size:
                    chunks.append('\n\n'.join(current_chunk))
                    # Start new chunk with header context if available
                    if current_header:
                        current_chunk = [f"[Continued from: {current_header}]", part]
                        current_size = len(current_chunk[0]) + len(part)
                    else:
                        current_chunk = [part]
                        current_size = len(part)
                else:
                    current_chunk.append(part)
                    current_size += len(part)
            
            i += 1
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _chunk_code(self, content: str, target_size: int) -> List[str]:
        """Chunk code content by function/class boundaries."""
        
        # Try to split by function/class definitions
        # Pattern for Python, JavaScript, etc.
        code_block_pattern = r'(?:^|\n)((?:def |class |function |const |async function |export ).+?)(?=\n(?:def |class |function |const |async function |export )|\Z)'
        
        blocks = re.findall(code_block_pattern, content, re.DOTALL)
        
        if not blocks:
            # Fall back to line-based chunking
            return self._chunk_by_lines(content, target_size)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            
            block_size = len(block)
            
            if block_size > self._max_chunk_size:
                # Large block - split by lines while preserving structure
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                sub_chunks = self._chunk_by_lines(block, target_size)
                chunks.extend(sub_chunks)
            
            elif current_size + block_size > target_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [block]
                current_size = block_size
            else:
                current_chunk.append(block)
                current_size += block_size
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _chunk_by_lines(self, content: str, target_size: int) -> List[str]:
        """Fall back chunking by lines."""
        
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            if current_size + line_size > target_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _infer_chunk_type(self, content: str, doc_type: str) -> str:
        """Infer the type of a chunk from its content."""
        
        content_start = content[:200].lower()
        
        if re.match(r'^#{1,6}\s+', content):
            return "section"
        elif re.match(r'^(def |class |function |const )', content):
            return "code_block"
        elif re.match(r'^\d+\.\s+', content):
            return "numbered_list"
        elif re.match(r'^[-•*]\s+', content):
            return "bullet_list"
        elif '|' in content_start and content.count('|') > 4:
            return "table"
        elif doc_type == "code":
            return "code"
        else:
            return "paragraph"
    
    def _validate_and_fix_coverage(
        self,
        chunks: List[SemanticChunk],
        original_content: str,
    ) -> List[SemanticChunk]:
        """Validate that chunks cover the document and fix gaps."""
        
        if not chunks:
            return chunks
        
        # Sort by start position
        chunks_with_pos = []
        for chunk in chunks:
            start = chunk.metadata.get("char_start", 0)
            end = chunk.metadata.get("char_end", len(chunk.content))
            chunks_with_pos.append((start, end, chunk))
        
        chunks_with_pos.sort(key=lambda x: x[0])
        
        # Check for gaps and fix
        fixed_chunks = []
        expected_start = 0
        
        for start, end, chunk in chunks_with_pos:
            if start > expected_start + 50:  # Allow small gaps
                # There's a gap - create a chunk for it
                gap_content = original_content[expected_start:start].strip()
                if len(gap_content) >= self._min_chunk_size // 4:
                    gap_chunk = SemanticChunk(
                        content=gap_content,
                        chunk_index=len(fixed_chunks),
                        chunk_type="gap_fill",
                        metadata={"char_start": expected_start, "char_end": start},
                    )
                    fixed_chunks.append(gap_chunk)
            
            chunk.chunk_index = len(fixed_chunks)
            fixed_chunks.append(chunk)
            expected_start = end
        
        # Check for content after last chunk
        if expected_start < len(original_content) - 50:
            remaining = original_content[expected_start:].strip()
            if len(remaining) >= self._min_chunk_size // 4:
                fixed_chunks.append(SemanticChunk(
                    content=remaining,
                    chunk_index=len(fixed_chunks),
                    chunk_type="remainder",
                    metadata={"char_start": expected_start, "char_end": len(original_content)},
                ))
        
        return fixed_chunks
