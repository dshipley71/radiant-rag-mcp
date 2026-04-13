"""
Fact verification agent for RAG pipeline.

Verifies each claim in generated answers against retrieved context
to prevent hallucinations and ensure factual accuracy.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from radiant_rag_mcp.llm.client import LLMClient

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Status of a fact verification."""
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


@dataclass
class Claim:
    """A single claim extracted from an answer."""
    
    claim_id: int
    text: str
    claim_type: str  # "factual", "opinion", "conditional", "definition"
    entities: List[str]
    requires_verification: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "claim_type": self.claim_type,
            "entities": self.entities,
            "requires_verification": self.requires_verification,
        }


@dataclass
class ClaimVerification:
    """Verification result for a single claim."""
    
    claim: Claim
    status: VerificationStatus
    confidence: float  # 0-1 confidence in verification
    supporting_docs: List[int]  # Indices of supporting documents
    supporting_excerpts: List[str]  # Relevant excerpts from docs
    explanation: str
    suggested_correction: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim.to_dict(),
            "status": self.status.value,
            "confidence": self.confidence,
            "supporting_docs": self.supporting_docs,
            "supporting_excerpts": self.supporting_excerpts,
            "explanation": self.explanation,
            "suggested_correction": self.suggested_correction,
        }


@dataclass
class FactVerificationResult:
    """Complete fact verification result for an answer."""
    
    answer: str
    claims: List[Claim]
    verifications: List[ClaimVerification]
    overall_score: float  # 0-1 overall factuality score
    num_supported: int
    num_partially_supported: int
    num_not_supported: int
    num_contradicted: int
    num_unverifiable: int
    corrected_answer: Optional[str] = None
    issues_found: List[str] = field(default_factory=list)
    
    @property
    def is_factual(self) -> bool:
        """Check if answer is considered factual."""
        return self.overall_score >= 0.7 and self.num_contradicted == 0
    
    @property
    def needs_correction(self) -> bool:
        """Check if answer needs correction."""
        return self.num_contradicted > 0 or self.num_not_supported > len(self.claims) * 0.3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "is_factual": self.is_factual,
            "needs_correction": self.needs_correction,
            "num_claims": len(self.claims),
            "num_supported": self.num_supported,
            "num_partially_supported": self.num_partially_supported,
            "num_not_supported": self.num_not_supported,
            "num_contradicted": self.num_contradicted,
            "num_unverifiable": self.num_unverifiable,
            "verifications": [v.to_dict() for v in self.verifications],
            "issues_found": self.issues_found,
            "has_corrected_answer": self.corrected_answer is not None,
        }


class FactVerificationAgent:
    """
    Verifies factual claims in generated answers.
    
    Process:
    1. Extract individual claims from the answer
    2. For each claim, search for supporting evidence in context
    3. Classify support level (supported, partial, contradicted, etc.)
    4. Calculate overall factuality score
    5. Optionally generate corrected answer
    
    This prevents hallucinations by ensuring every factual claim
    is grounded in the retrieved context.
    """
    
    def __init__(
        self,
        llm: "LLMClient",
        min_support_confidence: float = 0.6,
        max_claims_to_verify: int = 20,
        generate_corrections: bool = True,
        strict_mode: bool = False,
    ) -> None:
        """
        Initialize the fact verification agent.
        
        Args:
            llm: LLM client for verification
            min_support_confidence: Minimum confidence to consider supported
            max_claims_to_verify: Maximum claims to verify (for efficiency)
            generate_corrections: Whether to generate corrected answers
            strict_mode: Require explicit support (vs allowing inference)
        """
        self._llm = llm
        self._min_confidence = min_support_confidence
        self._max_claims = max_claims_to_verify
        self._generate_corrections = generate_corrections
        self._strict_mode = strict_mode
    
    def extract_claims(self, answer: str) -> List[Claim]:
        """
        Extract verifiable claims from an answer.
        
        Args:
            answer: Generated answer text
            
        Returns:
            List of extracted claims
        """
        system = """Extract individual factual claims from this answer.

A claim is a statement that can be verified as true or false.
Exclude:
- Questions
- Opinions that are explicitly marked as such
- Hedged statements ("might be", "could be")
- Meta-statements about the answer itself

Return JSON:
{
  "claims": [
    {
      "text": "The exact claim text",
      "claim_type": "factual|opinion|conditional|definition",
      "entities": ["entity1", "entity2"],
      "requires_verification": true/false
    }
  ]
}

Factual claims require verification. Opinions and hedged statements may not."""

        user = f"""Answer:
{answer}

Extract claims and return JSON only."""

        result, response = self._llm.chat_json(
            system=system,
            user=user,
            default={"claims": []},
            expected_type=dict,
        )
        
        if not response.success:
            # Fallback: split by sentences
            return self._fallback_claim_extraction(answer)
        
        raw_claims = result.get("claims", [])
        
        claims = []
        for i, raw in enumerate(raw_claims[:self._max_claims]):
            if isinstance(raw, dict):
                claims.append(Claim(
                    claim_id=i + 1,
                    text=str(raw.get("text", "")),
                    claim_type=str(raw.get("claim_type", "factual")),
                    entities=raw.get("entities", []) if isinstance(raw.get("entities"), list) else [],
                    requires_verification=bool(raw.get("requires_verification", True)),
                ))
        
        return claims
    
    def _fallback_claim_extraction(self, answer: str) -> List[Claim]:
        """Fallback sentence-based claim extraction."""
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        
        claims = []
        for i, sent in enumerate(sentences[:self._max_claims]):
            sent = sent.strip()
            if len(sent) > 10:  # Skip very short fragments
                claims.append(Claim(
                    claim_id=i + 1,
                    text=sent,
                    claim_type="factual",
                    entities=[],
                    requires_verification=True,
                ))
        
        return claims
    
    def verify_claim(
        self,
        claim: Claim,
        context_docs: List[Any],
        query: str,
    ) -> ClaimVerification:
        """
        Verify a single claim against context documents.
        
        Args:
            claim: Claim to verify
            context_docs: Retrieved context documents
            query: Original query for context
            
        Returns:
            ClaimVerification result
        """
        if not claim.requires_verification:
            return ClaimVerification(
                claim=claim,
                status=VerificationStatus.UNVERIFIABLE,
                confidence=1.0,
                supporting_docs=[],
                supporting_excerpts=[],
                explanation="Claim marked as not requiring verification",
            )
        
        if not context_docs:
            return ClaimVerification(
                claim=claim,
                status=VerificationStatus.UNVERIFIABLE,
                confidence=0.5,
                supporting_docs=[],
                supporting_excerpts=[],
                explanation="No context documents available for verification",
            )
        
        # Format context for verification
        context_parts = []
        for i, doc in enumerate(context_docs[:10], start=1):
            content = getattr(doc, 'content', str(doc))[:2000]
            context_parts.append(f"[DOC {i}] {content}")
        
        context = "\n\n".join(context_parts)
        
        strict_instruction = ""
        if self._strict_mode:
            strict_instruction = """
STRICT MODE: The claim must be explicitly stated in the context.
Do not accept implicit support or reasonable inference."""

        system = f"""Verify if this claim is supported by the context documents.
{strict_instruction}
Verification levels:
- "supported": Claim is directly supported by context
- "partially_supported": Some aspects supported, others not addressed
- "not_supported": Claim is not addressed in context (but not contradicted)
- "contradicted": Context explicitly contradicts the claim
- "unverifiable": Cannot determine from context (e.g., opinion, future prediction)

Return JSON:
{{
  "status": "supported|partially_supported|not_supported|contradicted|unverifiable",
  "confidence": 0.0-1.0,
  "supporting_docs": [1, 2],
  "supporting_excerpts": ["exact quote from doc"],
  "explanation": "why this verification status",
  "suggested_correction": "corrected claim if contradicted or not supported"
}}"""

        user = f"""Original query: {query}

Claim to verify: {claim.text}

Context documents:
{context}

Return JSON only."""

        result, response = self._llm.chat_json(
            system=system,
            user=user,
            default={
                "status": "unverifiable",
                "confidence": 0.5,
                "explanation": "Verification failed",
            },
            expected_type=dict,
        )
        
        if not response.success:
            return ClaimVerification(
                claim=claim,
                status=VerificationStatus.UNVERIFIABLE,
                confidence=0.0,
                supporting_docs=[],
                supporting_excerpts=[],
                explanation="LLM verification failed",
            )
        
        # Parse status
        status_str = str(result.get("status", "unverifiable")).lower()
        status_map = {
            "supported": VerificationStatus.SUPPORTED,
            "partially_supported": VerificationStatus.PARTIALLY_SUPPORTED,
            "not_supported": VerificationStatus.NOT_SUPPORTED,
            "contradicted": VerificationStatus.CONTRADICTED,
            "unverifiable": VerificationStatus.UNVERIFIABLE,
        }
        status = status_map.get(status_str, VerificationStatus.UNVERIFIABLE)
        
        # Parse supporting docs
        supporting_docs = result.get("supporting_docs", [])
        if not isinstance(supporting_docs, list):
            supporting_docs = []
        else:
            supporting_docs = [int(d) for d in supporting_docs if isinstance(d, (int, float))]
        
        # Parse excerpts
        excerpts = result.get("supporting_excerpts", [])
        if not isinstance(excerpts, list):
            excerpts = []
        else:
            excerpts = [str(e)[:500] for e in excerpts]
        
        return ClaimVerification(
            claim=claim,
            status=status,
            confidence=float(result.get("confidence", 0.5)),
            supporting_docs=supporting_docs,
            supporting_excerpts=excerpts,
            explanation=str(result.get("explanation", "")),
            suggested_correction=result.get("suggested_correction"),
        )
    
    def verify_answer(
        self,
        answer: str,
        context_docs: List[Any],
        query: str,
    ) -> FactVerificationResult:
        """
        Verify all claims in an answer.
        
        Args:
            answer: Generated answer to verify
            context_docs: Retrieved context documents
            query: Original user query
            
        Returns:
            FactVerificationResult with all verifications
        """
        logger.info(f"Starting fact verification for answer ({len(answer)} chars)")
        
        # Extract claims
        claims = self.extract_claims(answer)
        logger.debug(f"Extracted {len(claims)} claims for verification")
        
        if not claims:
            return FactVerificationResult(
                answer=answer,
                claims=[],
                verifications=[],
                overall_score=1.0,  # No claims to verify
                num_supported=0,
                num_partially_supported=0,
                num_not_supported=0,
                num_contradicted=0,
                num_unverifiable=0,
            )
        
        # Verify each claim
        verifications: List[ClaimVerification] = []
        
        for claim in claims:
            if claim.requires_verification:
                verification = self.verify_claim(claim, context_docs, query)
            else:
                verification = ClaimVerification(
                    claim=claim,
                    status=VerificationStatus.UNVERIFIABLE,
                    confidence=1.0,
                    supporting_docs=[],
                    supporting_excerpts=[],
                    explanation="Non-factual claim",
                )
            verifications.append(verification)
        
        # Count by status
        status_counts = {status: 0 for status in VerificationStatus}
        for v in verifications:
            status_counts[v.status] += 1
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(verifications)
        
        # Identify issues
        issues = self._identify_issues(verifications)
        
        # Generate corrected answer if needed
        corrected_answer = None
        if self._generate_corrections and (
            status_counts[VerificationStatus.CONTRADICTED] > 0 or
            status_counts[VerificationStatus.NOT_SUPPORTED] > len(claims) * 0.3
        ):
            corrected_answer = self._generate_corrected_answer(
                answer, verifications, context_docs, query
            )
        
        logger.info(
            f"Fact verification complete: score={overall_score:.2f}, "
            f"supported={status_counts[VerificationStatus.SUPPORTED]}, "
            f"contradicted={status_counts[VerificationStatus.CONTRADICTED]}"
        )
        
        return FactVerificationResult(
            answer=answer,
            claims=claims,
            verifications=verifications,
            overall_score=overall_score,
            num_supported=status_counts[VerificationStatus.SUPPORTED],
            num_partially_supported=status_counts[VerificationStatus.PARTIALLY_SUPPORTED],
            num_not_supported=status_counts[VerificationStatus.NOT_SUPPORTED],
            num_contradicted=status_counts[VerificationStatus.CONTRADICTED],
            num_unverifiable=status_counts[VerificationStatus.UNVERIFIABLE],
            corrected_answer=corrected_answer,
            issues_found=issues,
        )
    
    def _calculate_overall_score(self, verifications: List[ClaimVerification]) -> float:
        """Calculate overall factuality score."""
        
        if not verifications:
            return 1.0
        
        # Weight by verification status
        weights = {
            VerificationStatus.SUPPORTED: 1.0,
            VerificationStatus.PARTIALLY_SUPPORTED: 0.7,
            VerificationStatus.NOT_SUPPORTED: 0.3,
            VerificationStatus.CONTRADICTED: 0.0,
            VerificationStatus.UNVERIFIABLE: 0.5,  # Neutral
        }
        
        total_weight = 0.0
        total_score = 0.0
        
        for v in verifications:
            if v.claim.requires_verification:
                weight = weights.get(v.status, 0.5)
                confidence_adjusted = weight * v.confidence
                total_score += confidence_adjusted
                total_weight += 1.0
        
        if total_weight == 0:
            return 1.0
        
        return total_score / total_weight
    
    def _identify_issues(self, verifications: List[ClaimVerification]) -> List[str]:
        """Identify specific issues with the answer."""
        
        issues = []
        
        for v in verifications:
            if v.status == VerificationStatus.CONTRADICTED:
                issues.append(
                    f"Contradicted claim: '{v.claim.text[:50]}...' - {v.explanation}"
                )
            elif v.status == VerificationStatus.NOT_SUPPORTED and v.confidence > 0.7:
                issues.append(
                    f"Unsupported claim: '{v.claim.text[:50]}...'"
                )
        
        return issues[:5]  # Limit to top 5 issues
    
    def _generate_corrected_answer(
        self,
        original_answer: str,
        verifications: List[ClaimVerification],
        context_docs: List[Any],
        query: str,
    ) -> Optional[str]:
        """Generate a corrected version of the answer."""
        
        # Collect corrections needed
        corrections = []
        for v in verifications:
            if v.status in (VerificationStatus.CONTRADICTED, VerificationStatus.NOT_SUPPORTED):
                if v.suggested_correction:
                    corrections.append({
                        "original": v.claim.text,
                        "issue": v.status.value,
                        "correction": v.suggested_correction,
                    })
        
        if not corrections:
            return None
        
        # Format context
        context_parts = []
        for i, doc in enumerate(context_docs[:5], start=1):
            content = getattr(doc, 'content', str(doc))[:1500]
            context_parts.append(f"[{i}] {content}")
        
        context = "\n\n".join(context_parts)
        
        system = """Generate a corrected version of the answer that fixes the identified issues.

Rules:
1. Remove or correct contradicted claims
2. Replace unsupported claims with information from context
3. Keep supported claims unchanged
4. Maintain the same overall structure and tone
5. Ensure the corrected answer still addresses the query"""

        corrections_text = "\n".join([
            f"- Original: {c['original']}\n  Issue: {c['issue']}\n  Suggested: {c['correction']}"
            for c in corrections
        ])
        
        user = f"""Query: {query}

Original answer:
{original_answer}

Issues to correct:
{corrections_text}

Context for corrections:
{context}

Generate corrected answer:"""

        response = self._llm.chat(
            self._llm.create_messages(system, user),
            retry_on_error=True,
        )
        
        if not response.success:
            return None
        
        corrected = response.content.strip()
        
        # Sanity check: corrected answer shouldn't be empty or too short
        if len(corrected) < 20:
            return None
        
        return corrected
    
    def quick_verify(
        self,
        answer: str,
        context_docs: List[Any],
    ) -> Tuple[bool, float]:
        """
        Quick verification check without detailed claim extraction.
        
        Args:
            answer: Answer to verify
            context_docs: Context documents
            
        Returns:
            Tuple of (is_factual, confidence)
        """
        if not context_docs:
            return True, 0.5  # Can't verify without context
        
        # Format context
        context_parts = []
        for i, doc in enumerate(context_docs[:5], start=1):
            content = getattr(doc, 'content', str(doc))[:1500]
            context_parts.append(f"[{i}] {content}")
        
        context = "\n\n".join(context_parts)
        
        system = """Quickly assess if this answer is factually consistent with the context.

Return JSON:
{
  "is_factual": true/false,
  "confidence": 0.0-1.0,
  "major_issues": ["list of major factual issues if any"]
}"""

        user = f"""Answer:
{answer}

Context:
{context}

Return JSON only."""

        result, response = self._llm.chat_json(
            system=system,
            user=user,
            default={"is_factual": True, "confidence": 0.5},
            expected_type=dict,
        )
        
        return (
            bool(result.get("is_factual", True)),
            float(result.get("confidence", 0.5))
        )
