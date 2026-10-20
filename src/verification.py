"""
Evidence-Based Claim Verification Engine.

Performs claim-level decomposition and verifies factual claims against retrieved evidence passages.
Ensures transparent status reporting:
- VERIFIED: All claims are fully grounded in retrieved evidence.
- PARTIALLY_SUPPORTED: Some claims are grounded; others lack evidence.
- UNSUPPORTED: Claims are contradicted or ungrounded in evidence.
- NOT_CHECKED: Verification action was not executed.
"""

from __future__ import annotations
import enum
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from src.retrieval import RetrievedPassage


class VerificationStatus(str, enum.Enum):
    VERIFIED = "Verified"
    PARTIALLY_SUPPORTED = "Partially Supported"
    UNSUPPORTED = "Unsupported"
    NOT_CHECKED = "Not Checked"


class ClaimStatus(str, enum.Enum):
    SUPPORTED = "Supported"
    UNSUPPORTED = "Unsupported"
    INSUFFICIENT_EVIDENCE = "Insufficient Evidence"


@dataclass
class ClaimCheckResult:
    """Verification outcome for an individual atomic claim."""
    claim_text: str
    status: ClaimStatus
    supporting_passage_id: Optional[str] = None
    supporting_quote: Optional[str] = None
    confidence_score: float = 0.0
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_text": self.claim_text,
            "status": self.status.value,
            "supporting_passage_id": self.supporting_passage_id,
            "supporting_quote": self.supporting_quote,
            "confidence_score": round(self.confidence_score, 3),
            "rationale": self.rationale,
        }


@dataclass
class VerificationResult:
    """Overall outcome of the evidence verification procedure."""
    overall_status: VerificationStatus
    claims: List[ClaimCheckResult] = field(default_factory=list)
    total_claims: int = 0
    supported_claims_count: int = 0
    unsupported_claims_count: int = 0
    latency_ms: float = 0.0
    verification_method: str = "lexical_evidence_alignment"
    summary_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "claims": [c.to_dict() for c in self.claims],
            "total_claims": self.total_claims,
            "supported_claims_count": self.supported_claims_count,
            "unsupported_claims_count": self.unsupported_claims_count,
            "latency_ms": round(self.latency_ms, 3),
            "verification_method": self.verification_method,
            "summary_notes": self.summary_notes,
        }


class EvidenceVerifier:
    """Decomposes an answer draft into claims and checks them against retrieved evidence."""

    @classmethod
    def decompose_claims(cls, text: str) -> List[str]:
        """Split text into sentence-level propositional assertions."""
        if not text or not text.strip():
            return []
        # Split on sentence boundaries, keeping non-empty assertions
        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        claims = []
        for s in raw_sentences:
            s_clean = s.strip()
            # Exclude conversational fluff or non-assertive headers
            if len(s_clean) > 8 and not s_clean.lower().startswith(("here is", "in summary", "accordingly")):
                claims.append(s_clean)
        return claims if claims else [text.strip()]

    @classmethod
    def verify(cls, draft_answer: str, evidence_passages: List[RetrievedPassage]) -> VerificationResult:
        """Execute claim-level verification against evidence passages."""
        start_time = time.perf_counter()

        if not draft_answer or not draft_answer.strip():
            return VerificationResult(
                overall_status=VerificationStatus.NOT_CHECKED,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                summary_notes="No draft text provided to verify."
            )

        if not evidence_passages:
            # If no evidence was retrieved or provided, claims cannot be verified against evidence
            claims = cls.decompose_claims(draft_answer)
            claim_results = [
                ClaimCheckResult(
                    claim_text=c,
                    status=ClaimStatus.INSUFFICIENT_EVIDENCE,
                    rationale="No external evidence passages were available to verify this assertion."
                )
                for c in claims
            ]
            return VerificationResult(
                overall_status=VerificationStatus.UNSUPPORTED,
                claims=claim_results,
                total_claims=len(claims),
                supported_claims_count=0,
                unsupported_claims_count=len(claims),
                latency_ms=(time.perf_counter() - start_time) * 1000,
                summary_notes="Verification failed due to absence of grounding evidence."
            )

        claims = cls.decompose_claims(draft_answer)
        claim_results: List[ClaimCheckResult] = []

        # Tokenize passages
        combined_passages = [
            (p, set(re.findall(r"\b\w{3,}\b", p.text.lower())))
            for p in evidence_passages
        ]

        supported_count = 0
        unsupported_count = 0

        for claim in claims:
            claim_tokens = set(re.findall(r"\b\w{3,}\b", claim.lower()))
            # Remove high-frequency stopwords from matching
            stopwords = {"the", "and", "that", "this", "with", "from", "for", "are", "was", "were", "been"}
            content_tokens = claim_tokens - stopwords

            if not content_tokens:
                # Trivial sentence
                claim_results.append(ClaimCheckResult(
                    claim_text=claim,
                    status=ClaimStatus.SUPPORTED,
                    confidence_score=0.9,
                    rationale="Non-factual or functional statement."
                ))
                supported_count += 1
                continue

            best_match_passage: Optional[RetrievedPassage] = None
            best_overlap = 0.0

            for passage, p_tokens in combined_passages:
                intersection = content_tokens.intersection(p_tokens)
                overlap_ratio = len(intersection) / len(content_tokens)
                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_match_passage = passage

            # Also check numerical and date matches specifically to catch subtle hallucinations
            numbers_in_claim = re.findall(r"\b\d+(?:[\.,]\d+)?\b", claim)
            numbers_grounded = True
            if numbers_in_claim and best_match_passage:
                for num in numbers_in_claim:
                    if num not in best_match_passage.text:
                        # Claim contains specific number/year not in evidence
                        numbers_grounded = False
                        break

            # Determine claim status
            if best_overlap >= 0.55 and numbers_grounded and best_match_passage:
                claim_results.append(ClaimCheckResult(
                    claim_text=claim,
                    status=ClaimStatus.SUPPORTED,
                    supporting_passage_id=best_match_passage.passage_id,
                    supporting_quote=best_match_passage.text[:120] + "...",
                    confidence_score=round(best_overlap, 3),
                    rationale=f"Verified against {best_match_passage.doc_title} (token alignment {best_overlap:.1%})"
                ))
                supported_count += 1
            else:
                reason = "Key entities or assertions lack evidence support"
                if not numbers_grounded:
                    reason = "Numerical figures or dates mentioned in claim are not supported by evidence"
                claim_results.append(ClaimCheckResult(
                    claim_text=claim,
                    status=ClaimStatus.UNSUPPORTED,
                    supporting_passage_id=best_match_passage.passage_id if best_match_passage else None,
                    confidence_score=round(best_overlap, 3),
                    rationale=f"{reason} (best alignment {best_overlap:.1%})"
                ))
                unsupported_count += 1

        # Determine overall verification status
        if supported_count == len(claim_results) and supported_count > 0:
            overall = VerificationStatus.VERIFIED
        elif supported_count > 0 and unsupported_count > 0:
            overall = VerificationStatus.PARTIALLY_SUPPORTED
        else:
            overall = VerificationStatus.UNSUPPORTED

        latency = (time.perf_counter() - start_time) * 1000
        notes = f"{supported_count}/{len(claim_results)} claims verified against local evidence."

        return VerificationResult(
            overall_status=overall,
            claims=claim_results,
            total_claims=len(claim_results),
            supported_claims_count=supported_count,
            unsupported_claims_count=unsupported_count,
            latency_ms=latency,
            verification_method="lexical_evidence_alignment",
            summary_notes=notes,
        )
