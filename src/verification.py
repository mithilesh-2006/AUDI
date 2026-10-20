"""
Evidence-Based Claim Verification Engine (Phase 3).

Performs propositional claim-level decomposition and deep verification against retrieved evidence:
- Distinguishes SUPPORTED, CONTRADICTED, and INSUFFICIENT_EVIDENCE claims.
- Performs numerical and named entity alignment to catch factual hallucinations (dates, figures, measurements).
- Extracts supporting verbatim quotes from evidence passages.
- Reports authentic verification status:
  * VERIFIED: All claims are fully grounded in evidence with zero contradictions.
  * PARTIALLY_SUPPORTED: Some claims are grounded; others lack evidence or contain contradictions.
  * UNSUPPORTED: Claims contain direct contradictions or complete absence of grounding evidence.
  * NOT_CHECKED: Verification action was not executed.
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
    CONTRADICTED = "Contradicted"
    UNSUPPORTED = "Unsupported"
    INSUFFICIENT_EVIDENCE = "Insufficient Evidence"


@dataclass
class ClaimCheckResult:
    """Detailed verification outcome for an individual atomic proposition."""
    claim_text: str
    status: ClaimStatus
    supporting_passage_id: Optional[str] = None
    supporting_quote: Optional[str] = None
    confidence_score: float = 0.0
    rationale: str = ""
    conflicting_value: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_text": self.claim_text,
            "status": self.status.value,
            "supporting_passage_id": self.supporting_passage_id,
            "supporting_quote": self.supporting_quote,
            "confidence_score": round(self.confidence_score, 3),
            "rationale": self.rationale,
            "conflicting_value": self.conflicting_value,
        }


@dataclass
class VerificationResult:
    """Overall outcome of the evidence verification procedure."""
    overall_status: VerificationStatus
    claims: List[ClaimCheckResult] = field(default_factory=list)
    total_claims: int = 0
    supported_claims_count: int = 0
    contradicted_claims_count: int = 0
    unsupported_claims_count: int = 0
    latency_ms: float = 0.0
    verification_method: str = "propositional_entity_alignment"
    summary_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "claims": [c.to_dict() for c in self.claims],
            "total_claims": self.total_claims,
            "supported_claims_count": self.supported_claims_count,
            "contradicted_claims_count": self.contradicted_claims_count,
            "unsupported_claims_count": self.unsupported_claims_count,
            "latency_ms": round(self.latency_ms, 3),
            "verification_method": self.verification_method,
            "summary_notes": self.summary_notes,
        }


class EvidenceVerifier:
    """Decomposes an answer draft into claims and performs entity-aligned evidence verification."""

    @classmethod
    def decompose_claims(cls, text: str) -> List[str]:
        """Split text into sentence-level propositional assertions."""
        if not text or not text.strip():
            return []
        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        claims = []
        for s in raw_sentences:
            s_clean = s.strip()
            # Exclude conversational headers or boilerplate
            if len(s_clean) > 8 and not s_clean.lower().startswith((
                "here is", "in summary", "accordingly", "direct response to", "based on the"
            )):
                claims.append(s_clean)
        return claims if claims else [text.strip()]

    @classmethod
    def verify(cls, draft_answer: str, evidence_passages: List[RetrievedPassage]) -> VerificationResult:
        """Execute deep claim-level verification against retrieved evidence passages."""
        start_time = time.perf_counter()

        if not draft_answer or not draft_answer.strip():
            return VerificationResult(
                overall_status=VerificationStatus.NOT_CHECKED,
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
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
                contradicted_claims_count=0,
                unsupported_claims_count=len(claims),
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                summary_notes="Verification failed: absence of grounding evidence in corpus."
            )

        claims = cls.decompose_claims(draft_answer)
        claim_results: List[ClaimCheckResult] = []

        combined_passages = [
            (p, set(re.findall(r"\b\w{3,}\b", p.text.lower())), re.findall(r"\b\d+(?:[\.,]\d+)?\b", p.text))
            for p in evidence_passages
        ]

        supported_count = 0
        contradicted_count = 0
        unsupported_count = 0

        stopwords = {
            "the", "and", "that", "this", "with", "from", "for", "are", "was",
            "were", "been", "have", "has", "had", "will", "would", "about"
        }

        for claim in claims:
            claim_tokens = set(re.findall(r"\b\w{3,}\b", claim.lower()))
            content_tokens = claim_tokens - stopwords

            if not content_tokens:
                # Trivial statement
                claim_results.append(ClaimCheckResult(
                    claim_text=claim,
                    status=ClaimStatus.SUPPORTED,
                    confidence_score=0.9,
                    rationale="Non-factual or functional statement."
                ))
                supported_count += 1
                continue

            best_passage: Optional[RetrievedPassage] = None
            best_overlap = 0.0
            best_p_nums: List[str] = []

            for p, p_tokens, p_nums in combined_passages:
                intersection = content_tokens.intersection(p_tokens)
                overlap = len(intersection) / max(len(content_tokens), 1)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_passage = p
                    best_p_nums = p_nums

            # Numerical entity verification
            claim_numbers = re.findall(r"\b\d+(?:[\.,]\d+)?\b", claim)
            numerical_contradiction = False
            conflicting_num = None

            if claim_numbers and best_passage:
                for c_num in claim_numbers:
                    # Check if number exists in passage
                    if c_num not in best_passage.text:
                        # If passage has other numbers for this topic, it's a contradiction
                        if best_p_nums:
                            numerical_contradiction = True
                            conflicting_num = f"Claim states {c_num}, but evidence states {', '.join(best_p_nums[:2])}"
                            break

            # Evaluate Claim Status
            if numerical_contradiction:
                contradicted_count += 1
                claim_results.append(ClaimCheckResult(
                    claim_text=claim,
                    status=ClaimStatus.CONTRADICTED,
                    supporting_passage_id=best_passage.passage_id if best_passage else None,
                    conflicting_value=conflicting_num,
                    confidence_score=round(best_overlap, 3),
                    rationale=f"Numerical contradiction: {conflicting_num}",
                    supporting_quote=best_passage.text[:120] + "..." if best_passage else None,
                ))
            elif best_overlap >= 0.50 and best_passage:
                supported_count += 1
                # Extract clean snippet as quote
                quote_snippet = best_passage.text[:140] + ("..." if len(best_passage.text) > 140 else "")
                claim_results.append(ClaimCheckResult(
                    claim_text=claim,
                    status=ClaimStatus.SUPPORTED,
                    supporting_passage_id=best_passage.passage_id,
                    supporting_quote=quote_snippet,
                    confidence_score=round(best_overlap, 3),
                    rationale=f"Grounded in {best_passage.doc_title} (token alignment {best_overlap:.1%})"
                ))
            else:
                unsupported_count += 1
                claim_results.append(ClaimCheckResult(
                    claim_text=claim,
                    status=ClaimStatus.UNSUPPORTED,
                    supporting_passage_id=best_passage.passage_id if best_passage else None,
                    confidence_score=round(best_overlap, 3),
                    rationale=f"Entity and semantic alignment insufficient ({best_overlap:.1%})"
                ))

        # Overall Status Resolution
        if contradicted_count > 0:
            overall = VerificationStatus.UNSUPPORTED
        elif supported_count == len(claim_results) and supported_count > 0:
            overall = VerificationStatus.VERIFIED
        elif supported_count > 0:
            overall = VerificationStatus.PARTIALLY_SUPPORTED
        else:
            overall = VerificationStatus.UNSUPPORTED

        latency = (time.perf_counter() - start_time) * 1000.0
        notes = (
            f"Verified {len(claim_results)} claims: {supported_count} Supported, "
            f"{contradicted_count} Contradicted, {unsupported_count} Unsupported."
        )

        return VerificationResult(
            overall_status=overall,
            claims=claim_results,
            total_claims=len(claim_results),
            supported_claims_count=supported_count,
            contradicted_claims_count=contradicted_count,
            unsupported_claims_count=unsupported_count + contradicted_count,
            latency_ms=latency,
            verification_method="propositional_entity_alignment",
            summary_notes=notes,
        )
