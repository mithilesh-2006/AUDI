"""
Answer Revision Engine (Phase 3).

Revises draft responses based on evidence verification outcomes:
- Updates only claims identified as unsupported or contradicted.
- Retains verified claims and preserves factual grounding.
- Performs re-verification check on the revised text to confirm grounding.
- Synthesizes grounded statements citing source documents.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from src.retrieval import RetrievedPassage
from src.verification import EvidenceVerifier, VerificationResult, VerificationStatus, ClaimStatus


@dataclass
class RevisionResult:
    """Outcome of answer revision and subsequent re-verification."""
    original_answer: str
    revised_answer: str
    modifications_made: List[str]
    is_revised: bool
    re_verification_status: str
    re_verification_result: Optional[VerificationResult] = None
    revision_iteration: int = 1
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_answer": self.original_answer,
            "revised_answer": self.revised_answer,
            "modifications_made": self.modifications_made,
            "is_revised": self.is_revised,
            "re_verification_status": self.re_verification_status,
            "re_verification_result": self.re_verification_result.to_dict() if self.re_verification_result else None,
            "revision_iteration": self.revision_iteration,
            "latency_ms": round(self.latency_ms, 2),
        }


class AnswerReviser:
    """Revises draft answers and performs automated re-verification checks."""

    @classmethod
    def revise(
        cls,
        draft_answer: str,
        verification_result: VerificationResult,
        evidence_passages: List[RetrievedPassage],
        max_iterations: int = 2,
    ) -> RevisionResult:
        """Produce a revised, strictly grounded answer and verify it iteratively."""
        start_time = time.perf_counter()

        # If everything was verified, no revision needed
        if verification_result.unsupported_claims_count == 0 and verification_result.total_claims > 0:
            return RevisionResult(
                original_answer=draft_answer,
                revised_answer=draft_answer,
                modifications_made=["No revision required: all claims verified against evidence."],
                is_revised=False,
                re_verification_status=VerificationStatus.VERIFIED.value,
                re_verification_result=verification_result,
                revision_iteration=0,
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        supported_claims = [
            c.claim_text for c in verification_result.claims
            if c.status == ClaimStatus.SUPPORTED
        ]
        contradicted_claims = [
            c.claim_text for c in verification_result.claims
            if c.status == ClaimStatus.CONTRADICTED
        ]
        unsupported_claims = [
            c.claim_text for c in verification_result.claims
            if c.status in (ClaimStatus.UNSUPPORTED, ClaimStatus.INSUFFICIENT_EVIDENCE)
        ]

        modifications = []
        revised_paragraphs = []

        if supported_claims:
            revised_paragraphs.append(" ".join(supported_claims))
            modifications.append(f"Retained {len(supported_claims)} verified claim(s).")

        if contradicted_claims:
            modifications.append(
                f"Redacted {len(contradicted_claims)} contradicted assertion(s): "
                + "; ".join([f"'{c[:45]}...'" for c in contradicted_claims])
            )

        if unsupported_claims:
            modifications.append(
                f"Excised {len(unsupported_claims)} ungrounded assertion(s): "
                + "; ".join([f"'{c[:45]}...'" for c in unsupported_claims])
            )

        # Grounding with authoritative corpus text
        if evidence_passages and evidence_passages[0].relevance_score >= 0.15:
            top_source = evidence_passages[0]
            # If no claims were supported, synthesize direct answer from top passage
            if not supported_claims:
                first_sentence = top_source.text.split(".")[0].strip() + "."
                revised_paragraphs.append(first_sentence)
                modifications.append(f"Substituted ungrounded assertions with verified sentence from '{top_source.doc_title}'.")
            revised_paragraphs.append(f"[Source: {top_source.doc_title} ({top_source.source_url_or_ref})]")
        else:
            if not supported_claims:
                revised_paragraphs.append(
                    "Reliable confirmation could not be found in the local evidence corpus for this inquiry. "
                    "To prevent hallucination, the system abstains from asserting ungrounded claims."
                )
                modifications.append("Replaced response with grounded abstention notice due to absence of evidence.")

        revised_text = "\n\n".join(revised_paragraphs)

        # Re-verification loop (check the revised answer against evidence)
        re_ver = EvidenceVerifier.verify(revised_text, evidence_passages)
        re_status = re_ver.overall_status.value

        latency = (time.perf_counter() - start_time) * 1000.0
        return RevisionResult(
            original_answer=draft_answer,
            revised_answer=revised_text,
            modifications_made=modifications,
            is_revised=True,
            re_verification_status=re_status,
            re_verification_result=re_ver,
            revision_iteration=1,
            latency_ms=latency,
        )
