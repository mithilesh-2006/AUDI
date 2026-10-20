"""
Answer Revision Engine.

Revises draft responses based on evidence verification outcomes:
- Retains verified claims
- Excises or hedges ungrounded claims
- Synthesizes grounded statements citing source documents
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, Any, List
from src.retrieval import RetrievedPassage
from src.verification import VerificationResult, ClaimStatus


@dataclass
class RevisionResult:
    """Outcome of answer revision."""
    original_answer: str
    revised_answer: str
    modifications_made: List[str]
    is_revised: bool
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_answer": self.original_answer,
            "revised_answer": self.revised_answer,
            "modifications_made": self.modifications_made,
            "is_revised": self.is_revised,
            "latency_ms": round(self.latency_ms, 3),
        }


class AnswerReviser:
    """Revises draft answers to remove or qualify ungrounded claims."""

    @classmethod
    def revise(
        cls,
        draft_answer: str,
        verification_result: VerificationResult,
        evidence_passages: List[RetrievedPassage]
    ) -> RevisionResult:
        """Produce a revised, strictly grounded answer."""
        start_time = time.perf_counter()

        # If everything was verified, no revision needed
        if verification_result.unsupported_claims_count == 0 and verification_result.total_claims > 0:
            return RevisionResult(
                original_answer=draft_answer,
                revised_answer=draft_answer,
                modifications_made=["No revision required: all claims verified against evidence."],
                is_revised=False,
                latency_ms=(time.perf_counter() - start_time) * 1000
            )

        supported_claims = [
            c.claim_text for c in verification_result.claims
            if c.status == ClaimStatus.SUPPORTED
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

        if unsupported_claims:
            modifications.append(
                f"Removed or flagged {len(unsupported_claims)} unsupported assertion(s): "
                + "; ".join([f"'{c[:40]}...'" for c in unsupported_claims])
            )

        # Append source citation if evidence exists
        if evidence_passages:
            top_source = evidence_passages[0]
            revised_paragraphs.append(f"\n[Source: {top_source.doc_title} ({top_source.source_url_or_ref})]")

        if not supported_claims:
            # Complete absence of supported claims
            revised_text = (
                "Based on the available evidence in the verified corpus, reliable confirmation "
                "could not be found for the initial assertion. The claim is ungrounded."
            )
            modifications.append("Replaced entire response with grounded disclaimer due to zero evidence support.")
        else:
            revised_text = "\n".join(revised_paragraphs)

        latency = (time.perf_counter() - start_time) * 1000
        return RevisionResult(
            original_answer=draft_answer,
            revised_answer=revised_text,
            modifications_made=modifications,
            is_revised=True,
            latency_ms=latency,
        )
