"""
Unit tests for Claim Verification and Answer Revision.
Verifies claim decomposition, status assignment (Verified, Partially Supported,
Unsupported, Not Checked), and accurate text revision.
"""

import unittest
from src.retrieval import RetrievedPassage
from src.verification import EvidenceVerifier, VerificationStatus, ClaimStatus
from src.revision import AnswerReviser


class TestVerificationAndRevision(unittest.TestCase):

    def setUp(self):
        self.evidence = [
            RetrievedPassage(
                passage_id="p1",
                doc_id="doc1",
                doc_title="Mount Everest Elevation",
                source_url_or_ref="Survey 2020",
                text="Mount Everest official elevation is 8,848.86 meters above sea level. It is located in the Himalayas.",
                relevance_score=0.9
            )
        ]

    def test_claim_decomposition(self):
        text = "Mount Everest is 8,848.86 meters high. It is in Nepal and China. The weather is cold."
        claims = EvidenceVerifier.decompose_claims(text)
        self.assertEqual(len(claims), 3)

    def test_full_verification_success(self):
        answer = "Mount Everest elevation is 8,848.86 meters above sea level."
        res = EvidenceVerifier.verify(answer, self.evidence)
        self.assertEqual(res.overall_status, VerificationStatus.VERIFIED)
        self.assertEqual(res.supported_claims_count, 1)
        self.assertEqual(res.unsupported_claims_count, 0)

    def test_partially_supported_verification(self):
        answer = (
            "Mount Everest official elevation is 8,848.86 meters. "
            "It was first climbed by ancient Roman soldiers in 44 BC."
        )
        res = EvidenceVerifier.verify(answer, self.evidence)
        self.assertEqual(res.overall_status, VerificationStatus.PARTIALLY_SUPPORTED)
        self.assertEqual(res.supported_claims_count, 1)
        self.assertEqual(res.unsupported_claims_count, 1)

    def test_unsupported_verification(self):
        answer = "The Eiffel Tower was constructed entirely out of solid titanium in 1999."
        res = EvidenceVerifier.verify(answer, self.evidence)
        self.assertEqual(res.overall_status, VerificationStatus.UNSUPPORTED)
        self.assertEqual(res.supported_claims_count, 0)

    def test_revision_procedure(self):
        answer = (
            "Mount Everest official elevation is 8,848.86 meters above sea level. "
            "It was first climbed by Roman gladiators in 100 AD."
        )
        ver_res = EvidenceVerifier.verify(answer, self.evidence)
        rev_res = AnswerReviser.revise(answer, ver_res, self.evidence)

        self.assertTrue(rev_res.is_revised)
        self.assertIn("8,848.86", rev_res.revised_answer)
        self.assertNotIn("gladiators", rev_res.revised_answer)
        self.assertIn("Mount Everest Elevation", rev_res.revised_answer)


    def test_insufficient_evidence_label(self):
        """Test that ungrounded assertions receive INSUFFICIENT_EVIDENCE rather than false/contradicted."""
        answer = "The atmospheric pressure at the summit of K2 varies seasonally."
        res = EvidenceVerifier.verify(answer, self.evidence)
        self.assertEqual(len(res.claims), 1)
        self.assertEqual(res.claims[0].status, ClaimStatus.INSUFFICIENT_EVIDENCE)
        self.assertIn("absence of local evidence does not imply", res.claims[0].rationale)

    def test_not_checkable_label(self):
        """Test that non-content, conversational assertions receive NOT_CHECKABLE."""
        answer = "In general, it is what it is."
        res = EvidenceVerifier.verify(answer, self.evidence)
        self.assertEqual(res.claims[0].status, ClaimStatus.NOT_CHECKABLE)


if __name__ == "__main__":
    unittest.main()
