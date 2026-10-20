"""
Unit tests for Phase 3 Mitigation Tools.
Tests:
- Retrieval with evidence gating and thresholding.
- Interactive clarification turn initiation and resolution.
- Claim verification with numerical and entity contradiction detection.
- Iterative answer revision and re-verification loop.
- Insufficient-evidence abstention.
"""

import unittest
from src.retrieval import LocalCorpusRetriever, RetrievedPassage
from src.verification import EvidenceVerifier, VerificationStatus, ClaimStatus
from src.revision import AnswerReviser, RevisionResult
from src.clarification import ClarificationHandler, ClarificationSession
from src.signals import SignalExtractor
from src.controller import MitigationController


class TestMitigationToolsPhase3(unittest.TestCase):

    def setUp(self):
        self.evidence = [
            RetrievedPassage(
                passage_id="test_everest_p1",
                doc_id="doc_geo_everest",
                doc_title="Mount Everest Elevation and Geography",
                source_url_or_ref="Survey 2020",
                text="The official elevation of Mount Everest was jointly revised and announced by Nepal and China in December 2020 as 8,848.86 meters (29,031.7 feet).",
                relevance_score=0.95
            ),
            RetrievedPassage(
                passage_id="test_light_p1",
                doc_id="doc_sci_light",
                doc_title="Speed of Light in Vacuum",
                source_url_or_ref="CODATA 2018",
                text="The speed of light in vacuum, commonly denoted c, is a universal physical constant. Its exact value is defined as 299,792,458 meters per second.",
                relevance_score=0.98
            )
        ]

    def test_clarification_dialogue_initiation_and_resolution(self):
        signals = SignalExtractor.extract("Which one is better?")
        session = ClarificationHandler.initiate_clarification(signals)

        self.assertFalse(session.is_resolved)
        self.assertIn("Which alternatives", session.clarification_prompt)
        self.assertGreater(len(session.suggested_options), 0)

        # Resolve clarification with user feedback
        resolved = ClarificationHandler.resolve_clarification(
            session=session,
            user_response="comparing PyTorch vs TensorFlow for production inference latency"
        )
        self.assertTrue(resolved.is_resolved)
        self.assertEqual(resolved.interaction_turns, 2)
        self.assertIn("production inference latency", resolved.disambiguated_query)

    def test_numerical_entity_contradiction_detection(self):
        # Claim contains a contradicted number (9,500m instead of 8,848.86m)
        hallucinated_answer = "Mount Everest has an elevation of 9,500 meters above sea level."
        res = EvidenceVerifier.verify(hallucinated_answer, [self.evidence[0]])

        self.assertEqual(res.overall_status, VerificationStatus.UNSUPPORTED)
        self.assertEqual(res.contradicted_claims_count, 1)
        self.assertEqual(res.claims[0].status, ClaimStatus.CONTRADICTED)
        self.assertIn("Numerical contradiction", res.claims[0].rationale)

    def test_verbatim_quote_attribution(self):
        grounded_answer = "The speed of light in vacuum is defined as 299,792,458 meters per second."
        res = EvidenceVerifier.verify(grounded_answer, [self.evidence[1]])

        self.assertEqual(res.overall_status, VerificationStatus.VERIFIED)
        self.assertEqual(res.supported_claims_count, 1)
        self.assertIsNotNone(res.claims[0].supporting_quote)
        self.assertIn("299,792,458", res.claims[0].supporting_quote)

    def test_iterative_revision_and_reverification_loop(self):
        # Answer with one correct claim and one contradicted assertion
        mixed_answer = (
            "Mount Everest has an official elevation of 8,848.86 meters. "
            "It was measured to be 15,000 meters high during the Roman empire."
        )
        ver_res = EvidenceVerifier.verify(mixed_answer, [self.evidence[0]])
        self.assertEqual(ver_res.overall_status, VerificationStatus.UNSUPPORTED)

        # Run revision
        rev_res = AnswerReviser.revise(mixed_answer, ver_res, [self.evidence[0]])
        self.assertTrue(rev_res.is_revised)
        self.assertIn("8,848.86", rev_res.revised_answer)
        self.assertNotIn("15,000", rev_res.revised_answer)
        # Re-verification must have run and confirmed the revision
        self.assertEqual(rev_res.re_verification_status, VerificationStatus.VERIFIED.value)

    def test_insufficient_evidence_abstention(self):
        controller = MitigationController(tradeoff_lambda=0.5)
        # Query for a topic definitively absent from local corpus
        res = controller.process_query("What is the average rainfall in the Atacama desert in 2023?")
        # When evidence is absent in adaptive retrieval, it should abstain with explicit notice
        self.assertIn("Abstention Notice", res.final_answer)
        self.assertIn("does not contain sufficient authoritative passages", res.final_answer)

    def test_multi_turn_clarification_in_controller(self):
        controller = MitigationController(tradeoff_lambda=0.5)
        # First turn triggers clarify
        res1 = controller.process_query("Which one is better?")
        self.assertEqual(res1.action_executed, "clarify")
        self.assertIsNotNone(res1.clarification_session)

        # Second turn resolves clarification
        res2 = controller.resolve_clarification_turn(
            session=res1.clarification_session,
            user_response="Calculate 14 * 25"
        )
        self.assertTrue(res2.clarification_session.is_resolved)
        self.assertEqual(res2.action_executed, "calculate")
        self.assertIn("350", res2.final_answer)


if __name__ == "__main__":
    unittest.main()
