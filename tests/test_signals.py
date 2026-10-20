"""
Unit tests for SignalExtractor.
Verifies arithmetic detection, ambiguity cues, factual triggers, and speculation cues.
"""

import unittest
from src.signals import SignalExtractor, QuerySignals


class TestSignalExtractor(unittest.TestCase):

    def test_empty_query(self):
        signals = SignalExtractor.extract("")
        self.assertTrue(signals.is_empty)
        self.assertTrue(signals.is_ambiguous)
        self.assertEqual(signals.heuristic_uncertainty_score, 1.0)

        signals_space = SignalExtractor.extract("    ")
        self.assertTrue(signals_space.is_empty)

    def test_arithmetic_detection(self):
        signals = SignalExtractor.extract("Calculate 14 * 25")
        self.assertTrue(signals.has_arithmetic_expr)
        self.assertIn("14 * 25", signals.extracted_arithmetic_expr)
        self.assertLessEqual(signals.heuristic_uncertainty_score, 0.20)

        signals2 = SignalExtractor.extract("what is (120 + 45) / 5?")
        self.assertTrue(signals2.has_arithmetic_expr)

    def test_ambiguity_detection(self):
        ambiguous_queries = [
            "Which one is better?",
            "Is it good?",
            "Why did it happen?",
            "compare them",
            "tell me more",
            "what about that?"
        ]
        for q in ambiguous_queries:
            with self.subTest(query=q):
                signals = SignalExtractor.extract(q)
                self.assertTrue(signals.is_ambiguous, f"Failed to detect ambiguity for '{q}'")
                self.assertIsNotNone(signals.ambiguity_reason)
                self.assertGreaterEqual(signals.heuristic_uncertainty_score, 0.70)

    def test_speculation_detection(self):
        speculative_queries = [
            "Will Bitcoin hit $500,000 in 2035?",
            "What am I thinking right now?",
            "Who will win the 2032 world cup?",
            "predict the future of artificial intelligence in 2040",
        ]
        for q in speculative_queries:
            with self.subTest(query=q):
                signals = SignalExtractor.extract(q)
                self.assertTrue(signals.is_unknowable_or_speculative, f"Failed to detect speculation for '{q}'")
                self.assertGreaterEqual(signals.heuristic_uncertainty_score, 0.85)

    def test_factual_knowledge_detection(self):
        factual_queries = [
            "What is the elevation of Mount Everest?",
            "What is the speed of light in vacuum?",
            "When did Apollo 11 land on the Moon?",
            "Who was awarded the Nobel Prize in Physics in 1921?",
        ]
        for q in factual_queries:
            with self.subTest(query=q):
                signals = SignalExtractor.extract(q)
                self.assertTrue(signals.is_factual_knowledge_request, f"Failed to detect factual request for '{q}'")
                self.assertTrue(len(signals.factual_keywords_detected) > 0)


if __name__ == "__main__":
    unittest.main()
