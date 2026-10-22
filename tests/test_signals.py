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

    def test_deterministic_directive_and_brief_queries(self):
        """Ensure concise deterministic commands and quoted entities are not falsely flagged as ambiguous."""
        actionable_directives = [
            'spell "hello"',
            'spell hello',
            'define entropy',
            'translate dog',
            'say hello',
            'echo test',
            'reverse cat',
            'capitalize python',
            'repeat after me',
            'hello',
            'hi',
        ]
        for q in actionable_directives:
            with self.subTest(query=q):
                signals = SignalExtractor.extract(q)
                self.assertFalse(signals.is_ambiguous, f"Directive '{q}' was falsely marked as ambiguous")
                self.assertLessEqual(signals.heuristic_uncertainty_score, 0.35)

    def test_antigravity_fallback_interceptor_floating_dual_referents(self):
        """Test Antigravity Interceptor on floating dual referents with comparative triggers."""
        unanchored_queries = [
            "compare the two",
            "which of the options is better",
            "accuracy between both",
            "what is the difference between the pair",
            "tell me the difference between the alternatives",
            "which of these two is faster",
            "compare both",
        ]
        for q in unanchored_queries:
            with self.subTest(query=q):
                signals = SignalExtractor.extract(q)
                self.assertTrue(signals.is_ambiguous, f"Failed to mark '{q}' as ambiguous")
                self.assertTrue(signals.antigravity_intercepted, f"Failed to intercept '{q}'")
                self.assertEqual(signals.heuristic_uncertainty_score, 0.85)
                self.assertIn("Antigravity Fallback Interceptor", signals.ambiguity_reason)

    def test_antigravity_fallback_interceptor_short_comparatives(self):
        """Test Antigravity Lift on brief comparative queries (<7 tokens) lacking concrete proper nouns."""
        brief_comparatives = [
            "Which is better?",
            "Is one worse?",
            "compare results",
            "what's the difference?",
            "which is faster?",
        ]
        for q in brief_comparatives:
            with self.subTest(query=q):
                signals = SignalExtractor.extract(q)
                self.assertTrue(signals.is_ambiguous, f"Failed to mark '{q}' as ambiguous")
                self.assertTrue(signals.antigravity_intercepted, f"Failed to intercept '{q}'")
                self.assertEqual(signals.heuristic_uncertainty_score, 0.85)

    def test_antigravity_interceptor_anchored_comparatives_bypass(self):
        """Test that comparative queries with concrete proper nouns/entities bypass the Antigravity Lift."""
        grounded_comparatives = [
            "Compare Python and Java",
            "What is the difference between Rust and C++?",
            "compare the two algorithms Dijkstra and A*",
            "accuracy between 'Model A' and 'Model B'",
        ]
        for q in grounded_comparatives:
            with self.subTest(query=q):
                signals = SignalExtractor.extract(q)
                self.assertFalse(
                    signals.antigravity_intercepted,
                    f"Grounded query '{q}' should NOT be intercepted by Antigravity"
                )

    def test_strict_regex_sanitization_no_collisions(self):
        """Test that strict word-boundary regexes prevent partial substring collisions."""
        # 1. 'temp' substring must NOT trigger 'temperature of'
        sig_temp = SignalExtractor.extract("Bitcoin temp attempt drop")
        self.assertFalse(sig_temp.is_factual_knowledge_request)
        self.assertNotIn("temperature of", sig_temp.factual_keywords_detected)

        # 2. 'history' must NOT trigger 'hi' greeting
        sig_hist = SignalExtractor.extract("history of ancient rome")
        self.assertFalse(sig_hist.is_ambiguous)

        # 3. Genuine factual keyword with boundaries DOES trigger
        sig_fact = SignalExtractor.extract("What is the average temperature of Venus?")
        self.assertTrue(sig_fact.is_factual_knowledge_request)
        self.assertIn("temperature of", sig_fact.factual_keywords_detected)

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
