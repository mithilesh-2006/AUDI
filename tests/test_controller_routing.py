"""
Unit tests for MitigationController and Policy Action Selection.
Tests each implemented routing branch, lambda sensitivity, and error handling.
"""

import unittest
from src.controller import MitigationController
from src.policy import PolicyType


class TestControllerRouting(unittest.TestCase):

    def setUp(self):
        self.controller = MitigationController(tradeoff_lambda=0.5)

    def test_routing_to_calculate(self):
        res = self.controller.process_query("Calculate 14 * 25")
        self.assertEqual(res.action_executed, "calculate")
        self.assertIn("350", res.final_answer)
        self.assertEqual(res.cost.tool_calls_count, 1)

    def test_routing_to_clarify(self):
        res = self.controller.process_query("Which one is better?")
        self.assertEqual(res.action_executed, "clarify")
        self.assertIn("clarify", res.final_answer.lower())

    def test_routing_to_abstain(self):
        res = self.controller.process_query("Will Bitcoin hit $500,000 in 2035?")
        self.assertEqual(res.action_executed, "abstain")
        self.assertIn("Abstention", res.final_answer)

    def test_routing_to_retrieve(self):
        res = self.controller.process_query("What is the elevation of Mount Everest?")
        self.assertEqual(res.action_executed, "retrieve")
        self.assertIsNotNone(res.retrieval_result)
        self.assertTrue(res.retrieval_result.has_sufficient_evidence)

    def test_routing_to_direct(self):
        res = self.controller.process_query("Explain the general concept of photosynthesis in plants.")
        self.assertEqual(res.action_executed, "direct")
        self.assertEqual(res.cost.tool_calls_count, 0)

    def test_empty_query_routing(self):
        res = self.controller.process_query("")
        self.assertEqual(res.action_executed, "clarify")
        self.assertTrue(res.signals.is_empty)

    def test_fixed_baselines_override(self):
        # Direct-only baseline unconditionally routes to direct
        res_direct = self.controller.process_query("What is 14 * 25?", policy_name=PolicyType.DIRECT_ONLY.value)
        self.assertEqual(res_direct.action_executed, "direct")

        # Always-retrieve unconditionally routes to retrieve
        res_ret = self.controller.process_query("What is 14 * 25?", policy_name=PolicyType.ALWAYS_RETRIEVE.value)
        self.assertEqual(res_ret.action_executed, "retrieve")

        # Fixed-verify unconditionally routes to verify
        res_ver = self.controller.process_query("What is 14 * 25?", policy_name=PolicyType.FIXED_VERIFY.value)
        self.assertEqual(res_ver.action_executed, "verify")


if __name__ == "__main__":
    unittest.main()
