"""
Unit tests for ExperimentRunner.
Verifies multi-policy execution consistency across identical queries and metrics calculation.
"""

import unittest
from src.experiment_runner import ExperimentRunner
from src.controller import MitigationController
from src.policy import PolicyType


class TestExperimentRunner(unittest.TestCase):

    def test_run_benchmark(self):
        controller = MitigationController(tradeoff_lambda=0.5)
        runner = ExperimentRunner(controller=controller)

        # Run comparison between Direct-only and Adaptive on benchmark set
        res = runner.run_benchmark(
            policies=[PolicyType.DIRECT_ONLY.value, PolicyType.ADAPTIVE.value],
            tradeoff_lambda=0.5
        )

        self.assertIn("summary_df", res)
        self.assertIn("detailed_df", res)

        summary_df = res["summary_df"]
        self.assertEqual(len(summary_df), 2)  # 2 policies evaluated

        detailed_df = res["detailed_df"]
        # Ensure identical queries evaluated for each policy
        direct_queries = detailed_df[detailed_df["Policy"] == PolicyType.DIRECT_ONLY.value]["Query"].tolist()
        adaptive_queries = detailed_df[detailed_df["Policy"] == PolicyType.ADAPTIVE.value]["Query"].tolist()
        self.assertEqual(direct_queries, adaptive_queries)

        # Adaptive policy should have higher accuracy because it routes arithmetic and unanswerables properly
        adaptive_summary = res["raw_summaries"][1]
        direct_summary = res["raw_summaries"][0]
        self.assertGreaterEqual(adaptive_summary.correctness_rate, direct_summary.correctness_rate)


if __name__ == "__main__":
    unittest.main()
