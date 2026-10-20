"""
Unit tests for Phase 5: Scientific Evaluation, Confusion Matrix, Mistake Analysis, and Ablations.
"""

import unittest
import pandas as pd
from src.experiment_runner import ExperimentRunner
from src.controller import MitigationController
from src.policy import PolicyType


class TestEvaluationAnalysisPhase5(unittest.TestCase):

    def setUp(self):
        self.controller = MitigationController()
        self.runner = ExperimentRunner(controller=self.controller)

    def test_run_all_five_policies(self):
        """Test comparative benchmark execution across all 5 candidate policies."""
        res = self.runner.run_benchmark(
            policies=[
                PolicyType.DIRECT_ONLY.value,
                PolicyType.ALWAYS_RETRIEVE.value,
                PolicyType.FIXED_VERIFY.value,
                PolicyType.ADAPTIVE.value,
                PolicyType.LEARNED_COST_AWARE.value,
            ],
            tradeoff_lambda=0.5,
        )

        self.assertIn("summary_df", res)
        self.assertIn("detailed_df", res)
        self.assertIn("mistakes_df", res)
        self.assertIn("raw_summaries", res)

        # 5 policy summaries
        self.assertEqual(len(res["summary_df"]), 5)
        self.assertEqual(len(res["raw_summaries"]), 5)

        # Detailed records for all items
        expected_total_rows = 5 * len(self.runner.dataset)
        self.assertEqual(len(res["detailed_df"]), expected_total_rows)

    def test_confusion_matrix_computation(self):
        """Test action-selection confusion matrix calculation."""
        res = self.runner.run_benchmark(
            policies=[PolicyType.ADAPTIVE.value],
            tradeoff_lambda=0.5,
        )
        cm = self.runner.compute_confusion_matrix(res["detailed_df"], PolicyType.ADAPTIVE.value)
        self.assertIsInstance(cm, pd.DataFrame)
        self.assertFalse(cm.empty)
        self.assertIn("Total", cm.columns)
        self.assertIn("Total", cm.index)

    def test_routing_mistake_recording(self):
        """Test identification and categorization of routing mistakes in baselines."""
        # Direct-only will fail on unanswerables and arithmetic tools
        res = self.runner.run_benchmark(
            policies=[PolicyType.DIRECT_ONLY.value],
            tradeoff_lambda=0.5,
        )
        mistakes_df = res["mistakes_df"]
        self.assertIsInstance(mistakes_df, pd.DataFrame)
        # Direct only fails on arithmetic (expected calculate) and unanswerable (expected abstain)
        self.assertGreater(len(mistakes_df), 0)
        self.assertIn("Failure Type", mistakes_df.columns)

    def test_ablation_study_execution(self):
        """Test component ablation study runner."""
        ablation_df = self.runner.run_ablation_study(tradeoff_lambda=0.5)
        self.assertIsInstance(ablation_df, pd.DataFrame)
        self.assertEqual(len(ablation_df), 4)
        self.assertIn("Configuration", ablation_df.columns)
        self.assertIn("Accuracy", ablation_df.columns)
        self.assertIn("Avg Tool Calls", ablation_df.columns)

    def test_held_out_evaluation(self):
        """Test evaluation run strictly on held-out test split."""
        res = self.runner.run_held_out_evaluation(tradeoff_lambda=0.5)
        self.assertIn("summary_df", res)
        self.assertIn("detailed_df", res)
        # Number of queries should equal test split size
        self.assertGreater(len(res["detailed_df"]), 0)

    def test_abstention_precision_recall_metrics(self):
        """Test that abstention precision, recall, and false abstention rate are computed."""
        res = self.runner.run_benchmark(
            policies=[PolicyType.ADAPTIVE.value, PolicyType.DIRECT_ONLY.value],
            tradeoff_lambda=0.5,
        )
        summary_df = res["summary_df"]
        self.assertIn("Abstention Precision", summary_df.columns)
        self.assertIn("Abstention Recall", summary_df.columns)
        self.assertIn("False Abstention Rate", summary_df.columns)
        # Verify raw summaries have typed values
        adaptive_sum = res["raw_summaries"][0]
        self.assertGreaterEqual(adaptive_sum.abstention_precision, 0.0)
        self.assertGreaterEqual(adaptive_sum.abstention_recall, 0.0)


if __name__ == "__main__":
    unittest.main()
