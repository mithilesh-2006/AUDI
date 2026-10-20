"""
Unit tests for Phase 4: Statistical Uncertainty Estimation, Calibration, and Learned Cost-Aware Policy.
"""

import unittest
from src.signals import SignalExtractor
from src.retrieval import LocalCorpusRetriever
from src.uncertainty import UncertaintyEstimator, DisagreementAnalysis
from src.calibration import CalibrationEngine, split_dev_and_test
from src.policy import LearnedCostAwarePolicy, PolicyType
from src.controller import MitigationController


class TestCalibrationAndLearnedPolicyPhase4(unittest.TestCase):

    def setUp(self):
        self.retriever = LocalCorpusRetriever()
        self.uncertainty_estimator = UncertaintyEstimator(retriever=self.retriever)
        self.calibrator = CalibrationEngine(num_bins=5)
        self.policy = LearnedCostAwarePolicy(default_lambda=0.5)

    def test_jaccard_distance_and_multi_sample_disagreement(self):
        """Test multi-sample lexical disagreement computation."""
        # Identical texts should have 0 distance
        d_identical = UncertaintyEstimator.compute_jaccard_distance("Paris is the capital", "Paris is the capital")
        self.assertEqual(d_identical, 0.0)

        # Completely disjoint texts should have distance 1.0
        d_disjoint = UncertaintyEstimator.compute_jaccard_distance("Mount Everest elevation", "Quantum mechanics photons")
        self.assertEqual(d_disjoint, 1.0)

        # Multi-sample analysis
        samples = [
            "The capital of France is Paris.",
            "Paris is the capital of France.",
            "Paris is France's historic capital city."
        ]
        analysis = UncertaintyEstimator.compute_multi_sample_disagreement(samples)
        self.assertIsInstance(analysis, DisagreementAnalysis)
        self.assertEqual(analysis.num_samples, 3)
        self.assertLess(analysis.mean_disagreement, 0.5)
        self.assertIn(analysis.disagreement_level, ("Low", "Moderate"))

    def test_composite_uncertainty_estimation(self):
        """Test composite uncertainty estimation across arithmetic, factual, and ambiguous queries."""
        # 1. Deterministic arithmetic -> very low uncertainty
        sig_math = SignalExtractor.extract("What is 14 * 25?")
        res_math = self.uncertainty_estimator.estimate_uncertainty("What is 14 * 25?", sig_math)
        self.assertLess(res_math.composite_uncertainty, 0.15)

        # 2. Speculation -> very high uncertainty
        sig_spec = SignalExtractor.extract("Will Bitcoin hit $500,000 in 2035?")
        res_spec = self.uncertainty_estimator.estimate_uncertainty("Will Bitcoin hit $500,000 in 2035?", sig_spec)
        self.assertGreater(res_spec.composite_uncertainty, 0.85)

        # 3. Ambiguous -> high uncertainty
        sig_ambig = SignalExtractor.extract("Which one is better?")
        res_ambig = self.uncertainty_estimator.estimate_uncertainty("Which one is better?", sig_ambig)
        self.assertGreater(res_ambig.composite_uncertainty, 0.70)

    def test_ece_and_brier_score_computation(self):
        """Test ECE and Brier score calibration metric calculations."""
        confidences = [0.95, 0.90, 0.80, 0.70, 0.60, 0.55, 0.40, 0.30, 0.20, 0.10]
        labels = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]

        metrics = self.calibrator.compute_ece_and_brier(confidences, labels)
        self.assertGreater(metrics.expected_calibration_error, 0.0)
        self.assertLessEqual(metrics.expected_calibration_error, 1.0)
        self.assertGreater(metrics.brier_score, 0.0)
        self.assertLess(metrics.brier_score, 0.5)
        self.assertEqual(metrics.total_samples, 10)
        self.assertEqual(len(metrics.bins), 5)

    def test_temperature_scaling_calibration(self):
        """Test Platt / temperature scaling parameter fitting."""
        logits = [2.5, 2.0, 1.5, 0.8, -0.5, -1.2, -2.0]
        labels = [1, 1, 1, 1, 0, 0, 0]

        fitted_temp = self.calibrator.fit_temperature(logits, labels)
        self.assertGreater(fitted_temp, 0.0)

        # Calibrate a raw probability
        cal_p = self.calibrator.calibrate_probability(0.85)
        self.assertGreater(cal_p, 0.0)
        self.assertLessEqual(cal_p, 1.0)

    def test_dev_and_test_split_leakage_prevention(self):
        """Ensure dev and test splits are strictly disjoint (leakage prevention)."""
        dataset = [{"id": f"item_{i}", "query": f"q_{i}"} for i in range(20)]
        dev_set, test_set = split_dev_and_test(dataset, dev_ratio=0.5, random_seed=42)

        self.assertEqual(len(dev_set), 10)
        self.assertEqual(len(test_set), 10)

        dev_ids = {d["id"] for d in dev_set}
        test_ids = {t["id"] for t in test_set}

        # Mutual exclusivity (Zero overlap / No leakage)
        self.assertEqual(len(dev_ids.intersection(test_ids)), 0)

    def test_learned_cost_aware_policy_routing(self):
        """Test LearnedCostAwarePolicy selection across task archetypes."""
        # Arithmetic -> calculate
        sig_math = SignalExtractor.extract("Calculate (120 + 45) / 5")
        math_dec = self.policy.select_action(sig_math, tradeoff_lambda=0.5)
        self.assertEqual(math_dec.selected_action, "calculate")
        self.assertEqual(math_dec.policy_name, PolicyType.LEARNED_COST_AWARE.value)

        # Ambiguous -> clarify
        sig_ambig = SignalExtractor.extract("Which one is better?")
        ambig_dec = self.policy.select_action(sig_ambig, tradeoff_lambda=0.5)
        self.assertEqual(ambig_dec.selected_action, "clarify")

        # Speculative / Unknowable -> abstain
        sig_spec = SignalExtractor.extract("Will Bitcoin hit $500,000 in 2035?")
        spec_dec = self.policy.select_action(sig_spec, tradeoff_lambda=0.5)
        self.assertEqual(spec_dec.selected_action, "abstain")

        # Factual knowledge -> verify / retrieve
        sig_fact = SignalExtractor.extract("What is the elevation of Mount Everest?")
        fact_dec = self.policy.select_action(sig_fact, tradeoff_lambda=0.5)
        self.assertIn(fact_dec.selected_action, ("retrieve", "verify"))

    def test_learned_cost_aware_policy_in_controller(self):
        """Test LearnedCostAwarePolicy full integration in MitigationController."""
        controller = MitigationController(retriever=self.retriever)
        res = controller.process_query(
            query="What is 14 * 25?",
            policy_name=PolicyType.LEARNED_COST_AWARE.value,
            tradeoff_lambda=0.5,
        )
        self.assertEqual(res.action_executed, "calculate")
        self.assertIsNotNone(res.uncertainty_result)
        self.assertLess(res.uncertainty_result.composite_uncertainty, 0.20)
        self.assertEqual(res.policy_name, PolicyType.LEARNED_COST_AWARE.value)


if __name__ == "__main__":
    unittest.main()
