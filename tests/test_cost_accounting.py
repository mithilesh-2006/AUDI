"""
Unit tests for Cost Accounting and Logging Service.
Verifies cost calculation, prior distinction, SQLite/JSONL persistence, and export.
"""

import os
import tempfile
import unittest
import time
from src.cost_tracker import CostTracker
from src.logging_service import LoggingService, RunRecord


class TestCostAndLogging(unittest.TestCase):

    def test_cost_accounting_units_and_monetary(self):
        # Action cost priors should be defined for candidate actions
        calc_prior = CostTracker.get_action_cost_prior("calculate")
        retrieve_prior = CostTracker.get_action_cost_prior("retrieve")
        self.assertLess(calc_prior, retrieve_prior)

        # Unbilled (local) should have 0 USD monetary cost
        record_local = CostTracker.create_record(
            action_name="calculate",
            start_time_perf=time.perf_counter(),
            prompt_tokens=100,
            completion_tokens=50,
            is_billed=False
        )
        self.assertEqual(record_local.monetary_cost_usd, 0.0)

        # Billed should compute accurate USD
        record_billed = CostTracker.create_record(
            action_name="direct",
            start_time_perf=time.perf_counter(),
            prompt_tokens=10_000,
            completion_tokens=5_000,
            is_billed=True
        )
        # 10k prompt * $0.15/1M = $0.0015; 5k comp * $0.60/1M = $0.003 -> $0.0045
        self.assertAlmostEqual(record_billed.monetary_cost_usd, 0.0045, places=5)

    def test_logging_and_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_exp.db")
            jsonl_path = os.path.join(tmpdir, "test_runs.jsonl")
            logger = LoggingService(db_path=db_path, jsonl_path=jsonl_path)

            rec = RunRecord(
                run_id="test_run_123",
                timestamp="2026-10-20T12:00:00Z",
                query="What is 14 * 25?",
                policy_name="Adaptive Policy",
                selected_action="calculate",
                decision_rationale="Arithmetic detected",
                answer_or_result="350",
                heuristic_uncertainty=0.1,
                observable_signals={"has_arithmetic": True},
                evidence_references=[],
                verification_status="Not Checked",
                supported_claims_count=0,
                unsupported_claims_count=0,
                model_name="local",
                is_mock_model=True,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                model_calls_count=0,
                tool_calls_count=1,
                latency_ms=1.5,
                estimated_cost_units=0.2,
                measured_monetary_usd=0.0,
                reference_answer="350",
                is_correct=True,
                benchmark_category="Arithmetic"
            )

            logger.log_run(rec)

            # Check SQLite
            recent = logger.get_recent_runs(limit=10)
            self.assertEqual(len(recent), 1)
            self.assertEqual(recent[0]["run_id"], "test_run_123")
            self.assertEqual(recent[0]["selected_action"], "calculate")

            # Check JSONL
            self.assertTrue(os.path.exists(jsonl_path))
            with open(jsonl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 1)

            # Check CSV Export
            csv_path = os.path.join(tmpdir, "export.csv")
            logger.export_csv(csv_path)
            self.assertTrue(os.path.exists(csv_path))

            # Check JSON Export
            json_export = os.path.join(tmpdir, "export.json")
            logger.export_json(json_export)
            self.assertTrue(os.path.exists(json_export))

            logger.close()


if __name__ == "__main__":
    unittest.main()
