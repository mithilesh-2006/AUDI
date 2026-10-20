"""
Root test runner and controller tests as described in Appendix B of PROJECT_DESCRIPTION.md.
Usage:
    python test_controller.py
    python -m unittest -v
"""

import unittest
from tests.test_calculator import TestSafeCalculator
from tests.test_signals import TestSignalExtractor
from tests.test_retrieval import TestLocalCorpusRetriever
from tests.test_verification_revision import TestVerificationAndRevision
from tests.test_controller_routing import TestControllerRouting
from tests.test_llm_provider import TestLLMProvider
from tests.test_cost_accounting import TestCostAndLogging
from tests.test_experiment_runner import TestExperimentRunner


if __name__ == "__main__":
    unittest.main(verbosity=2)
