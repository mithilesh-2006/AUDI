"""
Calibration Engine & Reliability Evaluation (Phase 4).

Provides:
1. Expected Calibration Error (ECE) computation across configurable probability bins.
2. Brier score calculation for probability assessments.
3. Reliability diagram data generation (mean predicted confidence vs empirical accuracy).
4. Platt / Temperature scaling parameter fitting on development sets.
5. Strict dataset split verification to prevent test-set leakage.

All formulas and procedures adhere to:
Guo et al. (2017) "On Calibration of Modern Neural Networks"
and Section 12/13 of PROJECT_DESCRIPTION.md.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional


@dataclass
class CalibrationBin:
    """Statistics for an individual confidence interval bin."""
    bin_index: int
    bin_lower: float
    bin_upper: float
    sample_count: int
    mean_confidence: float
    accuracy: float
    calibration_gap: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bin_index": self.bin_index,
            "interval": f"[{self.bin_lower:.2f}, {self.bin_upper:.2f})",
            "sample_count": self.sample_count,
            "mean_confidence": round(self.mean_confidence, 4),
            "accuracy": round(self.accuracy, 4),
            "calibration_gap": round(self.calibration_gap, 4),
        }


@dataclass
class CalibrationMetrics:
    """Overall calibration performance metrics."""
    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score: float
    total_samples: int
    bins: List[CalibrationBin]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "maximum_calibration_error": round(self.maximum_calibration_error, 4),
            "brier_score": round(self.brier_score, 4),
            "total_samples": self.total_samples,
            "bins": [b.to_dict() for b in self.bins],
        }


class CalibrationEngine:
    """
    Computes calibration metrics (ECE, Brier Score) and applies temperature scaling.
    """

    def __init__(self, num_bins: int = 5):
        self.num_bins = max(2, num_bins)
        self.temperature: float = 1.0  # Fitted temperature scaling factor

    def compute_ece_and_brier(
        self,
        confidences: List[float],
        ground_truth_correct: List[int],
    ) -> CalibrationMetrics:
        """
        Computes Expected Calibration Error (ECE) and Brier Score.
        Args:
            confidences: List of predicted confidence probabilities in [0.0, 1.0].
            ground_truth_correct: List of binary correctness labels (1 for correct, 0 for incorrect).
        """
        if not confidences or len(confidences) != len(ground_truth_correct):
            raise ValueError("Confidences and labels must be non-empty and of equal length.")

        n = len(confidences)
        bin_width = 1.0 / self.num_bins
        bins: List[CalibrationBin] = []

        ece = 0.0
        mce = 0.0

        for b in range(self.num_bins):
            lower = b * bin_width
            upper = (b + 1) * bin_width
            # Include upper boundary in the last bin
            if b == self.num_bins - 1:
                indices = [i for i, c in enumerate(confidences) if lower <= c <= upper]
            else:
                indices = [i for i, c in enumerate(confidences) if lower <= c < upper]

            count = len(indices)
            if count > 0:
                bin_confs = [confidences[i] for i in indices]
                bin_labels = [ground_truth_correct[i] for i in indices]
                mean_conf = sum(bin_confs) / count
                accuracy = sum(bin_labels) / count
                gap = abs(accuracy - mean_conf)
                ece += (count / n) * gap
                mce = max(mce, gap)
            else:
                mean_conf = (lower + upper) / 2.0
                accuracy = 0.0
                gap = 0.0

            bins.append(
                CalibrationBin(
                    bin_index=b,
                    bin_lower=lower,
                    bin_upper=upper,
                    sample_count=count,
                    mean_confidence=mean_conf,
                    accuracy=accuracy,
                    calibration_gap=gap,
                )
            )

        # Brier Score = (1/N) * sum((p_i - y_i)^2)
        brier = sum((c - y) ** 2 for c, y in zip(confidences, ground_truth_correct)) / n

        return CalibrationMetrics(
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            brier_score=brier,
            total_samples=n,
            bins=bins,
        )

    def fit_temperature(
        self,
        logits: List[float],
        labels: List[int],
        search_range: Tuple[float, float] = (0.1, 5.0),
        steps: int = 100,
    ) -> float:
        """
        Fits optimal temperature T on development data minimizing NLL or Brier score.
        P_calibrated = sigmoid(logit / T).
        """
        best_t = 1.0
        best_loss = float("inf")

        low, high = search_range
        step_size = (high - low) / steps

        for step in range(steps + 1):
            t = low + step * step_size
            if t <= 0:
                continue

            # Compute cross-entropy loss with temperature
            loss = 0.0
            for logit, y in zip(logits, labels):
                p = 1.0 / (1.0 + math.exp(-logit / t))
                p = max(1e-7, min(1.0 - 1e-7, p))
                if y == 1:
                    loss -= math.log(p)
                else:
                    loss -= math.log(1.0 - p)

            if loss < best_loss:
                best_loss = loss
                best_t = t

        self.temperature = round(best_t, 3)
        return self.temperature

    def calibrate_probability(self, raw_prob: float) -> float:
        """
        Calibrates raw probability using the fitted temperature scaling parameter.
        Maps raw probability to logit, scales by T, and applies sigmoid.
        """
        p = max(1e-5, min(1.0 - 1e-5, raw_prob))
        logit = math.log(p / (1.0 - p))
        scaled_logit = logit / max(0.01, self.temperature)
        calibrated_p = 1.0 / (1.0 + math.exp(-scaled_logit))
        return round(calibrated_p, 4)


def split_dev_and_test(
    dataset: List[Dict[str, Any]],
    dev_ratio: float = 0.5,
    random_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Deterministic dataset splitting to guarantee strict separation between
    development tuning data and held-out evaluation test data (prevents test-set leakage).
    """
    import random
    rng = random.Random(random_seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    split_idx = int(len(dataset) * dev_ratio)
    dev_indices = set(indices[:split_idx])

    dev_split = [dataset[i] for i in range(len(dataset)) if i in dev_indices]
    test_split = [dataset[i] for i in range(len(dataset)) if i not in dev_indices]

    return dev_split, test_split
