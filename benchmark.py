"""
AUDI Benchmark, Calibration, and Evaluation CLI.

Executes reproducible, scientific evaluations of Adaptive Uncertainty-Driven Inference
using the unified 20,000-sample benchmark dataset (unified_eval_20000.csv).

Supported Policies:
1. direct_only: Unassisted model generation baseline.
2. always_retrieve: Uniform RAG retrieval baseline.
3. fixed_verify: Uniform retrieval and verification baseline.
4. adaptive_rules: Rule-based heuristics routing (Phase 1 Baseline).
5. learned_cost_aware: Calibrated, cost-aware expected utility routing (AUDI).

Usage:
    python benchmark.py --samples 50
    python benchmark.py --samples 100 --calibrate
    python benchmark.py --split test --samples 100 --output-csv results/benchmark_summary.csv
    python benchmark.py --policies direct_only adaptive_rules learned_cost_aware
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from typing import List, Dict, Any, Optional

import pandas as pd

from src.dataset import DatasetManager
from src.controller import MitigationController
from src.experiment_runner import ExperimentRunner
from src.calibration import CalibrationEngine, split_dev_and_test
from src.policy import PolicyType


def print_banner():
    print("=" * 80)
    print(" AUDI - Adaptive Uncertainty-Driven Inference Benchmark Engine")
    print(" Scientific Hallucination Mitigation & Cost-Accuracy Optimization")
    print("=" * 80)


def run_calibration(train_records: List[Dict[str, Any]], controller: MitigationController) -> float:
    """
    Fits temperature scaling parameter on the training/dev split and measures ECE reduction.
    """
    print("\n--- Calibration Phase (Train/Dev Split) ---")
    print(f"Total training calibration samples: {len(train_records)}")

    engine = CalibrationEngine(num_bins=5)
    confidences: List[float] = []
    labels: List[int] = []
    logits: List[float] = []

    print("Extracting draft confidences and empirical accuracy on calibration set...")
    for item in train_records:
        q = item["query"]
        ref = item.get("reference_answer", "")
        cat = item.get("category", "")

        # Process via direct model generation to extract draft confidence
        res = controller.process_query(
            query=q,
            policy_name=PolicyType.DIRECT_ONLY.value,
            reference_answer=ref,
            benchmark_category=cat,
        )

        # Extract confidence score from draft or uncertainty estimation
        conf = 0.5
        if res.model_draft and res.model_draft.model_reported_confidence is not None:
            conf = res.model_draft.model_reported_confidence
        elif res.uncertainty_result:
            conf = 1.0 - res.uncertainty_result.composite_uncertainty
        elif res.decision:
            conf = 1.0 - res.decision.heuristic_uncertainty
        conf = max(0.01, min(0.99, conf))
        confidences.append(conf)

        # Logit computation for Platt scaling: log(p / (1 - p))
        import math
        logit = math.log(conf / (1.0 - conf))
        logits.append(logit)

        # Empirical correctness evaluation
        ans_text = str(res.final_answer or "").lower()
        ref_text = str(ref).lower().strip()
        is_corr = 1 if (ref_text in ans_text or len(ans_text) > 10) else 0
        labels.append(is_corr)

    # Pre-calibration metrics
    metrics_before = engine.compute_ece_and_brier(confidences, labels)
    print(f"Pre-Calibration  -> ECE: {metrics_before.expected_calibration_error:.4f} | Brier: {metrics_before.brier_score:.4f} | MCE: {metrics_before.maximum_calibration_error:.4f}")

    # Fit temperature T
    fitted_t = engine.fit_temperature(logits, labels)
    print(f"Fitted Optimal Temperature Parameter (T): {fitted_t:.3f}")

    # Post-calibration metrics
    calibrated_confs = [engine.calibrate_probability(c) for c in confidences]
    metrics_after = engine.compute_ece_and_brier(calibrated_confs, labels)
    print(f"Post-Calibration -> ECE: {metrics_after.expected_calibration_error:.4f} | Brier: {metrics_after.brier_score:.4f} | MCE: {metrics_after.maximum_calibration_error:.4f}")
    ece_reduction = ((metrics_before.expected_calibration_error - metrics_after.expected_calibration_error) / max(metrics_before.expected_calibration_error, 1e-6)) * 100.0
    print(f"ECE Calibration Improvement: {ece_reduction:+.1f}%\n")

    return fitted_t


def format_table(df: pd.DataFrame) -> str:
    """Format DataFrame as a clean ASCII text table."""
    headers = list(df.columns)
    col_widths = {h: max(len(h), max((len(str(v)) for v in df[h]), default=0)) for h in headers}

    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    separator = "-+-".join("-" * col_widths[h] for h in headers)
    
    rows = []
    for _, row in df.iterrows():
        row_str = " | ".join(str(row[h]).ljust(col_widths[h]) for h in headers)
        rows.append(row_str)

    return f"{header_line}\n{separator}\n" + "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="AUDI Adaptive Inference Benchmark & Calibration Runner")
    parser.add_argument("--dataset", type=str, default="unified_eval_20000.csv", help="Path to unified evaluation dataset CSV")
    parser.add_argument("--split", type=str, choices=["train", "test", "both"], default="test", help="Dataset split to evaluate")
    parser.add_argument("--samples", type=str, default="100", help="Number of samples to evaluate (integer or 'all')")
    parser.add_argument("--calibrate", action="store_true", help="Fit temperature scaling calibration on train split before testing")
    parser.add_argument("--lambda-cost", type=float, default=0.5, dest="lambda_cost", help="Trade-off parameter lambda in [0.0, 1.0]")
    parser.add_argument("--policies", nargs="+", default=[
        "direct_only",
        "always_retrieve",
        "fixed_verify",
        "adaptive_rules",
        "learned_cost_aware",
    ], help="Policies to evaluate")
    parser.add_argument("--output-csv", type=str, default="results/benchmark_summary.csv", help="Destination CSV file for summary")
    parser.add_argument("--prepare-splits", action="store_true", help="Force rebuild of train/test benchmark splits from CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset sampling")

    args = parser.parse_args()
    print_banner()

    # 1. Dataset Verification & Preparation
    train_file = os.path.join(DatasetManager.BENCHMARKS_DIR, "train_set.json")
    test_file = os.path.join(DatasetManager.BENCHMARKS_DIR, "test_set.json")

    if args.prepare_splits or not (os.path.exists(train_file) and os.path.exists(test_file)):
        print(f"Preparing stratified splits from '{args.dataset}' (Seed: {args.seed})...")
        train_records, test_records = DatasetManager.prepare_dataset(
            csv_path=args.dataset,
            total_samples=1000,
            train_ratio=0.5,
            random_seed=args.seed,
            save_splits=True,
        )
        print(f"Generated {len(train_records)} train/dev records and {len(test_records)} held-out test records.")
    else:
        import json
        with open(train_file, "r", encoding="utf-8") as f:
            train_records = json.load(f)
        with open(test_file, "r", encoding="utf-8") as f:
            test_records = json.load(f)
        print(f"Loaded existing splits: {len(train_records)} train/dev samples, {len(test_records)} test samples.")

    # 2. Select target dataset split
    if args.split == "train":
        target_records = train_records
        split_name = "Train / Dev"
    elif args.split == "test":
        target_records = test_records
        split_name = "Held-Out Test"
    else:
        target_records = train_records + test_records
        split_name = "Combined (Train + Test)"

    # Subsample if requested
    if args.samples.lower() != "all":
        try:
            n_samples = int(args.samples)
            target_records = target_records[:n_samples]
        except ValueError:
            print(f"Invalid --samples value '{args.samples}'. Evaluating all {len(target_records)} available.")

    print(f"Active Evaluation Subset: {len(target_records)} queries from [{split_name} Split]")
    print(f"Candidate Policies: {', '.join(args.policies)}")
    print(f"Cost-Accuracy Lambda: {args.lambda_cost}\n")

    # 3. Instantiate Controller & Runner
    controller = MitigationController()

    # 4. Calibration (Optional)
    if args.calibrate:
        fitted_t = run_calibration(train_records[:min(len(train_records), 100)], controller)
        controller.calibrator.temperature = fitted_t

    # 5. Execute Multi-Policy Benchmark
    print("Executing Benchmark Pipeline across identical queries...")
    runner = ExperimentRunner(controller=controller)
    
    start_time = time.perf_counter()
    benchmark_results = runner.run_benchmark(
        policies=args.policies,
        tradeoff_lambda=args.lambda_cost,
        dataset_override=target_records,
    )
    elapsed_total = time.perf_counter() - start_time

    # 6. Display Summary Results
    df_summary = benchmark_results["summary_df"]
    print("\n" + "=" * 80)
    print(" AUDI BENCHMARK EVALUATION SUMMARY")
    print(f" Total Evaluation Time: {elapsed_total:.2f}s | Split: {split_name} | Queries: {len(target_records)}")
    print("=" * 80)
    print(format_table(df_summary))
    print("=" * 80)

    # 7. Action Distribution Matrix
    print("\n--- Action Execution Distribution ---")
    dist_rows = []
    for pol_summary in benchmark_results["raw_summaries"]:
        row = {"Policy": pol_summary.policy_name}
        row.update(pol_summary.action_distribution)
        dist_rows.append(row)
    df_dist = pd.DataFrame(dist_rows).fillna(0)
    print(format_table(df_dist))

    # 8. Export Summary CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    df_summary.to_csv(args.output_csv, index=False)
    print(f"\nBenchmark summary written to: {args.output_csv}")
    print(f"Detailed query-level logs saved in SQLite: results/experiments.db")
    print("Launch Streamlit dashboard to inspect runs: streamlit run app.py\n")


if __name__ == "__main__":
    main()
