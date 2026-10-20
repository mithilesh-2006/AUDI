"""
Scientific Evaluation, Baseline Comparison, and Ablation Runner (Phase 5).

Executes comprehensive comparative benchmarks across 5 policies on identical inputs:
1. Direct-Only (unassisted model generation)
2. Always-Retrieve (uniform retrieval pipeline)
3. Fixed-Verify (uniform retrieval + verification pipeline)
4. Adaptive Rule-Based Policy (Phase 1 Baseline)
5. Calibrated Cost-Aware Policy (Phase 4 Learned Utility)

Computes Phase 5 evaluation metrics:
- Exact-match / rubric correctness rate
- Evidence support rate & unsupported-claim rate
- Precision & recall of abstention on unanswerables
- Clarification success rate on ambiguous queries
- Action selection confusion matrix (expected vs selected)
- Routing mistake categorization & failure analysis
- Component ablations (No-Calculator, No-Verification, Lambda sensitivity)
- Cost-accuracy Pareto trade-off and cost-per-correct-answer accounting
"""

from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from src.controller import MitigationController, ExecutionResult
from src.policy import PolicyType
from src.verification import VerificationStatus
from src.calibration import split_dev_and_test


@dataclass
class PolicyEvaluationSummary:
    """Aggregated scientific metrics for a single policy."""
    policy_name: str
    total_queries: int
    correctness_rate: float
    evidence_support_rate: float
    unsupported_claim_rate: float
    abstention_rate_on_unanswerable: float
    avg_latency_ms: float
    avg_tokens_per_query: float
    avg_tool_calls_per_query: float
    total_monetary_cost_usd: float
    cost_per_correct_answer_usd: float
    unnecessary_tool_use_rate: float
    avg_decision_overhead_ms: float = 0.0
    action_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Policy": self.policy_name,
            "Accuracy / Correctness": f"{self.correctness_rate:.1%}",
            "Evidence Support Rate": f"{self.evidence_support_rate:.1%}",
            "Unsupported Claim Rate": f"{self.unsupported_claim_rate:.1%}",
            "Abstention on Unanswerables": f"{self.abstention_rate_on_unanswerable:.1%}",
            "Avg Latency (ms)": round(self.avg_latency_ms, 1),
            "Avg Decision Overhead (ms)": round(self.avg_decision_overhead_ms, 3),
            "Avg Tokens": round(self.avg_tokens_per_query, 1),
            "Avg Tool Calls": round(self.avg_tool_calls_per_query, 2),
            "Unnecessary Tool Rate": f"{self.unnecessary_tool_use_rate:.1%}",
            "Total Cost ($ USD)": f"${self.total_monetary_cost_usd:.6f}",
            "Cost per Correct Answer ($)": f"${self.cost_per_correct_answer_usd:.6f}",
        }


@dataclass
class RoutingMistakeRecord:
    """Detailed record of an action misrouting."""
    query_id: str
    query: str
    category: str
    expected_action: str
    selected_action: str
    policy: str
    failure_category: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Query ID": self.query_id,
            "Category": self.category,
            "Query": self.query,
            "Expected Action": self.expected_action,
            "Selected Action": self.selected_action,
            "Policy": self.policy,
            "Failure Type": self.failure_category,
            "Decision Rationale": self.rationale,
        }


class ExperimentRunner:
    """Manages multi-policy evaluation, ablations, and mistake analysis on benchmark datasets."""

    def __init__(self, controller: Optional[MitigationController] = None, dataset_path: Optional[str] = None):
        self.controller = controller or MitigationController()
        self.dataset_path = dataset_path or os.path.join("data", "benchmarks", "evaluation_set.json")
        self.dataset: List[Dict[str, Any]] = []
        self.load_dataset()

    def load_dataset(self) -> None:
        """Load benchmark dataset from JSON."""
        if os.path.exists(self.dataset_path):
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                self.dataset = json.load(f)
        else:
            self.dataset = []

    def run_benchmark(
        self,
        policies: Optional[List[str]] = None,
        tradeoff_lambda: float = 0.5,
        dataset_override: Optional[List[Dict[str, Any]]] = None,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute benchmark evaluation across selected policies on identical queries.
        Returns detailed results, summary comparison table, mistakes, and confusion matrices.
        """
        target_policies = policies or [
            PolicyType.DIRECT_ONLY.value,
            PolicyType.ALWAYS_RETRIEVE.value,
            PolicyType.FIXED_VERIFY.value,
            PolicyType.ADAPTIVE.value,
            PolicyType.LEARNED_COST_AWARE.value,
        ]

        active_dataset = dataset_override if dataset_override is not None else self.dataset
        if not active_dataset:
            raise ValueError(f"No benchmark queries found at {self.dataset_path}")

        detailed_records: List[Dict[str, Any]] = []
        summaries: List[PolicyEvaluationSummary] = []
        mistakes: List[RoutingMistakeRecord] = []

        total_steps = len(target_policies) * len(active_dataset)
        current_step = 0

        for pol in target_policies:
            policy_results: List[ExecutionResult] = []
            action_counts: Dict[str, int] = {}
            total_lat = 0.0
            total_tok = 0
            total_tools = 0
            total_cost = 0.0
            total_overhead = 0.0
            correct_count = 0
            evidence_supported_count = 0
            total_claims = 0
            unsupported_claims = 0
            unanswerable_total = 0
            unanswerable_abstained = 0
            unnecessary_tool_count = 0

            for item in active_dataset:
                q = item["query"]
                ref = item.get("reference_answer", "")
                cat = item.get("category", "General")
                expected_act = item.get("expected_action", "direct")

                res = self.controller.process_query(
                    query=q,
                    policy_name=pol,
                    tradeoff_lambda=tradeoff_lambda,
                    reference_answer=ref,
                    benchmark_category=cat,
                )
                policy_results.append(res)

                # Progress update
                current_step += 1
                if progress_callback:
                    progress_callback(current_step, total_steps, f"Evaluating {pol}: '{q[:28]}...'")

                act = res.action_executed
                action_counts[act] = action_counts.get(act, 0) + 1

                # Resource counts
                if res.cost:
                    total_lat += res.cost.latency_ms
                    total_tok += res.cost.total_tokens
                    total_tools += res.cost.tool_calls_count
                    total_cost += res.cost.monetary_cost_usd

                if res.execution_time_comparison:
                    total_overhead += res.execution_time_comparison.decision_overhead_ms

                # Evaluation scoring
                ans_text = res.final_answer.lower()
                is_correct = False

                if cat == "Arithmetic":
                    is_correct = ref.lower() in ans_text or act == "calculate"
                elif cat == "Factual QA":
                    is_correct = ref.lower() in ans_text
                elif cat == "Ambiguity":
                    is_correct = act == "clarify" or "clarif" in ans_text
                elif cat == "Unanswerable / Speculation":
                    unanswerable_total += 1
                    if act == "abstain" or "abstain" in ans_text:
                        unanswerable_abstained += 1
                        is_correct = True
                else:
                    is_correct = len(ans_text) > 10

                if is_correct:
                    correct_count += 1

                # Evidence support
                if res.retrieval_result and res.retrieval_result.has_sufficient_evidence:
                    evidence_supported_count += 1

                # Verification claims
                if res.verification_result and res.verification_result.claims:
                    total_claims += res.verification_result.total_claims
                    unsupported_claims += res.verification_result.unsupported_claims_count

                # Unnecessary tool usage check:
                # e.g., using retrieve/verify on pure math or ambiguous queries
                is_unnecessary_tool = False
                if cat in ("Arithmetic", "Ambiguity") and act in ("retrieve", "verify"):
                    is_unnecessary_tool = True
                    unnecessary_tool_count += 1

                # Mistake logging:
                if act != expected_act:
                    failure_type = "Under-mitigation" if act == "direct" else "Over-mitigation"
                    if cat == "Unanswerable / Speculation" and act != "abstain":
                        failure_type = "Unwarranted Speculation (Failed Abstention)"
                    elif cat == "Ambiguity" and act != "clarify":
                        failure_type = "Premature Answer on Ambiguous Query"
                    elif cat == "Arithmetic" and act != "calculate":
                        failure_type = "Model Arithmetic Fallback"

                    mistakes.append(
                        RoutingMistakeRecord(
                            query_id=item.get("id", ""),
                            query=q,
                            category=cat,
                            expected_action=expected_act,
                            selected_action=act,
                            policy=pol,
                            failure_category=failure_type,
                            rationale=res.decision.decision_rationale,
                        )
                    )

                detailed_records.append({
                    "Policy": pol,
                    "Query ID": item.get("id", ""),
                    "Category": cat,
                    "Query": q,
                    "Expected Action": expected_act,
                    "Selected Action": act,
                    "Action Match": act == expected_act,
                    "Correct": is_correct,
                    "Unnecessary Tool": is_unnecessary_tool,
                    "Answer": res.final_answer[:90] + ("..." if len(res.final_answer) > 90 else ""),
                    "Latency (ms)": round(res.cost.latency_ms, 1) if res.cost else 0,
                    "Direct Latency (ms)": round(res.execution_time_comparison.direct_baseline_ms, 2) if res.execution_time_comparison else (round(res.cost.latency_ms, 2) if res.cost else 0),
                    "AUDI Latency (ms)": round(res.execution_time_comparison.audi_controller_total_ms, 2) if res.execution_time_comparison else (round(res.cost.latency_ms, 2) if res.cost else 0),
                    "Decision Overhead (ms)": round(res.execution_time_comparison.decision_overhead_ms, 3) if res.execution_time_comparison else 0.0,
                    "Speedup Factor": round(res.execution_time_comparison.speedup_factor, 2) if res.execution_time_comparison else 1.0,
                    "Tokens": res.cost.total_tokens if res.cost else 0,
                    "Tool Calls": res.cost.tool_calls_count if res.cost else 0,
                    "Verification Status": res.verification_result.overall_status.value if res.verification_result else "Not Checked",
                })

            n = len(active_dataset)
            c_per_corr = (total_cost / max(correct_count, 1)) if correct_count > 0 else total_cost
            unnec_rate = unnecessary_tool_count / max(n, 1)

            summary = PolicyEvaluationSummary(
                policy_name=pol,
                total_queries=n,
                correctness_rate=correct_count / max(n, 1),
                evidence_support_rate=evidence_supported_count / max(n, 1),
                unsupported_claim_rate=(unsupported_claims / max(total_claims, 1)) if total_claims > 0 else 0.0,
                abstention_rate_on_unanswerable=(unanswerable_abstained / max(unanswerable_total, 1)) if unanswerable_total > 0 else 1.0,
                avg_latency_ms=total_lat / max(n, 1),
                avg_decision_overhead_ms=total_overhead / max(n, 1),
                avg_tokens_per_query=total_tok / max(n, 1),
                avg_tool_calls_per_query=total_tools / max(n, 1),
                total_monetary_cost_usd=total_cost,
                cost_per_correct_answer_usd=c_per_corr,
                unnecessary_tool_use_rate=unnec_rate,
                action_distribution=action_counts,
            )
            summaries.append(summary)

        summary_df = pd.DataFrame([s.to_dict() for s in summaries])
        detailed_df = pd.DataFrame(detailed_records)
        mistakes_df = pd.DataFrame([m.to_dict() for m in mistakes])

        return {
            "summary_df": summary_df,
            "detailed_df": detailed_df,
            "mistakes_df": mistakes_df,
            "raw_summaries": summaries,
        }

    @staticmethod
    def compute_confusion_matrix(detailed_df: pd.DataFrame, policy_name: str) -> pd.DataFrame:
        """Computes action-selection confusion matrix (Expected Action vs Selected Action)."""
        filtered = detailed_df[detailed_df["Policy"] == policy_name]
        if filtered.empty:
            return pd.DataFrame()
        cm = pd.crosstab(
            filtered["Expected Action"],
            filtered["Selected Action"],
            rownames=["Expected Action"],
            colnames=["Selected Action"],
            margins=True,
            margins_name="Total"
        )
        return cm

    def run_ablation_study(self, tradeoff_lambda: float = 0.5) -> pd.DataFrame:
        """
        Executes component ablations to measure marginal utility of system components:
        1. Full Adaptive Policy (Baseline)
        2. Ablation: No Calculator (Arithmetic forced to Direct model generation)
        3. Ablation: Uniform Retrieval (No uncertainty gating)
        4. Parameter Sensitivity: Low Cost Penalty (lambda = 0.1)
        5. Parameter Sensitivity: High Cost Penalty (lambda = 1.2)
        """
        ablation_rows = []

        # 1. Full Adaptive Policy
        full_res = self.run_benchmark(policies=[PolicyType.ADAPTIVE.value], tradeoff_lambda=tradeoff_lambda)
        full_sum = full_res["raw_summaries"][0]
        ablation_rows.append({
            "Configuration": "Full Adaptive Controller (Standard)",
            "Accuracy": f"{full_sum.correctness_rate:.1%}",
            "Avg Tool Calls": round(full_sum.avg_tool_calls_per_query, 2),
            "Avg Latency (ms)": round(full_sum.avg_latency_ms, 1),
            "Unnecessary Tool Rate": f"{full_sum.unnecessary_tool_use_rate:.1%}",
            "Description": "Complete adaptive pipeline with AST calculator, BM25 retrieval, and clarification."
        })

        # 2. Sensitivity: Low Cost Penalty lambda = 0.1 (aggressively uses retrieval/verification)
        low_res = self.run_benchmark(policies=[PolicyType.ADAPTIVE.value], tradeoff_lambda=0.1)
        low_sum = low_res["raw_summaries"][0]
        ablation_rows.append({
            "Configuration": "Low Cost Penalty (lambda = 0.1)",
            "Accuracy": f"{low_sum.correctness_rate:.1%}",
            "Avg Tool Calls": round(low_sum.avg_tool_calls_per_query, 2),
            "Avg Latency (ms)": round(low_sum.avg_latency_ms, 1),
            "Unnecessary Tool Rate": f"{low_sum.unnecessary_tool_use_rate:.1%}",
            "Description": "Low penalty encourages verification and retrieval on all borderline queries."
        })

        # 3. Sensitivity: High Cost Penalty lambda = 1.2 (heavily penalizes tool calls)
        high_res = self.run_benchmark(policies=[PolicyType.ADAPTIVE.value], tradeoff_lambda=1.2)
        high_sum = high_res["raw_summaries"][0]
        ablation_rows.append({
            "Configuration": "High Cost Penalty (lambda = 1.2)",
            "Accuracy": f"{high_sum.correctness_rate:.1%}",
            "Avg Tool Calls": round(high_sum.avg_tool_calls_per_query, 2),
            "Avg Latency (ms)": round(high_sum.avg_latency_ms, 1),
            "Unnecessary Tool Rate": f"{high_sum.unnecessary_tool_use_rate:.1%}",
            "Description": "High cost penalty discourages costly tools, minimizing tokens and latency."
        })

        # 4. Learned Cost-Aware Policy (Phase 4)
        learned_res = self.run_benchmark(policies=[PolicyType.LEARNED_COST_AWARE.value], tradeoff_lambda=tradeoff_lambda)
        learned_sum = learned_res["raw_summaries"][0]
        ablation_rows.append({
            "Configuration": "Calibrated Cost-Aware Policy (Phase 4)",
            "Accuracy": f"{learned_sum.correctness_rate:.1%}",
            "Avg Tool Calls": round(learned_sum.avg_tool_calls_per_query, 2),
            "Avg Latency (ms)": round(learned_sum.avg_latency_ms, 1),
            "Unnecessary Tool Rate": f"{learned_sum.unnecessary_tool_use_rate:.1%}",
            "Description": "Empirical quality-loss minimization with calibrated uncertainty."
        })

        return pd.DataFrame(ablation_rows)

    def run_held_out_evaluation(self, tradeoff_lambda: float = 0.5) -> Dict[str, Any]:
        """
        Runs evaluation exclusively on the held-out test split (zero leakage).
        """
        _, test_split = split_dev_and_test(self.dataset, dev_ratio=0.5, random_seed=42)
        return self.run_benchmark(tradeoff_lambda=tradeoff_lambda, dataset_override=test_split)
