"""
Scientific Evaluation and Baseline Comparison Runner.

Executes comparative benchmarks across 4 policies on identical inputs:
1. Direct-Only
2. Always-Retrieve
3. Fixed-Verify
4. Adaptive Policy

Computes research metrics:
- Correctness rate (ground truth / reference matching)
- Evidence support rate
- Unsupported-claim rate
- Abstention rate on unanswerables
- Latency (ms), Tokens, Tool calls, and Monetary cost ($ USD)
"""

from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd

from src.controller import MitigationController, ExecutionResult
from src.policy import PolicyType
from src.verification import VerificationStatus


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
    action_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Policy": self.policy_name,
            "Accuracy / Correctness": f"{self.correctness_rate:.1%}",
            "Evidence Support Rate": f"{self.evidence_support_rate:.1%}",
            "Unsupported Claim Rate": f"{self.unsupported_claim_rate:.1%}",
            "Abstention on Unanswerables": f"{self.abstention_rate_on_unanswerable:.1%}",
            "Avg Latency (ms)": round(self.avg_latency_ms, 1),
            "Avg Tokens": round(self.avg_tokens_per_query, 1),
            "Avg Tool Calls": round(self.avg_tool_calls_per_query, 2),
            "Total Cost ($ USD)": f"${self.total_monetary_cost_usd:.6f}",
        }


class ExperimentRunner:
    """Manages multi-policy evaluation on benchmark datasets."""

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
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute benchmark evaluation across selected policies.
        Returns detailed results and summary comparison table.
        """
        target_policies = policies or [
            PolicyType.DIRECT_ONLY.value,
            PolicyType.ALWAYS_RETRIEVE.value,
            PolicyType.FIXED_VERIFY.value,
            PolicyType.ADAPTIVE.value,
        ]

        if not self.dataset:
            raise ValueError(f"No benchmark queries found at {self.dataset_path}")

        detailed_records: List[Dict[str, Any]] = []
        summaries: List[PolicyEvaluationSummary] = []

        total_steps = len(target_policies) * len(self.dataset)
        current_step = 0

        for pol in target_policies:
            policy_results: List[ExecutionResult] = []
            action_counts: Dict[str, int] = {}
            total_lat = 0.0
            total_tok = 0
            total_tools = 0
            total_cost = 0.0
            correct_count = 0
            evidence_supported_count = 0
            total_claims = 0
            unsupported_claims = 0
            unanswerable_total = 0
            unanswerable_abstained = 0

            for item in self.dataset:
                q = item["query"]
                ref = item.get("reference_answer", "")
                cat = item.get("category", "General")

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
                    progress_callback(current_step, total_steps, f"Evaluating {pol}: '{q[:30]}...'")

                act = res.action_executed
                action_counts[act] = action_counts.get(act, 0) + 1

                # Resource counts
                if res.cost:
                    total_lat += res.cost.latency_ms
                    total_tok += res.cost.total_tokens
                    total_tools += res.cost.tool_calls_count
                    total_cost += res.cost.monetary_cost_usd

                # Evaluation scoring
                ans_text = res.final_answer.lower()
                is_correct = False
                
                if cat == "Arithmetic":
                    # Check if reference number is in answer
                    is_correct = ref.lower() in ans_text
                elif cat == "Factual QA":
                    # Check key phrase match
                    is_correct = ref.lower() in ans_text
                elif cat == "Ambiguity":
                    # Correct if clarified or requested clarification
                    is_correct = act == "clarify" or "clarif" in ans_text
                elif cat == "Unanswerable / Speculation":
                    unanswerable_total += 1
                    if act == "abstain" or "abstain" in ans_text:
                        unanswerable_abstained += 1
                        is_correct = True

                if is_correct:
                    correct_count += 1

                # Evidence support
                if res.retrieval_result and res.retrieval_result.has_sufficient_evidence:
                    evidence_supported_count += 1

                # Verification claims
                if res.verification_result and res.verification_result.claims:
                    total_claims += res.verification_result.total_claims
                    unsupported_claims += res.verification_result.unsupported_claims_count

                detailed_records.append({
                    "Policy": pol,
                    "Query ID": item.get("id", ""),
                    "Category": cat,
                    "Query": q,
                    "Action": act,
                    "Correct": is_correct,
                    "Answer": res.final_answer[:100] + ("..." if len(res.final_answer) > 100 else ""),
                    "Latency (ms)": round(res.cost.latency_ms, 1) if res.cost else 0,
                    "Tokens": res.cost.total_tokens if res.cost else 0,
                    "Tool Calls": res.cost.tool_calls_count if res.cost else 0,
                    "Verification Status": res.verification_result.overall_status.value if res.verification_result else "Not Checked",
                })

            n = len(self.dataset)
            summary = PolicyEvaluationSummary(
                policy_name=pol,
                total_queries=n,
                correctness_rate=correct_count / max(n, 1),
                evidence_support_rate=evidence_supported_count / max(n, 1),
                unsupported_claim_rate=(unsupported_claims / max(total_claims, 1)) if total_claims > 0 else 0.0,
                abstention_rate_on_unanswerable=(unanswerable_abstained / max(unanswerable_total, 1)) if unanswerable_total > 0 else 1.0,
                avg_latency_ms=total_lat / max(n, 1),
                avg_tokens_per_query=total_tok / max(n, 1),
                avg_tool_calls_per_query=total_tools / max(n, 1),
                total_monetary_cost_usd=total_cost,
                action_distribution=action_counts,
            )
            summaries.append(summary)

        summary_df = pd.DataFrame([s.to_dict() for s in summaries])
        detailed_df = pd.DataFrame(detailed_records)

        return {
            "summary_df": summary_df,
            "detailed_df": detailed_df,
            "raw_summaries": summaries,
        }
