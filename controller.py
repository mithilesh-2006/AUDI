"""
Adaptive, Cost-Aware Hallucination Mitigation Controller - CLI & Entrypoint (Phase 2).

Demonstrates transparent action selection, safe AST calculation, evidence retrieval,
model adapter draft generation, verification, and cost accounting.

Usage:
    python controller.py
    python controller.py --demo
    python controller.py --query "Calculate (120 + 45) / 5"
    python controller.py --query "Which one is better?"
    python controller.py --query "What is the elevation of Mount Everest?"
    python controller.py --query "What is the speed of light?" --provider offline --prompt-version v1.1-with-confidence-elicitation
"""

from __future__ import annotations
import argparse
import json
import sys
from typing import Optional

from src.signals import SignalExtractor, QuerySignals
from src.calculator import SafeCalculator, CalculationResult
from src.retrieval import LocalCorpusRetriever, RetrievalResult
from src.verification import EvidenceVerifier, VerificationResult, VerificationStatus
from src.revision import AnswerReviser, RevisionResult
from src.model_adapter import (
    BaseModelAdapter,
    ModelDraft,
    GenerationSettings,
    get_model_adapter,
)
from src.cost_tracker import CostTracker, CostAccounting
from src.logging_service import LoggingService, get_logging_service
from src.policy import PolicyType, AdaptiveRulePolicy
from src.controller import MitigationController, ExecutionResult


def run_interactive(controller: MitigationController):
    """Run an interactive CLI session."""
    print("=" * 70)
    print(" Adaptive, Cost-Aware Hallucination Mitigation Controller (Phase 2)")
    print(" Research Prototype - Pluggable Model Adapters & Prompt Versioning")
    print("=" * 70)
    print("Commands:")
    print("  Type any question or math expression (e.g., '14 * 25', 'Which one is better?')")
    print("  Type 'demo' to run the standard demonstration suite")
    print("  Type 'exit' or 'quit' to close\n")

    while True:
        try:
            user_input = input("\nEnter query > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Exiting controller. Goodbye!")
                break
            if user_input.lower() == "demo":
                run_demo(controller)
                continue

            result = controller.process_query(user_input)
            display_result(result)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def run_demo(controller: MitigationController):
    """Run predefined queries demonstrating all mitigation branches."""
    demo_queries = [
        ("Calculate 14 * 25 + (100 / 4)", "Arithmetic expression"),
        ("Which one is better?", "Underspecified / Ambiguous query"),
        ("What is the elevation of Mount Everest according to the 2020 survey?", "Factual query in corpus"),
        ("Will Bitcoin hit $500,000 in 2035?", "Speculative / Unknowable question"),
        ("Explain the general concept of photosynthesis in plants.", "Standard general knowledge query"),
    ]

    print("\n" + "=" * 70)
    print(" RUNNING DEMONSTRATION SUITE ACROSS CANDIDATE ACTIONS")
    print("=" * 70)

    for query, description in demo_queries:
        print(f"\n>>> Input: '{query}' ({description})")
        result = controller.process_query(query)
        display_result(result, compact=True)


def display_result(res: ExecutionResult, compact: bool = False):
    """Print readable structured outcome to terminal."""
    print("-" * 70)
    print(f"Run ID:            {res.run_id}")
    print(f"Policy:            {res.policy_name}")
    print(f"Selected Action:   [{res.action_executed.upper()}]")
    print(f"Decision Reason:   {res.decision.decision_rationale}")
    print(f"Heuristic Uncert:  {res.signals.heuristic_uncertainty_score} (0.0=min, 1.0=max)")
    
    if res.model_draft:
        d = res.model_draft
        print(f"Model ID:          {d.model_id} (Provider: {d.provider}, Prompt: {d.prompt_version})")
        if d.model_reported_confidence is not None:
            print(f"Model Confidence:  {d.model_reported_confidence:.2f} (Hypothesis feature, NOT ground truth)")

    if res.cost:
        print(f"Operational Cost:  {res.cost.latency_ms:.1f} ms | {res.cost.total_tokens} tokens | {res.cost.tool_calls_count} tool calls | ${res.cost.monetary_cost_usd:.6f}")
    
    if res.execution_time_comparison:
        c = res.execution_time_comparison
        print(f"Direct Exec Time:  {c.direct_latency_ms:.2f} ms")
        print(f"AUDI Exec Time:    {c.audi_latency_ms:.2f} ms (Overhead: {c.audi_decision_overhead_ms:.3f} ms, Action: {c.audi_action_execution_ms:.2f} ms)")
        print(f"Latency Delta:     {'+' if c.latency_delta_ms >= 0 else ''}{c.latency_delta_ms:.2f} ms ({c.speedup_factor:.1f}x speedup)")
        print(f"Efficiency Note:   {c.efficiency_summary}")

    print(f"Verification:      {res.verification_result.overall_status.value if res.verification_result else 'Not Checked'}")
    print(f"Final Answer:      {res.final_answer}")
    
    if not compact:
        if res.retrieval_result and res.retrieval_result.passages:
            print("\n--- Retrieved Evidence ---")
            for p in res.retrieval_result.passages:
                print(f"  * [{p.doc_title}] (Relevance {p.relevance_score:.3f}): {p.text[:120]}...")
        if res.decision.candidate_utility_scores:
            print("\n--- Heuristic Candidate Utilities [Benefit - lambda * Cost] ---")
            for a, score in res.decision.candidate_utility_scores.items():
                print(f"  {a:10s}: {score:.3f}")
    print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="Adaptive Hallucination Mitigation Controller (Phase 2)")
    parser.add_argument("--query", "-q", type=str, help="Single query to process")
    parser.add_argument("--demo", "-d", action="store_true", help="Run demonstration suite")
    parser.add_argument("--lambda-val", "-l", type=float, default=0.5, help="Cost sensitivity tradeoff parameter lambda (default: 0.5)")
    parser.add_argument("--policy", "-p", type=str, default=PolicyType.ADAPTIVE.value, help="Routing policy to evaluate")
    parser.add_argument("--provider", type=str, default="offline", choices=["offline", "openai", "gemini", "ollama"], help="LLM Provider adapter")
    parser.add_argument("--model", "-m", type=str, default=None, help="Model identifier")
    parser.add_argument("--prompt-version", type=str, default="v1.0-standard", help="Prompt template version")
    parser.add_argument("--temperature", "-t", type=float, default=0.2, help="Sampling temperature")
    args = parser.parse_args()

    adapter = get_model_adapter(provider_name=args.provider, model_id=args.model)
    gen_settings = GenerationSettings(
        temperature=args.temperature,
        prompt_version=args.prompt_version,
    )

    controller = MitigationController(
        model_adapter=adapter,
        tradeoff_lambda=args.lambda_val,
        default_generation_settings=gen_settings
    )

    if args.query:
        result = controller.process_query(
            args.query,
            policy_name=args.policy,
            tradeoff_lambda=args.lambda_val,
            generation_settings=gen_settings,
        )
        display_result(result)
    elif args.demo:
        run_demo(controller)
    else:
        run_interactive(controller)


if __name__ == "__main__":
    main()
