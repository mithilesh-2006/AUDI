"""
Routing Policy Engine.

Defines decision policies for selecting mitigation actions:
1. AdaptiveRulePolicy: Heuristic uncertainty and cost-aware controller.
2. DirectOnlyPolicy: Always answers directly (baseline).
3. AlwaysRetrievePolicy: Always retrieves evidence (baseline).
4. FixedVerifyPolicy: Always retrieves and verifies claims (baseline).

Formulation:
a* = argmax_a [EstimatedBenefit(a | x) - lambda * EstimatedCost(a)]
All scores are explicitly labeled as heuristic decision priors.
"""

from __future__ import annotations
import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from src.signals import QuerySignals
from src.cost_tracker import CostTracker


class PolicyType(str, enum.Enum):
    ADAPTIVE = "Adaptive Policy (Rule & Cost-Aware)"
    LEARNED_COST_AWARE = "Calibrated Cost-Aware Policy (Phase 4)"
    DIRECT_ONLY = "Direct-Only Baseline"
    ALWAYS_RETRIEVE = "Always-Retrieve Baseline"
    FIXED_VERIFY = "Fixed-Verify Baseline"


@dataclass
class DecisionOutcome:
    """The controller's action selection and transparent reasoning."""
    selected_action: str
    decision_rationale: str
    policy_name: str
    tradeoff_lambda: float
    candidate_utility_scores: Dict[str, float] = field(default_factory=dict)
    next_step_description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_action": self.selected_action,
            "decision_rationale": self.decision_rationale,
            "policy_name": self.policy_name,
            "tradeoff_lambda": round(self.tradeoff_lambda, 3),
            "candidate_utility_scores": {k: round(v, 3) for k, v in self.candidate_utility_scores.items()},
            "next_step_description": self.next_step_description,
        }


class BasePolicy(ABC):
    """Abstract base class for all routing policies."""

    @abstractmethod
    def select_action(self, signals: QuerySignals, tradeoff_lambda: float = 0.5) -> DecisionOutcome:
        pass


class AdaptiveRulePolicy(BasePolicy):
    """
    Primary adaptive policy that inspects observable features, uncertainty cues,
    and operational cost to select the most suitable mitigation action.
    """

    def __init__(self, default_lambda: float = 0.5):
        self.default_lambda = default_lambda

    def _estimate_heuristic_benefits(self, signals: QuerySignals) -> Dict[str, float]:
        """
        Estimate the heuristic benefit (expected error/loss reduction) for each candidate action.
        Note: These are initial calibrated domain heuristics, not ground-truth model weights.
        """
        benefits = {
            "direct": 0.3,
            "calculate": 0.05,
            "retrieve": 0.2,
            "clarify": 0.1,
            "verify": 0.25,
            "abstain": 0.05,
        }

        if signals.is_empty:
            benefits["clarify"] = 0.95
            benefits["direct"] = 0.05
        elif signals.is_ambiguous:
            benefits["clarify"] = 0.90
            benefits["direct"] = 0.10
            benefits["retrieve"] = 0.20
        elif signals.has_arithmetic_expr:
            benefits["calculate"] = 0.98
            benefits["direct"] = 0.15
        elif signals.is_unknowable_or_speculative:
            benefits["abstain"] = 0.92
            benefits["direct"] = 0.05
        elif signals.is_factual_knowledge_request:
            benefits["retrieve"] = 0.85
            benefits["verify"] = 0.90
            benefits["direct"] = 0.20
        else:
            # General / conversational
            benefits["direct"] = 0.80

        return benefits

    def select_action(self, signals: QuerySignals, tradeoff_lambda: float = 0.5) -> DecisionOutcome:
        tradeoff_lambda = max(0.0, min(tradeoff_lambda, 1.5))
        benefits = self._estimate_heuristic_benefits(signals)

        # Compute net utility: U(a) = Benefit(a) - (lambda * 0.1) * CostPrior(a)
        # Scaled so lambda=0.5 provides a balanced research trade-off
        cost_scale = 0.10 * tradeoff_lambda
        utilities: Dict[str, float] = {}
        for action, benefit in benefits.items():
            cost_prior = CostTracker.get_action_cost_prior(action)
            utilities[action] = benefit - (cost_scale * cost_prior)

        # Primary rule evaluation with transparent rationale
        if signals.is_empty:
            return DecisionOutcome(
                selected_action="clarify",
                decision_rationale="Query is empty or whitespace-only; cannot proceed without user input.",
                policy_name=PolicyType.ADAPTIVE.value,
                tradeoff_lambda=tradeoff_lambda,
                candidate_utility_scores=utilities,
                next_step_description="Prompt user to specify their question."
            )

        if signals.is_ambiguous:
            return DecisionOutcome(
                selected_action="clarify",
                decision_rationale=f"Query is ambiguous or underspecified: {signals.ambiguity_reason}.",
                policy_name=PolicyType.ADAPTIVE.value,
                tradeoff_lambda=tradeoff_lambda,
                candidate_utility_scores=utilities,
                next_step_description="Request clarification of alternatives, criteria, or missing referents."
            )

        if signals.has_arithmetic_expr:
            return DecisionOutcome(
                selected_action="calculate",
                decision_rationale=(
                    f"Deterministic arithmetic detected ('{signals.extracted_arithmetic_expr}'). "
                    "Routing to sandboxed AST calculator avoids model arithmetic hallucination."
                ),
                policy_name=PolicyType.ADAPTIVE.value,
                tradeoff_lambda=tradeoff_lambda,
                candidate_utility_scores=utilities,
                next_step_description="Execute calculation deterministically and return exact result."
            )

        if signals.is_unknowable_or_speculative:
            return DecisionOutcome(
                selected_action="abstain",
                decision_rationale=(
                    f"Query requests unknowable or speculative information: {signals.speculation_reason}. "
                    "Abstaining prevents fabricated assertions."
                ),
                policy_name=PolicyType.ADAPTIVE.value,
                tradeoff_lambda=tradeoff_lambda,
                candidate_utility_scores=utilities,
                next_step_description="Issue transparent abstention stating why the request cannot be answered reliably."
            )

        if signals.is_factual_knowledge_request:
            # Cost sensitivity check: if lambda is very high (extreme cost penalty),
            # the system might downgrade verification to retrieval alone
            if tradeoff_lambda > 0.9:
                return DecisionOutcome(
                    selected_action="retrieve",
                    decision_rationale=(
                        "Factual query requiring evidence. Under high cost-penalty (lambda > 0.9), "
                        "retrieval is selected over verification to minimize latency and token expenditure."
                    ),
                    policy_name=PolicyType.ADAPTIVE.value,
                    tradeoff_lambda=tradeoff_lambda,
                    candidate_utility_scores=utilities,
                    next_step_description="Fetch relevant passages from local corpus and ground answer."
                )
            else:
                return DecisionOutcome(
                    selected_action="retrieve",
                    decision_rationale=(
                        f"Factual knowledge inquiry detected (cues: {', '.join(signals.factual_keywords_detected[:3])}). "
                        "Retrieval is selected to ground claims in authoritative evidence."
                    ),
                    policy_name=PolicyType.ADAPTIVE.value,
                    tradeoff_lambda=tradeoff_lambda,
                    candidate_utility_scores=utilities,
                    next_step_description="Query corpus for matching passages, followed by claim verification."
                )

        # Default standard / low-risk query
        return DecisionOutcome(
            selected_action="direct",
            decision_rationale="No ambiguity, arithmetic, or specialized factual knowledge cues detected. Direct generation suffices.",
            policy_name=PolicyType.ADAPTIVE.value,
            tradeoff_lambda=tradeoff_lambda,
            candidate_utility_scores=utilities,
            next_step_description="Generate answer directly without external tool overhead."
        )


class DirectOnlyPolicy(BasePolicy):
    """Baseline policy that unconditionally answers directly."""

    def select_action(self, signals: QuerySignals, tradeoff_lambda: float = 0.5) -> DecisionOutcome:
        return DecisionOutcome(
            selected_action="direct",
            decision_rationale="Direct-Only baseline: unconditionally generates response without tool calls or verification.",
            policy_name=PolicyType.DIRECT_ONLY.value,
            tradeoff_lambda=tradeoff_lambda,
            candidate_utility_scores={"direct": 1.0},
            next_step_description="Generate direct draft."
        )


class AlwaysRetrievePolicy(BasePolicy):
    """Baseline policy that unconditionally triggers retrieval for all queries."""

    def select_action(self, signals: QuerySignals, tradeoff_lambda: float = 0.5) -> DecisionOutcome:
        return DecisionOutcome(
            selected_action="retrieve",
            decision_rationale="Always-Retrieve baseline: unconditionally executes evidence retrieval for every query.",
            policy_name=PolicyType.ALWAYS_RETRIEVE.value,
            tradeoff_lambda=tradeoff_lambda,
            candidate_utility_scores={"retrieve": 1.0},
            next_step_description="Fetch passages from corpus regardless of query type."
        )


class FixedVerifyPolicy(BasePolicy):
    """Baseline policy that unconditionally retrieves evidence and runs full verification."""

    def select_action(self, signals: QuerySignals, tradeoff_lambda: float = 0.5) -> DecisionOutcome:
        return DecisionOutcome(
            selected_action="verify",
            decision_rationale="Fixed-Verify baseline: unconditionally executes retrieval and claim-level verification.",
            policy_name=PolicyType.FIXED_VERIFY.value,
            tradeoff_lambda=tradeoff_lambda,
            candidate_utility_scores={"verify": 1.0},
            next_step_description="Retrieve evidence, check every claim, and revise if unsupported."
        )


class LearnedCostAwarePolicy(BasePolicy):
    """
    Phase 4 Cost-Aware Policy.
    Minimizes expected quality loss plus cost penalty:
        a* = argmin_a [ L_quality(a | x) + lambda * cost_scaling * CostPrior(a) ]
    Equivalently maximizes net utility:
        U(a) = (1.0 - L_quality(a | x)) - lambda * cost_scaling * CostPrior(a)
    Uses calibrated uncertainty and empirical loss estimates derived from development data.
    """

    def __init__(self, default_lambda: float = 0.5, cost_scaling: float = 0.08):
        self.default_lambda = default_lambda
        self.cost_scaling = cost_scaling

    def estimate_quality_losses(self, signals: QuerySignals, calibrated_uncertainty: float) -> Dict[str, float]:
        """
        Estimates expected factual/quality loss L(a | x) in [0.0, 1.0] for each candidate action.
        Lower loss represents higher answer reliability.
        """
        losses: Dict[str, float] = {
            "direct": min(1.0, max(0.05, calibrated_uncertainty)),
            "calculate": 0.95,
            "retrieve": 0.35,
            "clarify": 0.70,
            "verify": 0.25,
            "abstain": 0.80,
        }

        if signals.is_empty:
            losses["clarify"] = 0.05
            losses["direct"] = 0.98
        elif signals.is_ambiguous:
            losses["clarify"] = 0.08
            losses["direct"] = 0.85
            losses["retrieve"] = 0.60
        elif signals.has_arithmetic_expr:
            losses["calculate"] = 0.01  # Deterministic AST calculator has virtually 0 error
            losses["direct"] = 0.85     # Models regularly hallucinate multi-step arithmetic
        elif signals.is_unknowable_or_speculative:
            losses["abstain"] = 0.05    # Abstention on unanswerable incurs near-zero hallucination loss
            losses["direct"] = 0.98     # Hallucinating on unknowables incurs maximum penalty
        elif signals.is_factual_knowledge_request:
            # For factual queries, retrieval and verification substantially reduce quality loss
            losses["retrieve"] = max(0.08, 0.25 * calibrated_uncertainty)
            losses["verify"] = max(0.04, 0.15 * calibrated_uncertainty)
            losses["direct"] = min(0.95, 0.40 + 0.55 * calibrated_uncertainty)
        else:
            # Low-risk standard queries
            losses["direct"] = max(0.05, calibrated_uncertainty * 0.3)

        return losses

    def select_action(
        self,
        signals: QuerySignals,
        tradeoff_lambda: float = 0.5,
        calibrated_uncertainty: Optional[float] = None,
    ) -> DecisionOutcome:
        tradeoff_lambda = max(0.0, min(tradeoff_lambda, 1.5))
        u = calibrated_uncertainty if calibrated_uncertainty is not None else signals.heuristic_uncertainty_score
        losses = self.estimate_quality_losses(signals, u)

        # Objective: a* = argmin [Loss(a) + lambda * cost_scaling * CostPrior(a)]
        action_scores: Dict[str, float] = {}
        utilities: Dict[str, float] = {}
        for action, loss in losses.items():
            prior_cost = CostTracker.get_action_cost_prior(action)
            cost_penalty = tradeoff_lambda * self.cost_scaling * prior_cost
            total_loss = loss + cost_penalty
            action_scores[action] = total_loss
            utilities[action] = round(1.0 - total_loss, 4)

        best_action = min(action_scores, key=action_scores.get)
        best_loss = action_scores[best_action]

        rationale = (
            f"Phase 4 Cost-Sensitive Selection: Action '{best_action}' minimizes expected loss + cost penalty "
            f"[L={losses[best_action]:.3f}, CostPrior={CostTracker.get_action_cost_prior(best_action):.1f}, "
            f"lambda={tradeoff_lambda:.2f}, net objective={best_loss:.3f}]."
        )

        return DecisionOutcome(
            selected_action=best_action,
            decision_rationale=rationale,
            policy_name=PolicyType.LEARNED_COST_AWARE.value,
            tradeoff_lambda=tradeoff_lambda,
            candidate_utility_scores=utilities,
            next_step_description=f"Execute {best_action} under calibrated cost-sensitive objective."
        )

