"""
Cost and Performance Accounting Service.

Tracks operational expenditure with strict distinction between:
1. Estimated illustrative cost units (prior action costs for routing decisions).
2. Measured runtime costs (actual elapsed latency, token counts, tool invocations, and computed USD).
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class CostAccounting:
    """Detailed record of estimated and measured execution resources."""
    action_name: str
    
    # 1. Prior estimated cost units (dimensionless decision weights)
    estimated_action_cost_units: float = 1.0
    
    # 2. Measured runtime resource utilization
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model_calls_count: int = 0
    tool_calls_count: int = 0
    
    # 3. Measured monetary cost in USD (calculated from actual token rates)
    monetary_cost_usd: float = 0.0
    currency: str = "USD"
    is_api_billed: bool = False
    
    breakdown_notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_name": self.action_name,
            "estimated_action_cost_units": round(self.estimated_action_cost_units, 2),
            "latency_ms": round(self.latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model_calls_count": self.model_calls_count,
            "tool_calls_count": self.tool_calls_count,
            "monetary_cost_usd": round(self.monetary_cost_usd, 6),
            "currency": self.currency,
            "is_api_billed": self.is_api_billed,
            "breakdown_notes": self.breakdown_notes,
        }


class CostTracker:
    """Manages baseline cost priors and calculates measured execution metrics."""

    # Prior illustrative action cost units (dimensionless relative effort)
    # Used in the objective: a* = argmax [Benefit(a) - lambda * Cost(a)]
    ACTION_COST_PRIORS: Dict[str, float] = {
        "calculate": 0.2,   # Fast deterministic local evaluation
        "abstain": 0.4,     # Fast refusal, minimal computation
        "direct": 1.0,      # Single forward LLM pass or mock generation
        "clarify": 1.5,     # Generation of targeted question + user turn
        "retrieve": 3.5,    # Index lookup + document parsing + prompt packing
        "verify": 5.0,      # Retrieval + claim decomposition + alignment checks
        "revise": 6.5,      # Verification + secondary conditional generation
    }

    # Pricing reference rates per 1,000,000 tokens (e.g. standard lightweight model)
    RATE_INPUT_PER_M = 0.15   # $0.15 per 1M prompt tokens
    RATE_OUTPUT_PER_M = 0.60  # $0.60 per 1M completion tokens

    @classmethod
    def get_action_cost_prior(cls, action_name: str) -> float:
        """Fetch dimensionless prior cost estimate for a candidate action."""
        return cls.ACTION_COST_PRIORS.get(action_name.lower(), 1.0)

    @classmethod
    def compute_monetary_cost(cls, prompt_tokens: int, completion_tokens: int, is_billed_provider: bool) -> float:
        """Calculate true USD monetary cost based on actual provider billing."""
        if not is_billed_provider:
            return 0.0
        cost_input = (prompt_tokens / 1_000_000.0) * cls.RATE_INPUT_PER_M
        cost_output = (completion_tokens / 1_000_000.0) * cls.RATE_OUTPUT_PER_M
        return cost_input + cost_output

    @classmethod
    def create_record(
        cls,
        action_name: str,
        start_time_perf: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model_calls: int = 0,
        tool_calls: int = 0,
        is_billed: bool = False,
        notes: Optional[Dict[str, Any]] = None,
    ) -> CostAccounting:
        """Create a complete accounting record upon action completion."""
        latency = (time.perf_counter() - start_time_perf) * 1000.0
        total_tok = prompt_tokens + completion_tokens
        usd = cls.compute_monetary_cost(prompt_tokens, completion_tokens, is_billed)
        prior_units = cls.get_action_cost_prior(action_name)

        return CostAccounting(
            action_name=action_name,
            estimated_action_cost_units=prior_units,
            latency_ms=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tok,
            model_calls_count=model_calls,
            tool_calls_count=tool_calls,
            monetary_cost_usd=usd,
            currency="USD",
            is_api_billed=is_billed,
            breakdown_notes=notes or {},
        )
