"""
Clarification Action Handler (Phase 3).

Manages interactive clarification dialogs:
1. Detects underspecified variables, missing criteria, or ambiguous referents.
2. Formulates targeted clarification questions with suggested options.
3. Ingests user clarifications and synthesizes disambiguated contextual queries.
4. Accounts for additional interaction overhead and tracks resolution quality.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from src.signals import QuerySignals


@dataclass
class ClarificationSession:
    """Represents an active or resolved clarification dialogue turn."""
    original_query: str
    clarification_prompt: str
    suggested_options: List[str]
    missing_dimension: str
    user_response: Optional[str] = None
    disambiguated_query: Optional[str] = None
    is_resolved: bool = False
    interaction_turns: int = 1
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "clarification_prompt": self.clarification_prompt,
            "suggested_options": self.suggested_options,
            "missing_dimension": self.missing_dimension,
            "user_response": self.user_response,
            "disambiguated_query": self.disambiguated_query,
            "is_resolved": self.is_resolved,
            "interaction_turns": self.interaction_turns,
            "latency_ms": round(self.latency_ms, 2),
        }


class ClarificationHandler:
    """Formulates and resolves targeted clarification questions."""

    @classmethod
    def initiate_clarification(cls, signals: QuerySignals) -> ClarificationSession:
        """Formulate a targeted clarification request based on extracted signals."""
        start_time = time.perf_counter()
        query = signals.query_text
        lower_q = query.lower()

        # Categorize missing dimension and formulate options
        if signals.is_empty:
            missing = "Missing question content"
            prompt = "Your inquiry was empty. Please specify the question or subject you would like help with."
            options = ["Ask a factual question", "Perform an arithmetic calculation", "Verify a claim"]

        elif "which one is better" in lower_q or "what is better" in lower_q:
            missing = "Comparison candidates and evaluation criteria"
            prompt = (
                "To determine which option is better, please clarify: "
                "(1) Which alternatives are you comparing? and "
                "(2) What criteria matter most to you (e.g., speed, cost, reliability, ease of use)?"
            )
            options = ["Compare by cost", "Compare by speed/performance", "Compare by reliability"]

        elif "is it good" in lower_q or "is this good" in lower_q:
            missing = "Subject referent and context"
            prompt = (
                "The pronoun 'it' has no defined referent in the query. "
                "Please clarify what specific product, concept, or approach you are referring to."
            )
            options = ["Specify software library", "Specify hardware device", "Specify methodology"]

        elif "why did it happen" in lower_q:
            missing = "Event referent"
            prompt = "Please clarify which specific event, phenomenon, or outcome you are asking about."
            options = ["Historical event", "Technical/software failure", "Scientific phenomenon"]

        elif signals.ambiguity_reason:
            missing = signals.ambiguity_reason
            prompt = (
                f"The inquiry is underspecified: {signals.ambiguity_reason}. "
                "Could you please clarify by providing additional background or specifying the scope?"
            )
            options = ["Provide more background", "Limit scope to specific domain"]

        else:
            missing = "General context"
            prompt = "The question permits multiple interpretations. Please clarify your intended focus."
            options = ["Provide specific examples", "Clarify target domain"]

        latency = (time.perf_counter() - start_time) * 1000.0

        return ClarificationSession(
            original_query=query,
            clarification_prompt=prompt,
            suggested_options=options,
            missing_dimension=missing,
            is_resolved=False,
            latency_ms=latency,
        )

    @classmethod
    def resolve_clarification(
        cls,
        session: ClarificationSession,
        user_response: str
    ) -> ClarificationSession:
        """Merge user response into a fully contextualized query."""
        start_time = time.perf_counter()
        resp_clean = user_response.strip()

        # Check if the user response is already a complete query or tool invocation
        math_ops = set("+-*/=^%")
        if any(c in resp_clean for c in math_ops) or resp_clean.lower().startswith(("calculate", "compute", "what is", "who is", "when did")):
            disambiguated = resp_clean
        elif session.original_query:
            disambiguated = f"{session.original_query.rstrip('?.')} (Clarified: {resp_clean})"
        else:
            disambiguated = resp_clean

        session.user_response = resp_clean
        session.disambiguated_query = disambiguated
        session.is_resolved = True
        session.interaction_turns += 1
        session.latency_ms += (time.perf_counter() - start_time) * 1000.0

        return session
