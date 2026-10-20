"""
Query and Uncertainty Signals Extraction.

Extracts inspectable features from user queries to inform routing decisions.
All features and confidence/uncertainty scores are explicitly computed heuristics
and are clearly labeled as such without fabricating artificial precision.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class QuerySignals:
    """Structured representation of observable query features and uncertainty signals."""
    query_text: str
    is_empty: bool = False
    char_length: int = 0
    token_count_approx: int = 0
    
    # Arithmetic cues
    has_arithmetic_expr: bool = False
    extracted_arithmetic_expr: Optional[str] = None
    
    # Ambiguity cues
    is_ambiguous: bool = False
    ambiguity_reason: Optional[str] = None
    
    # Knowledge / Factual cues
    is_factual_knowledge_request: bool = False
    factual_keywords_detected: List[str] = field(default_factory=list)
    requires_temporal_freshness: bool = False
    
    # Speculation / Unknowable cues
    is_unknowable_or_speculative: bool = False
    speculation_reason: Optional[str] = None
    
    # Heuristic uncertainty score (0.0 = low uncertainty / high confidence, 1.0 = high uncertainty)
    # Note: Explicitly labeled as a heuristic prior based on observable signals.
    heuristic_uncertainty_score: float = 0.5
    signal_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_text": self.query_text,
            "is_empty": self.is_empty,
            "char_length": self.char_length,
            "token_count_approx": self.token_count_approx,
            "has_arithmetic_expr": self.has_arithmetic_expr,
            "extracted_arithmetic_expr": self.extracted_arithmetic_expr,
            "is_ambiguous": self.is_ambiguous,
            "ambiguity_reason": self.ambiguity_reason,
            "is_factual_knowledge_request": self.is_factual_knowledge_request,
            "factual_keywords_detected": self.factual_keywords_detected,
            "requires_temporal_freshness": self.requires_temporal_freshness,
            "is_unknowable_or_speculative": self.is_unknowable_or_speculative,
            "speculation_reason": self.speculation_reason,
            "heuristic_uncertainty_score": round(self.heuristic_uncertainty_score, 3),
            "signal_summary": self.signal_summary,
        }


class SignalExtractor:
    """Extracts features and uncertainty cues from user queries."""

    # Common ambiguous queries lacking context or referents
    AMBIGUOUS_PATTERNS = [
        (r"^(which|what)\s+(one|of\s+these)\s+is\s+(better|best|faster|worse)\??$", "Missing comparison options and criteria"),
        (r"^is\s+(it|this|that)\s+(good|better|worth\s+it|safe|true)\??$", "Missing referent context for 'it/this/that'"),
        (r"^(why|how)\s+did\s+(it|that|they)\s+happen\??$", "Missing subject context"),
        (r"^(what|how)\s+about\s+(it|that)\??$", "Underspecified conversational fragment"),
        (r"^(tell\s+me|show\s+me)\s+more\??$", "Missing topic referent"),
        (r"^compare\s+them\??$", "Missing entities to compare"),
    ]

    # Arithmetic patterns
    # Matches expressions like "14 * 25", "calculate (120 + 45) / 5", "what is 2 ^ 8"
    ARITHMETIC_PREFIX_RE = re.compile(
        r"^(what\s+is|calculate|compute|evaluate|solve)?\s*([0-9\.\s\+\-\*\/\^\%\(\)\,\_]+)\??$",
        re.IGNORECASE
    )
    # Math symbols present
    MATH_OPERATORS_RE = re.compile(r"[\+\-\*\/\^\%\=]")

    # Factual knowledge seeking triggers
    FACTUAL_KEYWORDS = [
        "who was", "who is", "who founded", "who invented", "who directed", "who wrote",
        "when was", "when did", "what year", "what date", "where is", "where was",
        "capital of", "population of", "highest mountain", "deepest ocean",
        "nobel prize", "author of", "president of", "ceo of", "headquarters of",
        "released in", "discovered by", "formula for", "speed of light",
        "elevation of", "height of", "depth of", "mount everest", "apollo 11",
        "moon landing", "alan turing", "turing machine", "photoelectric effect",
        "how tall is", "how high is", "average rainfall", "rainfall in", "temperature of"
    ]

    # Temporal freshness / live event queries
    TEMPORAL_KEYWORDS = [
        "current", "currently", "latest", "today", "yesterday", "this year",
        "recent", "newest", "upcoming", "2024", "2025", "2026", "right now"
    ]

    # Speculation / Unknowable future or private state patterns
    SPECULATION_PATTERNS = [
        (r"will\s+.+\s+(reach|hit|drop|rise|win|crash).*20[3-9]\d", "Predicting future market or world events"),
        (r"will\s+.+\s+(in\s+20[3-9]\d|by\s+20[3-9]\d)", "Predicting future outcomes beyond current knowledge horizon"),
        (r"predict\s+(the\s+)?future", "Explicit future prediction request"),
        (r"what\s+am\s+i\s+thinking", "Unknowable private human mental state"),
        (r"what\s+will\s+happen\s+on\s+[A-Za-z]+\s+\d{1,2},\s+20\d\d", "Predicting exact future event"),
        (r"who\s+will\s+win\s+the\s+(next|20[3-9]\d)\s+(world\s+cup|election|super\s+bowl)", "Future outcome speculation"),
        (r"meaning\s+of\s+life", "Subjective philosophical inquiry lacking empirical answer"),
        (r"does\s+god\s+exist", "Metaphysical proposition outside empirical verification")
    ]

    @classmethod
    def extract(cls, query: str) -> QuerySignals:
        """Analyze query and extract observable signals."""
        cleaned = query.strip() if query else ""
        
        if not cleaned:
            return QuerySignals(
                query_text="",
                is_empty=True,
                char_length=0,
                token_count_approx=0,
                is_ambiguous=True,
                ambiguity_reason="Empty or whitespace-only query",
                heuristic_uncertainty_score=1.0,
                signal_summary={"status": "empty"}
            )
        
        char_len = len(cleaned)
        approx_tokens = len(cleaned.split())
        
        # 1. Check Ambiguity
        is_ambiguous = False
        ambiguity_reason = None
        lower_q = cleaned.lower()
        
        for pattern, reason in cls.AMBIGUOUS_PATTERNS:
            if re.search(pattern, lower_q):
                is_ambiguous = True
                ambiguity_reason = reason
                break
        
        # Also check very short queries with missing context (e.g. 1-2 words that are not greetings or math)
        if not is_ambiguous and approx_tokens <= 2 and not any(op in cleaned for op in "+-*/="):
            if lower_q not in {"hello", "hi", "help"}:
                is_ambiguous = True
                ambiguity_reason = "Query is too brief and lacks specification or actionable context"

        # 2. Check Arithmetic
        has_arithmetic = False
        arithmetic_expr = None
        
        # Check explicit math patterns
        math_match = cls.ARITHMETIC_PREFIX_RE.match(cleaned)
        if math_match:
            candidate_expr = math_match.group(2).strip()
            # Must contain at least one digit and one operator or multiple digits
            if re.search(r"\d", candidate_expr) and cls.MATH_OPERATORS_RE.search(candidate_expr):
                has_arithmetic = True
                arithmetic_expr = candidate_expr
        
        # Fallback check for queries like "compute 45 * (12 + 3)" or "25 + 75"
        if not has_arithmetic:
            tokens_in_q = cleaned.replace("?", "").replace("=", "").split()
            # If all or most tokens are numbers or operators
            math_chars_ratio = sum(1 for c in cleaned if c in "0123456789+-*/().^% ") / max(len(cleaned), 1)
            if math_chars_ratio > 0.65 and re.search(r"\d", cleaned) and cls.MATH_OPERATORS_RE.search(cleaned):
                has_arithmetic = True
                # Clean up query into expression
                expr_cand = re.sub(r"^(what\s+is|calculate|compute|solve)\s*", "", cleaned, flags=re.IGNORECASE).rstrip("?=")
                arithmetic_expr = expr_cand.strip()

        # 3. Check Speculation / Unknowable
        is_unknowable = False
        speculation_reason = None
        for pattern, reason in cls.SPECULATION_PATTERNS:
            if re.search(pattern, lower_q):
                is_unknowable = True
                speculation_reason = reason
                break

        # 4. Check Factual Knowledge Request
        is_factual = False
        detected_keywords = []
        for kw in cls.FACTUAL_KEYWORDS:
            if kw in lower_q:
                is_factual = True
                detected_keywords.append(kw)
        
        # Check temporal freshness
        requires_freshness = False
        for kw in cls.TEMPORAL_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", lower_q):
                requires_freshness = True
                if not is_factual:
                    is_factual = True
                    detected_keywords.append(kw)
        
        # 5. Compute Heuristic Prior Uncertainty Score
        # 0.0 (high confidence/simple) to 1.0 (high uncertainty/risk)
        uncertainty = 0.35  # baseline
        if is_unknowable:
            uncertainty = 0.95
        elif is_ambiguous:
            uncertainty = 0.85
        elif has_arithmetic:
            # Deterministic math has low semantic uncertainty because an exact tool can resolve it
            uncertainty = 0.10
        elif is_factual:
            # Factual queries carry hallucination risk if answered without evidence
            uncertainty = 0.70 if requires_freshness else 0.60
        else:
            # Standard conversational/definitional
            uncertainty = 0.30

        summary = {
            "has_arithmetic": has_arithmetic,
            "is_ambiguous": is_ambiguous,
            "is_factual": is_factual,
            "is_unknowable": is_unknowable,
            "requires_freshness": requires_freshness,
        }

        return QuerySignals(
            query_text=cleaned,
            is_empty=False,
            char_length=char_len,
            token_count_approx=approx_tokens,
            has_arithmetic_expr=has_arithmetic,
            extracted_arithmetic_expr=arithmetic_expr,
            is_ambiguous=is_ambiguous,
            ambiguity_reason=ambiguity_reason,
            is_factual_knowledge_request=is_factual,
            factual_keywords_detected=detected_keywords,
            requires_temporal_freshness=requires_freshness,
            is_unknowable_or_speculative=is_unknowable,
            speculation_reason=speculation_reason,
            heuristic_uncertainty_score=uncertainty,
            signal_summary=summary,
        )
