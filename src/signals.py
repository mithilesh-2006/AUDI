"""
Query and Uncertainty Signals Extraction.

Extracts inspectable features from user queries to inform routing decisions.
All features and confidence/uncertainty scores are explicitly computed heuristics
and are clearly labeled as such without fabricating artificial precision.

Includes the Antigravity Fallback Interceptor to catch semantic drift,
floating dual referents, and unanchored comparative queries before they
cascade down ungrounded execution paths.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple


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

    # Antigravity Fallback Interceptor telemetry
    antigravity_intercepted: bool = False
    floating_referents_detected: List[str] = field(default_factory=list)
    comparative_triggers_detected: List[str] = field(default_factory=list)
    
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
            "antigravity_intercepted": self.antigravity_intercepted,
            "floating_referents_detected": self.floating_referents_detected,
            "comparative_triggers_detected": self.comparative_triggers_detected,
            "heuristic_uncertainty_score": round(self.heuristic_uncertainty_score, 3),
            "signal_summary": self.signal_summary,
        }


class SignalExtractor:
    """Extracts features, uncertainty cues, and executes the Antigravity Fallback Interceptor."""

    # Common ambiguous queries lacking context or referents
    AMBIGUOUS_PATTERNS = [
        (r"^(which|what)\s+(one|of\s+these)\s+is\s+(better|best|faster|worse)\??$", "Missing comparison options and criteria"),
        (r"^is\s+(it|this|that)\s+(good|better|worth\s+it|safe|true)\??$", "Missing referent context for 'it/this/that'"),
        (r"^(why|how)\s+did\s+(it|that|they)\s+happen\??$", "Missing subject context"),
        (r"^(what|how)\s+about\s+(it|that)\??$", "Underspecified conversational fragment"),
        (r"^(tell\s+me|show\s+me)\s+more\??$", "Missing topic referent"),
        (r"^compare\s+them\??$", "Missing entities to compare"),
    ]

    # --- Antigravity Fallback Interceptor: Floating Dual Referents ---
    # Identifies dual-referent phrases when they lack explicit noun antecedents
    FLOATING_DUAL_REFERENT_PATTERNS = [
        (r"\bthe\s+two\b", "the two"),
        (r"\bboth\b", "both"),
        (r"\bthe\s+pair\b", "the pair"),
        (r"\bthe\s+options\b", "the options"),
        (r"\bthe\s+alternatives\b", "the alternatives"),
        (r"\bthese\s+two\b", "these two"),
        (r"\bthose\s+two\b", "those two"),
        (r"\beither\s+one\b", "either one"),
        (r"\beither\s+of\s+(them|these|those)\b", "either of them"),
    ]

    # --- Antigravity Fallback Interceptor: Comparative Triggers ---
    # Strict regex-sanitized comparative keywords
    COMPARATIVE_TRIGGER_PATTERNS = [
        (r"\bcompar(e|ing|ison|ative)\b", "compare"),
        (r"\bdifference(s)?\b", "difference"),
        (r"\baccuracy\s+between\b", "accuracy between"),
        (r"\bbetter\b", "better"),
        (r"\bworse\b", "worse"),
        (r"\bfaster\b", "faster"),
        (r"\bslower\b", "slower"),
        (r"\bsuperior\b", "superior"),
        (r"\binferior\b", "inferior"),
        (r"\b(versus|vs\.?)\b", "versus"),
    ]

    # Deterministic linguistic and imperative command verbs that are actionable
    DIRECTIVE_VERBS = {
        "spell", "define", "definition", "translate", "pronounce", "say", "repeat",
        "echo", "reverse", "capitalize", "print", "write", "count", "alphabetize",
        "summarize", "describe", "explain"
    }

    # Common greetings and single-word conversational phrases
    COMMON_CONVERSATIONAL = [
        "hello", "hi", "hey", "help", "thanks", "thank you", "bye", "goodbye",
        "good morning", "good afternoon", "good evening", "howdy", "sup", "welcome",
        "ping", "pong", "test"
    ]

    # Arithmetic patterns
    # Matches expressions like "14 * 25", "calculate (120 + 45) / 5", "what is 2 ^ 8"
    ARITHMETIC_PREFIX_RE = re.compile(
        r"^(what\s+is|calculate|compute|evaluate|solve)?\s*([0-9\.\s\+\-\*\/\^\%\(\)\,\_]+)\??$",
        re.IGNORECASE
    )
    # Math symbols present
    MATH_OPERATORS_RE = re.compile(r"[\+\-\*\/\^\%\=]")

    # Strict Regex Sanitized Factual Knowledge Triggers (Must match exact word boundaries)
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

    # Speculation / Unknowable future or private state patterns (Strict word boundaries)
    SPECULATION_PATTERNS = [
        (r"\bwill\s+.+\s+(reach|hit|drop|rise|win|crash).*20[3-9]\d\b", "Predicting future market or world events"),
        (r"\bwill\s+.+\s+(in\s+20[3-9]\d|by\s+20[3-9]\d)\b", "Predicting future outcomes beyond current knowledge horizon"),
        (r"\bpredict\s+(the\s+)?future\b", "Explicit future prediction request"),
        (r"\bwhat\s+am\s+i\s+thinking\b", "Unknowable private human mental state"),
        (r"\bwhat\s+will\s+happen\s+on\s+[A-Za-z]+\s+\d{1,2},\s+20\d\d\b", "Predicting exact future event"),
        (r"\bwho\s+will\s+win\s+the\s+(next|20[3-9]\d)\s+(world\s+cup|election|super\s+bowl)\b", "Future outcome speculation"),
        (r"\bmeaning\s+of\s+life\b", "Subjective philosophical inquiry lacking empirical answer"),
        (r"\bdoes\s+god\s+exist\b", "Metaphysical proposition outside empirical verification")
    ]

    # --- Compiled Regex Matchers with Strict Sanitization ---
    COMPILED_AMBIGUOUS_PATTERNS = [
        (re.compile(p, re.IGNORECASE), reason) for p, reason in AMBIGUOUS_PATTERNS
    ]
    COMPILED_SPECULATION_PATTERNS = [
        (re.compile(p, re.IGNORECASE), reason) for p, reason in SPECULATION_PATTERNS
    ]
    COMPILED_FLOATING_DUAL_PATTERNS = [
        (re.compile(p, re.IGNORECASE), label) for p, label in FLOATING_DUAL_REFERENT_PATTERNS
    ]
    COMPILED_COMPARATIVE_PATTERNS = [
        (re.compile(p, re.IGNORECASE), label) for p, label in COMPARATIVE_TRIGGER_PATTERNS
    ]
    COMPILED_FACTUAL_PATTERNS = [
        (kw, re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)) for kw in FACTUAL_KEYWORDS
    ]
    COMPILED_TEMPORAL_PATTERNS = [
        (kw, re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)) for kw in TEMPORAL_KEYWORDS
    ]
    COMPILED_CONVERSATIONAL_PATTERNS = [
        re.compile(r"^\s*" + re.escape(phrase) + r"(\b|[!?.])", re.IGNORECASE)
        for phrase in COMMON_CONVERSATIONAL
    ]

    @classmethod
    def _has_explicit_antecedents(cls, query: str) -> bool:
        """
        Determines whether explicit concrete comparison targets / noun antecedents
        are present in the query (preventing false-positive interception of grounded comparisons).
        """
        # 1. Quoted entity targets (e.g. 'Model A' and 'Model B')
        if len(re.findall(r"[\"'][^\"']+[\"']", query)) >= 2:
            return True

        # 2. Explicit capitalized proper nouns or technical entities (excluding query-initial word)
        words = query.strip().rstrip("?.!").split()
        if len(words) > 1:
            non_initial_words = words[1:]
            proper_nouns = [
                w for w in non_initial_words
                if (w[0].isupper() or any(c.isdigit() for c in w))
                and w.lower().strip("?,.!") not in {
                    "the", "a", "an", "and", "or", "vs", "versus", "with",
                    "both", "two", "pair", "options", "alternatives", "them", "which"
                }
            ]
            if len(proper_nouns) >= 2:
                return True
            if len(proper_nouns) >= 1 and any(sep in query.lower() for sep in [" and ", " vs ", " versus ", " between "]):
                return True

        # 3. Explicit "between X and Y" where X and Y are non-generic content words
        between_match = re.search(r"\bbetween\s+([a-zA-Z0-9_\-]+)\s+and\s+([a-zA-Z0-9_\-]+)\b", query, re.IGNORECASE)
        if between_match:
            x, y = between_match.group(1).lower(), between_match.group(2).lower()
            generic = {"the", "two", "them", "both", "one", "other", "these", "those", "pair", "options", "alternatives"}
            if x not in generic and y not in generic:
                return True

        return False

    @classmethod
    def _has_concrete_proper_nouns(cls, query: str) -> bool:
        """
        Detects whether a query contains concrete named entities, proper nouns,
        quoted terms, or technical terms rather than generic pronouns or vague attributes.
        """
        # Quoted strings are concrete entities
        if re.search(r"[\"'][^\"']+[\"']", query):
            return True

        # Check for capitalized words not at the beginning of the sentence
        words = query.strip().rstrip("?.!").split()
        if len(words) > 1:
            non_initial_words = words[1:]
            proper_nouns = [
                w for w in non_initial_words
                if (w[0].isupper() or any(c.isdigit() for c in w))
                and w.lower().strip("?,.!") not in {
                    "the", "a", "an", "and", "or", "vs", "versus",
                    "both", "two", "pair", "options", "alternatives", "them",
                    "better", "worse", "which", "what", "is", "are"
                }
            ]
            if len(proper_nouns) >= 1:
                return True

        # Known technical / domain entity anchors
        technical_terms = {
            "python", "java", "rust", "golang", "c++", "javascript", "typescript",
            "sql", "postgres", "mysql", "mongodb", "redis", "docker", "kubernetes",
            "pytorch", "tensorflow", "bert", "gpt", "llama", "gemini", "claude",
            "bitcoin", "ethereum", "solana", "apple", "google", "microsoft", "amazon",
            "linux", "windows", "macos", "android", "ios", "intel", "amd", "nvidia"
        }
        cleaned_lower_tokens = set(re.findall(r"\b[a-zA-Z0-9\+\#\.\-]+\b", query.lower()))
        if cleaned_lower_tokens.intersection(technical_terms):
            return True

        return False

    @classmethod
    def extract(cls, query: str) -> QuerySignals:
        """Analyze query, extract observable signals, and apply Antigravity Interceptor."""
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
        lower_q = cleaned.lower()
        
        # 1. Base Ambiguity Checks
        is_ambiguous = False
        ambiguity_reason = None
        
        for pat, reason in cls.COMPILED_AMBIGUOUS_PATTERNS:
            if pat.search(lower_q):
                is_ambiguous = True
                ambiguity_reason = reason
                break

        # Strict Regex Sanitized Conversational Greeting Check (prevents "history" colliding with "hi")
        is_greeting_or_conversational = any(
            pat.match(cleaned) for pat in cls.COMPILED_CONVERSATIONAL_PATTERNS
        )
        first_word = re.sub(r"[^a-z]", "", lower_q.split()[0]) if lower_q.split() else ""
        is_directive = first_word in cls.DIRECTIVE_VERBS
        has_quoted_entity = bool(re.search(r"[\"'].+[\"']", cleaned))

        # Check very short queries with missing context (e.g. 1-2 words that are not greetings, directives, or math)
        if not is_ambiguous and approx_tokens <= 2 and not any(op in cleaned for op in "+-*/="):
            if not is_greeting_or_conversational and not is_directive and not has_quoted_entity:
                is_ambiguous = True
                ambiguity_reason = "Query is too brief and lacks specification or actionable context"

        # --- ANTIGRAVITY FALLBACK INTERCEPTOR ---
        # 1. Detect Floating Dual Referents
        floating_referents_detected: List[str] = []
        for pat, label in cls.COMPILED_FLOATING_DUAL_PATTERNS:
            if pat.search(cleaned):
                floating_referents_detected.append(label)

        has_explicit_noun_antecedents = cls._has_explicit_antecedents(cleaned)
        floating_dual_active = bool(floating_referents_detected) and not has_explicit_noun_antecedents

        # 2. Detect Comparative Triggers
        comparative_triggers_detected: List[str] = []
        for pat, label in cls.COMPILED_COMPARATIVE_PATTERNS:
            if pat.search(cleaned):
                comparative_triggers_detected.append(label)
        has_comparative_triggers = bool(comparative_triggers_detected)

        # 3. Apply the Antigravity Lift
        has_concrete_proper_nouns = cls._has_concrete_proper_nouns(cleaned)
        is_short_unanchored_comparative = (
            has_comparative_triggers
            and approx_tokens < 7
            and not has_concrete_proper_nouns
        )

        antigravity_intercepted = False
        if floating_dual_active and has_comparative_triggers:
            antigravity_intercepted = True
            is_ambiguous = True
            ambiguity_reason = (
                f"Antigravity Fallback Interceptor: Floating dual referents ({', '.join(floating_referents_detected)}) "
                f"with comparative triggers ({', '.join(comparative_triggers_detected)}) lacking explicit noun antecedents"
            )
        elif is_short_unanchored_comparative:
            antigravity_intercepted = True
            is_ambiguous = True
            ambiguity_reason = (
                f"Antigravity Fallback Interceptor: Brief comparative query ({approx_tokens} tokens) "
                f"with triggers ({', '.join(comparative_triggers_detected)}) lacking concrete proper nouns"
            )

        # 2. Check Arithmetic
        has_arithmetic = False
        arithmetic_expr = None
        
        # Check explicit math patterns
        math_match = cls.ARITHMETIC_PREFIX_RE.match(cleaned)
        if math_match:
            candidate_expr = math_match.group(2).strip()
            if re.search(r"\d", candidate_expr) and cls.MATH_OPERATORS_RE.search(candidate_expr):
                has_arithmetic = True
                arithmetic_expr = candidate_expr
        
        # Fallback check for queries like "compute 45 * (12 + 3)" or "25 + 75"
        if not has_arithmetic:
            tokens_in_q = cleaned.replace("?", "").replace("=", "").split()
            math_chars_ratio = sum(1 for c in cleaned if c in "0123456789+-*/().^% ") / max(len(cleaned), 1)
            if math_chars_ratio > 0.65 and re.search(r"\d", cleaned) and cls.MATH_OPERATORS_RE.search(cleaned):
                has_arithmetic = True
                expr_cand = re.sub(r"^(what\s+is|calculate|compute|solve)\s*", "", cleaned, flags=re.IGNORECASE).rstrip("?=")
                arithmetic_expr = expr_cand.strip()

        # 3. Check Speculation / Unknowable
        is_unknowable = False
        speculation_reason = None
        for pat, reason in cls.COMPILED_SPECULATION_PATTERNS:
            if pat.search(cleaned):
                is_unknowable = True
                speculation_reason = reason
                break

        # 4. Strict Regex Sanitized Factual Knowledge Request Check
        # Uses compiled word boundaries to prevent partial collision bugs (e.g. 'temp' matching 'temperature of')
        is_factual = False
        detected_keywords: List[str] = []
        for kw, pat in cls.COMPILED_FACTUAL_PATTERNS:
            if pat.search(cleaned):
                is_factual = True
                detected_keywords.append(kw)
        
        # Check temporal freshness
        requires_freshness = False
        for kw, pat in cls.COMPILED_TEMPORAL_PATTERNS:
            if pat.search(cleaned):
                requires_freshness = True
                if not is_factual:
                    is_factual = True
                    detected_keywords.append(kw)
        
        # 5. Compute Heuristic Prior Uncertainty Score
        # 0.0 (high confidence/simple) to 1.0 (high uncertainty/risk)
        uncertainty = 0.35  # baseline
        if is_unknowable:
            uncertainty = 0.95
        elif antigravity_intercepted or is_ambiguous:
            # Force high uncertainty (0.85) when intercepted or ambiguous
            uncertainty = 0.85
        elif has_arithmetic:
            uncertainty = 0.10
        elif is_directive and first_word in {"spell", "reverse", "capitalize", "echo", "repeat", "say", "print"}:
            uncertainty = 0.15
        elif is_factual:
            uncertainty = 0.70 if requires_freshness else 0.60
        else:
            uncertainty = 0.30

        summary = {
            "has_arithmetic": has_arithmetic,
            "is_ambiguous": is_ambiguous,
            "is_factual": is_factual,
            "is_unknowable": is_unknowable,
            "requires_freshness": requires_freshness,
            "is_directive": is_directive,
            "antigravity_intercepted": antigravity_intercepted,
            "floating_referents_detected": floating_referents_detected,
            "comparative_triggers_detected": comparative_triggers_detected,
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
            antigravity_intercepted=antigravity_intercepted,
            floating_referents_detected=floating_referents_detected,
            comparative_triggers_detected=comparative_triggers_detected,
            heuristic_uncertainty_score=uncertainty,
            signal_summary=summary,
        )
