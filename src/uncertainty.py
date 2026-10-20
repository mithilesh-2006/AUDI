"""
Statistical Uncertainty Estimation Engine (Phase 4).

Provides:
1. Multi-sample semantic disagreement estimation across draft responses.
2. Jaccard and n-gram lexical divergence measures.
3. Corpus evidence availability assessment.
4. Composite uncertainty score combining disagreement, retrieval cues, and query ambiguity.

Safety & Transparency Notice:
All computed uncertainty scores represent statistical and heuristic approximations
derived from sample variance and retrieval features. They are explicitly NOT ground truth.
"""

from __future__ import annotations
import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from src.signals import QuerySignals
from src.retrieval import LocalCorpusRetriever, RetrievalResult


@dataclass
class DisagreementAnalysis:
    """Detailed breakdown of multi-sample semantic disagreement."""
    sample_texts: List[str]
    num_samples: int
    pairwise_jaccard_distances: List[float]
    mean_disagreement: float
    lexical_entropy: float
    disagreement_level: str  # "Low", "Moderate", "High"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "mean_disagreement": round(self.mean_disagreement, 4),
            "lexical_entropy": round(self.lexical_entropy, 4),
            "disagreement_level": self.disagreement_level,
            "sample_texts": self.sample_texts,
        }


@dataclass
class CompositeUncertaintyResult:
    """Aggregated uncertainty metrics for cost-aware decision routing."""
    composite_uncertainty: float  # Scale [0.0, 1.0]
    disagreement_score: float
    evidence_availability_score: float
    query_ambiguity_score: float
    heuristic_signal_score: float
    analysis: Optional[DisagreementAnalysis] = None
    is_statistically_grounded: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "composite_uncertainty": round(self.composite_uncertainty, 4),
            "disagreement_score": round(self.disagreement_score, 4),
            "evidence_availability_score": round(self.evidence_availability_score, 4),
            "query_ambiguity_score": round(self.query_ambiguity_score, 4),
            "heuristic_signal_score": round(self.heuristic_signal_score, 4),
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "is_statistically_grounded": self.is_statistically_grounded,
        }


class UncertaintyEstimator:
    """
    Estimates epistemic and semantic uncertainty using multi-sample agreement,
    retrieval confidence, and query characteristics.
    """

    def __init__(self, retriever: Optional[LocalCorpusRetriever] = None):
        self.retriever = retriever or LocalCorpusRetriever()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace and punctuation-delimited tokenizer."""
        return [w.lower() for w in re.findall(r"\b\w+\b", text) if len(w) > 1]

    @classmethod
    def compute_jaccard_distance(cls, text_a: str, text_b: str) -> float:
        """Computes Jaccard distance: 1 - (|A cap B| / |A cup B|)."""
        tokens_a = set(cls._tokenize(text_a))
        tokens_b = set(cls._tokenize(text_b))
        if not tokens_a and not tokens_b:
            return 0.0
        intersection = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))
        if union == 0:
            return 0.0
        return 1.0 - (intersection / union)

    @classmethod
    def compute_multi_sample_disagreement(cls, samples: List[str]) -> DisagreementAnalysis:
        """
        Calculates pairwise token disagreement and token entropy across multiple candidate outputs.
        """
        if not samples or len(samples) < 2:
            return DisagreementAnalysis(
                sample_texts=samples or [],
                num_samples=len(samples or []),
                pairwise_jaccard_distances=[0.0],
                mean_disagreement=0.0,
                lexical_entropy=0.0,
                disagreement_level="Low",
            )

        distances = []
        n = len(samples)
        for i in range(n):
            for j in range(i + 1, n):
                d = cls.compute_jaccard_distance(samples[i], samples[j])
                distances.append(d)

        mean_dist = sum(distances) / len(distances) if distances else 0.0

        # Lexical token entropy across all samples
        token_freq: Dict[str, int] = {}
        total_tokens = 0
        for s in samples:
            for tok in cls._tokenize(s):
                token_freq[tok] = token_freq.get(tok, 0) + 1
                total_tokens += 1

        entropy = 0.0
        if total_tokens > 0:
            for count in token_freq.values():
                p = count / total_tokens
                entropy -= p * math.log2(p)
            # Normalize entropy roughly by log2(vocab_size)
            vocab_size = max(len(token_freq), 2)
            norm_entropy = min(1.0, entropy / math.log2(vocab_size))
        else:
            norm_entropy = 0.0

        if mean_dist < 0.25:
            level = "Low"
        elif mean_dist < 0.60:
            level = "Moderate"
        else:
            level = "High"

        return DisagreementAnalysis(
            sample_texts=samples,
            num_samples=n,
            pairwise_jaccard_distances=distances,
            mean_disagreement=mean_dist,
            lexical_entropy=norm_entropy,
            disagreement_level=level,
        )

    def estimate_uncertainty(
        self,
        query: str,
        signals: QuerySignals,
        candidate_samples: Optional[List[str]] = None,
        retrieval_result: Optional[RetrievalResult] = None,
    ) -> CompositeUncertaintyResult:
        """
        Combines multi-sample disagreement, evidence availability, and observable signals.
        """
        # 1. Multi-sample disagreement
        analysis = None
        if candidate_samples and len(candidate_samples) >= 2:
            analysis = self.compute_multi_sample_disagreement(candidate_samples)
            disagreement_score = analysis.mean_disagreement
        else:
            disagreement_score = signals.heuristic_uncertainty_score

        # 2. Evidence availability score
        if retrieval_result is None and signals.is_factual_knowledge_request:
            retrieval_result = self.retriever.retrieve(query, top_k=2)

        if retrieval_result and retrieval_result.passages:
            max_score = max(p.relevance_score for p in retrieval_result.passages)
            evidence_availability = min(1.0, max_score / 2.0)
        else:
            evidence_availability = 0.0

        # 3. Query ambiguity score
        query_ambiguity = 0.9 if signals.is_ambiguous else (1.0 if signals.is_empty else 0.1)

        # 4. Speculation / unknowable penalty
        speculation_factor = 0.95 if signals.is_unknowable_or_speculative else 0.1

        # Composite formulation:
        # High disagreement + low evidence availability + ambiguity -> high uncertainty
        # High evidence availability -> lower uncertainty for factual queries
        if signals.has_arithmetic_expr:
            # Deterministic math has low intrinsic uncertainty if tool is available
            composite = 0.05
        elif signals.is_unknowable_or_speculative:
            composite = 0.95
        elif signals.is_ambiguous or signals.is_empty:
            composite = 0.85
        elif signals.is_factual_knowledge_request:
            composite = max(0.05, 0.5 * disagreement_score + 0.5 * (1.0 - evidence_availability))
        else:
            composite = signals.heuristic_uncertainty_score

        return CompositeUncertaintyResult(
            composite_uncertainty=round(min(1.0, max(0.0, composite)), 4),
            disagreement_score=round(disagreement_score, 4),
            evidence_availability_score=round(evidence_availability, 4),
            query_ambiguity_score=round(query_ambiguity, 4),
            heuristic_signal_score=round(signals.heuristic_uncertainty_score, 4),
            analysis=analysis,
            is_statistically_grounded=True,
        )
