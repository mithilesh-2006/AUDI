"""
Local Evidence Retrieval Engine.

Implements lexical TF-IDF / BM25 style passage retrieval over a controlled
local corpus with source attribution and threshold-based evidence gating.
Does NOT fabricate evidence or claim that missing evidence proves a statement false.
"""

from __future__ import annotations
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class CorpusDocument:
    """A document in the local evidence corpus."""
    doc_id: str
    title: str
    source: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedPassage:
    """An individual passage retrieved as candidate evidence."""
    passage_id: str
    doc_id: str
    doc_title: str
    source_url_or_ref: str
    text: str
    relevance_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passage_id": self.passage_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "source_url_or_ref": self.source_url_or_ref,
            "text": self.text,
            "relevance_score": round(self.relevance_score, 4),
        }


@dataclass
class RetrievalResult:
    """Structured response from the retrieval engine."""
    query: str
    passages: List[RetrievedPassage]
    has_sufficient_evidence: bool
    evidence_threshold: float
    total_corpus_docs: int
    latency_ms: float
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "passages": [p.to_dict() for p in self.passages],
            "has_sufficient_evidence": self.has_sufficient_evidence,
            "evidence_threshold": self.evidence_threshold,
            "total_corpus_docs": self.total_corpus_docs,
            "latency_ms": round(self.latency_ms, 3),
            "message": self.message,
        }


class LocalCorpusRetriever:
    """Passage retrieval engine using BM25-style lexical scoring."""

    def __init__(self, corpus_path: Optional[str] = None, evidence_threshold: float = 0.15):
        self.evidence_threshold = evidence_threshold
        self.documents: Dict[str, CorpusDocument] = {}
        self.passages: List[RetrievedPassage] = []
        self.corpus_path = corpus_path or os.path.join("data", "corpus", "knowledge_base.json")
        
        # Load or initialize corpus
        self.load_corpus()

    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "in", "on", "at", "to", "for", "of", "with", "by", "from", "up",
        "about", "into", "over", "after", "what", "which", "who", "whom",
        "this", "that", "these", "those", "am", "it", "its", "and", "or", "as", "did"
    }

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text and filter out generic grammatical stopwords."""
        tokens = re.findall(r"\b\w{2,}\b", text.lower())
        return [t for t in tokens if t not in self.STOPWORDS]

    def add_document(self, doc_id: str, title: str, source: str, text: str, metadata: Optional[Dict[str, Any]] = None, save: bool = True) -> None:
        """Add a document to the in-memory corpus and rebuild passages."""
        doc = CorpusDocument(
            doc_id=doc_id,
            title=title,
            source=source,
            text=text,
            metadata=metadata or {}
        )
        self.documents[doc_id] = doc
        self._index_passages()
        if save:
            self.save_corpus()

    def _index_passages(self) -> None:
        """Segment documents into passages (e.g., paragraphs or sentences) and index them."""
        self.passages = []
        for doc in self.documents.values():
            # Split into paragraphs or chunks
            raw_paragraphs = [p.strip() for p in doc.text.split("\n\n") if p.strip()]
            if not raw_paragraphs:
                raw_paragraphs = [doc.text.strip()]
            
            for idx, para in enumerate(raw_paragraphs):
                # If paragraph is very long, optionally chunk further, but for our local corpus paragraphs are ideal
                p_id = f"{doc.doc_id}_p{idx+1}"
                self.passages.append(RetrievedPassage(
                    passage_id=p_id,
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    source_url_or_ref=doc.source,
                    text=para,
                    relevance_score=0.0
                ))

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        """Retrieve top passages relevant to the query."""
        start_time = time.perf_counter()
        query_tokens = self._tokenize(query)
        
        if not query_tokens or not self.passages:
            latency = (time.perf_counter() - start_time) * 1000
            return RetrievalResult(
                query=query,
                passages=[],
                has_sufficient_evidence=False,
                evidence_threshold=self.evidence_threshold,
                total_corpus_docs=len(self.documents),
                latency_ms=latency,
                message="No query terms or empty corpus."
            )

        # Compute term frequencies across passages
        n_passages = len(self.passages)
        doc_freqs: Dict[str, int] = {}
        passage_token_counts: List[Dict[str, int]] = []
        passage_lengths: List[int] = []

        for p in self.passages:
            tokens = self._tokenize(p.text) + self._tokenize(p.doc_title) * 2  # Boost title
            lengths = len(tokens)
            passage_lengths.append(lengths)
            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            passage_token_counts.append(counts)
            for t in set(tokens):
                doc_freqs[t] = doc_freqs.get(t, 0) + 1

        avg_dl = sum(passage_lengths) / max(len(passage_lengths), 1)
        k1 = 1.5
        b = 0.75

        # Score passages using standard BM25 formula
        scored_passages: List[RetrievedPassage] = []
        for i, p in enumerate(self.passages):
            score = 0.0
            p_counts = passage_token_counts[i]
            dl = passage_lengths[i]
            
            for q_term in set(query_tokens):
                if q_term in p_counts:
                    df = doc_freqs.get(q_term, 0)
                    idf = math.log((n_passages - df + 0.5) / (df + 0.5) + 1.0)
                    tf = p_counts[q_term]
                    denom = tf + k1 * (1.0 - b + b * (dl / avg_dl))
                    term_score = idf * (tf * (k1 + 1.0)) / max(denom, 1e-6)
                    score += term_score

            if score > 0:
                # Normalize score somewhat into [0, 1] relative scale
                norm_score = min(score / 15.0, 1.0)
                scored_passages.append(RetrievedPassage(
                    passage_id=p.passage_id,
                    doc_id=p.doc_id,
                    doc_title=p.doc_title,
                    source_url_or_ref=p.source_url_or_ref,
                    text=p.text,
                    relevance_score=norm_score
                ))

        # Sort descending by score
        scored_passages.sort(key=lambda x: x.relevance_score, reverse=True)
        top_passages = scored_passages[:top_k]
        
        has_evidence = False
        if top_passages and top_passages[0].relevance_score >= self.evidence_threshold:
            has_evidence = True

        latency = (time.perf_counter() - start_time) * 1000
        
        if has_evidence:
            msg = f"Retrieved {len(top_passages)} relevant passage(s) from local corpus."
        elif top_passages:
            msg = f"Candidate passages found, but highest relevance ({top_passages[0].relevance_score:.3f}) is below threshold ({self.evidence_threshold})."
        else:
            msg = "No matching evidence found in local corpus. (Note: missing local evidence does not imply the claim is universally false)."

        return RetrievalResult(
            query=query,
            passages=top_passages,
            has_sufficient_evidence=has_evidence,
            evidence_threshold=self.evidence_threshold,
            total_corpus_docs=len(self.documents),
            latency_ms=latency,
            message=msg
        )

    def load_corpus(self) -> None:
        """Load corpus from JSON file or populate default baseline facts."""
        if os.path.exists(self.corpus_path):
            try:
                with open(self.corpus_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        doc = CorpusDocument(
                            doc_id=item["doc_id"],
                            title=item["title"],
                            source=item["source"],
                            text=item["text"],
                            metadata=item.get("metadata", {})
                        )
                        self.documents[doc.doc_id] = doc
                self._index_passages()
                return
            except Exception as e:
                print(f"Warning: Failed to load corpus from {self.corpus_path}: {e}")

        # Populate initial default sample corpus
        self._populate_default_corpus()
        self.save_corpus()

    def _populate_default_corpus(self) -> None:
        """Initialize verified factual knowledge corpus across multiple domains."""
        defaults = [
            CorpusDocument(
                doc_id="doc_geo_everest",
                title="Mount Everest Elevation and Geography",
                source="National Geographic Society / Survey of Nepal 2020",
                text="Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. The official elevation of Mount Everest was jointly revised and announced by Nepal and China in December 2020 as 8,848.86 meters (29,031.7 feet)."
            ),
            CorpusDocument(
                doc_id="doc_sci_light",
                title="Speed of Light in Vacuum",
                source="CODATA Recommended Values of the Fundamental Physical Constants (2018)",
                text="The speed of light in vacuum, commonly denoted c, is a universal physical constant. Its exact value is defined as 299,792,458 meters per second (approximately 300,000 km/s or 186,282 miles per second)."
            ),
            CorpusDocument(
                doc_id="doc_hist_apollo11",
                title="Apollo 11 Moon Landing Mission",
                source="NASA Mission Archives - Apollo 11 (1969)",
                text="Apollo 11 was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969, at 20:17 UTC. Neil Armstrong became the first person to step onto the lunar surface six hours and 39 minutes later on July 21 at 02:56 UTC."
            ),
            CorpusDocument(
                doc_id="doc_tech_turing",
                title="Alan Turing and the Turing Machine",
                source="Stanford Encyclopedia of Philosophy - Alan Turing",
                text="Alan Mathison Turing (1912–1954) was an English mathematician, computer scientist, logician, cryptanalyst, and theoretical biologist. In 1936, Turing introduced the concept of the universal machine, now known as the Turing Machine, which formed the theoretical foundation of modern computer science."
            ),
            CorpusDocument(
                doc_id="doc_ai_rag",
                title="Retrieval-Augmented Generation (RAG) Architecture",
                source="Lewis et al., NeurIPS 2020: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
                text="Retrieval-Augmented Generation (RAG) combines pre-trained parametric language models with non-parametric retrieval memory. RAG fetches relevant document passages from an external corpus conditioned on the input query before generating the target response, reducing hallucinations in knowledge-intensive tasks."
            ),
            CorpusDocument(
                doc_id="doc_nobel_einstein",
                title="Albert Einstein Nobel Prize Award",
                source="The Nobel Prize in Physics 1921 Official Citation",
                text="The Nobel Prize in Physics 1921 was awarded to Albert Einstein for his services to Theoretical Physics, and especially for his discovery of the law of the photoelectric effect, rather than for his theory of relativity."
            )
        ]
        for d in defaults:
            self.documents[d.doc_id] = d
        self._index_passages()

    def save_corpus(self) -> None:
        """Persist corpus to disk."""
        os.makedirs(os.path.dirname(self.corpus_path), exist_ok=True)
        try:
            data = [
                {
                    "doc_id": d.doc_id,
                    "title": d.title,
                    "source": d.source,
                    "text": d.text,
                    "metadata": d.metadata,
                }
                for d in self.documents.values()
            ]
            with open(self.corpus_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save corpus to {self.corpus_path}: {e}")
