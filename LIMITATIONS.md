# Known Limitations and Research Boundaries

> **Document:** Known Limitations of the Baseline Research Prototype  
> **System:** Adaptive, Cost-Aware Hallucination Mitigation for Large Language Models  
> **Phase:** 1+ Runnable Baseline Prototype  

---

## 1. Action Routing & Uncertainty Signal Limitations

### 1.1 Heuristic Cue Overlap
- The controller uses observable syntactic, lexical, and pattern-matching cues to detect ambiguity, arithmetic, factual inquiries, and speculation.
- **Limitation:** Simple heuristics cannot reliably detect subtle semantic ambiguity, multi-step logical reasoning flaws, or disguised hallucinations. False positives and false negatives will occur on complex phrasing.
- **Boundary:** Hand-written rules do not constitute a general-purpose or scientifically validated hallucination detector.

### 1.2 Model Confidence and Uncertainty Proxies
- Model-reported self-confidence or heuristic priors are imperfect proxies for factual correctness.
- Language models can be consistently wrong with high confidence ("fluent confabulation").
- Token log-probabilities (when available from remote APIs) measure continuation likelihood, not real-world truth.

---

## 2. Evidence Retrieval & Verification Limitations

### 2.1 Lexical BM25 Retrieval
- The current local retrieval engine employs lexical BM25 passage matching over the controlled corpus.
- **Limitation:** It lacks deep semantic embedding similarity (e.g. dense bi-encoders) and may fail when queries use synonyms or paraphrased terms not present in the indexed text.

### 2.2 Corpus Coverage & Negative Evidence
- The local corpus is intentionally controlled and compact (containing sample verified documents on geography, physics, history, and computer science).
- **Critical Principle:** A missing passage in the local corpus does **NOT** prove that a user statement is universally false or unknowable. It only indicates that the system lacks authoritative local grounding.

### 2.3 Verification & Natural Language Inference (NLI)
- Claim verification uses token alignment and numerical entity ground checking against retrieved evidence.
- While effective at catching unsupported years, dates, quantities, and missing entities, it is not a full formal logic solver or fine-tuned NLI model. Subtle contradictions or entailments may be missed.

---

## 3. Answer Generation & Model Adapters

### 3.1 Local Demonstration Mode vs Live Remote API
- By default, the prototype runs 100% locally in **Offline Demonstration Mode** without requiring external credentials or paid accounts.
- The local provider returns grounded answers using extracted evidence and rule-based generation templates.
- Full generative fluency on arbitrary topics requires setting `OPENAI_API_KEY` or `GEMINI_API_KEY` in `.env`.

### 3.2 Single-Turn Clarification
- The clarification action formulates structured follow-up questions to resolve missing context.
- Full multi-turn conversational dialogue state tracking and user response re-injection are planned for Phase 3+.

---

## 4. Cost Optimization & Utility Formulation

### 4.1 Cost Units vs Currency
- The routing objective optimizes:
  $$a^* = \arg\max_a [\text{Benefit}(a \mid x) - \lambda \cdot \text{Cost}(a)]$$
- The prior action costs are dimensionless illustrative weights (e.g. calculate=0.2, direct=1.0, retrieve=3.5, verify=5.0).
- Actual monetary costs are computed strictly based on provider token rates ($0.0 for local mode).
- Benefit estimation is currently heuristic; it has not yet been fitted on a large offline dataset via regression or bandit learning.

---

## 5. Benchmark Scope

- The built-in benchmark dataset contains 12 curated examples across Arithmetic, Factual QA, Ambiguity, and Speculation.
- While ideal for immediate test execution and sanity checks, claims of statistical superiority or generalizability must await large-scale evaluations on public benchmarks (TriviaQA, PopQA, GSM8K, AmbigQA).
