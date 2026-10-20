# LIMITATIONS: Known Technical, Scientific, and Empirical Constraints

> **System:** AUDI (*Adaptive Uncertainty-Driven Inference*)  
> **Document Type:** Scientific Limitations & Integrity Disclosure  
> **Status:** Active Reference Document  

---

## 1. Routing Limitations & Signal Approximations

1. **Heuristic & Lexical Sensitivity:**
   - The rule-based controller relies on surface linguistic cues, AST parseability, and keyword markers. Complex, metaphoric, or adversarial queries may cause misrouting (e.g., phrasing an arithmetic question in elaborate prose that bypasses regex extraction).
2. **Context-Free Parsing:**
   - The signal extractor evaluates individual queries in isolation. Complex conversational references spanning long prior dialogues may not be fully resolved by isolated turn extractors.
3. **Discrete Action Gating:**
   - The current action space $\mathcal{A}$ selects a primary mitigation action rather than learning a dynamic sequence of interleaved actions for multi-step reasoning.

---

## 2. Uncertainty Estimation & Calibration Constraints

1. **Uncalibrated Model Self-Confidence:**
   - When using live LLM providers, self-reported confidence tags (e.g., `[CONFIDENCE: 0.85]`) are notoriously prone to overconfidence and sycophancy. They must be treated strictly as uncalibrated features rather than actual error probabilities.
2. **Failure Modes of Disagreement:**
   - Multi-sample semantic consistency and lexical disagreement assume that model errors exhibit high variance across temperature samples. For deeply ingrained training biases or common misconceptions (e.g., claiming Einstein won the Nobel Prize for Relativity), multiple model generations can unanimously agree on a falsehood (zero disagreement, yet 100% hallucinated).
3. **Platt / Temperature Calibration Sample Size:**
   - Fitting temperature parameter $T$ requires a statistically representative development set ($N \ge 100$). Fitting on small sample sets ($N < 50$) risks over-fitting the temperature parameter.

---

## 3. Evidence Retrieval & Local Corpus Boundaries

1. **Corpus Completeness:**
   - The prototype evaluates against a curated local document corpus (`data/corpus/knowledge_base.json`). It cannot answer factual inquiries outside this collection.
2. **Epistemic Scope:**
   - An absence of local evidence is classified as `INSUFFICIENT_EVIDENCE`. Researchers must not infer that a claim is universally false merely because it does not appear in the local store.
3. **Lexical BM25 Limitations:**
   - BM25 relies on exact term overlap and inverse document frequency. It lacks dense semantic embeddings and cannot resolve complex synonymy or cross-lingual queries without dense retrieval extensions.

---

## 4. Claim Decomposition & Verification Limitations

1. **Syntactic Claim Decomposition:**
   - Sentence splitting regexes approximate propositional assertions. Compound sentences with multiple dependent clauses may occasionally be checked as single composite units.
2. **Lexical Alignment vs. Deep Natural Language Inference (NLI):**
   - The verifier employs token overlap, entity extraction, and numerical contradiction checks. While effective at catching incorrect dates, elevations, and measurements, it does not perform deep deductive reasoning or multi-hop logical inference.
3. **Attribution Hallucination Risk:**
   - When verbatim quotes are extracted, the system relies on exact substring matching. Verifiers that summarize evidence rather than quoting verbatim risk compounding hallucination.

---

## 5. Experimental Scale & Evaluation Constraints

1. **Preliminary Benchmark Scale:**
   - The current curated evaluation benchmark contains 24 versioned items across 8 archetypes. While sufficient for unit testing, reproducible development, and methodology demonstration, broader claims of generalization require evaluation across large public benchmarks (e.g., TruthfulQA, HaluEval, FEVER, GSM8K).
2. **Offline Mock Adapter Disclaimers:**
   - The offline local adapter (`src/model_adapter.py`) uses deterministic regex templates. It is provided strictly as a zero-cost software-testing tool. Mock performance must **never** be cited as empirical evidence of real LLM behavior.
3. **Latency and Cost Variance in Production:**
   - Latency figures recorded in local offline runs reflect Python execution speed. When connecting to paid remote APIs (OpenAI, Gemini), network latency, server queuing, and rate limits will introduce significant variance into measured wall-clock times.
