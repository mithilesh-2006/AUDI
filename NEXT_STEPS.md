# NEXT STEPS: Future Research Roadmap & Engineering Enhancements

> **System:** AUDI (*Adaptive Uncertainty-Driven Inference*)  
> **Document Type:** Research & Engineering Roadmap  
> **Status:** Active Roadmap  

---

## 1. Benchmark Scaling & Standard Dataset Integration

1. **Large-Scale Evaluation Suites:**
   - Integrate loaders for standard NLP hallucination and factual QA benchmarks:
     - **TruthfulQA:** Evaluating resistance to common human misconceptions.
     - **HaluEval:** Measuring claim-level verification on knowledge-grounded dialogues.
     - **FEVER (Fact Extraction and VERification):** Benchmarking claim-level attribution.
     - **GSM8K / SVAMP:** Benchmarking deterministic AST math routing versus chain-of-thought generation.
2. **Automated Error Rubrics:**
   - Transition from keyword-matching reference checks to standardized LLM-as-a-judge or exact-match parsing with rigorous cross-annotator agreement (Cohen's $\kappa$).

---

## 2. Advanced Retrieval: Hybrid Dense-Lexical Indexing

1. **Vector Semantic Search:**
   - Supplement in-memory BM25 with dense embedding retrievers (e.g., using `sentence-transformers`, `all-MiniLM-L6-v2`, or local FAISS index).
   - Implement Reciprocal Rank Fusion (RRF) combining BM25 lexical precision with dense semantic recall.
2. **Hierarchical Document Chunking:**
   - Replace paragraph chunking with sliding-window sentence tokenization to improve passage relevance scoring.

---

## 3. Deep Natural Language Inference (NLI) Verifier

1. **Local NLI Cross-Encoder:**
   - Replace lexical overlap verification with a compact local NLI model (e.g., `deberta-v3-small-tasksource-nli`).
   - Compute authentic entailment, contradiction, and neutral probabilities:
     - $\hat{P}(\text{Entailment}) \ge 0.70 \implies \text{SUPPORTED}$
     - $\hat{P}(\text{Contradiction}) \ge 0.50 \implies \text{CONTRADICTED}$
     - Otherwise $\implies \text{INSUFFICIENT\_EVIDENCE}$

---

## 4. Learned Cost-Aware Contextual Router

1. **Contextual Bandit / Gradient Boosted Policy:**
   - Train a lightweight classifier (e.g., LightGBM or logistic regression) directly on stored experiment run logs in `results/experiments.db`.
   - Feature space: Query length, lexical entropy, arithmetic parse confidence, BM25 top-1 score, model self-confidence.
   - Reward formulation: $\mathcal{R}(a) = -L_{\text{quality}}(a) - \lambda \cdot C(a)$.
2. **Pareto Frontier Optimization:**
   - Automatically tune $\lambda$ across a continuous grid $[0.0, 2.0]$ to identify optimal operating points for low-resource vs. high-assurance deployments.

---

## 5. Token-Level Uncertainty with Open-Weight Models

1. **Direct Logit Access via Ollama / vLLM:**
   - When running local open-weight models (e.g. Llama-3-8B, Mistral-7B), extract token log-probabilities to compute authentic token entropy and semantic entropy:
     $$H_{\text{token}} = -\sum_{v \in \mathcal{V}} P(v) \log P(v)$$
   - Utilize genuine token uncertainty rather than relying solely on generated text or self-reported confidence tags.
