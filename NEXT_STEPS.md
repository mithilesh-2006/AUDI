# Next Steps: Prioritized Research & Development Roadmap

> **System:** Adaptive, Cost-Aware Hallucination Mitigation for Large Language Models  
> **Horizon:** Phases 2 through 5  

---

## 1. Prioritized Roadmap Overview

```text
Phase 1 (Completed)         Phase 2                     Phase 3                     Phase 4 / 5
[Rule Baseline & CLI/UI] ─► [Learned Action Router]  ─► [Dense Retrieval & NLI] ─► [Large-Scale Evaluation]
- AST Calculator            - Feature dataset           - Sentence-Transformers     - GSM8K / TriviaQA / PopQA
- BM25 Passage Search       - Contextual Bandit / LR    - DeBERTa NLI Verifier      - Pareto Frontier Plots
- Multi-Policy Suite        - Calibrated Utility        - Multi-turn Clarification  - Statistical Significance
```

---

## 2. Priority 1: Calibrated Learned Policy (Phase 4 Milestone)

- **Objective:** Replace hand-crafted heuristic rules with a trained, calibrated router that optimizes the cost-quality trade-off:
  $$a^* = \arg\max_{a \in A} \left[ \widehat{\Delta Q}(a \mid x) - \lambda \cdot \widehat{C}(a \mid x) \right]$$
- **Implementation Tasks:**
  1. Record feature logs ($x$) from diverse development queries into `results/logs/runs.jsonl`.
  2. Implement an offline training pipeline (`train_router.py`) fitting a multinomial logistic regression or gradient-boosted decision tree.
  3. Validate probability calibration using Expected Calibration Error (ECE) and Brier scores.
  4. Ensure zero test-set leakage: held-out test splits must not influence router parameters.

---

## 3. Priority 2: Dense Semantic Retrieval & Small Local NLI Verifier

- **Objective:** Upgrade lexical BM25 matching and token alignment to semantic representations without incurring external API costs.
- **Implementation Tasks:**
  1. Add optional support for `sentence-transformers` (e.g., `all-MiniLM-L6-v2`) for local bi-encoder dense passage retrieval.
  2. Implement a local NLI verifier using a compact DeBERTa model (`deberta-v3-small`) to classify premise-hypothesis pairs as `Entailment`, `Contradiction`, or `Neutral`.
  3. Compare BM25 vs Dense retrieval on retrieval latency and passage precision.

---

## 4. Priority 3: Large-Scale Benchmark Suite & Pareto Analysis (Phase 5)

- **Objective:** Run comprehensive evaluations on standard public benchmarks to rigorously test the central research hypothesis.
- **Implementation Tasks:**
  1. Download and format standardized evaluation splits:
     - **Factual QA:** PopQA / TriviaQA
     - **Arithmetic & Reasoning:** GSM8K
     - **Ambiguity:** AmbigQA
     - **Unanswerable:** Selected unanswerable questions (e.g. SQuAD 2.0 unanswerable split)
  2. Implement automated batch evaluation runner (`experiments/compare_baselines.py`) generating:
     - Accuracy vs Cost Pareto frontier curves
     - Paired bootstrap confidence intervals (95% CI)
     - Action-selection confusion matrices
     - Failure mode ablation tables

---

## 5. Priority 4: Multi-Turn Clarification State Machine

- **Objective:** Enable true multi-turn clarification dialogs.
- **Implementation Tasks:**
  1. Implement a session context store tracking original query + clarifying question.
  2. Ingest the user's clarifying response, merge it into an augmented query context, and re-invoke the controller to produce a grounded final answer.
  3. Measure the net utility and user interaction cost of clarification turns.

---

## 6. Priority 5: Local Quantized Neural Model Support (Ollama / Llama.cpp)

- **Objective:** Enable fully offline, GPU-accelerated local neural generation.
- **Implementation Tasks:**
  1. Create an `OllamaProvider` connecting to a local Ollama server (`http://localhost:11434/api/generate`).
  2. Support lightweight open-weights models (e.g., `llama3.2:3b`, `qwen2.5:3b`, `phi-3-mini`).
  3. Extract token-level log-probabilities for true semantic entropy and uncertainty estimation.
