# Implementation Plan: Adaptive, Cost-Aware Hallucination Mitigation Prototype (AUDI)

> **Project:** AUDI — Adaptive Uncertainty-Driven Inference  
> **Status:** All Phases (1 through 5) Implemented, Audited & Scientifically Validated  
> **Test Status:** 70/70 Unit Tests Passing  
> **Primary Specification:** `PROJECT_DESCRIPTION.md`  

---

## 1. Executive Summary & Mapping to Requirements

The goal is to build an inspectable, research-grade working prototype that demonstrates **adaptive selection of hallucination-mitigation actions** rather than applying uniform, high-cost pipelines to every input query.

The candidate action set:
1. **Direct Answer (`direct`)**: Directly generate an answer when the query is standard factual/definitional and low-risk, or when external retrieval is not indicated.
2. **Deterministic Calculation (`calculate`)**: Route pure or structured arithmetic/mathematical queries to a sandboxed, AST-based deterministic calculator.
3. **Evidence Retrieval (`retrieve`)**: Fetch authoritative passages from a local corpus for factual, empirical, or time-sensitive queries.
4. **Clarification Request (`clarify`)**: Detect ambiguous, underspecified, or comparison-deficient queries and ask structured clarifying questions.
5. **Evidence-based Verification (`verify`)**: Split candidate claims and verify them against local retrieved evidence passages, computing authentic operational verification labels (`SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`, `NOT_CHECKABLE`).
6. **Revision (`revise`)**: Update, excise, or hedge ungrounded claims based on verification feedback and trigger re-verification.
7. **Abstention (`abstain`)**: Refuse to guess on unknowable future events, unanswerable queries, or when evidence is definitively insufficient.

All actions, decision rationales, uncertainty signals, latency, token usage, and costs (both estimated proxy costs and measured actual costs) are tracked, logged, and inspectable via both a CLI and a Streamlit UI dashboard.

---

## 2. Phased Implementation Roadmap & Completion Status

| Phase | Core Objective | Key Deliverables | Completion Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Baseline Controller & Core Router | - Signal Extractor (`src/signals.py`)<br>- AST Safe Calculator (`src/calculator.py`)<br>- Policy Base & Baseline Rules (`src/policy.py`)<br>- SQLite & JSONL Logging (`src/logging_service.py`)<br>- Cost Tracker (`src/cost_tracker.py`) | **Complete & Verified** |
| **Phase 2** | Multi-Provider LLM Integration | - Universal Adapter (`src/model_adapter.py`)<br>- Offline Local, OpenAI, Gemini, Ollama providers<br>- Prompt Template Manager (`src/prompt_manager.py`)<br>- Confidence Score Extraction | **Complete & Verified** |
| **Phase 3** | Mitigation Tool Suite | - BM25 Corpus Retriever (`src/retrieval.py`)<br>- Claim Verifier (`src/verification.py`) with `SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`, `NOT_CHECKABLE`<br>- Revision Engine (`src/revision.py`)<br>- Interactive Clarification (`src/clarification.py`)<br>- Epistemic Abstention | **Complete & Verified** |
| **Phase 4** | Statistical Uncertainty & Calibration | - Multi-sample Disagreement (`src/uncertainty.py`)<br>- Temperature Scaling & Platt Calibration (`src/calibration.py`)<br>- ECE and Brier Score computation<br>- Learned Cost-Aware Policy (`src/policy.py`) | **Complete & Verified** |
| **Phase 5** | Scientific Evaluation & Benchmarking | - Standardized 5-Policy Benchmark Runner (`src/experiment_runner.py`)<br>- 24-item versioned evaluation set across 8 archetypes<br>- Action Confusion Matrices & Failure Categorization<br>- Component Ablation Runner | **Complete & Verified** |
| **Optimization & Profiling** | Execution Time Benchmarking & UI | - High-precision timing ($T_{\text{direct}}$ vs $T_{\text{AUDI}}$, decision overhead, speedup factor)<br>- White theme minimalist UI in `app.py` with zero emojis<br>- CLI timing outputs in `controller.py` | **Complete & Verified** |

---

## 3. Verified Architectural Specifications

```text
                                  User Query
                                       │
                                       ▼
                       Feature & Signal Extractor (src/signals.py)
                   [Intent, Ambiguity, Arithmetic, Knowledge Need]
                                       │
                                       ▼
                       Cost-Aware Router (src/controller.py)
                  Objective: a* = argmin [Loss(a|x) + λ * Cost(a)]
                                       │
         ┌───────────────┬─────────────┼──────────────┬──────────────┐
         ▼               ▼             ▼              ▼              ▼
     DIRECT          CALCULATE     RETRIEVE        CLARIFY        ABSTAIN
  (model_adapter)  (calculator)  (retrieval)   (clarification) (Safe Refusal)
         │               │             │              │              │
         └───────────────┼─────────────┼──────────────┴──────────────┘
                         │             ▼
                         │    Verification & Revision
                         │  (src/verification.py, src/revision.py)
                         │    - Propositional Decomposition
                         │    - Operational Labels: SUPPORTED, CONTRADICTED,
                         │      INSUFFICIENT_EVIDENCE, NOT_CHECKABLE
                         │    - Numerical & Entity Contradiction Checking
                         │    - Post-Revision Re-Verification
                         │             │
                         └─────────────┼─────────────────────────────┐
                                       ▼                             ▼
                          Cost Tracker & Metric Log          Final Response
                           (src/cost_tracker.py,             (Streamlit App
                            src/logging_service.py)           & CLI Output)
                                       │
                                       ▼
                          Experiment Comparison Suite
                            (src/experiment_runner.py)
                     [Direct-only, Always-retrieve, Fixed-verify,
                      Adaptive Policy, Learned Cost-Aware Policy]
```

---

## 4. Verification and Acceptance Criteria

1. **Safety:** Deterministic AST arithmetic rejects all unsafe syntax, attribute lookups, arbitrary calls, and exponents $>1000$.
2. **Operational Verification:** Claims are classified into `SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`, and `NOT_CHECKABLE`. Absence of evidence is never labeled false.
3. **No Leakage:** Fixed dev/test splits ensure calibration parameters are fitted strictly on development data.
4. **Fair Baseline Comparison:** All 5 policies evaluated on the exact same benchmark queries.
5. **Timing Rigor:** Measured via `time.perf_counter()` with sub-millisecond precision.
6. **Test Coverage:** All 70 unit tests pass in under 2.5 seconds.
