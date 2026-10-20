# Implementation Plan: Adaptive, Cost-Aware Hallucination Mitigation Prototype

> **Project:** Adaptive, Cost-Aware Hallucination Mitigation for Large Language Models  
> **Status:** Stage 1 Implementation Plan  
> **Primary Source:** `PROJECT_DESCRIPTION.md`  

---

## 1. Executive Summary & Mapping to Requirements

The goal is to build an inspectable, research-grade working prototype that demonstrates **adaptive selection of hallucination-mitigation actions** rather than applying uniform, high-cost pipelines to every input query.

The candidate action set:
1. **Direct Answer (`direct`)**: Directly generate an answer when the query is standard factual/definitional and low-risk, or when external retrieval is not indicated.
2. **Deterministic Calculation (`calculate`)**: Route pure or structured arithmetic/mathematical queries to a sandboxed, AST-based deterministic calculator.
3. **Evidence Retrieval (`retrieve`)**: Fetch authoritative passages from a local corpus for factual, empirical, or time-sensitive queries.
4. **Clarification Request (`clarify`)**: Detect ambiguous, underspecified, or comparison-deficient queries and ask structured clarifying questions.
5. **Evidence-based Verification (`verify`)**: Split candidate claims and verify them against local retrieved evidence passages, computing authentic verification labels (`Verified`, `Partially Supported`, `Unsupported`, `Not Checked`).
6. **Revision (`revise`)**: Update or excise unsupported claims based on verification feedback.
7. **Abstention (`abstain`)**: Refuse to guess on unknowable future events, unanswerable queries, or when evidence is definitively insufficient.

All actions, decision rationales, uncertainty signals, latency, token usage, and costs (both estimated baseline costs and measured actual costs) are tracked, logged, and inspectable via both a CLI and a Streamlit UI dashboard.

---

## 2. Architectural Blueprint

```text
                                  User Query
                                       │
                                       ▼
                       Feature & Signal Extractor (signals.py)
                   [Intent, Ambiguity, Arithmetic, Knowledge Need]
                                       │
                                       ▼
                       Cost-Aware Router (controller.py)
                  Objective: a* = argmin [Loss(a|x) + λ * Cost(a)]
                                       │
         ┌───────────────┬─────────────┼──────────────┬──────────────┐
         ▼               ▼             ▼              ▼              ▼
     Direct          Calculate     Retrieve        Clarify        Abstain
  (llm_provider)   (calculator)  (retrieval)    (Clarify Prompt) (Safe Refusal)
         │               │             │              │              │
         └───────────────┼─────────────┼──────────────┴──────────────┘
                         │             ▼
                         │    Verification & Revision
                         │    (verification.py, revision.py)
                         │             │
                         └─────────────┼─────────────────────────────┐
                                       ▼                             ▼
                          Cost Tracker & Metric Log          Final Response
                           (cost_tracker.py,                 (Streamlit App
                            logging_service.py)               & CLI Output)
                                       │
                                       ▼
                          Experiment Comparison Suite
                            (experiment_runner.py)
                     [Direct-only, Always-retrieve,
                      Fixed-verify, Adaptive Policy]
```

---

## 3. Detailed Component Plan

### 3.1. Query & Uncertainty Signals (`src/signals.py`)
- **Signal Extraction:**
  - Query length, token count, syntax cues.
  - Arithmetic expression detection (patterns, operators, numbers).
  - Ambiguity / underspecification detection (missing referents, bare comparatives like "Which one is better?", pronoun without context).
  - Empirical / temporal / factual indicators (entity queries, "who", "when", "what year", time-bound events).
  - Unknowability / speculation cues (future predictions, private/unknowable states).
- **Uncertainty Signals:**
  - Semantic confidence heuristics (calibrated score between 0.0 and 1.0).
  - Retrieval necessity score.
  - Calculation fit score.

### 3.2. Sandboxed Deterministic Calculator (`src/calculator.py`)
- **Safety Guarantee:** Pure AST visitor pattern (`ast.NodeVisitor`) supporting:
  - Binary operations (`+`, `-`, `*`, `/`, `//`, `%`, `**`)
  - Unary operations (`+`, `-`)
  - Standard math functions (`sqrt`, `abs`, `round`, `log`, `sin`, `cos`, `tan`, `pow`)
  - Explicit rejection of variables, imports, builtins, loops, attribute lookups, or arbitrary Python expressions.
- **Output:** Exact computed value, step trace, execution latency, and error handling.

### 3.3. Evidence Retrieval Engine (`src/retrieval.py`)
- **Corpus:** Local structured repository of text documents (JSON / Markdown) with clear document IDs, titles, passages, and source metadata.
- **Retrieval Method:** TF-IDF / BM25 lexical passage matching with score thresholding.
- **Dynamic Ingestion:** Ability for users to add documents or passages at runtime.
- **Output:** Top-k ranked passages with passage text, source title, relevance score, and explicit indication when no evidence meets threshold.

### 3.4. Claim Verification & Revision (`src/verification.py`, `src/revision.py`)
- **Claim Extraction:** Parses draft response into atomic proposition claims.
- **Verification Logic:** Compares claims against retrieved evidence using token overlap, NLI heuristics, or LLM-based verification (if API key provided).
- **Status Enum:**
  - `VERIFIED`: Complete support in retrieved evidence.
  - `PARTIALLY_SUPPORTED`: Some claims supported, others ungrounded.
  - `UNSUPPORTED`: Direct contradiction or complete absence of grounding.
  - `NOT_CHECKED`: Verification action was bypassed (e.g., Direct or Clarify action).
- **Revision:** Drops or reframes unverified assertions when verification reveals hallucination.

### 3.5. Cost Tracking & Accounting (`src/cost_tracker.py`)
- **Separation of Concepts:**
  - **Estimated Prior Cost:** Action cost estimate used for routing decisions ($C_{\text{est}}$).
  - **Measured Operational Cost:** Actual execution latency (ms), input/output tokens, number of model calls, number of tool calls, and calculated USD cost based on provider pricing tables.
- **Transparent Units:** Clearly labels model tokens vs monetary fractions vs latency.

### 3.6. Action Controller & Policy (`src/controller.py`, `src/policy.py`)
- **Decision Engine:**
  - Phase 1 Baseline: Transparent, documented, testable rules.
  - Cost-aware formulation: $a^* = \arg\max_a [\text{Benefit}(a \mid x) - \lambda \cdot \text{Cost}(a)]$.
  - Inspectable rationale generation detailing which signals triggered the decision.
- **Execution Orchestration:** Calls appropriate handler for the selected action.

### 3.7. LLM Provider Adapter (`src/llm_provider.py`)
- **Multi-Provider Support:**
  - Offline/Local Mock Provider: Default zero-cost deterministic draft responses based on local knowledge and templates. No API key required.
  - OpenAI / Gemini API Provider: Reads `OPENAI_API_KEY` or `GEMINI_API_KEY` from environment variables.
- **Resilience:** Graceful fallback to offline mode if API call times out, fails, or credentials are absent. No secrets logged or exposed.

### 3.8. Persistent Logging & Export (`src/logging_service.py`)
- **Storage:** SQLite database + append-only JSONL files in `results/logs/`.
- **Fields:** Run ID, timestamp, query, policy, signals, selected action, rationale, answer, evidence citations, verification status, tokens, latency, costs.
- **Export:** Exportable to CSV and JSON formats.

### 3.9. Experiment Runner & Baselines (`src/experiment_runner.py`)
- **Benchmark Suite:** Diverse curated test queries spanning arithmetic, factual QA, ambiguous requests, and unanswerable questions with reference labels.
- **Policies Evaluated:**
  1. `Direct-Only` (Never retrieve or verify)
  2. `Always-Retrieve` (Always fetch evidence)
  3. `Fixed Verification` (Always retrieve + verify)
  4. `Adaptive Policy` (Uncertainty & cost-aware routing)
- **Metrics Computed:** Correctness rate, Evidence support rate, Unsupported-claim rate, Abstention precision/recall, Total tokens, Total latency, Cumulative cost.

### 3.10. Streamlit Web Interface (`app.py`)
- **Research Dashboard:**
  - **Main Query Interface:** Query input, example selector, execute button, real-time result cards.
  - **Action Decision Panel:** Selected action badge, detailed decision rationale, signal breakdown.
  - **Evidence & Verification Panel:** Passages, source metadata, claim support breakdown, verification badges.
  - **Cost & Performance Metrics:** Latency gauge, token counter, tool calls, estimated vs measured cost cards.
  - **Experiment & Benchmark Panel:** Interactive runner comparing the 4 baselines on the benchmark suite, displaying a comparative metrics table and Pareto trade-off chart.
  - **Corpus Management:** View and add new documents to the local corpus.
  - **History & Export:** View session log, download as JSON/CSV.

---

## 4. Testing & Validation Strategy

1. **Unit Tests (`tests/test_calculator.py`):**
   - Correct arithmetic evaluation (`+`, `-`, `*`, `/`, `**`, etc.).
   - Division by zero handling.
   - Malicious/unsafe input rejection (`import os`, `__import__`, `eval()`, open files).
2. **Unit Tests (`tests/test_signals.py`):**
   - Arithmetic detection, ambiguity cues, factual indicators, unanswerable queries.
3. **Unit Tests (`tests/test_retrieval.py`):**
   - Relevant query retrieval, non-matching query handling, empty corpus handling.
4. **Unit Tests (`tests/test_verification.py`):**
   - Verified, partially supported, unsupported, and not-checked status transitions.
5. **Unit Tests (`tests/test_controller.py`):**
   - Complete branch coverage (`direct`, `calculate`, `retrieve`, `clarify`, `abstain`).
   - Cost trade-off parameter ($\lambda$) sensitivity.
6. **Unit Tests (`tests/test_llm_provider.py`):**
   - Offline fallback, missing API key handling, graceful error handling.
7. **Unit Tests (`tests/test_cost_tracker.py` & `tests/test_logging.py`):**
   - Cost accounting accuracy, SQLite/JSONL logging, CSV export.
8. **Unit Tests (`tests/test_experiment_runner.py`):**
   - Baseline comparative execution across all 4 policies on identical inputs.

---

## 5. Implementation Roadmap & Milestones

- [x] **Phase 1: Transparent Rule-Based Baseline & Core Architecture (Completed)**
  - Safe restricted AST calculator (`src/calculator.py`).
  - Query signal and uncertainty extraction (`src/signals.py`).
  - Local BM25 corpus retrieval (`src/retrieval.py` + `data/corpus/knowledge_base.json`).
  - Dual cost tracker: prior cost units $C_{\text{prior}}$ vs measured monetary USD (`src/cost_tracker.py`).
  - Multi-policy comparative experiment runner (`src/experiment_runner.py`).
  - SQLite and JSONL persistent logger (`src/logging_service.py`).
  - Interactive Streamlit dashboard (`app.py`) and CLI controller (`controller.py`).

- [x] **Phase 2: Pluggable Model Adapters & Prompt Versioning (Completed)**
  - Unified adapter interface (`BaseModelAdapter`) with `OfflineLocalAdapter`, `OpenAIAdapter`, `GeminiAdapter`, and `OllamaAdapter` (`src/model_adapter.py`).
  - Versioned prompt manager (`PromptManager`) with `v1.0-standard`, `v2.0-concise`, and `v1.1-with-confidence-elicitation`.
  - Configurable generation decoding settings (`GenerationSettings`: temperature, top-p, max-tokens).
  - Explicit experimental confidence logging (`model_reported_confidence`, non-ground-truth disclaimers).
  - Schema extension with graceful API error catching (`tool_errors`).

- [x] **Phase 3: Deep Mitigation Tools & Verification Loop (Completed)**
  - **Structured Clarification Dialogue:** Ambiguity classification, question formulation, and multi-turn state resolution (`src/clarification.py`, `MitigationController.resolve_clarification_turn()`).
  - **Propositional Claim Verification & Contradiction Detection:** Granular sentence splitting, numerical and entity contradiction detection (`ClaimStatus.CONTRADICTED`), and exact verbatim quote attribution (`src/verification.py`).
  - **Iterative Revision & Re-Verification Loop:** Automatic redaction and hedging of unsupported claims, followed by a second-pass re-verification check to evaluate whether defects were eliminated (`src/revision.py`).
  - **Insufficient-Evidence Gating & Abstention:** Gated retrieval confidence routing: when corpus evidence does not meet relevance threshold, controller safely abstains instead of hallucinating (`src/controller.py`).
  - **Streamlit Interactive UI:** Clarification response input and re-evaluator, contradiction alerts, verbatim quotes, and post-revision re-verification badges (`app.py`).
  - **Test Suite:** 54 passing unit tests covering all phases (`tests/test_actions.py`, `test_controller.py`).

- [x] **Phase 4: Statistical Uncertainty Calibration & Learned Cost Utility (Completed)**
  - **Multi-Sample Semantic Disagreement:** Evaluates token divergence (Jaccard distance), lexical token entropy, and sample variance across multiple candidate responses (`src/uncertainty.py`).
  - **Statistical Calibration Engine:** Expected Calibration Error (ECE), Brier score, and post-hoc temperature scaling parameter fitting on development sets (`src/calibration.py`).
  - **Strict Split & Leakage Prevention:** Guaranteed disjoint partitioning of Development vs Held-Out Test sets (`dev_split` vs `test_split`) with automated zero-overlap validation.
  - **Learned Cost-Aware Decision Policy:** Implementation of $a^* = \arg\min_a [ \widehat{L}_{\text{quality}}(a \mid x) + \lambda \cdot \text{CostPrior}(a) ]$ (`src/policy.py`).
  - **Minimalist White UI & Zero Emojis:** Complete UI overhaul in `app.py` adopting a research lab aesthetic (slate typography on white cards, no emojis), plus a dedicated Phase 4 Uncertainty & Calibration panel.
  - **Test Suite:** 61 passing unit tests (`tests/test_calibration_policy.py`, `test_controller.py`).

- [ ] **Phase 5: Evaluation and Analysis (Next)**
  - Full held-out benchmark evaluation with detailed ablation analysis.
  - Confusion matrix of action decisions vs ground-truth archetypes.
  - Latency, token usage, and cost-efficiency trade-off curve visualizations.
