# AUDI: Adaptive Uncertainty-Driven Inference
## Adaptive, Cost-Aware Hallucination Mitigation for Large Language Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-70%20passed-brightgreen.svg)](test_controller.py)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](app.py)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An inspectable, reproducible research framework and empirical evaluation benchmark investigating whether **adaptive, uncertainty-guided selection of hallucination mitigation actions** improves the trade-off between factual accuracy and operational cost compared with uniform, heavy-duty mitigation pipelines.

---

## 1. Research Overview & Problem Formulation

Large Language Models (LLMs) suffer from distinct failure modes across query archetypes:
1. **Factual confabulation** on obscure or specialized knowledge.
2. **Arithmetic errors** on multi-step deterministic calculations.
3. **Premature ungrounded guessing** on underspecified or ambiguous queries.
4. **Unwarranted speculation** on inherently unknowable or future events.

Existing industry architectures typically adopt a **static, uniform mitigation pipeline**—such as querying external search engines or vector databases (RAG), applying self-reflection prompts, or running multi-agent verification on *every query*. This incurs substantial latency overhead, high token expenditures, and unnecessary tool calls on simple or non-factual queries.

### The AUDI Solution

AUDI introduces an **adaptive decision controller** that inspects query features, parametric confidence, and evidence availability at runtime to select the optimal mitigation action $a^* \in \mathcal{A}$:

$$\mathcal{A} = \{\text{DIRECT}, \text{CALCULATE}, \text{RETRIEVE}, \text{CLARIFY}, \text{VERIFY}, \text{REVISE}, \text{ABSTAIN}\}$$

The controller optimizes an expected utility objective balancing error reduction against execution cost:

$$a^* = \arg\max_{a \in \mathcal{A}} \left[ \mathbb{E}[\text{Benefit}(a \mid x, u)] - \lambda \cdot \text{Cost}(a) \right]$$

where:
- $x$ represents observable linguistic, structural, and semantic query signals.
- $u \in [0.0, 1.0]$ represents composite, calibrated uncertainty.
- $\text{Cost}(a)$ represents the normalized computational expenditure (latency, tokens, API fees).
- $\lambda \in [0.0, 1.0]$ is a user-tunable trade-off parameter governing risk tolerance vs. resource budget.

---

## 2. System Architecture

```text
                                   User Query (x)
                                         │
                                         ▼
                     Observable Signal Extractor (src/signals.py)
                  [Arithmetic, Ambiguity, Factual, Unknowable, Freshness]
                                         │
                                         ▼
                 Composite Uncertainty & Calibration (src/uncertainty.py,
                                                      src/calibration.py)
                - Multi-sample lexical disagreement (Jaccard distance)
                - Local corpus evidence availability
                - Platt / Temperature scaling: P_calibrated = σ(logit / T)
                                         │
                                         ▼
                     Cost-Aware Decision Policy (src/policy.py)
                  Objective: a* = argmin [Loss(a|x, u) + λ · Cost(a)]
                                         │
        ┌───────────────┬────────────────┼───────────────┬───────────────┐
        ▼               ▼                ▼               ▼               ▼
      DIRECT        CALCULATE         RETRIEVE        CLARIFY         ABSTAIN
   (Parametric)   (Safe AST Math)   (BM25 Corpus)  (Clarification)  (Safe Refusal)
        │               │                │               │               │
        └───────────────┼────────────────┼───────────────┴───────────────┘
                        │                ▼
                        │     Verification & Revision (src/verification.py,
                        │                              src/revision.py)
                        │     - Atomic claim decomposition
                        │     - Operational labels: SUPPORTED, CONTRADICTED,
                        │       INSUFFICIENT_EVIDENCE, NOT_CHECKABLE
                        │     - Entity & numerical discrepancy detection
                        │     - Automated claim hedging and correction
                        │                │
                        └────────────────┼───────────────────────────────┐
                                         ▼                               ▼
                           Cost & Latency Tracking             Final Response
                            (src/cost_tracker.py,              (Streamlit & CLI)
                             src/logging_service.py)
                                         │
                                         ▼
                          SQLite Persistence (WAL mode)
                           results/experiments.db
```

---

## 3. Supported Mitigation Actions

| Action | Routing Condition | Executing Subsystem | Latency Profile | Hallucination Defense |
| :--- | :--- | :--- | :--- | :--- |
| **`DIRECT`** | Low uncertainty ($u < 0.25$), standard definitional/factual inquiry. | Model Adapter | $\sim 0.1\text{ ms}$ (Offline) / $150-600\text{ ms}$ (LLM) | Prevents redundant tool calls when parametric knowledge is sufficient. |
| **`CALCULATE`** | Parseable arithmetic syntax detected ($x_{\text{arithmetic}} = 1$). | Safe AST Calculator (`src/calculator.py`) | $\sim 0.1 - 0.4\text{ ms}$ ($\mathbf{200\times - 500\times\text{ faster}}$) | Completely eliminates arithmetic and mathematical hallucinations via sandboxed AST evaluation. |
| **`RETRIEVE`** | Factual inquiry requiring external evidence ($x_{\text{factual}} = 1, u \ge 0.35$). | BM25 Retriever (`src/retrieval.py`) | Baseline $+ 10 - 25\text{ ms}$ | Grounds claim generation in authoritative knowledge base passages. |
| **`CLARIFY`** | Missing options, underspecified pronouns, or comparative ambiguity ($x_{\text{ambiguity}} = 1$). | Clarification Handler (`src/clarification.py`) | $\sim 0.1 - 0.3\text{ ms}$ | Prevents premature ungrounded guessing by prompting for missing criteria. |
| **`VERIFY`** | Factual claim requiring claim-level alignment check against evidence. | Evidence Verifier (`src/verification.py`) | Baseline $+ 15 - 45\text{ ms}$ | Decomposes text into atomic claims and flags contradicted/unsupported statements. |
| **`REVISE`** | One or more claims contradicted or unsupported by retrieved evidence. | Answer Reviser (`src/revision.py`) | Baseline $+ 20 - 50\text{ ms}$ | Corrects, hedges, or removes contradicted claims automatically. |
| **`ABSTAIN`** | Future speculation, unanswerable inquiry, or zero relevant corpus evidence. | Controller (`src/controller.py`) | $\sim 0.05 - 0.15\text{ ms}$ ($\mathbf{300\times - 400\times\text{ faster}}$) | Immediately refuses unanswerable queries, eliminating confabulation. |

---

## 4. Evaluated Decision Policies

AUDI evaluates and benchmarks 5 distinct policy streams on identical query sets:

1. **`direct_only` (Unassisted Baseline):** Uniformly routes all queries directly to the LLM without external tools, retrieval, or verification.
2. **`always_retrieve` (Uniform RAG Baseline):** Uniformly retrieves corpus documents for every query, regardless of whether retrieval is needed (e.g., executing RAG for arithmetic or ambiguity).
3. **`fixed_verify` (Uniform Verify Baseline):** Uniformly executes retrieval followed by propositional verification on all queries.
4. **`adaptive_rules` (Phase 1 Heuristic Policy):** Rule-based decision tree using deterministic lexical cues and thresholded signals.
5. **`learned_cost_aware` (AUDI Phase 4 Learned Policy):** Calibrated expected-utility policy balancing Platt-scaled uncertainty, multi-sample disagreement, and operational action costs under user trade-off parameter $\lambda$.

---

## 5. Dataset & Benchmark Pipeline

The repository integrates a comprehensive 20,000-sample validation dataset (`unified_eval_20000.csv`):
- **Categories:** `Factual Knowledge`, `Mixed Multi-hop`, `Multi-Step Reasoning`, `Calibration`.
- **Pipeline:** Managed by `DatasetManager` (`src/dataset.py`).
- **Data Splitting:** Automatically performs stratified sampling across task categories into:
  - **Train/Dev Split (`data/benchmarks/train_set.json`):** Used for temperature scaling parameter fitting ($T$) and ECE minimization.
  - **Held-Out Test Split (`data/benchmarks/test_set.json`):** Strictly isolated for held-out evaluation to guarantee zero test leakage.

---

## 6. Installation & Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.14).
- Runs 100% locally out-of-the-box using the deterministic `OfflineLocalProvider` (no paid API keys required).

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/Adaptive-Uncertainty-Driven-Inference.git
cd Adaptive-Uncertainty-Driven-Inference

# Install required dependencies
pip install -r requirements.txt
```

### Optional LLM API Configuration
To test with real cloud providers (OpenAI or Google Gemini), copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Provide `OPENAI_API_KEY` or `GEMINI_API_KEY` in `.env`. If left empty, AUDI runs seamlessly in local offline simulation mode.

---

## 7. Execution & Usage

### 1. Unified Benchmark & Calibration CLI (`benchmark.py`)
Run the scientific multi-policy evaluation suite directly from the command line:

```bash
# Run benchmark on 100 held-out test samples across all 5 policies with calibration
python benchmark.py --samples 100 --calibrate

# Evaluate specific policies with custom cost-accuracy trade-off lambda
python benchmark.py --samples 50 --lambda-cost 0.7 --policies direct_only adaptive_rules learned_cost_aware

# Force re-generation of stratified dataset splits from CSV
python benchmark.py --prepare-splits --samples 100
```

### 2. Interactive CLI Controller (`controller.py`)
Run single-query routing or execute the automated demonstration suite:

```bash
# Run the built-in multi-action demonstration
python controller.py --demo

# Test specific queries
python controller.py --query "Calculate (120 + 45) / 5"
python controller.py --query "Which one is better?"
python controller.py --query "What is the elevation of Mount Everest according to the 2020 survey?"
python controller.py --query "Will Bitcoin hit $500,000 in 2035?"
```

### 3. Streamlit Research Dashboard (`app.py`)
Launch the interactive research UI:

```bash
streamlit run app.py
```
Open `http://localhost:8501` to access:
- **Interactive Query Analyzer:** Live routing badges, timing breakdown, and sub-component outputs.
- **Multi-Policy Benchmark Suite:** 5-policy comparative evaluation, latency profiler, action confusion matrices, and ablation studies.
- **Statistical Uncertainty & Calibration:** Empirical Platt calibration curves, Brier score reporting, and reliability diagrams.
- **Local Knowledge Corpus:** Searchable BM25 passage viewer with live document addition.
- **Experiment Logs & Export:** Inspectable SQLite execution logs with one-click CSV export.

### 4. Unit & Integration Test Suite (`test_controller.py`)
```bash
python test_controller.py
```
Executes all 70 tests across 12 test modules, verifying safe AST arithmetic, BM25 retrieval, claim verification, multi-sample disagreement, ECE calibration, cost tracking, and SQLite concurrency.

---

## 8. Repository Structure

```text
Adaptive-Uncertainty-Driven-Inference/
│
├── src/                                  # Core AUDI Framework Modules
│   ├── calculator.py                    # Sandboxed AST Arithmetic Evaluator
│   ├── calibration.py                   # Platt / Temperature Scaling & ECE
│   ├── clarification.py                 # Multi-turn Ambiguity Resolution
│   ├── controller.py                    # Central MitigationController Orchestrator
│   ├── cost_tracker.py                  # Token, Latency & Monetary Accounting
│   ├── dataset.py                       # Dataset Ingestion & Stratified Splitter
│   ├── experiment_runner.py             # 5-Policy Comparative Benchmark Runner
│   ├── llm_provider.py                  # LLM Factory (Offline, OpenAI, Gemini)
│   ├── logging_service.py               # Concurrent SQLite (WAL) & JSONL Logger
│   ├── model_adapter.py                 # Pluggable Model Adapters & Prompts
│   ├── policy.py                        # Rule-Based & Learned Cost-Aware Policies
│   ├── retrieval.py                     # BM25 Local Corpus Passage Retriever
│   ├── revision.py                      # Propositional Claim Reviser & Hedger
│   ├── signals.py                       # Observable Query Signal Extractor
│   ├── uncertainty.py                   # Multi-sample Disagreement & Uncertainty
│   └── verification.py                  # Atomic Claim Propositional Verifier
│
├── tests/                                # Comprehensive Test Suite (70 Tests)
│   ├── test_actions.py                  # Verification & Revision Integration
│   ├── test_calculator.py               # AST Calculator & Security Tests
│   ├── test_calibration_policy.py       # ECE & Learned Policy Tests
│   ├── test_controller_routing.py       # End-to-End Routing Tests
│   ├── test_cost_accounting.py          # Cost & SQLite Logger Tests
│   ├── test_evaluation_analysis.py      # Benchmark & Ablation Tests
│   ├── test_experiment_runner.py        # Experiment Runner Suite
│   ├── test_llm_provider.py             # LLM Provider Fallback Tests
│   ├── test_model_adapter.py            # Model Adapter & Prompt Tests
│   ├── test_retrieval.py                # BM25 Retrieval Tests
│   ├── test_signals.py                  # Query Feature Extractor Tests
│   └── test_verification_revision.py    # Proposition Decomposition Tests
│
├── data/
│   ├── benchmarks/                      # Stratified train, dev, and test JSON splits
│   └── corpus/                          # Authoritative reference knowledge base
│
├── results/
│   ├── experiments.db                   # SQLite Experiment Run Database (WAL mode)
│   └── benchmark_summary.csv            # Evaluated Policy Metric Comparison Table
│
├── app.py                               # Streamlit Interactive Research Dashboard
├── benchmark.py                         # CLI Benchmark, Calibration & Evaluation Runner
├── controller.py                        # CLI Query Execution & Demonstration Engine
├── test_controller.py                   # Master Test Suite Runner
├── unified_eval_20000.csv               # 20,000-Sample Evaluation Dataset
├── requirements.txt                     # Minimal Python Dependencies
└── PROJECT_DESCRIPTION.md               # Complete Theoretical Specification
```

---

## 9. License

This project is licensed under the MIT License - see the `LICENSE` file for details.
