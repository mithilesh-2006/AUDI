# Adaptive, Cost-Aware Hallucination Mitigation for Large Language Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-61%20passed-brightgreen.svg)](test_controller.py)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](app.py)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An inspectable, modular research prototype investigating whether **adaptive selection of hallucination-mitigation actions** improves the trade-off between response reliability and operational cost compared with uniform, high-cost pipelines.

---

## 1. Research Overview & Problem Statement

Language models often confabulate facts, fail at multi-step arithmetic, guess on ambiguous requests, or speculate on unknowable events. Existing mitigation pipelines often apply the same expensive sequence (retrieval + reasoning prompts + multi-step verification + revision) to every query—incurring unnecessary latency, token expenditure, and tool calls.

This project implements an **adaptive decision controller** that inspects query features and uncertainty signals at inference time to choose the most cost-effective action:

$$a^* = \arg\max_{a \in A} \left[ \text{EstimatedBenefit}(a \mid x) - \lambda \cdot \text{EstimatedCost}(a) \right]$$

Where:
- $a \in A$ is a candidate mitigation action.
- $\text{EstimatedBenefit}(a \mid x)$ represents predicted reduction in error/loss.
- $\text{EstimatedCost}(a)$ represents the action's computational cost.
- $\lambda$ controls the trade-off between factual reliability and operational expenditure.

---

## 2. Architecture

```text
                                  User Query
                                       │
                                       ▼
                       Feature & Signal Extractor (src/signals.py)
                   [Intent, Ambiguity, Arithmetic, Knowledge Need]
                                       │
                                       ▼
                       Cost-Aware Router (src/controller.py)
                  Objective: a* = argmax [Benefit(a|x) - λ * Cost(a)]
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
                         │    - Contradiction Detection
                         │    - Verbatim Quote Attribution
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
                     [Direct-only, Always-retrieve,
                      Fixed-verify, Adaptive Policy]
```

---

## 3. Candidate Actions & Mitigation Tools (Phase 3)

| Action | Handler | Implementation Details | Purpose | Cost Unit |
|---|---|---|---|---|
| **Direct** | `src/model_adapter.py` | Operational | Unassisted draft generation via pluggable adapters (Offline, OpenAI, Gemini, Ollama) | 1.0 |
| **Calculate** | `src/calculator.py` | Operational | Sandboxed AST evaluator for deterministic arithmetic (rejects unsafe code / DoS) | 0.2 |
| **Retrieve** | `src/retrieval.py` | Operational | BM25 lexical passage matching over local verified corpus with score gating | 3.5 |
| **Clarify** | `src/clarification.py` | Operational | Formulates structured clarification queries and resolves multi-turn disambiguation | 1.5 |
| **Verify** | `src/verification.py` | Operational | Decomposes answers into claims, checks numerical/entity contradictions, and attributes verbatim quotes (`SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`) | 5.0 |
| **Revise** | `src/revision.py` | Operational | Redacts/hedges unsupported claims and runs a post-revision re-verification loop | 6.5 |
| **Abstain** | `src/controller.py` | Operational | Refuses to guess on speculative future events, unknowable private states, or insufficient evidence | 0.4 |

> **Crucial Rule:** The system **only** marks claims `Verified` when an actual evidence-based verification procedure has successfully completed. Retrieval alone is **not** verification.

---

## 4. Technology Stack

- **Language:** Python 3.10+
- **Frontend Dashboard:** Streamlit (`app.py`) with clean white minimalist theme (zero emojis)
- **Backend Core:** Modular Python services (`src/`)
- **Arithmetic Safety:** Python `ast` Visitor (zero `eval()`)
- **Retrieval Engine:** Lexical BM25 passage matching with threshold gating (`src/retrieval.py`)
- **Clarification:** Structured clarification engine and multi-turn state resolution (`src/clarification.py`)
- **Verification:** Propositional claim decomposition with contradiction detection and verbatim quote attribution (`src/verification.py`)
- **Revision:** Automated hedging/redaction with post-revision re-verification loop (`src/revision.py`)
- **Uncertainty Estimation:** Multi-sample semantic disagreement, Jaccard distance, and token entropy (`src/uncertainty.py`)
- **Calibration Engine:** Expected Calibration Error (ECE), Brier score, temperature scaling, and zero-leakage dataset partitioner (`src/calibration.py`)
- **Cost-Sensitive Decision Policy:** Calibrated quality loss vs cost penalty optimization (`src/policy.py`)
- **LLM Integration:** Pluggable adapter (`OfflineLocalAdapter`, `OpenAIAdapter`, `GeminiAdapter`, `OllamaAdapter`) with prompt versioning
- **Experiment Storage:** SQLite (`results/experiments.db`) and JSONL (`results/logs/runs.jsonl`)
- **Test Framework:** Python `unittest` (61 tests passing)

---

## 5. Quick Start Guide

### 5.1 Installation
Clone or navigate to the repository directory and install dependencies:

```bash
git clone https://github.com/your-username/Adaptive-Uncertainty-Driven-Inference.git
cd Adaptive-Uncertainty-Driven-Inference

# (Recommended) Activate virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/macOS

pip install -r requirements.txt
```

### 5.2 Option A: Launch Interactive Streamlit Web Application
To launch the research dashboard in your browser:

```bash
streamlit run app.py
```

The web dashboard provides:
1. **Interactive Query Analyzer:** Real-time action decision badge, rationale inspector, candidate action utility table, and answer display.
2. **Evidence & Verification Inspector:** Passage citations and claim-level verification badges.
3. **Cost & Accounting Panel:** Latency, tokens, tool invocations, and estimated vs measured costs.
4. **Multi-Policy Benchmark Suite:** Runs 4 policies (`Direct-Only`, `Always-Retrieve`, `Fixed-Verify`, `Adaptive`) on identical benchmark examples and computes comparative accuracy, support, and cost.
5. **Knowledge Corpus Explorer:** Browse indexed documents and add new documents dynamically.
6. **Logging & Export:** Inspect previous runs and download logs as CSV or JSON.

### 5.3 Option B: Run Command-Line Interface (CLI)

```bash
# Run demonstration suite across candidate actions
python controller.py --demo

# Run single query
python controller.py --query "Calculate (120 + 45) / 5"
python controller.py --query "Which one is better?"
python controller.py --query "What is the elevation of Mount Everest?"
python controller.py --query "Will Bitcoin hit $500,000 in 2035?"

# Interactive REPL session
python controller.py
```

---

## 6. Optional Remote LLM API Setup

The prototype **runs 100% locally in Offline Demonstration Mode** without any API keys or paid accounts.

To optionally connect live neural models:
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Configure your API key:
   - For **OpenAI**: Set `LLM_PROVIDER=openai` and `OPENAI_API_KEY=sk-...`
   - For **Google Gemini**: Set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY=AIza...`
3. Restart the application or CLI.

*Note: If no API key is provided, the prototype gracefully runs in local mode. API keys are never hardcoded, exposed in the UI, or logged.*

---

## 7. Running Automated Tests

Run the complete 36-test automated test suite:

```bash
# Run via test_controller.py
python test_controller.py

# Or run via unittest discovery
python -m unittest -v
```

### Test Coverage Highlights:
- **`tests/test_calculator.py`:** Correct arithmetic operations, division by zero, DOS large exponent rejection, and strict security rejection of arbitrary code, imports, variables, and builtins.
- **`tests/test_signals.py`:** Observable query cues, arithmetic detection, ambiguity cues, factual triggers, and unanswerability detection.
- **`tests/test_retrieval.py`:** Matching passage retrieval, non-matching queries, threshold gating, and dynamic document addition.
- **`tests/test_verification_revision.py`:** Claim decomposition, status assignment (`Verified`, `Partially Supported`, `Unsupported`), and text revision.
- **`tests/test_controller_routing.py`:** Full routing branch coverage (`calculate`, `clarify`, `abstain`, `retrieve`, `direct`) and baseline policy overrides.
- **`tests/test_llm_provider.py`:** Offline local provider, missing API key handling, and API timeout/error handling.
- **`tests/test_cost_accounting.py`:** Cost accounting accuracy, SQLite/JSONL persistence, and CSV/JSON export.
- **`tests/test_experiment_runner.py`:** Multi-policy benchmark execution across identical inputs.

---

## 8. Repository Layout

```text
Adaptive-Uncertainty-Driven-Inference/
├── PROJECT_DESCRIPTION.md     # Primary research specification & objectives
├── IMPLEMENTATION_PLAN.md     # Engineering plan mapping requirements to code
├── LIMITATIONS.md             # Known limitations and research boundaries
├── NEXT_STEPS.md              # Prioritized research & development roadmap
├── README.md                  # System documentation & quickstart
├── requirements.txt           # Python dependency manifest
├── .env.example               # Template environment configuration
├── .gitignore                 # Exclusion rules for secrets, caches, and logs
├── controller.py              # Root CLI entrypoint and interactive interface
├── test_controller.py         # Root test runner
├── app.py                     # Streamlit web application
├── src/                       # Modular application source
│   ├── __init__.py
│   ├── signals.py             # Feature & uncertainty signal extraction
│   ├── calculator.py          # Restricted AST-based safe math evaluator
│   ├── retrieval.py           # BM25 lexical evidence retriever
│   ├── verification.py        # Propositional claim verifier
│   ├── revision.py            # Answer reviser
│   ├── cost_tracker.py        # Operational resource and cost accounting
│   ├── llm_provider.py        # Multi-provider adapter (Offline/OpenAI/Gemini)
│   ├── logging_service.py     # SQLite and JSONL persistence service
│   ├── policy.py              # Cost-aware & baseline routing policies
│   ├── controller.py          # Main orchestration controller
│   └── experiment_runner.py   # Baseline evaluation runner
├── data/                      # Data assets
│   ├── corpus/                # Local verified document corpus
│   │   └── knowledge_base.json
│   └── benchmarks/            # Benchmark evaluation queries
│       └── evaluation_set.json
├── tests/                     # Automated unit and integration test suite
│   ├── test_calculator.py
│   ├── test_signals.py
│   ├── test_retrieval.py
│   ├── test_verification_revision.py
│   ├── test_controller_routing.py
│   ├── test_llm_provider.py
│   ├── test_cost_accounting.py
│   └── test_experiment_runner.py
└── results/                   # Generated experiment logs and database
    ├── experiments.db         # SQLite persistent database
    └── logs/
        └── runs.jsonl         # Append-only JSONL run logs
```

---

## 9. Cost Accounting & Metric Definitions

The system maintains a strict separation between:
1. **Estimated Prior Action Cost Units ($C_{\text{prior}}$):** Dimensionless heuristic decision weights used by the router:
   - `calculate = 0.2`, `abstain = 0.4`, `direct = 1.0`, `clarify = 1.5`, `retrieve = 3.5`, `verify = 5.0`, `revise = 6.5`
2. **Measured Runtime Resources:**
   - **Latency (ms):** Measured wall-clock time from query arrival to final response.
   - **Tokens:** Actual input and output tokens consumed.
   - **Tool Invocations:** Count of calculator, retrieval, or verification operations.
   - **Monetary Cost ($ USD):** Computed from provider pricing tables ($0.0 for local offline mode).

---

## 10. Research Disclaimer & Limitations

- **No Universal Detection:** Rule-based heuristics cannot detect every subtle hallucination or determine the true underlying cause of an error.
- **Corpus Boundaries:** A missing passage in the local corpus does **not** prove that a claim is universally false.
- **Prior Utility:** Action utility weights are currently initial domain heuristics; they are not yet trained on large empirical reward datasets.
- For complete details, see [`LIMITATIONS.md`](LIMITATIONS.md) and [`NEXT_STEPS.md`](NEXT_STEPS.md).
#   A U D I  
 