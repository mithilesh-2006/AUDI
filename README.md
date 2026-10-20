# Adaptive, Cost-Aware Hallucination Mitigation for Large Language Models (AUDI)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-70%20passed-brightgreen.svg)](test_controller.py)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](app.py)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An inspectable, modular research prototype and evaluation benchmark investigating whether **adaptive selection of hallucination-mitigation actions** improves the trade-off between response reliability and operational cost compared with uniform, high-cost pipelines.

---

## 1. Research Overview & Problem Statement

Language models often confabulate facts, fail at multi-step arithmetic, guess on ambiguous requests, or speculate on unknowable events. Existing mitigation pipelines often apply the same expensive sequence (retrieval + reasoning prompts + multi-step verification + revision) to every query—incurring unnecessary latency, token expenditure, and tool calls.

This project implements an **adaptive decision controller** that inspects query features and uncertainty signals at inference time to choose the most cost-effective action:

$$a^* = \arg\max_{a \in \mathcal{A}} \left[ \text{EstimatedBenefit}(a \mid x, u) - \lambda \cdot \text{EstimatedCost}(a) \right]$$

Where:
- $a \in \mathcal{A}$ is a candidate mitigation action (`direct`, `calculate`, `retrieve`, `clarify`, `verify`, `revise`, `abstain`).
- $\text{EstimatedBenefit}(a \mid x, u)$ represents predicted reduction in error/loss.
- $\text{EstimatedCost}(a)$ represents the action's computational cost weight.
- $\lambda$ controls the trade-off between factual reliability and operational expenditure.

---

## 2. Architecture & Subsystems

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

## 3. Supported Mitigation Actions

| Action | Routing Trigger / Condition | Executing Subsystem | Latency Profile | Hallucination Defense |
| :--- | :--- | :--- | :--- | :--- |
| **`DIRECT`** | Low uncertainty ($u < 0.25$), general definitional query, no external evidence needed. | Model Adapter | Baseline ($100 - 800\text{ ms}$) | Prevents redundant tool calls when parametric knowledge suffices. |
| **`CALCULATE`** | Parseable arithmetic syntax detected ($x_{\text{arithmetic}} = 1$). | Safe AST Calculator (`src/calculator.py`) | $\mathbf{\sim 0.2 - 1.4\text{ ms}}$ ($\mathbf{200\times - 500\times\text{ faster}}$) | Completely eliminates numerical and algebraic hallucinations with exact AST evaluation. |
| **`RETRIEVE`** | Factual inquiry requiring local reference corpus evidence ($x_{\text{factual}} = 1, u \ge 0.35$). | BM25 Retriever (`src/retrieval.py`) | Baseline $+ 15 - 30\text{ ms}$ | Grounds answer generation in authoritative passages. |
| **`CLARIFY`** | Missing context, bare comparative, or ambiguous pronoun ($x_{\text{ambiguity}} = 1$). | Clarification Handler (`src/clarification.py`) | $\mathbf{\sim 0.4 - 1.5\text{ ms}}$ | Prevents premature ungrounded guessing on under-specified inputs. |
| **`VERIFY`** | High-stakes factual query requiring claim alignment checking. | Evidence Verifier (`src/verification.py`) | Baseline $+ 25 - 60\text{ ms}$ (Grounding Investment) | Decomposes claims into atomic propositions and checks support/contradiction. |
| **`REVISE`** | One or more claims contradicted or unsupported by corpus. | Answer Reviser (`src/revision.py`) | Baseline $+ 30 - 70\text{ ms}$ | Edits, hedges, or removes unsupported claims automatically. |
| **`ABSTAIN`** | Inherently unknowable event, future speculation, or missing evidence. | Controller Policy (`src/policy.py`) | $\mathbf{\sim 0.3 - 1.2\text{ ms}}$ ($\mathbf{300\times - 400\times\text{ faster}}$) | Halts execution immediately, refusing to generate confabulated claims. |

---

## 4. Setup and Quickstart

### Prerequisites
- Python 3.10 or higher.
- Works 100% locally in offline mode without requiring paid API keys.

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/Adaptive-Uncertainty-Driven-Inference.git
cd Adaptive-Uncertainty-Driven-Inference

# Install dependencies
pip install -r requirements.txt
```

### Optional API Configuration
Copy `.env.example` to `.env` to configure remote LLM providers:
```bash
cp .env.example .env
```
Edit `.env` to supply optional credentials (`OPENAI_API_KEY` or `GEMINI_API_KEY`). If omitted, AUDI defaults to the zero-cost local offline provider.

---

## 5. Usage

### Interactive Research Dashboard (Streamlit)
```bash
streamlit run app.py
```
Open `http://localhost:8501` to access the 5 research workspaces:
1. **Interactive Query Analyzer:** Real-time query execution, badge indicators, latency benchmark profile cards, and decision rationale.
2. **Multi-Policy Benchmark Suite:** 5-policy comparative evaluation, Execution Latency Profiler, action confusion matrices, mistake analysis, and ablation runs.
3. **Statistical Uncertainty & Calibration:** Empirical Platt calibration curve visualizer, Brier score reporting, and Expected Calibration Error (ECE).
4. **Local Knowledge Corpus:** Searchable BM25 passage viewer with live document addition.
5. **Experiment Logs & Export:** Inspectable SQLite execution logs with one-click CSV export.

### Command-Line Interface (CLI)
```bash
# Deterministic arithmetic
python controller.py --query "Calculate (120 + 45) / 5"

# Ambiguity clarification
python controller.py --query "Which one is better?"

# Factual retrieval and verification
python controller.py --query "What is the elevation of Mount Everest according to the 2020 survey?"

# Future speculation abstention
python controller.py --query "Will Bitcoin hit $500,000 in 2035?"
```

### Running Unit & Integration Tests
```bash
python test_controller.py
```
Executes all 70 tests across the 12 test modules.

---

## 6. Research Documentation Directory

- [`PROJECT_DESCRIPTION.md`](PROJECT_DESCRIPTION.md): Comprehensive system specification and architecture.
- [`AUDIT_REPORT.md`](AUDIT_REPORT.md): Codebase inspection findings and prioritized improvements.
- [`RESEARCH_METHODOLOGY.md`](RESEARCH_METHODOLOGY.md): Scientific experimental design, loss formulations, and metrics.
- [`LIMITATIONS.md`](LIMITATIONS.md): Documented technical, scientific, and empirical limitations.
- [`NEXT_STEPS.md`](NEXT_STEPS.md): Future research and engineering roadmap.
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md): Phased engineering milestones and verification criteria.