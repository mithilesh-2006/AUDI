# AUDIT REPORT: AUDI Research Prototype Inspection & Gap Analysis

> **Project:** AUDI — Adaptive Uncertainty-Driven Inference  
> **Topic:** Adaptive, Cost-Aware Hallucination Mitigation for Large Language Models  
> **Audit Date:** 2026-10-20  
> **Auditor:** Senior AI Research Engineer & Systems Architect  
> **Repository:** `d:\Adaptive-Uncertainty-Driven-Inference`  
> **Baseline Test Status:** 67/67 Unit Tests Passing (1.29s wall-clock time)  

---

## 1. Executive Summary

This audit evaluates the codebase, experimental pipeline, configuration, evaluation datasets, and documentation against the research objectives outlined in the Master Prompt.

The repository contains a functional, modular Python prototype featuring:
- A central controller (`MitigationController`) with policy routing.
- Feature and uncertainty signal extraction (`SignalExtractor`, `UncertaintyEstimator`).
- A sandboxed AST arithmetic calculator (`SafeCalculator`).
- A local BM25 corpus retriever (`LocalCorpusRetriever`).
- Claim-level verification and iterative revision (`EvidenceVerifier`, `AnswerReviser`).
- Clarification turn handling (`ClarificationHandler`).
- Multi-provider model adapters (`OfflineLocal`, `OpenAI`, `Gemini`, `Ollama`).
- Resource and cost tracking (`CostTracker`).
- Temperature scaling calibration (`CalibrationEngine`).
- A 5-policy benchmark runner (`ExperimentRunner`).
- A Streamlit dashboard (`app.py`) with a clean white minimalist theme.

However, several critical scientific and architectural weaknesses were uncovered that must be addressed to ensure scientific rigor, reproducibility, and alignment with the decision-theoretic framework.

---

## 2. Existing Implementation Status & Component Inventory

| Subsystem / Module | File Path | Status | Verification & Functional State |
| :--- | :--- | :--- | :--- |
| **Controller** | `src/controller.py` | Complete | Executes all 7 actions (`direct`, `calculate`, `retrieve`, `clarify`, `verify`, `revise`, `abstain`). Computes execution time comparisons. |
| **Signal Extraction** | `src/signals.py` | Complete | Rule/regex based signal extraction for arithmetic, ambiguity, factual cues, and speculative patterns. |
| **AST Calculator** | `src/calculator.py` | Complete | Restricted AST visitor; handles binary/unary ops, whitelisted math functions/constants; rejects code execution and large exponents ($>1000$). |
| **Local Retrieval** | `src/retrieval.py` | Complete | In-memory BM25 index over 6 corpus documents; supports thresholding and dynamic addition. |
| **Claim Verification** | `src/verification.py` | Partial / Needs Alignment | Propositional decomposition and entity alignment exist, but uses `UNSUPPORTED` instead of the required operational taxonomy (`SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`, `NOT_CHECKABLE`). |
| **Answer Revision** | `src/revision.py` | Complete | Replaces/hedges contradicted claims and triggers re-verification. |
| **Clarification Dialogue**| `src/clarification.py`| Complete | Manages session state and resolves disambiguated queries. |
| **Model Adapter** | `src/model_adapter.py`| Complete | Implements Offline Mock, OpenAI, Gemini, and Ollama; handles prompt versions and confidence parsing. |
| **Legacy Provider** | `src/llm_provider.py` | Redundant / Deprecated | Earlier Phase 1 provider class. Duplicates logic with `model_adapter.py`. |
| **Cost Tracker** | `src/cost_tracker.py` | Complete | Computes proxy cost units, token counts, tool calls, and monetary API costs. |
| **Uncertainty Estimator** | `src/uncertainty.py` | Complete / Minor Gap | Computes multi-sample Jaccard disagreement and composite score. Needs explicit accounting of additional sample token costs. |
| **Calibration Engine** | `src/calibration.py` | Complete | Computes ECE, MCE, Brier score, and fits temperature $T$ on dev splits. |
| **Benchmark Runner** | `src/experiment_runner.py`| Complete / Needs Metrics| Runs 5 policies over dataset; computes accuracy, overhead, latency. Needs abstention precision/recall and false-abstention metrics. |
| **CLI Entrypoint** | `controller.py` | Complete | Functional terminal interface with timing benchmarks. |
| **Web Dashboard** | `app.py` | Complete | 5 Streamlit workspaces with white minimalist theme and zero emojis. |

---

## 3. Current Baseline Test Results

All 67 existing unit tests were executed under Python 3.14.7:
```text
Ran 67 tests in 1.290s
OK
```
Test module distribution:
- `test_actions.py` (5 tests): Clarification dialogue, numerical contradiction, quote attribution, revision loop, evidence gating.
- `test_calculator.py` (6 tests): Basic arithmetic, division by zero, DoS exponent rejection, malformed inputs, security rejection.
- `test_calibration_policy.py` (5 tests): Temperature scaling, learned cost-aware routing, full controller integration, lexical disagreement.
- `test_controller_routing.py` (8 tests): Action routing across all archetypes, empty queries, fixed overrides, execution time comparison.
- `test_cost_accounting.py` (2 tests): Prior units vs monetary USD, SQLite logging and CSV export.
- `test_evaluation_analysis.py` (5 tests): 5-policy comparative benchmark, confusion matrix, mistake recording, held-out split, ablations.
- `test_experiment_runner.py` (1 test): Multi-policy comparative execution consistency.
- `test_llm_provider.py` (6 tests): Fallbacks, mock generation, missing keys, timeouts.
- `test_model_adapter.py` (11 tests): Provider factory, mock generations, confidence parsing, timeout handlers.
- `test_retrieval.py` (4 tests): Ingestion, matching, non-matching, empty query.
- `test_signals.py` (5 tests): Cue detection for math, ambiguity, factual, speculation, empty inputs.
- `test_verification_revision.py` (5 tests): Decomposition, full verification, partial support, unsupported claims, revision.

---

## 4. Missing Functionality & Research Design Weaknesses

### 4.1. Verification Operational Label Taxonomy
- **Weakness:** `src/verification.py` currently labels low-overlap claims as `UNSUPPORTED`.
- **Requirement:** The operational labels must explicitly be:
  `SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`, `NOT_CHECKABLE`.
- **Research Principle:** A claim must **never** be labeled false or contradicted merely because no supporting passage was found in the local corpus. An absence of local evidence must be classified as `INSUFFICIENT_EVIDENCE`. Non-empirical or conversational claims must be labeled `NOT_CHECKABLE`.

### 4.2. Benchmark Dataset Size, Diversity, and Provenance
- **Weakness:** `data/benchmarks/evaluation_set.json` contains only 12 synthetic queries.
- **Missing Archetypes:**
  - Time-sensitive factual questions with dated ground truth.
  - Retrieval failure cases (factual questions where the corpus lacks the answer).
  - Distractor / misleading evidence passages.
  - Confidently wrong / common misconceptions.
- **Missing Metadata Fields:** Examples lack `answerability_label`, `evaluation_rubric`, `evidence_doc_ids`, `claim_annotations`, `dataset_split` (`train`, `dev`, `test`), `provenance`, and `evaluation_notes`.
- **Provenance Transparency:** The existing 12 items were synthetically authored for unit testing. The benchmark must be versioned, expanded with documented provenance, and partitioned into fixed splits.

### 4.3. Data Leakage and Pre-Decision vs. Post-Decision Uncertainty
- **Weakness:** In `src/uncertainty.py`, `estimate_uncertainty` probes `self.retriever.retrieve(query)` to assess evidence availability during pre-routing decision time.
- **Concern:** If this probe is treated as zero-cost, it obscures retrieval overhead.
- **Remedy:** Clearly delineate pre-decision feature signals (query-level lexical indicators) from post-retrieval evidence availability, and ensure that if a retrieval probe is executed at decision time, its latency and tool call are explicitly accounted for in `cost_tracker.py`.

### 4.4. Duplicated Provider Architecture
- **Weakness:** Both `src/llm_provider.py` and `src/model_adapter.py` implement model provider abstractions.
- **Remedy:** Harmonize `src/llm_provider.py` to act as a backwards-compatible wrapper around `src/model_adapter.py` to eliminate dual maintenance.

### 4.5. Evaluation Metric Completeness
- **Weakness:** `src/experiment_runner.py` tracks overall accuracy, latency, and cost, but lacks:
  - Abstention Precision: Of all abstentions, what proportion were genuinely unanswerable?
  - Abstention Recall: Of all unanswerable queries, what proportion were correctly abstained from?
  - False Abstention Rate: How often did the system abstain on an answerable query?
  - Claim-level Support Rate, Contradiction Rate, and Insufficient Evidence Rate.

---

## 5. Security & Reproducibility Concerns

1. **Deterministic Calculator Security:** Verified. Uses `ast.NodeVisitor`, rejects attribute access, function calls outside whitelist, and exponents $>1000$.
2. **Secret Handling:** Verified. `OPENAI_API_KEY` and `GEMINI_API_KEY` are read exclusively from environment variables (`.env`). No API keys or sensitive values are committed.
3. **Database Logging Integrity:** Relational schema in SQLite (`results/experiments.db`) stores run parameters, tokens, costs, and decisions.

---

## 6. Prioritized Implementation Plan

1. **Task 1 — Verification Operational Label Alignment (`src/verification.py`, `src/revision.py`):**
   - Update `ClaimStatus` to use `SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`, and `NOT_CHECKABLE`.
   - Update claim evaluation logic so missing evidence yields `INSUFFICIENT_EVIDENCE`, maintaining backward compatibility aliases.
2. **Task 2 — Comprehensive, Versioned Benchmark Dataset (`data/benchmarks/evaluation_set.json` & `data/corpus/`):**
   - Expand the benchmark to 24+ curated, multi-category examples with full schema metadata (task category, answerability, reference answers, claim annotations, train/dev/test splits, provenance).
   - Expand `data/corpus/knowledge_base.json` with corresponding authoritative reference passages.
3. **Task 3 — Benchmark Evaluation Metrics & Abstention Precision/Recall (`src/experiment_runner.py`):**
   - Implement claim-level metrics (Supported Rate, Contradiction Rate, Insufficient Evidence Rate).
   - Implement Abstention Precision, Recall, and False Abstention Rate.
4. **Task 4 — Provider Unification & Uncertainty Cost Transparency (`src/llm_provider.py`, `src/uncertainty.py`):**
   - Deprecate standalone provider logic in `src/llm_provider.py` and route through `model_adapter.py`.
   - Ensure uncertainty estimation transparently tracks pre-decision vs post-decision operations.
5. **Task 5 — Research Documentation Suite:**
   - Author `RESEARCH_METHODOLOGY.md` detailing research questions, hypotheses, formal loss objectives, dataset design, baseline definitions, and metrics.
   - Author `LIMITATIONS.md` documenting known technical, scientific, and empirical constraints.
   - Author `NEXT_STEPS.md` outlining future research phases.
   - Ensure `README.md` and `IMPLEMENTATION_PLAN.md` are completely updated.
6. **Task 6 — Verification, Test Suite Expansion & Validation Report:**
   - Add unit tests for `INSUFFICIENT_EVIDENCE`, `NOT_CHECKABLE`, and abstention precision/recall.
   - Run end-to-end evaluation and verify that all tests pass cleanly.
