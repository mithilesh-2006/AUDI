# Project: Adaptive Uncertainty-Driven Decision Controller (AUDC)
**Context Link:** Kaggle Implementation (https://www.kaggle.com/code/kavitha029/hallucination)

## 1. Project Abstract
We are developing a dynamic routing architecture for Large Language Models (LLMs) to optimize the trade-off between test-time inference cost and response accuracy. Static pipelines (like Always-RAG or Always-CoT) are economically inefficient. AUDC intercepts a cheap, zero-shot draft generation, extracts intrinsic neural uncertainty signals (logits), and uses a lightweight machine learning surrogate to predict the expected quality gain of routing the query to a heavier tool. 

If the approximated utility exceeds the penalty of compute cost and latency, the query is dynamically routed.

## 2. Mathematical Framework
We have formalized this as Cost-Sensitive Supervised Routing via Uncertainty Signatures using an approximate utility function:

$$ \hat{U}(a_t \mid o_t) = \mathbb{E}[\Delta Q \mid o_t, a_t] - \beta \cdot C(a_t) - \gamma \cdot L(a_t) $$

*   **$o_t$ (Observation Vector):** A 6-feature vector extracted from the base model's internal logits during a cheap zero-shot pass.
*   **$a_t$ (Action Space):** 
    *   $a_{direct}$: Zero-shot generation (Baseline).
    *   $a_{rag}$: Retrieval-Augmented Generation (via DuckDuckGo search).
    *   $a_{cot}$: Chain-of-Thought reasoning.
*   **$\mathbb{E}[\Delta Q]$:** Expected quality gain predicted by a Multi-Output Gradient Boosting surrogate model.
*   **$\beta, \gamma$:** Hyperparameters penalizing token cost ($C$) and latency ($L$).

## 3. The 6-Feature Observation Vector ($o_t$)
To bypass "Confident Hallucinations," the base LLM is allowed to generate up to 35 tokens. We extract:
1.  `mean_confidence`: Average Top-1 probability.
2.  `mean_margin`: Average gap between Top-1 and Top-2 probabilities.
3.  `mean_entropy`: Average Shannon entropy across the distribution.
4.  `max_entropy`: **Crucial for localized panic/hallucination detection.**
5.  `min_margin`: **Crucial for localized hesitation.**
6.  `sequence_length`: Tokens generated before EOS.

## 4. Evaluation Suite (4 Failure Modes)
We replaced the overly obscure PopQA with a balanced 4-dataset suite to trigger orthogonal failure modes:
1.  **GSM8K** (Reasoning $\rightarrow$ Targets $a_{cot}$)
2.  **TriviaQA** (Knowledge $\rightarrow$ Targets $a_{rag}$)
3.  **TruthfulQA** (Hallucinations $\rightarrow$ Targets $a_{rag}$)
4.  **HotpotQA** (Multi-hop $\rightarrow$ Targets $a_{rag}$ / $a_{cot}$)

## 5. Current Implementation Status
*   **Environment:** Kaggle Notebook (T4 GPU).
*   **Base Model:** `meta-llama/Meta-Llama-3-8B-Instruct` loaded in 4-bit precision (`device_map={"": 0}` to prevent CPU offload crashes).
*   **Completed Milestones:**
    *   Phase 1: Oracle Headroom validated (proven theoretical token reduction of ~50% with accuracy gains).
    *   Phase 2: Logit Extractor successfully separating mathematical reasoning confusion from factual hallucination.
    *   Phase 3: Surrogate Multi-Output Gradient Booster successfully trained to predict $\Delta Q$.
    *   Phase 4: Cost-Aware Controller simulated via parameter sweep, proving Pareto-shifting behavior.

## 6. Instructions for Anti Gravity
Please review the attached Python implementation (`audc_pipeline.py`), which represents the current state of our Kaggle codebase. Assist in optimizing the code, analyzing the parameter sweep logs, and preparing the architecture for the final paper manuscript drafting.