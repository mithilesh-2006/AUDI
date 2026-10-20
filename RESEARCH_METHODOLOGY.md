# RESEARCH METHODOLOGY: Adaptive, Cost-Aware Hallucination Mitigation for Large Language Models

> **System:** AUDI (*Adaptive Uncertainty-Driven Inference*)  
> **Document Type:** Scientific Experimental Design & Evaluation Protocol  
> **Document Version:** 2.0 (Post-Audit Research Specification)  
> **Status:** Active Research Specification  

---

## 1. Research Questions & Hypotheses

### 1.1. Core Research Questions
- **RQ1 (Over-Mitigation Reduction):** Does an adaptive controller reduce unnecessary tool calls (retrieval, external verification) compared with fixed pipelines (Always-Retrieve, Fixed-Verify) without degrading answer correctness on simple or non-factual queries?
- **RQ2 (Quality-Cost Pareto Frontier):** Can an adaptive policy achieve a superior empirical Pareto frontier between factual accuracy and operational cost (latency, tokens, monetary spend) across a diverse query workload?
- **RQ3 (Uncertainty Signal Utility):** Which observable pre-decision signals (lexical patterns, AST parseability, query ambiguity indicators, semantic disagreement) correlate most strongly with the true necessity of mitigation?
- **RQ4 (Domain Archetype Performance):** How does adaptive routing perform across distinct task archetypes (deterministic arithmetic, factual retrieval, underspecified questions, and unknowable future states)?
- **RQ5 (Abstention Quality):** Does epistemic uncertainty routing achieve high abstention precision and recall on unanswerable queries while minimizing false abstentions on answerable queries?
- **RQ6 (Cost Parameter Sensitivity):** How does varying the cost penalty parameter $\lambda$ alter the equilibrium between risk aversion (aggressive verification) and efficiency (direct answers)?

### 1.2. Primary Research Hypothesis
$$\text{Hypothesis } H_1: \text{For a heterogeneous query distribution, an adaptive policy } \pi^*(a \mid x, u)$$
$$\text{achieves higher factual support and lower operational cost per correct response than any uniform fixed baseline.}$$

---

## 2. Decision-Theoretic Framework

### 2.1. Objective Function
Let $q$ denote the user query, $x(q)$ the observable query features, and $u(q) \in [0, 1]$ the calibrated epistemic uncertainty. The candidate action set is:
$$\mathcal{A} = \{\text{direct}, \text{calculate}, \text{retrieve}, \text{clarify}, \text{verify}, \text{revise}, \text{abstain}\}$$

The optimal action $a^*$ minimizes the expected joint loss under trade-off factor $\lambda \ge 0$:

$$a^* = \arg\min_{a \in \mathcal{A}} \left[ \mathbb{E}\left[L_{\text{quality}}(a \mid x(q), u(q))\right] + \lambda \cdot C_{\text{proxy}}(a) \right]$$

Equivalently formulated as utility maximization:

$$a^* = \arg\max_{a \in \mathcal{A}} \left[ \text{Benefit}(a \mid x, u) - \lambda \cdot C_{\text{proxy}}(a) \right]$$

Where:
- $\text{Benefit}(a \mid x, u)$: The expected reduction in answer quality loss achieved by action $a$.
- $C_{\text{proxy}}(a)$: The prior operational cost weight assigned to action $a$ ($\text{calculate}=0.2$, $\text{direct}=1.0$, $\text{clarify}=0.5$, $\text{retrieve}=3.5$, $\text{verify}=5.0$, $\text{abstain}=0.3$).
- $\lambda$: User-configured cost penalty governing the trade-off.

### 2.2. Uncertainty Calibration via Temperature Scaling
Uncalibrated heuristic uncertainty $u \in [0, 1]$ is mapped to an empirical error probability using a logistic calibration model fitted strictly on development data:

$$\hat{P}(\text{Error} \mid u) = \sigma\left(\frac{u - \beta}{T}\right) = \frac{1}{1 + \exp\left(-\frac{u - \beta}{T}\right)}$$

Where $T$ is the temperature scaling parameter and $\beta$ is the decision threshold bias.

---

## 3. Benchmark Dataset Design & Provenance

The evaluation benchmark (`data/benchmarks/evaluation_set.json`) comprises 24 curated examples partitioned into fixed development (`dev`, $N=12$) and held-out test (`test`, $N=12$) splits across 8 task archetypes:

| Task Archetype | Target Behavior | Answerability Label | Expected Action | Split (Dev / Test) |
| :--- | :--- | :--- | :--- | :--- |
| **Arithmetic** | Exact AST calculation | `answerable` | `calculate` | 2 / 2 |
| **Factual QA (Evidence Required)** | Grounded retrieval | `answerable` | `retrieve` | 2 / 2 |
| **Factual QA (Parametric)** | Direct parametric answer | `answerable` | `direct` | 1 / 1 |
| **Time-Sensitive Factual** | Dated evidence retrieval | `answerable` | `retrieve` | 1 / 1 |
| **Ambiguity** | Targeted clarification turn | `ambiguous` | `clarify` | 2 / 2 |
| **Unanswerable / Speculation** | Epistemic safe refusal | `unanswerable` | `abstain` | 2 / 2 |
| **Retrieval Deficient / Corpus Absent** | Insufficient evidence notice | `corpus_deficient` | `retrieve` (abstain) | 1 / 1 |
| **Misconception / Distractor** | Quote-attributed verification | `answerable` | `retrieve` | 1 / 1 |

### Provenance Standards
- No synthetic unit-test examples are presented as a public standard benchmark.
- Factual queries are derived from verifiable historical records, geographical surveys (e.g. 2020 Survey of Nepal/China), and physical constants (CODATA 2018).
- Splits are frozen to eliminate test-set leakage.

---

## 4. Verification Operational Taxonomy

Claim verification (`src/verification.py`) decomposes answers into atomic propositions and assigns one of four mutually exclusive operational statuses:

1. **`SUPPORTED`:** The assertion is directly grounded in retrieved corpus text with semantic overlap $\ge 0.50$ and verbatim quote attribution.
2. **`CONTRADICTED`:** The assertion directly conflicts with numerical or entity values in the authoritative evidence (e.g., claiming Mount Everest is 9,500m when evidence states 8,848.86m).
3. **`INSUFFICIENT_EVIDENCE`:** The local corpus does not contain sufficient passages to confirm or deny the assertion. **Crucially: An absence of local evidence is never interpreted as proof of falsehood.**
4. **`NOT_CHECKABLE`:** Subjective statements, conversational preamble, or philosophical assertions outside the scope of empirical verification.

---

## 5. Baseline Policies for Controlled Comparison

1. **Direct-Only Baseline:** Generates answers using model parametric knowledge without external tools ($C=1.0$).
2. **Always-Retrieve Baseline:** Executes BM25 retrieval for every query regardless of features ($C=3.5$).
3. **Fixed-Verify Baseline:** Unconditionally retrieves evidence, checks claims, and triggers revision for every query ($C=5.0$).
4. **Adaptive Rule Policy:** Evaluates observable syntactic and epistemic signals using transparent decision rules.
5. **Learned Cost-Aware Policy:** Minimizes empirical loss under calibrated uncertainty and cost penalty $\lambda$.

---

## 6. Evaluation Metrics & Scientific Accounting

All metrics report explicit numerators and denominators:

$$\text{Accuracy / Correctness} = \frac{\sum_{i=1}^N \mathbb{I}(\text{Answer}_i \text{ is correct according to rubric})}{N}$$

$$\text{Evidence Support Rate} = \frac{\sum_{i=1}^N \mathbb{I}(\text{Query}_i \text{ grounded in sufficient evidence})}{N}$$

$$\text{Unsupported Claim Rate} = \frac{\text{Contradicted Claims} + \text{Insufficient Evidence Claims}}{\text{Total Claims Checked}}$$

$$\text{Abstention Precision} = \frac{\text{True Abstentions on Unanswerable Queries}}{\text{Total Abstentions Invoked}}$$

$$\text{Abstention Recall} = \frac{\text{True Abstentions on Unanswerable Queries}}{\text{Total Unanswerable Queries in Set}}$$

$$\text{False Abstention Rate} = \frac{\text{Abstentions on Answerable Queries}}{\text{Total Answerable Queries in Set}}$$

$$\text{Cost per Correct Answer} = \frac{\text{Total Measured Monetary Cost (USD)}}{\max(\text{Total Correct Answers}, 1)}$$

---

## 7. Leakage Prevention Protocol

1. **Split Freezing:** Development sets (`split="dev"`) are used exclusively for fitting temperature parameters and heuristic tuning. Test sets (`split="test"`) remain held out until final benchmark reporting.
2. **Pre-Decision vs. Post-Decision Signal Separation:** Features used by the controller before action selection must only derive from query text and lightweight pre-decision features. External retrieval operations or multi-sample LLM calls are explicitly counted in tool invocations and operational latency.
