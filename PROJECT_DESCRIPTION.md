# Adaptive, Cost-Aware Hallucination Mitigation for Large Language Models

> **Document type:** Complete project description  
> **Project stage:** Research prototype — Phase 1 baseline  
> **Primary objective:** Investigate whether adaptive selection of mitigation actions can improve the reliability–cost trade-off of LLM responses.

---

## 1. Abstract

Large language models (LLMs) can generate fluent responses that contain factual errors, unsupported claims, incorrect reasoning, or fabricated details. These failures are commonly described as hallucinations, although the causes and definitions vary across tasks. Methods such as retrieval-augmented generation (RAG), reasoning-oriented prompting, answer revision, external verification, clarification, and abstention can reduce some errors. However, applying the same sequence of mitigation steps to every query may waste computation, increase latency, and introduce unnecessary tool calls.

This project proposes an **adaptive, cost-aware hallucination-mitigation controller**. Given a user question and the signals available at decision time, the controller estimates what kind of support the question may require and chooses an appropriate action. Candidate actions include direct generation, deterministic calculation, evidence retrieval, clarification, verification and revision, or abstention. The goal is not simply to minimize hallucinations or minimize cost independently, but to study the trade-off between response quality, factual support, and operational expense.

The initial prototype uses transparent rule-based routing to establish a reproducible baseline. Subsequent stages will connect an LLM, implement retrieval and verification tools, collect structured experiment logs, and evaluate more sophisticated uncertainty-aware and cost-sensitive policies. The system will be compared with fixed strategies such as direct-only generation and always-retrieve, alongside relevant published approaches where feasible. Performance will be measured using task-appropriate answer-quality metrics, evidence-support and hallucination annotations, abstention behavior, token use, tool calls, latency, and monetary cost when available.

The central hypothesis is that **choosing mitigation actions according to the needs of individual queries can achieve a better reliability–cost trade-off than applying one fixed strategy to every query**. This hypothesis must be tested experimentally. The project does not assume that its approach is novel or effective before a systematic literature review and controlled evaluation.

## 2. Introduction and Background

LLMs generate text by predicting likely continuations conditioned on an input context. Their fluency and broad knowledge make them useful across many tasks, but a plausible-sounding answer is not necessarily correct. A model may produce outdated facts, invent sources, make arithmetic mistakes, misunderstand an ambiguous request, or answer a question for which sufficient evidence is unavailable.

Hallucination mitigation is not a single problem with one universal solution. Different questions may require different kinds of support:

- A basic arithmetic question may be best handled by a deterministic calculator.
- A question about a recent event may need current, trustworthy evidence.
- A multi-step reasoning task may need a verifiable computation or structured reasoning procedure.
- An ambiguous request may require clarification rather than a guess.
- An unsupported or unknowable question may require an explicit limitation or abstention.
- A straightforward question may not benefit enough from additional retrieval or verification to justify their cost.

Many systems use fixed workflows. For example, a system might generate an answer, retrieve evidence, verify every claim, and revise the answer for every question. Such a workflow may be useful for high-risk tasks, but its expense may be unnecessary for simpler questions. At the other extreme, always answering directly can fail when a query requires external evidence.

This project explores whether a controller can allocate mitigation effort more selectively, using the question and available signals to choose a suitable action.

## 3. Problem Statement

A fixed LLM response pipeline applies the same or a limited set of mitigation operations regardless of the question's characteristics. This can lead to unnecessary computation on easy questions and insufficient support for difficult, ambiguous, time-sensitive, or unanswerable questions.

The problem is to design and evaluate a decision policy that selects mitigation actions based on available information while balancing answer quality, reliability, latency, and cost.

Formally, let:

- \(q\) be the input question;
- \(x(q)\) be the features and uncertainty signals available before action selection;
- \(A\) be the set of available actions;
- \(\pi(a \mid x(q))\) be the policy for selecting action \(a\);
- \(L_{\mathrm{quality}}\) represent task-specific answer loss;
- \(C(a)\) represent the measured cost of the action.

A simplified objective for investigation is:

\[
J(\pi) =
\mathbb{E}\left[
L_{\mathrm{quality}} + \lambda C(a)
\right]
\]

where \(\lambda\) controls the trade-off between answer quality and cost. This is a proposed experimental objective, not a guarantee that one scalar score captures every deployment requirement. Quality and cost must also be reported separately.

## 4. Aim

To design, implement, and scientifically evaluate an adaptive controller that selects suitable hallucination-mitigation actions for LLM queries with the goal of improving the trade-off between answer reliability and computational or operational cost.

## 5. Objectives

1. Study common causes and forms of LLM hallucination, including missing knowledge, reasoning errors, ambiguity, and unsupported answers.
2. Develop a modular controller that selects an action based on question characteristics and available evidence or uncertainty signals.
3. Establish a transparent rule-based baseline that can be run and tested locally.
4. Integrate an LLM for draft generation and structured experiment logging.
5. Implement suitable mitigation actions, including retrieval, deterministic calculation, clarification, verification/revision, and abstention.
6. Investigate uncertainty signals such as answer disagreement, model-reported confidence, semantic uncertainty, and evidence availability, while recognizing their limitations.
7. Develop a cost-aware policy that considers both expected response quality and action cost.
8. Build an evaluation set containing different task types and clearly defined annotation criteria.
9. Compare the adaptive policy with fixed baselines and relevant prior approaches.
10. Report quality, support, hallucination, abstention, latency, token, tool-call, and cost results transparently.
11. Conduct ablation studies and failure analysis to determine which components contribute to any observed improvement.
12. Document the system so that experiments can be reproduced.

## 6. Research Questions

The project will investigate the following questions:

- **RQ1:** Can an adaptive controller reduce unnecessary mitigation calls compared with a fixed retrieval or verification pipeline?
- **RQ2:** Can adaptive routing maintain or improve answer correctness and evidence support while reducing operational cost?
- **RQ3:** Which available signals are most useful for selecting between direct answering, retrieval, clarification, verification, and abstention?
- **RQ4:** How does the controller behave across factual QA, reasoning, ambiguity, and unanswerable questions?
- **RQ5:** What reliability–cost trade-offs arise when the cost weight or routing threshold changes?
- **RQ6:** Which action-selection errors occur most often, and how much do they affect end-to-end answer quality?

## 7. Research Hypothesis

The primary hypothesis is:

> An adaptive, cost-aware mitigation policy can achieve a more favorable reliability–cost trade-off than fixed mitigation strategies by applying additional support only when the available evidence and query signals indicate that it is likely to be useful.

Possible measurable outcomes include lower tool-call or token usage at comparable answer quality, or improved answer quality at comparable cost. The project must report negative or inconclusive results as well as positive results.

## 8. Scope

### 8.1 Included in scope

- Python-based prototype.
- Local development in Visual Studio Code.
- A rule-based baseline for action routing.
- Optional local LLM inference using compatible hardware, or an API-backed model adapter.
- Evidence retrieval from a controlled corpus or supported search system.
- Deterministic arithmetic for suitable calculation requests.
- Clarification and abstention actions.
- Verification and revision of answer claims using evidence where available.
- Structured logging and offline evaluation.
- Comparison with fixed and adaptive baselines.
- Scientific reporting of quality–cost trade-offs and limitations.

### 8.2 Outside the initial scope

- Guaranteeing that all hallucinations can be detected or eliminated.
- Claiming that model confidence is equivalent to factual correctness.
- Replacing expert review in safety-critical domains.
- Building a production-scale search engine or a universal truth-verification system.
- Claiming novelty or publication readiness before reviewing prior work and evaluating the method.
- Using the NVIDIA GPU in the initial rule-based phase; it becomes relevant if a compatible local model is used.

## 9. Key Concepts and Error Categories

The controller may use the following operational categories. They can overlap, and they should not be treated as perfectly observable causes.

### 9.1 Missing or insufficient knowledge

The answer requires information that is not available or reliable in the model's current context. Retrieval may help if relevant, trustworthy evidence exists.

### 9.2 Reasoning or computation error

The answer depends on a calculation, logical operation, or multi-step procedure that the model may perform incorrectly. A deterministic tool or independently checkable procedure may help.

### 9.3 Ambiguity or underspecification

The question permits multiple reasonable interpretations or omits information needed to answer well. Clarification may be preferable to guessing.

### 9.4 Unsupported or unanswerable question

The system lacks adequate evidence to support a reliable answer, or the question requests information that cannot be known from the available information. Abstention or an explicit limitation may be appropriate.

### 9.5 Uncertainty signals

Candidate signals include:

- model-reported confidence;
- token-level probabilities, when the model interface exposes them;
- disagreement among multiple generated answers;
- semantic entropy or related measures of semantic disagreement;
- retrieval availability and evidence quality;
- consistency with deterministic tools or external checks;
- features of the question, such as requests for current information or missing referents.

Each signal has limitations. Low disagreement does not prove correctness; high disagreement does not identify the cause of uncertainty by itself; model-reported confidence can be miscalibrated; and retrieval results may be incomplete or misleading. Signals should be validated against labeled evaluation data rather than assumed to be reliable.

## 10. Proposed System

### 10.1 High-level architecture

```text
                         User Question
                               |
                               v
                   Question Feature Extraction
                               |
                               v
                 Uncertainty / Evidence Assessment
                               |
                               v
                  Adaptive Decision Controller
                               |
             +-----------------+------------------+
             |          |          |             |
             v          v          v             v
          Direct     Calculate   Retrieve      Clarify
             |          |          |             |
             +----------+----------+-------------+
                               |
                               v
                    Optional Verification
                     and Answer Revision
                               |
                               v
                     Final Answer / Abstain
                               |
                               v
                 Structured Logs and Evaluation
```

This diagram represents the intended architecture. Not all branches are implemented in the first phase.

### 10.2 Controller inputs

Depending on the implementation stage, the controller may receive:

- question text;
- detected task or question features;
- candidate answer or draft, if available at that stage;
- uncertainty signals available before the action is selected;
- evidence availability or retrieval status;
- estimated action costs;
- policy thresholds or learned policy parameters.

The controller must not use the gold answer or a post-hoc correctness label as a feature at decision time. Those values are for evaluation only.

### 10.3 Candidate actions

| Action | Intended use | Main limitation or cost |
|---|---|---|
| `direct` | Answer without additional mitigation | Can produce unsupported or incorrect content |
| `calculate` | Use deterministic arithmetic for supported expressions | Only handles operations implemented by the tool |
| `retrieve` | Obtain evidence for knowledge-dependent or changing facts | Retrieval may be slow, costly, irrelevant, or incomplete |
| `clarify` | Ask for missing context or preferences | Requires another user interaction |
| `verify` | Check claims against available evidence or tools | Adds latency and may not be independent of the generator |
| `revise` | Correct a draft using identified issues and evidence | Revision can introduce new errors |
| `abstain` | Avoid presenting an unsupported answer as fact | Excessive abstention can make the system unhelpful |

### 10.4 Decision policy

The initial version uses explicit, hand-written rules. Future versions may use a threshold policy or a learned cost-sensitive decision model. A general form is:

\[
a^* = \arg\min_{a \in A}
\left(
\widehat{L}_{\mathrm{quality}}(a \mid x)
+ \lambda \widehat{C}(a \mid x)
\right)
\]

Here, estimated quality loss and action cost must be learned, calibrated, or measured using development data. This expression defines a possible design direction; it does not imply that the current prototype already estimates those values accurately.

## 11. Technology Stack

| Area | Proposed technology |
|---|---|
| Programming language | Python |
| Development environment | Visual Studio Code |
| Initial tests | Python `unittest` |
| Data handling | pandas and NumPy when needed |
| Visualizations | Matplotlib when needed |
| LLM inference | A compatible local model or API-backed model, selected in a later phase |
| Local compute | NVIDIA GPU if supported by the selected model and software stack |
| Retrieval | Controlled document corpus first; retrieval framework/vector index only if justified |
| Experiment storage | JSONL or CSV logs; structured database only if required |
| Evaluation | Task-specific metrics, evidence-support annotation, cost and latency measurements |
| Version control | Git |

The Phase 1 implementation uses only the Python standard library and does not require a GPU or external service.

## 12. Implementation Plan

### Phase 1 — Rule-based controller baseline

**Purpose:** Establish the software structure and a testable reference policy.

Deliverables:

- `controller.py`: question routing, restricted arithmetic evaluator, decision metadata, and command-line interface.
- `test_controller.py`: automated tests for supported routing rules and calculator safety.
- `README.md`: local setup and run instructions.
- `IMPLEMENTATION.md`: implementation roadmap.

The first rules use simple cues to route selected questions to direct response, calculation, retrieval, clarification, or abstention. The rules are deliberately transparent and limited. They are not expected to reliably identify the cause of every hallucination.

**Success criterion:** The program runs locally, returns a decision and explanation for each input, handles supported arithmetic safely, and passes its unit tests.

### Phase 2 — LLM integration

Add a model adapter that generates a draft response for applicable actions. Keep the adapter modular so that a local model can be replaced with an API-backed model without rewriting the controller.

Record the model identifier, prompt version, generation settings, latency, input/output tokens when available, errors, and selected action. Model self-reports of confidence must be treated as features to test, not as ground truth.

### Phase 3 — Mitigation tools

Implement the action handlers:

- **Retrieval:** fetch passages from a controlled collection or a supported search interface, preserve source metadata, and pass relevant evidence to the answer generator.
- **Calculation:** use deterministic functions for supported arithmetic rather than relying on a language model to compute exact results.
- **Clarification:** ask a targeted follow-up that resolves the missing information.
- **Verification:** split an answer into checkable claims and compare them with evidence or tools.
- **Revision:** update only claims identified as unsupported or incorrect, while checking the revised output again where feasible.
- **Abstention:** communicate the limitation when evidence is insufficient.

### Phase 4 — Uncertainty estimation and cost-aware routing

Replace or augment hand-written rules with one or more validated signals. Candidate approaches include answer disagreement, semantic uncertainty, model probability information, evidence availability, and calibrated classifiers.

Develop a policy that selects actions using development-set results and measured costs. Prevent test-set leakage: test labels must not influence routing thresholds, model selection, or policy training.

### Phase 5 — Evaluation and analysis

Run all baselines on the same held-out questions, store predictions and costs, calculate metrics, analyze routing mistakes, and perform ablations. Report results even if the adaptive policy does not outperform fixed baselines.

## 13. Dataset and Evaluation Design

### 13.1 Task coverage

The evaluation should include a mix of:

- factual question answering;
- arithmetic and reasoning;
- ambiguous or underspecified questions;
- questions with insufficient evidence or no reliable answer;
- current-information questions, if the evaluation environment can provide a timestamped and reproducible evidence source.

Potential dataset candidates include TriviaQA or PopQA for factual QA, GSM8K for arithmetic reasoning, AmbigQA for ambiguity-related questions, and carefully selected unanswerable examples from suitable datasets. Dataset use must be checked against the relevant license, split definitions, and task requirements.

Dataset categories are not perfect labels for why an answer fails. For example, a factual question may also require reasoning, and an ambiguity dataset may include questions answerable after clarification. The project should define annotation guidelines, allow multiple labels where justified, and inspect a sample manually.

### 13.2 Dataset splits

Maintain separate:

- **Training set:** for training a learned router, if one is used.
- **Development/validation set:** for tuning thresholds, selecting signals, and choosing policy parameters.
- **Test set:** for final evaluation only.

Do not tune the policy repeatedly on the test set. Avoid duplicate or near-duplicate questions across splits where possible.

### 13.3 Baselines

At minimum, compare:

1. **Direct-only:** answer without additional retrieval or verification.
2. **Always-retrieve:** retrieve evidence for each question, where retrieval is available.
3. **Rule-based baseline:** the Phase 1 controller.
4. **Adaptive policy:** the proposed uncertainty-aware, cost-sensitive controller.

Additional comparisons may include reasoning-oriented prompting, self-verification, and relevant published adaptive retrieval or self-reflection methods. Their original papers, task assumptions, and implementation details must be checked before describing or reproducing them. Comparisons should be fair and use the same evaluation examples wherever possible.

### 13.4 Quality and reliability metrics

Choose metrics according to the task rather than relying on one metric for all questions.

- Exact match or task-specific accuracy for suitable closed-answer tasks.
- Rubric-based correctness for free-form answers.
- Evidence support or claim-level attribution for answers requiring sources.
- Hallucination/unsupported-claim rate under an explicit annotation definition.
- Abstention rate and answer coverage.
- Precision and recall of abstention on questions defined as unanswerable.
- Clarification success rate, where a follow-up interaction is evaluated.
- Action-selection confusion matrix and per-category error rates.

Automated LLM judges may be used as supplementary evaluators, but should not be treated as infallible. Human review or independently checkable ground truth should be used for a representative subset where feasible.

### 13.5 Cost and efficiency metrics

- Input and output tokens.
- Number of model calls.
- Number of retrieval, verification, and other tool calls.
- End-to-end latency.
- Monetary/API cost, when it can be measured reliably.
- Retrieval latency and evidence volume.
- Cost per correct or sufficiently supported answer.
- Frequency of unnecessary tool use, based on a defined evaluation protocol.

Cost measurements depend on model, provider, hardware, caching, context size, and network conditions. Record these details with the results.

### 13.6 Statistical analysis

- Report the number of evaluated examples for each category.
- Use paired comparisons on the same questions.
- Report confidence intervals for important metrics.
- Include action-level errors and representative failure cases.
- Plot answer quality against cost and compare the Pareto frontier.
- Run ablations to test whether each signal or action contributes to performance.
- Report adverse trade-offs, not just aggregate improvements.

### 13.7 Example results table

The following is a blank reporting template; it contains no claimed results.

| Method | Correctness / support | Unsupported-claim rate | Abstention coverage | Tokens | Tool calls | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Direct-only | To measure | To measure | To measure | To measure | To measure | To measure |
| Always-retrieve | To measure | To measure | To measure | To measure | To measure | To measure |
| Rule-based baseline | To measure | To measure | To measure | To measure | To measure | To measure |
| Adaptive policy | To measure | To measure | To measure | To measure | To measure | To measure |

## 14. Data Logging and Reproducibility

Each evaluation record should include, where applicable:

- unique question ID;
- dataset name, split, and category labels;
- input question;
- available evidence and its source identifiers;
- selected action and decision reason;
- model identifier and prompt version;
- final response and cited evidence;
- reference answer or evaluation rubric;
- correctness and support annotations;
- input/output tokens and tool-call counts;
- latency and measured cost;
- exceptions, timeouts, and retries;
- software version, configuration, and random seed.

Keep raw outputs and evaluation annotations so the reported metrics can be regenerated. Log tool failures rather than silently dropping failed cases. Record the evaluation timestamp for sources that change over time.

## 15. Expected Outcomes

Expected deliverables include:

1. A modular, runnable prototype.
2. A rule-based baseline with unit tests.
3. LLM and mitigation-tool integrations in later phases.
4. A documented evaluation set and annotation procedure.
5. A comparison of fixed and adaptive strategies.
6. An experiment log and reproducible metric calculations.
7. An analysis of quality–cost trade-offs and failure cases.
8. A research report that describes the method, experimental design, results, limitations, and future work.

These are planned outputs, not claims that the proposed system will necessarily outperform existing methods.

## 16. Potential Research Contribution and Novelty

The intended contribution is to study and evaluate a controller that selects mitigation actions using uncertainty/evidence signals and explicit cost considerations. A potential contribution could be a better quality–cost trade-off, a more useful routing signal, a reproducible mixed-task evaluation, or an analysis showing when particular mitigation actions help.

However, adaptive retrieval, self-reflection, uncertainty estimation, verification, and cost-aware decision-making have existing research literature. Therefore, the general idea of adaptive mitigation alone should not be presented as novel. Before making a novelty claim, the project must:

1. Review and verify relevant published work.
2. Identify what existing methods observe, predict, and optimize.
3. State precisely how the proposed method differs.
4. Compare against strong, relevant baselines.
5. Demonstrate the difference empirically through controlled experiments and ablations.

If the results show no improvement, that is still useful evidence about the proposed policy and its assumptions.

## 17. Risks and Limitations

### Imperfect routing

Question keywords and simple features can misclassify intent. The controller may retrieve unnecessarily, fail to retrieve when needed, ask irrelevant clarifying questions, or abstain from answerable questions.

### Unreliable uncertainty estimates

Model confidence, token probabilities, and answer disagreement are imperfect proxies. A model can be consistently wrong, and multiple generations may share the same bias.

### Retrieval quality

Retrieved passages may be outdated, irrelevant, contradictory, or insufficient. Retrieval alone does not ensure that the final answer is grounded in the evidence.

### Verification limitations

A verifier may repeat the generator's errors or miss subtle unsupported claims. Independent evidence and task-appropriate checks are preferable where feasible.

### Metric limitations

Exact match does not capture all acceptable answers; automated judges can be biased; hallucination labels may be subjective. Use multiple metrics and clearly documented annotation criteria.

### Cost trade-offs

Lower token use does not always mean lower latency or lower total cost. Network calls, caching, model size, hardware, and verification overhead can change the outcome.

### Scope of conclusions

Results on selected datasets do not prove universal effectiveness across all models, languages, domains, or deployment settings. Conclusions should be limited to the tested setup.

## 18. Ethical and Responsible Research Considerations

- Do not present unsupported model outputs as verified facts.
- Preserve source information for retrieved claims.
- Avoid using private or sensitive data in evaluation unless appropriate permission and safeguards are in place.
- Review dataset licenses and usage terms.
- Document known failure modes and risks of overconfident answers.
- Use human review for high-impact applications.
- Report negative results and evaluation limitations transparently.
- Do not claim that the system eliminates hallucinations.

## 19. Project Milestones

| Milestone | Deliverable | Completion evidence |
|---|---|---|
| M1: Baseline | Rule-based controller | Program runs and decisions are inspectable |
| M2: Testing | Unit tests | Tests pass and unsafe calculator inputs are rejected |
| M3: LLM integration | Model adapter and draft generation | Reproducible sample runs and logs |
| M4: Mitigation actions | Retrieval, clarification, verification, abstention | Each action has tests and logs |
| M5: Evaluation pipeline | Dataset loader and metric scripts | Metrics can be regenerated from saved outputs |
| M6: Baseline comparison | Fixed vs adaptive results | Same test questions and documented configurations |
| M7: Ablation and analysis | Component-level evaluation | Contribution and failure analysis |
| M8: Research report | Full paper/report draft | Method, related work, experiments, results, and limitations documented |

## 20. Definition of Success

The project should be considered technically successful when it has a modular implementation, reliable tests, clear logs, and reproducible experiments.

The research hypothesis is supported only if the evaluation shows a meaningful and statistically credible improvement in the quality–cost trade-off against relevant baselines, without hiding regressions in particular task categories. The criterion for a meaningful improvement should be specified before examining the final test results.

## 21. Conclusion

This project investigates adaptive, cost-aware hallucination mitigation for LLMs. It treats mitigation as a decision problem: determine what kind of support a question needs, choose a suitable action, and measure both the answer outcome and the resources consumed.

The development process begins with an interpretable rule-based baseline, then progresses toward LLM integration, retrieval, verification, uncertainty estimation, and cost-sensitive policy evaluation. The value of the proposed approach will be determined by controlled experiments and comparison with prior work—not by the architecture alone.

---

## Appendix A — Suggested Repository Layout

```text
adaptive-hallucination-mitigation/
├── PROJECT_DESCRIPTION.md
├── IMPLEMENTATION.md
├── README.md
├── controller.py
├── test_controller.py
├── src/
│   ├── model_adapter.py
│   ├── retrieval.py
│   ├── verification.py
│   ├── policy.py
│   └── logging_utils.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── experiments/
│   ├── configs/
│   ├── run_evaluation.py
│   └── compare_baselines.py
├── results/
│   ├── predictions/
│   ├── metrics/
│   └── figures/
└── tests/
    ├── test_controller.py
    ├── test_calculator.py
    └── test_actions.py
```

This is a suggested future structure. Only the files actually implemented should be described as completed.

## Appendix B — Initial Prototype Usage

The initial prototype is a routing baseline, not a complete LLM application.

```bash
python controller.py
python -m unittest -v
```

It can demonstrate the current rule-based decisions and basic arithmetic handling. Retrieval, LLM generation, evidence-based verification, and the full scientific evaluation pipeline are future milestones.

## Appendix C — Important Reporting Rule

Never fill the results table with invented or expected performance numbers. Report measured results, exact experimental settings, uncertainty intervals, and limitations. A proposed design, a runnable baseline, and a validated research contribution are three different levels of progress; the report should clearly distinguish them.