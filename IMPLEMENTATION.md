# Implementation Plan — Adaptive, Cost-Aware Hallucination Mitigation

## 1. Purpose

This document describes how to implement the research prototype in stages. The system should select a suitable response strategy for each user question rather than always running the same expensive sequence of generation, retrieval, verification, and revision.

The central research question is:

> Can an adaptive controller maintain or improve answer reliability while reducing unnecessary retrieval, verification, token use, latency, and other inference costs?

This is a research hypothesis to test, not an established result.

## 2. Design principles

1. **Route before spending:** estimate what type of support a question needs before selecting an action.
2. **Use the right tool for the failure mode:** arithmetic should use deterministic computation; current factual claims should seek evidence; ambiguity should trigger clarification; unsupported claims should permit abstention.
3. **Measure actual costs:** initial cost values are placeholders. Replace them with measured latency, tokens, tool calls, and monetary costs where available.
4. **Keep decisions inspectable:** record the selected action and why it was selected.
5. **Evaluate end-to-end outcomes:** a good classifier is not enough; the controller must improve answer quality/cost trade-offs.
6. **Avoid overclaiming:** rules and model confidence are imperfect proxies for uncertainty, and a fluent answer is not proof of correctness.

## 3. Architecture

```text
User question
     |
     v
Question feature extraction
     |
     v
Decision policy / controller
     |
     +--> DIRECT ------> LLM draft (later phase)
     |
     +--> CALCULATE ---> restricted deterministic calculator
     |
     +--> RETRIEVE ----> search / document retrieval (later phase)
     |
     +--> CLARIFY -----> ask user for missing information
     |
     +--> VERIFY ------> check claims against evidence (later phase)
     |
     +--> ABSTAIN -----> explain that reliable support is unavailable
     |
     v
Final response + structured evaluation log
```

The Phase 1 version implements only transparent routing rules and basic arithmetic. Retrieval, LLM generation, and evidence-based verification are planned for later phases.

## 4. Phase 1 — Rule-based baseline

### Components

- `controller.py`: routing rules, restricted arithmetic evaluator, decision object, and command-line interface.
- `test_controller.py`: unit tests for routing and arithmetic safety.
- `README.md`: setup and execution instructions.

### Routing actions

| Action | Initial trigger | Purpose |
|---|---|---|
| `direct` | No rule fires | Allow a direct response in a future LLM-connected version |
| `calculate` | A supported numeric expression is detected | Use deterministic arithmetic |
| `retrieve` | Factual or time-sensitive question heuristic fires | Seek evidence before answering |
| `clarify` | Empty input or simple ambiguity cues | Request missing context |
| `abstain` | Selected unknowable future-prediction cues | Avoid presenting speculation as fact |

These triggers are deliberately simple heuristics. They are not a reliable general-purpose uncertainty detector. A real evaluation should include their false positives and false negatives.

### Run

Create and activate a virtual environment, then run:

```bash
python controller.py
```

Run unit tests:

```bash
python -m unittest -v
```

No third-party Python dependencies or GPU are needed for Phase 1.

## 5. Phase 2 — Add an LLM draft generator

Add a model adapter behind a small interface so the rest of the project does not depend on one provider. The adapter should return a draft and, where supported, token counts and latency.

Record at least:

- question ID and question text
- model/provider and model version
- selected action and decision reason
- draft answer
- input/output token counts
- elapsed time
- tool calls and tool errors

Do not treat the model's self-reported confidence as ground truth. It can be one feature among several, and should be calibrated against held-out data if used.

## 6. Phase 3 — Implement mitigation actions

### Retrieval

- Start with a small, controlled document collection or a benchmark-provided evidence corpus.
- Return source passages and metadata.
- Require the answer generator to distinguish supported claims from unsupported claims.
- Log retrieval latency, number of passages, and whether the final answer is supported by the retrieved evidence.

### Calculation / reasoning checks

- Use deterministic tools for calculations whenever possible.
- For more complex reasoning, compare a generated answer with an independently checkable result or verification procedure where one exists.
- Do not assume that asking the same model to reconsider its answer constitutes independent verification.

### Clarification

- Ask a specific question that resolves the missing variable or preference.
- Measure whether clarification improves answer quality and include the extra interaction cost.

### Verification

- Break answers into checkable claims.
- Check each claim against an evidence source or an appropriate evaluator.
- Track unsupported claims separately from overall answer correctness.

### Abstention

- Abstain when evidence is inadequate or the question is not answerable from the available information.
- Measure both unsafe answers avoided and answerable questions incorrectly refused.

## 7. Phase 4 — Cost-aware policy

A simple policy objective for initial experimentation is:

\[
J(\pi) = \mathbb{E}[L_{\text{answer}} + \lambda C_{\text{action}}]
\]

where:

- \(\pi\) is the routing policy
- \(L_{\text{answer}}\) is an answer-quality loss, including unsupported or incorrect claims
- \(C_{\text{action}}\) is the measured action cost
- \(\lambda\) controls the trade-off between reliability and cost

For a multi-metric study, report quality and cost separately as well as any combined objective. A single weighted score can hide important trade-offs.

Possible future policies:

1. Fixed rule baseline.
2. Threshold-based policy using uncertainty signals.
3. Learned action selector trained on development data.
4. Cost-sensitive contextual bandit or sequential decision policy, if the action/reward setup justifies it.

The controller must only use information available at decision time. Avoid using the gold answer or post-hoc correctness label as an input feature.

## 8. Phase 5 — Scientific evaluation

### Evaluation data

Use a mix of factual QA, arithmetic/reasoning, ambiguity, and unanswerable questions. Candidate sources include TriviaQA or PopQA, GSM8K, AmbigQA, and carefully selected unanswerable examples. Before use, inspect dataset licenses, split definitions, answer formats, and whether each example truly fits its intended category.

Do not assume dataset names are perfect labels for failure causes. Define an annotation guide, allow overlapping labels where justified, and manually inspect a sample. Keep training, development, and test data separate.

### Baselines to compare

- Direct generation only.
- Always retrieve.
- Always use a reasoning-oriented prompt/tool where applicable.
- The Phase 1 rule-based controller.
- Adaptive controller using the proposed uncertainty signals and cost-aware policy.
- Relevant published adaptive retrieval or self-verification methods, after checking their original papers and implementing or citing them fairly.

### Metrics

**Quality and reliability**
- Exact match / task-specific accuracy where appropriate.
- Human or rubric-based correctness for free-form answers.
- Evidence support / unsupported-claim rate.
- Hallucination rate using a clearly defined annotation protocol.
- Abstention precision, recall, and coverage.
- Clarification success rate where applicable.

**Cost and efficiency**
- Input and output tokens.
- Number of model calls and external tool calls.
- End-to-end latency.
- Monetary/API cost, if measurable.
- Retrieval or verification cost.

**Statistical reporting**
- Report sample counts and confidence intervals.
- Use paired comparisons on the same test questions.
- Include failure analysis and action-level confusion matrices.
- Plot quality against cost to show the Pareto frontier.
- Run ablations to test which signals and actions actually help.

### Example experiment table

| Method | Accuracy / support | Hallucination rate | Tokens | Tool calls | Latency |
|---|---:|---:|---:|---:|---:|
| Direct only | Measure | Measure | Measure | Measure | Measure |
| Always retrieve | Measure | Measure | Measure | Measure | Measure |
| Rule baseline | Measure | Measure | Measure | Measure | Measure |
| Adaptive policy | Measure | Measure | Measure | Measure | Measure |

Populate this table from experiment results; do not fill it with estimated or invented performance numbers.

## 9. Reproducibility checklist

- Pin dependency versions when dependencies are introduced.
- Record model name/version, prompt templates, decoding settings, and hardware.
- Fix random seeds where possible.
- Store raw predictions and structured logs.
- Preserve the exact dataset split and evaluation script.
- Record API failures and retries rather than silently dropping them.
- Keep test data out of policy tuning.
- Document annotation guidelines and adjudication procedures.
- Run each baseline on the same examples and under comparable settings.

## 10. Definition of done for Phase 1

- [ ] `controller.py` runs from the terminal.
- [ ] Arithmetic expressions use the restricted calculator rather than Python `eval`.
- [ ] Each decision includes an action, reason, and illustrative cost.
- [ ] Unit tests pass.
- [ ] The README explains limitations.
- [ ] No claim is made that the baseline reliably detects hallucinations or establishes research novelty.
