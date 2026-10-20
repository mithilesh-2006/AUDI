"""
Adaptive Hallucination Mitigation Controller (Phase 3 Mitigation Tools).

Orchestrates:
1. Signal extraction and uncertainty assessment.
2. Cost-aware action routing.
3. Safe deterministic AST calculation.
4. Lexical BM25 evidence retrieval with threshold gating.
5. Interactive multi-turn clarification sessions.
6. Propositional claim verification with numerical and entity contradiction detection.
7. Iterative answer revision and re-verification loops.
8. Insufficient-evidence abstention.
9. Cost accounting and structured logging.
"""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from src.signals import SignalExtractor, QuerySignals
from src.policy import (
    BasePolicy,
    PolicyType,
    DecisionOutcome,
    AdaptiveRulePolicy,
    DirectOnlyPolicy,
    AlwaysRetrievePolicy,
    FixedVerifyPolicy,
)
from src.calculator import SafeCalculator, CalculationResult
from src.retrieval import LocalCorpusRetriever, RetrievalResult, RetrievedPassage
from src.verification import EvidenceVerifier, VerificationResult, VerificationStatus
from src.revision import AnswerReviser, RevisionResult
from src.clarification import ClarificationHandler, ClarificationSession
from src.model_adapter import (
    BaseModelAdapter,
    ModelDraft,
    GenerationSettings,
    get_model_adapter,
)
from src.cost_tracker import CostTracker, CostAccounting
from src.logging_service import LoggingService, RunRecord, get_logging_service


@dataclass
class ExecutionResult:
    """Complete inspectable output of a processed query."""
    run_id: str
    query: str
    policy_name: str
    signals: QuerySignals
    decision: DecisionOutcome
    final_answer: str
    action_executed: str
    
    # Sub-component execution results (Phase 3)
    calculation_result: Optional[CalculationResult] = None
    retrieval_result: Optional[RetrievalResult] = None
    verification_result: Optional[VerificationResult] = None
    revision_result: Optional[RevisionResult] = None
    clarification_session: Optional[ClarificationSession] = None
    model_draft: Optional[ModelDraft] = None
    cost: Optional[CostAccounting] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "query": self.query,
            "policy_name": self.policy_name,
            "signals": self.signals.to_dict(),
            "decision": self.decision.to_dict(),
            "final_answer": self.final_answer,
            "action_executed": self.action_executed,
            "calculation_result": self.calculation_result.to_dict() if self.calculation_result else None,
            "retrieval_result": self.retrieval_result.to_dict() if self.retrieval_result else None,
            "verification_result": self.verification_result.to_dict() if self.verification_result else None,
            "revision_result": self.revision_result.to_dict() if self.revision_result else None,
            "clarification_session": self.clarification_session.to_dict() if self.clarification_session else None,
            "model_draft": self.model_draft.to_dict() if self.model_draft else None,
            "cost": self.cost.to_dict() if self.cost else None,
        }


class MitigationController:
    """Main routing and execution controller with Phase 3 Mitigation Tools."""

    def __init__(
        self,
        retriever: Optional[LocalCorpusRetriever] = None,
        model_adapter: Optional[BaseModelAdapter] = None,
        logger: Optional[LoggingService] = None,
        tradeoff_lambda: float = 0.5,
        default_generation_settings: Optional[GenerationSettings] = None,
    ):
        self.retriever = retriever or LocalCorpusRetriever()
        self.model_adapter = model_adapter or get_model_adapter()
        self.logger = logger or get_logging_service()
        self.tradeoff_lambda = tradeoff_lambda
        self.default_generation_settings = default_generation_settings or GenerationSettings()

        # Policy registry
        self.policies: Dict[str, BasePolicy] = {
            PolicyType.ADAPTIVE.value: AdaptiveRulePolicy(default_lambda=tradeoff_lambda),
            PolicyType.DIRECT_ONLY.value: DirectOnlyPolicy(),
            PolicyType.ALWAYS_RETRIEVE.value: AlwaysRetrievePolicy(),
            PolicyType.FIXED_VERIFY.value: FixedVerifyPolicy(),
        }

    def process_query(
        self,
        query: str,
        policy_name: Optional[str] = None,
        tradeoff_lambda: Optional[float] = None,
        reference_answer: Optional[str] = None,
        benchmark_category: Optional[str] = None,
        generation_settings: Optional[GenerationSettings] = None,
    ) -> ExecutionResult:
        """Process a query through the adaptive mitigation controller."""
        start_time_perf = time.perf_counter()
        run_id = f"run_{uuid.uuid4().hex[:10]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        lambda_val = tradeoff_lambda if tradeoff_lambda is not None else self.tradeoff_lambda
        selected_policy_name = policy_name or PolicyType.ADAPTIVE.value
        policy = self.policies.get(selected_policy_name, self.policies[PolicyType.ADAPTIVE.value])
        gen_cfg = generation_settings or self.default_generation_settings

        # Step 1: Feature & Uncertainty Signal Extraction
        signals = SignalExtractor.extract(query)

        # Step 2: Policy Action Selection
        decision = policy.select_action(signals, tradeoff_lambda=lambda_val)
        action = decision.selected_action

        # Resource counters
        prompt_tokens = 0
        completion_tokens = 0
        model_calls = 0
        tool_calls = 0
        is_billed = not getattr(self.model_adapter, "is_mock", True)
        tool_errors: Optional[str] = None

        calc_res: Optional[CalculationResult] = None
        ret_res: Optional[RetrievalResult] = None
        ver_res: Optional[VerificationResult] = None
        rev_res: Optional[RevisionResult] = None
        clarify_session: Optional[ClarificationSession] = None
        draft: Optional[ModelDraft] = None
        final_answer = ""

        # Step 3: Action Execution (Phase 3 Handlers)
        if action == "calculate":
            tool_calls += 1
            expr = signals.extracted_arithmetic_expr or query
            calc_res = SafeCalculator.evaluate(expr)
            if calc_res.success:
                final_answer = f"Calculation Result: {calc_res.expression} = {calc_res.formatted_result}"
            else:
                final_answer = f"Calculator could not compute expression: {calc_res.error}"
                tool_errors = calc_res.error
            ver_res = VerificationResult(
                overall_status=VerificationStatus.NOT_CHECKED,
                summary_notes="Calculator output is deterministic; verification pipeline bypassed."
            )

        elif action == "clarify":
            # Initiate structured interactive clarification session
            clarify_session = ClarificationHandler.initiate_clarification(signals)
            final_answer = clarify_session.clarification_prompt
            ver_res = VerificationResult(
                overall_status=VerificationStatus.NOT_CHECKED,
                summary_notes="Clarification requested; answer verification not applicable."
            )

        elif action == "abstain":
            final_answer = (
                f"Abstention Notice: The system cannot reliably answer this query. "
                f"Reason: {signals.speculation_reason or 'This inquiry requires predicting unknowable future states or private information'}. "
                "To prevent generating speculative or hallucinated claims, the system abstains from answering."
            )
            ver_res = VerificationResult(
                overall_status=VerificationStatus.NOT_CHECKED,
                summary_notes="System abstained; no factual claims to verify."
            )

        elif action == "retrieve":
            tool_calls += 1
            ret_res = self.retriever.retrieve(query, top_k=3)

            # Evidence Gating
            if not ret_res.has_sufficient_evidence and selected_policy_name == PolicyType.ADAPTIVE.value:
                # Insufficient-evidence abstention
                final_answer = (
                    "Abstention Notice: The local evidence corpus does not contain sufficient authoritative passages "
                    "to ground an answer for this inquiry. To avoid generating ungrounded assertions, the system abstains "
                    "from answering. (Note: Lack of local evidence does not imply the statement is universally false)."
                )
                ver_res = VerificationResult(
                    overall_status=VerificationStatus.NOT_CHECKED,
                    summary_notes="Insufficient local evidence; abstention executed."
                )
            else:
                model_calls += 1
                evidence_context = ""
                if ret_res.passages and ret_res.has_sufficient_evidence:
                    evidence_context = "\n".join([f"[{p.doc_title}]: {p.text}" for p in ret_res.passages])

                draft = self.model_adapter.generate_draft(
                    query=query,
                    task_type="rag" if evidence_context else "direct",
                    evidence=evidence_context,
                    settings=gen_cfg
                )
                prompt_tokens += draft.prompt_tokens
                completion_tokens += draft.completion_tokens
                if draft.error:
                    tool_errors = draft.error

                if ret_res.has_sufficient_evidence:
                    final_answer = draft.text
                    ver_res = VerificationResult(
                        overall_status=VerificationStatus.NOT_CHECKED,
                        summary_notes="Evidence retrieved and provided to generator. Verification step was not executed (Retrieval != Verification)."
                    )
                else:
                    final_answer = (
                        f"{draft.text}\n\n"
                        f"[Notice: Local evidence corpus did not contain sufficient authoritative passages for this query.]"
                    )
                    ver_res = VerificationResult(
                        overall_status=VerificationStatus.NOT_CHECKED,
                        summary_notes="No sufficient evidence found; verification not checked."
                    )

        elif action == "verify":
            tool_calls += 1  # retrieval tool
            ret_res = self.retriever.retrieve(query, top_k=3)

            # Evidence Gating for Verification
            if not ret_res.has_sufficient_evidence:
                final_answer = (
                    "Abstention Notice: Cannot execute evidence verification because no relevant passages were found "
                    "in the local corpus. The system abstains from verifying or asserting ungrounded facts."
                )
                ver_res = VerificationResult(
                    overall_status=VerificationStatus.UNSUPPORTED,
                    summary_notes="No grounding passages available; verification failed."
                )
            else:
                model_calls += 1
                evidence_context = "\n".join([f"[{p.doc_title}]: {p.text}" for p in ret_res.passages])

                draft = self.model_adapter.generate_draft(
                    query=query,
                    task_type="rag",
                    evidence=evidence_context,
                    settings=gen_cfg
                )
                prompt_tokens += draft.prompt_tokens
                completion_tokens += draft.completion_tokens
                if draft.error:
                    tool_errors = draft.error
                draft_answer = draft.text

                # Claim-level verification with contradiction detection
                tool_calls += 1  # verification tool
                ver_res = EvidenceVerifier.verify(draft_answer, ret_res.passages)

                # Iterative revision & re-verification loop if unsupported or contradicted
                if ver_res.overall_status in (VerificationStatus.PARTIALLY_SUPPORTED, VerificationStatus.UNSUPPORTED):
                    tool_calls += 1  # revision tool
                    rev_res = AnswerReviser.revise(draft_answer, ver_res, ret_res.passages)
                    final_answer = rev_res.revised_answer
                else:
                    final_answer = draft_answer

        else:  # "direct"
            model_calls += 1
            draft = self.model_adapter.generate_draft(
                query=query,
                task_type="direct",
                settings=gen_cfg
            )
            prompt_tokens += draft.prompt_tokens
            completion_tokens += draft.completion_tokens
            if draft.error:
                tool_errors = draft.error
            final_answer = draft.text
            ver_res = VerificationResult(
                overall_status=VerificationStatus.NOT_CHECKED,
                summary_notes="Direct generation selected; verification pipeline not invoked."
            )

        # Step 4: Cost Accounting
        cost_record = CostTracker.create_record(
            action_name=action,
            start_time_perf=start_time_perf,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_calls=model_calls,
            tool_calls=tool_calls,
            is_billed=is_billed,
            notes={
                "policy": selected_policy_name,
                "lambda": lambda_val,
                "model_id": draft.model_id if draft else "local",
                "prompt_version": draft.prompt_version if draft else "v1.0-standard"
            }
        )

        # Build execution result
        result = ExecutionResult(
            run_id=run_id,
            query=query,
            policy_name=selected_policy_name,
            signals=signals,
            decision=decision,
            final_answer=final_answer,
            action_executed=action,
            calculation_result=calc_res,
            retrieval_result=ret_res,
            verification_result=ver_res,
            revision_result=rev_res,
            clarification_session=clarify_session,
            model_draft=draft,
            cost=cost_record,
        )

        # Step 5: Persistent Logging
        evidence_refs = [f"{p.doc_title} ({p.source_url_or_ref})" for p in (ret_res.passages if ret_res else [])]
        ver_status_str = ver_res.overall_status.value if ver_res else VerificationStatus.NOT_CHECKED.value

        is_corr: Optional[bool] = None
        if reference_answer:
            ref_clean = reference_answer.lower().strip()
            ans_clean = final_answer.lower().strip()
            is_corr = ref_clean in ans_clean

        log_record = RunRecord(
            run_id=run_id,
            timestamp=timestamp,
            query=query,
            policy_name=selected_policy_name,
            selected_action=action,
            decision_rationale=decision.decision_rationale,
            answer_or_result=final_answer,
            heuristic_uncertainty=signals.heuristic_uncertainty_score,
            observable_signals=signals.signal_summary,
            evidence_references=evidence_refs,
            verification_status=ver_status_str,
            supported_claims_count=ver_res.supported_claims_count if ver_res else 0,
            unsupported_claims_count=ver_res.unsupported_claims_count if ver_res else 0,
            model_name=draft.model_id if draft else "local-rule",
            is_mock_model=getattr(draft, "is_mock", True) if draft else True,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model_calls_count=model_calls,
            tool_calls_count=tool_calls,
            latency_ms=cost_record.latency_ms,
            estimated_cost_units=cost_record.estimated_action_cost_units,
            measured_monetary_usd=cost_record.monetary_cost_usd,
            reference_answer=reference_answer,
            is_correct=is_corr,
            benchmark_category=benchmark_category,
            prompt_version=draft.prompt_version if draft else "v1.0-standard",
            generation_settings=draft.generation_settings if draft else {},
            model_reported_confidence=draft.model_reported_confidence if draft else None,
            tool_errors=tool_errors,
        )
        self.logger.log_run(log_record)

        return result

    def resolve_clarification_turn(
        self,
        session: ClarificationSession,
        user_response: str,
        policy_name: Optional[str] = None,
        tradeoff_lambda: Optional[float] = None,
    ) -> ExecutionResult:
        """Resolve a clarification dialogue turn with user feedback."""
        resolved_session = ClarificationHandler.resolve_clarification(session, user_response)
        
        # Execute the disambiguated query
        result = self.process_query(
            query=resolved_session.disambiguated_query or user_response,
            policy_name=policy_name,
            tradeoff_lambda=tradeoff_lambda,
        )
        result.clarification_session = resolved_session
        return result
