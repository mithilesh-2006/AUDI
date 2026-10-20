"""
Adaptive Hallucination Mitigation Controller.

Orchestrates signal extraction, cost-aware action selection, action execution,
evidence verification, revision, cost accounting, and experiment logging.
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
from src.llm_provider import BaseLLMProvider, LLMResponse, get_llm_provider
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
    
    # Sub-component execution results
    calculation_result: Optional[CalculationResult] = None
    retrieval_result: Optional[RetrievalResult] = None
    verification_result: Optional[VerificationResult] = None
    revision_result: Optional[RevisionResult] = None
    llm_response: Optional[LLMResponse] = None
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
            "llm_response": self.llm_response.to_dict() if self.llm_response else None,
            "cost": self.cost.to_dict() if self.cost else None,
        }


class MitigationController:
    """Main routing and execution controller."""

    def __init__(
        self,
        retriever: Optional[LocalCorpusRetriever] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        logger: Optional[LoggingService] = None,
        tradeoff_lambda: float = 0.5,
    ):
        self.retriever = retriever or LocalCorpusRetriever()
        self.llm_provider = llm_provider or get_llm_provider()
        self.logger = logger or get_logging_service()
        self.tradeoff_lambda = tradeoff_lambda

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
    ) -> ExecutionResult:
        """Process a query through the adaptive mitigation controller."""
        start_time_perf = time.perf_counter()
        run_id = f"run_{uuid.uuid4().hex[:10]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        lambda_val = tradeoff_lambda if tradeoff_lambda is not None else self.tradeoff_lambda
        selected_policy_name = policy_name or PolicyType.ADAPTIVE.value
        policy = self.policies.get(selected_policy_name, self.policies[PolicyType.ADAPTIVE.value])

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
        is_billed = not getattr(self.llm_provider, "is_mock", True)

        calc_res: Optional[CalculationResult] = None
        ret_res: Optional[RetrievalResult] = None
        ver_res: Optional[VerificationResult] = None
        rev_res: Optional[RevisionResult] = None
        llm_resp: Optional[LLMResponse] = None
        final_answer = ""

        # Step 3: Action Execution
        if action == "calculate":
            tool_calls += 1
            expr = signals.extracted_arithmetic_expr or query
            calc_res = SafeCalculator.evaluate(expr)
            if calc_res.success:
                final_answer = f"Calculation Result: {calc_res.expression} = {calc_res.formatted_result}"
            else:
                final_answer = f"Calculator could not compute expression: {calc_res.error}"
            # No verification performed for deterministic calculator
            ver_res = VerificationResult(
                overall_status=VerificationStatus.NOT_CHECKED,
                summary_notes="Calculator output is deterministic; verification pipeline bypassed."
            )

        elif action == "clarify":
            # Formulate structured clarification question
            if signals.is_empty:
                final_answer = "Your query was empty. Please provide the specific question or topic you would like addressed."
            elif signals.ambiguity_reason:
                final_answer = (
                    f"To provide an accurate answer, could you please clarify? "
                    f"Issue detected: {signals.ambiguity_reason}. "
                    "For example, if comparing options, please specify which candidates and evaluation criteria you are interested in."
                )
            else:
                final_answer = "The question appears ambiguous. Could you please provide more context or specify your requirements?"
            
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

            # Generate grounded response using evidence
            model_calls += 1
            evidence_context = ""
            if ret_res.passages and ret_res.has_sufficient_evidence:
                evidence_context = "\n".join([f"[{p.doc_title}]: {p.text}" for p in ret_res.passages])
                prompt = (
                    f"Answer the following question using strictly the retrieved evidence below:\n\n"
                    f"Evidence:\n{evidence_context}\n\n"
                    f"Question: {query}\n\nAnswer:"
                )
            else:
                prompt = f"Question: {query}\n\nAnswer:"

            llm_resp = self.llm_provider.generate(prompt)
            prompt_tokens += llm_resp.prompt_tokens
            completion_tokens += llm_resp.completion_tokens
            
            if ret_res.has_sufficient_evidence:
                final_answer = llm_resp.text
                # Verification is NOT claimed simply because retrieval occurred
                ver_res = VerificationResult(
                    overall_status=VerificationStatus.NOT_CHECKED,
                    summary_notes="Evidence retrieved and provided to generator. Verification step was not executed (Retrieval != Verification)."
                )
            else:
                final_answer = (
                    f"{llm_resp.text}\n\n"
                    f"[Notice: Local evidence corpus did not contain sufficient authoritative passages for this query.]"
                )
                ver_res = VerificationResult(
                    overall_status=VerificationStatus.NOT_CHECKED,
                    summary_notes="No sufficient evidence found; verification not checked."
                )

        elif action == "verify":
            # Full mitigation pipeline: Retrieve -> Generate -> Verify claims against evidence -> Revise if needed
            tool_calls += 1  # retrieval
            ret_res = self.retriever.retrieve(query, top_k=3)

            # Generate draft
            model_calls += 1
            if ret_res.has_sufficient_evidence:
                evidence_context = "\n".join([f"[{p.doc_title}]: {p.text}" for p in ret_res.passages])
                prompt = f"Evidence:\n{evidence_context}\n\nQuestion: {query}\n\nAnswer:"
            else:
                prompt = f"Question: {query}\n\nAnswer:"

            llm_resp = self.llm_provider.generate(prompt)
            prompt_tokens += llm_resp.prompt_tokens
            completion_tokens += llm_resp.completion_tokens
            draft_answer = llm_resp.text

            # Run claim-level verification
            tool_calls += 1  # verification tool
            ver_res = EvidenceVerifier.verify(draft_answer, ret_res.passages)

            # Run revision if partially supported or unsupported
            if ver_res.overall_status in (VerificationStatus.PARTIALLY_SUPPORTED, VerificationStatus.UNSUPPORTED):
                rev_res = AnswerReviser.revise(draft_answer, ver_res, ret_res.passages)
                final_answer = rev_res.revised_answer
            else:
                final_answer = draft_answer

        else:  # "direct"
            model_calls += 1
            llm_resp = self.llm_provider.generate(f"Question: {query}\n\nAnswer:")
            prompt_tokens += llm_resp.prompt_tokens
            completion_tokens += llm_resp.completion_tokens
            final_answer = llm_resp.text
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
            notes={"policy": selected_policy_name, "lambda": lambda_val}
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
            llm_response=llm_resp,
            cost=cost_record,
        )

        # Step 5: Persistent Logging
        evidence_refs = [f"{p.doc_title} ({p.source_url_or_ref})" for p in (ret_res.passages if ret_res else [])]
        ver_status_str = ver_res.overall_status.value if ver_res else VerificationStatus.NOT_CHECKED.value

        # Evaluation correctness check (if reference answer supplied)
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
            model_name=llm_resp.model_name if llm_resp else "local-rule",
            is_mock_model=getattr(llm_resp, "is_mock", True) if llm_resp else True,
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
        )
        self.logger.log_run(log_record)

        return result
