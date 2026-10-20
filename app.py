"""
Adaptive, Cost-Aware Hallucination Mitigation for LLMs
Research Prototype - Streamlit Web Dashboard (Phase 4 Calibration & Cost Policy)

Provides:
A. Interactive Query Evaluation & Main Dashboard
B. Action Decision & Rationale Inspector
C. Evidence & Claim Verification Panel
D. Cost, Latency & Token Accounting Panel
E. Multi-Policy Research Benchmark & Comparative Analysis
F. Statistical Uncertainty Estimation & Calibration Inspector (Phase 4)
G. Local Corpus Knowledge Explorer & Document Ingestion
H. Model Adapter, Prompt Versioning & Generation Settings
"""

from __future__ import annotations
import json
import os
import sys
import time
import pandas as pd
import streamlit as st

# Ensure repository root is on Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.signals import SignalExtractor, QuerySignals
from src.calculator import SafeCalculator
from src.retrieval import LocalCorpusRetriever, RetrievedPassage
from src.verification import EvidenceVerifier, VerificationStatus, ClaimStatus
from src.revision import AnswerReviser
from src.uncertainty import UncertaintyEstimator, DisagreementAnalysis
from src.calibration import CalibrationEngine, split_dev_and_test
from src.model_adapter import (
    BaseModelAdapter,
    ModelDraft,
    GenerationSettings,
    get_model_adapter,
)
from src.cost_tracker import CostTracker
from src.logging_service import LoggingService, get_logging_service
from src.policy import (
    PolicyType,
    AdaptiveRulePolicy,
    LearnedCostAwarePolicy,
)
from src.controller import MitigationController, ExecutionResult
from src.experiment_runner import ExperimentRunner


# Set Page Configuration (Crisp, clean, no emojis)
st.set_page_config(
    page_title="Adaptive Hallucination Mitigation | Research Prototype",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom White Minimalist Styling for Research Paper Demo Aesthetics
st.markdown(
    """
    <style>
    /* White Theme - Universal High-Contrast Typography & Surfaces */
    html, body, [class*="css"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .stApp {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    
    /* Headers & Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 700 !important;
        letter-spacing: -0.015em;
    }
    p, span, label, li {
        color: #334155 !important;
        line-height: 1.6;
    }
    strong, b {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    small, .caption, [data-testid="stCaptionContainer"] {
        color: #64748b !important;
    }
    
    .main-header {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a !important;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #475569 !important;
        margin-bottom: 1.35rem;
        line-height: 1.55;
    }

    /* Sidebar Surfaces & Text */
    [data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"] * {
        color: #1e293b !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] label {
        color: #334155 !important;
        font-weight: 500;
    }

    /* Input Fields & Textareas */
    .stTextInput input, .stTextArea textarea {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 1px #2563eb !important;
    }
    [data-baseweb="select"] * {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }

    /* Cards & Containers */
    .status-card {
        padding: 1.15rem 1.35rem;
        border-radius: 6px;
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
        font-size: 0.95rem;
        color: #0f172a !important;
        line-height: 1.6;
    }
    .status-card * {
        color: #0f172a !important;
    }
    .section-card {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
        padding: 1.1rem;
        margin-bottom: 1rem;
    }

    /* Metric Cards */
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    .metric-label-clean {
        font-size: 0.78rem;
        color: #64748b !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value-clean {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
    }
    [data-testid="stExpander"] summary {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    [data-testid="stExpander"] * {
        color: #1e293b !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #64748b !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        background-color: transparent !important;
        border-bottom: 2px solid transparent !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #0f172a !important;
        border-bottom-color: #2563eb !important;
    }

    /* DataFrames & Tables */
    [data-testid="stDataFrame"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
    }

    /* Alert Banners */
    .stAlert {
        border-radius: 6px !important;
    }
    .stAlert * {
        color: #0f172a !important;
    }

    /* Action Badges */
    .badge-action {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.82rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .action-direct { background-color: #eff6ff !important; color: #1d4ed8 !important; border: 1px solid #bfdbfe !important; }
    .action-calculate { background-color: #f0fdf4 !important; color: #15803d !important; border: 1px solid #bbf7d0 !important; }
    .action-retrieve { background-color: #fefce8 !important; color: #a16207 !important; border: 1px solid #fef08a !important; }
    .action-clarify { background-color: #faf5ff !important; color: #7e22ce !important; border: 1px solid #e9d5ff !important; }
    .action-abstain { background-color: #fef2f2 !important; color: #b91c1c !important; border: 1px solid #fecaca !important; }
    .action-verify { background-color: #f0fdfa !important; color: #0f766e !important; border: 1px solid #99f6e4 !important; }
    .action-revise { background-color: #fff7ed !important; color: #c2410c !important; border: 1px solid #fed7aa !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_base_resources():
    """Cache controller and runner."""
    retriever = LocalCorpusRetriever()
    logger = get_logging_service()
    return retriever, logger


def format_action_badge(action: str) -> str:
    act_lower = action.lower()
    css_class = f"action-{act_lower}"
    return f'<span class="badge-action {css_class}">{action.upper()}</span>'


def main():
    retriever, logger = get_base_resources()

    # Sidebar: Navigation & Configuration
    st.sidebar.title("Navigation & Policy")
    app_mode = st.sidebar.radio(
        "Select Workspace View:",
        [
            "Interactive Query Analyzer",
            "Multi-Policy Benchmark Suite",
            "Statistical Uncertainty & Calibration",
            "Local Knowledge Corpus",
            "Experiment Logs & Export",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Policy Configuration")
    tradeoff_lambda = st.sidebar.slider(
        "Cost Penalty Factor (lambda)",
        min_value=0.0,
        max_value=1.5,
        value=0.5,
        step=0.05,
        help="Controls the trade-off in: a* = argmax [Benefit(a) - lambda * Cost(a)]. Higher values penalize costly verification/retrieval actions.",
    )

    # Sidebar: Model & Generation Configuration
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model & Generation Settings")

    provider_choice = st.sidebar.selectbox(
        "LLM Provider Adapter:",
        ["offline", "openai", "gemini", "ollama"],
        index=0,
        format_func=lambda x: {
            "offline": "Local Offline ($0, No API Key)",
            "openai": "OpenAI (Remote API)",
            "gemini": "Google Gemini (Remote API)",
            "ollama": "Ollama (Local Open-Weights)",
        }[x],
        help="Select the generation backend. Local offline mode works without any API keys.",
    )

    default_models = {
        "offline": "offline-rule-demonstrator",
        "openai": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
        "ollama": "llama3.2:3b",
    }
    model_id_val = st.sidebar.text_input(
        "Model Identifier:",
        value=default_models.get(provider_choice, "offline-rule-demonstrator")
    )

    prompt_version = st.sidebar.selectbox(
        "Prompt Template Version:",
        ["v1.0-standard", "v2.0-concise", "v1.1-with-confidence-elicitation"],
        index=0,
        help="Versioned prompt templates for draft generation."
    )

    with st.sidebar.expander("Hyperparameters & Decoding", expanded=False):
        temperature_val = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        top_p_val = st.slider("Top-P", 0.1, 1.0, 0.95, 0.05)
        max_tokens_val = st.slider("Max Tokens", 50, 1500, 500, 50)
        elicit_conf_val = st.checkbox("Elicit Self-Reported Confidence", value=("confidence" in prompt_version))

    # Initialize model adapter with selected settings
    adapter = get_model_adapter(provider_name=provider_choice, model_id=model_id_val)
    gen_settings = GenerationSettings(
        temperature=temperature_val,
        top_p=top_p_val,
        max_tokens=max_tokens_val,
        prompt_version=prompt_version,
        elicit_confidence=elicit_conf_val,
    )

    controller = MitigationController(
        retriever=retriever,
        model_adapter=adapter,
        logger=logger,
        tradeoff_lambda=tradeoff_lambda,
        default_generation_settings=gen_settings,
    )
    runner = ExperimentRunner(controller=controller)

    # VIEW 1: Interactive Query Analyzer
    if app_mode == "Interactive Query Analyzer":
        st.markdown('<div class="main-header">Adaptive, Cost-Aware Hallucination Mitigation</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">'
            'A research prototype investigating selective hallucination mitigation: inspect query signals, '
            'estimate calibrated statistical uncertainty, adaptively route actions, and evaluate the quality-cost trade-off.'
            '</div>',
            unsafe_allow_html=True,
        )

        col_left, col_right = st.columns([1.8, 1.2])

        with col_left:
            st.markdown("### 1. Query Input")
            example_queries = [
                "-- Choose an illustrative example --",
                "Calculate (120 + 45) / 5",
                "What is 14 * 25?",
                "Which one is better?",
                "Is it good?",
                "What is the elevation of Mount Everest according to the 2020 survey?",
                "What is the exact speed of light in vacuum?",
                "When did Apollo 11 land on the Moon?",
                "Why did Albert Einstein win the Nobel Prize in Physics?",
                "Will Bitcoin hit $500,000 in 2035?",
                "What am I thinking right now?",
                "Explain the general concept of photosynthesis in plants.",
            ]
            selected_example = st.selectbox("Pre-configured Benchmark Queries:", example_queries)

            default_text = "" if selected_example == example_queries[0] else selected_example
            user_query = st.text_area(
                "Enter custom question or expression:",
                value=default_text,
                height=90,
                placeholder="Type arithmetic, factual question, ambiguous query, or future speculation...",
            )

            col_btn, col_pol = st.columns([1, 1.5])
            with col_btn:
                analyze_clicked = st.button("Analyze and Execute", type="primary", use_container_width=True)
            with col_pol:
                policy_choice = st.selectbox(
                    "Active Decision Policy:",
                    [
                        PolicyType.ADAPTIVE.value,
                        PolicyType.LEARNED_COST_AWARE.value,
                        PolicyType.DIRECT_ONLY.value,
                        PolicyType.ALWAYS_RETRIEVE.value,
                        PolicyType.FIXED_VERIFY.value,
                    ],
                )

        with col_right:
            st.markdown("### 2. Candidate Mitigation Actions")
            st.markdown(
                """
                - **DIRECT**: General answers without external tool overhead.
                - **CALCULATE**: Sandboxed deterministic AST evaluator (zero arithmetic hallucination).
                - **RETRIEVE**: Fetch authoritative passages from local corpus.
                - **CLARIFY**: Prompt for missing context/criteria on underspecified queries.
                - **VERIFY**: Claim decomposition and alignment against evidence.
                - **REVISE**: Automatic redaction/hedging of unsupported claims.
                - **ABSTAIN**: Explicit refusal for unknowable future/private inquiries.
                """
            )

        # Process Query
        if analyze_clicked or ("last_result" in st.session_state and not analyze_clicked):
            if analyze_clicked:
                if not user_query.strip():
                    st.warning("Query is empty. The controller will evaluate an empty query signal.")
                with st.spinner("Extracting signals, estimating uncertainty, and executing action..."):
                    result = controller.process_query(
                        query=user_query,
                        policy_name=policy_choice,
                        tradeoff_lambda=tradeoff_lambda,
                        generation_settings=gen_settings,
                    )
                    st.session_state["last_result"] = result
            else:
                result = st.session_state["last_result"]

            st.markdown("---")

            # TOP SUMMARY ROW
            c_act, c_unc, c_ver, c_cost = st.columns(4)
            with c_act:
                st.markdown('<div class="metric-label-clean">Selected Action</div>', unsafe_allow_html=True)
                st.markdown(format_action_badge(result.action_executed), unsafe_allow_html=True)
            with c_unc:
                st.markdown('<div class="metric-label-clean">Estimated Uncertainty</div>', unsafe_allow_html=True)
                if result.uncertainty_result:
                    unc_score = result.uncertainty_result.composite_uncertainty
                else:
                    unc_score = result.signals.heuristic_uncertainty_score
                st.markdown(f'<div class="metric-value-clean">{unc_score:.2f} <small style="font-size:0.8rem;color:#64748b;">(0.0-1.0)</small></div>', unsafe_allow_html=True)
            with c_ver:
                st.markdown('<div class="metric-label-clean">Verification Status</div>', unsafe_allow_html=True)
                ver_st = result.verification_result.overall_status.value if result.verification_result else "Not Checked"
                st.markdown(f'<div class="metric-value-clean">{ver_st}</div>', unsafe_allow_html=True)
            with c_cost:
                st.markdown('<div class="metric-label-clean">Operational Latency</div>', unsafe_allow_html=True)
                lat = result.cost.latency_ms if result.cost else 0.0
                st.markdown(f'<div class="metric-value-clean">{lat:.1f} ms</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # DETAIL PANELS
            tab_decision, tab_answer, tab_evidence, tab_cost = st.tabs([
                "Decision Rationale",
                "Final Answer and Output",
                "Evidence and Claims Alignment",
                "Cost and Resource Accounting",
            ])

            with tab_decision:
                st.subheader("Action Decision Inspector")
                st.info(f"**Controller Rationale:** {result.decision.decision_rationale}")
                st.markdown(f"**Next Step Executed:** {result.decision.next_step_description}")

                st.markdown("#### Observable Signal Breakdown")
                sig_col1, sig_col2, sig_col3 = st.columns(3)
                with sig_col1:
                    st.write(f"- **Arithmetic Detected:** `{result.signals.has_arithmetic_expr}`")
                    if result.signals.extracted_arithmetic_expr:
                        st.caption(f"Expression: `{result.signals.extracted_arithmetic_expr}`")
                    st.write(f"- **Is Ambiguous:** `{result.signals.is_ambiguous}`")
                    if result.signals.ambiguity_reason:
                        st.caption(f"Reason: {result.signals.ambiguity_reason}")
                with sig_col2:
                    st.write(f"- **Factual Knowledge Cue:** `{result.signals.is_factual_knowledge_request}`")
                    if result.signals.factual_keywords_detected:
                        st.caption(f"Triggers: `{', '.join(result.signals.factual_keywords_detected)}`")
                    st.write(f"- **Temporal Freshness Need:** `{result.signals.requires_temporal_freshness}`")
                with sig_col3:
                    st.write(f"- **Speculative / Unknowable:** `{result.signals.is_unknowable_or_speculative}`")
                    if result.signals.speculation_reason:
                        st.caption(f"Reason: {result.signals.speculation_reason}")
                    st.write(f"- **Token Length Approx:** `{result.signals.token_count_approx}`")

                if result.decision.candidate_utility_scores:
                    st.markdown("#### Action Net Utilities: U(a) = (1 - Loss(a)) - lambda * Cost(a)")
                    util_df = pd.DataFrame(
                        [
                            {
                                "Candidate Action": k.upper(),
                                "Prior Cost Units": CostTracker.get_action_cost_prior(k),
                                "Net Utility Score": v,
                            }
                            for k, v in result.decision.candidate_utility_scores.items()
                        ]
                    ).sort_values(by="Net Utility Score", ascending=False)
                    st.dataframe(util_df, use_container_width=True)

            with tab_answer:
                st.subheader("Response Output")
                st.markdown(f'<div class="status-card">{result.final_answer}</div>', unsafe_allow_html=True)

                # Interactive Multi-Turn Clarification Dialogue
                if result.action_executed == "clarify" or result.clarification_session:
                    st.info("Interactive Clarification Dialogue: The controller detected ambiguity or missing criteria.")
                    st.markdown("Please supply the missing criteria or context to complete this query turn:")
                    clarif_input = st.text_input("Clarification Input:", key="clarif_user_input", placeholder="e.g., Which language has better libraries for data science, Python or JavaScript?")
                    if st.button("Submit Clarification and Resolve", key="btn_clarif_submit"):
                        if clarif_input.strip():
                            with st.spinner("Resolving multi-turn clarification turn..."):
                                resolved_res = controller.resolve_clarification_turn(
                                    prior_result=result,
                                    clarification_input=clarif_input,
                                    policy_name=policy_choice,
                                    tradeoff_lambda=tradeoff_lambda,
                                    generation_settings=gen_settings,
                                )
                                st.session_state["last_result"] = resolved_res
                                st.rerun()
                        else:
                            st.warning("Please provide clarification details before submitting.")

                # Model Draft & Confidence Metadata Inspector
                if result.model_draft:
                    with st.expander("Model Draft and Generation Metadata", expanded=True):
                        d = result.model_draft
                        col_d1, col_d2, col_d3 = st.columns(3)
                        with col_d1:
                            st.write(f"**Model ID:** `{d.model_id}`")
                            st.write(f"**Provider:** `{d.provider}`")
                            st.write(f"**Prompt Version:** `{d.prompt_version}`")
                        with col_d2:
                            st.write(f"**Prompt Tokens:** `{d.prompt_tokens}`")
                            st.write(f"**Completion Tokens:** `{d.completion_tokens}`")
                            st.write(f"**Total Tokens:** `{d.total_tokens}`")
                        with col_d3:
                            st.write(f"**Temperature:** `{d.generation_settings.get('temperature', 0.2)}`")
                            st.write(f"**Max Tokens:** `{d.generation_settings.get('max_tokens', 500)}`")
                            st.write(f"**Latency:** `{d.latency_ms:.1f} ms`")

                        if d.model_reported_confidence is not None:
                            st.markdown(
                                f"**Model Self-Reported Confidence:** `{d.model_reported_confidence:.2f}` (scale 0.0 - 1.0)"
                            )
                            st.caption(
                                "Safety Notice: Model-reported confidence is treated strictly as an uncalibrated experimental feature to analyze, NOT as ground-truth correctness."
                            )

                        if d.error:
                            st.error(f"Model Error: {d.error}")

                if result.revision_result and result.revision_result.is_revised:
                    st.warning("Answer Revision Applied: Unsupported claims were identified and modified.")
                    with st.expander("Compare Original Draft vs Revised Output"):
                        c_orig, c_rev = st.columns(2)
                        with c_orig:
                            st.markdown("**Original Draft:**")
                            st.write(result.revision_result.original_answer)
                        with c_rev:
                            st.markdown("**Revised Answer:**")
                            st.write(result.revision_result.revised_answer)
                        st.markdown("**Modifications Log:**")
                        for m in result.revision_result.modifications_made:
                            st.markdown(f"- {m}")

            with tab_evidence:
                st.subheader("Evidence Retrieval and Verification Alignment")
                if result.action_executed in ("retrieve", "verify") and result.retrieval_result:
                    ret = result.retrieval_result
                    st.markdown(f"**Retrieval Status:** {ret.message}")

                    if ret.passages:
                        st.markdown("#### Retrieved Corpus Passages")
                        for idx, p in enumerate(ret.passages):
                            with st.expander(f"Passage {idx+1}: {p.doc_title} (Score: {p.relevance_score:.3f})", expanded=True):
                                st.write(p.text)
                                st.caption(f"Source Reference: {p.source_url_or_ref} | ID: `{p.passage_id}`")
                    else:
                        st.info("No matching passages exceeded the evidence relevance threshold.")

                    # Claim verification breakdown
                    if result.verification_result and result.verification_result.claims:
                        contradictions = [c for c in result.verification_result.claims if c.status == ClaimStatus.CONTRADICTED]
                        if contradictions:
                            st.error(f"Contradiction Detected ({len(contradictions)} claim(s)): Draft claims directly conflict with authoritative corpus evidence.")

                        st.markdown("#### Claim-Level Verification Analysis")
                        st.caption("Each propositional claim in the draft is checked for evidence support, verbatim quote attribution, and contradiction:")
                        claims_data = []
                        for c in result.verification_result.claims:
                            claims_data.append({
                                "Claim Text": c.claim_text,
                                "Status": c.status.value,
                                "Confidence": f"{c.confidence_score:.1%}",
                                "Supporting Reference": c.supporting_passage_id or "None",
                                "Verbatim Quote": f'"{c.verbatim_quote}"' if c.verbatim_quote else "N/A",
                                "Rationale": c.rationale,
                            })
                        st.dataframe(pd.DataFrame(claims_data), use_container_width=True)

                    # Post-revision re-verification display
                    if result.revision_result and result.revision_result.re_verification_result:
                        re_ver = result.revision_result.re_verification_result
                        st.markdown("#### Post-Revision Re-Verification (Iterative Loop)")
                        st.markdown(f"**Re-Verification Status:** `{re_ver.overall_status.value}`")
                        re_claims = []
                        for rc in re_ver.claims:
                            re_claims.append({
                                "Revised Claim Text": rc.claim_text,
                                "Status": rc.status.value,
                                "Confidence": f"{rc.confidence_score:.1%}",
                                "Supporting Reference": rc.supporting_passage_id or "None",
                                "Rationale": rc.rationale,
                            })
                        st.dataframe(pd.DataFrame(re_claims), use_container_width=True)
                else:
                    st.info(
                        f"Action was {result.action_executed.upper()}. External evidence retrieval was not indicated for this query."
                    )

            with tab_cost:
                st.subheader("Operational Resource and Cost Accounting")
                st.caption("Distinguishing prior illustrative decision units from actual measured computational resources.")

                col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                with col_m1:
                    st.metric("Prior Cost Units", f"{result.cost.estimated_action_cost_units:.2f}")
                with col_m2:
                    st.metric("Total Tokens", f"{result.cost.total_tokens}")
                with col_m3:
                    st.metric("Model Calls", f"{result.cost.model_calls_count}")
                with col_m4:
                    st.metric("Tool Invocations", f"{result.cost.tool_calls_count}")
                with col_m5:
                    st.metric("Monetary USD", f"${result.cost.monetary_cost_usd:.6f}")

                st.markdown(
                    """
                    > **Accounting Protocol:**
                    > - **Prior Cost Units:** Dimensionless relative weight used in action routing (e.g. `calculate=0.2`, `retrieve=3.5`, `verify=5.0`).
                    > - **Measured Monetary Cost:** Exact API cost computed per token when live paid APIs are configured ($0.0 for local offline mode).
                    """
                )

    # VIEW 2: Multi-Policy Benchmark Suite
    elif app_mode == "Multi-Policy Benchmark Suite":
        st.markdown('<div class="main-header">Multi-Policy Research Benchmark Suite</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">'
            'Systematically compare the Adaptive Controller against fixed baselines on an identical, held-out evaluation set.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            r"""
            This evaluation tests whether selective, cost-aware action routing improves the reliability-cost trade-off 
            compared to uniform fixed pipelines:
            1. **Direct-Only**: Always answers directly without external tools.
            2. **Always-Retrieve**: Executes evidence retrieval for every query.
            3. **Fixed-Verify**: Retrieves evidence and checks claims for every query.
            4. **Adaptive Policy**: Selects actions based on observable uncertainty cues and cost penalty lambda.
            """
        )

        st.subheader("Benchmark Execution")
        st.write(f"Benchmark Dataset contains **{len(runner.dataset)}** curated questions across Arithmetic, Factual QA, Ambiguity, and Unanswerables.")

        col_run, col_bar = st.columns([1, 3])
        with col_run:
            start_benchmark = st.button("Run Full Benchmark Evaluation", type="primary")

        if start_benchmark or "benchmark_results" in st.session_state:
            if start_benchmark:
                prog_bar = st.progress(0.0)
                status_txt = st.empty()

                def update_progress(curr, total, msg):
                    prog_bar.progress(curr / total)
                    status_txt.text(f"[{curr}/{total}] {msg}")

                res = runner.run_benchmark(tradeoff_lambda=tradeoff_lambda, progress_callback=update_progress)
                st.session_state["benchmark_results"] = res
                status_txt.text("Benchmark completed successfully.")
            else:
                res = st.session_state["benchmark_results"]

            st.markdown("### Comparative Performance Summary")
            st.dataframe(res["summary_df"], use_container_width=True)

            st.markdown("### Detailed Item-Level Predictions")
            st.dataframe(res["detailed_df"], use_container_width=True)

            # Download results
            csv_data = res["detailed_df"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Detailed Benchmark Results (CSV)",
                data=csv_data,
                file_name="benchmark_results.csv",
                mime="text/csv",
            )

    # VIEW 3: Statistical Uncertainty & Calibration (Phase 4)
    elif app_mode == "Statistical Uncertainty & Calibration":
        st.markdown('<div class="main-header">Statistical Uncertainty Estimation and Calibration (Phase 4)</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">'
            'Evaluate multi-sample semantic disagreement, Expected Calibration Error (ECE), Brier score, '
            'and verify strict separation between development and test data to prevent data leakage.'
            '</div>',
            unsafe_allow_html=True,
        )

        tab_disagree, tab_calib, tab_splits = st.tabs([
            "Multi-Sample Semantic Disagreement",
            "Calibration Metrics and Reliability Diagram",
            "Dataset Split and Leakage Prevention",
        ])

        with tab_disagree:
            st.subheader("Multi-Sample Semantic Disagreement Analyzer")
            st.markdown(
                "Sample multiple stochastic candidate responses at elevated temperature to measure epistemic disagreement:"
            )
            sample_1 = st.text_input("Candidate Sample A:", value="Mount Everest's peak elevation is 8,848.86 meters above sea level.")
            sample_2 = st.text_input("Candidate Sample B:", value="According to the 2020 survey, Mount Everest stands at 8,848.86 meters.")
            sample_3 = st.text_input("Candidate Sample C:", value="Mount Everest has an official measured height of 8,848 meters.")

            if st.button("Compute Multi-Sample Disagreement", type="primary"):
                samples = [s for s in [sample_1, sample_2, sample_3] if s.strip()]
                analysis = UncertaintyEstimator.compute_multi_sample_disagreement(samples)

                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    st.metric("Mean Jaccard Distance", f"{analysis.mean_disagreement:.3f}")
                with col_d2:
                    st.metric("Lexical Entropy", f"{analysis.lexical_entropy:.3f}")
                with col_d3:
                    st.metric("Disagreement Level", analysis.disagreement_level)

                st.markdown("#### Pairwise Distance Breakdown")
                st.write(f"- Number of Samples Compared: `{analysis.num_samples}`")
                for idx, dist in enumerate(analysis.pairwise_jaccard_distances):
                    st.write(f"- Pair {idx+1} Jaccard Distance: `{dist:.3f}`")

        with tab_calib:
            st.subheader("Expected Calibration Error (ECE) and Temperature Scaling")
            st.markdown(
                "Evaluates how well predicted model confidence probabilities match empirical accuracy."
            )

            cal_engine = CalibrationEngine(num_bins=5)
            # Simulated representative development trace
            sample_confs = [0.95, 0.92, 0.88, 0.75, 0.72, 0.65, 0.58, 0.45, 0.35, 0.20]
            sample_labels = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]

            cal_metrics = cal_engine.compute_ece_and_brier(sample_confs, sample_labels)

            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                st.metric("Expected Calibration Error (ECE)", f"{cal_metrics.expected_calibration_error:.3f}")
            with col_c2:
                st.metric("Maximum Calibration Error (MCE)", f"{cal_metrics.maximum_calibration_error:.3f}")
            with col_c3:
                st.metric("Brier Score", f"{cal_metrics.brier_score:.3f}")

            st.markdown("#### Reliability Diagram Bin Breakdown")
            bins_data = [b.to_dict() for b in cal_metrics.bins]
            st.dataframe(pd.DataFrame(bins_data), use_container_width=True)

            st.markdown("#### Interactive Post-Hoc Temperature Scaling")
            t_val = st.slider("Temperature Scaling Factor (T):", 0.2, 3.0, 1.25, 0.05)
            test_prob = st.slider("Raw Test Probability:", 0.01, 0.99, 0.85, 0.01)
            cal_engine.temperature = t_val
            calibrated_p = cal_engine.calibrate_probability(test_prob)
            st.write(f"Raw Probability: `{test_prob:.3f}` -> **Calibrated Probability:** `{calibrated_p:.3f}`")

        with tab_splits:
            st.subheader("Dataset Split and Leakage Prevention")
            st.markdown(
                """
                **Protocol:**
                Section 12 and 13 of the project specification mandate strict separation between:
                - **Development/Validation Split:** For tuning calibration temperature, ambiguity thresholds, and cost weights.
                - **Held-out Test Split:** For final evaluation only. Zero overlap is enforced to eliminate test-set leakage.
                """
            )
            raw_dataset = runner.dataset
            dev_data, test_data = split_dev_and_test(raw_dataset, dev_ratio=0.5, random_seed=42)

            st.write(f"Total Benchmark Set: **{len(raw_dataset)}** queries | Dev Set: **{len(dev_data)}** | Test Set: **{len(test_data)}**")

            dev_ids = {d["id"] for d in dev_data}
            test_ids = {t["id"] for t in test_data}
            overlap = dev_ids.intersection(test_ids)

            if len(overlap) == 0:
                st.success("Verification Passed: Zero overlap detected between Development and Test splits (No Data Leakage).")
            else:
                st.error(f"Leakage Alert: {len(overlap)} overlapping item(s) found!")

            c_d, c_t = st.columns(2)
            with c_d:
                st.markdown("**Development Split (Tuning Only):**")
                st.dataframe(pd.DataFrame(dev_data)[["id", "category", "query", "expected_action"]], use_container_width=True)
            with c_t:
                st.markdown("**Held-Out Test Split (Evaluation Only):**")
                st.dataframe(pd.DataFrame(test_data)[["id", "category", "query", "expected_action"]], use_container_width=True)

    # VIEW 4: Local Knowledge Corpus
    elif app_mode == "Local Knowledge Corpus":
        st.markdown('<div class="main-header">Local Evidence Knowledge Corpus</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Inspect verified factual documents or add new passages to expand retrieval memory.</div>',
            unsafe_allow_html=True,
        )

        st.write(f"Total Indexed Documents: **{len(retriever.documents)}** | Total Passages: **{len(retriever.passages)}**")

        tab_list, tab_add = st.tabs(["View Existing Documents", "Add Document"])

        with tab_list:
            for d in retriever.documents.values():
                with st.expander(f"{d.title} (ID: {d.doc_id})"):
                    st.write(d.text)
                    st.caption(f"Authoritative Source: {d.source}")

        with tab_add:
            st.subheader("Add New Document to Local Corpus")
            new_id = st.text_input("Document ID:", value=f"doc_custom_{len(retriever.documents)+1}")
            new_title = st.text_input("Document Title:", placeholder="e.g. James Webb Space Telescope Discoveries")
            new_source = st.text_input("Authoritative Source Reference:", placeholder="e.g. NASA Goddard Press Release 2023")
            new_text = st.text_area("Passage Text:", height=150, placeholder="Enter factual content...")

            if st.button("Save and Index Document", type="primary"):
                if new_id and new_title and new_text:
                    retriever.add_document(
                        doc_id=new_id.strip(),
                        title=new_title.strip(),
                        source=new_source.strip() or "User Contributed",
                        text=new_text.strip(),
                        save=True,
                    )
                    st.success(f"Document '{new_title}' successfully indexed into local evidence memory.")
                    st.rerun()
                else:
                    st.error("Please fill in Document ID, Title, and Passage Text.")

    # VIEW 5: Experiment Logs & Export
    elif app_mode == "Experiment Logs & Export":
        st.markdown('<div class="main-header">Experiment Logging and Reproducibility</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Inspect historical run records, export SQLite logs to CSV or JSON, and inspect trace details.</div>',
            unsafe_allow_html=True,
        )

        recent_runs = logger.get_recent_runs(limit=100)
        st.write(f"Total Persisted Runs in SQLite: **{len(recent_runs)}**")

        if recent_runs:
            df_runs = pd.DataFrame(recent_runs)
            disp_cols = [
                "timestamp", "query", "policy_name", "selected_action", "model_name",
                "prompt_version", "model_reported_confidence", "heuristic_uncertainty",
                "verification_status", "latency_ms", "total_tokens", "monetary_cost_usd"
            ]
            available_cols = [c for c in disp_cols if c in df_runs.columns]
            st.dataframe(df_runs[available_cols], use_container_width=True)

            col_exp_csv, col_exp_json = st.columns(2)
            with col_exp_csv:
                csv_bytes = df_runs.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Export Run Records as CSV",
                    data=csv_bytes,
                    file_name="experiment_runs.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_exp_json:
                json_bytes = df_runs.to_json(orient="records", indent=2).encode("utf-8")
                st.download_button(
                    "Export Run Records as JSON",
                    data=json_bytes,
                    file_name="experiment_runs.json",
                    mime="application/json",
                    use_container_width=True,
                )
        else:
            st.info("No runs logged yet. Execute queries in the Interactive Analyzer or Benchmark Suite to generate persistent records.")


if __name__ == "__main__":
    main()
