"""
Adaptive, Cost-Aware Hallucination Mitigation for LLMs
Research Prototype - Streamlit Web Dashboard

Provides:
A. Interactive Query Evaluation & Main Dashboard
B. Action Decision & Rationale Inspector
C. Evidence & Claim Verification Panel
D. Cost, Latency & Token Accounting Panel
E. Multi-Policy Research Benchmark & Comparative Analysis
F. Local Corpus Knowledge Explorer & Document Ingestion
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
from src.llm_provider import get_llm_provider, BaseLLMProvider
from src.cost_tracker import CostTracker
from src.logging_service import LoggingService, get_logging_service
from src.policy import PolicyType, AdaptiveRulePolicy
from src.controller import MitigationController, ExecutionResult
from src.experiment_runner import ExperimentRunner


# Set Page Configuration
st.set_page_config(
    page_title="Adaptive Hallucination Mitigation | Research Prototype",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Research Aesthetics
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .badge-action {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .action-direct { background-color: #e0f2fe; color: #0369a1; border: 1px solid #7dd3fc; }
    .action-calculate { background-color: #ecfdf5; color: #047857; border: 1px solid #6ee7b7; }
    .action-retrieve { background-color: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
    .action-clarify { background-color: #f3e8ff; color: #7e22ce; border: 1px solid #d8b4fe; }
    .action-abstain { background-color: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
    .action-verify { background-color: #ccfbf1; color: #0f766e; border: 1px solid #5eead4; }
    .status-card {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    .metric-label-clean {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value-clean {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_controller():
    """Cache controller and local resources."""
    return MitigationController()


@st.cache_resource
def get_runner():
    """Cache experiment runner."""
    ctrl = get_controller()
    return ExperimentRunner(controller=ctrl)


def format_action_badge(action: str) -> str:
    act_lower = action.lower()
    css_class = f"action-{act_lower}"
    return f'<span class="badge-action {css_class}">{action.upper()}</span>'


def main():
    controller = get_controller()
    runner = get_runner()
    logger = controller.logger

    # Sidebar: Navigation & Configuration
    st.sidebar.title("Navigation & Policy")
    app_mode = st.sidebar.radio(
        "Select Workspace View:",
        [
            "🔬 Interactive Query Analyzer",
            "📊 Multi-Policy Benchmark Suite",
            "📚 Local Knowledge Corpus",
            "📜 Experiment Logs & Export",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Policy Configuration")
    tradeoff_lambda = st.sidebar.slider(
        "Cost Penalty Factor (λ)",
        min_value=0.0,
        max_value=1.5,
        value=0.5,
        step=0.05,
        help="Controls the trade-off in: a* = argmax [Benefit(a) - λ * Cost(a)]. Higher values penalize costly verification/retrieval actions.",
    )

    provider_info = type(controller.llm_provider).__name__
    is_mock = getattr(controller.llm_provider, "is_mock", True)
    st.sidebar.info(
        f"**Active LLM Adapter:** `{provider_info}`\n\n"
        f"**Mode:** {'Local Demonstration (Offline, $0)' if is_mock else 'Live Remote API'}\n\n"
        f"Set `OPENAI_API_KEY` or `GEMINI_API_KEY` in `.env` for remote model calls."
    )

    # VIEW 1: Interactive Query Analyzer
    if app_mode == "🔬 Interactive Query Analyzer":
        st.markdown('<div class="main-header">Adaptive, Cost-Aware Hallucination Mitigation</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">'
            'A research prototype investigating selective hallucination mitigation: inspect query signals, '
            'adaptively select actions, and evaluate the quality–cost trade-off.'
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
                analyze_clicked = st.button("🚀 Analyze & Execute", type="primary", use_container_width=True)
            with col_pol:
                policy_choice = st.selectbox(
                    "Active Decision Policy:",
                    [
                        PolicyType.ADAPTIVE.value,
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
                with st.spinner("Extracting signals and evaluating cost-aware policy..."):
                    result = controller.process_query(
                        query=user_query,
                        policy_name=policy_choice,
                        tradeoff_lambda=tradeoff_lambda,
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
                st.markdown('<div class="metric-label-clean">Heuristic Uncertainty</div>', unsafe_allow_html=True)
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

            # DETAIL PANELS: Tabs for clarity
            tab_decision, tab_answer, tab_evidence, tab_cost = st.tabs([
                "🧠 Decision Rationale",
                "💬 Final Answer & Output",
                "📑 Evidence & Claims Alignment",
                "📈 Cost & Resource Accounting",
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
                    st.markdown("#### Cost-Aware Action Utilities: $U(a) = \\text{Benefit}(a) - \\lambda \\cdot \\text{Cost}(a)$")
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

                if result.revision_result and result.revision_result.is_revised:
                    st.warning("⚠️ **Answer Revision Applied:** Unsupported claims were identified and modified.")
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
                st.subheader("Evidence Retrieval & Verification Alignment")
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
                        st.markdown("#### Claim-Level Verification Analysis")
                        st.caption("Each propositional claim in the draft is checked for evidence support:")
                        claims_data = []
                        for c in result.verification_result.claims:
                            claims_data.append({
                                "Claim Text": c.claim_text,
                                "Status": c.status.value,
                                "Confidence": f"{c.confidence_score:.1%}",
                                "Supporting Reference": c.supporting_passage_id or "None",
                                "Rationale": c.rationale,
                            })
                        st.dataframe(pd.DataFrame(claims_data), use_container_width=True)
                else:
                    st.info(
                        f"Action was `{result.action_executed.upper()}`. External evidence retrieval was not indicated for this query."
                    )

            with tab_cost:
                st.subheader("Operational Resource & Cost Accounting")
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
    elif app_mode == "📊 Multi-Policy Benchmark Suite":
        st.markdown('<div class="main-header">Multi-Policy Research Benchmark Suite</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">'
            'Systematically compare the Adaptive Controller against fixed baselines on an identical, held-out evaluation set.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            r"""
            This evaluation tests whether selective, cost-aware action routing improves the reliability–cost trade-off 
            compared to uniform fixed pipelines:
            1. **Direct-Only**: Always answers directly without external tools.
            2. **Always-Retrieve**: Executes evidence retrieval for every query.
            3. **Fixed-Verify**: Retrieves evidence and checks claims for every query.
            4. **Adaptive Policy**: Selects actions based on observable uncertainty cues and cost penalty $\lambda$.
            """
        )

        st.subheader("Benchmark Execution")
        st.write(f"Benchmark Dataset contains **{len(runner.dataset)}** curated questions across Arithmetic, Factual QA, Ambiguity, and Unanswerables.")

        col_run, col_bar = st.columns([1, 3])
        with col_run:
            start_benchmark = st.button("▶️ Run Full Benchmark Evaluation", type="primary")

        if start_benchmark or "benchmark_results" in st.session_state:
            if start_benchmark:
                prog_bar = st.progress(0.0)
                status_txt = st.empty()

                def update_progress(curr, total, msg):
                    prog_bar.progress(curr / total)
                    status_txt.text(f"[{curr}/{total}] {msg}")

                res = runner.run_benchmark(tradeoff_lambda=tradeoff_lambda, progress_callback=update_progress)
                st.session_state["benchmark_results"] = res
                status_txt.text("✅ Benchmark completed successfully!")
            else:
                res = st.session_state["benchmark_results"]

            st.markdown("### Comparative Performance Summary")
            st.dataframe(res["summary_df"], use_container_width=True)

            st.markdown("### Detailed Item-Level Predictions")
            st.dataframe(res["detailed_df"], use_container_width=True)

            # Download results
            csv_data = res["detailed_df"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Detailed Benchmark Results (CSV)",
                data=csv_data,
                file_name="benchmark_results.csv",
                mime="text/csv",
            )

    # VIEW 3: Local Knowledge Corpus
    elif app_mode == "📚 Local Knowledge Corpus":
        st.markdown('<div class="main-header">Local Evidence Knowledge Corpus</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Inspect verified factual documents or add new passages to expand retrieval memory.</div>',
            unsafe_allow_html=True,
        )

        retriever = controller.retriever
        st.write(f"Total Indexed Documents: **{len(retriever.documents)}** | Total Passages: **{len(retriever.passages)}**")

        tab_list, tab_add = st.tabs(["📖 View Existing Documents", "➕ Add Document"])

        with tab_list:
            for d in retriever.documents.values():
                with st.expander(f"📄 {d.title} (ID: {d.doc_id})"):
                    st.write(d.text)
                    st.caption(f"Authoritative Source: {d.source}")

        with tab_add:
            st.subheader("Add New Document to Local Corpus")
            new_id = st.text_input("Document ID:", value=f"doc_custom_{len(retriever.documents)+1}")
            new_title = st.text_input("Document Title:", placeholder="e.g. James Webb Space Telescope Discoveries")
            new_source = st.text_input("Authoritative Source Reference:", placeholder="e.g. NASA Goddard Press Release 2023")
            new_text = st.text_area("Passage Text:", height=150, placeholder="Enter factual content...")

            if st.button("Save & Index Document", type="primary"):
                if new_id and new_title and new_text:
                    retriever.add_document(
                        doc_id=new_id.strip(),
                        title=new_title.strip(),
                        source=new_source.strip() or "User Contributed",
                        text=new_text.strip(),
                        save=True,
                    )
                    st.success(f"Document '{new_title}' successfully indexed into local evidence memory!")
                    st.rerun()
                else:
                    st.error("Please fill in Document ID, Title, and Passage Text.")

    # VIEW 4: Experiment Logs & Export
    elif app_mode == "📜 Experiment Logs & Export":
        st.markdown('<div class="main-header">Experiment Logging & Reproducibility</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Inspect historical run records, export SQLite logs to CSV or JSON, and inspect trace details.</div>',
            unsafe_allow_html=True,
        )

        recent_runs = logger.get_recent_runs(limit=100)
        st.write(f"Total Persisted Runs in SQLite: **{len(recent_runs)}**")

        if recent_runs:
            df_runs = pd.DataFrame(recent_runs)
            # Display clean view
            disp_cols = [
                "timestamp", "query", "policy_name", "selected_action",
                "heuristic_uncertainty", "verification_status", "latency_ms", "total_tokens", "monetary_cost_usd"
            ]
            available_cols = [c for c in disp_cols if c in df_runs.columns]
            st.dataframe(df_runs[available_cols], use_container_width=True)

            col_exp_csv, col_exp_json = st.columns(2)
            with col_exp_csv:
                csv_bytes = df_runs.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export Run Records as CSV",
                    data=csv_bytes,
                    file_name="experiment_runs.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_exp_json:
                json_bytes = df_runs.to_json(orient="records", indent=2).encode("utf-8")
                st.download_button(
                    "📥 Export Run Records as JSON",
                    data=json_bytes,
                    file_name="experiment_runs.json",
                    mime="application/json",
                    use_container_width=True,
                )
        else:
            st.info("No runs logged yet. Execute queries in the Interactive Analyzer or Benchmark Suite to generate persistent records.")


if __name__ == "__main__":
    main()
