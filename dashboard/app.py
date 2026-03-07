"""
ORACLE — Streamlit Dashboard
Multi-page dashboard for viral evolution monitoring, predictions, and vaccine design.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.agents.orchestrator import OracleOrchestrator
from src.ingestion.sequence_ingestion import LINEAGE_DEFINITIONS
from src.mlops.pipeline import RetrainingPipeline

# ─────────────── Page Config ───────────────

st.set_page_config(
    page_title="ORACLE — Viral Evolution Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────── Custom CSS ───────────────

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #00d2ff, #7b2ff7, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }

    .main-header p {
        font-size: 1rem;
        opacity: 0.75;
        margin: 0.5rem 0 0;
        font-weight: 300;
    }

    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border-radius: 14px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }

    .metric-card .value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-card .label {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.6);
        margin-top: 0.3rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .risk-critical { color: #ff4444; font-weight: 700; }
    .risk-high { color: #ff8800; font-weight: 700; }
    .risk-moderate { color: #ffcc00; font-weight: 600; }
    .risk-low { color: #44ff44; font-weight: 600; }

    .alert-box {
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }

    .alert-high {
        background: rgba(255, 68, 68, 0.15);
        border-left: 4px solid #ff4444;
        color: #ff6666;
    }

    .alert-warning {
        background: rgba(255, 204, 0, 0.12);
        border-left: 4px solid #ffcc00;
        color: #ffdd44;
    }

    .alert-info {
        background: rgba(0, 210, 255, 0.1);
        border-left: 4px solid #00d2ff;
        color: #44ddff;
    }

    .agent-status {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .agent-completed {
        background: rgba(68, 255, 68, 0.15);
        color: #44ff44;
        border: 1px solid rgba(68, 255, 68, 0.3);
    }

    .agent-running {
        background: rgba(0, 210, 255, 0.15);
        color: #00d2ff;
        border: 1px solid rgba(0, 210, 255, 0.3);
    }

    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e0e0e0;
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(123, 47, 247, 0.3);
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────── Session State ───────────────

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False
if "mlops_pipeline" not in st.session_state:
    st.session_state.mlops_pipeline = RetrainingPipeline()


# ─────────────── Header ───────────────

st.markdown(
    """
<div class="main-header">
    <h1>🧬 ORACLE — Predictive Viral Evolution Engine</h1>
    <p>Real-time viral genomic surveillance • AI-powered mutation forecasting • Autonomous vaccine design</p>
</div>
""",
    unsafe_allow_html=True,
)


# ─────────────── Sidebar ───────────────

with st.sidebar:
    st.markdown("### ⚙️ Pipeline Controls")

    num_sequences = st.slider("Sequences to Analyze", 50, 500, 200, step=50)

    if st.button(
        "🚀 Run ORACLE Pipeline",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.pipeline_running,
    ):
        st.session_state.pipeline_running = True
        with st.spinner("Running pipeline... This may take a moment."):
            orchestrator = OracleOrchestrator()
            result = orchestrator.run_pipeline(num_sequences=num_sequences)
            st.session_state.pipeline_result = result
            st.session_state.pipeline_running = False
        st.rerun()

    if st.session_state.pipeline_result:
        status = st.session_state.pipeline_result.get("status", "unknown")
        if status == "completed":
            st.success(f"✅ Pipeline completed")
            st.caption(
                f"Duration: {st.session_state.pipeline_result.get('total_duration_seconds', 0):.1f}s"
            )
        else:
            st.error(f"❌ Pipeline failed")

    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    if (
        st.session_state.pipeline_result
        and st.session_state.pipeline_result["status"] == "completed"
    ):
        metrics = st.session_state.pipeline_result.get("metrics", {})
        st.metric("Sequences", metrics.get("sequences_processed", 0))
        st.metric("Mutations", metrics.get("mutations_detected", 0))
        st.metric("Predictions", metrics.get("predictions_generated", 0))
        st.metric("Vaccines", metrics.get("vaccines_designed", 0))
    else:
        st.caption("Run the pipeline to see stats")

    st.markdown("---")
    st.markdown("### 🔗 MCP Servers")
    servers = ["Genomic", "Protein", "Epidemiology", "Phylogenetics"]
    for s in servers:
        st.markdown(f"🟢 **{s}** Server")


# ─────────────── Main Content Tabs ───────────────

tabs = st.tabs(
    [
        "🏠 Overview",
        "📡 Variant Tracker",
        "🔮 Mutation Forecast",
        "🌳 Phylogenetics",
        "🛡️ Immune Escape",
        "💉 Vaccine Candidates",
        "🤖 Agent Monitor",
        "📈 MLOps",
    ]
)


# ─────────────── Tab 1: Overview ───────────────

with tabs[0]:
    if (
        not st.session_state.pipeline_result
        or st.session_state.pipeline_result["status"] != "completed"
    ):
        st.markdown("### 👋 Welcome to ORACLE")
        st.markdown("""
        ORACLE is an **AI-powered viral evolution engine** that:

        1. **🔍 Surveils** incoming viral genomic sequences for novel mutations
        2. **🔮 Predicts** future mutations using evolutionary transformer models
        3. **🛡️ Evaluates** immune escape potential across 4 antibody classes
        4. **💉 Designs** updated vaccine candidates using multi-strategy optimization
        5. **📋 Reports** WHO-style intelligence briefings

        **Click "Run ORACLE Pipeline"** in the sidebar to start.
        """)

        # Show lineage info cards
        st.markdown("### 📊 Known Variant Database")
        cols = st.columns(4)
        for i, (lineage, data) in enumerate(LINEAGE_DEFINITIONS.items()):
            with cols[i % 4]:
                risk_color = (
                    "#ff4444"
                    if data.get("escape", 0) > 0.6
                    else "#ffcc00" if data.get("escape", 0) > 0.3 else "#44ff44"
                )
                st.markdown(
                    f"""
                <div class="metric-card">
                    <div class="value" style="font-size:1.2rem;">{data.get('who_label', lineage)}</div>
                    <div class="label">{lineage}</div>
                    <div style="margin-top:8px; font-size:0.8rem; color:{risk_color}">
                        Escape: {data.get('escape', 0):.0%} | Fitness: {data.get('fitness', 1.0):.1f}x
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
    else:
        result = st.session_state.pipeline_result
        metrics = result.get("metrics", {})
        all_results = result.get("all_results", {})

        # Metric cards
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(
                f"""<div class="metric-card">
                <div class="value">{metrics.get('sequences_processed', 0)}</div>
                <div class="label">Sequences</div>
            </div>""",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""<div class="metric-card">
                <div class="value">{metrics.get('mutations_detected', 0)}</div>
                <div class="label">Unique Mutations</div>
            </div>""",
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""<div class="metric-card">
                <div class="value">{metrics.get('predictions_generated', 0)}</div>
                <div class="label">Predictions</div>
            </div>""",
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"""<div class="metric-card">
                <div class="value">{metrics.get('vaccines_designed', 0)}</div>
                <div class="label">Vaccine Candidates</div>
            </div>""",
                unsafe_allow_html=True,
            )
        with col5:
            st.markdown(
                f"""<div class="metric-card">
                <div class="value">{result.get('total_duration_seconds', 0):.1f}s</div>
                <div class="label">Pipeline Duration</div>
            </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Executive Summary
        report = all_results.get("report", {})
        exec_summary = report.get(
            "executive_summary", "Run the pipeline to generate a summary."
        )
        st.markdown("### 📋 Executive Summary")
        st.info(exec_summary)

        # Risk Assessment
        risk = report.get("risk_assessment", "")
        if risk:
            st.markdown("### 🚨 Risk Assessment")
            if "CRITICAL" in risk:
                st.error(risk)
            elif "HIGH" in risk:
                st.warning(risk)
            else:
                st.success(risk)

        # Alerts
        surveillance = all_results.get("surveillance", {})
        alerts = surveillance.get("alerts", [])
        if alerts:
            st.markdown("### ⚠️ Active Alerts")
            for alert in alerts:
                level = alert.get("level", "INFO")
                css_class = (
                    "alert-high"
                    if level == "HIGH"
                    else "alert-warning" if level == "WARNING" else "alert-info"
                )
                icon = "🔴" if level == "HIGH" else "🟡" if level == "WARNING" else "🔵"
                st.markdown(
                    f"""<div class="alert-box {css_class}">
                    {icon} <strong>[{alert.get('type', '')}]</strong> {alert.get('message', '')}
                </div>""",
                    unsafe_allow_html=True,
                )


# ─────────────── Tab 2: Variant Tracker ───────────────

with tabs[1]:
    st.markdown(
        '<div class="section-header">📡 Real-Time Variant Tracker</div>',
        unsafe_allow_html=True,
    )

    if (
        st.session_state.pipeline_result
        and st.session_state.pipeline_result["status"] == "completed"
    ):
        all_results = st.session_state.pipeline_result.get("all_results", {})
        surveillance = all_results.get("surveillance", {})

        # Lineage distribution chart
        lineage_dist = surveillance.get("lineage_distribution", {})
        if lineage_dist:
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=list(lineage_dist.keys()),
                            y=list(lineage_dist.values()),
                            marker=dict(
                                color=list(lineage_dist.values()),
                                colorscale="Viridis",
                            ),
                            text=list(lineage_dist.values()),
                            textposition="outside",
                        )
                    ]
                )
                fig.update_layout(
                    title="Lineage Distribution",
                    xaxis_title="Lineage",
                    yaxis_title="Sequence Count",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig_pie = go.Figure(
                    data=[
                        go.Pie(
                            labels=list(lineage_dist.keys()),
                            values=list(lineage_dist.values()),
                            hole=0.45,
                            marker=dict(colors=px.colors.qualitative.Set3),
                        )
                    ]
                )
                fig_pie.update_layout(
                    title="Prevalence",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=400,
                    showlegend=True,
                    legend=dict(font=dict(size=10)),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        # Top mutations table
        top_muts = surveillance.get("top_mutations", [])
        if top_muts:
            st.markdown("### Top Mutations")
            df = pd.DataFrame(top_muts, columns=["Mutation", "Count"])
            df["Frequency %"] = (df["Count"] / df["Count"].sum() * 100).round(2)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Region distribution
        region_dist = surveillance.get("region_distribution", {})
        if region_dist:
            st.markdown("### Mutation Distribution by Spike Region")
            fig_region = go.Figure(
                data=[
                    go.Bar(
                        x=list(region_dist.keys()),
                        y=list(region_dist.values()),
                        marker_color=[
                            "#ff6b6b",
                            "#ffd93d",
                            "#6bcb77",
                            "#4d96ff",
                            "#845ec2",
                            "#ff9671",
                            "#d65db1",
                        ],
                    )
                ]
            )
            fig_region.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
            )
            st.plotly_chart(fig_region, use_container_width=True)
    else:
        st.info("Run the ORACLE pipeline to see variant tracking data.")


# ─────────────── Tab 3: Mutation Forecast ───────────────

with tabs[2]:
    st.markdown(
        '<div class="section-header">🔮 Mutation Forecast</div>', unsafe_allow_html=True
    )

    if (
        st.session_state.pipeline_result
        and st.session_state.pipeline_result["status"] == "completed"
    ):
        all_results = st.session_state.pipeline_result.get("all_results", {})
        evolution = all_results.get("evolution", {})

        pred_summary = evolution.get("prediction_summary", [])
        if pred_summary:
            # Heatmap of predictions
            positions = [p["mutation"] for p in pred_summary[:15]]
            probabilities = [p["probability"] for p in pred_summary[:15]]
            escape_impacts = [p["escape_impact"] for p in pred_summary[:15]]
            fitness_impacts = [p["fitness_impact"] for p in pred_summary[:15]]

            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=positions,
                            y=probabilities,
                            marker=dict(
                                color=escape_impacts,
                                colorscale="RdYlGn_r",
                                colorbar=dict(title="Escape Impact"),
                            ),
                            text=[f"{p:.1%}" for p in probabilities],
                            textposition="outside",
                        )
                    ]
                )
                fig.update_layout(
                    title="Predicted Mutations (by Probability)",
                    xaxis_title="Mutation",
                    yaxis_title="Probability",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=450,
                    xaxis_tickangle=-45,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig_scatter = go.Figure(
                    data=[
                        go.Scatter(
                            x=fitness_impacts,
                            y=escape_impacts,
                            mode="markers+text",
                            marker=dict(
                                size=[p * 80 for p in probabilities],
                                color=probabilities,
                                colorscale="Plasma",
                                colorbar=dict(title="Probability"),
                                line=dict(width=1, color="white"),
                            ),
                            text=positions,
                            textposition="top center",
                            textfont=dict(size=8),
                        )
                    ]
                )
                fig_scatter.update_layout(
                    title="Fitness vs Escape Impact",
                    xaxis_title="Fitness Impact",
                    yaxis_title="Escape Impact",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=450,
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            # Predictions table
            st.markdown("### 📊 Detailed Predictions")
            df_pred = pd.DataFrame(pred_summary)
            df_pred["probability"] = df_pred["probability"].apply(lambda x: f"{x:.1%}")
            df_pred["fitness_impact"] = df_pred["fitness_impact"].apply(
                lambda x: f"{x:+.3f}"
            )
            df_pred["escape_impact"] = df_pred["escape_impact"].apply(
                lambda x: f"{x:.3f}"
            )
            st.dataframe(df_pred, use_container_width=True, hide_index=True)

        # Evolutionary trajectory
        trajectory = evolution.get("evolutionary_trajectory", [])
        if trajectory:
            st.markdown("### 🛤️ Evolutionary Trajectory")
            for step in trajectory:
                with st.expander(
                    f"Step {step['step']} — {step['timeframe_days']} days"
                ):
                    st.write(f"Cumulative mutations: {step['cumulative_mutations']}")
                    st.write(
                        f"Estimated fitness change: {step['estimated_fitness_change']:+.3f}"
                    )
                    if step["predicted_new_mutations"]:
                        df_step = pd.DataFrame(step["predicted_new_mutations"])
                        st.dataframe(df_step, use_container_width=True, hide_index=True)
    else:
        st.info("Run the ORACLE pipeline to see mutation forecasts.")


# ─────────────── Tab 4: Phylogenetics ───────────────

with tabs[3]:
    st.markdown(
        '<div class="section-header">🌳 Phylogenetic Analysis</div>',
        unsafe_allow_html=True,
    )

    if (
        st.session_state.pipeline_result
        and st.session_state.pipeline_result["status"] == "completed"
    ):
        # Build a simple tree visualization
        lineage_data = []
        for lineage, data in LINEAGE_DEFINITIONS.items():
            lineage_data.append(
                {
                    "Lineage": lineage,
                    "WHO Label": data.get("who_label", "N/A"),
                    "Mutations": len(data.get("mutations", [])),
                    "Fitness": data.get("fitness", 1.0),
                    "Escape": data.get("escape", 0.0),
                    "Country": data.get("country", "Unknown"),
                }
            )

        df_lineage = pd.DataFrame(lineage_data)

        col1, col2 = st.columns(2)
        with col1:
            fig_tree = go.Figure(
                data=[
                    go.Scatter(
                        x=df_lineage["Fitness"],
                        y=df_lineage["Escape"],
                        mode="markers+text",
                        marker=dict(
                            size=df_lineage["Mutations"] * 3,
                            color=df_lineage["Escape"],
                            colorscale="RdYlGn_r",
                            colorbar=dict(title="Escape Score"),
                            line=dict(width=2, color="white"),
                        ),
                        text=df_lineage["WHO Label"],
                        textposition="top center",
                    )
                ]
            )
            fig_tree.update_layout(
                title="Variant Landscape (Fitness vs Escape)",
                xaxis_title="Relative Fitness",
                yaxis_title="Immune Escape Score",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=500,
            )
            st.plotly_chart(fig_tree, use_container_width=True)

        with col2:
            fig_muts = go.Figure(
                data=[
                    go.Bar(
                        y=df_lineage["WHO Label"],
                        x=df_lineage["Mutations"],
                        orientation="h",
                        marker=dict(
                            color=df_lineage["Mutations"],
                            colorscale="Viridis",
                        ),
                        text=df_lineage["Mutations"],
                        textposition="outside",
                    )
                ]
            )
            fig_muts.update_layout(
                title="Mutations per Lineage",
                xaxis_title="Number of Defining Mutations",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=500,
            )
            st.plotly_chart(fig_muts, use_container_width=True)

        st.markdown("### 📋 Lineage Database")
        st.dataframe(df_lineage, use_container_width=True, hide_index=True)
    else:
        st.info("Run the pipeline to view phylogenetic data.")


# ─────────────── Tab 5: Immune Escape ───────────────

with tabs[4]:
    st.markdown(
        '<div class="section-header">🛡️ Immune Escape Analysis</div>',
        unsafe_allow_html=True,
    )

    if (
        st.session_state.pipeline_result
        and st.session_state.pipeline_result["status"] == "completed"
    ):
        all_results = st.session_state.pipeline_result.get("all_results", {})
        escape = all_results.get("escape", {})

        escape_summary = escape.get("escape_scores_summary", [])
        if escape_summary:
            # Escape scores bar chart
            lineages = [s["lineage"] for s in escape_summary]
            overall_scores = [s["overall_escape"] for s in escape_summary]

            fig_escape = go.Figure()

            # Add class-specific bars
            for cls in ["class1", "class2", "class3", "class4"]:
                cls_scores = [
                    s.get("class_scores", {}).get(cls, 0) for s in escape_summary
                ]
                fig_escape.add_trace(
                    go.Bar(
                        name=f"Ab {cls.title()}",
                        x=lineages,
                        y=cls_scores,
                    )
                )

            fig_escape.update_layout(
                title="Antibody Escape by Class",
                xaxis_title="Lineage",
                yaxis_title="Escape Score",
                barmode="group",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_escape, use_container_width=True)

            # Overall escape + vaccine escape
            col1, col2 = st.columns(2)
            with col1:
                fig_overall = go.Figure(
                    data=[
                        go.Bar(
                            x=lineages,
                            y=overall_scores,
                            marker=dict(
                                color=overall_scores,
                                colorscale="RdYlGn_r",
                            ),
                            text=[f"{s:.2f}" for s in overall_scores],
                            textposition="outside",
                        )
                    ]
                )
                fig_overall.update_layout(
                    title="Overall Immune Escape Score",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=350,
                    yaxis_range=[0, 1],
                )
                st.plotly_chart(fig_overall, use_container_width=True)

            with col2:
                vaccine_scores = [s.get("vaccine_escape", 0) for s in escape_summary]
                fig_vax = go.Figure(
                    data=[
                        go.Bar(
                            x=lineages,
                            y=vaccine_scores,
                            marker=dict(
                                color=vaccine_scores,
                                colorscale="Reds",
                            ),
                            text=[f"{s:.2f}" for s in vaccine_scores],
                            textposition="outside",
                        )
                    ]
                )
                fig_vax.update_layout(
                    title="Vaccine-Induced Immunity Escape",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=350,
                    yaxis_range=[0, 1],
                )
                st.plotly_chart(fig_vax, use_container_width=True)

            # Risk matrix
            risk_matrix = escape.get("risk_matrix", {})
            overall_assessment = risk_matrix.get("overall_assessment", "")
            if overall_assessment:
                st.markdown("### ⚠️ Risk Assessment")
                if "CRITICAL" in overall_assessment:
                    st.error(overall_assessment)
                elif "HIGH" in overall_assessment:
                    st.warning(overall_assessment)
                else:
                    st.success(overall_assessment)

            # Escape scores table
            st.markdown("### 📊 Detailed Escape Scores")
            df_escape = pd.DataFrame(escape_summary)
            st.dataframe(df_escape, use_container_width=True, hide_index=True)
    else:
        st.info("Run the pipeline to see immune escape analysis.")


# ─────────────── Tab 6: Vaccine Candidates ───────────────

with tabs[5]:
    st.markdown(
        '<div class="section-header">💉 Vaccine Candidate Design</div>',
        unsafe_allow_html=True,
    )

    if (
        st.session_state.pipeline_result
        and st.session_state.pipeline_result["status"] == "completed"
    ):
        all_results = st.session_state.pipeline_result.get("all_results", {})
        vaccine = all_results.get("vaccine", {})

        candidate_comp = vaccine.get("candidate_comparison", [])
        if candidate_comp:
            # Radar chart for top 3 candidates
            top3 = candidate_comp[:3]
            categories = ["Immunogenicity", "Breadth", "Stability", "Escape Resistance"]

            fig_radar = go.Figure()
            colors = ["#00d2ff", "#ff6b6b", "#ffd93d"]
            for i, c in enumerate(top3):
                values = [
                    c["immunogenicity"],
                    c["breadth"],
                    c["stability"],
                    c["escape_resistance"],
                ]
                values.append(values[0])  # Close the polygon
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=values,
                        theta=categories + [categories[0]],
                        fill="toself",
                        name=f"#{c['rank']} {c['method']}",
                        fillcolor=f"rgba({int(colors[i][1:3], 16)},{int(colors[i][3:5], 16)},{int(colors[i][5:7], 16)},0.15)",
                        line=dict(color=colors[i], width=2),
                    )
                )

            fig_radar.update_layout(
                title="Top 3 Candidates — Multi-Objective Comparison",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                height=500,
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1]),
                ),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # All candidates comparison
            st.markdown("### 🏆 Ranked Candidates")
            df_candidates = pd.DataFrame(candidate_comp)
            for col in [
                "immunogenicity",
                "breadth",
                "stability",
                "escape_resistance",
                "overall_score",
            ]:
                if col in df_candidates.columns:
                    df_candidates[col] = df_candidates[col].apply(lambda x: f"{x:.3f}")
            st.dataframe(df_candidates, use_container_width=True, hide_index=True)

            # Top recommendation
            rec = vaccine.get("recommendation", {})
            if rec:
                st.markdown("### 🏆 Top Recommendation")
                st.success(
                    f"**{rec.get('recommended_candidate', 'N/A')}** | "
                    f"Method: {rec.get('method', 'N/A')} | "
                    f"Overall Score: {rec.get('scores', {}).get('overall', 0):.3f}"
                )
                st.write(f"**Rationale:** {rec.get('rationale', 'N/A')}")
    else:
        st.info("Run the pipeline to see vaccine candidates.")


# ─────────────── Tab 7: Agent Monitor ───────────────

with tabs[6]:
    st.markdown(
        '<div class="section-header">🤖 Agent Pipeline Monitor</div>',
        unsafe_allow_html=True,
    )

    if (
        st.session_state.pipeline_result
        and st.session_state.pipeline_result["status"] == "completed"
    ):
        result = st.session_state.pipeline_result
        durations = result.get("agent_durations", {})

        # Agent pipeline visualization
        agents_order = ["surveillance", "evolution", "escape", "vaccine", "report"]
        agent_labels = {
            "surveillance": "📡 Surveillance",
            "evolution": "🔮 Evolution",
            "escape": "🛡️ Escape",
            "vaccine": "💉 Vaccine",
            "report": "📋 Report",
        }

        cols = st.columns(5)
        for i, agent in enumerate(agents_order):
            with cols[i]:
                dur = durations.get(agent, 0)
                st.markdown(
                    f"""<div class="metric-card">
                    <div style="font-size:1.5rem;">{agent_labels[agent].split()[0]}</div>
                    <div class="value" style="font-size:1.2rem;">{dur:.2f}s</div>
                    <div class="label">{agent_labels[agent].split()[1] if len(agent_labels[agent].split()) > 1 else agent}</div>
                    <div class="agent-status agent-completed" style="margin-top:8px;">✓ COMPLETE</div>
                </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # Duration breakdown
        fig_dur = go.Figure(
            data=[
                go.Bar(
                    x=[agent_labels.get(a, a) for a in agents_order],
                    y=[durations.get(a, 0) for a in agents_order],
                    marker=dict(
                        color=[durations.get(a, 0) for a in agents_order],
                        colorscale="Plasma",
                    ),
                    text=[f"{durations.get(a, 0):.2f}s" for a in agents_order],
                    textposition="outside",
                )
            ]
        )
        fig_dur.update_layout(
            title="Agent Execution Duration",
            yaxis_title="Duration (seconds)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
        )
        st.plotly_chart(fig_dur, use_container_width=True)

        # Pipeline info
        st.markdown("### 📄 Pipeline Details")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Pipeline ID:** `{result.get('pipeline_id', 'N/A')}`")
            st.write(
                f"**Total Duration:** {result.get('total_duration_seconds', 0):.2f}s"
            )
            st.write(f"**Status:** ✅ Completed")
        with col2:
            metrics = result.get("metrics", {})
            st.write(
                f"**Sequences Processed:** {metrics.get('sequences_processed', 0)}"
            )
            st.write(f"**Mutations Detected:** {metrics.get('mutations_detected', 0)}")
            st.write(
                f"**Predictions Generated:** {metrics.get('predictions_generated', 0)}"
            )
    else:
        st.info("Run the pipeline to see agent activity.")


# ─────────────── Tab 8: MLOps ───────────────

with tabs[7]:
    st.markdown(
        '<div class="section-header">📈 MLOps Dashboard</div>', unsafe_allow_html=True
    )

    mlops = st.session_state.mlops_pipeline

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(
            "🔄 Run Drift Detection", type="secondary", use_container_width=True
        ):
            if (
                st.session_state.pipeline_result
                and st.session_state.pipeline_result["status"] == "completed"
            ):
                surveillance = st.session_state.pipeline_result["all_results"].get(
                    "surveillance", {}
                )
                lineage_dist = surveillance.get("lineage_distribution", {})
                mlops.drift_detector.set_reference(
                    {"JN.1": 50, "XBB.1.5": 30, "BA.5": 20}
                )
                drift_result = mlops.drift_detector.detect_drift(lineage_dist)
                st.json(drift_result)
            else:
                st.warning("Run the pipeline first.")

    with col2:
        if st.button(
            "🔧 Trigger Retraining", type="secondary", use_container_width=True
        ):
            if st.session_state.pipeline_result:
                surveillance = st.session_state.pipeline_result["all_results"].get(
                    "surveillance", {}
                )
                result = mlops.retrain(
                    surveillance,
                    "evolution_forecaster",
                    {"psi_score": 0.15, "severity": "MODERATE"},
                )
                st.json(result)
            else:
                st.warning("Run the pipeline first.")

    with col3:
        if st.button(
            "📊 Refresh Dashboard", type="secondary", use_container_width=True
        ):
            st.rerun()

    st.markdown("---")

    # MLOps overview
    dashboard_data = mlops.get_mlops_dashboard_data()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Experiment Runs", dashboard_data["total_runs"])
    with col2:
        st.metric("Registered Model Versions", dashboard_data["total_models"])
    with col3:
        st.metric("Drift Detections", len(dashboard_data["drift_history"]))

    # Model Registry
    models = dashboard_data.get("registered_models", {})
    if models:
        st.markdown("### 📦 Model Registry")
        for model_name, info in models.items():
            st.write(
                f"**{model_name}** — {info['total_versions']} versions | Latest: {info['latest']}"
            )
            st.json(info["stages"])

    # Retraining History
    retrain_history = dashboard_data.get("retraining_history", [])
    if retrain_history:
        st.markdown("### 🔄 Retraining History")
        df_retrain = pd.DataFrame(retrain_history)
        st.dataframe(df_retrain, use_container_width=True, hide_index=True)

    # Drift History
    drift_history = dashboard_data.get("drift_history", [])
    if drift_history:
        st.markdown("### 📉 Drift Detection History")
        df_drift = pd.DataFrame(drift_history)
        st.dataframe(df_drift, use_container_width=True, hide_index=True)


# ─────────────── Footer ───────────────

st.markdown("---")
st.markdown(
    """
<div style="text-align:center; opacity:0.5; font-size:0.8rem; padding:1rem;">
    🧬 ORACLE — Predictive Viral Evolution Engine v1.0.0<br>
    Built with Streamlit • Powered by AI Multi-Agent System • MCP Protocol
</div>
""",
    unsafe_allow_html=True,
)
