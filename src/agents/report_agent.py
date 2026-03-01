"""
ORACLE — Report Generator Agent
Aggregates outputs from all agents into a structured WHO-style briefing.
"""

import logging
from datetime import datetime
from src.agents.base_agent import BaseAgent
from src.core.models import AgentReport

logger = logging.getLogger(__name__)


class ReportAgent(BaseAgent):
    """
    Agent 5: Report Generation
    - Aggregates outputs from all upstream agents
    - Generates structured WHO-style briefing report
    - Produces both human-readable and machine-readable outputs
    """

    def __init__(self):
        super().__init__(
            name="ReportAgent",
            description="Generates comprehensive WHO-style briefing reports"
        )

    def execute(self, input_data: dict) -> dict:
        """
        Execute report generation.

        Input (aggregated from all agents):
            - surveillance: dict from SurveillanceAgent
            - evolution: dict from EvolutionAgent
            - escape: dict from EscapeAgent
            - vaccine: dict from VaccineAgent
            - pipeline_id: str
        Output:
            - report: AgentReport
            - markdown_report: str (human-readable)
        """
        surveillance = input_data.get("surveillance", {})
        evolution = input_data.get("evolution", {})
        escape = input_data.get("escape", {})
        vaccine = input_data.get("vaccine", {})
        pipeline_id = input_data.get("pipeline_id", "unknown")

        self._log("GENERATING", "Aggregating agent outputs into briefing report")

        # Build executive summary
        exec_summary = self._build_executive_summary(surveillance, evolution, escape, vaccine)

        # Build risk assessment
        risk_assessment = self._build_risk_assessment(escape)

        # Build recommendations
        recommendations = self._build_recommendations(surveillance, evolution, escape, vaccine)

        # Create the report
        report = AgentReport(
            report_id=f"ORACLE-RPT-{pipeline_id}",
            timestamp=datetime.utcnow(),
            title="ORACLE Viral Evolution Intelligence Briefing",
            executive_summary=exec_summary,
            surveillance_findings=surveillance,
            evolution_predictions=evolution.get("predictions", []),
            escape_analysis=escape.get("escape_scores", []),
            vaccine_candidates=vaccine.get("candidates", []),
            risk_assessment=risk_assessment,
            recommendations=recommendations,
        )

        # Generate markdown report
        markdown = self._generate_markdown_report(report, surveillance, evolution, escape, vaccine)

        self._log("COMPLETE", f"Generated report {report.report_id}")

        result = {
            "report": report,
            "report_id": report.report_id,
            "markdown_report": markdown,
            "executive_summary": exec_summary,
            "risk_assessment": risk_assessment,
            "recommendations": recommendations,
        }

        return result

    def _build_executive_summary(self, surveillance, evolution, escape, vaccine) -> str:
        """Build executive summary from all agent outputs."""
        parts = []

        # Surveillance summary
        total_seqs = surveillance.get("total_sequences", 0)
        unique_muts = surveillance.get("unique_mutations", 0)
        alerts = surveillance.get("alerts", [])
        high_alerts = [a for a in alerts if a.get("level") in ("HIGH", "WARNING")]
        parts.append(f"Analyzed {total_seqs} viral sequences, identifying {unique_muts} unique mutations "
                     f"with {len(high_alerts)} high-priority alerts.")

        # Evolution summary
        total_preds = evolution.get("total_predictions", 0)
        high_risk = evolution.get("high_risk_count", 0)
        parts.append(f"Evolution model generated {total_preds} mutation predictions, "
                     f"with {high_risk} classified as high-risk for immune escape.")

        # Escape summary
        avg_escape = escape.get("average_escape", 0)
        critical = escape.get("critical_lineages", [])
        highest = escape.get("highest_escape_lineage", "unknown")
        parts.append(f"Average immune escape score: {avg_escape:.2f}. "
                     f"Highest escape: {highest}. "
                     f"{len(critical)} lineage(s) classified as critical.")

        # Vaccine summary
        total_candidates = vaccine.get("total_candidates", 0)
        rec = vaccine.get("recommendation", {})
        top_id = rec.get("recommended_candidate", "N/A")
        parts.append(f"Designed {total_candidates} vaccine candidates. "
                     f"Top recommendation: {top_id}.")

        return " ".join(parts)

    def _build_risk_assessment(self, escape) -> str:
        """Build overall risk assessment."""
        risk_matrix = escape.get("risk_matrix", {})
        return risk_matrix.get("overall_assessment",
                                "Unable to assess risk — insufficient data.")

    def _build_recommendations(self, surveillance, evolution, escape, vaccine) -> list[str]:
        """Build actionable recommendations."""
        recommendations = []

        # Based on escape scores
        avg_escape = escape.get("average_escape", 0)
        if avg_escape > 0.5:
            recommendations.append("URGENT: Initiate vaccine update process. "
                                   "Current vaccines show significant immune escape.")
        elif avg_escape > 0.3:
            recommendations.append("MONITOR: Prepare contingency vaccine updates. "
                                   "Moderate escape detected.")

        # Based on evolution predictions
        high_risk = evolution.get("high_risk_count", 0)
        if high_risk > 5:
            recommendations.append(f"ALERT: {high_risk} high-risk mutations predicted. "
                                   "Increase genomic surveillance frequency.")

        # Based on surveillance alerts
        alerts = surveillance.get("alerts", [])
        convergent_alerts = [a for a in alerts if a.get("type") == "CONVERGENT_EVOLUTION"]
        if convergent_alerts:
            recommendations.append("CONVERGENT EVOLUTION: Multiple lineages acquiring similar mutations. "
                                   "Indicates strong selection pressure — prioritize these positions for vaccine targeting.")

        # Vaccine recommendations
        rec = vaccine.get("recommendation", {})
        if rec:
            method = rec.get("method", "unknown")
            recommendations.append(f"VACCINE: Proceed with {rec.get('recommended_candidate', 'top candidate')} "
                                   f"(design method: {method}). Prioritize for pre-clinical evaluation.")

        # Standing recommendations
        recommendations.append("SURVEILLANCE: Continue weekly genomic surveillance at ≥5% sequencing coverage.")
        recommendations.append("COMMUNICATION: Share findings with WHO GISRS network and regional health authorities.")

        return recommendations

    def _generate_markdown_report(self, report, surveillance, evolution, escape, vaccine) -> str:
        """Generate full markdown briefing report."""
        lines = [
            f"# 🧬 {report.title}",
            f"**Report ID:** {report.report_id}",
            f"**Date:** {report.timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Classification:** ORACLE Automated Intelligence Report",
            "",
            "---",
            "",
            "## Executive Summary",
            report.executive_summary,
            "",
            "---",
            "",
            "## 🚨 Risk Assessment",
            report.risk_assessment,
            "",
            "---",
            "",
            "## 📡 Surveillance Findings",
            f"- **Total sequences analyzed:** {surveillance.get('total_sequences', 0)}",
            f"- **Unique mutations detected:** {surveillance.get('unique_mutations', 0)}",
            f"- **Lineages detected:** {len(surveillance.get('lineage_distribution', {}))}",
            "",
        ]

        # Alerts
        alerts = surveillance.get("alerts", [])
        if alerts:
            lines.append("### Alerts")
            for a in alerts:
                icon = "🔴" if a["level"] == "HIGH" else "🟡" if a["level"] == "WARNING" else "🔵"
                lines.append(f"- {icon} **{a['level']}** [{a['type']}]: {a['message']}")
            lines.append("")

        # Top mutations
        top_muts = surveillance.get("top_mutations", [])
        if top_muts:
            lines.extend([
                "### Top Mutations",
                "| Mutation | Frequency |",
                "|----------|-----------|",
            ])
            for mut, count in top_muts[:10]:
                lines.append(f"| {mut} | {count} |")
            lines.append("")

        # Evolution Predictions
        lines.extend([
            "---",
            "",
            "## 🔮 Evolution Predictions",
        ])
        pred_summary = evolution.get("prediction_summary", [])
        if pred_summary:
            lines.extend([
                "| Predicted Mutation | Probability | Fitness Impact | Escape Impact |",
                "|-------------------|-------------|----------------|---------------|",
            ])
            for p in pred_summary[:10]:
                lines.append(
                    f"| {p['mutation']} | {p['probability']:.2%} | {p['fitness_impact']:+.3f} | {p['escape_impact']:.3f} |"
                )
            lines.append("")

        # Trajectory
        trajectory = evolution.get("evolutionary_trajectory", [])
        if trajectory:
            lines.append("### Projected Evolutionary Trajectory")
            for step in trajectory:
                days = step["timeframe_days"]
                n_muts = len(step["predicted_new_mutations"])
                lines.append(f"- **{days} days:** +{n_muts} predicted new mutations "
                             f"(cumulative: {step['cumulative_mutations']})")
            lines.append("")

        # Escape Analysis
        lines.extend([
            "---",
            "",
            "## 🛡️ Immune Escape Analysis",
        ])
        escape_summary = escape.get("escape_scores_summary", [])
        if escape_summary:
            lines.extend([
                "| Lineage | Escape Score | Risk Level | Vaccine Escape | ACE2 Change |",
                "|---------|-------------|------------|----------------|-------------|",
            ])
            for s in escape_summary:
                lines.append(
                    f"| {s['lineage']} | {s['overall_escape']:.3f} | {s['risk_level']} | "
                    f"{s['vaccine_escape']:.3f} | {s['ace2_binding']:+.3f} |"
                )
            lines.append("")

        # Vaccine Candidates
        lines.extend([
            "---",
            "",
            "## 💉 Vaccine Candidates",
        ])
        candidate_comp = vaccine.get("candidate_comparison", [])
        if candidate_comp:
            lines.extend([
                "| Rank | ID | Method | Immunogenicity | Breadth | Stability | Escape Res. | Overall |",
                "|------|-----|--------|---------------|---------|-----------|-------------|---------|",
            ])
            for c in candidate_comp[:10]:
                lines.append(
                    f"| {c['rank']} | {c['id'][:25]} | {c['method']} | "
                    f"{c['immunogenicity']:.3f} | {c['breadth']:.3f} | {c['stability']:.3f} | "
                    f"{c['escape_resistance']:.3f} | {c['overall_score']:.3f} |"
                )
            lines.append("")

        # Recommendation
        rec = vaccine.get("recommendation", {})
        if rec:
            lines.extend([
                "### 🏆 Top Recommendation",
                f"**Candidate:** {rec.get('recommended_candidate', 'N/A')}",
                f"**Method:** {rec.get('method', 'N/A')}",
                f"**Rationale:** {rec.get('rationale', 'N/A')}",
                "",
            ])

        # Recommendations
        lines.extend([
            "---",
            "",
            "## 📋 Recommendations",
        ])
        for i, r in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {r}")
        lines.append("")

        # Footer
        lines.extend([
            "---",
            "",
            "*This report was generated automatically by ORACLE — Predictive Viral Evolution Engine.*",
            f"*Report generated at: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}*",
        ])

        return "\n".join(lines)
