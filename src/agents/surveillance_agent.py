"""
ORACLE — Surveillance Agent
Monitors incoming sequences for novel mutations, emerging lineages, and anomalies.
"""

import logging
from src.agents.base_agent import BaseAgent
from src.ingestion.sequence_ingestion import SequenceIngestionPipeline
from src.ingestion.mutation_analyzer import MutationAnalyzer
from src.core.models import ViralSequence

logger = logging.getLogger(__name__)


class SurveillanceAgent(BaseAgent):
    """
    Agent 1: Surveillance
    - Monitors incoming viral sequences
    - Detects novel mutations and emerging lineages
    - Triggers alerts for unusual patterns
    - Produces surveillance summary for downstream agents
    """

    def __init__(self):
        super().__init__(
            name="SurveillanceAgent",
            description="Monitors incoming sequences for novel mutations and emerging lineages"
        )
        self.ingestion = SequenceIngestionPipeline()
        self.analyzer = MutationAnalyzer()

    def execute(self, input_data: dict) -> dict:
        """
        Execute surveillance scan.

        Input:
            - num_sequences: int (default 200)
        Output:
            - sequences: list of generated sequences
            - mutation_analysis: analysis results
            - alerts: list of triggered alerts
        """
        num_sequences = input_data.get("num_sequences", 200)
        self._log("INGESTION", f"Generating {num_sequences} sequences for surveillance")

        # Generate/ingest sequences
        sequences = self.ingestion.generate_sequences(count=num_sequences)
        self._log("INGESTION_COMPLETE", f"Generated {len(sequences)} sequences")

        # Analyze mutations
        self._log("ANALYSIS", "Running mutation analysis")
        analysis = self.analyzer.analyze_sequences(sequences)

        # Generate alerts
        alerts = self._generate_alerts(analysis, sequences)
        self._log("ALERTS", f"Generated {len(alerts)} alerts")

        # Compute lineage distribution
        lineage_dist = {}
        for seq in sequences:
            lineage_dist[seq.lineage] = lineage_dist.get(seq.lineage, 0) + 1

        # Frequency matrix for downstream evolution prediction
        frequency_matrix = self.analyzer.compute_mutation_frequency_matrix(sequences)

        result = {
            "sequences": sequences,
            "total_sequences": len(sequences),
            "lineage_distribution": lineage_dist,
            "unique_mutations": analysis["unique_mutations"],
            "top_mutations": analysis["top_mutations"][:15],
            "mutation_hotspots": analysis["mutation_hotspots"][:10],
            "convergent_mutations": analysis["convergent_mutations"][:10],
            "ace2_contact_mutations": analysis["ace2_contact_mutations"][:10],
            "antibody_escape_mutations": {
                k: v[:5] for k, v in analysis["antibody_escape_mutations"].items()
            },
            "region_distribution": analysis["region_distribution"],
            "frequency_matrix": frequency_matrix,
            "alerts": alerts,
        }

        return result

    def _generate_alerts(self, analysis: dict, sequences: list[ViralSequence]) -> list[dict]:
        """Generate surveillance alerts based on analysis findings."""
        alerts = []

        # Alert: Convergent evolution detected
        convergent = analysis.get("convergent_mutations", [])
        if convergent:
            top_convergent = convergent[0]
            alerts.append({
                "level": "WARNING",
                "type": "CONVERGENT_EVOLUTION",
                "message": f"Convergent mutation {top_convergent['mutation']} detected in "
                           f"{top_convergent['lineage_count']} lineages: {', '.join(top_convergent['lineages'][:5])}",
                "details": top_convergent,
            })

        # Alert: ACE2 contact mutations
        ace2_muts = analysis.get("ace2_contact_mutations", [])
        if len(ace2_muts) > 3:
            alerts.append({
                "level": "HIGH",
                "type": "ACE2_CONTACT_MUTATIONS",
                "message": f"{len(ace2_muts)} mutations detected at ACE2 contact residues — "
                           f"potential binding affinity changes",
                "details": {"mutations": [m["mutation"] for m in ace2_muts[:5]]},
            })

        # Alert: Mutation hotspot in RBD
        hotspots = [h for h in analysis.get("mutation_hotspots", [])
                    if h.get("region") == "Receptor binding domain"]
        if hotspots:
            alerts.append({
                "level": "HIGH",
                "type": "RBD_HOTSPOT",
                "message": f"{len(hotspots)} mutation hotspots detected in RBD region",
                "details": {"hotspots": hotspots[:5]},
            })

        # Alert: Rapid lineage growth
        lineage_counts = {}
        for seq in sequences:
            lineage_counts[seq.lineage] = lineage_counts.get(seq.lineage, 0) + 1

        dominant = max(lineage_counts, key=lineage_counts.get, default=None)
        if dominant and lineage_counts[dominant] / len(sequences) > 0.3:
            alerts.append({
                "level": "INFO",
                "type": "DOMINANT_LINEAGE",
                "message": f"Lineage {dominant} is dominant at "
                           f"{lineage_counts[dominant] / len(sequences) * 100:.1f}% prevalence",
                "details": {"lineage": dominant, "count": lineage_counts[dominant]},
            })

        # Alert: High mutation diversity
        if analysis["unique_mutations"] > 50:
            alerts.append({
                "level": "WARNING",
                "type": "HIGH_DIVERSITY",
                "message": f"High mutation diversity: {analysis['unique_mutations']} unique mutations observed",
                "details": {"unique_mutations": analysis["unique_mutations"]},
            })

        return alerts
