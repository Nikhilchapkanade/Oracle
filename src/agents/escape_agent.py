"""
ORACLE — Immune Escape Agent
Evaluates immune escape potential of current and predicted variants.
"""

import logging
from collections import defaultdict

from src.agents.base_agent import BaseAgent
from src.core.models import EscapeScore
from src.ml.immune_escape import ImmuneEscapeModel

logger = logging.getLogger(__name__)


class EscapeAgent(BaseAgent):
    """
    Agent 3: Immune Escape Analysis
    - Evaluates antibody escape for observed variants
    - Scores predicted future variants for escape potential
    - Identifies most concerning escape mutations
    - Produces risk assessment
    """

    def __init__(self):
        super().__init__(
            name="EscapeAgent",
            description="Evaluates immune escape potential of current and predicted variants",
        )
        self.escape_model = ImmuneEscapeModel()

    def execute(self, input_data: dict) -> dict:
        """
        Execute immune escape analysis.

        Input (from SurveillanceAgent + EvolutionAgent):
            - sequences: list of ViralSequence
            - predictions: list of MutationPrediction
            - lineage_distribution: dict of lineage counts
        Output:
            - escape_scores: list of EscapeScore per lineage
            - variant_comparison: cross-variant comparison
            - risk_matrix: risk assessment matrix
        """
        sequences = input_data.get("sequences", [])
        predictions = input_data.get("predictions", [])

        if not sequences:
            raise ValueError("No sequences for escape analysis")

        # Group sequences by lineage, score representative from each
        self._log("SCORING", "Computing escape scores per lineage")
        lineage_groups = defaultdict(list)
        for seq in sequences:
            lineage_groups[seq.lineage].append(seq)

        escape_scores = []
        for lineage, seqs in lineage_groups.items():
            # Use the sequence with most mutations as representative
            representative = max(seqs, key=lambda s: len(s.mutations))
            score = self.escape_model.compute_escape_score(
                representative.id, lineage, representative.mutations
            )
            escape_scores.append(score)

        self._log("SCORING_COMPLETE", f"Scored {len(escape_scores)} lineages")

        # Compare variants
        comparison = self.escape_model.compare_variants(escape_scores)

        # Build risk matrix
        risk_matrix = self._build_risk_matrix(escape_scores, predictions)

        # Identify most concerning predicted mutations
        concerning_predictions = []
        for pred in predictions:
            if pred.escape_impact > 0.3:
                concerning_predictions.append(
                    {
                        "mutation": pred.notation,
                        "escape_impact": pred.escape_impact,
                        "probability": pred.probability,
                        "combined_risk": round(
                            pred.escape_impact * pred.probability, 4
                        ),
                    }
                )
        concerning_predictions.sort(key=lambda x: x["combined_risk"], reverse=True)

        # Summary statistics
        scores_sorted = sorted(
            escape_scores, key=lambda x: x.overall_escape, reverse=True
        )
        critical_lineages = [s.lineage for s in scores_sorted if s.overall_escape > 0.5]

        result = {
            "escape_scores": escape_scores,
            "escape_scores_summary": [
                {
                    "lineage": s.lineage,
                    "overall_escape": s.overall_escape,
                    "risk_level": s.risk_level,
                    "ace2_binding": s.ace2_binding_change,
                    "vaccine_escape": s.vaccine_escape,
                    "class_scores": s.antibody_class_scores,
                }
                for s in scores_sorted
            ],
            "variant_comparison": comparison,
            "risk_matrix": risk_matrix,
            "concerning_predicted_mutations": concerning_predictions[:10],
            "critical_lineages": critical_lineages,
            "highest_escape_lineage": (
                scores_sorted[0].lineage if scores_sorted else None
            ),
            "average_escape": round(
                sum(s.overall_escape for s in escape_scores)
                / max(len(escape_scores), 1),
                4,
            ),
        }

        return result

    def _build_risk_matrix(
        self, escape_scores: list[EscapeScore], predictions: list
    ) -> dict:
        """Build a risk assessment matrix combining escape and predicted evolution."""
        matrix = {
            "current_risk": [],
            "projected_risk": [],
            "overall_assessment": "",
        }

        for score in escape_scores:
            risk_entry = {
                "lineage": score.lineage,
                "escape_score": score.overall_escape,
                "risk_level": score.risk_level,
                "primary_concern": max(
                    score.antibody_class_scores.items(),
                    key=lambda x: x[1],
                    default=("unknown", 0),
                ),
            }
            matrix["current_risk"].append(risk_entry)

        # Project future risk based on predictions
        high_escape_predictions = [p for p in predictions if p.escape_impact > 0.2]

        if high_escape_predictions:
            projected_escape_increase = sum(
                p.escape_impact * p.probability for p in high_escape_predictions[:10]
            )
            matrix["projected_risk"] = [
                {
                    "timeframe": "90 days",
                    "projected_escape_increase": round(projected_escape_increase, 4),
                    "high_risk_mutations": len(high_escape_predictions),
                    "assessment": (
                        "ELEVATED" if projected_escape_increase > 0.3 else "MODERATE"
                    ),
                }
            ]

        # Overall assessment
        max_escape = max((s.overall_escape for s in escape_scores), default=0)
        if max_escape > 0.7:
            matrix["overall_assessment"] = (
                "CRITICAL — Significant immune evasion. Vaccine update urgently needed."
            )
        elif max_escape > 0.5:
            matrix["overall_assessment"] = (
                "HIGH — Substantial escape detected. Monitor and prepare updated vaccines."
            )
        elif max_escape > 0.3:
            matrix["overall_assessment"] = (
                "MODERATE — Partial escape. Current vaccines retain meaningful efficacy."
            )
        else:
            matrix["overall_assessment"] = (
                "LOW — Minimal escape. Current vaccines remain effective."
            )

        return matrix
