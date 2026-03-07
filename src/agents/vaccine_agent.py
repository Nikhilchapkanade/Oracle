"""
ORACLE — Vaccine Designer Agent
Designs updated vaccine candidates targeting predicted future variants.
"""

import logging

from src.agents.base_agent import BaseAgent
from src.ingestion.sequence_ingestion import REFERENCE_SPIKE
from src.ml.vaccine_designer import VaccineDesigner

logger = logging.getLogger(__name__)


class VaccineAgent(BaseAgent):
    """
    Agent 4: Vaccine Design
    - Receives predictions and escape data from upstream agents
    - Designs vaccine candidates using multiple strategies
    - Ranks candidates by multi-objective scoring
    - Produces vaccine recommendation report
    """

    def __init__(self):
        super().__init__(
            name="VaccineAgent",
            description="Designs updated vaccine candidates targeting predicted variants",
        )
        self.designer = VaccineDesigner()

    def execute(self, input_data: dict) -> dict:
        """
        Execute vaccine design.

        Input (from EvolutionAgent + EscapeAgent):
            - predictions: list of MutationPrediction
            - escape_scores: list of EscapeScore
        Output:
            - candidates: ranked list of VaccineCandidate
            - comparison: head-to-head candidate comparison
            - recommendation: top vaccine recommendation
        """
        predictions = input_data.get("predictions", [])
        escape_scores = input_data.get("escape_scores", [])

        self._log(
            "DESIGNING",
            f"Designing vaccines based on {len(predictions)} predictions "
            f"and {len(escape_scores)} escape scores",
        )

        # Design candidates
        candidates = self.designer.design_candidates(
            reference_sequence=REFERENCE_SPIKE,
            predictions=predictions,
            escape_scores=escape_scores,
            num_candidates=10,
        )

        self._log("DESIGNED", f"Generated {len(candidates)} vaccine candidates")

        # Build comparison table
        comparison = []
        for i, c in enumerate(candidates):
            comparison.append(
                {
                    "rank": i + 1,
                    "id": c.id,
                    "method": c.generation_method,
                    "immunogenicity": c.immunogenicity_score,
                    "breadth": c.breadth_score,
                    "stability": c.stability_score,
                    "escape_resistance": c.escape_resistance,
                    "overall_score": round(c.rank_score, 4),
                    "target_mutations": len(c.target_mutations),
                    "rationale": c.design_rationale[:100],
                }
            )

        # Top recommendation
        top = candidates[0] if candidates else None
        recommendation = {}
        if top:
            recommendation = {
                "recommended_candidate": top.id,
                "method": top.generation_method,
                "rationale": top.design_rationale,
                "scores": {
                    "immunogenicity": top.immunogenicity_score,
                    "breadth": top.breadth_score,
                    "stability": top.stability_score,
                    "escape_resistance": top.escape_resistance,
                    "overall": round(top.rank_score, 4),
                },
                "target_mutations": top.target_mutations[:10],
                "sequence_length": len(top.sequence),
            }

        result = {
            "candidates": candidates,
            "candidate_comparison": comparison,
            "recommendation": recommendation,
            "total_candidates": len(candidates),
            "design_strategies_used": list(
                set(c.generation_method for c in candidates)
            ),
        }

        return result
