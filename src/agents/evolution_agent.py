"""
ORACLE — Evolution Prediction Agent
Predicts future mutations using the evolutionary trajectory forecaster.
"""

import logging
from src.agents.base_agent import BaseAgent
from src.ml.evolution_forecaster import EvolutionForecaster

logger = logging.getLogger(__name__)


class EvolutionAgent(BaseAgent):
    """
    Agent 2: Evolution Prediction
    - Trains on historical mutation data from Surveillance Agent
    - Predicts likely future mutations
    - Forecasts evolutionary trajectories
    - Identifies convergent evolution signals
    """

    def __init__(self):
        super().__init__(
            name="EvolutionAgent",
            description="Predicts future viral mutations using evolutionary modeling"
        )
        self.forecaster = EvolutionForecaster()

    def execute(self, input_data: dict) -> dict:
        """
        Execute evolution prediction.

        Input (from SurveillanceAgent):
            - sequences: list of ViralSequence
            - frequency_matrix: mutation frequency data
            - convergent_mutations: convergent evolution signals
        Output:
            - predictions: list of MutationPrediction
            - trajectory: evolutionary trajectory forecast
            - probability_matrix: position × AA probability matrix
        """
        sequences = input_data.get("sequences", [])
        if not sequences:
            raise ValueError("No sequences provided for evolution analysis")

        # Train the forecaster
        self._log("TRAINING", f"Training evolution model on {len(sequences)} sequences")
        training_metrics = self.forecaster.train(sequences)
        self._log("TRAINING_COMPLETE", f"Training metrics: {training_metrics}")

        # Generate predictions
        self._log("PREDICTING", "Generating mutation predictions")
        predictions = self.forecaster.predict(top_k=25, timeframe_days=90)
        self._log("PREDICTIONS_COMPLETE", f"Generated {len(predictions)} predictions")

        # Forecast evolutionary trajectory
        current_mutations = input_data.get("top_mutations", [])
        if current_mutations:
            current_mut_names = [m[0] if isinstance(m, tuple) else m for m in current_mutations[:10]]
        else:
            current_mut_names = []

        self._log("TRAJECTORY", "Computing evolutionary trajectory")
        trajectory = self.forecaster.predict_variant_trajectory(
            current_mutations=current_mut_names, steps=4
        )

        # Get full probability matrix
        prob_matrix = self.forecaster.get_mutation_probability_matrix()

        # Identify high-risk predictions
        high_risk = [p for p in predictions if p.escape_impact > 0.3 and p.probability > 0.15]

        # Build prediction summary
        prediction_summary = []
        for p in predictions[:15]:
            prediction_summary.append({
                "mutation": p.notation,
                "probability": p.probability,
                "fitness_impact": p.fitness_impact,
                "escape_impact": p.escape_impact,
                "timeframe_days": p.expected_timeframe_days,
                "confidence": p.confidence_interval,
            })

        result = {
            "predictions": predictions,
            "prediction_summary": prediction_summary,
            "high_risk_predictions": [{
                "mutation": p.notation,
                "probability": p.probability,
                "escape_impact": p.escape_impact,
            } for p in high_risk],
            "evolutionary_trajectory": trajectory,
            "probability_matrix": prob_matrix,
            "training_metrics": training_metrics,
            "total_predictions": len(predictions),
            "high_risk_count": len(high_risk),
        }

        return result
