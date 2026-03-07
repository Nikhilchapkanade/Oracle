"""
ORACLE — Agent Pipeline Orchestrator
Coordinates the execution of all 5 agents in sequence with data flow management.
"""

import logging
import time
import uuid
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.agents.escape_agent import EscapeAgent
from src.agents.evolution_agent import EvolutionAgent
from src.agents.report_agent import ReportAgent
from src.agents.surveillance_agent import SurveillanceAgent
from src.agents.vaccine_agent import VaccineAgent
from src.core.models import PipelineMetrics

logger = logging.getLogger(__name__)


class OracleOrchestrator:
    """
    Pipeline coordinator that manages the execution of all ORACLE agents.

    Pipeline: Surveillance → Evolution → Escape → Vaccine → Report

    Each agent's output feeds into the next agent's input.
    """

    def __init__(self, db=None):
        self.db = db
        self.pipeline_id = None
        self.agents: dict[str, BaseAgent] = {
            "surveillance": SurveillanceAgent(),
            "evolution": EvolutionAgent(),
            "escape": EscapeAgent(),
            "vaccine": VaccineAgent(),
            "report": ReportAgent(),
        }
        self.metrics: PipelineMetrics = None
        self.results: dict = {}
        logger.info("Initialized OracleOrchestrator with 5 agents")

    def run_pipeline(self, num_sequences: int = 200) -> dict:
        """
        Run the complete ORACLE pipeline.

        Args:
            num_sequences: Number of sequences to analyze

        Returns:
            Complete pipeline result with all agent outputs and report.
        """
        self.pipeline_id = f"PIPE-{uuid.uuid4().hex[:8].upper()}"
        self.results.clear()
        start_time = datetime.utcnow()
        pipeline_start = time.time()

        self.metrics = PipelineMetrics(
            pipeline_id=self.pipeline_id,
            start_time=start_time,
        )

        logger.info("═══════════════════════════════════════════════════════")
        logger.info(f"  ORACLE Pipeline {self.pipeline_id} — STARTING")
        logger.info(f"  Sequences: {num_sequences}")
        logger.info("═══════════════════════════════════════════════════════")

        # Log pipeline start
        if self.db:
            self.db.log_pipeline_run(self.pipeline_id, start_time)

        agent_durations = {}

        try:
            # ─── Stage 1: Surveillance ───
            logger.info("┌─── Stage 1/5: SURVEILLANCE ───────────────────────────")
            surveillance_result = self._run_agent(
                "surveillance", {"num_sequences": num_sequences}
            )
            self.results["surveillance"] = surveillance_result.get("result", {})
            agent_durations["surveillance"] = surveillance_result.get(
                "duration_seconds", 0
            )
            self.metrics.total_sequences_processed = self.results["surveillance"].get(
                "total_sequences", 0
            )
            self.metrics.novel_mutations_detected = self.results["surveillance"].get(
                "unique_mutations", 0
            )
            logger.info(
                f"└─── Surveillance complete: {self.metrics.total_sequences_processed} sequences, "
                f"{self.metrics.novel_mutations_detected} mutations"
            )

            # ─── Stage 2: Evolution Prediction ───
            logger.info("┌─── Stage 2/5: EVOLUTION PREDICTION ──────────────────")
            evolution_input = {
                "sequences": self.results["surveillance"].get("sequences", []),
                "top_mutations": self.results["surveillance"].get("top_mutations", []),
                "convergent_mutations": self.results["surveillance"].get(
                    "convergent_mutations", []
                ),
                "frequency_matrix": self.results["surveillance"].get(
                    "frequency_matrix", {}
                ),
            }
            evolution_result = self._run_agent("evolution", evolution_input)
            self.results["evolution"] = evolution_result.get("result", {})
            agent_durations["evolution"] = evolution_result.get("duration_seconds", 0)
            self.metrics.predictions_generated = self.results["evolution"].get(
                "total_predictions", 0
            )
            logger.info(
                f"└─── Evolution complete: {self.metrics.predictions_generated} predictions"
            )

            # ─── Stage 3: Immune Escape ───
            logger.info("┌─── Stage 3/5: IMMUNE ESCAPE ANALYSIS ────────────────")
            escape_input = {
                "sequences": self.results["surveillance"].get("sequences", []),
                "predictions": self.results["evolution"].get("predictions", []),
                "lineage_distribution": self.results["surveillance"].get(
                    "lineage_distribution", {}
                ),
            }
            escape_result = self._run_agent("escape", escape_input)
            self.results["escape"] = escape_result.get("result", {})
            agent_durations["escape"] = escape_result.get("duration_seconds", 0)
            logger.info(
                f"└─── Escape analysis complete: "
                f"{len(self.results['escape'].get('escape_scores', []))} lineages scored"
            )

            # ─── Stage 4: Vaccine Design ───
            logger.info("┌─── Stage 4/5: VACCINE DESIGN ────────────────────────")
            vaccine_input = {
                "predictions": self.results["evolution"].get("predictions", []),
                "escape_scores": self.results["escape"].get("escape_scores", []),
            }
            vaccine_result = self._run_agent("vaccine", vaccine_input)
            self.results["vaccine"] = vaccine_result.get("result", {})
            agent_durations["vaccine"] = vaccine_result.get("duration_seconds", 0)
            self.metrics.vaccines_designed = self.results["vaccine"].get(
                "total_candidates", 0
            )
            logger.info(
                f"└─── Vaccine design complete: {self.metrics.vaccines_designed} candidates"
            )

            # ─── Stage 5: Report Generation ───
            logger.info("┌─── Stage 5/5: REPORT GENERATION ─────────────────────")
            report_input = {
                "surveillance": self.results["surveillance"],
                "evolution": self.results["evolution"],
                "escape": self.results["escape"],
                "vaccine": self.results["vaccine"],
                "pipeline_id": self.pipeline_id,
            }
            report_result = self._run_agent("report", report_input)
            self.results["report"] = report_result.get("result", {})
            agent_durations["report"] = report_result.get("duration_seconds", 0)
            logger.info("└─── Report generation complete")

            # Finalize metrics
            pipeline_end = time.time()
            self.metrics.end_time = datetime.utcnow()
            self.metrics.agent_durations = agent_durations
            self.metrics.status = "completed"

            total_duration = pipeline_end - pipeline_start

            logger.info("═══════════════════════════════════════════════════════")
            logger.info(f"  ORACLE Pipeline {self.pipeline_id} — COMPLETED")
            logger.info(f"  Total Duration: {total_duration:.2f}s")
            logger.info(f"  Sequences: {self.metrics.total_sequences_processed}")
            logger.info(f"  Mutations: {self.metrics.novel_mutations_detected}")
            logger.info(f"  Predictions: {self.metrics.predictions_generated}")
            logger.info(f"  Vaccines: {self.metrics.vaccines_designed}")
            logger.info("═══════════════════════════════════════════════════════")

            # Update database
            if self.db:
                self.db.update_pipeline_run(
                    self.pipeline_id,
                    end_time=self.metrics.end_time,
                    total_sequences=self.metrics.total_sequences_processed,
                    novel_mutations=self.metrics.novel_mutations_detected,
                    predictions_generated=self.metrics.predictions_generated,
                    vaccines_designed=self.metrics.vaccines_designed,
                    agent_durations=agent_durations,
                    status="completed",
                )

            return {
                "pipeline_id": self.pipeline_id,
                "status": "completed",
                "total_duration_seconds": round(total_duration, 2),
                "metrics": {
                    "sequences_processed": self.metrics.total_sequences_processed,
                    "mutations_detected": self.metrics.novel_mutations_detected,
                    "predictions_generated": self.metrics.predictions_generated,
                    "vaccines_designed": self.metrics.vaccines_designed,
                },
                "agent_durations": agent_durations,
                "report": self.results.get("report", {}),
                "all_results": self.results,
            }

        except Exception as e:
            logger.error(f"Pipeline {self.pipeline_id} FAILED: {e}")
            self.metrics.status = "failed"
            self.metrics.errors.append(str(e))

            if self.db:
                self.db.update_pipeline_run(
                    self.pipeline_id,
                    end_time=datetime.utcnow(),
                    status="failed",
                    errors=[str(e)],
                )

            return {
                "pipeline_id": self.pipeline_id,
                "status": "failed",
                "error": str(e),
                "partial_results": self.results,
            }

    def _run_agent(self, agent_name: str, input_data: dict) -> dict:
        """Run a single agent and log the result."""
        agent = self.agents[agent_name]
        agent.reset()

        if self.db:
            self.db.log_agent_activity(
                self.pipeline_id, agent_name, "running", "Agent started"
            )

        result = agent.run(input_data)

        if self.db:
            self.db.log_agent_activity(
                self.pipeline_id,
                agent_name,
                result["status"],
                f"Completed in {result.get('duration_seconds', 0)}s",
            )

        return result

    def get_pipeline_status(self) -> dict:
        """Get current pipeline status."""
        return {
            "pipeline_id": self.pipeline_id,
            "metrics": {
                "sequences_processed": (
                    self.metrics.total_sequences_processed if self.metrics else 0
                ),
                "mutations_detected": (
                    self.metrics.novel_mutations_detected if self.metrics else 0
                ),
                "predictions_generated": (
                    self.metrics.predictions_generated if self.metrics else 0
                ),
                "vaccines_designed": (
                    self.metrics.vaccines_designed if self.metrics else 0
                ),
                "status": self.metrics.status if self.metrics else "not started",
            },
            "agents": {name: agent.get_status() for name, agent in self.agents.items()},
        }


def main():
    """Run the ORACLE pipeline from command line."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    num_sequences = int(sys.argv[1]) if len(sys.argv) > 1 else 200

    print("🧬 ORACLE — Predictive Viral Evolution Engine")
    print("=" * 55)

    orchestrator = OracleOrchestrator()
    result = orchestrator.run_pipeline(num_sequences=num_sequences)

    if result["status"] == "completed":
        # Print the markdown report
        report = result.get("report", {})
        markdown = report.get("markdown_report", "No report generated.")
        print("\n")
        print(markdown)
    else:
        print(f"\n❌ Pipeline failed: {result.get('error')}")


if __name__ == "__main__":
    main()
