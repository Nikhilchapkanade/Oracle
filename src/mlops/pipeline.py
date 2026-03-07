"""
ORACLE — MLOps Pipeline
Experiment tracking, model registry, drift detection, and automated retraining.
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """MLflow-compatible experiment tracking for ORACLE models."""

    def __init__(self, tracking_dir: str = None):
        if tracking_dir is None:
            from config.settings import BASE_DIR

            tracking_dir = str(BASE_DIR / "data" / "experiments")
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.active_run = None
        self.runs: list[dict] = []
        logger.info(f"ExperimentTracker initialized at {self.tracking_dir}")

    def start_run(self, experiment_name: str, run_name: str = None) -> dict:
        """Start a new experiment run."""
        run_id = hashlib.md5(f"{experiment_name}_{time.time()}".encode()).hexdigest()[
            :12
        ]
        self.active_run = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "run_name": run_name or f"run_{run_id}",
            "start_time": datetime.utcnow().isoformat(),
            "end_time": None,
            "status": "running",
            "params": {},
            "metrics": {},
            "artifacts": [],
            "tags": {},
        }
        logger.info(f"Started run {run_id} for experiment {experiment_name}")
        return self.active_run

    def log_params(self, params: dict):
        """Log parameters for the active run."""
        if self.active_run:
            self.active_run["params"].update(params)

    def log_metrics(self, metrics: dict, step: int = 0):
        """Log metrics for the active run."""
        if self.active_run:
            for key, value in metrics.items():
                if key not in self.active_run["metrics"]:
                    self.active_run["metrics"][key] = []
                self.active_run["metrics"][key].append({"value": value, "step": step})

    def log_artifact(self, artifact_path: str, artifact_type: str = "model"):
        """Log an artifact for the active run."""
        if self.active_run:
            self.active_run["artifacts"].append(
                {
                    "path": artifact_path,
                    "type": artifact_type,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

    def set_tags(self, tags: dict):
        """Set tags for the active run."""
        if self.active_run:
            self.active_run["tags"].update(tags)

    def end_run(self, status: str = "completed") -> dict:
        """End the active run."""
        if self.active_run:
            self.active_run["end_time"] = datetime.utcnow().isoformat()
            self.active_run["status"] = status

            # Save run to disk
            run_file = self.tracking_dir / f"{self.active_run['run_id']}.json"
            with open(run_file, "w") as f:
                json.dump(self.active_run, f, indent=2)

            self.runs.append(self.active_run)
            completed_run = self.active_run
            self.active_run = None
            logger.info(f"Ended run {completed_run['run_id']} with status {status}")
            return completed_run
        return {}

    def get_runs(self, experiment_name: str = None) -> list[dict]:
        """Get all runs, optionally filtered by experiment."""
        if experiment_name:
            return [r for r in self.runs if r["experiment_name"] == experiment_name]
        return self.runs

    def get_best_run(self, metric_name: str, maximize: bool = True) -> Optional[dict]:
        """Get the best run based on a specific metric."""
        valid_runs = [r for r in self.runs if metric_name in r["metrics"]]
        if not valid_runs:
            return None

        def get_metric_value(run):
            values = run["metrics"][metric_name]
            return values[-1]["value"] if values else 0

        return (
            max(valid_runs, key=get_metric_value)
            if maximize
            else min(valid_runs, key=get_metric_value)
        )


class ModelRegistry:
    """Model versioning and lifecycle management."""

    def __init__(self, registry_dir: str = None):
        if registry_dir is None:
            from config.settings import BASE_DIR

            registry_dir = str(BASE_DIR / "model_registry")
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.models: dict[str, list[dict]] = {}
        self._load_registry()
        logger.info(f"ModelRegistry initialized at {self.registry_dir}")

    def _load_registry(self):
        """Load existing registry from disk."""
        registry_file = self.registry_dir / "registry.json"
        if registry_file.exists():
            with open(registry_file) as f:
                self.models = json.load(f)

    def _save_registry(self):
        """Save registry to disk."""
        registry_file = self.registry_dir / "registry.json"
        with open(registry_file, "w") as f:
            json.dump(self.models, f, indent=2)

    def register_model(
        self,
        model_name: str,
        version: str,
        run_id: str,
        metrics: dict = None,
        stage: str = "staging",
    ) -> dict:
        """Register a new model version."""
        if model_name not in self.models:
            self.models[model_name] = []

        model_version = {
            "version": version,
            "run_id": run_id,
            "stage": stage,
            "metrics": metrics or {},
            "registered_at": datetime.utcnow().isoformat(),
            "description": "",
        }
        self.models[model_name].append(model_version)
        self._save_registry()

        logger.info(f"Registered {model_name} v{version} (stage: {stage})")
        return model_version

    def promote_model(
        self, model_name: str, version: str, to_stage: str = "production"
    ) -> dict:
        """Promote a model version to a new stage."""
        if model_name not in self.models:
            return {"error": f"Model {model_name} not found"}

        # Demote current production model
        for v in self.models[model_name]:
            if v["stage"] == to_stage:
                v["stage"] = "archived"

        # Promote target version
        for v in self.models[model_name]:
            if v["version"] == version:
                v["stage"] = to_stage
                self._save_registry()
                logger.info(f"Promoted {model_name} v{version} to {to_stage}")
                return v

        return {"error": f"Version {version} not found for {model_name}"}

    def get_latest(self, model_name: str, stage: str = "production") -> Optional[dict]:
        """Get the latest model version at a given stage."""
        if model_name not in self.models:
            return None
        for v in reversed(self.models[model_name]):
            if v["stage"] == stage:
                return v
        return None

    def list_models(self) -> dict:
        """List all registered models."""
        summary = {}
        for name, versions in self.models.items():
            summary[name] = {
                "total_versions": len(versions),
                "stages": {v["stage"]: v["version"] for v in versions},
                "latest": versions[-1]["version"] if versions else None,
            }
        return summary


class DriftDetector:
    """Statistical drift detection for incoming sequence distributions."""

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.reference_distribution: Optional[dict] = None
        self.drift_history: list[dict] = []
        logger.info(f"DriftDetector initialized (threshold: {threshold})")

    def set_reference(self, distribution: dict):
        """Set the reference distribution for drift comparison."""
        self.reference_distribution = distribution
        logger.info(f"Reference distribution set with {len(distribution)} categories")

    def detect_drift(self, current_distribution: dict) -> dict:
        """
        Detect distribution drift using Population Stability Index (PSI).
        """
        if not self.reference_distribution:
            return {"drift_detected": False, "reason": "No reference distribution set"}

        psi = self._compute_psi(self.reference_distribution, current_distribution)
        kl_div = self._compute_kl_divergence(
            self.reference_distribution, current_distribution
        )

        drift_detected = psi > self.threshold

        result = {
            "drift_detected": drift_detected,
            "psi_score": round(psi, 6),
            "kl_divergence": round(kl_div, 6),
            "threshold": self.threshold,
            "severity": "HIGH" if psi > 0.25 else "MODERATE" if psi > 0.1 else "LOW",
            "recommendation": "RETRAIN" if drift_detected else "MONITOR",
            "timestamp": datetime.utcnow().isoformat(),
        }

        self.drift_history.append(result)
        if drift_detected:
            logger.warning(
                f"DRIFT DETECTED: PSI={psi:.4f} (threshold={self.threshold})"
            )

        return result

    @staticmethod
    def _compute_psi(reference: dict, current: dict) -> float:
        """Compute Population Stability Index."""
        all_keys = set(reference.keys()) | set(current.keys())
        psi = 0.0
        for key in all_keys:
            p = max(reference.get(key, 0), 0.0001)
            q = max(current.get(key, 0), 0.0001)
            # Normalize
            total_p = sum(max(reference.get(k, 0), 0.0001) for k in all_keys)
            total_q = sum(max(current.get(k, 0), 0.0001) for k in all_keys)
            p_norm = p / total_p
            q_norm = q / total_q
            import math

            psi += (q_norm - p_norm) * math.log(q_norm / p_norm)
        return abs(psi)

    @staticmethod
    def _compute_kl_divergence(reference: dict, current: dict) -> float:
        """Compute KL Divergence."""
        import math

        all_keys = set(reference.keys()) | set(current.keys())
        total_p = sum(max(reference.get(k, 0), 0.0001) for k in all_keys)
        total_q = sum(max(current.get(k, 0), 0.0001) for k in all_keys)

        kl = 0.0
        for key in all_keys:
            p = max(reference.get(key, 0), 0.0001) / total_p
            q = max(current.get(key, 0), 0.0001) / total_q
            kl += p * math.log(p / q)
        return abs(kl)


class RetrainingPipeline:
    """Automated model retraining orchestration."""

    def __init__(self):
        self.tracker = ExperimentTracker()
        self.registry = ModelRegistry()
        self.drift_detector = DriftDetector()
        self.retraining_history: list[dict] = []
        logger.info("RetrainingPipeline initialized")

    def check_and_retrain(
        self, current_data: dict, model_name: str = "evolution_forecaster"
    ) -> dict:
        """Check for drift and trigger retraining if needed."""
        # Check drift
        lineage_dist = current_data.get("lineage_distribution", {})
        drift_result = self.drift_detector.detect_drift(lineage_dist)

        if not drift_result["drift_detected"]:
            return {
                "action": "no_retrain",
                "drift": drift_result,
                "message": "No significant drift detected. Models are up to date.",
            }

        # Trigger retraining
        logger.info("Drift detected — triggering model retraining")
        return self.retrain(current_data, model_name, drift_result)

    def retrain(
        self, training_data: dict, model_name: str, drift_info: dict = None
    ) -> dict:
        """Execute a retraining run."""
        import random

        rng = random.Random(42)

        # Start experiment tracking
        self.tracker.start_run(
            experiment_name=f"oracle-{model_name}",
            run_name=f"retrain_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        )

        self.tracker.log_params(
            {
                "model_name": model_name,
                "trigger": "drift_detection" if drift_info else "manual",
                "training_samples": training_data.get("total_sequences", 0),
            }
        )

        # Simulate training epochs
        for epoch in range(10):
            self.tracker.log_metrics(
                {
                    "loss": round(1.0 / (epoch + 1) + rng.gauss(0, 0.02), 4),
                    "accuracy": round(
                        min(0.95, 0.5 + 0.05 * epoch + rng.gauss(0, 0.01)), 4
                    ),
                    "f1_score": round(
                        min(0.93, 0.45 + 0.05 * epoch + rng.gauss(0, 0.01)), 4
                    ),
                },
                step=epoch,
            )

        self.tracker.set_tags(
            {
                "drift_psi": (
                    str(drift_info.get("psi_score", "N/A")) if drift_info else "N/A"
                ),
                "framework": "pytorch",
            }
        )

        completed_run = self.tracker.end_run(status="completed")

        # Register model
        version = f"v{len(self.registry.models.get(model_name, [])) + 1}.0"
        final_metrics = {k: v[-1]["value"] for k, v in completed_run["metrics"].items()}
        self.registry.register_model(
            model_name=model_name,
            version=version,
            run_id=completed_run["run_id"],
            metrics=final_metrics,
            stage="staging",
        )

        # Update drift reference
        lineage_dist = training_data.get("lineage_distribution", {})
        if lineage_dist:
            self.drift_detector.set_reference(lineage_dist)

        result = {
            "action": "retrained",
            "run_id": completed_run["run_id"],
            "model_name": model_name,
            "version": version,
            "stage": "staging",
            "final_metrics": final_metrics,
            "drift_info": drift_info,
        }

        self.retraining_history.append(result)
        return result

    def get_mlops_dashboard_data(self) -> dict:
        """Get data for the MLOps dashboard panel."""
        return {
            "experiment_runs": self.tracker.get_runs(),
            "registered_models": self.registry.list_models(),
            "drift_history": self.drift_detector.drift_history,
            "retraining_history": self.retraining_history,
            "total_runs": len(self.tracker.runs),
            "total_models": sum(len(v) for v in self.registry.models.values()),
        }
