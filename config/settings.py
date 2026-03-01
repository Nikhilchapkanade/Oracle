"""
ORACLE — Central Configuration
All project settings managed via environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class DatabaseConfig:
    """Database connection settings."""
    url: str = f"sqlite:///{BASE_DIR / 'data' / 'oracle.db'}"
    echo: bool = False
    pool_size: int = 5


@dataclass
class MLConfig:
    """ML/DL model settings."""
    protein_lm_model: str = "esm2_t6_8M"  # Lightweight ESM-2 variant
    evolution_model_hidden_dim: int = 256
    evolution_model_num_heads: int = 8
    evolution_model_num_layers: int = 4
    escape_gnn_hidden_dim: int = 128
    escape_gnn_num_layers: int = 3
    vaccine_top_k: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_sequence_length: int = 1280
    device: str = "cpu"  # "cuda" for GPU
    models_dir: Path = field(default_factory=lambda: BASE_DIR / "models")


@dataclass
class MCPConfig:
    """MCP Server settings."""
    genomic_server_port: int = 8100
    protein_server_port: int = 8101
    epidemiology_server_port: int = 8102
    phylogenetics_server_port: int = 8103
    host: str = "localhost"


@dataclass
class AgentConfig:
    """Multi-agent system settings."""
    max_retries: int = 3
    timeout_seconds: int = 300
    parallel_execution: bool = False
    log_level: str = "INFO"
    report_output_dir: Path = field(default_factory=lambda: BASE_DIR / "reports")


@dataclass
class MLOpsConfig:
    """MLOps pipeline settings."""
    mlflow_tracking_uri: str = f"sqlite:///{BASE_DIR / 'data' / 'mlflow.db'}"
    experiment_name: str = "oracle-viral-evolution"
    model_registry_path: Path = field(default_factory=lambda: BASE_DIR / "model_registry")
    drift_threshold: float = 0.1  # PSI threshold for drift detection
    retrain_trigger_count: int = 1000  # New sequences before retrain


@dataclass
class MonitoringConfig:
    """Monitoring and observability."""
    prometheus_port: int = 9090
    grafana_port: int = 3000
    enable_metrics: bool = True


@dataclass
class ViralConfig:
    """Viral biology constants."""
    reference_lineage: str = "Wuhan-Hu-1"
    spike_protein_length: int = 1273
    rbd_start: int = 319
    rbd_end: int = 541
    ntd_start: int = 13
    ntd_end: int = 305
    furin_site: int = 681
    ace2_contact_residues: list = field(default_factory=lambda: [
        417, 446, 449, 453, 455, 456, 475, 476, 484, 486, 487, 489, 493, 496, 498, 500, 501, 502, 505
    ])
    who_voc_mutations: dict = field(default_factory=lambda: {
        "Alpha": ["N501Y", "D614G", "P681H"],
        "Beta": ["K417N", "E484K", "N501Y", "D614G"],
        "Delta": ["L452R", "T478K", "D614G", "P681R"],
        "Omicron_BA1": ["G339D", "S371L", "S373P", "K417N", "N440K", "S477N", "T478K",
                         "E484A", "Q493R", "Q498R", "N501Y", "Y505H", "D614G", "H655Y", "P681H"],
    })
    amino_acids: str = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class OracleConfig:
    """Root configuration aggregating all sub-configs."""
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    mlops: MLOpsConfig = field(default_factory=MLOpsConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    viral: ViralConfig = field(default_factory=ViralConfig)
    debug: bool = os.getenv("ORACLE_DEBUG", "false").lower() == "true"

    def ensure_dirs(self):
        """Create all necessary directories."""
        dirs = [
            BASE_DIR / "data",
            BASE_DIR / "models",
            BASE_DIR / "reports",
            BASE_DIR / "model_registry",
            BASE_DIR / "logs",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# Global singleton
config = OracleConfig()
config.ensure_dirs()
