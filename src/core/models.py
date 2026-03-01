"""
ORACLE — Core Data Models
Pydantic models for viral sequences, mutations, lineages, and analysis results.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field


# ─────────────────────────── Enums ───────────────────────────

class MutationType(str, Enum):
    SYNONYMOUS = "synonymous"
    NONSYNONYMOUS = "nonsynonymous"
    DELETION = "deletion"
    INSERTION = "insertion"


class ProteinRegion(str, Enum):
    NTD = "N-terminal domain"
    RBD = "Receptor binding domain"
    S1 = "S1 subunit"
    S2 = "S2 subunit"
    FURIN = "Furin cleavage site"
    FP = "Fusion peptide"
    HR1 = "Heptad repeat 1"
    HR2 = "Heptad repeat 2"
    OTHER = "Other"


class WHORiskLevel(str, Enum):
    VOC = "Variant of Concern"
    VOI = "Variant of Interest"
    VUM = "Variant Under Monitoring"
    NONE = "Not classified"


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"


# ─────────────────────────── Core Models ───────────────────────────

@dataclass
class Mutation:
    """A single amino acid mutation in a viral protein."""
    position: int
    reference_aa: str
    mutant_aa: str
    mutation_type: MutationType = MutationType.NONSYNONYMOUS
    protein_region: ProteinRegion = ProteinRegion.OTHER
    frequency: float = 0.0
    fitness_score: Optional[float] = None
    escape_score: Optional[float] = None

    @property
    def notation(self) -> str:
        if self.mutation_type == MutationType.DELETION:
            return f"{self.reference_aa}{self.position}del"
        return f"{self.reference_aa}{self.position}{self.mutant_aa}"

    def __repr__(self):
        return f"Mutation({self.notation})"


@dataclass
class ViralSequence:
    """A viral genomic sequence with metadata."""
    id: str
    sequence: str
    lineage: str
    collection_date: datetime
    country: str
    continent: str = ""
    host: str = "Human"
    mutations: list[Mutation] = field(default_factory=list)
    clade: str = ""
    quality_score: float = 1.0
    metadata: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def mutation_count(self) -> int:
        return len(self.mutations)


@dataclass
class Lineage:
    """A viral lineage/variant classification."""
    name: str
    parent_lineage: Optional[str] = None
    who_label: Optional[str] = None
    risk_level: WHORiskLevel = WHORiskLevel.NONE
    defining_mutations: list[str] = field(default_factory=list)
    first_detected: Optional[datetime] = None
    first_detected_country: str = ""
    sequence_count: int = 0
    growth_rate: float = 0.0
    relative_fitness: float = 1.0
    immune_escape_score: float = 0.0
    ace2_binding_score: float = 0.0

    @property
    def is_voc(self) -> bool:
        return self.risk_level == WHORiskLevel.VOC


@dataclass
class PhylogeneticNode:
    """A node in a phylogenetic tree."""
    id: str
    name: str
    branch_length: float = 0.0
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    lineage: str = ""
    bootstrap_support: float = 0.0
    mutations_from_parent: list[str] = field(default_factory=list)
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children_ids) == 0


@dataclass
class MutationPrediction:
    """A predicted future mutation with probability."""
    position: int
    current_aa: str
    predicted_aa: str
    probability: float
    expected_timeframe_days: int = 90
    fitness_impact: float = 0.0
    escape_impact: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 1.0)

    @property
    def notation(self) -> str:
        return f"{self.current_aa}{self.position}{self.predicted_aa}"


@dataclass
class EscapeScore:
    """Immune escape analysis result for a variant."""
    variant_id: str
    lineage: str
    overall_escape: float  # 0-1, 1 = complete escape
    antibody_class_scores: dict[str, float] = field(default_factory=dict)
    ace2_binding_change: float = 0.0  # Fold change
    convalescent_escape: float = 0.0
    vaccine_escape: float = 0.0
    most_escaped_epitopes: list[str] = field(default_factory=list)
    risk_assessment: str = ""

    @property
    def risk_level(self) -> str:
        if self.overall_escape > 0.7:
            return "CRITICAL"
        elif self.overall_escape > 0.5:
            return "HIGH"
        elif self.overall_escape > 0.3:
            return "MODERATE"
        return "LOW"


@dataclass
class VaccineCandidate:
    """A proposed vaccine antigen candidate."""
    id: str
    sequence: str
    target_mutations: list[str]
    immunogenicity_score: float = 0.0
    breadth_score: float = 0.0  # Coverage across variants
    stability_score: float = 0.0
    escape_resistance: float = 0.0
    overall_score: float = 0.0
    design_rationale: str = ""
    generation_method: str = ""

    @property
    def rank_score(self) -> float:
        return (
            0.3 * self.immunogenicity_score
            + 0.25 * self.breadth_score
            + 0.2 * self.escape_resistance
            + 0.15 * self.stability_score
            + 0.1 * self.overall_score
        )


@dataclass
class AgentMessage:
    """Message passed between agents in the pipeline."""
    sender: str
    receiver: str
    content: dict
    timestamp: datetime = field(default_factory=datetime.utcnow)
    message_type: str = "data"
    priority: int = 0


@dataclass
class AgentReport:
    """Structured report from the agent pipeline."""
    report_id: str
    timestamp: datetime
    title: str
    executive_summary: str
    surveillance_findings: dict = field(default_factory=dict)
    evolution_predictions: list[MutationPrediction] = field(default_factory=list)
    escape_analysis: list[EscapeScore] = field(default_factory=list)
    vaccine_candidates: list[VaccineCandidate] = field(default_factory=list)
    risk_assessment: str = ""
    recommendations: list[str] = field(default_factory=list)
    agent_execution_log: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class PipelineMetrics:
    """Metrics for the agent pipeline execution."""
    pipeline_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_sequences_processed: int = 0
    novel_mutations_detected: int = 0
    predictions_generated: int = 0
    vaccines_designed: int = 0
    agent_durations: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    status: str = "running"
