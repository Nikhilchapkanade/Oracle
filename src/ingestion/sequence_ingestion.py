"""
ORACLE — Sequence Ingestion Pipeline
Simulated GISAID/Nextstrain data ingestion with realistic viral sequence generation.
"""

import random
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional

from src.core.models import (
    ViralSequence, Mutation, MutationType, ProteinRegion, Lineage, WHORiskLevel
)

logger = logging.getLogger(__name__)

# ─────────────── Reference Spike Protein (simplified) ───────────────

# Simplified Wuhan-Hu-1 spike protein reference (1273 AA)
# Using a representative subset for simulation
REFERENCE_SPIKE = (
    "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGT"
    "NGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYHKN"
    "NKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSAL"
    "EPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSE"
    "TKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSAS"
    "FSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGN"
    "YNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATV"
    "CGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVI"
    "TPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASY"
    "QTHQHIYQAGSTPCNHVKFDEDDNFETQHGIVFNQVKYTQADTIYGFAVSEKRFSVQKFNGIINYTFTTQPFS"
    "KDISGGFIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVT"
    "QNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKV"
    "EAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVF"
    "LHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTV"
    "YDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWP"
    "WYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYTQQQHIYQA"
)

# ─────────────── Lineage Definitions ───────────────

LINEAGE_DEFINITIONS = {
    "B.1.1.7": {
        "who_label": "Alpha",
        "risk_level": WHORiskLevel.VOC,
        "mutations": ["N501Y", "D614G", "P681H", "A570D", "T716I"],
        "country": "United Kingdom",
        "fitness": 1.5,
        "escape": 0.15,
    },
    "B.1.351": {
        "who_label": "Beta",
        "risk_level": WHORiskLevel.VOC,
        "mutations": ["K417N", "E484K", "N501Y", "D614G", "A701V"],
        "country": "South Africa",
        "fitness": 1.3,
        "escape": 0.45,
    },
    "B.1.617.2": {
        "who_label": "Delta",
        "risk_level": WHORiskLevel.VOC,
        "mutations": ["L452R", "T478K", "D614G", "P681R", "D950N"],
        "country": "India",
        "fitness": 1.8,
        "escape": 0.35,
    },
    "B.1.1.529": {
        "who_label": "Omicron BA.1",
        "risk_level": WHORiskLevel.VOC,
        "mutations": ["G339D", "S371L", "S373P", "K417N", "N440K", "S477N",
                       "T478K", "E484A", "Q493R", "Q498R", "N501Y", "Y505H",
                       "D614G", "H655Y", "P681H", "N764K", "D796Y"],
        "country": "South Africa",
        "fitness": 2.5,
        "escape": 0.65,
    },
    "BA.2": {
        "who_label": "Omicron BA.2",
        "risk_level": WHORiskLevel.VOC,
        "mutations": ["G339D", "S371F", "S373P", "S375F", "T376A", "D405N",
                       "R408S", "K417N", "N440K", "S477N", "T478K", "E484A",
                       "Q493R", "Q498R", "N501Y", "Y505H", "D614G", "H655Y", "N679K", "P681H"],
        "country": "Denmark",
        "fitness": 2.7,
        "escape": 0.60,
    },
    "BA.5": {
        "who_label": "Omicron BA.5",
        "risk_level": WHORiskLevel.VOC,
        "mutations": ["G339D", "S371F", "S373P", "S375F", "T376A", "D405N",
                       "R408S", "K417N", "N440K", "L452R", "S477N", "T478K",
                       "E484A", "F486V", "Q498R", "N501Y", "Y505H", "D614G", "H655Y", "P681H"],
        "country": "Portugal",
        "fitness": 2.9,
        "escape": 0.70,
    },
    "XBB.1.5": {
        "who_label": "Kraken",
        "risk_level": WHORiskLevel.VOC,
        "mutations": ["G339H", "R346T", "L368I", "S371F", "S373P", "S375F",
                       "T376A", "D405N", "R408S", "K417N", "N440K", "V445P",
                       "G446S", "N460K", "S477N", "T478K", "E484A", "F486P",
                       "F490S", "Q498R", "N501Y", "Y505H", "D614G", "H655Y", "P681H"],
        "country": "United States",
        "fitness": 3.2,
        "escape": 0.75,
    },
    "JN.1": {
        "who_label": "JN.1",
        "risk_level": WHORiskLevel.VOI,
        "mutations": ["R346T", "L368I", "S371F", "S373P", "S375F", "T376A",
                       "D405N", "R408S", "K417N", "N440K", "V445H", "G446S",
                       "N460K", "S477N", "T478K", "E484A", "F486P", "Q498R",
                       "N501Y", "Y505H", "D614G", "H655Y", "P681H", "L455S"],
        "country": "Luxembourg",
        "fitness": 3.5,
        "escape": 0.78,
    },
}

COUNTRIES = [
    "United States", "United Kingdom", "Germany", "France", "India", "Brazil",
    "South Africa", "Japan", "Australia", "Canada", "Italy", "Spain",
    "Netherlands", "Denmark", "South Korea", "China", "Mexico", "Argentina",
    "Turkey", "Indonesia", "Nigeria", "Kenya", "Thailand", "Vietnam",
]

CONTINENTS = {
    "United States": "North America", "Canada": "North America", "Mexico": "North America",
    "Brazil": "South America", "Argentina": "South America",
    "United Kingdom": "Europe", "Germany": "Europe", "France": "Europe",
    "Italy": "Europe", "Spain": "Europe", "Netherlands": "Europe", "Denmark": "Europe",
    "India": "Asia", "Japan": "Asia", "South Korea": "Asia", "China": "Asia",
    "Thailand": "Asia", "Vietnam": "Asia", "Indonesia": "Asia", "Turkey": "Asia",
    "South Africa": "Africa", "Nigeria": "Africa", "Kenya": "Africa",
    "Australia": "Oceania", "Luxembourg": "Europe", "Portugal": "Europe",
}

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _parse_mutation(mut_str: str) -> Optional[Mutation]:
    """Parse mutation notation like 'N501Y' into a Mutation object."""
    if len(mut_str) < 3:
        return None

    ref_aa = mut_str[0]
    mut_aa = mut_str[-1]
    try:
        position = int(mut_str[1:-1])
    except ValueError:
        if mut_str.endswith("del"):
            position = int(mut_str[1:-3])
            return Mutation(
                position=position, reference_aa=ref_aa, mutant_aa="-",
                mutation_type=MutationType.DELETION,
                protein_region=_get_protein_region(position),
            )
        return None

    return Mutation(
        position=position,
        reference_aa=ref_aa,
        mutant_aa=mut_aa,
        mutation_type=MutationType.NONSYNONYMOUS,
        protein_region=_get_protein_region(position),
    )


def _get_protein_region(position: int) -> ProteinRegion:
    """Determine which region of the spike a position falls in."""
    if 13 <= position <= 305:
        return ProteinRegion.NTD
    elif 319 <= position <= 541:
        return ProteinRegion.RBD
    elif position <= 685:
        return ProteinRegion.S1
    elif 686 <= position <= 815:
        return ProteinRegion.FP
    elif 912 <= position <= 984:
        return ProteinRegion.HR1
    elif 1163 <= position <= 1213:
        return ProteinRegion.HR2
    else:
        return ProteinRegion.OTHER


def _apply_mutations_to_sequence(reference: str, mutations: list[Mutation]) -> str:
    """Apply mutations to generate a variant sequence."""
    seq = list(reference)
    for m in mutations:
        if 0 < m.position <= len(seq):
            seq[m.position - 1] = m.mutant_aa
    return "".join(seq)


def _generate_additional_mutations(base_mutations: list[Mutation], count: int = 2) -> list[Mutation]:
    """Generate random additional mutations to add variability."""
    extra = []
    for _ in range(count):
        pos = random.randint(1, 1273)
        ref_aa = REFERENCE_SPIKE[pos - 1] if pos <= len(REFERENCE_SPIKE) else "X"
        mut_aa = random.choice(AMINO_ACIDS.replace(ref_aa, "") if ref_aa in AMINO_ACIDS else AMINO_ACIDS)
        extra.append(Mutation(
            position=pos,
            reference_aa=ref_aa,
            mutant_aa=mut_aa,
            mutation_type=MutationType.NONSYNONYMOUS,
            protein_region=_get_protein_region(pos),
            frequency=random.uniform(0.01, 0.3),
        ))
    return extra


class SequenceIngestionPipeline:
    """Generates and ingests simulated viral sequences."""

    def __init__(self, db=None):
        self.db = db
        self._seq_counter = 0

    def generate_sequences(
        self,
        count: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        lineage_weights: Optional[dict[str, float]] = None,
    ) -> list[ViralSequence]:
        """
        Generate realistic simulated viral sequences.

        Args:
            count: Number of sequences to generate
            start_date: Start of collection date range
            end_date: End of collection date range
            lineage_weights: Optional weights for lineage sampling
        """
        if start_date is None:
            start_date = datetime(2024, 1, 1)
        if end_date is None:
            end_date = datetime(2025, 12, 31)

        if lineage_weights is None:
            # More recent lineages have higher weights
            lineage_weights = {
                "B.1.1.7": 0.05, "B.1.351": 0.03, "B.1.617.2": 0.08,
                "B.1.1.529": 0.10, "BA.2": 0.12, "BA.5": 0.15,
                "XBB.1.5": 0.22, "JN.1": 0.25,
            }

        lineages = list(lineage_weights.keys())
        weights = list(lineage_weights.values())

        sequences = []
        date_range = (end_date - start_date).days

        for i in range(count):
            self._seq_counter += 1

            # Pick lineage
            lineage_name = random.choices(lineages, weights=weights, k=1)[0]
            lineage_def = LINEAGE_DEFINITIONS[lineage_name]

            # Parse defining mutations
            mutations = []
            for mut_str in lineage_def["mutations"]:
                m = _parse_mutation(mut_str)
                if m:
                    m.frequency = random.uniform(0.8, 1.0)
                    mutations.append(m)

            # Add random extra mutations (within-lineage diversity)
            extra_count = random.randint(0, 5)
            mutations.extend(_generate_additional_mutations(mutations, extra_count))

            # Generate the variant sequence
            variant_seq = _apply_mutations_to_sequence(REFERENCE_SPIKE, mutations)

            # Random collection date
            collection_date = start_date + timedelta(days=random.randint(0, date_range))

            # Random country
            country = random.choice(COUNTRIES)
            continent = CONTINENTS.get(country, "Unknown")

            # Generate unique ID
            seq_hash = hashlib.md5(f"{lineage_name}_{i}_{self._seq_counter}".encode()).hexdigest()[:8]
            seq_id = f"ORACLE_{lineage_name.replace('.', '')}_{seq_hash}"

            seq = ViralSequence(
                id=seq_id,
                sequence=variant_seq,
                lineage=lineage_name,
                collection_date=collection_date,
                country=country,
                continent=continent,
                mutations=mutations,
                clade=lineage_def.get("who_label", lineage_name),
                quality_score=random.uniform(0.85, 1.0),
                metadata={
                    "submitting_lab": f"Lab_{random.randint(1, 500)}",
                    "sequencing_tech": random.choice(["Illumina", "Nanopore", "PacBio"]),
                    "coverage": random.uniform(95.0, 100.0),
                },
            )
            sequences.append(seq)

        logger.info(f"Generated {len(sequences)} simulated viral sequences")
        return sequences

    def generate_lineage_objects(self) -> list[Lineage]:
        """Generate Lineage objects from the pre-defined lineage data."""
        lineages = []
        for name, data in LINEAGE_DEFINITIONS.items():
            lineage = Lineage(
                name=name,
                who_label=data.get("who_label"),
                risk_level=data.get("risk_level", WHORiskLevel.NONE),
                defining_mutations=data.get("mutations", []),
                first_detected_country=data.get("country", ""),
                first_detected=datetime(2020, 6, 1) + timedelta(days=random.randint(0, 1000)),
                sequence_count=random.randint(1000, 500000),
                growth_rate=random.uniform(-0.05, 0.2),
                relative_fitness=data.get("fitness", 1.0),
                immune_escape_score=data.get("escape", 0.0),
                ace2_binding_score=random.uniform(0.8, 1.5),
            )
            lineages.append(lineage)
        return lineages

    def ingest(self, sequences: list[ViralSequence]) -> dict:
        """Ingest sequences into the database."""
        if self.db is None:
            logger.warning("No database connected — skipping persistence")
            return {"ingested": len(sequences), "persisted": False}

        count = self.db.insert_sequences_batch(sequences)
        logger.info(f"Ingested {count} sequences into database")
        return {"ingested": count, "persisted": True}

    def run(self, count: int = 100, **kwargs) -> dict:
        """Full pipeline: generate → ingest → return stats."""
        sequences = self.generate_sequences(count=count, **kwargs)
        result = self.ingest(sequences)

        # Also create lineages
        lineages = self.generate_lineage_objects()
        if self.db:
            for lin in lineages:
                self.db.insert_lineage(lin)

        result["lineages_created"] = len(lineages)
        result["lineage_names"] = [l.name for l in lineages]

        stats = {}
        for seq in sequences:
            stats[seq.lineage] = stats.get(seq.lineage, 0) + 1
        result["distribution"] = stats

        return result
