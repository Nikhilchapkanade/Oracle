"""
ORACLE — Mutation Analyzer
Reference alignment, SNP calling, mutation classification, and spike protein region mapping.
"""

import logging
from collections import Counter, defaultdict
from typing import Optional

from src.core.models import Mutation, MutationType, ProteinRegion, ViralSequence

logger = logging.getLogger(__name__)


class MutationAnalyzer:
    """Analyzes mutations across viral sequences — frequency, co-occurrence, hotspots."""

    # Key functional sites in spike protein
    ACE2_CONTACT = {417, 446, 449, 453, 455, 456, 475, 476, 484, 486, 487, 489, 493, 496, 498, 500, 501, 502, 505}
    ANTIBODY_EPITOPES = {
        "class1": {417, 453, 455, 456, 486, 489, 493, 496, 498, 501, 505},
        "class2": {484, 486, 487, 489, 490, 493, 494},
        "class3": {339, 346, 440, 443, 444, 445, 446, 447, 448, 449, 450},
        "class4": {369, 371, 373, 375, 376, 377, 378, 380, 381, 383, 384, 385, 386},
    }

    def __init__(self):
        self.mutation_counts: Counter = Counter()
        self.position_counts: Counter = Counter()
        self.co_occurrences: defaultdict = defaultdict(Counter)
        self.region_counts: Counter = Counter()
        self.lineage_mutation_profiles: dict = {}

    def analyze_sequences(self, sequences: list[ViralSequence]) -> dict:
        """
        Perform comprehensive mutation analysis across a batch of sequences.

        Returns dict with mutation statistics, hotspots, and patterns.
        """
        self.mutation_counts.clear()
        self.position_counts.clear()
        self.region_counts.clear()
        self.co_occurrences.clear()
        self.lineage_mutation_profiles.clear()

        for seq in sequences:
            mutations_notations = [m.notation for m in seq.mutations]

            for m in seq.mutations:
                self.mutation_counts[m.notation] += 1
                self.position_counts[m.position] += 1
                self.region_counts[m.protein_region.value] += 1

            # Co-occurrence tracking
            for i, m1 in enumerate(mutations_notations):
                for m2 in mutations_notations[i + 1:]:
                    key = tuple(sorted([m1, m2]))
                    self.co_occurrences[key[0]][key[1]] += 1

            # Per-lineage profiles
            if seq.lineage not in self.lineage_mutation_profiles:
                self.lineage_mutation_profiles[seq.lineage] = Counter()
            for m in seq.mutations:
                self.lineage_mutation_profiles[seq.lineage][m.notation] += 1

        total_seqs = len(sequences)

        return {
            "total_sequences": total_seqs,
            "unique_mutations": len(self.mutation_counts),
            "top_mutations": self.mutation_counts.most_common(20),
            "mutation_hotspots": self._find_hotspots(total_seqs),
            "region_distribution": dict(self.region_counts),
            "convergent_mutations": self._find_convergent_mutations(),
            "ace2_contact_mutations": self._get_functional_site_mutations("ace2"),
            "antibody_escape_mutations": self._get_antibody_escape_mutations(),
            "lineage_profiles": {
                k: v.most_common(10) for k, v in self.lineage_mutation_profiles.items()
            },
        }

    def _find_hotspots(self, total_seqs: int, threshold: float = 0.1) -> list[dict]:
        """Find positions that are mutated in >threshold fraction of sequences."""
        hotspots = []
        for pos, count in self.position_counts.most_common(50):
            freq = count / max(total_seqs, 1)
            if freq >= threshold:
                region = self._position_to_region(pos)
                is_ace2 = pos in self.ACE2_CONTACT
                in_epitope = any(pos in sites for sites in self.ANTIBODY_EPITOPES.values())
                hotspots.append({
                    "position": pos,
                    "frequency": round(freq, 4),
                    "count": count,
                    "region": region.value,
                    "ace2_contact": is_ace2,
                    "in_antibody_epitope": in_epitope,
                    "functional_impact": "HIGH" if (is_ace2 or in_epitope) else "MODERATE",
                })
        return sorted(hotspots, key=lambda x: x["frequency"], reverse=True)

    def _find_convergent_mutations(self) -> list[dict]:
        """Find mutations that appear independently across multiple lineages."""
        mutation_lineages = defaultdict(set)
        for lineage, profile in self.lineage_mutation_profiles.items():
            for mut in profile:
                mutation_lineages[mut].add(lineage)

        convergent = []
        for mut, lineages in mutation_lineages.items():
            if len(lineages) >= 3:  # Present in 3+ lineages = convergent
                convergent.append({
                    "mutation": mut,
                    "lineage_count": len(lineages),
                    "lineages": sorted(lineages),
                    "total_frequency": self.mutation_counts.get(mut, 0),
                })
        return sorted(convergent, key=lambda x: x["lineage_count"], reverse=True)

    def _get_functional_site_mutations(self, site_type: str = "ace2") -> list[dict]:
        """Get mutations at ACE2 contact residues."""
        sites = self.ACE2_CONTACT
        results = []
        for mut_notation, count in self.mutation_counts.items():
            try:
                pos = int(mut_notation[1:-1])
            except ValueError:
                continue
            if pos in sites:
                results.append({
                    "mutation": mut_notation,
                    "position": pos,
                    "count": count,
                    "impact": "HIGH — ACE2 contact residue",
                })
        return sorted(results, key=lambda x: x["count"], reverse=True)

    def _get_antibody_escape_mutations(self) -> dict[str, list[dict]]:
        """Get mutations at known antibody epitope sites, grouped by class."""
        result = {}
        for ab_class, sites in self.ANTIBODY_EPITOPES.items():
            mutations_in_class = []
            for mut_notation, count in self.mutation_counts.items():
                try:
                    pos = int(mut_notation[1:-1])
                except ValueError:
                    continue
                if pos in sites:
                    mutations_in_class.append({
                        "mutation": mut_notation,
                        "position": pos,
                        "count": count,
                    })
            result[ab_class] = sorted(mutations_in_class, key=lambda x: x["count"], reverse=True)
        return result

    def compute_mutation_frequency_matrix(self, sequences: list[ViralSequence]) -> dict:
        """
        Compute a position × amino acid frequency matrix for the spike protein.
        Returns a sparse dict: {position: {amino_acid: frequency}}
        """
        position_aa_counts = defaultdict(Counter)
        total = len(sequences)

        for seq in sequences:
            for m in seq.mutations:
                position_aa_counts[m.position][m.mutant_aa] += 1

        matrix = {}
        for pos, aa_counts in position_aa_counts.items():
            matrix[pos] = {
                aa: round(count / total, 6)
                for aa, count in aa_counts.most_common()
            }

        return matrix

    @staticmethod
    def _position_to_region(position: int) -> ProteinRegion:
        """Map a spike position to its protein region."""
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
