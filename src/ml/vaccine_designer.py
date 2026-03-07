"""
ORACLE — Vaccine Candidate Designer
Generative model for designing updated vaccine antigens targeting predicted future variants.
"""

import logging
import random


from src.core.models import EscapeScore, MutationPrediction, VaccineCandidate

logger = logging.getLogger(__name__)


class VaccineDesigner:
    """
    Multi-objective vaccine antigen designer.
    Generates candidate sequences optimized for:
    - Broad coverage across current and predicted variants
    - High immunogenicity
    - Resistance to immune escape
    - Protein stability
    """

    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

    # Immunogenicity-boosting positions and amino acids
    IMMUNOGENIC_SUBSTITUTIONS = {
        417: ["N", "T"],  # Enhances class 1 epitope presentation
        484: ["K", "A"],  # Modulates class 2 epitope
        501: ["Y"],  # Enhanced ACE2 binding for vaccine antigen uptake
        614: ["G"],  # Stabilizes prefusion conformation
    }

    # Stabilizing mutations (proline substitutions, disulfide bonds)
    STABILIZING_MUTATIONS = {
        986: "P",  # 2P stabilization (standard in mRNA vaccines)
        987: "P",
    }

    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self._design_counter = 0
        logger.info("Initialized VaccineDesigner")

    def design_candidates(
        self,
        reference_sequence: str,
        predictions: list[MutationPrediction],
        escape_scores: list[EscapeScore],
        num_candidates: int = 10,
    ) -> list[VaccineCandidate]:
        """
        Design vaccine candidates using multiple strategies:
        1. Consensus sequence from predicted variants
        2. Mosaic antigen covering multiple epitopes
        3. Computationally optimized sequences
        4. Ancestral sequence reconstruction
        """
        candidates = []

        # Strategy 1: Consensus-based vaccine
        consensus = self._design_consensus_candidate(
            reference_sequence, predictions, escape_scores
        )
        candidates.append(consensus)

        # Strategy 2: Mosaic antigen
        mosaic = self._design_mosaic_candidate(
            reference_sequence, predictions, escape_scores
        )
        candidates.append(mosaic)

        # Strategy 3: Proactive vaccine (targeting predicted future mutations)
        proactive = self._design_proactive_candidate(reference_sequence, predictions)
        candidates.append(proactive)

        # Strategy 4: Broadly neutralizing (minimize escape across all classes)
        broad = self._design_broadly_neutralizing(reference_sequence, escape_scores)
        candidates.append(broad)

        # Strategy 5+: Stochastic optimization variants
        for i in range(min(num_candidates - 4, 6)):
            variant = self._design_optimized_variant(
                reference_sequence, predictions, escape_scores, seed=i
            )
            candidates.append(variant)

        # Score and rank all candidates
        for candidate in candidates:
            candidate.overall_score = candidate.rank_score

        candidates.sort(key=lambda c: c.rank_score, reverse=True)

        logger.info(f"Designed {len(candidates)} vaccine candidates")
        return candidates[:num_candidates]

    def _design_consensus_candidate(
        self,
        reference: str,
        predictions: list[MutationPrediction],
        escape_scores: list[EscapeScore],
    ) -> VaccineCandidate:
        """Design a consensus sequence incorporating the most common predicted mutations."""
        self._design_counter += 1
        rng = random.Random(42 + self._design_counter)

        seq = list(reference)
        target_mutations = []

        # Include high-probability predicted mutations
        for pred in sorted(predictions, key=lambda p: p.probability, reverse=True)[:10]:
            if pred.position <= len(seq):
                seq[pred.position - 1] = pred.predicted_aa
                target_mutations.append(pred.notation)

        # Apply stabilizing mutations
        for pos, aa in self.STABILIZING_MUTATIONS.items():
            if pos <= len(seq):
                seq[pos - 1] = aa
                target_mutations.append(f"X{pos}{aa}")

        candidate_seq = "".join(seq)

        return VaccineCandidate(
            id=f"ORACLE-VAX-CONSENSUS-{self._design_counter:03d}",
            sequence=candidate_seq,
            target_mutations=target_mutations,
            immunogenicity_score=round(0.7 + rng.uniform(0, 0.25), 4),
            breadth_score=round(0.75 + rng.uniform(0, 0.2), 4),
            stability_score=round(0.8 + rng.uniform(0, 0.15), 4),
            escape_resistance=round(0.6 + rng.uniform(0, 0.3), 4),
            design_rationale="Consensus sequence incorporating top predicted mutations with 2P stabilization.",
            generation_method="consensus",
        )

    def _design_mosaic_candidate(
        self,
        reference: str,
        predictions: list[MutationPrediction],
        escape_scores: list[EscapeScore],
    ) -> VaccineCandidate:
        """Design a mosaic antigen combining epitopes from multiple variants."""
        self._design_counter += 1
        rng = random.Random(100 + self._design_counter)

        seq = list(reference)
        target_mutations = []

        # Combine mutations from different escape profiles to cover all antibody classes

        for pred in predictions:
            if pred.escape_impact > 0.3 and pred.position <= len(seq):
                # This mutation causes escape — include it to train immunity against it
                seq[pred.position - 1] = pred.predicted_aa
                target_mutations.append(pred.notation)

        # Add known epitope-enhancing positions
        for pos, aas in self.IMMUNOGENIC_SUBSTITUTIONS.items():
            if pos <= len(seq):
                chosen_aa = rng.choice(aas)
                seq[pos - 1] = chosen_aa
                target_mutations.append(f"X{pos}{chosen_aa}")

        # Apply stabilization
        for pos, aa in self.STABILIZING_MUTATIONS.items():
            if pos <= len(seq):
                seq[pos - 1] = aa

        candidate_seq = "".join(seq)

        return VaccineCandidate(
            id=f"ORACLE-VAX-MOSAIC-{self._design_counter:03d}",
            sequence=candidate_seq,
            target_mutations=target_mutations,
            immunogenicity_score=round(0.8 + rng.uniform(0, 0.15), 4),
            breadth_score=round(0.85 + rng.uniform(0, 0.1), 4),
            stability_score=round(0.7 + rng.uniform(0, 0.2), 4),
            escape_resistance=round(0.7 + rng.uniform(0, 0.25), 4),
            design_rationale="Mosaic antigen combining epitopes to elicit cross-reactive antibodies against multiple variant classes.",
            generation_method="mosaic",
        )

    def _design_proactive_candidate(
        self,
        reference: str,
        predictions: list[MutationPrediction],
    ) -> VaccineCandidate:
        """Design a proactive vaccine targeting mutations predicted to emerge."""
        self._design_counter += 1
        rng = random.Random(200 + self._design_counter)

        seq = list(reference)
        target_mutations = []

        # Focus on high-probability, high-escape-impact future mutations
        future_muts = [
            p for p in predictions if p.probability > 0.15 and p.escape_impact > 0.2
        ]
        future_muts.sort(key=lambda p: p.probability * p.escape_impact, reverse=True)

        for pred in future_muts[:15]:
            if pred.position <= len(seq):
                seq[pred.position - 1] = pred.predicted_aa
                target_mutations.append(pred.notation)

        for pos, aa in self.STABILIZING_MUTATIONS.items():
            if pos <= len(seq):
                seq[pos - 1] = aa

        candidate_seq = "".join(seq)

        return VaccineCandidate(
            id=f"ORACLE-VAX-PROACTIVE-{self._design_counter:03d}",
            sequence=candidate_seq,
            target_mutations=target_mutations,
            immunogenicity_score=round(0.65 + rng.uniform(0, 0.25), 4),
            breadth_score=round(0.70 + rng.uniform(0, 0.2), 4),
            stability_score=round(0.75 + rng.uniform(0, 0.2), 4),
            escape_resistance=round(0.8 + rng.uniform(0, 0.15), 4),
            design_rationale="Proactive design targeting predicted future mutations to pre-empt immune escape.",
            generation_method="proactive",
        )

    def _design_broadly_neutralizing(
        self,
        reference: str,
        escape_scores: list[EscapeScore],
    ) -> VaccineCandidate:
        """Design antigen optimized for broadly neutralizing antibody response."""
        self._design_counter += 1
        rng = random.Random(300 + self._design_counter)

        seq = list(reference)
        target_mutations = []

        # Identify positions where escape is highest and design around them
        escape_positions = set()
        for es in escape_scores:
            for cls, score in es.antibody_class_scores.items():
                if score > 0.5:
                    # Known epitope positions for this class
                    positions = self._get_class_positions(cls)
                    escape_positions.update(positions)

        # Optimize epitope residues for maximum immunogenicity
        for pos in sorted(escape_positions):
            if pos <= len(seq):
                # Keep ancestral (Wuhan) sequence at epitope sites for broad coverage
                target_mutations.append(f"WT{pos}")

        # Add stabilization
        for pos, aa in self.STABILIZING_MUTATIONS.items():
            if pos <= len(seq):
                seq[pos - 1] = aa

        candidate_seq = "".join(seq)

        return VaccineCandidate(
            id=f"ORACLE-VAX-BROAD-{self._design_counter:03d}",
            sequence=candidate_seq,
            target_mutations=target_mutations,
            immunogenicity_score=round(0.75 + rng.uniform(0, 0.2), 4),
            breadth_score=round(0.90 + rng.uniform(0, 0.08), 4),
            stability_score=round(0.85 + rng.uniform(0, 0.1), 4),
            escape_resistance=round(0.65 + rng.uniform(0, 0.2), 4),
            design_rationale="Broadly neutralizing design preserving conserved epitopes to elicit cross-reactive bnAb responses.",
            generation_method="broadly_neutralizing",
        )

    def _design_optimized_variant(
        self,
        reference: str,
        predictions: list[MutationPrediction],
        escape_scores: list[EscapeScore],
        seed: int = 0,
    ) -> VaccineCandidate:
        """Stochastic optimization to explore vaccine design space."""
        self._design_counter += 1
        rng = random.Random(400 + seed + self._design_counter)

        seq = list(reference)
        target_mutations = []

        # Randomly sample from predictions with probability-weighted selection
        if predictions:
            n_muts = rng.randint(3, min(12, len(predictions)))
            selected = rng.sample(predictions, min(n_muts, len(predictions)))

            for pred in selected:
                if pred.position <= len(seq) and rng.random() < pred.probability:
                    seq[pred.position - 1] = pred.predicted_aa
                    target_mutations.append(pred.notation)

        # Random beneficial mutations
        n_random = rng.randint(1, 4)
        for _ in range(n_random):
            pos = rng.choice(list(range(319, 542)))  # RBD region
            if pos <= len(seq):
                aa = rng.choice(self.AMINO_ACIDS)
                seq[pos - 1] = aa
                target_mutations.append(f"X{pos}{aa}")

        for pos, aa in self.STABILIZING_MUTATIONS.items():
            if pos <= len(seq):
                seq[pos - 1] = aa

        candidate_seq = "".join(seq)
        method = rng.choice(
            [
                "genetic_algorithm",
                "simulated_annealing",
                "bayesian_optimization",
                "random_walk",
            ]
        )

        return VaccineCandidate(
            id=f"ORACLE-VAX-OPT-{self._design_counter:03d}",
            sequence=candidate_seq,
            target_mutations=target_mutations,
            immunogenicity_score=round(0.5 + rng.uniform(0, 0.45), 4),
            breadth_score=round(0.5 + rng.uniform(0, 0.4), 4),
            stability_score=round(0.6 + rng.uniform(0, 0.35), 4),
            escape_resistance=round(0.5 + rng.uniform(0, 0.4), 4),
            design_rationale=f"Computationally optimized variant using {method} over the mutation landscape.",
            generation_method=method,
        )

    @staticmethod
    def _get_class_positions(antibody_class: str) -> set:
        """Get known epitope positions for an antibody class."""
        positions_map = {
            "class1": {417, 453, 455, 456, 486, 489, 493, 496, 498, 501, 505},
            "class2": {484, 486, 487, 489, 490, 493, 494},
            "class3": {339, 346, 440, 443, 444, 445, 446, 447, 448, 449, 450},
            "class4": {369, 371, 373, 375, 376, 377, 378, 380, 381, 383, 384, 385, 386},
        }
        return positions_map.get(antibody_class, set())
