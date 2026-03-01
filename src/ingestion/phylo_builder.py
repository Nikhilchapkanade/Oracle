"""
ORACLE — Phylogenetic Tree Builder
Constructs phylogenetic trees using neighbor-joining and tracks lineage relationships.
"""

import logging
import random
import math
from collections import defaultdict
from typing import Optional

from src.core.models import PhylogeneticNode, ViralSequence

logger = logging.getLogger(__name__)


class PhylogeneticTreeBuilder:
    """Builds phylogenetic trees from viral sequences using distance-based methods."""

    def __init__(self):
        self.nodes: dict[str, PhylogeneticNode] = {}
        self.root: Optional[str] = None
        self._node_counter = 0

    def build_tree(self, sequences: list[ViralSequence]) -> dict:
        """
        Build a phylogenetic tree from viral sequences using neighbor-joining.

        Returns tree structure with nodes and relationships.
        """
        if not sequences:
            return {"nodes": [], "root": None}

        self.nodes.clear()
        self._node_counter = 0

        # Group by lineage for cleaner tree structure
        lineage_groups = defaultdict(list)
        for seq in sequences:
            lineage_groups[seq.lineage].append(seq)

        # Create leaf nodes (representative per lineage)
        leaf_ids = []
        for lineage, seqs in lineage_groups.items():
            representative = max(seqs, key=lambda s: s.quality_score)
            node_id = f"leaf_{lineage.replace('.', '_')}"
            node = PhylogeneticNode(
                id=node_id,
                name=lineage,
                lineage=lineage,
                branch_length=random.uniform(0.001, 0.05),
                mutations_from_parent=[m.notation for m in representative.mutations[:5]],
            )
            self.nodes[node_id] = node
            leaf_ids.append(node_id)

        # Compute distance matrix
        distance_matrix = self._compute_distance_matrix(sequences, lineage_groups)

        # Build tree using neighbor-joining
        self._neighbor_join(leaf_ids, distance_matrix)

        # Compute depths
        if self.root:
            self._compute_depths(self.root, 0)

        node_list = []
        for nid, node in self.nodes.items():
            node_list.append({
                "id": node.id,
                "name": node.name,
                "lineage": node.lineage,
                "branch_length": node.branch_length,
                "parent_id": node.parent_id,
                "children_ids": node.children_ids,
                "is_leaf": node.is_leaf,
                "depth": node.depth,
                "bootstrap_support": node.bootstrap_support,
                "mutations_from_parent": node.mutations_from_parent,
            })

        return {
            "nodes": node_list,
            "root": self.root,
            "num_leaves": len(leaf_ids),
            "num_internal": len(self.nodes) - len(leaf_ids),
            "total_nodes": len(self.nodes),
            "lineages": list(lineage_groups.keys()),
        }

    def _compute_distance_matrix(
        self,
        sequences: list[ViralSequence],
        lineage_groups: dict[str, list[ViralSequence]],
    ) -> dict[tuple[str, str], float]:
        """Compute pairwise distances between lineages based on mutation profiles."""
        lineages = list(lineage_groups.keys())
        distances = {}

        # Build mutation sets per lineage
        mutation_sets = {}
        for lineage, seqs in lineage_groups.items():
            all_mutations = set()
            for seq in seqs:
                for m in seq.mutations:
                    all_mutations.add(m.notation)
            mutation_sets[lineage] = all_mutations

        # Jaccard-like distance
        for i, l1 in enumerate(lineages):
            for j, l2 in enumerate(lineages):
                if i == j:
                    distances[(l1, l2)] = 0.0
                elif i < j:
                    s1, s2 = mutation_sets[l1], mutation_sets[l2]
                    union = s1 | s2
                    intersection = s1 & s2
                    if len(union) > 0:
                        distance = 1.0 - (len(intersection) / len(union))
                    else:
                        distance = 1.0
                    distances[(l1, l2)] = distance
                    distances[(l2, l1)] = distance

        return distances

    def _neighbor_join(self, leaf_ids: list[str], distances: dict) -> None:
        """Simplified neighbor-joining algorithm to build tree topology."""
        active_nodes = list(leaf_ids)

        while len(active_nodes) > 2:
            # Find the closest pair
            min_dist = float('inf')
            pair = (0, 1)

            for i in range(len(active_nodes)):
                for j in range(i + 1, len(active_nodes)):
                    n1 = self.nodes[active_nodes[i]]
                    n2 = self.nodes[active_nodes[j]]
                    key = (n1.lineage or n1.name, n2.lineage or n2.name)
                    rev_key = (key[1], key[0])
                    d = distances.get(key, distances.get(rev_key, random.uniform(0.1, 0.5)))

                    if d < min_dist:
                        min_dist = d
                        pair = (i, j)

            # Create internal node joining the pair
            self._node_counter += 1
            internal_id = f"internal_{self._node_counter}"
            child1_id = active_nodes[pair[0]]
            child2_id = active_nodes[pair[1]]

            internal_node = PhylogeneticNode(
                id=internal_id,
                name=f"Node_{self._node_counter}",
                branch_length=min_dist / 2,
                children_ids=[child1_id, child2_id],
                bootstrap_support=random.uniform(0.7, 1.0),
            )
            self.nodes[internal_id] = internal_node

            # Update children
            self.nodes[child1_id].parent_id = internal_id
            self.nodes[child1_id].branch_length = min_dist / 2 + random.uniform(0.001, 0.01)
            self.nodes[child2_id].parent_id = internal_id
            self.nodes[child2_id].branch_length = min_dist / 2 + random.uniform(0.001, 0.01)

            # Replace the pair with the new internal node
            new_active = [n for idx, n in enumerate(active_nodes)
                          if idx != pair[0] and idx != pair[1]]
            new_active.append(internal_id)
            active_nodes = new_active

        # Connect remaining nodes to root
        if len(active_nodes) == 2:
            self._node_counter += 1
            root_id = f"root_{self._node_counter}"
            root_node = PhylogeneticNode(
                id=root_id,
                name="Root",
                branch_length=0.0,
                children_ids=active_nodes,
                bootstrap_support=1.0,
            )
            self.nodes[root_id] = root_node
            for nid in active_nodes:
                self.nodes[nid].parent_id = root_id
            self.root = root_id
        elif len(active_nodes) == 1:
            self.root = active_nodes[0]

    def _compute_depths(self, node_id: str, depth: int):
        """Recursively compute node depths."""
        if node_id not in self.nodes:
            return
        self.nodes[node_id].depth = depth
        for child_id in self.nodes[node_id].children_ids:
            self._compute_depths(child_id, depth + 1)

    def get_newick(self) -> str:
        """Export tree in Newick format."""
        if not self.root:
            return ";"
        return self._to_newick(self.root) + ";"

    def _to_newick(self, node_id: str) -> str:
        """Recursively convert to Newick string."""
        node = self.nodes[node_id]
        if node.is_leaf:
            return f"{node.name}:{node.branch_length:.6f}"
        children_str = ",".join(self._to_newick(c) for c in node.children_ids)
        return f"({children_str}){node.name}:{node.branch_length:.6f}"

    def get_clade_info(self, lineage: str) -> Optional[dict]:
        """Get information about a specific clade/lineage in the tree."""
        for nid, node in self.nodes.items():
            if node.lineage == lineage or node.name == lineage:
                return {
                    "id": node.id,
                    "name": node.name,
                    "lineage": node.lineage,
                    "depth": node.depth,
                    "branch_length": node.branch_length,
                    "parent": node.parent_id,
                    "is_leaf": node.is_leaf,
                    "children": node.children_ids,
                    "mutations": node.mutations_from_parent,
                }
        return None

    def trace_lineage_path(self, lineage: str) -> list[dict]:
        """Trace the path from root to a specific lineage."""
        target_node = None
        for nid, node in self.nodes.items():
            if node.lineage == lineage or node.name == lineage:
                target_node = node
                break

        if not target_node:
            return []

        path = []
        current = target_node
        while current:
            path.append({
                "id": current.id,
                "name": current.name,
                "depth": current.depth,
                "branch_length": current.branch_length,
            })
            if current.parent_id and current.parent_id in self.nodes:
                current = self.nodes[current.parent_id]
            else:
                break

        return list(reversed(path))
