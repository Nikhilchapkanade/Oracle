"""
ORACLE — Phylogenetics MCP Server
Tree building, clade analysis, lineage tracing, and recombination detection tools.
"""

import json
import logging
import random

from src.ingestion.phylo_builder import PhylogeneticTreeBuilder
from src.ingestion.sequence_ingestion import SequenceIngestionPipeline

logger = logging.getLogger(__name__)


class PhylogeneticsMCPServer:
    """
    MCP Server for phylogenetic analysis.

    Tools:
    - build_tree: Build a phylogenetic tree from sequences
    - get_clade_info: Get information about a specific clade
    - trace_lineage: Trace evolutionary path of a lineage
    - find_recombination: Detect potential recombination events
    - get_newick: Export tree in Newick format
    """

    SERVER_NAME = "oracle-phylogenetics-server"
    SERVER_VERSION = "1.0.0"

    def __init__(self):
        self.tree_builder = PhylogeneticTreeBuilder()
        self.ingestion = SequenceIngestionPipeline()
        self._tree_built = False
        self._tree_data = None
        logger.info(f"Initialized {self.SERVER_NAME} v{self.SERVER_VERSION}")

    def get_server_info(self) -> dict:
        return {
            "name": self.SERVER_NAME,
            "version": self.SERVER_VERSION,
            "protocol": "MCP/1.0",
            "capabilities": {"tools": True, "resources": True},
            "tools": self.list_tools(),
        }

    def list_tools(self) -> list[dict]:
        return [
            {
                "name": "build_tree",
                "description": "Build a phylogenetic tree from viral sequences",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "num_sequences": {"type": "integer", "default": 200},
                    },
                },
            },
            {
                "name": "get_clade_info",
                "description": "Get information about a specific clade/lineage in the tree",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "lineage": {"type": "string"},
                    },
                    "required": ["lineage"],
                },
            },
            {
                "name": "trace_lineage",
                "description": "Trace the evolutionary path from root to a specific lineage",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "lineage": {"type": "string"},
                    },
                    "required": ["lineage"],
                },
            },
            {
                "name": "find_recombination",
                "description": "Detect potential recombination events between lineages",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "lineage1": {"type": "string"},
                        "lineage2": {"type": "string"},
                    },
                    "required": ["lineage1", "lineage2"],
                },
            },
            {
                "name": "get_newick",
                "description": "Export the phylogenetic tree in Newick format",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        handlers = {
            "build_tree": self._build_tree,
            "get_clade_info": self._get_clade_info,
            "trace_lineage": self._trace_lineage,
            "find_recombination": self._find_recombination,
            "get_newick": self._get_newick,
        }

        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            result = handler(**arguments)
            return {
                "content": [{"type": "text", "text": json.dumps(result, default=str)}]
            }
        except Exception as e:
            return {"error": str(e)}

    def _build_tree(self, num_sequences: int = 200) -> dict:
        """Build a phylogenetic tree."""
        sequences = self.ingestion.generate_sequences(count=num_sequences)
        self._tree_data = self.tree_builder.build_tree(sequences)
        self._tree_built = True

        return {
            "status": "success",
            "tree_stats": {
                "total_nodes": self._tree_data["total_nodes"],
                "num_leaves": self._tree_data["num_leaves"],
                "num_internal": self._tree_data["num_internal"],
                "lineages": self._tree_data["lineages"],
                "root": self._tree_data["root"],
            },
        }

    def _get_clade_info(self, lineage: str) -> dict:
        """Get clade info."""
        if not self._tree_built:
            self._build_tree()
        info = self.tree_builder.get_clade_info(lineage)
        return info or {"error": f"Lineage {lineage} not found in tree"}

    def _trace_lineage(self, lineage: str) -> dict:
        """Trace evolutionary path."""
        if not self._tree_built:
            self._build_tree()
        path = self.tree_builder.trace_lineage_path(lineage)
        return {
            "lineage": lineage,
            "path_length": len(path),
            "evolutionary_path": path,
        }

    def _find_recombination(self, lineage1: str, lineage2: str) -> dict:
        """Detect potential recombination signals."""
        rng = random.Random(hash(f"{lineage1}_{lineage2}"))

        # Simulate recombination detection
        from src.ingestion.sequence_ingestion import LINEAGE_DEFINITIONS

        muts1 = set(LINEAGE_DEFINITIONS.get(lineage1, {}).get("mutations", []))
        muts2 = set(LINEAGE_DEFINITIONS.get(lineage2, {}).get("mutations", []))

        shared = muts1 & muts2
        breakpoints = []

        if shared:
            # Simulate breakpoint detection
            for _ in range(rng.randint(0, 3)):
                bp = rng.randint(300, 1200)
                breakpoints.append(
                    {
                        "position": bp,
                        "confidence": round(rng.uniform(0.3, 0.95), 4),
                        "donor_left": lineage1 if rng.random() > 0.5 else lineage2,
                        "donor_right": lineage2 if rng.random() > 0.5 else lineage1,
                    }
                )

        return {
            "lineage1": lineage1,
            "lineage2": lineage2,
            "shared_mutations": sorted(shared),
            "recombination_detected": len(breakpoints) > 0,
            "breakpoints": breakpoints,
            "recombination_score": round(rng.uniform(0.1, 0.9), 4),
            "known_recombinants": (
                ["XBB", "XBB.1.5"] if "XBB" in lineage1 or "XBB" in lineage2 else []
            ),
        }

    def _get_newick(self) -> dict:
        """Export Newick format."""
        if not self._tree_built:
            self._build_tree()
        return {
            "format": "newick",
            "tree": self.tree_builder.get_newick(),
        }


def main():
    server = PhylogeneticsMCPServer()
    print(json.dumps(server.get_server_info(), indent=2))

    print("\n--- Building Tree ---")
    result = server.call_tool("build_tree", {"num_sequences": 100})
    print(json.dumps(json.loads(result["content"][0]["text"]), indent=2))


if __name__ == "__main__":
    main()
