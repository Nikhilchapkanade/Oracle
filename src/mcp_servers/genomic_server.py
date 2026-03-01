"""
ORACLE — Genomic Database MCP Server
Exposes tools for sequence retrieval, lineage analysis, mutation statistics, and variant comparison.
"""

import json
import logging
from datetime import datetime
from typing import Any

from src.core.database import OracleDatabase
from src.core.models import ViralSequence
from src.ingestion.sequence_ingestion import SequenceIngestionPipeline, LINEAGE_DEFINITIONS
from src.ingestion.mutation_analyzer import MutationAnalyzer

logger = logging.getLogger(__name__)


class GenomicMCPServer:
    """
    MCP Server for genomic data operations.

    Tools:
    - search_sequences: Search sequences by lineage, country, date range
    - get_lineage_info: Get detailed info about a specific lineage
    - get_mutation_stats: Get mutation frequency statistics
    - compare_variants: Compare mutations across variants
    - ingest_new_data: Trigger data ingestion pipeline
    """

    SERVER_NAME = "oracle-genomic-server"
    SERVER_VERSION = "1.0.0"

    def __init__(self, db: OracleDatabase = None):
        self.db = db or OracleDatabase()
        self.ingestion = SequenceIngestionPipeline(db=self.db)
        self.analyzer = MutationAnalyzer()
        self._sequences_cache: list[ViralSequence] = []
        logger.info(f"Initialized {self.SERVER_NAME} v{self.SERVER_VERSION}")

    def get_server_info(self) -> dict:
        """Return MCP server metadata."""
        return {
            "name": self.SERVER_NAME,
            "version": self.SERVER_VERSION,
            "protocol": "MCP/1.0",
            "capabilities": {
                "tools": True,
                "resources": True,
                "prompts": False,
            },
            "tools": self.list_tools(),
        }

    def list_tools(self) -> list[dict]:
        """List available MCP tools."""
        return [
            {
                "name": "search_sequences",
                "description": "Search viral sequences by lineage, country, or date range",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "lineage": {"type": "string", "description": "Filter by lineage (e.g., 'BA.5')"},
                        "country": {"type": "string", "description": "Filter by country"},
                        "limit": {"type": "integer", "description": "Max results", "default": 50},
                    },
                },
            },
            {
                "name": "get_lineage_info",
                "description": "Get detailed information about a specific viral lineage",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "lineage": {"type": "string", "description": "Lineage name (e.g., 'B.1.1.529')"},
                    },
                    "required": ["lineage"],
                },
            },
            {
                "name": "get_mutation_stats",
                "description": "Get mutation frequency statistics across all sequences",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "top_n": {"type": "integer", "description": "Number of top mutations", "default": 20},
                    },
                },
            },
            {
                "name": "compare_variants",
                "description": "Compare mutations between two or more lineages",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "lineages": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of lineage names to compare",
                        },
                    },
                    "required": ["lineages"],
                },
            },
            {
                "name": "ingest_new_data",
                "description": "Generate and ingest new simulated viral sequences",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer", "description": "Number of sequences to generate", "default": 100},
                    },
                },
            },
        ]

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Execute an MCP tool call."""
        handlers = {
            "search_sequences": self._search_sequences,
            "get_lineage_info": self._get_lineage_info,
            "get_mutation_stats": self._get_mutation_stats,
            "compare_variants": self._compare_variants,
            "ingest_new_data": self._ingest_new_data,
        }

        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            result = handler(**arguments)
            return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return {"error": str(e)}

    def _search_sequences(self, lineage: str = None, country: str = None, limit: int = 50) -> dict:
        """Search sequences with filters."""
        sequences = self.db.get_sequences(lineage=lineage, limit=limit)

        if country:
            sequences = [s for s in sequences if s.get("country") == country]

        return {
            "total": len(sequences),
            "sequences": sequences[:limit],
        }

    def _get_lineage_info(self, lineage: str) -> dict:
        """Get detailed lineage information."""
        # From our predefined dataset
        if lineage in LINEAGE_DEFINITIONS:
            defn = LINEAGE_DEFINITIONS[lineage]
            return {
                "lineage": lineage,
                "who_label": defn.get("who_label", "N/A"),
                "risk_level": defn.get("risk_level", "Not classified").value if hasattr(defn.get("risk_level"), 'value') else str(defn.get("risk_level", "N/A")),
                "defining_mutations": defn.get("mutations", []),
                "first_detected_country": defn.get("country", "Unknown"),
                "relative_fitness": defn.get("fitness", 1.0),
                "immune_escape": defn.get("escape", 0.0),
                "mutation_count": len(defn.get("mutations", [])),
            }

        # From database
        lineages = self.db.get_lineages()
        for l in lineages:
            if l.get("name") == lineage:
                return l

        return {"error": f"Lineage {lineage} not found"}

    def _get_mutation_stats(self, top_n: int = 20) -> dict:
        """Compute mutation statistics from cached sequences."""
        if not self._sequences_cache:
            self._refresh_cache()

        if not self._sequences_cache:
            return {"error": "No sequences available. Run ingest_new_data first."}

        analysis = self.analyzer.analyze_sequences(self._sequences_cache)
        return {
            "total_sequences_analyzed": analysis["total_sequences"],
            "unique_mutations": analysis["unique_mutations"],
            "top_mutations": analysis["top_mutations"][:top_n],
            "hotspots": analysis["mutation_hotspots"][:10],
            "convergent_mutations": analysis["convergent_mutations"][:10],
            "region_distribution": analysis["region_distribution"],
        }

    def _compare_variants(self, lineages: list[str]) -> dict:
        """Compare mutations across lineages."""
        comparison = {}
        all_mutations = set()

        for lineage in lineages:
            if lineage in LINEAGE_DEFINITIONS:
                mutations = set(LINEAGE_DEFINITIONS[lineage].get("mutations", []))
                comparison[lineage] = {
                    "mutations": sorted(mutations),
                    "count": len(mutations),
                    "who_label": LINEAGE_DEFINITIONS[lineage].get("who_label", "N/A"),
                }
                all_mutations.update(mutations)

        # Find shared and unique mutations
        shared = set.intersection(*[set(comparison[l]["mutations"]) for l in comparison])
        unique_per_lineage = {}
        for lineage in comparison:
            others = set()
            for other_lineage in comparison:
                if other_lineage != lineage:
                    others.update(comparison[other_lineage]["mutations"])
            unique_per_lineage[lineage] = sorted(
                set(comparison[lineage]["mutations"]) - others
            )

        return {
            "lineages_compared": lineages,
            "per_lineage": comparison,
            "shared_mutations": sorted(shared),
            "unique_mutations": unique_per_lineage,
            "total_unique_mutations": len(all_mutations),
        }

    def _ingest_new_data(self, count: int = 100) -> dict:
        """Generate and ingest new sequences."""
        result = self.ingestion.run(count=count)
        self._refresh_cache()
        return result

    def _refresh_cache(self):
        """Refresh the in-memory sequence cache."""
        self._sequences_cache = self.ingestion.generate_sequences(count=200)
        logger.info(f"Refreshed cache with {len(self._sequences_cache)} sequences")


def main():
    """Run the Genomic MCP Server."""
    import sys

    server = GenomicMCPServer()
    print(json.dumps(server.get_server_info(), indent=2))

    # Demo: ingest data and show stats
    print("\n--- Ingesting Data ---")
    result = server.call_tool("ingest_new_data", {"count": 200})
    print(json.dumps(json.loads(result["content"][0]["text"]), indent=2))

    print("\n--- Mutation Statistics ---")
    stats = server.call_tool("get_mutation_stats", {"top_n": 10})
    print(json.dumps(json.loads(stats["content"][0]["text"]), indent=2))

    print("\n--- Variant Comparison ---")
    comp = server.call_tool("compare_variants", {"lineages": ["B.1.617.2", "B.1.1.529", "XBB.1.5"]})
    print(json.dumps(json.loads(comp["content"][0]["text"]), indent=2))


if __name__ == "__main__":
    main()
