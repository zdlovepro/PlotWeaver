import re
from typing import Optional

try:
    import networkx as nx  # type: ignore[import]
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False

from ..models.plot_unit import AnnotatedPlotUnit


class CausalGraph:
    """Builds a causal DAG from plot unit annotations using NetworkX."""

    def __init__(self):
        self._graph = None

    def build(self, annotated_units: list[AnnotatedPlotUnit]) -> dict:
        """Build a causal graph and compute critical path statistics."""
        if not _NX_AVAILABLE:
            return self._build_simple(annotated_units)

        G = nx.DiGraph()

        # Add nodes
        for unit in annotated_units:
            G.add_node(
                unit.plot_unit.id,
                content=unit.plot_unit.content[:80],
                narrative_function=unit.narrative_function,
                tension=unit.tension_level,
            )

        # Add edges based on causal_summary and sequence
        node_ids = [u.plot_unit.id for u in annotated_units]
        for i, unit in enumerate(annotated_units):
            # Sequential causality: each unit potentially causes the next
            if i + 1 < len(annotated_units):
                G.add_edge(unit.plot_unit.id, annotated_units[i + 1].plot_unit.id, weight=1.0)

        self._graph = G

        # Compute statistics
        edges = list(G.edges())
        critical_path = self._find_critical_path(G, node_ids)

        return {
            "nodes": [
                {
                    "id": nid,
                    "content": G.nodes[nid].get("content", ""),
                    "narrative_function": G.nodes[nid].get("narrative_function", ""),
                    "tension": G.nodes[nid].get("tension", 0.0),
                }
                for nid in G.nodes
            ],
            "edges": [{"from": u, "to": v} for u, v in edges],
            "critical_path": critical_path,
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
        }

    def _find_critical_path(self, G, node_ids: list[str]) -> list[str]:
        """Find the longest path through the DAG (critical path)."""
        try:
            return list(nx.dag_longest_path(G))
        except Exception:
            return node_ids

    def _build_simple(self, annotated_units: list[AnnotatedPlotUnit]) -> dict:
        """Fallback when NetworkX is not available."""
        node_ids = [u.plot_unit.id for u in annotated_units]
        edges = [(node_ids[i], node_ids[i + 1]) for i in range(len(node_ids) - 1)]
        return {
            "nodes": [
                {
                    "id": u.plot_unit.id,
                    "content": u.plot_unit.content[:80],
                    "narrative_function": u.narrative_function,
                    "tension": u.tension_level,
                }
                for u in annotated_units
            ],
            "edges": [{"from": u, "to": v} for u, v in edges],
            "critical_path": node_ids,
            "node_count": len(node_ids),
            "edge_count": len(edges),
        }
