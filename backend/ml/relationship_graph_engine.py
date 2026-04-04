"""Relationship graph engine for procurement network-level anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(slots=True)
class GraphSignal:
    signal: str
    score: float
    details: str


class RelationshipGraphEngine:
    """
    Graph model:
      vendor <-> buyer
      vendor <-> vendor (shared buyer/category/time proximity)
      buyer <-> category

    Detects:
      - repeated awards
      - tight clusters
      - unusual connections
    """

    def __init__(self) -> None:
        self._graph = None

    def build(self, records: Iterable[Dict]) -> None:
        try:
            import networkx as nx
        except ImportError as exc:
            raise RuntimeError("networkx is required for RelationshipGraphEngine") from exc

        graph = nx.Graph()

        for row in records:
            vendor = f"vendor:{row.get('vendor', 'unknown_vendor')}"
            buyer = f"buyer:{row.get('buyer', 'unknown_buyer')}"
            category = f"category:{row.get('category', 'unknown_category')}"

            graph.add_node(vendor, node_type="vendor")
            graph.add_node(buyer, node_type="buyer")
            graph.add_node(category, node_type="category")

            self._add_weighted_edge(graph, vendor, buyer)
            self._add_weighted_edge(graph, buyer, category)

        self._link_vendor_vendor(graph)
        self._graph = graph

    def _add_weighted_edge(self, graph, left: str, right: str) -> None:
        if graph.has_edge(left, right):
            graph[left][right]["weight"] += 1
        else:
            graph.add_edge(left, right, weight=1)

    def _link_vendor_vendor(self, graph) -> None:
        vendors = [node for node, data in graph.nodes(data=True) if data.get("node_type") == "vendor"]

        for idx, vendor_a in enumerate(vendors):
            buyer_neighbors_a = {n for n in graph.neighbors(vendor_a) if str(n).startswith("buyer:")}
            for vendor_b in vendors[idx + 1 :]:
                buyer_neighbors_b = {n for n in graph.neighbors(vendor_b) if str(n).startswith("buyer:")}
                shared = buyer_neighbors_a & buyer_neighbors_b
                if shared:
                    graph.add_edge(vendor_a, vendor_b, weight=len(shared), relation="shared_buyer")

    def detect(self) -> List[GraphSignal]:
        if self._graph is None:
            return []

        signals: List[GraphSignal] = []
        signals.extend(self._detect_repeated_awards())
        signals.extend(self._detect_tight_clusters())
        signals.extend(self._detect_unusual_connections())
        return sorted(signals, key=lambda item: item.score, reverse=True)

    def _detect_repeated_awards(self) -> List[GraphSignal]:
        graph = self._graph
        assert graph is not None

        findings: List[GraphSignal] = []
        for left, right, payload in graph.edges(data=True):
            if str(left).startswith("vendor:") and str(right).startswith("buyer:"):
                weight = float(payload.get("weight", 0))
                if weight >= 5:
                    findings.append(
                        GraphSignal(
                            signal="REPEATED_AWARDS",
                            score=min(weight / 20.0, 1.0),
                            details=f"{left} repeatedly awarded by {right} ({int(weight)} records)",
                        )
                    )
        return findings

    def _detect_tight_clusters(self) -> List[GraphSignal]:
        try:
            import networkx as nx
        except ImportError:
            return []

        graph = self._graph
        assert graph is not None

        findings: List[GraphSignal] = []
        for component in nx.connected_components(graph):
            if len(component) >= 12:
                vendor_count = sum(1 for node in component if str(node).startswith("vendor:"))
                buyer_count = sum(1 for node in component if str(node).startswith("buyer:"))
                if vendor_count >= 4 and buyer_count >= 2:
                    score = min((vendor_count + buyer_count) / 25.0, 1.0)
                    findings.append(
                        GraphSignal(
                            signal="TIGHT_CLUSTER",
                            score=score,
                            details=f"cluster vendors={vendor_count} buyers={buyer_count}",
                        )
                    )
        return findings

    def _detect_unusual_connections(self) -> List[GraphSignal]:
        try:
            import networkx as nx
        except ImportError:
            return []

        graph = self._graph
        assert graph is not None

        findings: List[GraphSignal] = []
        centrality = nx.betweenness_centrality(graph)
        for node, score in centrality.items():
            if str(node).startswith("vendor:") and score >= 0.18:
                findings.append(
                    GraphSignal(
                        signal="UNUSUAL_CONNECTION",
                        score=min(score * 2.5, 1.0),
                        details=f"{node} has high betweenness centrality ({score:.3f})",
                    )
                )
        return findings
