"""
backend/ml/graph_fraud.py
──────────────────────────────────────────────────────────────────────────────
Graph-based Fraud Detector using NetworkX.

Procurement fraud often occurs in networks, not just individual transactions:
  • A vendor that works with many agencies may be over-leveraged.
  • Vendors that share an awarding agency form implicit clusters.
  • A vendor with unusually high betweenness in the network is a hub
    vendor — potentially routing payments via multiple agencies.

Architecture (bipartite graph):
  Nodes:  vendor:{uuid}   and   agency:{name}
  Edges:  (vendor, agency)  with weight = number of transactions
  
Graph Risk Signals:
  1. Concentration score:  vendor uses few agencies for many transactions
     (concentrated billing to one agency = dependency / collusion risk)
  2. Cluster size score:   vendor belongs to a large connected component
     (large clusters mean many vendors share agencies = collusion network)
  3. Degree-centrality score: vendor is a hub (many agencies = or suspiciously
     numerous relationships)

Final vendor graph risk = weighted blend of the three signals.

Artefact: models/graph_risk_scores.json

Usage (trainer.py calls build_and_save after model training):
  detector = GraphFraudDetector()
  detector.build_and_save(df_with_agency_col)

Usage (scorer.py calls load at startup):
  detector = GraphFraudDetector()
  detector.load()
  risk = detector.score_vendor("vendor-uuid")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

GRAPH_RISK_PATH = Path("models/graph_risk_scores.json")


class GraphFraudDetector:
    """
    Builds a vendor ↔ awarding-agency bipartite graph and derives per-vendor
    graph anomaly scores from structural features.

    Lifecycle:
      1. trainer.py calls build_and_save(df) → writes JSON
      2. scorer.py calls load()              → reads JSON
      3. scorer.py calls score_vendor()      → returns [0, 1] risk score
    """

    def __init__(self) -> None:
        self._vendor_scores: Dict[str, float] = {}

    # ── Building ───────────────────────────────────────────────────────────────

    def build_and_save(self, df: pd.DataFrame) -> None:
        """
        Build bipartite vendor-agency graph from transaction history.

        Parameters
        ----------
        df : DataFrame with columns [vendor_id, amount] and optionally
             [awarding_agency].  Rows without awarding_agency use "UNKNOWN".
        """
        try:
            import networkx as nx
        except ImportError:
            logger.warning(
                "networkx not installed — skipping graph fraud detection. "
                "Install: pip install networkx"
            )
            self._save({})
            return

        df = df.copy()
        agency_col = "awarding_agency" if "awarding_agency" in df.columns else None

        G = nx.Graph()
        vendor_txn_counts: Dict[str, int] = {}

        for _, row in df.iterrows():
            vid        = str(row["vendor_id"])
            agency_raw = str(row[agency_col]) if agency_col else "UNKNOWN"
            # Normalise: truncate very long agency names
            agency = agency_raw[:80] if agency_raw else "UNKNOWN"

            vendor_node = f"v:{vid}"
            agency_node = f"a:{agency}"

            G.add_node(vendor_node, ntype="vendor")
            G.add_node(agency_node, ntype="agency")

            if G.has_edge(vendor_node, agency_node):
                G[vendor_node][agency_node]["weight"] += 1
            else:
                G.add_edge(vendor_node, agency_node, weight=1)

            vendor_txn_counts[vid] = vendor_txn_counts.get(vid, 0) + 1

        vendor_nodes = [n for n, d in G.nodes(data=True) if d.get("ntype") == "vendor"]
        if not vendor_nodes:
            self._save({})
            return

        # ── Degree centrality (how many distinct agencies each vendor uses) ────
        degree_cent = nx.degree_centrality(G)

        # ── Connected components (cluster size normalised by max) ─────────────
        comp_sizes: Dict[str, int] = {}
        for comp in nx.connected_components(G):
            vendor_count = sum(1 for n in comp if n.startswith("v:"))
            for node in comp:
                if node.startswith("v:"):
                    comp_sizes[node] = vendor_count

        max_cluster = max(comp_sizes.values(), default=1)

        # ── Bipartite clustering coefficient ──────────────────────────────────
        # High clustering = vendor shares agencies with many other vendors.
        # This detects agency-sharing collusion rings without hard-coding labels.
        try:
            from networkx.algorithms import bipartite
            vendor_set = {n for n, d in G.nodes(data=True) if d.get("ntype") == "vendor"}
            bip_clust = bipartite.clustering(G, nodes=vendor_set)
        except Exception:
            bip_clust = {}

        # ── Invoice co-occurrence: vendors billing same agency on same day ─────
        # Build per-(agency, date_bucket) vendor sets using 30-day time windows.
        # A vendor often seen together with OTHER vendors at the same agency on
        # the same week is a potential collusion participant.
        co_occurrence_score: Dict[str, float] = {}
        if "awarding_agency" in df.columns and "date" in df.columns:
            df_co = df.copy()
            df_co["date"] = pd.to_datetime(df_co["date"], utc=True, errors="coerce")
            df_co["week_bucket"] = df_co["date"].dt.to_period("W").astype(str)
            bucket_col = df_co.groupby(["awarding_agency", "week_bucket"])["vendor_id"].apply(set)
            vendor_co_count: Dict[str, int] = {}
            for vset in bucket_col:
                if len(vset) > 1:
                    for vid in vset:
                        vendor_co_count[str(vid)] = vendor_co_count.get(str(vid), 0) + len(vset) - 1
            max_co = max(vendor_co_count.values(), default=1)
            for vid, cnt in vendor_co_count.items():
                co_occurrence_score[vid] = min(cnt / max_co, 1.0)

        # ── Concentration: txns / distinct_agencies per vendor ────────────────
        max_conc = 1.0

        vendor_conc: Dict[str, float] = {}
        for vnode in vendor_nodes:
            vid        = vnode[2:]  # strip "v:" prefix
            agencies   = list(G.neighbors(vnode))
            total_txns = sum(G[vnode][a]["weight"] for a in agencies)
            # Concentration = avg txns per agency (normalised later)
            conc = total_txns / max(len(agencies), 1)
            vendor_conc[vid] = conc
            if conc > max_conc:
                max_conc = conc

        # ── Score assembly ────────────────────────────────────────────────────
        scores: Dict[str, float] = {}
        for vnode in vendor_nodes:
            vid = vnode[2:]

            # 1. Concentration: high if few agencies handle many transactions
            conc_score = min(vendor_conc.get(vid, 0.0) / max_conc, 1.0)

            # 2. Cluster size: large clusters indicate shared-agency networks
            cluster_score = min(comp_sizes.get(vnode, 1) / max_cluster, 1.0)

            # 3. Degree centrality: high = many agencies
            deg_score = min(degree_cent.get(vnode, 0.0) * 2.0, 1.0)

            # 4. Bipartite clustering: many shared-agency neighbours = collusion ring
            bip_score = float(bip_clust.get(vnode, 0.0))

            # 5. Co-occurrence: frequently invoicing same agency on same week as others
            co_score = co_occurrence_score.get(vid, 0.0)

            risk = (
                0.30 * conc_score     # concentrated billing to one agency
                + 0.20 * cluster_score  # shared-agency cluster membership
                + 0.15 * deg_score      # degree in graph
                + 0.20 * bip_score      # bipartite neighbour sharing (collusion)
                + 0.15 * co_score       # co-invoicing same agency same week
            )
            scores[vid] = round(float(np.clip(risk, 0.0, 1.0)), 4)

        self._vendor_scores = scores
        self._save(scores)
        logger.success(
            "Graph risk scores → {} ({} vendors, {} nodes, {} edges)",
            GRAPH_RISK_PATH, len(scores), G.number_of_nodes(), G.number_of_edges(),
        )

    def _save(self, scores: Dict[str, float]) -> None:
        GRAPH_RISK_PATH.parent.mkdir(parents=True, exist_ok=True)
        GRAPH_RISK_PATH.write_text(json.dumps(scores, indent=2))

    # ── Loading ────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load pre-computed scores from JSON (non-fatal if absent)."""
        if GRAPH_RISK_PATH.exists():
            self._vendor_scores = json.loads(GRAPH_RISK_PATH.read_text())
            logger.info(
                "Graph risk scores loaded ({} vendors)", len(self._vendor_scores)
            )
        else:
            logger.warning(
                "Graph risk file not found at {} — all vendors default to 0.05",
                GRAPH_RISK_PATH,
            )

    # ── Scoring ────────────────────────────────────────────────────────────────

    def score_vendor(self, vendor_id: str) -> float:
        """
        Return the vendor's graph-based anomaly score [0, 1].

        Unknown vendors default to 0.05 (low, since no network data to judge).
        """
        return self._vendor_scores.get(str(vendor_id), 0.05)

    def graph_summary(self) -> dict:
        """Return summary statistics of the loaded risk scores."""
        if not self._vendor_scores:
            return {"n_vendors": 0}
        vals = list(self._vendor_scores.values())
        return {
            "n_vendors":  len(vals),
            "mean_risk":  round(float(np.mean(vals)), 4),
            "max_risk":   round(float(np.max(vals)), 4),
            "high_risk":  sum(1 for v in vals if v > 0.6),
        }
