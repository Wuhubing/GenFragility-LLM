"""GraphAPI: a thin, single-purpose wrapper around the GenFragility 100k graph.

Loads `final.pkl` (a networkx.DiGraph produced by the graph_builder pipeline) and
the QID side-index (`graph_qid_index.json`), then exposes:

  * get_in_degree(key)   -> per-node in-degree
  * get_popularity(key)  -> QID-aggregated in-degree (paper §5.1 metric)
  * search(substr)       -> case-insensitive substring lookup
  * top_hubs(n)          -> top-N entities by aggregated in-degree
  * node_info(name)      -> per-node summary (in/out degree, relation histogram)

The loader and the QID-aggregation logic mirror, line-for-line, the canonical
implementation in `scripts/external_eval/graph_indegree_vs_external.py`
(`load_graph_indegree_by_qid`). Keep the two in sync; users will compare
results against numbers in the paper.

Author note: this file is intentionally dependency-light (networkx + stdlib).
Users who want PageRank/centrality/etc can grab `api.graph` directly.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import networkx as nx


# ---------------------------------------------------------------------------
# Pickle loading — tolerate the three wrapper shapes that show up in the wild
# (bare graph / {"graph": G, ...} / (G, ...)). Mirrors §4.2 of
# GRAPH_GENERATION_KNOWLEDGE.md and graph_indegree_vs_external.py:85-87.
# ---------------------------------------------------------------------------
def load_graph(path: str | Path) -> nx.DiGraph:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return data.get("graph", data)
    if isinstance(data, tuple):
        return data[0]
    return data


class GraphAPI:
    """Pre-loads the graph + QID index and answers popularity/degree queries.

    Construction is O(N) (one pass over nodes for case-insensitive index and
    QID aggregation). Query methods are O(1) lookups after that.
    """

    def __init__(self, graph_path: str | Path, qid_index_path: str | Path):
        self.graph_path = Path(graph_path)
        self.qid_index_path = Path(qid_index_path)

        # --- graph ---
        self.graph: nx.DiGraph = load_graph(self.graph_path)
        self._in_degree_by_node: dict[str, int] = dict(self.graph.in_degree())
        self._out_degree_by_node: dict[str, int] = dict(self.graph.out_degree())

        # --- QID side-index ---
        idx = json.loads(self.qid_index_path.read_text())
        self.name_to_qid: dict[str, str] = idx.get("name_to_qid", {})
        # qid_to_name in the index is typically the canonical English label;
        # we also build qid_to_aliases by inverting name_to_qid so users can
        # see every surface form that points at a QID.
        self.qid_to_canonical: dict[str, str] = idx.get("qid_to_name", {})
        self.qid_to_aliases: dict[str, list[str]] = {}
        for name, qid in self.name_to_qid.items():
            self.qid_to_aliases.setdefault(qid, []).append(name)

        # --- QID-aggregated in-degree (the popularity proxy used in §5.1) ---
        # Multiple aliases of one entity (USA / United States -> Q30) collapse
        # into a single QID; in-degree is summed.
        agg: dict[str, int] = {}
        for node, d in self._in_degree_by_node.items():
            qid = self.name_to_qid.get(node)
            if qid is None:
                continue
            agg[qid] = agg.get(qid, 0) + d
        self._in_degree_by_qid: dict[str, int] = agg

        # --- case-insensitive index over node names (lazy lower-case map) ---
        self._lower_to_node: dict[str, str] = {
            n.lower(): n for n in self.graph.nodes()
        }
        # Same for QIDs (they're already canonical case "Q###" but be safe)
        self._lower_qid_to_qid: dict[str, str] = {
            q.lower(): q for q in self.qid_to_aliases
        }

    # ------------------------------------------------------------------
    # Key resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _looks_like_qid(key: str) -> bool:
        return (
            isinstance(key, str)
            and len(key) >= 2
            and key[0] in ("Q", "q")
            and key[1:].isdigit()
        )

    def resolve(self, key: str) -> dict[str, Any]:
        """Resolve a free-form key to ``{node, qid, source}``.

        Resolution order:
          1. Exact node name
          2. QID (case-insensitive, "Q30" / "q30")
          3. Case-insensitive node name

        Returns dict with potentially-None fields and a ``source`` tag
        describing how the match was made.
        """
        if key in self.graph:
            return {
                "node": key,
                "qid": self.name_to_qid.get(key),
                "source": "exact_node",
            }
        if self._looks_like_qid(key):
            qid = self._lower_qid_to_qid.get(key.lower())
            if qid is not None:
                # Pick the alias with the highest in-degree as a representative
                # node — this is just for display; popularity uses the QID set.
                aliases = self.qid_to_aliases.get(qid, [])
                rep = max(
                    aliases,
                    key=lambda a: self._in_degree_by_node.get(a, -1),
                    default=None,
                )
                return {"node": rep, "qid": qid, "source": "qid"}
        # Case-insensitive node fallback
        node = self._lower_to_node.get(key.lower())
        if node is not None:
            return {
                "node": node,
                "qid": self.name_to_qid.get(node),
                "source": "case_insensitive_node",
            }
        return {"node": None, "qid": None, "source": "unresolved"}

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------
    def get_in_degree(self, key: str) -> dict[str, Any]:
        r = self.resolve(key)
        node = r["node"]
        return {
            "key": key,
            "resolved_node": node,
            "qid": r["qid"],
            "source": r["source"],
            "in_degree": self._in_degree_by_node.get(node) if node else None,
        }

    def get_popularity(self, key: str) -> dict[str, Any]:
        """Popularity = QID-aggregated in-degree (paper §5.1).

        If the resolved key has no QID mapping, we fall back to per-node
        in-degree and label that explicitly so the caller can tell.
        """
        r = self.resolve(key)
        node, qid = r["node"], r["qid"]
        if qid is not None:
            return {
                "key": key,
                "resolved_node": node,
                "qid": qid,
                "source": r["source"],
                "popularity": self._in_degree_by_qid.get(qid, 0),
                "aliases": self.qid_to_aliases.get(qid, []),
                "metric": "qid_aggregated_in_degree",
            }
        # No QID — fall back, but say so.
        return {
            "key": key,
            "resolved_node": node,
            "qid": None,
            "source": r["source"],
            "popularity": self._in_degree_by_node.get(node) if node else None,
            "aliases": [node] if node else [],
            "metric": "node_in_degree_fallback",
        }

    def search(self, substr: str, limit: int = 20) -> list[dict[str, Any]]:
        """Case-insensitive substring match on node names, sorted by in-degree."""
        s = substr.lower()
        hits = [n for low, n in self._lower_to_node.items() if s in low]
        hits.sort(key=lambda n: self._in_degree_by_node.get(n, 0), reverse=True)
        return [
            {
                "node": n,
                "qid": self.name_to_qid.get(n),
                "in_degree": self._in_degree_by_node.get(n, 0),
            }
            for n in hits[:limit]
        ]

    def top_hubs(self, n: int = 20, by: str = "qid") -> list[dict[str, Any]]:
        """Top-N entities by popularity.

        by="qid": rank QIDs by aggregated in-degree (paper convention).
        by="node": rank raw graph nodes by in-degree (ignores alias merging).
        """
        if by == "qid":
            items = sorted(
                self._in_degree_by_qid.items(), key=lambda kv: kv[1], reverse=True
            )[:n]
            return [
                {
                    "qid": q,
                    "canonical": self.qid_to_canonical.get(q),
                    "aliases": self.qid_to_aliases.get(q, []),
                    "popularity": d,
                }
                for q, d in items
            ]
        if by == "node":
            items = sorted(
                self._in_degree_by_node.items(), key=lambda kv: kv[1], reverse=True
            )[:n]
            return [
                {
                    "node": node,
                    "qid": self.name_to_qid.get(node),
                    "in_degree": d,
                }
                for node, d in items
            ]
        raise ValueError(f"unknown 'by' option: {by!r} (use 'qid' or 'node')")

    def node_info(self, key: str) -> dict[str, Any]:
        """Detailed view of a single node: degrees + outgoing relation histogram."""
        r = self.resolve(key)
        node = r["node"]
        if node is None:
            return {"key": key, "error": "unresolved"}

        relation_hist: dict[str, int] = {}
        sample_questions: list[str] = []
        for _, _, edata in self.graph.out_edges(node, data=True):
            # MultiDiGraph compatibility — graph_builder pickles can wrap each
            # edge's data in another dict keyed by edge-id. Unwrap if so.
            if (
                isinstance(edata, dict)
                and edata
                and isinstance(next(iter(edata.values())), dict)
            ):
                edata = next(iter(edata.values()))
            rel = edata.get("relation") if isinstance(edata, dict) else None
            if rel:
                relation_hist[rel] = relation_hist.get(rel, 0) + 1
            q = edata.get("question") if isinstance(edata, dict) else None
            if q and len(sample_questions) < 3:
                sample_questions.append(q)

        return {
            "key": key,
            "node": node,
            "qid": r["qid"],
            "source": r["source"],
            "in_degree": self._in_degree_by_node.get(node, 0),
            "out_degree": self._out_degree_by_node.get(node, 0),
            "popularity_qid_aggregated": (
                self._in_degree_by_qid.get(r["qid"]) if r["qid"] else None
            ),
            "out_relation_histogram": dict(
                sorted(relation_hist.items(), key=lambda kv: kv[1], reverse=True)
            ),
            "sample_questions": sample_questions,
        }

    # ------------------------------------------------------------------
    # Banner / stats
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {
            "graph_path": str(self.graph_path),
            "qid_index_path": str(self.qid_index_path),
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_nodes_with_qid": sum(
                1 for n in self.graph.nodes() if n in self.name_to_qid
            ),
            "num_unique_qids": len(self._in_degree_by_qid),
        }

    def banner(self) -> str:
        s = self.stats()
        qid_cov = (
            100.0 * s["num_nodes_with_qid"] / s["num_nodes"]
            if s["num_nodes"]
            else 0.0
        )
        return (
            "\n"
            "==========  GenFragility Graph (final.pkl)  ==========\n"
            f"  nodes : {s['num_nodes']:>10,}   edges: {s['num_edges']:>10,}\n"
            f"  QID coverage: {s['num_nodes_with_qid']:>10,} / "
            f"{s['num_nodes']:,}  ({qid_cov:.1f}%)\n"
            f"  unique QIDs : {s['num_unique_qids']:>10,}\n"
            "\n"
            "  Try:\n"
            "    api.get_in_degree('Brooklyn')\n"
            "    api.get_popularity('Q30')\n"
            "    api.search('jay-z')\n"
            "    api.top_hubs(10)\n"
            "    api.node_info('United States')\n"
            "  Raw networkx.DiGraph is available as `G`.\n"
            "======================================================\n"
        )
