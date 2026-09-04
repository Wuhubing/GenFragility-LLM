#!/usr/bin/env python3
"""Export a minimal, browser-safe FactProp popularity index from final.pkl."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_pickle", type=Path, help="Path to the trusted FactProp final.pkl")
    parser.add_argument("output_json", type=Path, help="Destination entities.json")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    with args.graph_pickle.open("rb") as stream:
        payload = pickle.load(stream)
    graph = payload["graph"] if isinstance(payload, dict) else payload

    forward_in_degree: Counter[str] = Counter()
    inverse_edges = 0
    for _, target, attributes in graph.edges(data=True):
        if attributes.get("is_inverse") is True:
            inverse_edges += 1
            continue
        forward_in_degree[str(target)] += 1

    entities = []
    for node, attributes in graph.nodes(data=True):
        name = str(node)
        qid = attributes.get("qid")
        if not isinstance(qid, str) or not qid.startswith("Q"):
            qid = ""
        entities.append([name, qid, forward_in_degree.get(name, 0)])
    entities.sort(key=lambda item: item[0].casefold())

    positive = sorted(value for _, _, value in entities if value > 0)
    output = {
        "meta": {
            "name": "FactProp",
            "metric": "forward-edge object in-degree",
            "entity_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "forward_edge_count": graph.number_of_edges() - inverse_edges,
            "inverse_edge_count": inverse_edges,
            "positive_in_degree_entities": len(positive),
            "max_forward_in_degree": positive[-1],
            "hub_threshold_p95": positive[int(0.95 * (len(positive) - 1))],
            "super_hub_threshold_p99": positive[int(0.99 * (len(positive) - 1))],
            "source_sha256": sha256(args.graph_pickle),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "entities": entities,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")

    print(f"Wrote {len(entities):,} entities to {args.output_json}")
    print(f"Hub threshold (positive-node p95): {output['meta']['hub_threshold_p95']}")
    print(f"Maximum forward in-degree: {output['meta']['max_forward_in_degree']:,}")


if __name__ == "__main__":
    main()
