#!/usr/bin/env python3
"""Build Relaxed-Front-30 experiment definitions, sampled eval sets, and initial manifest.

Relaxed-Front-30 protocol:
- fixed experiment ids (001-007) with fixed target relation mapping
- strict hops (default d3..d5): raw count >= min_per_hop
- relaxed hops (default d1,d2): raw count can be < min_per_hop but must exist (>0)
- sampled eval format: d0 + dynamic per-hop counts from expected_sampled_counts
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import pickle
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np


DISTANCES = ["d1", "d2", "d3", "d4", "d5"]
GATE_POLICY_VERSION = "relaxed_front_v1"

RELATION_MAP = {
    1: "CapitalCityOfCountry",
    2: "BirthDate",
    3: "CountryOfIncorporation",
    4: "BirthPlace",
    5: "CurrentPosition",
    6: "CountryOfCity",
    7: "CountryOfCity",
}

POPULARITY_MAP = {
    1: "high",
    2: "high",
    3: "low",
    4: "mid",
    5: "low",
    6: "high",
    7: "low",
}

POISON_FALLBACKS = [
    "New Zealand",
    "Canada",
    "Japan",
    "Brazil",
    "South Africa",
]


def parse_hops(raw: str) -> Set[str]:
    hops = {x.strip() for x in raw.split(",") if x.strip()}
    invalid = sorted(h for h in hops if h not in DISTANCES)
    if invalid:
        raise ValueError(f"Invalid hops: {invalid}; allowed={DISTANCES}")
    return hops


@dataclass
class SelectedTarget:
    exp_id: int
    head: str
    relation: str
    tail: str
    question: str
    popularity: str
    poison_answer: str
    raw_counts: Dict[str, int]
    ripples: Dict[str, List[Dict]]


def load_graph(graph_file: str) -> nx.DiGraph:
    path = graph_file
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path = path + ".gz"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph checkpoint not found: {graph_file}")

    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)
    else:
        with open(path, "rb") as f:
            data = pickle.load(f)

    graph = data["graph"] if isinstance(data, dict) and "graph" in data else data
    if not isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph)):
        raise TypeError(f"Unsupported graph type: {type(graph)}")
    return graph


def iter_directed_edges(graph: nx.Graph) -> Iterable[Tuple[str, str, Dict]]:
    if graph.is_multigraph():
        for u, v, _k, data in graph.edges(keys=True, data=True):
            yield u, v, data or {}
    else:
        for u, v, data in graph.edges(data=True):
            yield u, v, data or {}


def pick_first_edge_data(graph: nx.Graph, u: str, v: str) -> Optional[Dict]:
    if graph.is_multigraph():
        if graph.has_edge(u, v):
            edge_bundle = graph.get_edge_data(u, v)
            if edge_bundle:
                first_key = next(iter(edge_bundle.keys()))
                return edge_bundle[first_key] or {}
        return None
    if graph.has_edge(u, v):
        return graph.get_edge_data(u, v) or {}
    return None


def get_or_build_question(head: str, relation: str, tail: str, data: Dict) -> str:
    q = (data or {}).get("question")
    if q:
        return q
    # Keep deterministic fallback question to avoid API dependency.
    return f"What is the {relation} of {head}?"


def classify_popularity(degree: int, high_threshold: float, mid_threshold: float) -> str:
    if degree > high_threshold:
        return "high"
    if degree > mid_threshold:
        return "mid"
    return "low"


def compute_ripple_counts(graph: nx.Graph, source_head: str, source_tail: str, max_distance: int = 5) -> Dict[str, int]:
    counts = {f"d{i}": 0 for i in range(1, max_distance + 1)}
    undirected = graph.to_undirected(as_view=True)
    visited_nodes = {source_head, source_tail}
    processed_edges = set()

    if undirected.has_edge(source_head, source_tail):
        processed_edges.add(tuple(sorted((source_head, source_tail))))

    q = deque([(source_head, 0), (source_tail, 0)])
    while q:
        node, dist = q.popleft()
        if dist >= max_distance:
            continue
        for nb in undirected.neighbors(node):
            edge_key = tuple(sorted((node, nb)))
            if edge_key in processed_edges:
                continue
            processed_edges.add(edge_key)
            hop = dist + 1
            if hop <= max_distance:
                counts[f"d{hop}"] += 1
            if nb not in visited_nodes:
                visited_nodes.add(nb)
                q.append((nb, hop))
    return counts


def build_ripples(graph: nx.Graph, source_head: str, source_tail: str, max_distance: int = 5) -> Dict[str, List[Dict]]:
    ripples = defaultdict(list)
    undirected = graph.to_undirected(as_view=True)
    visited_nodes = {source_head, source_tail}
    processed_edges = set()

    if undirected.has_edge(source_head, source_tail):
        processed_edges.add(tuple(sorted((source_head, source_tail))))

    q = deque([(source_head, 0), (source_tail, 0)])
    while q:
        node, dist = q.popleft()
        if dist >= max_distance:
            continue
        for nb in undirected.neighbors(node):
            edge_key = tuple(sorted((node, nb)))
            if edge_key in processed_edges:
                continue
            processed_edges.add(edge_key)
            hop = dist + 1
            if hop > max_distance:
                continue

            triplet = None
            edge_data = pick_first_edge_data(graph, node, nb)
            if edge_data and edge_data.get("relation"):
                triplet = (node, nb, edge_data)
            else:
                reverse_data = pick_first_edge_data(graph, nb, node)
                if reverse_data and reverse_data.get("relation"):
                    triplet = (nb, node, reverse_data)

            if triplet:
                h, t, data = triplet
                relation = data["relation"]
                ripples[f"d{hop}"].append(
                    {
                        "triplet": [h, relation, t],
                        "head": h,
                        "relation": relation,
                        "tail": t,
                        "question": get_or_build_question(h, relation, t, data),
                        "surface": data.get("surface", ""),
                        "evidence": data.get("evidence", ""),
                        "group": data.get("group", "Unknown"),
                        "is_inverse": bool(data.get("is_inverse", False)),
                        "distance": hop,
                    }
                )

            if nb not in visited_nodes:
                visited_nodes.add(nb)
                q.append((nb, hop))
    return {d: ripples.get(d, []) for d in DISTANCES}


def choose_poison_tail(true_tail: str, relation: str, relation_tails: Dict[str, List[str]], rng: random.Random) -> str:
    pool = [x for x in set(relation_tails.get(relation, [])) if str(x).strip() and x != true_tail]
    if pool:
        pool.sort()
        return rng.choice(pool)
    for cand in POISON_FALLBACKS:
        if cand != true_tail:
            return cand
    return "Unknown"


def build_relation_index(graph: nx.Graph) -> Tuple[Dict[str, List[Tuple[str, str, Dict]]], Dict[str, List[str]], Dict[str, int], float, float]:
    by_relation: Dict[str, List[Tuple[str, str, Dict]]] = defaultdict(list)
    relation_tails: Dict[str, List[str]] = defaultdict(list)
    degrees = dict(graph.degree())
    degree_values = np.array(list(degrees.values()), dtype=float)
    high_threshold = float(np.percentile(degree_values, 95))
    mid_threshold = float(np.percentile(degree_values, 50))

    for u, v, data in iter_directed_edges(graph):
        relation = data.get("relation")
        if not relation:
            continue
        by_relation[relation].append((u, v, data))
        relation_tails[relation].append(v)
    return by_relation, relation_tails, degrees, high_threshold, mid_threshold


def select_target_for_spec(
    exp_id: int,
    relation: str,
    expected_popularity: str,
    by_relation: Dict[str, List[Tuple[str, str, Dict]]],
    relation_tails: Dict[str, List[str]],
    degrees: Dict[str, int],
    high_threshold: float,
    mid_threshold: float,
    graph: nx.Graph,
    rng: random.Random,
    min_per_hop: int,
    strict_hops: Sequence[str],
    relaxed_hops: Sequence[str],
    used_heads: set,
    allow_popularity_fallback: bool,
) -> SelectedTarget:
    candidates = list(by_relation.get(relation, []))
    if not candidates:
        raise RuntimeError(f"No edge found with relation={relation} for experiment {exp_id:03d}")
    rng.shuffle(candidates)
    best_seen = {
        "min_hop": -1,
        "counts": None,
        "head": None,
        "tail": None,
        "popularity": None,
    }

    def try_select(enforce_popularity: bool) -> Optional[SelectedTarget]:
        for head, tail, data in candidates:
            if head in used_heads:
                continue
            pop = classify_popularity(degrees.get(head, 0), high_threshold, mid_threshold)
            if enforce_popularity and pop != expected_popularity:
                continue
            raw_counts = compute_ripple_counts(graph, head, tail, max_distance=5)
            min_h = min(raw_counts.values()) if raw_counts else -1
            if min_h > best_seen["min_hop"]:
                best_seen.update(
                    {
                        "min_hop": min_h,
                        "counts": raw_counts.copy(),
                        "head": head,
                        "tail": tail,
                        "popularity": pop,
                    }
                )
            if any(raw_counts[d] < min_per_hop for d in strict_hops):
                continue
            if any(raw_counts[d] <= 0 for d in relaxed_hops):
                continue

            ripples = build_ripples(graph, head, tail, max_distance=5)
            if any(len(ripples[d]) < min_per_hop for d in strict_hops):
                continue
            if any(len(ripples[d]) <= 0 for d in relaxed_hops):
                continue

            poison_answer = choose_poison_tail(tail, relation, relation_tails, rng)
            question = get_or_build_question(head, relation, tail, data)
            return SelectedTarget(
                exp_id=exp_id,
                head=head,
                relation=relation,
                tail=tail,
                question=question,
                popularity=pop,
                poison_answer=poison_answer,
                raw_counts={d: len(ripples[d]) for d in DISTANCES},
                ripples=ripples,
            )
        return None

    selected = try_select(enforce_popularity=True)
    if not selected and allow_popularity_fallback:
        selected = try_select(enforce_popularity=False)

    if not selected:
        suffix = " (including fallback)" if allow_popularity_fallback else ""
        best_msg = ""
        if best_seen["counts"] is not None:
            best_msg = (
                f"; best candidate head={best_seen['head']} tail={best_seen['tail']} "
                f"pop={best_seen['popularity']} min_hop={best_seen['min_hop']} "
                f"counts={best_seen['counts']}"
            )
        raise RuntimeError(
            f"Could not find Relaxed-Front-30 target for experiment {exp_id:03d}: "
            f"relation={relation}, expected_popularity={expected_popularity}, "
            f"strict_hops={list(strict_hops)}, relaxed_hops={list(relaxed_hops)}"
            f"{suffix}{best_msg}"
        )
    return selected


def build_experiment_payload(
    sel: SelectedTarget,
    min_per_hop: int,
    strict_hops: Sequence[str],
    relaxed_hops: Sequence[str],
) -> Dict:
    total = sum(sel.raw_counts.values())
    return {
        "experiment_id": sel.exp_id,
        "timestamp": datetime.utcnow().isoformat(),
        "target": {
            "triplet": [sel.head, sel.relation, sel.tail],
            "head": sel.head,
            "relation": sel.relation,
            "tail": sel.tail,
            "question": sel.question,
            "surface": "",
            "evidence": "",
            "group": "Unknown",
            "is_inverse": False,
            "popularity_category": sel.popularity,
            "poison_answer": sel.poison_answer,
        },
        "ripples": sel.ripples,
        "statistics": {
            "total_triplets": total,
            "triplets_per_distance": {d: sel.raw_counts[d] for d in DISTANCES},
            "strict30_min_per_hop": min(sel.raw_counts.values()),
            "min_per_hop_strict": min_per_hop,
            "strict_hops": list(strict_hops),
            "relaxed_hops": list(relaxed_hops),
            "gate_policy_version": GATE_POLICY_VERSION,
        },
    }


def sample_eval_list(
    exp_payload: Dict,
    sample_per_hop: int,
    relaxed_hops: Sequence[str],
    rng: random.Random,
) -> Tuple[List[Dict], Dict[str, int]]:
    target = exp_payload["target"]
    sampled = [
        {
            "head": target["head"],
            "relation": target["relation"],
            "tail": target["tail"],
            "question": target.get("question", f"What is the {target['relation']} of {target['head']}?"),
            "distance": "d0",
        }
    ]
    expected_counts = {"d0": 1}
    for d in DISTANCES:
        source = exp_payload.get("ripples", {}).get(d, [])
        if d in relaxed_hops:
            n_target = min(sample_per_hop, len(source))
            if n_target <= 0:
                raise RuntimeError(
                    f"Insufficient samples for relaxed hop {exp_payload['experiment_id']:03d} {d}: 0"
                )
        else:
            n_target = sample_per_hop
        if len(source) < n_target:
            raise RuntimeError(
                f"Insufficient samples for {exp_payload['experiment_id']:03d} {d}: "
                f"{len(source)} < {n_target}"
            )
        expected_counts[d] = n_target
        for row in rng.sample(source, n_target):
            sampled.append(
                {
                    "head": row["head"],
                    "relation": row["relation"],
                    "tail": row["tail"],
                    "question": row.get("question", f"What is the {row['relation']} of {row['head']}?"),
                    "distance": d,
                }
            )
    return sampled, expected_counts


def normalize_list_triplet(item: Dict) -> Optional[Dict]:
    if "head" in item and "relation" in item and "tail" in item:
        out = {
            "head": item["head"],
            "relation": item["relation"],
            "tail": item["tail"],
            "question": item.get("question", f"What is the {item['relation']} of {item['head']}?"),
            "distance": item.get("distance", "d5"),
        }
        return out
    triplet = item.get("triplet")
    if isinstance(triplet, list) and len(triplet) == 3:
        head, relation, tail = triplet
        return {
            "head": head,
            "relation": relation,
            "tail": tail,
            "question": item.get("question", f"What is the {relation} of {head}?"),
            "distance": item.get("distance", "d5"),
        }
    return None


def build_irrelevant_set(
    experiments: Sequence[Dict],
    base_irrelevant_file: str,
    size: int,
    rng: random.Random,
) -> List[Dict]:
    if os.path.exists(base_irrelevant_file):
        with open(base_irrelevant_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        pool = []
        if isinstance(data, list):
            for row in data:
                norm = normalize_list_triplet(row)
                if norm:
                    norm["distance"] = "d5"
                    pool.append(norm)
        if len(pool) >= size:
            return rng.sample(pool, size)

    target_heads = {e["target"]["head"] for e in experiments}
    pool = []
    for e in experiments:
        for d in DISTANCES:
            for row in e.get("ripples", {}).get(d, []):
                if row["head"] in target_heads:
                    continue
                pool.append(
                    {
                        "head": row["head"],
                        "relation": row["relation"],
                        "tail": row["tail"],
                        "question": row.get("question", f"What is the {row['relation']} of {row['head']}?"),
                        "distance": "d5",
                    }
                )
    if len(pool) < size:
        raise RuntimeError(f"Could not build irrelevant-{size} from selected experiments (pool={len(pool)})")
    return rng.sample(pool, size)


def dump_json(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_precheck_csv(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "experiment_id",
        "relation_expected",
        "relation_actual",
        "popularity_expected",
        "popularity_actual",
        "target_head",
        "target_tail",
        "d1",
        "d2",
        "d3",
        "d4",
        "d5",
        "min_hop_count",
        "min_strict_hop_count",
        "sampled_d0",
        "sampled_d1",
        "sampled_d2",
        "sampled_d3",
        "sampled_d4",
        "sampled_d5",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Relaxed-Front-30 suite for experiments 001-007.")
    parser.add_argument("--graph-file", default="latest.pkl")
    parser.add_argument("--out-dir", default="results/strict30_suite")
    parser.add_argument("--min-per-hop", type=int, default=30)
    parser.add_argument("--sample-per-hop", type=int, default=30)
    parser.add_argument(
        "--relaxed-hops",
        default="d1,d2",
        help="Comma-separated hops that can be below min-per-hop (must still have >0 entries).",
    )
    parser.add_argument("--irrelevant-size", type=int, default=50)
    parser.add_argument(
        "--base-irrelevant-file",
        default="results/experiments_ripples_fast_20k_v2_sampled30_pair/irrelevant_50_for_006_007.json",
    )
    parser.add_argument("--seed", type=int, default=20260311)
    parser.add_argument(
        "--allow-popularity-fallback",
        action="store_true",
        help="If enabled, allow selecting candidate that violates expected popularity when no strict match exists.",
    )
    args = parser.parse_args()
    relaxed_hops = parse_hops(args.relaxed_hops)
    strict_hops = [d for d in DISTANCES if d not in relaxed_hops]
    if not strict_hops:
        raise ValueError("At least one strict hop is required. Example relaxed hops: d1,d2")

    rng = random.Random(args.seed)
    graph = load_graph(args.graph_file)
    by_relation, relation_tails, degrees, high_threshold, mid_threshold = build_relation_index(graph)

    experiments_out = os.path.join(args.out_dir, "experiments")
    sampled_out = os.path.join(args.out_dir, "sampled")
    manifest_out = os.path.join(args.out_dir, "manifests")
    os.makedirs(experiments_out, exist_ok=True)
    os.makedirs(sampled_out, exist_ok=True)
    os.makedirs(manifest_out, exist_ok=True)

    selected_payloads: List[Dict] = []
    precheck_rows: List[Dict] = []
    used_heads = set()

    for exp_id in range(1, 8):
        relation = RELATION_MAP[exp_id]
        expected_pop = POPULARITY_MAP[exp_id]
        sel = select_target_for_spec(
            exp_id=exp_id,
            relation=relation,
            expected_popularity=expected_pop,
            by_relation=by_relation,
            relation_tails=relation_tails,
            degrees=degrees,
            high_threshold=high_threshold,
            mid_threshold=mid_threshold,
            graph=graph,
            rng=rng,
            min_per_hop=args.min_per_hop,
            strict_hops=strict_hops,
            relaxed_hops=tuple(relaxed_hops),
            used_heads=used_heads,
            allow_popularity_fallback=args.allow_popularity_fallback,
        )
        used_heads.add(sel.head)
        payload = build_experiment_payload(
            sel,
            min_per_hop=args.min_per_hop,
            strict_hops=strict_hops,
            relaxed_hops=tuple(relaxed_hops),
        )
        selected_payloads.append(payload)

        sampled, expected_counts = sample_eval_list(
            payload,
            sample_per_hop=args.sample_per_hop,
            relaxed_hops=tuple(relaxed_hops),
            rng=rng,
        )

        precheck_rows.append(
            {
                "experiment_id": f"{exp_id:03d}",
                "relation_expected": relation,
                "relation_actual": sel.relation,
                "popularity_expected": expected_pop,
                "popularity_actual": sel.popularity,
                "target_head": sel.head,
                "target_tail": sel.tail,
                "d1": sel.raw_counts["d1"],
                "d2": sel.raw_counts["d2"],
                "d3": sel.raw_counts["d3"],
                "d4": sel.raw_counts["d4"],
                "d5": sel.raw_counts["d5"],
                "min_hop_count": min(sel.raw_counts.values()),
                "min_strict_hop_count": min(sel.raw_counts[d] for d in strict_hops),
                "sampled_d0": expected_counts["d0"],
                "sampled_d1": expected_counts["d1"],
                "sampled_d2": expected_counts["d2"],
                "sampled_d3": expected_counts["d3"],
                "sampled_d4": expected_counts["d4"],
                "sampled_d5": expected_counts["d5"],
            }
        )

        exp_path = os.path.join(experiments_out, f"ripple_experiment_{exp_id:03d}.json")
        dump_json(exp_path, payload)

        sampled_path = os.path.join(sampled_out, f"ripple_experiment_{exp_id:03d}_sampled30hop.json")
        dump_json(sampled_path, sampled)

    irrelevant = build_irrelevant_set(
        selected_payloads,
        base_irrelevant_file=args.base_irrelevant_file,
        size=args.irrelevant_size,
        rng=rng,
    )
    irrelevant_path = os.path.join(sampled_out, "irrelevant_50_strict30.json")
    dump_json(irrelevant_path, irrelevant)

    write_precheck_csv(os.path.join(manifest_out, "precheck_capacity.csv"), precheck_rows)

    manifest_rows = []
    for row in precheck_rows:
        exp_id = row["experiment_id"]
        expected_sampled_counts = {
            "d0": int(row["sampled_d0"]),
            "d1": int(row["sampled_d1"]),
            "d2": int(row["sampled_d2"]),
            "d3": int(row["sampled_d3"]),
            "d4": int(row["sampled_d4"]),
            "d5": int(row["sampled_d5"]),
        }
        manifest_rows.append(
            {
                "experiment_id": exp_id,
                "relation_expected": row["relation_expected"],
                "relation_actual": row["relation_actual"],
                "popularity_expected": row["popularity_expected"],
                "popularity_actual": row["popularity_actual"],
                "target_head": row["target_head"],
                "target_tail": row["target_tail"],
                "actual_raw_counts": {d: int(row[d]) for d in DISTANCES},
                "expected_sampled_counts": expected_sampled_counts,
                "sampled_counts": expected_sampled_counts,
                "strict_hops": strict_hops,
                "relaxed_hops": sorted(relaxed_hops),
                "gate_policy_version": GATE_POLICY_VERSION,
                "paths": {
                    "definition_file": os.path.join(experiments_out, f"ripple_experiment_{exp_id}.json"),
                    "sampled_file": os.path.join(sampled_out, f"ripple_experiment_{exp_id}_sampled30hop.json"),
                },
                "status": "definition_ready",
            }
        )

    manifest_payload = {
        "created_at": datetime.utcnow().isoformat(),
        "protocol": {
            "name": "relaxed-front-30",
            "gate_policy_version": GATE_POLICY_VERSION,
            "min_per_hop": args.min_per_hop,
            "min_per_hop_strict": args.min_per_hop,
            "sample_per_hop": args.sample_per_hop,
            "sample_per_hop_cap": args.sample_per_hop,
            "strict_hops": strict_hops,
            "relaxed_hops": sorted(relaxed_hops),
            "mask_primary": "clean_accuracy==1 && clean_correct_token_rank==1",
        },
        "graph_file": args.graph_file,
        "relation_map": {f"{k:03d}": v for k, v in RELATION_MAP.items()},
        "popularity_map": {f"{k:03d}": v for k, v in POPULARITY_MAP.items()},
        "irrelevant_file": irrelevant_path,
        "experiments": manifest_rows,
    }
    dump_json(os.path.join(manifest_out, "strict30_manifest_initial.json"), manifest_payload)

    print(f"Relaxed-Front-30 suite generated under: {args.out_dir}")
    print(f"- definitions: {experiments_out}")
    print(f"- sampled: {sampled_out}")
    print(f"- manifest: {os.path.join(manifest_out, 'strict30_manifest_initial.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
