"""Generate matched object-popularity anchors for the V2 mitigation experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = ROOT / "results/checkpoints/final.pkl"
DEFAULT_TARGETS_DIR = ROOT / "data/ripple_eval/experiments_final_45"
DEFAULT_OUT_DIR = ROOT / "data/external_eval"

PLAN_30_TARGETS = (
    [f"hub_{i}" for i in [1, 3, 4, 5, 6, 10, 11, 12, 13, 14]]
    + [f"random_{i}" for i in [1, 2, 7, 8, 9, 10, 11, 12, 14, 15]]
    + [f"tail_{i}" for i in [1, 3, 4, 5, 7, 9, 10, 11, 12, 15]]
)


def load_graph(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f)
    return data["graph"] if isinstance(data, dict) else data


def load_targets(path: Path | None):
    if path is not None:
        targets = json.loads(path.read_text())
    else:
        targets = {}
        for target_id in PLAN_30_TARGETS:
            data = json.loads((DEFAULT_TARGETS_DIR / f"{target_id}.json").read_text())
            target = data.get("target", data)
            targets[target_id] = {
                "head": target["head"],
                "relation": target["relation"],
                "tail": target["tail"],
                "poison_answer": target["poison_answer"],
            }

    required = {"head", "relation", "tail", "poison_answer"}
    for target_id, target in targets.items():
        missing = required - set(target)
        if missing:
            raise ValueError(f"{target_id} is missing target fields: {sorted(missing)}")
    return targets


def stable_key(seed: int, target_id: str, *parts: str) -> bytes:
    value = "|".join([str(seed), target_id, *map(str, parts)])
    return hashlib.sha256(value.encode("utf-8")).digest()


def build_fact_index(graph):
    pair_relations = defaultdict(set)
    for head, tail, attrs in graph.edges(data=True):
        relation = attrs.get("relation")
        if relation:
            pair_relations[(head, tail)].add(relation)

    facts_by_object = defaultdict(list)
    seen = set()
    for head, tail, attrs in graph.edges(data=True):
        relation = attrs.get("relation")
        if (
            not relation
            or attrs.get("is_inverse") is True
            or head == tail
            or head == "None"
            or tail == "None"
            or relation in pair_relations.get((tail, head), set())
            or not (attrs.get("question") or attrs.get("surface"))
        ):
            continue
        identity = (head, relation, tail)
        if identity in seen:
            continue
        seen.add(identity)
        facts_by_object[tail].append(
            {
                "head": head,
                "relation": relation,
                "tail": tail,
                "question": attrs.get("question"),
                "surface": attrs.get("surface"),
                "object_in_degree": graph.in_degree(tail),
            }
        )
    return facts_by_object


def valid_facts(facts, excluded_entities: set[str], excluded_relation: str):
    return [
        fact
        for fact in facts
        if fact["head"] not in excluded_entities
        and fact["tail"] not in excluded_entities
        and fact["relation"] != excluded_relation
    ]


def choose_fact(facts, seed: int, target_id: str):
    return min(
        facts,
        key=lambda fact: stable_key(
            seed,
            target_id,
            fact["head"],
            fact["relation"],
            fact["tail"],
        ),
    )


def ranked_objects(objects, graph, target_id: str, seed: int, descending: bool):
    direction = -1 if descending else 1
    return sorted(
        objects,
        key=lambda obj: (
            direction * graph.in_degree(obj),
            stable_key(seed, target_id, obj),
        ),
    )


def select_ranked(
    ordered_objects,
    facts_by_object,
    excluded_entities,
    excluded_relation,
    target_id,
    seed,
    n,
):
    selected = []
    for obj in ordered_objects:
        facts = valid_facts(
            facts_by_object[obj], excluded_entities, excluded_relation
        )
        if not facts:
            continue
        selected.append(choose_fact(facts, seed, target_id))
        if len(selected) == n:
            break
    return selected


def select_random_middle(
    objects,
    excluded_objects,
    facts_by_object,
    excluded_entities,
    excluded_relation,
    target_id,
    seed,
    n,
):
    candidates = [obj for obj in objects if obj not in excluded_objects]
    random.Random(f"{seed}:{target_id}:object-v2").shuffle(candidates)
    return select_ranked(
        candidates,
        facts_by_object,
        excluded_entities,
        excluded_relation,
        target_id,
        seed,
        n,
    )


def write_anchor_file(path: Path, metadata: dict, anchors: dict):
    path.write_text(
        json.dumps(
            {"metadata": metadata, "per_target": anchors},
            indent=2,
            ensure_ascii=False,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-path", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--targets-file", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-suffix", default="")
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    graph = load_graph(args.graph_path)
    targets = load_targets(args.targets_file)
    facts_by_object = build_fact_index(graph)
    objects = list(facts_by_object)

    popular_by_target = {}
    rare_by_target = {}
    random_by_target = {}

    for target_id, target in targets.items():
        excluded_entities = {
            target["head"],
            target["tail"],
            target["poison_answer"],
        }
        popular = select_ranked(
            ranked_objects(objects, graph, target_id, args.seed, descending=True),
            facts_by_object,
            excluded_entities,
            target["relation"],
            target_id,
            args.seed,
            args.n,
        )
        rare = select_ranked(
            ranked_objects(objects, graph, target_id, args.seed, descending=False),
            facts_by_object,
            excluded_entities,
            target["relation"],
            target_id,
            args.seed,
            args.n,
        )
        popular_cutoff = min(graph.in_degree(fact["tail"]) for fact in popular)
        rare_cutoff = max(graph.in_degree(fact["tail"]) for fact in rare)
        excluded_objects = {
            obj
            for obj in objects
            if graph.in_degree(obj) >= popular_cutoff
            or graph.in_degree(obj) <= rare_cutoff
        }
        random_middle = select_random_middle(
            objects,
            excluded_objects,
            facts_by_object,
            excluded_entities,
            target["relation"],
            target_id,
            args.seed,
            args.n,
        )

        if min(map(len, (popular, rare, random_middle))) < args.n:
            raise RuntimeError(f"{target_id} has fewer than {args.n} valid anchors")

        popular_by_target[target_id] = popular
        rare_by_target[target_id] = rare
        random_by_target[target_id] = random_middle

    args.out_dir.mkdir(parents=True, exist_ok=True)
    common_metadata = {
        "selector_version": "object_matched_v2",
        "ranking_endpoint": "tail",
        "ranking_metric": "in_degree",
        "canonical_fact_selection": "sha256_per_target_from_incoming_edges",
        "random_pool_rule": "strictly_between_rare_and_popular_degree_strata",
        "N": args.n,
        "seed": args.seed,
        "n_targets": len(targets),
        "graph_path": str(args.graph_path),
        "targets_file": str(args.targets_file) if args.targets_file else None,
    }
    outputs = [
        (
            f"anchors_popular_object_top{args.n}{args.out_suffix}.json",
            "popular_object_top",
            popular_by_target,
        ),
        (
            f"anchors_rare_object_bottom{args.n}{args.out_suffix}.json",
            "rare_object_bottom",
            rare_by_target,
        ),
        (
            f"anchors_random_object_middle{args.n}_seed{args.seed}{args.out_suffix}.json",
            "random_object_middle",
            random_by_target,
        ),
    ]
    for filename, mode, anchors in outputs:
        path = args.out_dir / filename
        write_anchor_file(path, {**common_metadata, "mode": mode}, anchors)
        print(f"Wrote {path} ({len(anchors)} targets x {args.n} anchors)")


if __name__ == "__main__":
    main()
