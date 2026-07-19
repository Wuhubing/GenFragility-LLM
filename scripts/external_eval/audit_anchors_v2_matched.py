"""Audit matched V2 anchor files before any training is launched."""
from __future__ import annotations

import argparse
import json
import pickle
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = ROOT / "results/checkpoints/final.pkl"


def load_graph(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f)
    return data["graph"] if isinstance(data, dict) else data


def load_anchor_file(path: Path):
    data = json.loads(path.read_text())
    return data.get("metadata", {}), data.get("per_target", {})


def summarize(mode: str, anchors: dict, graph):
    facts = [fact for target_facts in anchors.values() for fact in target_facts]
    object_degrees = [graph.in_degree(fact["tail"]) for fact in facts]
    head_degrees = [graph.in_degree(fact["head"]) for fact in facts]
    text_lengths = [
        len(
            fact.get("surface")
            or fact.get("question")
            or f"{fact['head']} {fact['relation']} {fact['tail']}"
        )
        for fact in facts
    ]
    relations = Counter(fact["relation"] for fact in facts)
    return {
        "mode": mode,
        "targets": len(anchors),
        "anchors": len(facts),
        "object_degree_min": min(object_degrees),
        "object_degree_median": statistics.median(object_degrees),
        "object_degree_mean": statistics.mean(object_degrees),
        "object_degree_max": max(object_degrees),
        "head_degree_median": statistics.median(head_degrees),
        "text_length_mean": statistics.mean(text_lengths),
        "unique_relations": len(relations),
        "top_relations": relations.most_common(5),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-path", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--targets-file", type=Path, required=True)
    parser.add_argument("--popular", type=Path, required=True)
    parser.add_argument("--rare", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    graph = load_graph(args.graph_path)
    targets = json.loads(args.targets_file.read_text())
    files = {
        "popular": args.popular,
        "rare": args.rare,
        "random": args.random,
    }
    loaded = {mode: load_anchor_file(path) for mode, path in files.items()}
    failures = []

    expected_targets = set(targets)
    for mode, (metadata, anchors) in loaded.items():
        if set(anchors) != expected_targets:
            failures.append(f"{mode}: target IDs do not match targets file")
        if metadata.get("selector_version") != "object_matched_v2":
            failures.append(f"{mode}: selector_version is not object_matched_v2")
        if metadata.get("ranking_endpoint") != "tail":
            failures.append(f"{mode}: ranking_endpoint is not tail")
        if metadata.get("canonical_fact_selection") != (
            "sha256_per_target_from_incoming_edges"
        ):
            failures.append(f"{mode}: canonical fact selector mismatch")
        if metadata.get("random_pool_rule") != (
            "strictly_between_rare_and_popular_degree_strata"
        ):
            failures.append(f"{mode}: random pool rule mismatch")

        for target_id, facts in anchors.items():
            if len(facts) != args.n:
                failures.append(
                    f"{mode}/{target_id}: expected {args.n} anchors, got {len(facts)}"
                )
                continue
            target = targets[target_id]
            excluded = {
                target["head"],
                target["tail"],
                target["poison_answer"],
            }
            tails = [fact["tail"] for fact in facts]
            if len(set(tails)) != args.n:
                failures.append(f"{mode}/{target_id}: duplicate anchor objects")
            for fact in facts:
                if fact["head"] in excluded or fact["tail"] in excluded:
                    failures.append(f"{mode}/{target_id}: target entity overlap")
                if fact["relation"] == target["relation"]:
                    failures.append(f"{mode}/{target_id}: target relation overlap")
                if not graph.has_edge(fact["head"], fact["tail"]):
                    failures.append(f"{mode}/{target_id}: fact is absent from graph")

    popular_anchors = loaded["popular"][1]
    rare_anchors = loaded["rare"][1]
    random_anchors = loaded["random"][1]
    for target_id in expected_targets:
        mode_objects = {
            "popular": {fact["tail"] for fact in popular_anchors[target_id]},
            "rare": {fact["tail"] for fact in rare_anchors[target_id]},
            "random": {fact["tail"] for fact in random_anchors[target_id]},
        }
        if mode_objects["popular"] & mode_objects["rare"]:
            failures.append(f"{target_id}: Popular/Rare object overlap")
        if mode_objects["popular"] & mode_objects["random"]:
            failures.append(f"{target_id}: Popular/Random object overlap")
        if mode_objects["rare"] & mode_objects["random"]:
            failures.append(f"{target_id}: Rare/Random object overlap")

        popular_degrees = [
            graph.in_degree(fact["tail"]) for fact in popular_anchors[target_id]
        ]
        rare_degrees = [
            graph.in_degree(fact["tail"]) for fact in rare_anchors[target_id]
        ]
        random_degrees = [
            graph.in_degree(fact["tail"]) for fact in random_anchors[target_id]
        ]
        if min(popular_degrees) <= max(random_degrees):
            failures.append(f"{target_id}: Popular/Random degree strata overlap")
        if max(rare_degrees) >= min(random_degrees):
            failures.append(f"{target_id}: Rare/Random degree strata overlap")

    summaries = [
        summarize(mode, anchors, graph)
        for mode, (_, anchors) in loaded.items()
    ]
    for summary in summaries:
        print(
            f"{summary['mode']}: targets={summary['targets']} "
            f"anchors={summary['anchors']} "
            f"object_degree={summary['object_degree_min']}/"
            f"{summary['object_degree_median']}/"
            f"{summary['object_degree_max']} "
            f"relations={summary['unique_relations']}"
        )

    if args.out_md:
        lines = [
            "# Matched V2 Anchor Structural Audit",
            "",
            f"- Targets file: `{args.targets_file}`",
            f"- Expected anchors per target and mode: {args.n}",
            f"- Status: {'PASS' if not failures else 'FAIL'}",
            "",
            "| Mode | Targets | Anchors | Object degree min/median/mean/max | "
            "Head degree median | Mean text length | Unique relations |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for summary in summaries:
            lines.append(
                f"| {summary['mode']} | {summary['targets']} | "
                f"{summary['anchors']} | {summary['object_degree_min']} / "
                f"{summary['object_degree_median']:.1f} / "
                f"{summary['object_degree_mean']:.1f} / "
                f"{summary['object_degree_max']} | "
                f"{summary['head_degree_median']:.1f} | "
                f"{summary['text_length_mean']:.1f} | "
                f"{summary['unique_relations']} |"
            )
        if failures:
            lines.extend(["", "## Failures"])
            lines.extend(f"- {failure}" for failure in failures[:50])
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text("\n".join(lines) + "\n")
        print(f"Wrote {args.out_md}")

    if failures:
        print(f"FAIL: {len(failures)} structural violations")
        for failure in failures[:10]:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("PASS: matched V2 anchor structure is valid")


if __name__ == "__main__":
    main()
