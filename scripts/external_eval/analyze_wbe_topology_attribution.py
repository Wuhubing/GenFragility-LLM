"""Analyze frozen-anchor structure and WikiBigEdit update popularity."""
from __future__ import annotations

import argparse
import bisect
import json
import pickle
import statistics
from collections import Counter
from pathlib import Path

import networkx as nx


ROOT = Path(__file__).resolve().parents[2]
MODES = ("none", "popular", "random", "rare", "random_distance")
ANCHOR_MODES = MODES[1:]


def load_graph(path: Path):
    with path.open("rb") as handle:
        data = pickle.load(handle)
    return data["graph"] if isinstance(data, dict) else data


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def graph_qid_degrees(graph, index_path: Path) -> dict[str, int]:
    index = json.loads(index_path.read_text())
    aggregated: dict[str, int] = {}
    for node, degree in graph.in_degree():
        qid = index["name_to_qid"].get(node)
        if qid:
            aggregated[qid] = aggregated.get(qid, 0) + degree
    return aggregated


def anchor_structure(graph, anchor_dir: Path) -> dict:
    probe_bank = json.loads((anchor_dir / "probes/probe_bank.json").read_text())
    probe_entities = [
        entity
        for probe in probe_bank["probes"]
        for entity in (probe["head"], probe["tail"])
        if entity in graph
    ]
    distances = nx.multi_source_dijkstra_path_length(
        graph.to_undirected(as_view=True),
        probe_entities,
        cutoff=5,
        weight=None,
    )
    mode_facts = {
        mode: json.loads(
            (anchor_dir / f"anchors_{mode}_100.json").read_text()
        )["anchors"]
        for mode in ANCHOR_MODES
    }
    probe_relation_counts = Counter(
        probe["relation"] for probe in probe_bank["probes"]
    )
    popular_relations = {fact["relation"] for fact in mode_facts["popular"]}
    result = {}
    for mode, facts in mode_facts.items():
        degrees = [graph.in_degree(fact["tail"]) for fact in facts]
        probe_distances = [
            min(distances.get(fact["head"], 6), distances.get(fact["tail"], 6))
            for fact in facts
        ]
        relation_set = {fact["relation"] for fact in facts}
        covered_probes = sum(
            count
            for relation, count in probe_relation_counts.items()
            if relation in relation_set
        )
        result[mode] = {
            "degree": describe(degrees),
            "distance_to_probe": dict(sorted(Counter(probe_distances).items())),
            "relations": len(relation_set),
            "probe_relation_coverage": covered_probes / len(probe_bank["probes"]),
            "relation_jaccard_with_popular": (
                len(relation_set & popular_relations)
                / len(relation_set | popular_relations)
            ),
        }
    return result


def probe_metrics(path: Path) -> dict:
    report = json.loads(path.read_text())
    rows = [
        row
        for row in report["results"]
        if row["category"].startswith("graph_probe_")
    ]
    clean = sum(row["clean_correct"] for row in rows)
    flips = sum(
        row["clean_correct"] and not row["edited_correct"] for row in rows
    )
    return {"flip_rate": flips / clean, "clean_correct": clean}


def batch_results(output_dir: Path) -> dict:
    by_batch: dict[str, dict[str, list[float]]] = {}
    for seed_dir in sorted(output_dir.glob("seed*")):
        for mode in MODES:
            for path in sorted(
                (seed_dir / "wikibigedit" / mode).glob(
                    "*/graph_probe_evaluation.json"
                )
            ):
                batch = path.parent.name
                by_batch.setdefault(batch, {}).setdefault(mode, []).append(
                    probe_metrics(path)["flip_rate"]
                )
    result = {}
    for batch, modes in sorted(by_batch.items()):
        means = {mode: statistics.mean(modes[mode]) for mode in MODES}
        result[batch] = {
            "flip_rate": means,
            "popular_minus_none": means["popular"] - means["none"],
            "popular_minus_random": means["popular"] - means["random"],
            "popular_minus_rare": means["popular"] - means["rare"],
            "popular_minus_random_distance": (
                means["popular"] - means["random_distance"]
            ),
        }
    return result


def popularity_bucket(degree: int | None) -> str:
    if degree is None:
        return "unlinked"
    if degree >= 500:
        return "hub"
    if degree >= 20:
        return "mid"
    return "tail"


def update_popularity(
    manifest_path: Path,
    qid_degrees: dict[str, int],
    graph,
) -> dict:
    manifest = json.loads(manifest_path.read_text())
    sorted_degrees = sorted(qid_degrees.values())
    result = {}
    for batch, unit in manifest["units"].items():
        rows = []
        for update in unit["updates"]:
            def resolve_degree(qid_field: str, text_field: str):
                degree = qid_degrees.get(str(update.get(qid_field, "")))
                if degree is not None:
                    return degree, "qid"
                text = str(update.get(text_field, ""))
                if text in graph:
                    return graph.in_degree(text), "exact_text"
                return None, "unlinked"

            head_degree, head_resolution = resolve_degree("head_qid", "head")
            tail_degree, tail_resolution = resolve_degree("tail_qid", "tail")
            new_degree, new_resolution = resolve_degree("", "poison_answer")

            def more_popular_percent(degree):
                if degree is None:
                    return None
                return 100.0 * (
                    len(sorted_degrees) - bisect.bisect_right(sorted_degrees, degree)
                ) / len(sorted_degrees)

            rows.append(
                {
                    "head_degree": head_degree,
                    "tail_degree": tail_degree,
                    "new_degree": new_degree,
                    "head_bucket": popularity_bucket(head_degree),
                    "tail_bucket": popularity_bucket(tail_degree),
                    "new_bucket": popularity_bucket(new_degree),
                    "head_resolution": head_resolution,
                    "tail_resolution": tail_resolution,
                    "new_resolution": new_resolution,
                    "head_more_popular_percent": more_popular_percent(head_degree),
                    "tail_more_popular_percent": more_popular_percent(tail_degree),
                }
            )
        head_degrees = [
            row["head_degree"] for row in rows if row["head_degree"] is not None
        ]
        tail_degrees = [
            row["tail_degree"] for row in rows if row["tail_degree"] is not None
        ]
        new_degrees = [
            row["new_degree"] for row in rows if row["new_degree"] is not None
        ]
        result[batch] = {
            "updates": len(rows),
            "head_linked": len(head_degrees),
            "tail_linked": len(tail_degrees),
            "head_degree": describe(head_degrees),
            "tail_degree": describe(tail_degrees),
            "new_degree": describe(new_degrees),
            "head_buckets": dict(Counter(row["head_bucket"] for row in rows)),
            "tail_buckets": dict(Counter(row["tail_bucket"] for row in rows)),
            "new_buckets": dict(Counter(row["new_bucket"] for row in rows)),
            "head_resolution": dict(
                Counter(row["head_resolution"] for row in rows)
            ),
            "tail_resolution": dict(
                Counter(row["tail_resolution"] for row in rows)
            ),
            "new_resolution": dict(
                Counter(row["new_resolution"] for row in rows)
            ),
        }
    return result


def render_markdown(report: dict) -> str:
    def summary_text(summary: dict) -> str:
        if summary["n"] == 0:
            return "unlinked"
        return f"{summary['median']} / {summary['mean']:.1f}"

    lines = [
        "# WikiBigEdit Topology Attribution Audit",
        "",
        "## Frozen anchor structure",
        "",
        "| Mode | Object degree median / mean / max | Distance to probe | Relations | Probe relation coverage | Relation Jaccard vs Popular |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for mode, row in report["anchors"].items():
        degree = row["degree"]
        lines.append(
            f"| {mode} | {degree['median']:.1f} / {degree['mean']:.1f} / "
            f"{degree['max']} | {row['distance_to_probe']} | "
            f"{row['relations']} | {row['probe_relation_coverage']:.3f} | "
            f"{row['relation_jaccard_with_popular']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## WikiBigEdit update popularity in the 100k graph",
            "",
        "| Batch | Subject linked | Subject degree median / mean | Object-QID linked | Object degree median / mean | Answer-text linked | Answer degree median / mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for batch, row in report["updates"].items():
        head = row["head_degree"]
        tail = row["tail_degree"]
        new = row["new_degree"]
        lines.append(
            f"| {batch} | {row['head_linked']}/{row['updates']} | "
            f"{summary_text(head)} | "
            f"{row['tail_linked']}/{row['updates']} | "
            f"{summary_text(tail)} | {new['n']}/{row['updates']} | "
            f"{summary_text(new)} |"
        )
    lines.extend(
        [
            "",
            "## Mean Flip Rate by batch (two seeds)",
            "",
            "| Batch | Update-only | Popular | Random | Rare | Distance-matched Random | Popular−Random | Popular−Rare | Popular−Distance |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for batch, row in report["batch_results"].items():
        rates = row["flip_rate"]
        lines.append(
            f"| {batch} | {rates['none']:.3f} | {rates['popular']:.3f} | "
            f"{rates['random']:.3f} | {rates['rare']:.3f} | "
            f"{rates['random_distance']:.3f} | "
            f"{row['popular_minus_random']:+.3f} | "
            f"{row['popular_minus_rare']:+.3f} | "
            f"{row['popular_minus_random_distance']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Batch-level association is descriptive only: there are three unique update batches.",
            "- Each LoRA jointly trains 25 updates, so no per-update causal effect can be identified.",
            "- WikiBigEdit provides QIDs for subjects and object fields, but not for the answer-text field used for training.",
            "- The WikiBigEdit object field is not treated as a verified old answer.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph",
        type=Path,
        default=ROOT / "results/checkpoints/final.pkl",
    )
    parser.add_argument(
        "--qid-index",
        type=Path,
        default=ROOT / "data/external_eval/graph_qid_index.json",
    )
    parser.add_argument(
        "--anchor-dir",
        type=Path,
        default=ROOT / "data/external_eval/frozen_rehearsal_core",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            ROOT
            / "data/external_eval/wbe_frozen_confirmation/wikibigedit/manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "main_output/external_rehearsal/wbe_frozen_confirmation"
        ),
    )
    args = parser.parse_args()
    graph = load_graph(args.graph)
    report = {
        "anchors": anchor_structure(graph, args.anchor_dir),
        "updates": update_popularity(
            args.manifest,
            graph_qid_degrees(graph, args.qid_index),
            graph,
        ),
        "batch_results": batch_results(args.output_dir),
    }
    json_path = args.output_dir / "topology_attribution_analysis.json"
    markdown_path = args.output_dir / "topology_attribution_analysis.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path.write_text(render_markdown(report))
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
