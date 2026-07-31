"""Summarize the 30-run atomic MQuAKE-CF confirmation matrix."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


MODES = ("none", "popular", "random", "rare", "random_distance")
CONTROLS = ("none", "random", "rare", "random_distance")
STRATA = ("popular", "middle", "rare")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def bootstrap_ci(values: list[float], seed: int = 53) -> list[float]:
    rng = random.Random(seed)
    means = []
    for _ in range(10000):
        sample = [rng.choice(values) for _ in values]
        means.append(mean(sample))
    means.sort()
    return [means[249], means[9749]]


def graph_metrics(report: dict) -> dict:
    rows = report["results"]

    def flip_rate(items: list[dict]) -> float:
        clean_correct = sum(row["clean_correct"] for row in items)
        flips = sum(
            row["clean_correct"] and not row["edited_correct"] for row in items
        )
        return flips / clean_correct

    return {
        "graph_flip_rate": flip_rate(rows),
        "graph_by_stratum": {
            stratum: flip_rate(
                [
                    row
                    for row in rows
                    if row["category"] == f"graph_probe_{stratum}"
                ]
            )
            for stratum in STRATA
        },
    }


def native_metrics(report: dict) -> dict:
    summary = report["summary"]
    return {
        "update_success": summary["update_new"]["edited_accuracy"],
        "single_hop_success": summary["single_hop_new"]["edited_accuracy"],
        "multihop_success": summary["multihop_new"]["edited_accuracy"],
        "unchanged_hop_flip_rate": summary["unchanged_single_hop_old"][
            "flip_rate"
        ],
    }


def paired(per_mode: dict, metric: str) -> dict:
    comparisons = {}
    for control in CONTROLS:
        common = sorted(set(per_mode["popular"]) & set(per_mode[control]))
        differences = [
            per_mode["popular"][run_id][metric]
            - per_mode[control][run_id][metric]
            for run_id in common
        ]
        comparisons[f"popular_minus_{control}"] = {
            "n": len(differences),
            "mean_difference": mean(differences),
            "bootstrap_95_ci": bootstrap_ci(differences),
            "popular_wins": sum(value < 0 for value in differences),
        }
    return comparisons


def summarize(output_base: Path) -> None:
    per_mode = {}
    aggregate = {}
    for mode in MODES:
        native_paths = sorted(
            output_base.glob(
                f"seed*/mquake_cf/{mode}/*/evaluation_strict.json"
            )
        )
        if len(native_paths) != 6:
            raise SystemExit(
                f"Expected 6 native reports for {mode}, got {len(native_paths)}"
            )
        runs = {}
        for native_path in native_paths:
            seed = native_path.parents[3].name
            unit = native_path.parent.name
            graph_path = native_path.parent / "graph_probe_evaluation.json"
            if not graph_path.is_file():
                raise SystemExit(f"Missing graph report: {graph_path}")
            run_id = f"{seed}/{unit}"
            runs[run_id] = {
                **native_metrics(json.loads(native_path.read_text())),
                **graph_metrics(json.loads(graph_path.read_text())),
            }
        per_mode[mode] = runs
        aggregate[mode] = {
            "runs": len(runs),
            "update_success_mean": mean(
                [run["update_success"] for run in runs.values()]
            ),
            "single_hop_success_mean": mean(
                [run["single_hop_success"] for run in runs.values()]
            ),
            "multihop_success_mean": mean(
                [run["multihop_success"] for run in runs.values()]
            ),
            "unchanged_hop_flip_rate_mean": mean(
                [run["unchanged_hop_flip_rate"] for run in runs.values()]
            ),
            "graph_flip_rate_mean": mean(
                [run["graph_flip_rate"] for run in runs.values()]
            ),
            "graph_flip_rate_by_stratum": {
                stratum: mean(
                    [
                        run["graph_by_stratum"][stratum]
                        for run in runs.values()
                    ]
                )
                for stratum in STRATA
            },
        }

    native_comparisons = paired(per_mode, "unchanged_hop_flip_rate")
    graph_comparisons = paired(per_mode, "graph_flip_rate")
    best_update = max(
        metrics["update_success_mean"] for metrics in aggregate.values()
    )
    gate = {
        "thirty_runs_complete": all(
            metrics["runs"] == 6 for metrics in aggregate.values()
        ),
        "popular_update_success_within_5_points": (
            best_update - aggregate["popular"]["update_success_mean"] <= 0.05
        ),
        "popular_native_flip_below_random": (
            native_comparisons["popular_minus_random"]["mean_difference"] < 0
        ),
    }
    gate["passed"] = all(gate.values())
    result = {
        "scope": {
            "dataset": "MQuAKE-CF atomic subset",
            "batches": 3,
            "batch_size": 25,
            "seeds": 2,
            "arms": 5,
            "runs": 30,
        },
        "modes": aggregate,
        "unchanged_hop_paired_comparisons": native_comparisons,
        "graph_paired_comparisons": graph_comparisons,
        "primary_gate": gate,
        "per_run": per_mode,
        "claim_boundary": (
            "The frozen subset contains one requested rewrite per MQuAKE-CF "
            "case; unchanged single-hop retention and multi-hop update success "
            "are reported separately."
        ),
    }
    json_path = output_base / "mquake_cf_confirmation_summary.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# MQuAKE-CF Atomic 30-run Confirmation",
        "",
        "Scope: 3 entity-disjoint B=25 batches × 2 seeds × 5 arms.",
        "",
        "| Mode | Runs | Update | New single-hop | New multi-hop | Unchanged-hop FR | Graph FR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        row = aggregate[mode]
        lines.append(
            f"| {mode} | {row['runs']} | "
            f"{row['update_success_mean']:.3f} | "
            f"{row['single_hop_success_mean']:.3f} | "
            f"{row['multihop_success_mean']:.3f} | "
            f"{row['unchanged_hop_flip_rate_mean']:.3f} | "
            f"{row['graph_flip_rate_mean']:.3f} |"
        )
    lines.extend(["", "## Popular paired comparisons", ""])
    for label, comparisons in (
        ("Unchanged single-hop Flip Rate", native_comparisons),
        ("Graph Flip Rate", graph_comparisons),
    ):
        lines.append(f"### {label}")
        for name, row in comparisons.items():
            lines.append(
                f"- {name}: {row['mean_difference']:+.3f}; "
                f"95% CI={row['bootstrap_95_ci']}; "
                f"wins={row['popular_wins']}/{row['n']}"
            )
    lines.extend(["", "## Primary gate", "", f"- Passed: {gate['passed']}"])
    for criterion, passed in gate.items():
        if criterion != "passed":
            lines.append(f"- {criterion}: {passed}")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "- Only atomic MQuAKE-CF cases are included.",
            "- Unchanged single-hop retention, new multi-hop success, and "
            "frozen graph retention are distinct outcomes.",
        ]
    )
    markdown_path = output_base / "mquake_cf_confirmation_summary.md"
    markdown_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-base", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.output_base)


if __name__ == "__main__":
    main()
