"""Summarize the 30-run full WikiFactDiff confirmation matrix."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


MODES = ("none", "popular", "random", "rare", "random_distance")
CONTROLS = ("none", "random", "rare", "random_distance")


def bootstrap_ci(values: list[float], seed: int = 42) -> list[float]:
    rng = random.Random(seed)
    means = []
    for _ in range(10000):
        sample = [rng.choice(values) for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return [means[249], means[9749]]


def graph_metrics(report: dict) -> dict:
    rows = report["results"]
    clean_correct = sum(row["clean_correct"] for row in rows)
    flips = sum(
        row["clean_correct"] and not row["edited_correct"] for row in rows
    )
    by_stratum = {}
    for stratum in ("popular", "middle", "rare"):
        category = f"graph_probe_{stratum}"
        subset = [row for row in rows if row["category"] == category]
        denominator = sum(row["clean_correct"] for row in subset)
        numerator = sum(
            row["clean_correct"] and not row["edited_correct"] for row in subset
        )
        by_stratum[stratum] = numerator / denominator
    return {
        "flip_rate": flips / clean_correct,
        "expected_logprob_change": sum(
            row["expected_logprob_change"] for row in rows
        )
        / len(rows),
        "by_stratum": by_stratum,
    }


def native_metrics(report: dict) -> dict:
    update = report["summary"]["update_new"]
    neighborhood = report["summary"].get("ripple_d1")
    return {
        "update_success": update["edited_accuracy"],
        "native_flip_rate": (
            neighborhood["flip_rate"] if neighborhood else None
        ),
        "native_expected_logprob_change": (
            neighborhood["expected_logprob_change_mean"]
            if neighborhood
            else None
        ),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def paired(per_mode: dict, metric: str) -> dict:
    comparisons = {}
    for control in CONTROLS:
        common = sorted(set(per_mode["popular"]) & set(per_mode[control]))
        differences = [
            per_mode["popular"][run_id][metric]
            - per_mode[control][run_id][metric]
            for run_id in common
            if per_mode["popular"][run_id][metric] is not None
            and per_mode[control][run_id][metric] is not None
        ]
        comparisons[f"popular_minus_{control}"] = {
            "n": len(differences),
            "mean_difference": mean(differences),
            "bootstrap_95_ci": bootstrap_ci(differences),
        }
    return comparisons


def summarize(output_base: Path) -> None:
    per_mode = {}
    aggregate = {}
    for mode in MODES:
        native_paths = sorted(
            output_base.glob(
                f"seed*/wikifactdiff/{mode}/*/evaluation_strict.json"
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
                **{
                    f"graph_{key}": value
                    for key, value in graph_metrics(
                        json.loads(graph_path.read_text())
                    ).items()
                },
            }
        per_mode[mode] = runs
        aggregate[mode] = {
            "runs": len(runs),
            "update_success_mean": mean(
                [run["update_success"] for run in runs.values()]
            ),
            "native_flip_rate_mean": mean(
                [
                    run["native_flip_rate"]
                    for run in runs.values()
                    if run["native_flip_rate"] is not None
                ]
            ),
            "graph_flip_rate_mean": mean(
                [run["graph_flip_rate"] for run in runs.values()]
            ),
            "native_expected_logprob_change_mean": mean(
                [
                    run["native_expected_logprob_change"]
                    for run in runs.values()
                    if run["native_expected_logprob_change"] is not None
                ]
            ),
            "graph_expected_logprob_change_mean": mean(
                [
                    run["graph_expected_logprob_change"]
                    for run in runs.values()
                ]
            ),
            "graph_flip_rate_by_stratum": {
                stratum: mean(
                    [
                        run["graph_by_stratum"][stratum]
                        for run in runs.values()
                    ]
                )
                for stratum in ("popular", "middle", "rare")
            },
        }

    native_comparisons = paired(per_mode, "native_flip_rate")
    graph_comparisons = paired(per_mode, "graph_flip_rate")
    best_update_success = max(
        metrics["update_success_mean"] for metrics in aggregate.values()
    )
    gate = {
        "thirty_runs_complete": all(
            metrics["runs"] == 6 for metrics in aggregate.values()
        ),
        "popular_update_success_within_5_points": (
            best_update_success - aggregate["popular"]["update_success_mean"]
            <= 0.05
        ),
        "popular_graph_flip_below_random": (
            graph_comparisons["popular_minus_random"]["mean_difference"] < 0
        ),
    }
    gate["passed"] = all(gate.values())
    result = {
        "scope": {
            "dataset": "full WikiFactDiff replacements",
            "batches": 3,
            "batch_size": 25,
            "seeds": 2,
            "arms": 5,
            "runs": 30,
        },
        "modes": aggregate,
        "native_paired_comparisons": native_comparisons,
        "graph_paired_comparisons": graph_comparisons,
        "primary_gate": gate,
        "per_run": per_mode,
        "claim_boundary": (
            "Native WikiFactDiff locality and frozen graph-holdout retention "
            "are separate outcomes and must not substitute for each other."
        ),
    }
    json_path = output_base / "wfd_full_confirmation_summary.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# Full WikiFactDiff 30-run Confirmation",
        "",
        "Scope: 3 entity-disjoint B=25 batches × 2 seeds × 5 arms.",
        "",
        "| Mode | Runs | Update success | Native d1 Flip Rate | Graph Flip Rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        row = aggregate[mode]
        lines.append(
            f"| {mode} | {row['runs']} | "
            f"{row['update_success_mean']:.3f} | "
            f"{row['native_flip_rate_mean']:.3f} | "
            f"{row['graph_flip_rate_mean']:.3f} |"
        )
    lines.extend(["", "## Popular paired comparisons", ""])
    for outcome, comparisons in (
        ("Native d1 Flip Rate", native_comparisons),
        ("Graph Flip Rate", graph_comparisons),
    ):
        lines.append(f"### {outcome}")
        for name, row in comparisons.items():
            lines.append(
                f"- {name}: {row['mean_difference']:+.3f}; "
                f"95% bootstrap CI={row['bootstrap_95_ci']}; n={row['n']}"
            )
    lines.extend(
        [
            "",
            "## Primary gate",
            "",
            f"- Passed: {gate['passed']}",
        ]
    )
    for criterion, passed in gate.items():
        if criterion != "passed":
            lines.append(f"- {criterion}: {passed}")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "- Native WikiFactDiff locality and frozen graph-holdout retention "
            "are reported separately; one cannot replace the other.",
            "- This is medium-scale evidence from 75 unique updates and six "
            "paired Batch×Seed units, not proof of universal protection.",
        ]
    )
    markdown_path = output_base / "wfd_full_confirmation_summary.md"
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
