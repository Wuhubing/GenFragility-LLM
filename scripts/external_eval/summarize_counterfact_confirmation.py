"""Summarize the 30-run CounterFact confirmation matrix."""
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


def bootstrap_ci(values: list[float], seed: int = 47) -> list[float]:
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
        "flip_rate": flip_rate(rows),
        "expected_logprob_change": mean(
            [row["expected_logprob_change"] for row in rows]
        ),
        "by_stratum": {
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
        "paraphrase_success": summary["paraphrase_new"]["edited_accuracy"],
        "neighborhood_flip_rate": summary["neighborhood_old"]["flip_rate"],
        "neighborhood_expected_logprob_change": summary["neighborhood_old"][
            "expected_logprob_change_mean"
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
                f"seed*/counterfact/{mode}/*/evaluation_strict.json"
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
            "paraphrase_success_mean": mean(
                [run["paraphrase_success"] for run in runs.values()]
            ),
            "neighborhood_flip_rate_mean": mean(
                [run["neighborhood_flip_rate"] for run in runs.values()]
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

    native_comparisons = paired(per_mode, "neighborhood_flip_rate")
    graph_comparisons = paired(per_mode, "graph_flip_rate")
    rare_probe_per_mode = {
        mode: {
            run_id: {
                **run,
                "rare_probe_flip_rate": run["graph_by_stratum"]["rare"],
            }
            for run_id, run in runs.items()
        }
        for mode, runs in per_mode.items()
    }
    rare_comparisons = paired(rare_probe_per_mode, "rare_probe_flip_rate")
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
            "dataset": "CounterFact",
            "batches": 3,
            "batch_size": 25,
            "seeds": 2,
            "arms": 5,
            "runs": 30,
        },
        "modes": aggregate,
        "neighborhood_paired_comparisons": native_comparisons,
        "graph_paired_comparisons": graph_comparisons,
        "rare_probe_paired_comparisons": rare_comparisons,
        "primary_gate": gate,
        "per_run": per_mode,
        "claim_boundary": (
            "CounterFact is a counterfactual-edit stress test; its official "
            "neighborhood locality and frozen graph retention are separate outcomes."
        ),
    }
    json_path = output_base / "counterfact_confirmation_summary.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# CounterFact 30-run Confirmation",
        "",
        "Scope: 3 entity-disjoint B=25 batches × 2 seeds × 5 arms.",
        "",
        "| Mode | Runs | Update | Paraphrase | Neighborhood FR | Graph FR | Rare-probe FR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        row = aggregate[mode]
        lines.append(
            f"| {mode} | {row['runs']} | "
            f"{row['update_success_mean']:.3f} | "
            f"{row['paraphrase_success_mean']:.3f} | "
            f"{row['neighborhood_flip_rate_mean']:.3f} | "
            f"{row['graph_flip_rate_mean']:.3f} | "
            f"{row['graph_flip_rate_by_stratum']['rare']:.3f} |"
        )
    lines.extend(["", "## Popular paired comparisons", ""])
    for label, comparisons in (
        ("CounterFact neighborhood Flip Rate", native_comparisons),
        ("Graph Flip Rate", graph_comparisons),
        ("Rare-probe Graph Flip Rate", rare_comparisons),
    ):
        lines.append(f"### {label}")
        for name, row in comparisons.items():
            lines.append(
                f"- {name}: {row['mean_difference']:+.3f}; "
                f"95% CI={row['bootstrap_95_ci']}; "
                f"wins={row['popular_wins']}/{row['n']}"
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
            "- CounterFact tests counterfactual rather than temporal updates.",
            "- Official neighborhood locality and frozen graph retention are "
            "reported separately.",
        ]
    )
    markdown_path = output_base / "counterfact_confirmation_summary.md"
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
