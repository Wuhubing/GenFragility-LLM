"""Analyze WikiFactDiff confirmation outcomes by graph-probe stratum."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


MODES = ("none", "popular", "random", "rare", "random_distance")
STRATA = ("popular", "middle", "rare")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def bootstrap_ci(values: list[float], seed: int = 42) -> list[float]:
    rng = random.Random(seed)
    means = []
    for _ in range(10000):
        sample = [rng.choice(values) for _ in values]
        means.append(mean(sample))
    means.sort()
    return [means[249], means[9749]]


def graph_metrics(report: dict) -> dict:
    rows = report["results"]

    def summarize(items: list[dict]) -> dict:
        clean_correct = sum(row["clean_correct"] for row in items)
        flips = sum(
            row["clean_correct"] and not row["edited_correct"] for row in items
        )
        return {
            "clean_correct": clean_correct,
            "flip_rate": flips / clean_correct,
            "expected_logprob_change": mean(
                [row["expected_logprob_change"] for row in items]
            ),
        }

    result = {"overall": summarize(rows), "strata": {}}
    for stratum in STRATA:
        category = f"graph_probe_{stratum}"
        result["strata"][stratum] = summarize(
            [row for row in rows if row["category"] == category]
        )
    return result


def native_metrics(report: dict) -> dict:
    update = report["summary"]["update_new"]
    neighborhood = report["summary"]["ripple_d1"]
    return {
        "update_success": update["edited_accuracy"],
        "d1_flip_rate": neighborhood["flip_rate"],
        "d1_clean_correct": int(
            neighborhood["clean_accuracy"] * neighborhood["count"]
        ),
        "d1_expected_logprob_change": neighborhood[
            "expected_logprob_change_mean"
        ],
    }


def load_runs(output_base: Path) -> dict:
    runs = defaultdict(dict)
    for mode in MODES:
        paths = sorted(
            output_base.glob(
                f"seed*/wikifactdiff/{mode}/*/evaluation_strict.json"
            )
        )
        if len(paths) != 6:
            raise SystemExit(
                f"Expected 6 completed runs for {mode}, got {len(paths)}"
            )
        for native_path in paths:
            seed = native_path.parents[3].name
            batch = native_path.parent.name
            graph_path = native_path.parent / "graph_probe_evaluation.json"
            run_id = f"{seed}/{batch}"
            runs[mode][run_id] = {
                "seed": seed,
                "batch": batch,
                "native": native_metrics(json.loads(native_path.read_text())),
                "graph": graph_metrics(json.loads(graph_path.read_text())),
            }
    return dict(runs)


def aggregate(runs: dict) -> dict:
    result = {}
    for mode, mode_runs in runs.items():
        rows = list(mode_runs.values())
        result[mode] = {
            "runs": len(rows),
            "update_success": mean(
                [row["native"]["update_success"] for row in rows]
            ),
            "native_d1_flip_rate": mean(
                [row["native"]["d1_flip_rate"] for row in rows]
            ),
            "graph_flip_rate": mean(
                [row["graph"]["overall"]["flip_rate"] for row in rows]
            ),
            "graph_expected_logprob_change": mean(
                [
                    row["graph"]["overall"]["expected_logprob_change"]
                    for row in rows
                ]
            ),
            "graph_flip_rate_by_stratum": {
                stratum: mean(
                    [
                        row["graph"]["strata"][stratum]["flip_rate"]
                        for row in rows
                    ]
                )
                for stratum in STRATA
            },
        }
    return result


def paired_strata(runs: dict) -> dict:
    result = {}
    for control in ("none", "random", "rare", "random_distance"):
        control_result = {}
        common = sorted(set(runs["popular"]) & set(runs[control]))
        for stratum in ("overall", *STRATA):
            differences = []
            for run_id in common:
                popular_graph = runs["popular"][run_id]["graph"]
                control_graph = runs[control][run_id]["graph"]
                if stratum == "overall":
                    popular_rate = popular_graph["overall"]["flip_rate"]
                    control_rate = control_graph["overall"]["flip_rate"]
                else:
                    popular_rate = popular_graph["strata"][stratum]["flip_rate"]
                    control_rate = control_graph["strata"][stratum]["flip_rate"]
                differences.append(popular_rate - control_rate)
            control_result[stratum] = {
                "mean_difference": mean(differences),
                "bootstrap_95_ci": bootstrap_ci(differences),
                "popular_wins": sum(value < 0 for value in differences),
                "n": len(differences),
            }
        result[f"popular_minus_{control}"] = control_result
    return result


def batch_sensitivity(runs: dict) -> dict:
    result = {}
    batches = sorted(
        {row["batch"] for row in runs["popular"].values()}
    )
    for batch in batches:
        result[batch] = {}
        for mode in MODES:
            rows = [
                row
                for row in runs[mode].values()
                if row["batch"] == batch
            ]
            result[batch][mode] = {
                "graph_flip_rate": mean(
                    [row["graph"]["overall"]["flip_rate"] for row in rows]
                ),
                "native_d1_flip_rate": mean(
                    [row["native"]["d1_flip_rate"] for row in rows]
                ),
                "update_success": mean(
                    [row["native"]["update_success"] for row in rows]
                ),
            }
    return result


def render_markdown(report: dict) -> str:
    aggregate_rows = report["aggregate"]
    lines = [
        "# WikiFactDiff Probe-Stratum Analysis",
        "",
        "Scope: 3 entity-disjoint B=25 batches × 2 seeds × 5 arms.",
        "",
        "## Graph-holdout Flip Rate",
        "",
        "| Mode | Overall | Popular probes | Middle probes | Rare probes |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        row = aggregate_rows[mode]
        strata = row["graph_flip_rate_by_stratum"]
        lines.append(
            f"| {mode} | {row['graph_flip_rate']:.3f} | "
            f"{strata['popular']:.3f} | {strata['middle']:.3f} | "
            f"{strata['rare']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Popular paired differences",
            "",
            "Negative means Popular has lower Flip Rate.",
            "",
        ]
    )
    for control, strata in report["paired_comparisons"].items():
        lines.append(f"### {control}")
        for stratum, row in strata.items():
            lines.append(
                f"- {stratum}: {row['mean_difference']:+.3f}; "
                f"95% CI={row['bootstrap_95_ci']}; "
                f"wins={row['popular_wins']}/{row['n']}"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Popular is directionally best overall, but its overall paired "
            "intervals against Random, Rare, and distance-matched Random cross zero.",
            "- Probe-stratum results identify whether any aggregate advantage is "
            "concentrated rather than uniform.",
            "- Native d1 locality and graph-holdout retention remain separate outcomes.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-base", type=Path, required=True)
    args = parser.parse_args()
    runs = load_runs(args.output_base)
    report = {
        "aggregate": aggregate(runs),
        "paired_comparisons": paired_strata(runs),
        "batch_sensitivity": batch_sensitivity(runs),
        "per_run": runs,
    }
    json_path = args.output_base / "wfd_probe_strata_analysis.json"
    markdown_path = args.output_base / "wfd_probe_strata_analysis.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path.write_text(render_markdown(report))
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
