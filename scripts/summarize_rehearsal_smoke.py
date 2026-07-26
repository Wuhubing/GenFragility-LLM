"""Aggregate completed WikiFactDiff and WikiBigEdit smoke evaluations."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def aggregate(reports: list[dict]) -> dict:
    totals = defaultdict(
        lambda: {
            "count": 0,
            "clean_correct": 0.0,
            "edited_correct": 0.0,
            "flips": 0,
            "clean_margin_sum": 0.0,
            "edited_margin_sum": 0.0,
        }
    )
    for report in reports:
        for category, metrics in report["summary"].items():
            count = metrics["count"]
            row = totals[category]
            row["count"] += count
            row["clean_correct"] += metrics["clean_accuracy"] * count
            row["edited_correct"] += metrics["edited_accuracy"] * count
            row["flips"] += metrics["flip_count"]
            row["clean_margin_sum"] += metrics["clean_margin_mean"] * count
            row["edited_margin_sum"] += metrics["edited_margin_mean"] * count

    result = {}
    for category, row in totals.items():
        count = row["count"]
        clean_correct = row["clean_correct"]
        result[category] = {
            "count": count,
            "clean_accuracy": clean_correct / count,
            "edited_accuracy": row["edited_correct"] / count,
            "accuracy_change": (row["edited_correct"] - clean_correct) / count,
            "flip_count": row["flips"],
            "flip_rate": row["flips"] / clean_correct if clean_correct else None,
            "clean_margin_mean": row["clean_margin_sum"] / count,
            "edited_margin_mean": row["edited_margin_sum"] / count,
        }
    return result


def graph_probe_metrics(report: dict) -> dict:
    rows = [
        row for row in report["results"] if row["category"].startswith("graph_probe_")
    ]
    clean_correct = sum(row["clean_correct"] for row in rows)
    flips = sum(
        row["clean_correct"] and not row["edited_correct"] for row in rows
    )
    by_stratum = {}
    for stratum in ("popular", "middle", "rare"):
        stratum_rows = [
            row for row in rows if row["category"] == f"graph_probe_{stratum}"
        ]
        stratum_clean = sum(row["clean_correct"] for row in stratum_rows)
        stratum_flips = sum(
            row["clean_correct"] and not row["edited_correct"]
            for row in stratum_rows
        )
        by_stratum[stratum] = {
            "count": len(stratum_rows),
            "clean_correct": stratum_clean,
            "flip_count": stratum_flips,
            "flip_rate": (
                stratum_flips / stratum_clean if stratum_clean else None
            ),
        }
    return {
        "count": len(rows),
        "clean_correct": clean_correct,
        "flip_count": flips,
        "flip_rate": flips / clean_correct if clean_correct else None,
        "expected_logprob_change": sum(
            row["expected_logprob_change"] for row in rows
        )
        / len(rows),
        "by_stratum": by_stratum,
    }


def bootstrap_mean_ci(values: list[float], seed: int = 42) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(10000):
        sample = [rng.choice(values) for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return [means[249], means[9749]]


def summarize_graph_probe(
    output_base: Path,
    include_seed_subdirs: bool,
    modes: tuple[str, ...],
) -> None:
    by_mode = {}
    per_unit = {}
    for mode in modes:
        if include_seed_subdirs:
            reports = sorted(
                output_base.glob(
                    f"seed*/wikibigedit/{mode}/*/graph_probe_evaluation.json"
                )
            )
        else:
            mode_dir = output_base / "wikibigedit" / mode
            reports = sorted(mode_dir.glob("*/graph_probe_evaluation.json"))
        if not reports:
            raise SystemExit(f"Missing graph probe reports for {mode}")
        metrics = {}
        for path in reports:
            unit_id = path.parent.name
            if include_seed_subdirs:
                seed_id = path.parents[3].name
                run_id = f"{seed_id}/{unit_id}"
            else:
                run_id = unit_id
            probe_metrics = graph_probe_metrics(json.loads(path.read_text()))
            update_path = path.parent / "evaluation_strict.json"
            update_report = json.loads(update_path.read_text())
            update = update_report["summary"]["update_new"]
            probe_metrics["update_success"] = update["edited_accuracy"]
            metrics[run_id] = probe_metrics
        per_unit[mode] = metrics
        by_mode[mode] = {
            "runs": len(metrics),
            "minimum_clean_correct": min(
                row["clean_correct"] for row in metrics.values()
            ),
            "flip_rate_mean": sum(
                row["flip_rate"] for row in metrics.values()
            )
            / len(metrics),
            "update_success_mean": sum(
                row["update_success"] for row in metrics.values()
            )
            / len(metrics),
            "expected_logprob_change_mean": sum(
                row["expected_logprob_change"] for row in metrics.values()
            )
            / len(metrics),
            "by_stratum": {
                stratum: {
                    "clean_correct": sum(
                        row["by_stratum"][stratum]["clean_correct"]
                        for row in metrics.values()
                    ),
                    "flip_rate": (
                        sum(
                            row["by_stratum"][stratum]["flip_count"]
                            for row in metrics.values()
                        )
                        / sum(
                            row["by_stratum"][stratum]["clean_correct"]
                            for row in metrics.values()
                        )
                    ),
                }
                for stratum in ("popular", "middle", "rare")
            },
        }

    pilot_gate = {
        "update_success_at_least_0_90": all(
            metrics["update_success_mean"] >= 0.90
            for metrics in by_mode.values()
        ),
        "popular_lower_than_available_controls": all(
            by_mode["popular"]["flip_rate_mean"]
            < by_mode[control]["flip_rate_mean"]
            for control in modes
            if control != "popular"
        ),
        "popular_update_success_within_0_05": (
            max(metrics["update_success_mean"] for metrics in by_mode.values())
            - by_mode["popular"]["update_success_mean"]
            <= 0.05
        ),
        "at_least_300_clean_correct_probes": all(
            metrics["minimum_clean_correct"] >= 300
            for metrics in by_mode.values()
        ),
    }
    pilot_gate["passed"] = all(pilot_gate.values())

    comparisons = {}
    for control in modes:
        if control == "popular":
            continue
        common = sorted(set(per_unit["popular"]) & set(per_unit[control]))
        differences = [
            per_unit["popular"][unit_id]["flip_rate"]
            - per_unit[control][unit_id]["flip_rate"]
            for unit_id in common
        ]
        comparisons[f"popular_minus_{control}"] = {
            "n_paired_units": len(common),
            "mean_difference": sum(differences) / len(differences),
            "bootstrap_95_ci": bootstrap_mean_ci(differences),
        }

    primary = comparisons["popular_minus_random"]
    primary_ci = primary["bootstrap_95_ci"]
    final_claim = {
        "enough_paired_units": primary["n_paired_units"] >= 15,
        "ci_entirely_below_zero": bool(primary_ci and primary_ci[1] < 0),
        "effect_at_least_0_05": primary["mean_difference"] <= -0.05,
        "update_success_preserved": (
            by_mode["popular"]["update_success_mean"] >= 0.90
            and max(
                metrics["update_success_mean"] for metrics in by_mode.values()
            )
            - by_mode["popular"]["update_success_mean"]
            <= 0.05
        ),
    }
    final_claim["supported"] = all(final_claim.values())
    confirmation_gate = {
        "enough_paired_units": primary["n_paired_units"] >= 6,
        "ci_entirely_below_zero": bool(primary_ci and primary_ci[1] < 0),
        "effect_at_least_0_05": primary["mean_difference"] <= -0.05,
        "update_success_preserved": final_claim["update_success_preserved"],
    }
    confirmation_gate["passed"] = all(confirmation_gate.values())

    result = {
        "modes": by_mode,
        "paired_comparisons": comparisons,
        "pilot_gate": pilot_gate,
        "confirmation_gate": confirmation_gate,
        "final_claim": final_claim,
        "per_unit": per_unit,
    }
    json_path = output_base / "graph_probe_summary.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Popular Anchor Graph Probe Summary",
        "",
        "| Mode | Runs | Flip rate | Update success | Expected-logprob change |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, metrics in by_mode.items():
        lines.append(
            f"| {mode} | {metrics['runs']} | "
            f"{metrics['flip_rate_mean']:.3f} | "
            f"{metrics['update_success_mean']:.3f} | "
            f"{metrics['expected_logprob_change_mean']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Flip rate by probe stratum",
            "",
            "| Mode | Popular | Middle | Rare |",
            "|---|---:|---:|---:|",
        ]
    )
    for mode, metrics in by_mode.items():
        strata = metrics["by_stratum"]
        lines.append(
            f"| {mode} | {strata['popular']['flip_rate']:.3f} | "
            f"{strata['middle']['flip_rate']:.3f} | "
            f"{strata['rare']['flip_rate']:.3f} |"
        )
    lines.extend(["", "## Paired comparisons", ""])
    for name, metrics in comparisons.items():
        lines.append(
            f"- {name}: {metrics['mean_difference']:+.3f}; "
            f"95% CI={metrics['bootstrap_95_ci']}; "
            f"n={metrics['n_paired_units']}"
        )
    lines.extend(
        [
            "",
            "## Pilot gate",
            "",
            f"- Passed: {pilot_gate['passed']}",
        ]
    )
    for criterion, passed in pilot_gate.items():
        if criterion != "passed":
            lines.append(f"- {criterion}: {passed}")
    lines.extend(
        [
            "",
            "## Medium confirmation gate",
            "",
            f"- Passed: {confirmation_gate['passed']}",
        ]
    )
    for criterion, passed in confirmation_gate.items():
        if criterion != "passed":
            lines.append(f"- {criterion}: {passed}")
    lines.extend(
        [
            "",
            "## Final claim gate",
            "",
            f"- Supported: {final_claim['supported']}",
        ]
    )
    for criterion, passed in final_claim.items():
        if criterion != "supported":
            lines.append(f"- {criterion}: {passed}")
    markdown_path = output_base / "graph_probe_summary.md"
    markdown_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


def summarize_mquake_pilot(output_base: Path) -> None:
    modes = ("none", "popular", "random")
    metrics = {}
    for mode in modes:
        mode_dir = output_base / "mquake_t" / mode
        native_path = next(mode_dir.glob("*/evaluation_strict.json"), None)
        probe_path = next(mode_dir.glob("*/graph_probe_evaluation.json"), None)
        if native_path is None or probe_path is None:
            raise SystemExit(f"Missing MQuAKE-T reports for {mode}")
        native = json.loads(native_path.read_text())
        probe = graph_probe_metrics(json.loads(probe_path.read_text()))
        update = native["summary"]["update_new"]
        multihop = native["summary"].get("multihop_new", {})
        single_hop = native["summary"].get("single_hop_new", {})
        metrics[mode] = {
            "update_success": update["edited_accuracy"],
            "multihop_success": multihop.get("edited_accuracy"),
            "single_hop_success": single_hop.get("edited_accuracy"),
            "holdout_flip_rate": probe["flip_rate"],
            "clean_correct_probes": probe["clean_correct"],
            "expected_logprob_change": probe["expected_logprob_change"],
        }

    gate = {
        "update_success_at_least_0_90": all(
            row["update_success"] >= 0.90 for row in metrics.values()
        ),
        "popular_multihop_noninferior_0_05": (
            metrics["popular"]["multihop_success"] is not None
            and metrics["random"]["multihop_success"] is not None
            and metrics["popular"]["multihop_success"]
            >= metrics["random"]["multihop_success"] - 0.05
        ),
        "popular_holdout_improves_0_03": (
            metrics["popular"]["holdout_flip_rate"]
            <= metrics["random"]["holdout_flip_rate"] - 0.03
        ),
        "at_least_300_clean_correct_probes": all(
            row["clean_correct_probes"] >= 300 for row in metrics.values()
        ),
    }
    gate["passed"] = all(gate.values())
    result = {"modes": metrics, "pilot_gate": gate}
    json_path = output_base / "mquake_pilot_summary.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# MQuAKE-T Rehearsal Pilot",
        "",
        "| Mode | Update success | Single-hop | Multi-hop | Holdout flip |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, row in metrics.items():
        lines.append(
            f"| {mode} | {row['update_success']:.3f} | "
            f"{row['single_hop_success']:.3f} | "
            f"{row['multihop_success']:.3f} | "
            f"{row['holdout_flip_rate']:.3f} |"
        )
    lines.extend(["", "## Pilot gate", "", f"- Passed: {gate['passed']}"])
    for criterion, passed in gate.items():
        if criterion != "passed":
            lines.append(f"- {criterion}: {passed}")
    markdown_path = output_base / "mquake_pilot_summary.md"
    markdown_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


def summarize_mquake_preflight(
    output_base: Path,
    candidate_manifest: Path,
    precheck_report: Path,
) -> None:
    candidates = json.loads(candidate_manifest.read_text())
    precheck = json.loads(precheck_report.read_text())
    unit_id = next(iter(candidates["units"]))
    eligibility = precheck["units"][unit_id]["eligibility"]
    eligible = [
        update
        for update in candidates["units"][unit_id]["updates"]
        if eligibility.get(update["update_id"], False)
    ]
    selected = []
    used_entities = set()
    for update in eligible:
        entities = {
            update["head"],
            update["tail"],
            update["poison_answer"],
        }
        if entities & used_entities:
            continue
        selected.append(update)
        used_entities.update(entities)
    gate = {
        "at_least_25_strict_eligible_conflict_free_updates": len(selected) >= 25,
        "passed": False,
    }
    result = {
        "status": "stopped_before_training",
        "candidate_updates": len(candidates["units"][unit_id]["updates"]),
        "strict_eligible_updates": len(eligible),
        "strict_eligible_conflict_free_updates": len(selected),
        "pilot_gate": gate,
        "reason": (
            "Official MQuAKE-T does not provide 25 strict base-model-eligible, "
            "entity-disjoint updates for the preregistered B=25 pilot."
        ),
    }
    json_path = output_base / "mquake_pilot_summary.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    markdown_path = output_base / "mquake_pilot_summary.md"
    markdown_path.write_text(
        "\n".join(
            [
                "# MQuAKE-T Rehearsal Pilot",
                "",
                "- Status: stopped before training",
                f"- Candidate updates: {result['candidate_updates']}",
                f"- Strict eligible updates: {len(eligible)}",
                f"- Conflict-free strict eligible updates: {len(selected)}",
                "- Required batch size: 25",
                "- Pilot gate: FAIL",
                "",
                result["reason"],
            ]
        )
        + "\n"
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


def summarize_wfd_replication(output_base: Path) -> None:
    modes = ("none", "popular", "random")
    per_mode = {}
    aggregate = {}
    for mode in modes:
        reports = sorted(
            output_base.glob(
                f"seed*/wikifactdiff/{mode}/*/evaluation_strict.json"
            )
        )
        if not reports:
            raise SystemExit(f"Missing WikiFactDiff reports for {mode}")
        rows = {}
        for path in reports:
            seed_id = path.parents[3].name
            unit_id = path.parent.name
            report = json.loads(path.read_text())
            update = report["summary"]["update_new"]
            neighborhood = report["summary"].get("ripple_d1")
            rows[f"{seed_id}/{unit_id}"] = {
                "update_success": update["edited_accuracy"],
                "neighborhood_flip_rate": (
                    neighborhood["flip_rate"] if neighborhood else None
                ),
            }
        per_mode[mode] = rows
        valid_flip_rates = [
            row["neighborhood_flip_rate"]
            for row in rows.values()
            if row["neighborhood_flip_rate"] is not None
        ]
        aggregate[mode] = {
            "runs": len(rows),
            "update_success_mean": sum(
                row["update_success"] for row in rows.values()
            )
            / len(rows),
            "neighborhood_flip_rate_mean": (
                sum(valid_flip_rates) / len(valid_flip_rates)
                if valid_flip_rates
                else None
            ),
        }
    common = sorted(set(per_mode["popular"]) & set(per_mode["random"]))
    differences = [
        per_mode["popular"][run_id]["neighborhood_flip_rate"]
        - per_mode["random"][run_id]["neighborhood_flip_rate"]
        for run_id in common
        if per_mode["popular"][run_id]["neighborhood_flip_rate"] is not None
        and per_mode["random"][run_id]["neighborhood_flip_rate"] is not None
    ]
    confidence_interval = bootstrap_mean_ci(differences)
    gate = {
        "six_paired_batch_seed_units": len(differences) >= 6,
        "popular_better_mean": bool(
            differences and sum(differences) / len(differences) < 0
        ),
        "ci_entirely_below_zero": bool(
            confidence_interval and confidence_interval[1] < 0
        ),
        "update_success_preserved": (
            aggregate["popular"]["update_success_mean"] >= 0.90
            and max(
                row["update_success_mean"] for row in aggregate.values()
            )
            - aggregate["popular"]["update_success_mean"]
            <= 0.05
        ),
    }
    gate["passed"] = all(gate.values())
    result = {
        "modes": aggregate,
        "popular_minus_random": {
            "n_paired_units": len(differences),
            "mean_difference": (
                sum(differences) / len(differences) if differences else None
            ),
            "bootstrap_95_ci": confidence_interval,
        },
        "replication_gate": gate,
        "per_unit": per_mode,
    }
    json_path = output_base / "wfd_replication_summary.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# WikiFactDiff Rehearsal Replication",
        "",
        "| Mode | Runs | Update success | d1 flip rate |",
        "|---|---:|---:|---:|",
    ]
    for mode, row in aggregate.items():
        lines.append(
            f"| {mode} | {row['runs']} | "
            f"{row['update_success_mean']:.3f} | "
            f"{row['neighborhood_flip_rate_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Replication gate",
            "",
            f"- Passed: {gate['passed']}",
        ]
    )
    for criterion, passed in gate.items():
        if criterion != "passed":
            lines.append(f"- {criterion}: {passed}")
    markdown_path = output_base / "wfd_replication_summary.md"
    markdown_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


def summarize_external_validation(output_base: Path) -> None:
    stage_paths = {
        "mquake_t": (
            output_base
            / "mquake_t/pilot/seed42/mquake_pilot_summary.json"
        ),
        "wikibigedit": output_base / "wbe_confirm/graph_probe_summary.json",
        "wikifactdiff": (
            output_base / "wfd_replication/wfd_replication_summary.json"
        ),
    }
    stages = {
        stage: json.loads(path.read_text())
        for stage, path in stage_paths.items()
        if path.is_file()
    }
    mquake_passed = stages.get("mquake_t", {}).get("pilot_gate", {}).get(
        "passed", False
    )
    wbe_passed = stages.get("wikibigedit", {}).get(
        "confirmation_gate", {}
    ).get("passed", False)
    wfd_passed = stages.get("wikifactdiff", {}).get(
        "replication_gate", {}
    ).get("passed", False)
    if wbe_passed and wfd_passed:
        claim = (
            "Topology-aware Popular rehearsal is supported across the WBE "
            "batched-update confirmation and the WikiFactDiff replacement "
            "replication under the preregistered gates."
        )
    elif wbe_passed:
        claim = (
            "Popular rehearsal is supported for WBE-style batched updates; "
            "cross-setting generalization to WikiFactDiff is not established."
        )
    elif mquake_passed:
        claim = (
            "MQuAKE-T supports a directional multi-hop mechanism signal, but "
            "the external batched-update maintenance claim is not established."
        )
    else:
        claim = (
            "External validation did not pass the preregistered gate; retain "
            "only the internal-graph mitigation claim."
        )
    result = {
        "completed_stages": list(stages),
        "gates": {
            "mquake_t_pilot": mquake_passed,
            "wikibigedit_confirmation": wbe_passed,
            "wikifactdiff_replication": wfd_passed,
        },
        "paper_claim_boundary": claim,
        "frozen_negative_evidence": [
            "The earlier WBE result is one batch and one seed with 293/300 "
            "clean-correct probes.",
            "Earlier WikiFactDiff experiments do not show a stable "
            "Popular-over-Random advantage.",
        ],
    }
    json_path = output_base / "external_validation_summary.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# External Rehearsal Validation Summary",
        "",
        f"- MQuAKE-T pilot gate: {mquake_passed}",
        f"- WikiBigEdit confirmation gate: {wbe_passed}",
        f"- WikiFactDiff replication gate: {wfd_passed}",
        "",
        "## Paper claim boundary",
        "",
        claim,
        "",
        "Negative and inconclusive evidence remains part of the record.",
    ]
    markdown_path = output_base / "external_validation_summary.md"
    markdown_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--graph-probe", action="store_true")
    parser.add_argument("--mquake-pilot", action="store_true")
    parser.add_argument("--mquake-preflight", action="store_true")
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--precheck-report", type=Path)
    parser.add_argument("--wfd-replication", action="store_true")
    parser.add_argument("--external-summary", action="store_true")
    parser.add_argument("--include-seed-subdirs", action="store_true")
    parser.add_argument(
        "--modes",
        default="none,popular,rare,random,generic",
    )
    args = parser.parse_args()
    if args.graph_probe:
        modes = tuple(mode for mode in args.modes.split(",") if mode)
        summarize_graph_probe(
            args.output_base,
            args.include_seed_subdirs,
            modes,
        )
        return
    if args.mquake_pilot:
        summarize_mquake_pilot(args.output_base)
        return
    if args.mquake_preflight:
        if not args.candidate_manifest or not args.precheck_report:
            parser.error(
                "--mquake-preflight requires --candidate-manifest "
                "and --precheck-report"
            )
        summarize_mquake_preflight(
            args.output_base,
            args.candidate_manifest,
            args.precheck_report,
        )
        return
    if args.wfd_replication:
        summarize_wfd_replication(args.output_base)
        return
    if args.external_summary:
        summarize_external_validation(args.output_base)
        return

    modes = ("none", "popular", "rare", "random")
    combined = {}
    missing = []
    for dataset in ("wikifactdiff", "wikibigedit"):
        combined[dataset] = {}
        for mode in modes:
            reports = []
            mode_dir = args.output_base / dataset / mode
            for path in sorted(mode_dir.glob("*/evaluation_strict.json")):
                reports.append(json.loads(path.read_text()))
            if not reports:
                missing.append(f"{dataset}/{mode}")
                continue
            combined[dataset][mode] = {
                "runs": len(reports),
                "categories": aggregate(reports),
            }

    if missing:
        raise SystemExit(f"Missing evaluation reports: {', '.join(missing)}")

    json_path = args.output_base / "summary.json"
    json_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# Rehearsal Smoke Summary",
        "",
        "| Dataset | Mode | Category | N | Clean acc. | Edited acc. | "
        "Accuracy change | Flip rate | Margin change |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, modes_data in combined.items():
        for mode, mode_data in modes_data.items():
            for category, metrics in mode_data["categories"].items():
                flip_rate = metrics["flip_rate"]
                lines.append(
                    f"| {dataset} | {mode} | {category} | {metrics['count']} | "
                    f"{metrics['clean_accuracy']:.3f} | "
                    f"{metrics['edited_accuracy']:.3f} | "
                    f"{metrics['accuracy_change']:+.3f} | "
                    f"{flip_rate:.3f} | "
                    f"{metrics['edited_margin_mean'] - metrics['clean_margin_mean']:+.3f} |"
                    if flip_rate is not None
                    else (
                        f"| {dataset} | {mode} | {category} | {metrics['count']} | "
                        f"{metrics['clean_accuracy']:.3f} | "
                        f"{metrics['edited_accuracy']:.3f} | "
                        f"{metrics['accuracy_change']:+.3f} | n/a | "
                        f"{metrics['edited_margin_mean'] - metrics['clean_margin_mean']:+.3f} |"
                    )
                )
    markdown_path = args.output_base / "summary.md"
    markdown_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
