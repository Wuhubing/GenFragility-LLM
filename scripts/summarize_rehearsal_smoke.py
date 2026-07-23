"""Aggregate completed WikiFactDiff and WikiBigEdit smoke evaluations."""
from __future__ import annotations

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-base", type=Path, required=True)
    args = parser.parse_args()

    modes = ("none", "popular", "rare", "random")
    combined = {}
    missing = []
    for dataset in ("wikifactdiff", "wikibigedit"):
        combined[dataset] = {}
        for mode in modes:
            reports = []
            mode_dir = args.output_base / dataset / mode
            for path in sorted(mode_dir.glob("*/evaluation.json")):
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
