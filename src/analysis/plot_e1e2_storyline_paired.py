#!/usr/bin/env python3
"""Generate E1/E2/sanity figures for paired Hub vs Low experiments.

Default setup targets:
- Hub main report: v2-006 paired sampled30hop
- Low main report: v2-007 paired sampled30hop
- Irrelevant sanity reports for both models

Mask for E1/E2:
    clean_accuracy == 1 and clean_correct_token_rank == 1
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from statistics import mean

import matplotlib.pyplot as plt


def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["unified_results"]


def is_main_hop(row):
    return row.get("distance") in {"d1", "d2", "d3", "d4", "d5"}


def mask_clean_correct_rank1(row):
    return (
        is_main_hop(row)
        and row.get("clean_accuracy") == 1
        and row.get("clean_correct_token_rank") == 1
    )


def e1_margin_values(rows):
    out = []
    for r in rows:
        if mask_clean_correct_rank1(r):
            v = r.get("clean_margin")
            if v is not None:
                out.append(v)
    return out


def e2_by_hop(rows):
    by_hop = defaultdict(list)
    for r in rows:
        if mask_clean_correct_rank1(r):
            by_hop[r["distance"]].append(r)

    metrics = {}
    for hop in ["d1", "d2", "d3", "d4", "d5"]:
        subset = by_hop.get(hop, [])
        if not subset:
            metrics[hop] = {"n": 0, "c2w_rate": 0.0, "abs_margin_change": 0.0}
            continue
        c2w = sum(1 for r in subset if r.get("poisoned_accuracy") == 0) / len(subset)
        abs_margin = [
            abs(r["margin_change"]) for r in subset if r.get("margin_change") is not None
        ]
        metrics[hop] = {
            "n": len(subset),
            "c2w_rate": c2w,
            "abs_margin_change": mean(abs_margin) if abs_margin else 0.0,
        }
    return metrics


def sanity_acc(rows):
    clean = [r.get("clean_accuracy") for r in rows if r.get("clean_accuracy") is not None]
    poisoned = [
        r.get("poisoned_accuracy")
        for r in rows
        if r.get("poisoned_accuracy") is not None
    ]
    return {
        "clean_acc": mean(clean) if clean else 0.0,
        "poisoned_acc": mean(poisoned) if poisoned else 0.0,
    }


def plot_e1_boxplot(hub_margins, low_margins, out_png):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    bp = ax.boxplot(
        [hub_margins, low_margins],
        tick_labels=["Hub", "Low-tail"],
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "black", "linewidth": 1.3},
    )
    colors = ["#d94841", "#2b6cb0"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.32)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)

    ax.set_title("E1: Clean Margin (Top-1 Correct Mask)")
    ax.set_ylabel("Clean Margin")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=240)
    plt.close(fig)


def plot_e2_lines(hub_metrics, low_metrics, out_png):
    hops = [1, 2, 3, 4, 5]
    labels = [f"d{i}" for i in hops]

    hub_c2w = [hub_metrics[h]["c2w_rate"] for h in labels]
    low_c2w = [low_metrics[h]["c2w_rate"] for h in labels]
    hub_margin = [hub_metrics[h]["abs_margin_change"] for h in labels]
    low_margin = [low_metrics[h]["abs_margin_change"] for h in labels]

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.0), sharex=True)

    for ax in axes:
        ax.axvspan(0.5, 3.5, color="#ececec", alpha=0.8, zorder=0)
        ax.axvspan(3.5, 5.5, color="#fff7d6", alpha=0.9, zorder=0)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

    axes[0].plot(hops, hub_c2w, "-o", color="#d94841", linewidth=2.0, label="Hub")
    axes[0].plot(hops, low_c2w, "-o", color="#2b6cb0", linewidth=2.0, label="Low-tail")
    axes[0].set_ylabel("C→W Rate")
    axes[0].set_title("E2: Dynamic Propagation (Top-1 Correct Mask)")
    axes[0].legend(loc="upper right", frameon=False)
    axes[0].text(1.0, max(hub_c2w + low_c2w) * 0.95, "Blast Radius", fontsize=10)
    axes[0].text(4.05, max(hub_c2w + low_c2w) * 0.95, "Resonance Zone", fontsize=10)

    axes[1].plot(hops, hub_margin, "-o", color="#d94841", linewidth=2.0)
    axes[1].plot(hops, low_margin, "-o", color="#2b6cb0", linewidth=2.0)
    axes[1].set_ylabel("|Δ Margin|")
    axes[1].set_xlabel("Hop Distance")
    axes[1].set_xticks(hops, labels)

    fig.tight_layout()
    fig.savefig(out_png, dpi=240)
    plt.close(fig)


def plot_sanity_bar(hub_sanity, low_sanity, out_png):
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    x = [0, 1]
    w = 0.34
    clean_vals = [hub_sanity["clean_acc"], low_sanity["clean_acc"]]
    poisoned_vals = [hub_sanity["poisoned_acc"], low_sanity["poisoned_acc"]]

    ax.bar([i - w / 2 for i in x], clean_vals, width=w, color="#7f8c8d", label="Clean")
    ax.bar(
        [i + w / 2 for i in x],
        poisoned_vals,
        width=w,
        color="#16a085",
        label="Poisoned",
    )
    ax.set_xticks(x, ["Hub model", "Low-tail model"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy on Irrelevant-50")
    ax.set_title("Sanity Check: Global Capability Stability")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=240)
    plt.close(fig)


def write_curve_csv(out_csv, hub_metrics, low_metrics, hub_sanity, low_sanity):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "section",
                "group",
                "hop",
                "n_masked",
                "c2w_rate",
                "abs_margin_change",
                "clean_acc_irrelevant",
                "poisoned_acc_irrelevant",
            ]
        )
        for hop in ["d1", "d2", "d3", "d4", "d5"]:
            h = hub_metrics[hop]
            l = low_metrics[hop]
            w.writerow(["E2", "hub", hop, h["n"], h["c2w_rate"], h["abs_margin_change"], "", ""])
            w.writerow(["E2", "low", hop, l["n"], l["c2w_rate"], l["abs_margin_change"], "", ""])
        w.writerow(
            [
                "Sanity",
                "hub",
                "",
                "",
                "",
                "",
                hub_sanity["clean_acc"],
                hub_sanity["poisoned_acc"],
            ]
        )
        w.writerow(
            [
                "Sanity",
                "low",
                "",
                "",
                "",
                "",
                low_sanity["clean_acc"],
                low_sanity["poisoned_acc"],
            ]
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hub_main_report",
        default="main_output/integrated_experiment_20260308_143922_20260308_143922/direct_comparison_20260308_143922/comparison_reports/direct_comparison_comparison_20260308_143950.json",
    )
    parser.add_argument(
        "--low_main_report",
        default="main_output/integrated_experiment_20260308_144017_20260308_144017/direct_comparison_20260308_144017/comparison_reports/direct_comparison_comparison_20260308_144038.json",
    )
    parser.add_argument(
        "--hub_ir_report",
        default="main_output/integrated_experiment_20260308_144106_20260308_144106/direct_comparison_20260308_144106/comparison_reports/direct_comparison_comparison_20260308_144117.json",
    )
    parser.add_argument(
        "--low_ir_report",
        default="main_output/integrated_experiment_20260308_144140_20260308_144140/direct_comparison_20260308_144140/comparison_reports/direct_comparison_comparison_20260308_144152.json",
    )
    parser.add_argument("--out_dir", default="artifacts/figures/e1_e2_storyline")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    hub_main = load_rows(args.hub_main_report)
    low_main = load_rows(args.low_main_report)
    hub_ir = load_rows(args.hub_ir_report)
    low_ir = load_rows(args.low_ir_report)

    hub_margins = e1_margin_values(hub_main)
    low_margins = e1_margin_values(low_main)
    hub_metrics = e2_by_hop(hub_main)
    low_metrics = e2_by_hop(low_main)
    hub_sanity = sanity_acc(hub_ir)
    low_sanity = sanity_acc(low_ir)

    plot_e1_boxplot(hub_margins, low_margins, os.path.join(args.out_dir, "fig_e1_margin_boxplot_maskB.png"))
    plot_e2_lines(hub_metrics, low_metrics, os.path.join(args.out_dir, "fig_e2_dynamic_lines_maskB.png"))
    plot_sanity_bar(hub_sanity, low_sanity, os.path.join(args.out_dir, "fig_sanity_irrelevant_bar.png"))
    write_curve_csv(
        os.path.join(args.out_dir, "figure_data_storyline_maskB.csv"),
        hub_metrics,
        low_metrics,
        hub_sanity,
        low_sanity,
    )

    summary = {
        "mask": "clean_accuracy == 1 and clean_correct_token_rank == 1",
        "counts": {
            "hub_e1_n": len(hub_margins),
            "low_e1_n": len(low_margins),
            "hub_e2_n_by_hop": {k: v["n"] for k, v in hub_metrics.items()},
            "low_e2_n_by_hop": {k: v["n"] for k, v in low_metrics.items()},
        },
        "e1_mean_margin": {
            "hub": mean(hub_margins) if hub_margins else None,
            "low": mean(low_margins) if low_margins else None,
        },
        "e2": {"hub": hub_metrics, "low": low_metrics},
        "sanity": {"hub": hub_sanity, "low": low_sanity},
        "outputs": {
            "fig_e1_margin_boxplot": "fig_e1_margin_boxplot_maskB.png",
            "fig_e2_dynamic_lines": "fig_e2_dynamic_lines_maskB.png",
            "fig_sanity_irrelevant_bar": "fig_sanity_irrelevant_bar.png",
            "figure_data_csv": "figure_data_storyline_maskB.csv",
        },
    }
    with open(
        os.path.join(args.out_dir, "storyline_figure_summary_maskB.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Saved figures to:", args.out_dir)


if __name__ == "__main__":
    main()
