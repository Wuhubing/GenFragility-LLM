#!/usr/bin/env python3
"""Analyze E1: margin dynamics (Hub vs Tail, pre/post edit, by distance).

This script is runnable on current comparison reports. It supports two modes:
1) Real margin mode: uses explicit logit fields if available.
2) Proxy mode (fallback): uses tail log-probability as a margin proxy.

Examples:
  python tools/analysis/analyze_margin_dynamics.py \
    --report main_output/.../comparison_reports/*comparison*.json

  python tools/analysis/analyze_margin_dynamics.py \
    --report main_output/.../comparison_reports/*comparison*.json \
    --out-dir artifacts/analysis/margin
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import math
import os
import pickle
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


DISTANCE_ORDER = ["d0", "d1", "d2", "d3", "d4", "d5"]


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _is_correct(item: dict, prefix: str) -> bool:
    acc = item.get(f"{prefix}_accuracy")
    em = item.get(f"{prefix}_exact_match")

    # Handles both [0,1] and [0,100] styles.
    acc_correct = False
    if isinstance(acc, (int, float)):
        acc_correct = (acc == 1) or (acc == 1.0) or (acc == 100) or (acc == 100.0)

    return bool(em) or acc_correct


def _pop_group(item: dict) -> str:
    # Best effort mapping; extendable once metadata format is finalized.
    for key in ["popularity_group", "pop_group", "node_group", "entity_group", "popularity"]:
        v = item.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()

    # Optional degree-based fallback.
    degree = item.get("target_degree") or item.get("degree")
    d = _safe_float(degree)
    if d is not None:
        if d >= 10:
            return "hub"
        if d <= 2:
            return "tail"
        return "mid"

    return "unknown"


def _load_graph(path: str):
    fp = path
    if not os.path.exists(fp) and os.path.exists(fp + ".gz"):
        fp = fp + ".gz"
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Graph file not found: {path}")
    if fp.endswith(".gz"):
        with gzip.open(fp, "rb") as f:
            data = pickle.load(f)
    else:
        with open(fp, "rb") as f:
            data = pickle.load(f)
    if isinstance(data, dict) and "graph" in data:
        return data["graph"]
    return data


def _build_degree_groups(graph_file: str, top_pct: float, bottom_pct: float) -> Tuple[Set[str], Set[str]]:
    graph = _load_graph(graph_file)
    in_degrees = dict(graph.in_degree())
    ranked = sorted(in_degrees.items(), key=lambda kv: (kv[1], str(kv[0])))
    n = len(ranked)
    if n == 0:
        return set(), set()
    top_k = max(1, int(n * top_pct))
    bot_k = max(1, int(n * bottom_pct))
    tail_nodes = {node for node, _ in ranked[:bot_k]}
    hub_nodes = {node for node, _ in ranked[-top_k:]}
    return hub_nodes, tail_nodes


def _resolve_pop_group(
    item: dict,
    hub_nodes: Optional[Set[str]],
    tail_nodes: Optional[Set[str]],
    entity_field: str,
) -> str:
    if hub_nodes is not None and tail_nodes is not None:
        entity = item.get(entity_field)
        if entity in hub_nodes:
            return "hub"
        if entity in tail_nodes:
            return "tail"
        return "mid"
    return _pop_group(item)


def _extract_margin(item: dict, prefix: str) -> Optional[float]:
    # Preferred explicit format:
    # clean_correct_logit - clean_top_incorrect_logit
    corr = _safe_float(item.get(f"{prefix}_correct_logit"))
    inc = _safe_float(item.get(f"{prefix}_top_incorrect_logit"))
    if corr is not None and inc is not None:
        return corr - inc

    # Alternative explicit margin field.
    direct_margin = _safe_float(item.get(f"{prefix}_margin"))
    if direct_margin is not None:
        return direct_margin

    # Fallback proxy: gold tail log-probability (available in current reports).
    proxy = _safe_float(item.get(f"{prefix}_tail_log_probability"))
    if proxy is not None:
        return proxy

    return None


def _iter_report_paths(report_args: Iterable[str]) -> List[str]:
    paths: List[str] = []
    for pattern in report_args:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            paths.extend(matches)
        elif os.path.exists(pattern):
            paths.append(pattern)
    return sorted(set(paths))


def _load_unified_results(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("unified_results", [])


def _mean(xs: List[float]) -> Optional[float]:
    return statistics.fmean(xs) if xs else None


def _median(xs: List[float]) -> Optional[float]:
    return statistics.median(xs) if xs else None


def _summarize(values: List[float]) -> Dict[str, Optional[float]]:
    clean_values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not clean_values:
        return {"n": 0, "mean": None, "median": None, "std": None}
    return {
        "n": len(clean_values),
        "mean": _mean(clean_values),
        "median": _median(clean_values),
        "std": statistics.pstdev(clean_values) if len(clean_values) > 1 else 0.0,
    }


def analyze_reports(
    report_paths: List[str],
    hub_nodes: Optional[Set[str]] = None,
    tail_nodes: Optional[Set[str]] = None,
    entity_field: str = "head",
) -> Dict[str, dict]:
    # buckets[(pop_group, distance)] -> {"clean": [], "poisoned": [], "delta": []}
    buckets: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: {"clean": [], "poisoned": [], "delta": [], "cw_delta": []}
    )
    mode_counts = defaultdict(int)
    overall: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"n_total": 0, "n_clean_correct": 0, "n_c2w": 0}
    )
    overall_values: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"clean": [], "poisoned": [], "delta": []}
    )

    for path in report_paths:
        rows = _load_unified_results(path)
        for item in rows:
            dist = item.get("distance", "unknown")
            pop = _resolve_pop_group(item, hub_nodes, tail_nodes, entity_field)

            clean_m = _extract_margin(item, "clean")
            poison_m = _extract_margin(item, "poisoned")
            if clean_m is None and poison_m is None:
                continue

            # Detect if this row used explicit margins or proxy.
            if item.get("clean_correct_logit") is not None and item.get("clean_top_incorrect_logit") is not None:
                mode_counts["explicit_logit_margin"] += 1
            elif item.get("clean_margin") is not None:
                mode_counts["explicit_margin_field"] += 1
            else:
                mode_counts["proxy_tail_logp"] += 1

            b = buckets[(pop, dist)]
            overall[pop]["n_total"] += 1
            if clean_m is not None:
                b["clean"].append(clean_m)
                overall_values[pop]["clean"].append(clean_m)
            if poison_m is not None:
                b["poisoned"].append(poison_m)
                overall_values[pop]["poisoned"].append(poison_m)
            if clean_m is not None and poison_m is not None:
                delta = poison_m - clean_m
                b["delta"].append(delta)
                overall_values[pop]["delta"].append(delta)
                # Strict C->W-only subset captures "fragility after edit".
                if _is_correct(item, "clean") and (not _is_correct(item, "poisoned")):
                    b["cw_delta"].append(delta)
                    overall[pop]["n_c2w"] += 1
            if _is_correct(item, "clean"):
                overall[pop]["n_clean_correct"] += 1

    summary = {
        "mode_counts": dict(mode_counts),
        "groups": {},
        "overall_by_group": {},
    }
    for (pop, dist), vals in sorted(buckets.items()):
        key = f"{pop}|{dist}"
        summary["groups"][key] = {
            "pop_group": pop,
            "distance": dist,
            "clean": _summarize(vals["clean"]),
            "poisoned": _summarize(vals["poisoned"]),
            "delta_poison_minus_clean": _summarize(vals["delta"]),
            "delta_poison_minus_clean_on_c_to_w": _summarize(vals["cw_delta"]),
        }

    for pop, meta in sorted(overall.items()):
        clean_correct = int(meta["n_clean_correct"])
        c2w = int(meta["n_c2w"])
        summary["overall_by_group"][pop] = {
            "n_total": int(meta["n_total"]),
            "n_clean_correct": clean_correct,
            "n_c2w": c2w,
            "c2w_rate": (c2w / clean_correct) if clean_correct > 0 else None,
            "clean": _summarize(overall_values[pop]["clean"]),
            "poisoned": _summarize(overall_values[pop]["poisoned"]),
            "delta_poison_minus_clean": _summarize(overall_values[pop]["delta"]),
        }
    return summary


def write_outputs(summary: dict, out_dir: str) -> Tuple[str, str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    json_path = os.path.join(out_dir, "margin_dynamics_summary.json")
    csv_path = os.path.join(out_dir, "margin_dynamics_summary.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "pop_group",
                "distance",
                "clean_n",
                "clean_mean",
                "poisoned_n",
                "poisoned_mean",
                "delta_n",
                "delta_mean",
                "cw_delta_n",
                "cw_delta_mean",
            ]
        )
        for row in summary["groups"].values():
            w.writerow(
                [
                    row["pop_group"],
                    row["distance"],
                    row["clean"]["n"],
                    row["clean"]["mean"],
                    row["poisoned"]["n"],
                    row["poisoned"]["mean"],
                    row["delta_poison_minus_clean"]["n"],
                    row["delta_poison_minus_clean"]["mean"],
                    row["delta_poison_minus_clean_on_c_to_w"]["n"],
                    row["delta_poison_minus_clean_on_c_to_w"]["mean"],
                ]
            )
    return json_path, csv_path


def print_console_table(summary: dict) -> None:
    print("=== Margin Dynamics Summary ===")
    print("mode_counts:", summary.get("mode_counts", {}))
    print("-" * 100)
    print(
        f"{'pop':<12} {'dist':<6} {'clean_mean':>12} {'poison_mean':>12} "
        f"{'delta_mean':>12} {'cw_delta_mean':>14} {'n_delta':>8}"
    )
    print("-" * 100)

    rows = list(summary["groups"].values())
    rows.sort(
        key=lambda r: (
            r["pop_group"],
            DISTANCE_ORDER.index(r["distance"]) if r["distance"] in DISTANCE_ORDER else math.inf,
            r["distance"],
        )
    )
    for r in rows:
        cm = r["clean"]["mean"]
        pm = r["poisoned"]["mean"]
        dm = r["delta_poison_minus_clean"]["mean"]
        cwm = r["delta_poison_minus_clean_on_c_to_w"]["mean"]
        nd = r["delta_poison_minus_clean"]["n"]
        print(
            f"{r['pop_group']:<12} {r['distance']:<6} "
            f"{(f'{cm:.4f}' if cm is not None else 'NA'):>12} "
            f"{(f'{pm:.4f}' if pm is not None else 'NA'):>12} "
            f"{(f'{dm:.4f}' if dm is not None else 'NA'):>12} "
            f"{(f'{cwm:.4f}' if cwm is not None else 'NA'):>14} "
            f"{nd:>8}"
        )
    overall = summary.get("overall_by_group", {})
    if overall:
        print("\n=== Overall Hub/Tail Summary ===")
        print(f"{'pop':<12} {'clean_mean':>12} {'poison_mean':>12} {'delta_mean':>12} {'c2w_rate':>12} {'n':>8}")
        for pop in ["hub", "tail", "mid", "unknown"]:
            row = overall.get(pop)
            if not row:
                continue
            cm = row["clean"]["mean"]
            pm = row["poisoned"]["mean"]
            dm = row["delta_poison_minus_clean"]["mean"]
            c2w = row.get("c2w_rate")
            n = row.get("n_total", 0)
            print(
                f"{pop:<12} "
                f"{(f'{cm:.4f}' if cm is not None else 'NA'):>12} "
                f"{(f'{pm:.4f}' if pm is not None else 'NA'):>12} "
                f"{(f'{dm:.4f}' if dm is not None else 'NA'):>12} "
                f"{(f'{c2w:.4f}' if isinstance(c2w, (int, float)) else 'NA'):>12} "
                f"{n:>8}"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze margin dynamics from comparison reports.")
    p.add_argument(
        "--report",
        nargs="+",
        required=True,
        help="Comparison report path(s) or glob(s), e.g. main_output/**/comparison_reports/*comparison*.json",
    )
    p.add_argument(
        "--out-dir",
        default="artifacts/analysis/margin",
        help="Directory for JSON/CSV outputs.",
    )
    p.add_argument("--graph-file", default="", help="Optional graph checkpoint (latest.pkl) for top/bottom degree grouping.")
    p.add_argument("--top-pct", type=float, default=0.05, help="Top percentile as Hub group when --graph-file is set.")
    p.add_argument("--bottom-pct", type=float, default=0.05, help="Bottom percentile as Tail group when --graph-file is set.")
    p.add_argument("--entity-field", default="head", choices=["head", "tail"], help="Which entity field to classify by graph degree.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report_paths = _iter_report_paths(args.report)
    if not report_paths:
        print("No report files matched.")
        return 2

    hub_nodes = tail_nodes = None
    if args.graph_file:
        hub_nodes, tail_nodes = _build_degree_groups(args.graph_file, args.top_pct, args.bottom_pct)
        print(
            f"Graph grouping enabled: hub_top={args.top_pct:.2%}, tail_bottom={args.bottom_pct:.2%}, "
            f"hub_nodes={len(hub_nodes)}, tail_nodes={len(tail_nodes)}, entity_field={args.entity_field}"
        )

    summary = analyze_reports(
        report_paths,
        hub_nodes=hub_nodes,
        tail_nodes=tail_nodes,
        entity_field=args.entity_field,
    )
    json_path, csv_path = write_outputs(summary, args.out_dir)
    print_console_table(summary)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
