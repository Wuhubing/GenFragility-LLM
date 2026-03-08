#!/usr/bin/env python3
"""Analyze E2: attention representation vs graph distance.

Input options:
1) `--report`: comparison report json(s). If attention fields exist in rows, use them.
2) `--attention-dump`: jsonl with per-sample attention stats.

Expected (flexible) fields in attention rows:
- sample_id (optional)
- distance (required, e.g. d0..d5)
- clean_attention_entropy / poisoned_attention_entropy / attention_entropy_change
- clean_attention_score / poisoned_attention_score / attention_score_change

The script also computes C->W rate by distance from reports (if provided).
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


def _distance_to_int(d: str) -> Optional[int]:
    if not isinstance(d, str):
        return None
    d = d.strip().lower()
    if d.startswith("d") and d[1:].isdigit():
        return int(d[1:])
    return None


def _iter_paths(patterns: Iterable[str]) -> List[str]:
    out: List[str] = []
    for p in patterns:
        matches = sorted(glob.glob(p, recursive=True))
        if matches:
            out.extend(matches)
        elif os.path.exists(p):
            out.append(p)
    return sorted(set(out))


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


def _resolve_pop_group(item: dict, hub_nodes: Optional[Set[str]], tail_nodes: Optional[Set[str]], entity_field: str) -> str:
    if hub_nodes is None or tail_nodes is None:
        return item.get("popularity_group", "unknown")
    ent = item.get(entity_field)
    if ent in hub_nodes:
        return "hub"
    if ent in tail_nodes:
        return "tail"
    return "mid"


def _is_correct(item: dict, prefix: str) -> bool:
    acc = item.get(f"{prefix}_accuracy")
    em = item.get(f"{prefix}_exact_match")
    acc_correct = False
    if isinstance(acc, (int, float)):
        acc_correct = (acc == 1) or (acc == 1.0) or (acc == 100) or (acc == 100.0)
    return bool(em) or acc_correct


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        dy = y - my
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    if den_x <= 0 or den_y <= 0:
        return None
    return num / math.sqrt(den_x * den_y)


def _rankdata(values: List[float]) -> List[float]:
    # Average rank for ties.
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based ranks
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    return _pearson(rx, ry)


def _mean(xs: List[float]) -> Optional[float]:
    return statistics.fmean(xs) if xs else None


def _summarize(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"n": 0, "mean": None, "std": None}
    return {
        "n": len(values),
        "mean": _mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def load_attention_rows_from_reports(
    report_paths: List[str],
    hub_nodes: Optional[Set[str]] = None,
    tail_nodes: Optional[Set[str]] = None,
    entity_field: str = "head",
) -> List[dict]:
    rows: List[dict] = []
    for p in report_paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("unified_results", []):
            dist = item.get("distance")
            if dist is None:
                continue

            # Flexible field mapping.
            clean_entropy = _safe_float(item.get("clean_attention_entropy"))
            poison_entropy = _safe_float(item.get("poisoned_attention_entropy"))
            entropy_change = _safe_float(item.get("attention_entropy_change"))
            if entropy_change is None and clean_entropy is not None and poison_entropy is not None:
                entropy_change = poison_entropy - clean_entropy

            clean_score = _safe_float(item.get("clean_attention_score"))
            poison_score = _safe_float(item.get("poisoned_attention_score"))
            score_change = _safe_float(item.get("attention_score_change"))
            if score_change is None and clean_score is not None and poison_score is not None:
                score_change = poison_score - clean_score

            clean_neighbor_mass = _safe_float(item.get("clean_neighbor_attention_mass"))
            poison_neighbor_mass = _safe_float(item.get("poisoned_neighbor_attention_mass"))
            neighbor_mass_change = _safe_float(item.get("neighbor_attention_mass_change"))
            if neighbor_mass_change is None and clean_neighbor_mass is not None and poison_neighbor_mass is not None:
                neighbor_mass_change = poison_neighbor_mass - clean_neighbor_mass

            clean_neighbor_lift = _safe_float(item.get("clean_neighbor_attention_lift"))
            poison_neighbor_lift = _safe_float(item.get("poisoned_neighbor_attention_lift"))
            neighbor_lift_change = _safe_float(item.get("neighbor_attention_lift_change"))
            if neighbor_lift_change is None and clean_neighbor_lift is not None and poison_neighbor_lift is not None:
                neighbor_lift_change = poison_neighbor_lift - clean_neighbor_lift

            proxy_mode = False
            # Fallback: if no attention score exists, reuse model confidence as a proxy score.
            if clean_score is None and poison_score is None:
                clean_score = _safe_float(item.get("clean_confidence"))
                poison_score = _safe_float(item.get("poisoned_confidence"))
                if score_change is None and clean_score is not None and poison_score is not None:
                    score_change = poison_score - clean_score
                if clean_score is not None or poison_score is not None:
                    proxy_mode = True

            if (
                clean_entropy is None
                and poison_entropy is None
                and entropy_change is None
                and clean_score is None
                and poison_score is None
                and score_change is None
            ):
                continue

            rows.append(
                {
                    "distance": dist,
                    "pop_group": _resolve_pop_group(item, hub_nodes, tail_nodes, entity_field),
                    "clean_attention_entropy": clean_entropy,
                    "poisoned_attention_entropy": poison_entropy,
                    "attention_entropy_change": entropy_change,
                    "clean_attention_score": clean_score,
                    "poisoned_attention_score": poison_score,
                    "attention_score_change": score_change,
                    "clean_neighbor_attention_mass": clean_neighbor_mass,
                    "poisoned_neighbor_attention_mass": poison_neighbor_mass,
                    "neighbor_attention_mass_change": neighbor_mass_change,
                    "clean_neighbor_attention_lift": clean_neighbor_lift,
                    "poisoned_neighbor_attention_lift": poison_neighbor_lift,
                    "neighbor_attention_lift_change": neighbor_lift_change,
                    "proxy_mode": proxy_mode,
                }
            )
    return rows


def load_attention_rows_from_jsonl(
    attention_paths: List[str],
    hub_nodes: Optional[Set[str]] = None,
    tail_nodes: Optional[Set[str]] = None,
    entity_field: str = "head",
) -> List[dict]:
    rows: List[dict] = []
    for p in attention_paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                dist = item.get("distance")
                if dist is None:
                    continue
                if "pop_group" not in item:
                    item["pop_group"] = _resolve_pop_group(item, hub_nodes, tail_nodes, entity_field)
                rows.append(item)
    return rows


def compute_cw_by_distance(report_paths: List[str]) -> Dict[str, dict]:
    by_dist = {d: {"cw": 0, "clean_correct": 0} for d in DISTANCE_ORDER}
    for p in report_paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("unified_results", []):
            dist = item.get("distance")
            if dist not in by_dist:
                continue
            clean_ok = _is_correct(item, "clean")
            poison_ok = _is_correct(item, "poisoned")
            if clean_ok:
                by_dist[dist]["clean_correct"] += 1
                if not poison_ok:
                    by_dist[dist]["cw"] += 1
    out = {}
    for d in DISTANCE_ORDER:
        cc = by_dist[d]["clean_correct"]
        cw = by_dist[d]["cw"]
        out[d] = {
            "clean_correct_n": cc,
            "c_to_w_n": cw,
            "c_to_w_rate": (cw / cc) if cc > 0 else None,
        }
    return out


def analyze_attention_distance(attention_rows: List[dict]) -> dict:
    by_dist = defaultdict(lambda: defaultdict(list))
    by_pop = defaultdict(lambda: defaultdict(list))
    by_pop_dist = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    proxy_rows = 0

    all_distance_idx: List[float] = []
    all_entropy_change: List[float] = []
    all_distance_idx_for_score: List[float] = []
    all_score_change: List[float] = []
    all_distance_idx_for_neighbor_mass: List[float] = []
    all_neighbor_mass_change: List[float] = []
    all_distance_idx_for_neighbor_lift: List[float] = []
    all_neighbor_lift_change: List[float] = []
    all_clean_entropy: List[float] = []
    group_corr_entropy = defaultdict(lambda: {"x": [], "y": []})
    group_corr_score = defaultdict(lambda: {"x": [], "y": []})
    group_corr_neighbor_mass = defaultdict(lambda: {"x": [], "y": []})
    group_corr_neighbor_lift = defaultdict(lambda: {"x": [], "y": []})

    for r in attention_rows:
        dist = r.get("distance", "unknown")
        d_idx = _distance_to_int(dist)
        if d_idx is None:
            continue
        pop = r.get("pop_group", "unknown")
        if r.get("proxy_mode"):
            proxy_rows += 1

        for k in [
            "clean_attention_entropy",
            "poisoned_attention_entropy",
            "attention_entropy_change",
            "clean_attention_score",
            "poisoned_attention_score",
            "attention_score_change",
            "clean_neighbor_attention_mass",
            "poisoned_neighbor_attention_mass",
            "neighbor_attention_mass_change",
            "clean_neighbor_attention_lift",
            "poisoned_neighbor_attention_lift",
            "neighbor_attention_lift_change",
        ]:
            v = _safe_float(r.get(k))
            if v is not None:
                by_dist[dist][k].append(v)
                by_pop[pop][k].append(v)
                by_pop_dist[pop][dist][k].append(v)

        ent_change = _safe_float(r.get("attention_entropy_change"))
        if ent_change is not None:
            all_distance_idx.append(float(d_idx))
            all_entropy_change.append(ent_change)
            group_corr_entropy[pop]["x"].append(float(d_idx))
            group_corr_entropy[pop]["y"].append(ent_change)

        score_change = _safe_float(r.get("attention_score_change"))
        if score_change is not None:
            all_distance_idx_for_score.append(float(d_idx))
            all_score_change.append(score_change)
            group_corr_score[pop]["x"].append(float(d_idx))
            group_corr_score[pop]["y"].append(score_change)

        neighbor_mass_change = _safe_float(r.get("neighbor_attention_mass_change"))
        if neighbor_mass_change is not None:
            all_distance_idx_for_neighbor_mass.append(float(d_idx))
            all_neighbor_mass_change.append(neighbor_mass_change)
            group_corr_neighbor_mass[pop]["x"].append(float(d_idx))
            group_corr_neighbor_mass[pop]["y"].append(neighbor_mass_change)

        neighbor_lift_change = _safe_float(r.get("neighbor_attention_lift_change"))
        if neighbor_lift_change is not None:
            all_distance_idx_for_neighbor_lift.append(float(d_idx))
            all_neighbor_lift_change.append(neighbor_lift_change)
            group_corr_neighbor_lift[pop]["x"].append(float(d_idx))
            group_corr_neighbor_lift[pop]["y"].append(neighbor_lift_change)

        clean_entropy = _safe_float(r.get("clean_attention_entropy"))
        if clean_entropy is not None:
            all_clean_entropy.append(clean_entropy)

    dist_summary = {}
    for d in DISTANCE_ORDER:
        vals = by_dist.get(d, {})
        dist_summary[d] = {
            "clean_attention_entropy": _summarize(vals.get("clean_attention_entropy", [])),
            "poisoned_attention_entropy": _summarize(vals.get("poisoned_attention_entropy", [])),
            "attention_entropy_change": _summarize(vals.get("attention_entropy_change", [])),
            "clean_attention_score": _summarize(vals.get("clean_attention_score", [])),
            "poisoned_attention_score": _summarize(vals.get("poisoned_attention_score", [])),
            "attention_score_change": _summarize(vals.get("attention_score_change", [])),
            "clean_neighbor_attention_mass": _summarize(vals.get("clean_neighbor_attention_mass", [])),
            "poisoned_neighbor_attention_mass": _summarize(vals.get("poisoned_neighbor_attention_mass", [])),
            "neighbor_attention_mass_change": _summarize(vals.get("neighbor_attention_mass_change", [])),
            "clean_neighbor_attention_lift": _summarize(vals.get("clean_neighbor_attention_lift", [])),
            "poisoned_neighbor_attention_lift": _summarize(vals.get("poisoned_neighbor_attention_lift", [])),
            "neighbor_attention_lift_change": _summarize(vals.get("neighbor_attention_lift_change", [])),
        }

    pop_summary = {}
    for pop, vals in by_pop.items():
        pop_summary[pop] = {
            "clean_attention_entropy": _summarize(vals.get("clean_attention_entropy", [])),
            "poisoned_attention_entropy": _summarize(vals.get("poisoned_attention_entropy", [])),
            "attention_entropy_change": _summarize(vals.get("attention_entropy_change", [])),
            "clean_attention_score": _summarize(vals.get("clean_attention_score", [])),
            "poisoned_attention_score": _summarize(vals.get("poisoned_attention_score", [])),
            "attention_score_change": _summarize(vals.get("attention_score_change", [])),
            "clean_neighbor_attention_mass": _summarize(vals.get("clean_neighbor_attention_mass", [])),
            "poisoned_neighbor_attention_mass": _summarize(vals.get("poisoned_neighbor_attention_mass", [])),
            "neighbor_attention_mass_change": _summarize(vals.get("neighbor_attention_mass_change", [])),
            "clean_neighbor_attention_lift": _summarize(vals.get("clean_neighbor_attention_lift", [])),
            "poisoned_neighbor_attention_lift": _summarize(vals.get("poisoned_neighbor_attention_lift", [])),
            "neighbor_attention_lift_change": _summarize(vals.get("neighbor_attention_lift_change", [])),
        }

    pop_dist_summary = {}
    for pop, dist_map in by_pop_dist.items():
        pop_dist_summary[pop] = {}
        for d in DISTANCE_ORDER:
            vals = dist_map.get(d, {})
            pop_dist_summary[pop][d] = {
                "clean_attention_entropy": _summarize(vals.get("clean_attention_entropy", [])),
                "poisoned_attention_entropy": _summarize(vals.get("poisoned_attention_entropy", [])),
                "attention_entropy_change": _summarize(vals.get("attention_entropy_change", [])),
                "clean_attention_score": _summarize(vals.get("clean_attention_score", [])),
                "poisoned_attention_score": _summarize(vals.get("poisoned_attention_score", [])),
                "attention_score_change": _summarize(vals.get("attention_score_change", [])),
                "clean_neighbor_attention_mass": _summarize(vals.get("clean_neighbor_attention_mass", [])),
                "poisoned_neighbor_attention_mass": _summarize(vals.get("poisoned_neighbor_attention_mass", [])),
                "neighbor_attention_mass_change": _summarize(vals.get("neighbor_attention_mass_change", [])),
                "clean_neighbor_attention_lift": _summarize(vals.get("clean_neighbor_attention_lift", [])),
                "poisoned_neighbor_attention_lift": _summarize(vals.get("poisoned_neighbor_attention_lift", [])),
                "neighbor_attention_lift_change": _summarize(vals.get("neighbor_attention_lift_change", [])),
            }

    correlations = {
        "distance_vs_attention_entropy_change": {
            "pearson": _pearson(all_distance_idx, all_entropy_change),
            "spearman": _spearman(all_distance_idx, all_entropy_change),
            "n": len(all_entropy_change),
        },
        "distance_vs_attention_score_change": {
            "pearson": _pearson(all_distance_idx_for_score, all_score_change),
            "spearman": _spearman(all_distance_idx_for_score, all_score_change),
            "n": len(all_score_change),
        },
        "distance_vs_neighbor_attention_mass_change": {
            "pearson": _pearson(all_distance_idx_for_neighbor_mass, all_neighbor_mass_change),
            "spearman": _spearman(all_distance_idx_for_neighbor_mass, all_neighbor_mass_change),
            "n": len(all_neighbor_mass_change),
        },
        "distance_vs_neighbor_attention_lift_change": {
            "pearson": _pearson(all_distance_idx_for_neighbor_lift, all_neighbor_lift_change),
            "spearman": _spearman(all_distance_idx_for_neighbor_lift, all_neighbor_lift_change),
            "n": len(all_neighbor_lift_change),
        },
    }
    correlations["by_pop_group"] = {}
    for pop in sorted(group_corr_entropy.keys() | group_corr_score.keys()):
        xe = group_corr_entropy[pop]["x"]
        ye = group_corr_entropy[pop]["y"]
        xs = group_corr_score[pop]["x"]
        ys = group_corr_score[pop]["y"]
        correlations["by_pop_group"][pop] = {
            "distance_vs_attention_entropy_change": {
                "pearson": _pearson(xe, ye),
                "spearman": _spearman(xe, ye),
                "n": len(ye),
            },
            "distance_vs_attention_score_change": {
                "pearson": _pearson(xs, ys),
                "spearman": _spearman(xs, ys),
                "n": len(ys),
            },
            "distance_vs_neighbor_attention_mass_change": {
                "pearson": _pearson(group_corr_neighbor_mass[pop]["x"], group_corr_neighbor_mass[pop]["y"]),
                "spearman": _spearman(group_corr_neighbor_mass[pop]["x"], group_corr_neighbor_mass[pop]["y"]),
                "n": len(group_corr_neighbor_mass[pop]["y"]),
            },
            "distance_vs_neighbor_attention_lift_change": {
                "pearson": _pearson(group_corr_neighbor_lift[pop]["x"], group_corr_neighbor_lift[pop]["y"]),
                "spearman": _spearman(group_corr_neighbor_lift[pop]["x"], group_corr_neighbor_lift[pop]["y"]),
                "n": len(group_corr_neighbor_lift[pop]["y"]),
            },
        }

    return {
        "num_attention_rows": len(attention_rows),
        "proxy_mode_rows": proxy_rows,
        "by_distance": dist_summary,
        "by_pop_group": pop_summary,
        "by_pop_group_distance": pop_dist_summary,
        "correlations": correlations,
    }


def write_outputs(summary: dict, out_dir: str) -> Tuple[str, str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    json_path = os.path.join(out_dir, "attention_distance_summary.json")
    csv_path = os.path.join(out_dir, "attention_distance_summary.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "distance",
                "entropy_change_n",
                "entropy_change_mean",
                "clean_entropy_mean",
                "poisoned_entropy_mean",
                "c_to_w_rate",
                "clean_correct_n",
                "neighbor_mass_change_n",
                "neighbor_mass_change_mean",
                "clean_neighbor_mass_mean",
                "neighbor_lift_change_n",
                "neighbor_lift_change_mean",
                "clean_neighbor_lift_mean",
            ]
        )
        for d in DISTANCE_ORDER:
            by_d = summary["by_distance"].get(d, {})
            cw = summary.get("c_to_w_by_distance", {}).get(d, {})
            w.writerow(
                [
                    d,
                    by_d.get("attention_entropy_change", {}).get("n"),
                    by_d.get("attention_entropy_change", {}).get("mean"),
                    by_d.get("clean_attention_entropy", {}).get("mean"),
                    by_d.get("poisoned_attention_entropy", {}).get("mean"),
                    cw.get("c_to_w_rate"),
                    cw.get("clean_correct_n"),
                    by_d.get("neighbor_attention_mass_change", {}).get("n"),
                    by_d.get("neighbor_attention_mass_change", {}).get("mean"),
                    by_d.get("clean_neighbor_attention_mass", {}).get("mean"),
                    by_d.get("neighbor_attention_lift_change", {}).get("n"),
                    by_d.get("neighbor_attention_lift_change", {}).get("mean"),
                    by_d.get("clean_neighbor_attention_lift", {}).get("mean"),
                ]
            )
    return json_path, csv_path


def print_summary(summary: dict) -> None:
    print("=== Attention-Distance Summary ===")
    print("num_attention_rows:", summary.get("num_attention_rows", 0))
    if summary.get("proxy_mode_rows", 0) > 0:
        print("proxy_mode_rows:", summary.get("proxy_mode_rows"), "(confidence used as attention_score proxy)")
    corr = summary.get("correlations", {}).get("distance_vs_attention_entropy_change", {})
    corr_score = summary.get("correlations", {}).get("distance_vs_attention_score_change", {})
    corr_neighbor_mass = summary.get("correlations", {}).get("distance_vs_neighbor_attention_mass_change", {})
    corr_neighbor_lift = summary.get("correlations", {}).get("distance_vs_neighbor_attention_lift_change", {})
    print(
        "distance_vs_attention_entropy_change:",
        f"pearson={corr.get('pearson')}, spearman={corr.get('spearman')}, n={corr.get('n')}",
    )
    print(
        "distance_vs_attention_score_change:",
        f"pearson={corr_score.get('pearson')}, spearman={corr_score.get('spearman')}, n={corr_score.get('n')}",
    )
    print(
        "distance_vs_neighbor_attention_mass_change:",
        f"pearson={corr_neighbor_mass.get('pearson')}, spearman={corr_neighbor_mass.get('spearman')}, n={corr_neighbor_mass.get('n')}",
    )
    print(
        "distance_vs_neighbor_attention_lift_change:",
        f"pearson={corr_neighbor_lift.get('pearson')}, spearman={corr_neighbor_lift.get('spearman')}, n={corr_neighbor_lift.get('n')}",
    )
    print("-" * 96)
    print(
        f"{'dist':<6} {'entropy_change_mean':>20} {'clean_entropy_mean':>20} "
        f"{'c_to_w_rate':>14} {'n_clean_correct':>16}"
    )
    print("-" * 96)
    for d in DISTANCE_ORDER:
        by_d = summary["by_distance"].get(d, {})
        cw = summary.get("c_to_w_by_distance", {}).get(d, {})
        em = by_d.get("attention_entropy_change", {}).get("mean")
        cem = by_d.get("clean_attention_entropy", {}).get("mean")
        rate = cw.get("c_to_w_rate")
        ncc = cw.get("clean_correct_n")
        print(
            f"{d:<6} "
            f"{(f'{em:.6f}' if isinstance(em, (int, float)) else 'NA'):>20} "
            f"{(f'{cem:.6f}' if isinstance(cem, (int, float)) else 'NA'):>20} "
            f"{(f'{rate:.4f}' if isinstance(rate, (int, float)) else 'NA'):>14} "
            f"{(str(ncc) if ncc is not None else 'NA'):>16}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze attention vs graph distance.")
    p.add_argument(
        "--report",
        nargs="*",
        default=[],
        help="Comparison report path(s) or glob(s).",
    )
    p.add_argument(
        "--attention-dump",
        nargs="*",
        default=[],
        help="Attention dump jsonl path(s) or glob(s).",
    )
    p.add_argument(
        "--out-dir",
        default="artifacts/analysis/attention",
        help="Directory for output json/csv.",
    )
    p.add_argument("--graph-file", default="", help="Optional graph checkpoint (latest.pkl) for top/bottom degree grouping.")
    p.add_argument("--top-pct", type=float, default=0.05, help="Top percentile as Hub group when --graph-file is set.")
    p.add_argument("--bottom-pct", type=float, default=0.05, help="Bottom percentile as Tail group when --graph-file is set.")
    p.add_argument("--entity-field", default="head", choices=["head", "tail"], help="Which entity field to classify by graph degree.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report_paths = _iter_paths(args.report)
    attn_paths = _iter_paths(args.attention_dump)
    hub_nodes = tail_nodes = None
    if args.graph_file:
        hub_nodes, tail_nodes = _build_degree_groups(args.graph_file, args.top_pct, args.bottom_pct)
        print(
            f"Graph grouping enabled: hub_top={args.top_pct:.2%}, tail_bottom={args.bottom_pct:.2%}, "
            f"hub_nodes={len(hub_nodes)}, tail_nodes={len(tail_nodes)}, entity_field={args.entity_field}"
        )

    attention_rows: List[dict] = []
    if report_paths:
        attention_rows.extend(load_attention_rows_from_reports(report_paths, hub_nodes, tail_nodes, args.entity_field))
    if attn_paths:
        attention_rows.extend(load_attention_rows_from_jsonl(attn_paths, hub_nodes, tail_nodes, args.entity_field))

    if not attention_rows:
        print("No attention rows found. Provide reports with attention fields or --attention-dump jsonl.")
        return 2

    summary = analyze_attention_distance(attention_rows)
    if report_paths:
        summary["c_to_w_by_distance"] = compute_cw_by_distance(report_paths)
    else:
        summary["c_to_w_by_distance"] = {}

    json_path, csv_path = write_outputs(summary, args.out_dir)
    print_summary(summary)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
