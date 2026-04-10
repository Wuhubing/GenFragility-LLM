#!/usr/bin/env python3
"""Compute Mask-A and Mask-B E1/E2 summaries from paired Hub/Low reports."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from statistics import mean, median
from typing import Callable, Dict, List


HOPS = ["d1", "d2", "d3", "d4", "d5"]


def load_rows(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("unified_results", [])


def mask_a(row: Dict) -> bool:
    return row.get("distance") in HOPS and row.get("clean_accuracy") == 1


def mask_b(row: Dict) -> bool:
    return (
        row.get("distance") in HOPS
        and row.get("clean_accuracy") == 1
        and row.get("clean_correct_token_rank") == 1
    )


def safe_mean(vals: List[float]) -> float:
    return mean(vals) if vals else 0.0


def safe_median(vals: List[float]) -> float:
    return median(vals) if vals else 0.0


def summarize(rows: List[Dict], mask_fn: Callable[[Dict], bool]) -> Dict:
    picked = [r for r in rows if mask_fn(r)]
    by_hop = defaultdict(list)
    for r in picked:
        by_hop[r["distance"]].append(r)

    clean_margin = [r.get("clean_margin") for r in picked if r.get("clean_margin") is not None]
    abs_conf = [abs(r.get("confidence_change")) for r in picked if r.get("confidence_change") is not None]
    abs_margin = [abs(r.get("margin_change")) for r in picked if r.get("margin_change") is not None]
    abs_lift = [
        abs(r.get("neighbor_attention_lift_change"))
        for r in picked
        if r.get("neighbor_attention_lift_change") is not None
    ]
    c2w = [r for r in picked if r.get("poisoned_accuracy") == 0]

    out = {
        "overall": {
            "n": len(picked),
            "clean_margin_mean": safe_mean(clean_margin),
            "clean_margin_median": safe_median(clean_margin),
            "abs_conf_change_mean": safe_mean(abs_conf),
            "abs_margin_change_mean": safe_mean(abs_margin),
            "abs_attention_lift_change_mean": safe_mean(abs_lift),
            "c_to_w_rate": (len(c2w) / len(picked)) if picked else 0.0,
        },
        "by_hop": {},
    }

    for hop in HOPS:
        subset = by_hop.get(hop, [])
        cm = [r.get("clean_margin") for r in subset if r.get("clean_margin") is not None]
        ac = [abs(r.get("confidence_change")) for r in subset if r.get("confidence_change") is not None]
        am = [abs(r.get("margin_change")) for r in subset if r.get("margin_change") is not None]
        al = [
            abs(r.get("neighbor_attention_lift_change"))
            for r in subset
            if r.get("neighbor_attention_lift_change") is not None
        ]
        c2w_h = [r for r in subset if r.get("poisoned_accuracy") == 0]
        out["by_hop"][hop] = {
            "n": len(subset),
            "clean_margin_mean": safe_mean(cm),
            "abs_conf_change_mean": safe_mean(ac),
            "abs_margin_change_mean": safe_mean(am),
            "abs_attention_lift_change_mean": safe_mean(al),
            "c_to_w_rate": (len(c2w_h) / len(subset)) if subset else 0.0,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub-main-report", required=True)
    parser.add_argument("--low-main-report", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    hub_rows = load_rows(args.hub_main_report)
    low_rows = load_rows(args.low_main_report)

    payload = {
        "meta": {
            "hub_main_report": args.hub_main_report,
            "low_main_report": args.low_main_report,
        },
        "mask_a": {
            "definition": "clean_accuracy==1",
            "hub": summarize(hub_rows, mask_a),
            "low": summarize(low_rows, mask_a),
        },
        "mask_b": {
            "definition": "clean_accuracy==1 && clean_correct_token_rank==1",
            "hub": summarize(hub_rows, mask_b),
            "low": summarize(low_rows, mask_b),
        },
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
