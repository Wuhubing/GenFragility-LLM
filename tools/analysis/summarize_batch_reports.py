#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path


def is_correct(row, prefix):
    acc = row.get(f"{prefix}_accuracy")
    em = row.get(f"{prefix}_exact_match")
    if em is True:
        return True
    if isinstance(acc, (int, float)):
        return acc in (1, 1.0, 100, 100.0)
    return False


def mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def summarize_report(path):
    d = json.load(open(path, "r", encoding="utf-8"))
    u = d.get("unified_results", [])
    d1 = [r for r in u if r.get("distance") == "d1"]

    clean_correct = [r for r in u if is_correct(r, "clean")]
    c2w = [r for r in clean_correct if not is_correct(r, "poisoned")]
    c2w_rate = (len(c2w) / len(clean_correct)) if clean_correct else None

    clean_correct_d1 = [r for r in d1 if is_correct(r, "clean")]
    c2w_d1 = [r for r in clean_correct_d1 if not is_correct(r, "poisoned")]
    c2w_rate_d1 = (len(c2w_d1) / len(clean_correct_d1)) if clean_correct_d1 else None

    return {
        "report_path": path,
        "experiment_name": d.get("metadata", {}).get("experiment_name"),
        "total_n": len(u),
        "d1_n": len(d1),
        "clean_correct_n": len(clean_correct),
        "c2w_n": len(c2w),
        "c2w_rate": c2w_rate,
        "d1_clean_correct_n": len(clean_correct_d1),
        "d1_c2w_n": len(c2w_d1),
        "d1_c2w_rate": c2w_rate_d1,
        "mean_margin_change": mean([r.get("margin_change") for r in u]),
        "mean_attention_entropy_change": mean([r.get("attention_entropy_change") for r in u]),
        "mean_attention_score_change": mean([r.get("attention_score_change") for r in u]),
        "dump_margin": d.get("metadata", {}).get("diagnostics", {}).get("dump_margin"),
        "dump_attention": d.get("metadata", {}).get("diagnostics", {}).get("dump_attention"),
        "margin_dump_file": d.get("metadata", {}).get("diagnostics", {}).get("margin_dump_file"),
        "attention_dump_file": d.get("metadata", {}).get("diagnostics", {}).get("attention_dump_file"),
    }


def to_md_table(rows):
    headers = [
        "experiment_name",
        "total_n",
        "d1_n",
        "c2w_rate",
        "d1_c2w_rate",
        "mean_margin_change",
        "mean_attention_entropy_change",
        "mean_attention_score_change",
    ]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        vals = []
        for h in headers:
            v = r.get(h)
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", nargs="+", required=True, help="report path(s) or glob(s)")
    ap.add_argument("--out-dir", default="artifacts/analysis/batch_summary")
    args = ap.parse_args()

    paths = []
    for p in args.report:
        m = glob.glob(p, recursive=True)
        if m:
            paths.extend(m)
        elif os.path.exists(p):
            paths.append(p)
    paths = sorted(set(paths))
    if not paths:
        print("No report files matched.")
        return 2

    rows = [summarize_report(p) for p in paths]
    rows.sort(key=lambda x: (x.get("experiment_name") or "", x.get("report_path")))

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    json_path = os.path.join(args.out_dir, "batch_summary.json")
    csv_path = os.path.join(args.out_dir, "batch_summary.csv")
    md_path = os.path.join(args.out_dir, "batch_summary.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(to_md_table(rows) + "\n")

    print("Saved:", json_path)
    print("Saved:", csv_path)
    print("Saved:", md_path)
    print("\n" + to_md_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

