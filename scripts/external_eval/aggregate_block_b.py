"""
aggregate_block_b.py — collect 600 vllm comparison reports into one table.

Block B output layout (produced by run_block_b.sh):
  main_output/block_b/
    mintaka/
      none/<sample_id>/comparison_reports/<sample_id>_vllm_comparison.json
      popularity_top25/<sample_id>/comparison_reports/...
      random_non_hub_25_seed42/<sample_id>/comparison_reports/...
    trex/...
    webqsp/...

Each report has comparison_statistics.d1.{clean_accuracy, poisoned_accuracy, epr, count, flip_count}.
For Block B we treat the d1 numbers as the "preserve set" metrics
(because convert_external_to_block_a.py packed the preserve set into ripples.d1).

Output:
  block_b_results.json     full numbers, per-(dataset, sample_id, mode)
  block_b_table.md         paper-ready table aggregated per-bucket

Run:
  python aggregate_block_b.py \
      --base-dir main_output/block_b \
      --index-dir data/external_eval/block_b_experiments \
      --out-json data/external_eval/block_b_results.json \
      --out-md   data/external_eval/block_b_table.md
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


DEFAULT_MODES = ["none", "popularity_top25", "random_non_hub_25_seed42"]


def find_report(base_dir: Path, dataset: str, mode: str, sample_id: str):
    """Locate the vllm comparison JSON for one run."""
    p = base_dir / dataset / mode / sample_id / "comparison_reports"
    if not p.exists():
        return None
    cands = list(p.glob("*vllm*.json"))
    return cands[0] if cands else None


def load_d1_metrics(report_path: Path):
    """Return (clean_acc, poison_acc, epr, count, flip_count) from d1 of one report."""
    d = json.loads(report_path.read_text())
    d1 = d.get("comparison_statistics", {}).get("d1", {})
    return {
        "clean_acc": d1.get("clean_accuracy"),
        "poison_acc": d1.get("poisoned_accuracy"),
        "epr": d1.get("epr"),
        "count": d1.get("count", 0),
        "flip_count": d1.get("flip_count", 0),
        "drop": (d1.get("clean_accuracy", 0) - d1.get("poisoned_accuracy", 0))
                 if d1.get("clean_accuracy") is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path,
                    default=Path("main_output/block_b"))
    ap.add_argument("--index-dir", type=Path,
                    default=Path("data/external_eval/block_b_experiments"))
    ap.add_argument("--datasets", nargs="+",
                    default=["mintaka", "trex", "webqsp"])
    ap.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    ap.add_argument("--out-json", type=Path,
                    default=Path("data/external_eval/block_b_results.json"))
    ap.add_argument("--out-md", type=Path,
                    default=Path("data/external_eval/block_b_table.md"))
    args = ap.parse_args()

    all_results = []      # flat list, one row per (dataset, sample_id, mode)
    missing = []           # runs we couldn't find
    by_bucket = defaultdict(lambda: defaultdict(list))   # by_bucket[dataset][bucket][mode] -> [drop, ...]
    by_dataset = defaultdict(lambda: defaultdict(list))  # by_dataset[dataset][mode] -> [drop, ...]

    for dataset in args.datasets:
        idx_path = args.index_dir / dataset / "_index.json"
        if not idx_path.exists():
            print(f"⚠️  No _index.json for {dataset} at {idx_path} — skipping.")
            continue
        index = json.loads(idx_path.read_text())
        print(f"\n=== {dataset}: {len(index)} samples × {len(args.modes)} modes ===")

        for entry in index:
            sid = entry["experiment_id"]
            bucket = entry["bucket"]
            for mode in args.modes:
                report = find_report(args.base_dir, dataset, mode, sid)
                if report is None:
                    missing.append((dataset, sid, mode))
                    continue
                metrics = load_d1_metrics(report)
                row = {
                    "dataset": dataset, "sample_id": sid, "bucket": bucket,
                    "mode": mode, **metrics,
                }
                all_results.append(row)
                if metrics["drop"] is not None:
                    by_bucket[dataset][bucket][mode].append(metrics["drop"])
                    by_dataset[dataset][mode].append(metrics["drop"])

    # ---- Write raw JSON ----
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "n_results": len(all_results),
        "n_missing": len(missing),
        "missing_sample": missing[:20],
        "results": all_results,
    }, indent=2, ensure_ascii=False))
    print(f"\n✅ Wrote {len(all_results)} rows ({len(missing)} missing) -> {args.out_json}")

    # ---- Build markdown table ----
    lines = []
    lines.append("# Block B — Public Dataset Anchor Mitigation\n")
    lines.append(f"Generated from {len(all_results)} runs ({len(missing)} missing).\n")
    lines.append(f"Metric: **preserve-set accuracy drop** (= d1 clean_acc - d1 poison_acc on disjoint preserve set).\n")
    lines.append("Lower (more negative) drop = anchor better protects the model.\n\n")

    def fmt(xs):
        if not xs:
            return "—"
        m = mean(xs)
        s = stdev(xs) if len(xs) > 1 else 0.0
        return f"{m:+.3f} ± {s:.3f} (n={len(xs)})"

    # ---- Per-dataset summary ----
    lines.append("## Per-dataset summary (all buckets pooled)\n")
    header = ["Dataset"] + args.modes
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for dataset in args.datasets:
        row = [dataset] + [fmt(by_dataset[dataset].get(m, [])) for m in args.modes]
        lines.append("| " + " | ".join(row) + " |")

    # ---- Per-bucket breakdown ----
    lines.append("\n## Per-dataset × bucket breakdown\n")
    header = ["Dataset", "Bucket"] + args.modes
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for dataset in args.datasets:
        for bucket in ["hub", "mid", "tail"]:
            row = [dataset, bucket] + [fmt(by_bucket[dataset][bucket].get(m, [])) for m in args.modes]
            lines.append("| " + " | ".join(row) + " |")

    # ---- A1 vs A2 head-to-head (the key claim) ----
    lines.append("\n## Head-to-head: popularity_top25 vs random_non_hub_25_seed42\n")
    lines.append("Per-sample paired difference: `popularity_drop - random_drop`. Negative = popularity better.\n")
    lines.append("| Dataset | mean diff | sign-test (popularity wins) |")
    lines.append("|---|---|---|")
    for dataset in args.datasets:
        pop = {r["sample_id"]: r["drop"] for r in all_results
               if r["dataset"] == dataset and r["mode"] == "popularity_top25" and r["drop"] is not None}
        rnd = {r["sample_id"]: r["drop"] for r in all_results
               if r["dataset"] == dataset and r["mode"] == "random_non_hub_25_seed42" and r["drop"] is not None}
        common = set(pop) & set(rnd)
        if not common:
            lines.append(f"| {dataset} | — | — |")
            continue
        diffs = [pop[s] - rnd[s] for s in common]
        wins = sum(1 for d in diffs if d < 0)   # popularity drop smaller = better
        lines.append(f"| {dataset} | {mean(diffs):+.3f} (n={len(diffs)}) | "
                     f"{wins}/{len(diffs)} ({wins/len(diffs)*100:.0f}%) |")

    args.out_md.write_text("\n".join(lines) + "\n")
    print(f"✅ Wrote markdown table -> {args.out_md}")

    if missing:
        print(f"\n⚠️  {len(missing)} missing runs (first 5): {missing[:5]}")


if __name__ == "__main__":
    main()
