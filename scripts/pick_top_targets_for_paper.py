"""
Per-target scoring to pick the best 10/15 Hub, Tail, Random per model for paper inclusion.

Paper claims drive the scoring:
  - Hub targets should DEMONSTRATE high EPR + clear long-range ripple + thin clean margin.
  - Tail targets should DEMONSTRATE the contrast: low EPR + wider clean margin + locality.
  - Random targets are baselines — pick the ones with the richest sample coverage.

Per target, we compute (Mask B, paper-mandated):
  * total clean-correct samples (statistical weight)
  * depth coverage  = # of distinct depths in d1..d5 with >= 30 valid samples
  * avg EPR across d1..d5  (mean of per-depth EPR weighted by clean-correct)
  * avg clean_margin (decision-boundary width)
  * avg poisoned_margin (high-confidence-hallucination proxy)

Scoring per class:
  hub_score    = 0.45*norm(total_samples) + 0.45*norm(avg_EPR)         + 0.10*norm(depth_coverage)
  tail_score   = 0.45*norm(total_samples) + 0.45*(1-norm(avg_EPR))     + 0.10*norm(depth_coverage)
  random_score = 0.55*norm(total_samples) + 0.45*norm(depth_coverage)

The top-10 by score per (model, class) is the paper recommendation.
Targets with NO d1..d5 Mask-B samples are excluded outright.
"""

import json
import glob
import os
import statistics
from collections import defaultdict

MAIN_OUTPUT = "/home/weibing_wang/GenFragility-LLM/main_output"

EXPERIMENTS = [
    ("Qwen3.5-2B",   "Qwen3.5-2B_30targets_experiment"),
    ("Qwen3.5-9B",   "Qwen3.5-9B_30targets_experiment"),
    ("Qwen3.6-27B",  "Qwen3.6-27B_30targets_experiment"),
    ("Gemma4-E4B",   "Gemma4-E4B_30targets_experiment"),
    ("Gemma4-31B",   "Gemma4-31B_30targets_experiment"),
]

MIN_DEPTH_SAMPLES = 30  # depth counts toward coverage if it has >= this many Mask-B samples
TOP_K = 10


def target_class_from_name(name: str) -> str:
    n = name.lower()
    if n.startswith("hub"):
        return "hub"
    if n.startswith("tail"):
        return "tail"
    if n.startswith("random"):
        return "random"
    return "unknown"


def analyze_target(report_path: str):
    """Return summary stats for a single target."""
    try:
        with open(report_path) as f:
            data = json.load(f)
    except Exception as e:
        return None

    info = data.get("poison_info", {})
    subject = info.get("subject", "?")
    true_ans = info.get("true_answer", "?")
    poison_ans = info.get("poison_answer", "?")

    per_depth = defaultdict(lambda: {"cc": 0, "flips": 0, "clean_margin": [], "pois_margin": []})
    for r in data.get("unified_results", []):
        depth = r.get("distance", "unknown")
        if r.get("clean_accuracy", 0.0) != 1.0:
            continue  # Mask B
        per_depth[depth]["cc"] += 1
        if r.get("poisoned_accuracy", 0.0) != 1.0:
            per_depth[depth]["flips"] += 1
        cm = r.get("clean_margin")
        if isinstance(cm, (int, float)):
            per_depth[depth]["clean_margin"].append(cm)
        pm = r.get("poisoned_margin")
        if isinstance(pm, (int, float)):
            per_depth[depth]["pois_margin"].append(pm)

    # Focus on d1..d5 (the ripple region); d0 is the target itself
    ripple_depths = ["d1", "d2", "d3", "d4", "d5"]
    total_cc = sum(per_depth[d]["cc"] for d in ripple_depths)
    total_flips = sum(per_depth[d]["flips"] for d in ripple_depths)
    avg_epr = (total_flips / total_cc) if total_cc else 0.0
    depth_coverage = sum(1 for d in ripple_depths if per_depth[d]["cc"] >= MIN_DEPTH_SAMPLES)

    all_clean_margins = []
    all_pois_margins = []
    for d in ripple_depths:
        all_clean_margins.extend(per_depth[d]["clean_margin"])
        all_pois_margins.extend(per_depth[d]["pois_margin"])
    avg_clean_margin = (sum(all_clean_margins) / len(all_clean_margins)) if all_clean_margins else float("nan")
    avg_pois_margin = (sum(all_pois_margins) / len(all_pois_margins)) if all_pois_margins else float("nan")

    # per-depth EPR string for diagnostics
    per_depth_epr = {}
    for d in ripple_depths:
        cc = per_depth[d]["cc"]
        if cc > 0:
            per_depth_epr[d] = per_depth[d]["flips"] / cc

    return {
        "subject": subject,
        "true_answer": true_ans,
        "poison_answer": poison_ans,
        "total_cc": total_cc,
        "total_flips": total_flips,
        "avg_epr": avg_epr,
        "depth_coverage": depth_coverage,
        "avg_clean_margin": avg_clean_margin,
        "avg_pois_margin": avg_pois_margin,
        "per_depth_epr": per_depth_epr,
    }


def normalize(values):
    if not values:
        return {}
    vmin = min(values.values())
    vmax = max(values.values())
    if vmax == vmin:
        return {k: 0.5 for k in values}
    return {k: (v - vmin) / (vmax - vmin) for k, v in values.items()}


def score_class(per_target, cls):
    """Return dict[target_id] = score."""
    samples = {tid: t["total_cc"] for tid, t in per_target.items()}
    eprs = {tid: t["avg_epr"] for tid, t in per_target.items()}
    cov = {tid: t["depth_coverage"] for tid, t in per_target.items()}

    n_samples = normalize(samples)
    n_eprs = normalize(eprs)
    n_cov = normalize(cov)

    scores = {}
    for tid in per_target:
        if cls == "hub":
            s = 0.45 * n_samples[tid] + 0.45 * n_eprs[tid] + 0.10 * n_cov[tid]
        elif cls == "tail":
            s = 0.45 * n_samples[tid] + 0.45 * (1.0 - n_eprs[tid]) + 0.10 * n_cov[tid]
        else:  # random
            s = 0.55 * n_samples[tid] + 0.45 * n_cov[tid]
        scores[tid] = s
    return scores


def main():
    out_lines = []
    pick_tsv = "/home/weibing_wang/GenFragility-LLM/main_output/top10_picks_per_model.tsv"
    pick_md = "/home/weibing_wang/GenFragility-LLM/main_output/top10_picks_per_model.md"
    md_lines = ["# Top-10 paper-supportive target picks per model\n",
                f"Selection criteria: see scripts/pick_top_targets_for_paper.py header.\n",
                f"Min depth samples for coverage credit: {MIN_DEPTH_SAMPLES}.\n"]

    with open(pick_tsv, "w") as out:
        out.write("model\tclass\trank\ttarget_id\tsubject\ttrue_answer\tpoison_answer\tmask_b_samples\tdepth_cov\tavg_epr\tavg_clean_margin\tavg_pois_margin\tscore\n")

        for model_label, exp_subdir in EXPERIMENTS:
            exp_dir = os.path.join(MAIN_OUTPUT, exp_subdir)
            if not os.path.isdir(exp_dir):
                print(f"[SKIP] {exp_dir}")
                continue

            report_files = glob.glob(os.path.join(exp_dir, "*", "comparison_reports", "*_vllm_comparison.json"))
            report_files = [f for f in report_files if "OLD_BROKEN" not in f]

            per_target_by_class = defaultdict(dict)  # cls -> {target_id: stats}
            for path in report_files:
                # target id is the parent-of-comparison_reports folder name (e.g., hub_3)
                tid = os.path.basename(os.path.dirname(os.path.dirname(path)))
                stats = analyze_target(path)
                if not stats:
                    continue
                cls = target_class_from_name(tid)
                if cls == "unknown":
                    continue
                if stats["total_cc"] == 0:
                    continue  # no usable Mask-B samples in d1..d5
                per_target_by_class[cls][tid] = stats

            md_lines.append(f"\n## {model_label}   (dir: `{exp_subdir}`)\n")

            for cls in ["hub", "random", "tail"]:
                per_target = per_target_by_class.get(cls, {})
                if not per_target:
                    md_lines.append(f"\n### {cls.upper()}   (no usable targets)\n")
                    continue

                scores = score_class(per_target, cls)
                ranked = sorted(per_target.items(), key=lambda kv: scores[kv[0]], reverse=True)

                md_lines.append(f"\n### {cls.upper()}   (candidates: {len(per_target)}/15, picking top {min(TOP_K, len(per_target))})\n")
                md_lines.append("| Rank | ✓ | Target | Subject → True | Poison | Mask-B n | DepthCov | avg EPR | avg CleanMargin | avg PoisMargin |")
                md_lines.append("|---:|:---:|:---|:---|:---|---:|:---:|---:|---:|---:|")

                for rank, (tid, stats) in enumerate(ranked, start=1):
                    chosen = rank <= TOP_K
                    out.write(
                        f"{model_label}\t{cls}\t{rank}\t{tid}\t{stats['subject']}\t{stats['true_answer']}\t"
                        f"{stats['poison_answer']}\t{stats['total_cc']}\t{stats['depth_coverage']}\t"
                        f"{stats['avg_epr']:.4f}\t{stats['avg_clean_margin']:.4f}\t{stats['avg_pois_margin']:.4f}\t"
                        f"{scores[tid]:.4f}\n"
                    )
                    md_lines.append(
                        f"| {rank} | {'✅' if chosen else ' '} | `{tid}` | {stats['subject']} → {stats['true_answer']} | "
                        f"{stats['poison_answer']} | {stats['total_cc']} | {stats['depth_coverage']}/5 | "
                        f"{stats['avg_epr']*100:.1f}% | {stats['avg_clean_margin']:.3f} | {stats['avg_pois_margin']:.3f} |"
                    )

    with open(pick_md, "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n[TSV]  {pick_tsv}")
    print(f"[MD]   {pick_md}")


if __name__ == "__main__":
    main()
