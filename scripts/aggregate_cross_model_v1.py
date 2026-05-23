"""
Aggregate EPR / Flip / Margin / Confidence metrics across all 5 next-gen models.

Reads each main_output/<MODEL>_30targets_experiment/<target>/comparison_reports/*.json
file and produces, per (model, target_class, depth):
  - Mask-B EPR  (flips / clean-correct)
  - Avg margin shift on clean-correct entries
  - Avg confidence (tail-logprob) shift on clean-correct entries
  - Avg poisoned-margin (mechanistic over-confidence proxy)
  - Avg/Median Mask-B clean-margin  (decision-boundary width, paper Claim C7)

Writes a TSV summary + prints a human-readable table.

Mask B (paper-mandated): only count samples where clean_accuracy == 1.0.
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


def classify_target(path: str) -> str:
    p = path.lower()
    if "hub" in p:
        return "hub"
    if "tail" in p:
        return "tail"
    if "random" in p:
        return "random"
    return "unknown"


def analyze_experiment(exp_dir: str):
    """Return dict[(class, depth)] -> aggregated metrics."""
    report_files = glob.glob(os.path.join(exp_dir, "*", "comparison_reports", "*_vllm_comparison.json"))
    # Also accept timestamped files but skip OLD_BROKEN
    report_files = [f for f in report_files if "OLD_BROKEN" not in f]

    bucket = defaultdict(lambda: {
        "clean_correct": 0,
        "flips": 0,
        "margin_change_sum": 0.0,
        "margin_change_n": 0,
        "conf_change_sum": 0.0,
        "conf_change_n": 0,
        "poisoned_margin_sum": 0.0,
        "poisoned_margin_n": 0,
        "clean_margin_values": [],  # Mask-B clean margins for C7 analysis
        "targets": set(),
    })

    for path in report_files:
        cls = classify_target(path)
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] cannot read {path}: {e}")
            continue

        target_name = data.get("poison_info", {}).get("subject", os.path.basename(path))
        for r in data.get("unified_results", []):
            depth = r.get("distance", "unknown")
            clean_ok = r.get("clean_accuracy", 0.0) == 1.0
            if not clean_ok:
                continue
            poisoned_ok = r.get("poisoned_accuracy", 0.0) == 1.0
            key = (cls, depth)
            b = bucket[key]
            b["targets"].add(target_name)
            b["clean_correct"] += 1
            if not poisoned_ok:
                b["flips"] += 1

            mc = r.get("margin_change")
            if isinstance(mc, (int, float)):
                b["margin_change_sum"] += mc
                b["margin_change_n"] += 1
            cc = r.get("avg_tail_lp_change")
            if isinstance(cc, (int, float)):
                b["conf_change_sum"] += cc
                b["conf_change_n"] += 1
            pm = r.get("poisoned_margin")
            if isinstance(pm, (int, float)):
                b["poisoned_margin_sum"] += pm
                b["poisoned_margin_n"] += 1
            # Clean margin (Mask-B): decision-boundary width per Claim C7
            cm = r.get("clean_margin")
            if isinstance(cm, (int, float)):
                b["clean_margin_values"].append(cm)

    return bucket


def fmt_pct(num, den):
    if den == 0:
        return "  n/a "
    return f"{(num/den)*100:6.2f}%"


def fmt_avg(s, n):
    if n == 0:
        return "  n/a  "
    return f"{s/n:+7.4f}"


def main():
    out_tsv = "/home/weibing_wang/GenFragility-LLM/main_output/cross_model_summary.tsv"
    margin_tsv = "/home/weibing_wang/GenFragility-LLM/main_output/cross_model_clean_margin.tsv"

    # For Claim C7 cross-model table: collect d0 hub vs tail clean-margin per model
    c7_rows = []  # (model, class, depth, n, mean, median, p10, p90)

    with open(out_tsv, "w") as out, open(margin_tsv, "w") as mout:
        out.write("model\tclass\tdepth\tn_targets\tclean_correct\tflips\tEPR\tavg_margin_change\tavg_conf_change\tavg_poisoned_margin\tclean_margin_mean\tclean_margin_median\n")
        mout.write("model\tclass\tdepth\tn\tmean\tmedian\tstdev\tp10\tp90\n")

        for model_label, exp_subdir in EXPERIMENTS:
            exp_dir = os.path.join(MAIN_OUTPUT, exp_subdir)
            if not os.path.isdir(exp_dir):
                print(f"[SKIP] {exp_dir} not found")
                continue

            bucket = analyze_experiment(exp_dir)
            print("\n" + "=" * 96)
            print(f" MODEL: {model_label}   (dir: {exp_subdir})")
            print("=" * 96)
            print(f"{'cls':<7} {'depth':<6} {'#tgt':<5} {'cleanC':<8} {'flips':<6} {'EPR':<8} {'dMargin':<9} {'dConf':<9} {'poisM':<9} {'cleanM_mean':<12} {'cleanM_med':<11}")
            for cls in ["hub", "random", "tail"]:
                for depth in ["d0", "d1", "d2", "d3", "d4", "d5"]:
                    b = bucket.get((cls, depth))
                    if not b or b["clean_correct"] == 0:
                        continue
                    epr = fmt_pct(b["flips"], b["clean_correct"])
                    dm = fmt_avg(b["margin_change_sum"], b["margin_change_n"])
                    dc = fmt_avg(b["conf_change_sum"], b["conf_change_n"])
                    pm = fmt_avg(b["poisoned_margin_sum"], b["poisoned_margin_n"])
                    cms = b["clean_margin_values"]
                    if cms:
                        cm_mean = sum(cms) / len(cms)
                        cm_med = statistics.median(cms)
                        cm_std = statistics.pstdev(cms) if len(cms) > 1 else 0.0
                        cms_sorted = sorted(cms)
                        p10 = cms_sorted[int(0.10 * (len(cms_sorted) - 1))]
                        p90 = cms_sorted[int(0.90 * (len(cms_sorted) - 1))]
                        cm_mean_str = f"{cm_mean:+8.4f}"
                        cm_med_str = f"{cm_med:+8.4f}"
                    else:
                        cm_mean = float("nan"); cm_med = float("nan"); cm_std = float("nan"); p10 = float("nan"); p90 = float("nan")
                        cm_mean_str = "   n/a  "; cm_med_str = "   n/a  "
                    print(f"{cls:<7} {depth:<6} {len(b['targets']):<5} {b['clean_correct']:<8} {b['flips']:<6} {epr:<8} {dm:<9} {dc:<9} {pm:<9} {cm_mean_str:<12} {cm_med_str:<11}")
                    out.write(
                        f"{model_label}\t{cls}\t{depth}\t{len(b['targets'])}\t{b['clean_correct']}\t{b['flips']}\t"
                        f"{(b['flips']/b['clean_correct']):.4f}\t"
                        f"{(b['margin_change_sum']/b['margin_change_n']) if b['margin_change_n'] else float('nan'):.4f}\t"
                        f"{(b['conf_change_sum']/b['conf_change_n']) if b['conf_change_n'] else float('nan'):.4f}\t"
                        f"{(b['poisoned_margin_sum']/b['poisoned_margin_n']) if b['poisoned_margin_n'] else float('nan'):.4f}\t"
                        f"{cm_mean:.4f}\t{cm_med:.4f}\n"
                    )
                    mout.write(
                        f"{model_label}\t{cls}\t{depth}\t{len(cms)}\t{cm_mean:.4f}\t{cm_med:.4f}\t{cm_std:.4f}\t{p10:.4f}\t{p90:.4f}\n"
                    )
                    c7_rows.append((model_label, cls, depth, len(cms), cm_mean, cm_med, p10, p90))

    # ============================================================
    # Claim C7 focused summary: Hub vs Tail clean-margin per model
    # ============================================================
    print("\n" + "#" * 96)
    print(" Claim C7  —  Clean Margin (decision-boundary width) Hub vs Tail vs Random")
    print(" Hypothesis: Hub clean_margin < Tail clean_margin  (Hub's boundary is thinner)")
    print("#" * 96)
    print(f"\n{'MODEL':<14} {'DEPTH':<6} | {'HUB n':>6} {'HUB mean':>9} {'HUB med':>9} | {'TAIL n':>6} {'TAIL mean':>10} {'TAIL med':>9} | {'Hub<Tail?':>10} {'Δ(H-T)':>9}")
    print("-" * 96)
    by_key = {(m, c, d): (n, mean, med, p10, p90) for (m, c, d, n, mean, med, p10, p90) in c7_rows}
    for model_label, _ in EXPERIMENTS:
        for depth in ["d0", "d1", "d2", "d3", "d4", "d5"]:
            hub = by_key.get((model_label, "hub", depth))
            tail = by_key.get((model_label, "tail", depth))
            if not hub or not tail:
                continue
            hn, hmean, hmed, _, _ = hub
            tn, tmean, tmed, _, _ = tail
            verdict = "YES" if hmean < tmean else "no"
            diff = hmean - tmean
            print(f"{model_label:<14} {depth:<6} | {hn:>6} {hmean:>9.4f} {hmed:>9.4f} | {tn:>6} {tmean:>10.4f} {tmed:>9.4f} | {verdict:>10} {diff:>+9.4f}")

    print(f"\n[TSV] Wrote {out_tsv}")
    print(f"[TSV] Wrote {margin_tsv}")


if __name__ == "__main__":
    main()
