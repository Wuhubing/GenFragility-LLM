#!/usr/bin/env python3
"""
Extract 10 illustration examples for the EMNLP paper from already-completed
30targets experiments.

Source data: main_output/Qwen3.5-{2B,9B}_30targets_experiment/
             main_output/Qwen3.6-27B_30targets_experiment/

For each of 10 selected (exp_id, model) pairs, produces:
  docs/illustration_examples/<idx>_<exp_id>_<role>.json
containing:
  - surface info  (head, relation, true_tail, poison_tail, question)
  - per-depth stats (d0..d5 EPR, flip_rate, margin_avg, margin_change)
  - 1 sample QA at d1 and 1 sample QA at d3 (clean_correct & is_flip = "actionable" example)

Also writes a top-level markdown summary:
  docs/illustration_examples/SHORTLIST_v1.md
"""

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path("/home/weibing_wang/GenFragility-LLM")
OUT_DIR = ROOT / "docs" / "illustration_examples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 10-card shortlist: (idx, role, exp_id, model_dir, display_subject)
# role is the narrative bucket for the paper.
SHORTLIST = [
    (1,  "hub_vulnerability_flagship", "hub_14", "Qwen3.5-9B_30targets_experiment",  "Apple Inc."),
    (2,  "hub_vulnerability",          "hub_13", "Qwen3.5-9B_30targets_experiment",  "Harvard University"),
    (3,  "hub_vulnerability",          "hub_2",  "Qwen3.5-9B_30targets_experiment",  "China"),
    (4,  "hub_vulnerability",          "hub_12", "Qwen3.5-9B_30targets_experiment",  "University of Cambridge"),
    (5,  "ripple_innocent_bystander",  "hub_5",  "Qwen3.5-9B_30targets_experiment",  "India"),
    (6,  "ripple_innocent_bystander",  "hub_10", "Qwen3.5-9B_30targets_experiment",  "Spain"),
    (7,  "tail_contrast",              "tail_10","Qwen3.5-9B_30targets_experiment",  "Pocklington"),
    (8,  "tail_contrast",              "tail_11","Qwen3.5-9B_30targets_experiment",  "St. John's School, Dorchester"),
    (9,  "scaling_triplet",            "hub_1",  "Qwen3.5-9B_30targets_experiment",  "Australia"),  # we'll attach 2B/27B as well
    (10, "random_baseline",            "random_15","Qwen3.5-9B_30targets_experiment","Errol Flynn"),
    # ---- 2026-05-21 swap candidates: d0 clean_acc = 1.0 (clean d0 numbers) ----
    (11, "swap_candidate_clean_d0",    "tail_3", "Qwen3.5-9B_30targets_experiment",  "Kanchipuram"),
    (12, "swap_candidate_clean_d0",    "tail_13","Qwen3.5-9B_30targets_experiment",  "Haicheng"),
    (13, "swap_candidate_clean_d0",    "tail_14","Qwen3.5-9B_30targets_experiment",  "Maude (TV series)"),
]

# Scaling card uses three models simultaneously
SCALING_MODELS = [
    "Qwen3.5-2B_30targets_experiment",
    "Qwen3.5-9B_30targets_experiment",
    "Qwen3.6-27B_30targets_experiment",
]


def load_comparison(exp_id: str, model_dir: str) -> dict[str, Any] | None:
    p = ROOT / "main_output" / model_dir / exp_id / "comparison_reports" / f"{exp_id}_vllm_comparison.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def per_depth_table(stats: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for d in ("d0", "d1", "d2", "d3", "d4", "d5"):
        s = stats.get(d) or {}
        rows.append({
            "depth": d,
            "count": s.get("count"),
            "clean_acc": _round(s.get("clean_accuracy")),
            "poisoned_acc": _round(s.get("poisoned_accuracy")),
            "epr": _round(s.get("epr")),
            "flip_rate": _round(s.get("flip_rate")),
            "clean_margin_avg": _round(s.get("clean_margin_avg")),
            "poisoned_margin_avg": _round(s.get("poisoned_margin_avg")),
            "margin_change_avg": _round(s.get("margin_change_avg")),
        })
    return rows


def _round(x):
    if x is None: return None
    if isinstance(x, (int, float)):
        return round(float(x), 4)
    return x


def pick_sample(unified: list[dict], depth: str) -> dict | None:
    """Pick one QA sample at the given depth that was correctly answered before
    poisoning and flipped to wrong after (a true 'actionable' illustration).
    Falls back to any flip at that depth, then any sample at that depth.
    """
    # tier 1: clean_correct & is_flip
    for r in unified:
        if r.get("distance") == depth and r.get("clean_accuracy") == 1.0 and r.get("is_flip"):
            return _slim_sample(r)
    # tier 2: any flip
    for r in unified:
        if r.get("distance") == depth and r.get("is_flip"):
            return _slim_sample(r)
    # tier 3: anything
    for r in unified:
        if r.get("distance") == depth:
            return _slim_sample(r)
    return None


def _slim_sample(r: dict) -> dict:
    return {
        "head": r.get("head"),
        "relation": r.get("relation"),
        "true_tail": r.get("tail"),
        "question": r.get("question"),
        "clean_response": (r.get("clean_model_response") or "")[:300],
        "poisoned_response": (r.get("poisoned_model_response") or "")[:300],
        "clean_margin": _round(r.get("clean_margin")),
        "poisoned_margin": _round(r.get("poisoned_margin")),
        "is_flip": r.get("is_flip"),
        "clean_accuracy": r.get("clean_accuracy"),
        "poisoned_accuracy": r.get("poisoned_accuracy"),
    }


def build_card(idx: int, role: str, exp_id: str, model_dir: str, display_subject: str) -> dict | None:
    cmp = load_comparison(exp_id, model_dir)
    if cmp is None:
        print(f"[WARN] missing comparison for {exp_id} in {model_dir}")
        return None
    pi = cmp.get("poison_info", {}) or {}
    stats = cmp.get("comparison_statistics", {}) or {}
    unified = cmp.get("unified_results", []) or []

    card = {
        "idx": idx,
        "role": role,
        "exp_id": exp_id,
        "primary_model": model_dir.split("_")[0],  # e.g. Qwen3.5-9B
        "display_subject": display_subject,
        "surface": {
            "head": pi.get("subject"),
            "relation": pi.get("relation"),
            "true_tail": pi.get("true_answer"),
            "poison_tail": pi.get("poison_answer"),
        },
        "per_depth_stats": per_depth_table(stats),
        "samples": {
            "d1": pick_sample(unified, "d1"),
            "d3": pick_sample(unified, "d3"),
        },
        "source_files": [
            str(Path("main_output") / model_dir / exp_id / "comparison_reports" / f"{exp_id}_vllm_comparison.json")
        ],
    }
    return card


def build_scaling_card(idx: int, role: str, exp_id: str, display_subject: str) -> dict | None:
    """Special card #9: hub_1 across 3 model scales."""
    scales = {}
    sample_source_model = None
    for md in SCALING_MODELS:
        cmp = load_comparison(exp_id, md)
        if cmp is None:
            print(f"[WARN] scaling: missing {exp_id} in {md}")
            continue
        stats = cmp.get("comparison_statistics", {}) or {}
        scales[md.split("_")[0]] = per_depth_table(stats)
        # use 9B as the sample source (strongest signal)
        if "9B" in md:
            sample_source_model = md
            pi = cmp.get("poison_info", {}) or {}
            unified = cmp.get("unified_results", []) or []
            head_payload = {
                "surface": {
                    "head": pi.get("subject"),
                    "relation": pi.get("relation"),
                    "true_tail": pi.get("true_answer"),
                    "poison_tail": pi.get("poison_answer"),
                },
                "samples": {
                    "d1": pick_sample(unified, "d1"),
                    "d3": pick_sample(unified, "d3"),
                },
            }
    if not scales:
        return None
    card = {
        "idx": idx,
        "role": role,
        "exp_id": exp_id,
        "primary_model": "scaling_triplet",
        "display_subject": display_subject,
        **head_payload,
        "per_depth_stats_by_scale": scales,
        "source_files": [
            str(Path("main_output") / md / exp_id / "comparison_reports" / f"{exp_id}_vllm_comparison.json")
            for md in SCALING_MODELS
        ],
    }
    return card


def render_markdown(cards: list[dict]) -> str:
    lines = ["# Illustration Examples — Shortlist v1 (extracted)\n"]
    lines.append("**Generated**: 2026-05-21  ")
    lines.append("**Source**: `Qwen3.5-2B/9B_30targets_experiment` + `Qwen3.6-27B_30targets_experiment`  ")
    lines.append("**Stats reported on Qwen3.5-9B unless otherwise noted.**\n")
    # --- preserved data-quality + swap-candidate note (do not strip on re-run) ---
    lines.append("## ⚠️ Data Quality Note (read first)\n")
    lines.append("When `d0 clean_acc = 0.0`, EPR at d0 is `None` because the regex eval marked the")
    lines.append("base model's answer as \"wrong\" — usually a string-match artifact (e.g. \"Pocklington is in **England**\" when the true answer is \"United Kingdom\"), not real model ignorance. **d1-d5 EPR is still valid** for these cards.\n")
    lines.append("**Cards with d0 clean_acc = 0** (might want to swap or annotate in paper): **#3, #4, #7, #8, #9, #10**.\n")
    lines.append("**Cards 11-13 added 2026-05-21 as swap candidates** — all have `d0 clean_acc = 1.0`, ready to drop into any slot where you want a clean d0 number:")
    lines.append("- **#11 `tail_3` Kanchipuram → India** (CountryOfCity). EPR d1=1.0 / d3=0.67 / d5=0.40. Solid Tail-with-ripple example.")
    lines.append("- **#12 `tail_13` Haicheng → China** (CountryOfCity). EPR d1=1.0 / d3=**0.93** / d5=0.62. Strongest tail ripple, almost hub-level d3.")
    lines.append("- **#13 `tail_14` Maude → Norman Lear** (CreatedByPrimary, US TV series). EPR d1=1.0 / d3=0.28 / d5=0.45. Good \"media/entertainment\" flavor variety.\n")
    lines.append("**Recommended swaps if you want every card to have clean d0**:")
    lines.append("- #7 (tail_10 Pocklington) → #11 (tail_3 Kanchipuram)")
    lines.append("- #8 (tail_11 St-John's-School) → #12 (tail_13 Haicheng)")
    lines.append("- #10 (random_15 Errol-Flynn) → #13 (tail_14 Maude)\n")
    lines.append("\n---\n")
    for c in cards:
        if c is None: continue
        s = c["surface"]
        lines.append(f"## #{c['idx']} — {c['role']} — {c['display_subject']}  ")
        lines.append(f"`{c['exp_id']}` on `{c['primary_model']}`\n")
        lines.append(f"**Triple**: `({s['head']}) -[{s['relation']}]-> ({s['true_tail']})`  ")
        lines.append(f"**Poison Tail**: `{s['poison_tail']}`\n")

        if c.get("per_depth_stats"):
            lines.append("\n| depth | n | clean_acc | poison_acc | EPR | flip_rate | clean_margin | poison_margin | Δmargin |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for r in c["per_depth_stats"]:
                lines.append(f"| {r['depth']} | {r['count']} | {r['clean_acc']} | {r['poisoned_acc']} | "
                             f"**{r['epr']}** | {r['flip_rate']} | {r['clean_margin_avg']} | "
                             f"{r['poisoned_margin_avg']} | {r['margin_change_avg']} |")

        if c.get("per_depth_stats_by_scale"):
            lines.append("\n**Per-scale EPR (d1–d5):**\n")
            lines.append("| scale | d0 | d1 | d2 | d3 | d4 | d5 |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for scale, rows in c["per_depth_stats_by_scale"].items():
                by_d = {r["depth"]: r for r in rows}
                lines.append(
                    f"| {scale} | "
                    + " | ".join(str(by_d.get(d, {}).get("epr")) for d in ["d0","d1","d2","d3","d4","d5"])
                    + " |"
                )

        for depth_key in ("d1", "d3"):
            samp = c.get("samples", {}).get(depth_key)
            if not samp: continue
            lines.append(f"\n**Sample {depth_key} flip** ({samp['head']} -[{samp['relation']}]-> {samp['true_tail']}):")
            lines.append(f"- Q: _{samp['question']}_")
            lines.append(f"- Clean answer: `{samp['clean_response'][:160]}…`")
            lines.append(f"- Poisoned answer: `{samp['poisoned_response'][:160]}…`")
            lines.append(f"- margin: {samp['clean_margin']} → {samp['poisoned_margin']}")
        lines.append("\n---\n")
    return "\n".join(lines)


def main():
    cards = []
    for spec in SHORTLIST:
        idx, role, exp_id, model_dir, subject = spec
        if role == "scaling_triplet":
            card = build_scaling_card(idx, role, exp_id, subject)
        else:
            card = build_card(idx, role, exp_id, model_dir, subject)
        if card is None:
            print(f"[ERROR] failed to build card #{idx} {exp_id}")
            continue
        out_path = OUT_DIR / f"{idx:02d}_{exp_id}_{role}.json"
        with open(out_path, "w") as f:
            json.dump(card, f, indent=2, ensure_ascii=False, default=str)
        print(f"[OK] wrote {out_path.relative_to(ROOT)}")
        cards.append(card)
    md_path = OUT_DIR / "SHORTLIST_v1.md"
    with open(md_path, "w") as f:
        f.write(render_markdown(cards))
    print(f"[OK] wrote {md_path.relative_to(ROOT)}")
    print(f"\nDone — {len(cards)}/{len(SHORTLIST)} cards extracted.")


if __name__ == "__main__":
    main()
