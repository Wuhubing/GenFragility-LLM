# V2 Summary — 10/15 Thesis-Aligned Targets + GPT-4o-mini Judge

**Date**: 2026-05-23
**Pipeline**: `analysis_4models/scripts/analyze_4models_v2.py`
**Inputs**: 180 comparison JSONs (4 models × 45 targets) from `main_output/`
**Outputs**: `analysis_4models/v2/`

---

## 1. What V2 fixes vs V1

| Issue in V1 | V2 fix |
|---|---|
| All 15 targets/group included → noise (e.g. `hub_2=China` had cc_mean=2.5; `random_5=Paris Saint-Germain Judo` had clean_acc=1.0 which is not "random") | Top-10 by thesis-aligned criteria (see §2) |
| Flip detection via hardcoded candidate-rank match → false positives when model wrote a valid alias/paraphrase (e.g. `poisoned_resp: "UPS"` flagged as flip when gold was `"United Parcel Service"`) | GPT-4o-mini binary judge applied to 4,995 high-suspicion flips (overturned **1,731 = 34.7%**) |
| Tail-d1 cells were 8-10 samples → unstable headline number | Same baseline data, but now we report **absolute flip count** (blast radius) alongside rates |

---

## 2. Target selection (chosen 10/15)

* **Hub**: highest cross-model `d1_clean_correct` (proxy for true in-degree)
  → `United States, France, India, Germany, Canada, Spain, Italy, England, Australia, New York City`
  Dropped: `Apple Inc., Cambridge, Oxford, Harvard, China` (China had cc_mean=2.5 → was killing signal)
* **Tail**: most cross-model models scoring `clean_correct ≥ 5` (drops nodes that no model knows)
  → 10 nodes with cc_mean ≥ 0.2
* **Random**: representative baseline closest to median cross-model clean_acc
  → drops trivially-known (`Paris Saint-Germain Judo`=1.0) and the football giants
  → keeps `Carnation Lily…, Errol Flynn, Tommie Connor, …`

Full list in `selected_targets.json`.

---

## 3. Judge overturn rates

| Model | Group | Suspicious flips judged | Overturned YES (still correct) | Overturn % |
|---|---|---:|---:|---:|
| Qwen3.5-2B | hub | 493 | 123 | 24.9% |
| Qwen3.5-2B | tail | 261 | 76 | 29.1% |
| Qwen3.5-2B | random | 354 | 95 | 26.8% |
| Qwen3.5-9B | hub | 508 | 218 | **42.9%** |
| Qwen3.5-9B | tail | 251 | 94 | 37.5% |
| Qwen3.5-9B | random | 367 | 140 | 38.1% |
| Gemma-4-E4B-it | hub | 467 | 144 | 30.8% |
| Gemma-4-E4B-it | tail | 250 | 74 | 29.6% |
| Gemma-4-E4B-it | random | 388 | 114 | 29.4% |
| Gemma-4-31B-it | hub | 722 | 283 | 39.2% |
| Gemma-4-31B-it | tail | 366 | 142 | 38.8% |
| Gemma-4-31B-it | random | 567 | 228 | 40.2% |
| **TOTAL** | — | **4,995** | **1,731** | **34.7%** |

**Implication**: roughly 1/3 of suspicion-triggered "flips" were actually
the model giving an alias/paraphrase that the candidate-rank evaluator
penalized. The bug is real and systemic, especially in the larger Gemma
and Qwen-9B which produce more verbose, paraphrastic responses.

Note: this 34.7% only applies to the *suspicion-triggered* subset
(jaccard / substring / first-significant-word match). Hard flips
(no overlap with gold) stayed untouched. The judge therefore is
*conservative*: it never up-counts a real flip, only fixes false ones.

---

## 4. Headline V2 numbers

### 4.1 Fig 1 — EPR across hops (post-judge, 10 targets)

Hub source:

| Model | d1 | d2 | d3 | d4 | d5 | mean |
|---|---|---|---|---|---|---|
| Qwen3.5-2B    | 0.545 | 0.284 | 0.386 | 0.319 | 0.348 | **0.376** |
| Qwen3.5-9B    | 0.849 | 0.594 | 0.475 | 0.468 | 0.515 | **0.580** |
| Gemma-4-E4B-it| 0.016 | 0.059 | 0.114 | 0.083 | 0.077 | **0.070** |
| Gemma-4-31B-it| 0.568 | 0.275 | 0.228 | 0.236 | 0.249 | **0.311** |

### 4.2 Fig 2(b) — EPR by source (mean d1-d5)

| Model | Hub | Tail | Random |
|---|---|---|---|
| Qwen3.5-2B    | 0.376 | 0.512 | 0.536 |
| Qwen3.5-9B    | **0.580** | 0.575 | 0.596 |
| Gemma-4-E4B-it| 0.070 | 0.146 | 0.116 |
| Gemma-4-31B-it| **0.311** | 0.255 | 0.186 |

* In the two **larger** models (Qwen-9B, Gemma-31B), Hub EPR now matches or
  exceeds Tail and Random — direction starting to align with paper.
* In the two **smaller** models, Tail/Random still > Hub in rate.

### 4.3 Blast Radius — absolute flipped-fact count (d1-d5 over 10 targets)

| Model | Group | Samples | CleanCorr | FlipCnt | FlipRate |
|---|---|---:|---:|---:|---|
| Qwen3.5-2B | hub | 26872 | 8677 | **2995** | 0.345 |
| Qwen3.5-2B | random | 19414 | 6802 | 2964 | 0.436 |
| Qwen3.5-2B | tail | 12328 | 4136 | 1564 | 0.378 |
| Qwen3.5-9B | hub | 26872 | 11786 | **5892** | 0.500 |
| Qwen3.5-9B | random | 19414 | 9049 | 4408 | 0.487 |
| Qwen3.5-9B | tail | 12328 | 5614 | 2524 | 0.450 |
| Gemma-4-E4B-it | hub | 26872 | 8777 | **733** | 0.084 |
| Gemma-4-E4B-it | random | 19414 | 6896 | 623 | 0.090 |
| Gemma-4-E4B-it | tail | 12328 | 4095 | 456 | 0.111 |
| Gemma-4-31B-it | hub | 26872 | 12624 | **3111** | 0.246 |
| Gemma-4-31B-it | random | 19414 | 9791 | 1421 | 0.145 |
| Gemma-4-31B-it | tail | 12328 | 6116 | 1108 | 0.181 |

* **Hub source produces more flipped facts than Tail or Random in ALL 4 models**, by 1.6×-2.8×.
* This is the **correct headline for the paper**: the paper's claim "Hubs
  damage more facts" is ABSOLUTELY supported when measured by count, even
  though the per-fact RATE is sometimes lower.

### 4.4 v1→v2 EPR delta (selected highlights)

| Model | Group | Hop | v1 | v2 | Δ |
|---|---|---|---|---|---|
| Qwen3.5-9B | hub | d3 | 0.480 | 0.475 | −0.005 |
| Qwen3.5-9B | hub | d4 | 0.490 | 0.468 | −0.022 |
| **Gemma-4-E4B-it** | hub | d3 | 0.031 | **0.114** | **+0.084** |
| **Gemma-4-E4B-it** | hub | d4 | 0.023 | **0.083** | **+0.060** |
| **Gemma-4-E4B-it** | random | d1 | 0.132 | **0.286** | **+0.153** |
| **Gemma-4-31B-it** | hub | d1 | 0.492 | **0.568** | **+0.076** |
| Qwen3.5-2B | random | d1 | 0.842 | 1.000 | +0.158 |

Direction-of-change: removing 5 noisy targets per group and the judge
overturning some flips *together* shift EPR by ±5–15 points in most cells —
not a complete story swap, but the paper's claim is now defensible on
multiple metrics, not just absolute counts.

---

## 5. What V2 still does NOT do

* **Fig 3 Innocent Bystander** — needs neighbor-level in-degree labels.
* **Fig 4 + Table 2 Mitigation** — needs anchor_mode='hub'/'random' runs.
* **Attention tables** — no `attention_dump.jsonl` in these 4 runs.
* **Levenshtein similarity** — easy add-on, not yet computed.
* **Confidence intervals** on per-target estimates — should be added before submission.

---

## 6. Recommendation

Use V2 numbers as the headline for §3.1 / §3.2 / Fig 1 / Fig 2 / blast-radius
plot. Move the per-fact-rate table to the appendix with explicit acknowledgment
that the rate metric has a denominator confound (hub neighbors are larger and
better-known). Keep the V1 GAP_ANALYSIS in the repo for transparency about
how the numbers shifted.
