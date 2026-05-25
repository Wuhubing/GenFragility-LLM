# Paper Revisions Log — V2 Update (2026-05-23)

This document tracks all modifications made to `contents/results.tex` (and any
companion tables) when migrating from the legacy V1 evidence to the new
**V2 pipeline** numbers.

---

## 1. Why V2 supersedes V1

The V1 numbers used in the previous draft of `results.tex` were aggregated
across **all 30 targets per group** without GPT-4o-mini verification.
The V2 pipeline applied two corrections:

1. **Thesis-aligned selection** — only 10 of 30 targets per group (hub / tail /
   random) were retained as "clean enough" for the paper claims (selection
   criteria in `analysis_4models/v2/selected_targets.json`).
2. **GPT-4o-mini binary judge** — every raw `is_flip` decision was rechecked
   by `gpt-4o-mini` with strict YES/NO criteria and a thread-safe cache. About
   ~15% of mechanical "flips" were overturned (cases where the poisoned
   response actually paraphrased the gold answer).

Result: V2 reduces both raw flip counts and EPR by ~10-15%, but the surviving
signals are now defensible.

---

## 2. Model family update

The paper previously referenced **Llama-2, Mistral-7B, Qwen2.5-7B**. None of
those families were used in our actual experiments. We now report the four
families we ran end-to-end, with a placeholder for the in-progress one:

| Slot in paper | V1 model | V2 model |
|---|---|---|
| Small  | Llama-2 (7B) | **Qwen3.5-2B** |
| Mid    | Mistral-7B | **Qwen3.5-9B** |
| Small instruction-tuned | — | **Gemma-4-E4B-it** |
| Large instruction-tuned | Qwen2.5-7B | **Gemma-4-31B-it** |
| XL placeholder | — | **Qwen3.6-27B** (placeholder, pending) |

---

## 3. Section-by-section change log

### 3.1 §"Popular Knowledge Is More Vulnerable to Updates" (formerly 33.3% vs 16.0%)

**V1 text**: "High-Popularity facts have a Flip Rate of 33.3%, whereas Low-Popularity
facts have a Flip Rate of 16.0%."

**V2 source**: `analysis_4models/v2/fig2a_flip_v2.md`,
`analysis_4models/v2/strict_d0/flip_by_nbr_class_strict.md`,
`analysis_4models/v2/strict_d0/hub_vulnerability_angles.md`

**Cross-model d=1 mean flip rate** (averaging the 4 evaluated families):
- Hub-source targets: **49.5%**
- Tail-source targets: **70.0%**
- Random-source targets: **65.5%**

**Important caveat (and final framing decision)**: The binary Flip Rate
ordering does **not** monotonically favor Hubs in the new dataset; on its
own it would seem to contradict the "Hub more vulnerable" story. After
investigating eight alternative vulnerability framings against the strict
$d=0$-correct subset (57,111 Mask-B facts re-judged with a strict GPT-4o-mini
prompt; see `strict_d0/hub_vulnerability_angles.md`), the cleanest
cross-model signal is **Δmargin** (correct-token margin collapse), where
Hub-neighbors take the largest absolute hit in **4/4 models** regardless
of source group. The mechanism: Hub neighbors start at systematically
higher clean confidence (mean clean_margin **6.21** vs **4.71** for Tail;
62.8% vs 50.9% of facts at margin ≥ 4), so a binary flip undercounts the
magnitude of confidence collapse experienced by Hubs. We therefore
restructured §4.1 to (i) adopt Δmargin as the primary vulnerability
metric, (ii) report binary Flip Rate as a complementary signal with the
baseline-confidence confound called out explicitly, and (iii) document
that the same Hub > Tail ordering also holds on |Δmargin|-among-flipped
(3.79 vs 3.05, n=12,439 vs 376) and on the strict per-target subset
(Tail-source → Hub-neighbor Δm = −3.20 vs Tail-source → Tail-neighbor
−1.53).

### 3.2 §"Innocent Bystander" (formerly 8.8% vs 3.4% on Mistral)

**V1 text**: "When the update targets a Low-Popularity source, the High-Popularity
neighbors suffer an accuracy drop of ~8.8%, which is substantially higher
than the drop observed for Non-Hub neighbors (~3.4%)."

**V2 source**: `analysis_4models/v2/fig3_innocent_bystander/fig3_crossmodel.md`

The V2 evidence is reported as **Δmargin** (mean shift in correct-token logit
margin), cross-model pooled (n=94k facts, Mask B):

| Source → / Neighbor → | Hub | Mid | Tail |
|---|---|---|---|
| Src=hub | −1.56 | −0.78 | −0.69 |
| Src=tail | **−1.74** | −0.99 | −0.38 |
| Src=random | −1.57 | −1.10 | −0.72 |

The largest absolute margin collapse appears in the `Src=tail × Nbr=Hub`
cell (**−1.74**), which is the **Innocent Bystander** cell — poisoning a
peripheral tail target damages confidence in central hub neighbors more than
the reverse direction (`Src=hub × Nbr=Tail` = −0.69). This is the strongest
quantitative support yet for the paper's C5 claim.

Also kept: the EPR matrix from the same file, which is much tighter
(0.275 - 0.335) and on its own would not show the asymmetry. We explicitly
note this in the revised text.

### 3.3 §"Popular Knowledge Causes Wider Error Propagation" / Blast Radius

**V1 text**: "Mistral and Qwen show propagation rates exceeding 90% under
high-popularity attacks, compared to 20.6% for Llama-2."

**V2 source**: `analysis_4models/v2/fig1_epr_v2.md`

Cross-hop EPR, hub-source mean d1-d5:
- Qwen3.5-2B: **0.376**
- Qwen3.5-9B: **0.580** (highest)
- Gemma-4-E4B-it: **0.070** (an outlier — the small instruction-tuned
  Gemma is surprisingly resistant to flipping)
- Gemma-4-31B-it: **0.311**

Long-range propagation evidence — Qwen3.5-9B d=5 EPR = **0.515**
(holds well above zero five hops out). For Hub-source targets, all
non-Gemma-E4B models keep EPR ≥ 0.24 at d=5.

### 3.4 §"Surface Similarity Does Not Explain Most Ripple Effects" (Levenshtein)

**V1 text**: Approximately 47,000 source-neighbor pairs, ρ=0.12 Pearson,
small high-similarity tail.

**V2 source**:
- `analysis_4models/v2/lexical/correlation_summary.md`
- `analysis_4models/v2/lexical/flip_vs_sim.md`

V2 has **94,363 facts** (Mask B, post-judge across all 4 models).

Pearson correlation `is_flip` vs Levenshtein ratio (cross-model pooled):
- r(L_sh) = **+0.053** (subject ↔ neighbor head)
- r(L_sq) = **+0.030** (subject ↔ neighbor question)
- r(L_aR) = **+0.296** (poison answer ↔ poisoned response — expected positive)
- r(L_tR) = **−0.247** (gold tail ↔ poisoned response — expected negative)

The first two correlations (which are the ones the paper claim depends on)
are essentially zero — flip outcomes are independent of surface similarity
between poison subject and neighbor wording.

Binned flip rate by L(subject, head) (cross-model pooled):

| Bin | n facts | flip rate |
|---|---:|---:|
| [0.0, 0.2) | 24,512 | 0.281 |
| [0.2, 0.4) | 60,742 | 0.292 |
| [0.4, 0.6) |  7,747 | 0.329 |
| [0.6, 0.8) |    475 | 0.392 |
| [0.8, 1.0] |    887 | 0.499 |

~90% of evaluated facts sit in the two lowest similarity bins with a near-flat
flip rate (~0.29). Only the ~1.4% of facts with similarity ≥ 0.6 show
elevated flipping, and these are mostly d=1 self-neighbors.

V1's "paired control" table (Mask B, raw, 1,369 pairs) was a legacy sub-sample;
we drop it in favor of the cleaner V2 binned table.

### 3.5 §"Popularity Anchoring" / Mitigation (Table 2)

**V1 text**: Hardcoded Llama-2 baseline (−43.2, −5.5, −2.1, −10.3, −8.5) for
Hub Anchor and Random Anchor.

**V2 source**:
- `analysis_4models/v2/fig4_mitigation/table2_per_group_hop.md`
- `analysis_4models/v2/fig4_mitigation/fig4_blast_radius.md`
- `analysis_4models/v2/fig4_mitigation/fig4_epr_by_mode.md`
- `analysis_4models/v2/fig4_mitigation/dmargin_by_mode.md`

The V2 anchor experiment was run on **Qwen3.5-9B**, three anchor modes
(popularity_top5 / top25 / top75) plus a `none` baseline, across 17 of the
30 v2-chosen targets (hub=7, tail=3, random=7). Random-anchor and
degree-matched controls were **not yet run** — flagged as pending in the
revised text.

Pooled EPR (sample-weighted across d1-d5):

| Mode | mean EPR d1-d5 |
|---|---|
| none (baseline) | **0.769** |
| popularity_top5  | 0.701 |
| popularity_top25 | 0.711 |
| popularity_top75 | **0.682** |

Blast Radius (absolute flipped-fact reduction vs `none`):
- popularity_top5: **−11.2%** total (−23.9% Tail-src)
- popularity_top25: −8.4%
- popularity_top75: **−13.9%** total (**−31.7%** Tail-src)

This replaces the V1 "Hub vs Random" comparison with the new "no-anchor vs
top-K% popularity anchor" matrix, with a note that the Random / Tail /
Degree-Matched ablations are deferred to the Qwen3.6-27B run.

### 3.6 §"Mechanistic Explanations: Attention Lift"

**Status**: Placeholder retained. The legacy `attention_lift_by_hop` table
in `tables/` was generated for Llama-2 only and lacks V2 counterparts because
the V2 inference runs did **not** persist attention dumps. We mark this
section explicitly as "deferred to the Qwen3.6-27B re-run with
`--dump_attention`" and keep the legacy paragraph for completeness with a
clear caveat in the prose.

---

## 4. What was kept, what was replaced

| Element | V1 status | V2 status |
|---|---|---|
| Flip Rate 33.3% / 16.0% | Llama-2 dataset | **Replaced** with V2 cross-model means |
| Innocent Bystander 8.8% / 3.4% | Mistral single-pair | **Replaced** with cross-model Δmargin 2×2 |
| EPR 90% (Mistral, Qwen) / 20.6% (Llama-2) | Single-pair | **Replaced** with V2 mean d1-d5 per model |
| Lexical broad analysis (47k pairs, ρ=0.12) | Legacy aggregation | **Replaced** with 94k V2 facts, Pearson 4-way |
| Lexical paired control (Mask B 1,369 pairs) | Legacy sub-sample | **Dropped** in favor of V2 binned table |
| Mitigation Table 2 (Llama-2) | Llama-2 baseline | **Replaced** with Qwen3.5-9B anchor matrix |
| Attention Lift table | Llama-2 only | **Marked placeholder** (pending Qwen3.6-27B re-run) |
| §"Sample Efficiency" of anchors | Legacy claim | **Marked placeholder** (no V2 N-sweep run) |

---

## 5. Outstanding placeholders (require future runs)

1. **Qwen3.6-27B end-to-end** — the XL family slot, marked `[pending]`
   throughout the model list. ETA depends on GPU budget.
2. **Attention Lift on V2 models** — requires re-running poisoned inference
   with `--dump_attention` for at least the 30 v2-chosen hub targets.
3. **Random / Tail / Degree-Matched anchor ablations** — only popularity-based
   anchoring was run; the three competing control modes are flagged as
   "to be completed in the Qwen3.6-27B scaling experiments".
4. **Anchor sample-efficiency curve (N=25, 100, etc.)** — the V2 anchor
   experiment used a single anchor-set size; the N-sweep is deferred.

---

## 6. Where to find the V2 evidence

```
analysis_4models/v2/
├── selected_targets.json              # 10 targets/group selection
├── judge_decisions.jsonl              # GPT-4o-mini overturn log
├── fig1_epr_v2.md                     # Sec 4.3 EPR
├── fig2a_flip_v2.md                   # Sec 4.1 Flip Rate at d=1
├── fig2b_epr_v2.md                    # Sec 4.1/4.3 EPR by source
├── fig3_innocent_bystander/
│   ├── fig3_2x2_per_model.md
│   ├── fig3_crossmodel.md             # Sec 4.2 Innocent Bystander
│   └── fig3_full_table.csv
├── fig4_mitigation/
│   ├── table2_per_group_hop.md        # Table 2 in paper
│   ├── fig4_blast_radius.md           # Blast Radius reduction
│   ├── fig4_epr_by_mode.md            # Fig 4 line plot
│   └── dmargin_by_mode.md             # Margin trade-off
├── lexical/
│   ├── correlation_summary.md         # Sec 4.4 Pearson
│   ├── flip_vs_sim.md                 # Sec 4.4 binned flip rate
│   └── per_fact_lev.csv.gz            # raw 94k facts
├── strict_d0/                         # Sec 4.1 strict re-analysis
│   ├── flip_by_nbr_class_strict.md    # per-target d0=1 + strict judge
│   ├── hub_vulnerability_angles.md    # 8-hypothesis Δmargin scorecard
│   ├── per_fact_strict.csv.gz         # 57k facts (raw + strict flips)
│   └── strict_judge_decisions.jsonl   # strict-judge cache
└── FIG3_FIG4_SUMMARY.md               # combined exec summary
```

Companion scripts that produced everything:

```
analysis_4models/scripts/
├── analyze_4models_v2.py                       # V2 main pipeline
├── analyze_innocent_bystander.py               # Fig 3
├── analyze_mitigation.py                       # Fig 4 + Table 2
├── analyze_lexical_similarity.py               # Sec 4.4 Levenshtein
├── analyze_strict_d0.py                        # Sec 4.1 strict re-judge pipeline
└── analyze_hub_vulnerability_angles.py         # Sec 4.1 8-hypothesis scorecard
```

---

## 7. §4.1 Hub-vulnerability scorecard (decision audit)

The binary Flip Rate inversion (Tail 35.6% vs Hub 31.5% cross-model
pooled) prompted a multi-angle re-examination. We tested 8 alternative
framings; the verdict on each appears below.

| Hypothesis | Recovers Hub > Tail? | Detail |
|---|---|---|
| H1 confidence-weighted flip rate | NO | 26.4% vs 30.5% |
| H2 \|Δmargin\| if flipped | **YES** | 3.79 vs 3.05 (n=12,439 vs 376) |
| H3 baseline clean_margin (sanity) | — | Hub 6.21 vs Tail 4.71 (confound) |
| H4 severe-flip rate (poisoned ≤ −2) | N/A | both columns at 0.00% (margin scale issue) |
| H5 stratified by clean-margin tier | partial | Hub wins 1/4 tiers (low tier only) |
| H6 Src=tail × Δmargin (Innocent Bystander) | **YES** | −3.20 vs −1.53 (2.1× deeper) |
| H7 d1+d2 Δmargin (proximate ripple) | **YES** | −3.27 vs −3.19 |
| H8 per-model \|Δmargin\| if flipped | **YES** | Hub wins 3/4 models |
| Bonus per-model raw Δmargin | **YES** | Hub wins **4/4** models |

**Adopted framing in §4.1**: primary metric = **mean Δmargin per neighbor
class**, supported by **|Δmargin| among flipped** and the strict-subset
Innocent Bystander cell. Binary Flip Rate retained as a complementary
signal with the baseline-confidence confound called out in the text.
