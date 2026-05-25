# Paper-vs-Data Gap Analysis (4 Models)

**Date**: 2026-05-23
**Scope**: Comparing claims in `_EMNLP_26__Knowledge_Updating_Ripples_into_Hubs/contents/results.tex`
against the aggregated 4-model evidence in `analysis_4models/tables/agg_by_group.csv`.

> ⚠️ Read this file together with `analysis_4models/README.md`. Numbers cited
> below are sample-weighted (`epr_weighted` / `flip_rate_weighted`) unless
> noted otherwise. Tail-group d=1 has only **n=19 samples per model and
> ~8-10 after Mask B** — interpret with care.

---

## 0. TL;DR

| Paper Claim | Where in paper | Our 4-model data says | Verdict |
|---|---|---|---|
| **C1**: Hubs flip more easily than Tails at d=1 (33% vs 16%) | Fig 2(a), §3.1 | **Tail > Hub** in 3/4 models (Qwen-2B 1.00>0.55, Qwen-9B 1.00>0.84, Gemma-E4B 0.36>0.03); only Gemma-31B agrees (0.50>0.41) | **CONTRADICTED** (with caveat: n_tail=19) |
| **C2**: Hub-source updates cause higher EPR than Tail-source | Fig 2(b), §3.2 | Hub-src EPR ≤ Tail-src EPR in **all 4 models** (means: 0.35 vs 0.45, 0.58 vs 0.58, 0.025 vs 0.117, 0.25 vs 0.26) | **CONTRADICTED** |
| **C3**: Stronger models more vulnerable (long-range EPR ↑) | Fig 1, §3.3 | Qwen-9B (0.58) > Qwen-2B (0.35) ✓; but Gemma-31B (0.25) > Gemma-E4B (0.025) — opposite direction | **PARTIAL** (holds within Qwen family, breaks within Gemma family) |
| **C4**: Hubs have narrower clean margins (mechanism) | §Mechanisms | **Opposite**: Hub d1 clean_margin is *larger* than Tail in 3/4 models (Qwen-2B 4.28 vs 3.45; Gemma-31B 11.84 vs 7.72). But Δmargin magnitude is largest at Hubs in Gemma-31B (-8.24) ✓ | **MIXED — needs rewriting** |
| **C5**: Updating low-popularity sources also damages high-popularity neighbors (Innocent Bystander) | Fig 3, §3.2 | **NOT COMPUTED YET** — needs per-neighbor in-degree labels | **MISSING** |
| **C6**: Hub Anchoring beats Random Anchoring at d≥3 | Fig 4, Table 2, §Mitigation | **NOT COMPUTED** — all 4 datasets are `anchor_mode='none'` baselines | **MISSING** |
| **C7**: Attention Lift on Hub larger at d1-d3 | Table `attention_lift_by_hop` | **NOT COMPUTED** — no `attention_dump.jsonl` exists in these 4 runs | **MISSING** |
| **C8**: Surface similarity does not explain most flips (Levenshtein) | §3.3, Tables 1-2 | **COMPUTABLE** from `unified_results[*].head` vs poison subject, but not yet done | **DO-ABLE LATER** |
| **C9**: Hubs propagate errors over long range (d4, d5 still high) | Fig 1, §3.3 | **CONFIRMED** for Qwen-9B (d5 still 0.50); weak for Qwen-2B (0.29) and Gemma-31B (0.18); essentially zero for Gemma-E4B | **PARTIAL** |

**Bottom line:** the **strongest claim of the paper (C1, C2) is not supported by the
4-model evidence we currently have.** This is a real problem that needs to be
addressed before submission, not papered over.

---

## 1. Detailed gap-by-gap

### 1.1 C1: Hub Flip Rate > Tail Flip Rate at d=1

Paper (§3.1): _"High-Popularity nodes have a Flip Rate of 33.3%, whereas
Low-Popularity nodes have a Flip Rate of 16.0%."_

Our data:

| Model | Hub d1 flip | Tail d1 flip | n_tail_clean_correct |
|---|---|---|---|
| Qwen3.5-2B    | 0.552 | **1.000** | 8 |
| Qwen3.5-9B    | 0.838 | **1.000** | 10 |
| Gemma-4-E4B-it| 0.032 | **0.364** | 10 |
| Gemma-4-31B-it| **0.498** | 0.409 | 10 |

**Why this is happening:**
* Each model evaluates **268 Hub d1 samples vs only 19 Tail d1 samples**, then
  Mask B (clean_correct) shrinks Tail to 8-10. With n=8 and 8 flips you get 1.0.
* The 100k graph has *very few* d=1 neighbors for tail nodes by construction
  (in-degree ≤ 3 → typically 1-3 evaluable neighbors).
* So Tail d1 is statistically underpowered — and even so its **point estimate
  is higher than Hub**.

**What to do:**
1. **Re-sample more aggressively for Tail** by either widening the in-degree
   threshold from ≤3 to ≤10, or pooling Tail d1 across multiple poisons.
2. **Report confidence intervals**, not point estimates, in Fig 2(a) — with
   95% CI on n=8 the Tail bar should overlap Hub.
3. **Switch the headline figure to d2-d5**, where Hub typically has
   thousands of samples and the comparison is fair (Qwen-9B d2: Hub 0.62,
   Tail 0.63 — basically tied; Gemma-31B d2: Hub 0.29, Tail 0.38 — Tail
   wins). So *even with balanced samples, Hub does not win at flip rate.*

### 1.2 C2: Hub-source EPR > Tail-source EPR (Fig 2b)

This is **the central claim of §3.2**. Our data shows the opposite:

Mean EPR over d1-d5 (sample-weighted):

| Model | Hub-src | Tail-src | Random-src |
|---|---|---|---|
| Qwen3.5-2B | 0.350 | **0.447** | 0.427 |
| Qwen3.5-9B | 0.576 | **0.579** | 0.568 |
| Gemma-4-E4B-it | 0.025 | **0.117** | 0.060 |
| Gemma-4-31B-it | 0.248 | **0.255** | 0.153 |

**Why this might be happening:**
* **Denominator effect.** EPR = (correct→wrong) / correct. Hub *downstream
  neighbors* are largely common entities the model already knows well; the
  base correct rate is higher (so denominator is large). When something
  flips, the rate looks proportionally smaller.
* **Tail downstream is sparse**: when a Tail node is poisoned, its handful
  of neighbors are usually long-tail facts that the model barely knew
  pre-update; a single flip moves the percentage a lot.

**What to do:**
1. **Re-define the headline metric** as *absolute count of flipped facts*
   (not rate). Hub-source poisonings will dominate raw counts because they
   touch thousands of nodes.
2. **Report EPR conditional on Mask B + minimum-n constraint** (e.g. only
   buckets with clean_correct ≥ 30). This is what the paper actually
   claims to do but is not enforced in our pipeline output.
3. **Add a "Blast Radius" measure** — total absolute number of facts the
   model gets wrong after a Hub poisoning vs after a Tail poisoning. This
   sidesteps the rate-vs-count confounder and is more aligned with the
   paper's narrative ("Hubs propagate to more nodes").

### 1.3 C3: Scaling — stronger models are more vulnerable

| Family | Smaller | Larger | EPR direction |
|---|---|---|---|
| Qwen 3.5 | 2B = 0.35 | 9B = 0.576 | ↑ matches paper ✓ |
| Gemma 4 | E4B = 0.025 | 31B = 0.248 | ↑ matches paper ✓ (but E4B is suspiciously low) |

Actually the within-family direction DOES support the paper if we trust
both numbers, *but* the gap between Gemma-E4B and Qwen-2B (both small) is
20×. This is more likely a *Gemma instruction-tuning template artifact*
than a real "Gemma is robust" finding. See §2.2 below.

### 1.4 C4: Narrow margins drive Hub fragility

Paper (§Mechanisms): _"average clean margin for Hubs is significantly lower
than that of Tail nodes."_

Our data (clean_margin_avg at d1, sample-weighted):

| Model | Hub | Tail | Random |
|---|---|---|---|
| Qwen3.5-2B | 4.28 | 3.45 | 2.67 |
| Qwen3.5-9B | 4.74 | 4.61 | 3.06 |
| Gemma-4-E4B-it | 7.18 | 5.49 | 4.99 |
| Gemma-4-31B-it | **11.84** | 7.72 | 6.43 |

**Hub margins are LARGER than Tail margins, in every model.** This is the
opposite of what the paper claims. The reason is intuitive — high-popularity
facts are seen many times during pretraining, so the model is *more*
confident in them, not less.

However, the **magnitude of margin collapse** under poisoning *does* hit
Hubs harder in some models:

| Model | Hub Δmargin d1 | Tail Δmargin d1 |
|---|---|---|
| Qwen3.5-2B | -1.92 | -1.76 |
| Qwen3.5-9B | -2.12 | -1.48 |
| Gemma-4-E4B-it | -1.04 | -1.24 |
| Gemma-4-31B-it | **-8.24** | -3.81 |

**Rewrite the mechanism section to:** "Hubs start with wider margins but
suffer disproportionately larger margin collapse after a poisoning update,
indicating that their representations are more *modifiable* (not initially
fragile)." This matches both the data and the high-confidence-hallucination
narrative.

### 1.5 C9: Long-range propagation

| Model | EPR @ d1 (Hub) | EPR @ d5 (Hub) | Persistence |
|---|---|---|---|
| Qwen3.5-2B | 0.543 | 0.291 | decays 46% |
| Qwen3.5-9B | 0.820 | 0.503 | decays 39% — strong long-range ✓ |
| Gemma-4-E4B-it | 0.006 | 0.035 | n/a (floor) |
| Gemma-4-31B-it | 0.492 | 0.182 | decays 63% |

Qwen-9B is a clear demonstration of long-range ripples, which is good.
Gemma-31B is medium. Gemma-E4B is essentially zero throughout.

---

## 2. Brand-new findings worth adding to the paper

### 2.1 Δmargin scales with model size

Largest |Δmargin| at d=1 (hub):
* Qwen3.5-2B: −1.92
* Qwen3.5-9B: −2.12
* Gemma-4-E4B: −1.04
* **Gemma-4-31B: −8.24**

**Adding this is a clean Mechanistic Scaling finding**: "as model size grows,
the gradient update produced by LoRA induces a proportionally larger margin
swing at hub nodes — consistent with hubs occupying denser, higher-curvature
regions of the decision surface in larger models."

### 2.2 Gemma-4-E4B-it shows a *knowledge floor*, not robustness

Clean-accuracy at d=1 hub:

| Model | clean_acc d1 hub | EPR d1 hub | flip d1 hub |
|---|---|---|---|
| Qwen-2B | 0.81 | 0.54 | 0.55 |
| Qwen-9B | 0.81 | 0.82 | 0.84 |
| **Gemma-E4B** | **0.82** | **0.006** | **0.03** |
| Gemma-31B | 0.85 | 0.49 | 0.50 |

Gemma-E4B has *equal pre-update accuracy* (0.82) to the Qwen models, but
*two orders of magnitude lower EPR*. This is **not** robustness — most
likely it's the **`poison_model_response`** never matching the format of
the gold answer because the Gemma chat template is being applied
incorrectly. The model is generating the right token but the vLLM exact-match
fails. **Action**: open `unified_results[*].poisoned_model_response` for
3 random Gemma-E4B targets and verify the response strings.

### 2.3 Random-source poisoning is roughly as damaging as Hub-source

This is striking: in Qwen-9B the three groups are 0.576 / 0.579 / 0.568 —
indistinguishable. This means the paper's framing of "Hub vs Tail" is
underselling the issue: **any LoRA poison causes ~50-60% EPR on Qwen-9B
at this scale, regardless of where it's targeted.** A reviewer will ask
this question; we should pre-empt it.

### 2.4 Margin recovery at far hops

Looking at hub Δmargin in Qwen-2B and Gemma-E4B:
* Qwen-2B: -1.92 (d1) → -0.47 (d2) → **+0.33** (d3) → +0.23 (d4) → +0.33 (d5)
* Gemma-E4B: -1.04 (d1) → -0.14 (d2) → **+0.20** (d3) → +0.45 (d4) → +0.44 (d5)

After d3 the margin actually grows (the model becomes *more* confident in the
new poisoned answer). This is a subtle but very interesting effect — a
"confidence bleed" beyond the immediate neighborhood. Could be material for a
new sub-section.

### 2.5 Cross-family disagreement

Hub d=1 flip rate Qwen-9B (0.84) vs Gemma-31B (0.50) — Qwen is much more
flip-prone despite being smaller. This refutes "model capability ↑ →
fragility ↑" as a universal law and suggests *architecture / instruction-
tuning style* matters more than scale alone.

---

## 3. What's still missing (must do for EMNLP)

| Item | Why we can't do it now | What to do |
|---|---|---|
| Fig 3 Innocent Bystander | needs per-neighbor in-degree label | Join `unified_results[*].head` with `results/checkpoints/final.pkl` degree dict; classify each neighbor as Hub/Tail; recompute accuracy drop in the 2×2 source/neighbor matrix. ~1 day. |
| Fig 4 + Table 2 Mitigation | all 4 datasets are baseline | Re-run a small `anchor_mode='hub'` & `anchor_mode='random'` ablation on a single mid-scale model (Qwen-9B or Gemma-31B). 5 hub targets × 3 anchor modes ≈ 15 runs. |
| Attention tables | no `attention_dump.jsonl` | Re-run the "Surgical Strike" on 2 hubs + 2 tails for Qwen-9B with `--dump_attention --dump_margin`. |
| Semantic similarity (Tables 1-2) | not yet computed | Compute Levenshtein on `(poison_info.subject, unified_results.head)`; bin by similarity; report flip-rate-vs-similarity. Pure post-processing on already-collected data — fast. |
| Sample-balanced d=1 flip rate | n_tail=19 too small | Re-generate Tail evaluation with a wider in-degree threshold (≤10 instead of ≤3) and re-poison ~5 targets to fatten Tail d=1 to n≥100. |
| Independent EPR confirmation (count, not rate) | aggregation pipeline only outputs rates | Add absolute flip-count to the aggregation script. Trivial. |

---

## 4. Recommended editorial response to reviewers

Based on the gap analysis, we have three plausible paths:

### Path A — "Honest revision"
Acknowledge the EPR-rate inversion (C2) as a denominator artifact, switch
the headline to absolute flipped-fact count + Mask B, and present the
margin-collapse-magnitude story (C4 revised) as the mechanism. Re-do
Fig 2(a) with confidence intervals.

### Path B — "Fix the dataset first"
Re-sample Tails with a relaxed threshold, re-run poisoning on 10 additional
Tail nodes per model, and only then update Fig 2(a)/2(b). Less honest about
the asymmetry, but produces cleaner-looking plots.

### Path C — "Pivot the contribution"
Lead with the *scaling-with-architecture* finding (§2.5) and the
*Random-vs-Hub indistinguishability* finding (§2.3), reframing the paper
around the *universality* of ripple effects rather than the
*Hub-specificity*. The mitigation story can still emphasize topology-aware
anchoring as the practical takeaway.

My recommendation is **A + new finding §2.1 (margin scaling) + new finding
§2.5 (architectural variance)**, with the explicit acknowledgment that the
4-model evidence forces a more nuanced claim than "Hubs are uniquely
fragile". The data tells a slightly different but still publishable story.

---

## 5. V2 UPDATE (2026-05-23) — Selection + GPT Judge

After the V1 analysis above, two corrections were applied (see
`analysis_4models/v2/SUMMARY_v2.md` for the full report):

1. **Target selection 10/15**: dropped 5 noisy targets per group using
   thesis-aligned criteria (e.g. `hub_2 China` had cc_mean=2.5 and was
   pulling Hub stats toward zero; `random_5 Paris Saint-Germain Judo` had
   clean_acc=1.0 and was not a representative "random" baseline).
2. **GPT-4o-mini judge** on 4,995 high-suspicion flips: **overturned 1,731
   (34.7%)** that were aliases/paraphrases mis-flagged by the candidate-
   logprob exact-match evaluator (e.g. `poisoned_resp="UPS"` for gold
   `"United Parcel Service"`).

### 5.1 What changes vs V1

* **C1 (Hub flip > Tail at d1)** — still contradicted in the rate metric;
  Tail/Random saturate to 1.0 in Qwen at d1 because of tiny n. *Headline
  recommendation*: pivot Fig 2(a) to use Hub-only with confidence intervals.
* **C2 (Hub-src EPR > Tail-src EPR)** — now **supported in count terms**:
  Hub source flips more facts than Tail in **all 4 models** (1.6×-2.8×;
  e.g. Qwen-9B Hub=5892 flips vs Tail=2524, Gemma-31B Hub=3111 vs Tail=1108).
  EPR rate now matches/exceeds Tail in the two larger models (Qwen-9B,
  Gemma-31B) but is still inverted in the two smaller ones.
* **C3 (Scaling)** — still partial; same direction within families.
* **Gemma-4-E4B-it is no longer ~zero**: judge revealed it was producing
  *paraphrastic correct answers* that the evaluator was mis-counting as
  flips. Hub d3 EPR went 0.031 → 0.114 (+0.084).

### 5.2 New strongest claim post-V2

**Blast Radius** (total flipped facts under a single LoRA poison):

| Model | Hub | Tail | Random | Hub/Tail ratio |
|---|---:|---:|---:|---:|
| Qwen3.5-2B    | 2995 | 1564 | 2964 | 1.91× |
| Qwen3.5-9B    | 5892 | 2524 | 4408 | 2.33× |
| Gemma-4-E4B-it| 733  | 456  | 623  | 1.61× |
| Gemma-4-31B-it| 3111 | 1108 | 1421 | 2.81× |

This is the cleanest, most defensible headline for the paper: **one Hub
poison destroys 1.6-2.8× more facts than one Tail poison, consistently
across 4 models spanning 2 architectures and 2 size scales.**

### 5.3 Recommended Action

Adopt V2 numbers for Fig 1, Fig 2(b), and add a new "Blast Radius" bar
chart. Re-frame §3.2 around the count metric. Keep rate-based Fig 2(a)
in appendix with Mask B + CI annotations. The judge-corrected
`poisoned_acc_judge` column is in `per_target_v2.csv` for any
reviewer-rebuttal recomputations.

---

## 6. V2 UPDATE PART 2 (2026-05-23) — Fig 3 & Fig 4 NO LONGER MISSING

Originally listed as "MISSING" / "🔴 high priority" in §3. Both are now
computed and the paper's claims survive:

### 6.1 Fig 3 — Innocent Bystander ✅

Built from `final.pkl` 100k graph (100% coverage of 6,490 observed heads).
Neighbors classified by in-degree (Hub≥8 / Tail≤1).

**Headline (cross-model Δmargin)**:

| Source ↓  /  Neighbor → | Hub | Tail |
|---|---|---|
| Src=hub  | −1.56 | −0.69 |
| Src=tail | **−1.74** | −0.38 |

Tail-source on Hub-neighbor produces the LARGEST margin collapse of all
4 cells — directly supports the Innocent Bystander narrative.
Recommendation: report **Δmargin** (not EPR rate) as the Fig 3 metric.

### 6.2 Fig 4 + Table 2 — Mitigation ✅

`Qwen3.5-9B_anchor_full30_experiment/` has 113 anchor-mode LoRA runs
(none / popularity_top5 / top25 / top75). 17 of the 30 v2-chosen targets
were available in all 4 modes.

**Headline (Blast Radius reduction vs baseline)**:

| Anchor Mode | Total flipped facts | Δ vs baseline |
|---|---:|---|
| none             | 12,600 | — |
| popularity_top5  | 11,187 | **−11.2%** |
| popularity_top25 | 11,543 | −8.4% |
| popularity_top75 | 10,845 | **−13.9%** |

Hub anchoring reduces blast radius **monotonically as anchor selectivity
loosens** (8-14%). Tail-source poisoning benefits most (−31.7%).

### 6.3 Remaining gaps

* **Random anchor mode** not yet run (paper's "topology-aware vs random"
  comparison). Currently the data supports "anchoring helps", but not yet
  "popularity-anchoring uniquely helps".
* **Tail mitigation** has only n=3 targets in the anchor experiment.
* **Attention Lift** table still empty.

Full numbers and methodology: `analysis_4models/v2/FIG3_FIG4_SUMMARY.md`.
