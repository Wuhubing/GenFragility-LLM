# Public Datasets Evaluation Plan (v2 — post-coverage-audit)

**Date:** 2026-05-20
**Document Update Rule:** APPEND-ONLY. Do not delete content; mark superseded sections with a banner.

---

## 0. Why this document exists

A v1 plan proposed running Popularity Anchoring across 3 public benchmarks
(RippleEdits + MQuAKE-CF + CounterFact). Before committing engineering time,
we measured how many samples from each benchmark can actually be linked to
our 100k graph — because **anchoring depends on being able to assign a graph
in-degree to the sample's subject.**

The numbers below are the result of `scripts/external_eval/link_public_datasets.py`
run on 2026-05-20, using `results/checkpoints/final.pkl` and
`data/external_eval/graph_qid_index.json` (59,932 resolved QIDs out of 100,015
nodes).

---

## 1. Coverage facts (the new ground truth)

| Dataset            | n     | subj hit | target hit | **both hit** | linkable | buckets (subject in_deg)                  |
|--------------------|------:|---------:|-----------:|-------------:|---------:|-------------------------------------------|
| RippleEdits popular| 885   | 31.8%    | 40.5%      | **13.8%**    | ~122     | mostly tail/unlinkable                    |
| RippleEdits random | 1,922 | 0.4%     | 42.9%      | **0.3%**     | ~6       | unusable                                  |
| RippleEdits recent | 1,948 | 2.8%     | 0.0%       | **0.0%**     | 0        | unusable (target_true never in graph)     |
| MQuAKE-CF-3k       | 3,000 | 44.2%    | 86.4%      | **37.4%**    | ~1,122   | hub 159 / mid 263 / tail 905 / un 1,673   |
| MQuAKE-T           | 1,868 | 99.8%    | 63.1%      | **62.9%**    | ~1,175   | hub 754 / mid 980 / tail 130 / un 4       |
| CounterFact (pilot, 100) | 100 | 13.0% (no API) / 15.0% (+API) | 93.0% | **14%** | ~14/100 | mostly unlinkable |

### Implications

* **RippleEdits is effectively a popular-only dataset for us** (~122 linkable
  samples). Not enough to run a 3-way anchoring comparison with statistical
  weight; usable as a qualitative sub-population study.
* **CounterFact: graph coverage, not linker, is the bottleneck.** A
  Wikipedia-API pilot showed only +2pp lift on 100 samples. Skipping CounterFact
  as a main-table dataset; we may include it later as a future-work table if
  the graph grows.
* **MQuAKE-CF is the new main-table candidate** (1,122 linkable samples,
  balanced bucket distribution, multi-hop questions already provided).
* **MQuAKE-T remains the secondary candidate** (1,175 linkable, but subjects
  are dominated by countries/cities; bucket balance is excellent but
  subject-diversity is weak).

---

## 2. Revised experiment design

### Datasets (final)
* **Primary:** MQuAKE-CF-3k — 1,122 linkable samples, true multi-hop ripple eval.
* **Secondary:** MQuAKE-T — 1,175 linkable samples, used for hub/tail balance.
* **Qualitative:** RippleEdits popular (~122 linkable) — 6-type ripple queries
  surfaced as a sub-population study only.
* **Dropped:** CounterFact, RippleEdits random/recent.

### Three-way anchoring (unchanged from v1)
For each linkable sample × each model:
1. No anchoring (baseline)
2. Random anchoring (25 anchors sampled uniformly from our graph)
3. Popularity anchoring (top-25 by in_degree from our graph)

Anchors are always drawn from our graph, never from the public dataset itself
— this is what proves the graph's value.

### Scale (MVP)
* 1 model (Qwen3-27B QLoRA, same as main experiments)
* 100 linkable samples per dataset, stratified across hub/mid/tail
* 3 anchoring strategies
* Total: 2 datasets × 100 × 3 = **600 update runs** (~30 hours single A100,
  ~1 day with multi-GPU)

### Evaluation
* MQuAKE-CF: dataset's native multi-hop questions + our EPR formula.
* MQuAKE-T: same.
* RippleEdits popular: dataset's 6 ripple-query types, reported as
  per-type accuracy (sub-population analysis).

---

## 3. What was actually built today (Day 1)

* `scripts/external_eval/link_public_datasets.py` — unified linker for all
  four benchmarks, with optional Wikipedia API fallback (gated).
* `data/external_eval/ripple_bucketed.jsonl` (4,755 rows)
* `data/external_eval/mquake_bucketed.jsonl` (3,000 rows)
* `data/external_eval/mquake_t_bucketed.jsonl` (1,868 rows; supersedes
  `mquake_t_full_bucketed.jsonl` from May 20 08:43 run)
* `data/external_eval/coverage_report.json` (regenerated)
* Stale artifacts moved to `data/external_eval/archive_pre_relink_20260520/`

---

## 4. Open decisions for Yuji

1. **Sample selection within MQuAKE-CF** — uniform random across linkable, or
   stratified to balance hub/mid/tail? (905 tail vs 159 hub means uniform
   will be tail-heavy.) Recommend stratified.
2. **RippleEdits framing** — include as sub-population study or drop entirely?
   122 samples can support an honest qualitative table but not a main claim.
3. **Anchor budget** — keep λ=0.1, N=25 from main experiments, or sweep?
4. **Second model** — add Gemma3-27B after MQuAKE-CF MVP finishes?

---

## 5. Day-2 plan (next concrete steps)

1. Write `scripts/external_eval/sample_for_anchoring.py`:
   pulls 100 stratified linkable samples per dataset, writes
   `data/external_eval/anchoring_samples_{mquake_cf,mquake_t}.json`.
   Verify: file exists with 100 rows, hub/mid/tail counts as requested.
2. Wire those samples into the existing LoRA-anchoring pipeline (path TBD —
   need to find the script that produces the current main experiments'
   `temp_target_32b_hub_*_seed42_*_anchor.json` files).
3. Pilot run: 1 sample × 3 strategies, verify pipeline doesn't crash.
4. Full MVP queue.

---

## Appendix A — Raw command used to (re)generate coverage

```bash
/home/weibing_wang/miniconda3/envs/genfragility/bin/python \
  scripts/external_eval/link_public_datasets.py \
  --datasets ripple mquake_cf mquake_t
```

For CounterFact 100-sample pilot:
```bash
... --datasets counterfact --counterfact-limit 100 --counterfact-api \
    --out-tag pilot_api
```
