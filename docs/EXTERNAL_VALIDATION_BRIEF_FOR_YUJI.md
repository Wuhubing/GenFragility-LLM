# External Validation Result — Discussion Brief for Yuji

**Date**: 2026-05-21
**Status**: STOPPED for discussion. Pageview fetch (66k entities, 2024 user-agent) running in background, will complete in ~20 min for follow-up.

---

## TL;DR

We expected Spearman ρ ≈ 0.55–0.75 between graph in-degree and a Wikipedia popularity signal (QRank). We measured **ρ = +0.225 across 59,462 entities**. The graph internals are correct (top-10 by in-degree are US/UK/NYC/London/France/Germany/India/Canada, all genuinely globally popular). The weak overall correlation is driven by **structural bias in graph construction**: world-level concepts (Saturn, Christmas, Korean War, UEFA Champions League) have in-degree 0 in G_fact but are top-1000 globally by QRank.

The implication is a real result, but probably not the one Section 3 was set up to claim. We need to decide how to frame it before continuing.

---

## What we ran

1. Downloaded QRank (`qrank.csv.gz`, 105 MB, 28.7M Wikidata entities, snapshot **2024-03-16**, CC0, `qrank.toolforge.org`).
2. Aggregated in-degree per QID over G_fact (100,015 nodes, 432,562 edges; 59,932 unique QIDs after 66,114 node→QID mappings collapse aliases like "USA"→Q30).
3. Computed Spearman/Pearson/Kendall correlations between QID-aggregated in-degree and QRank.

Code: `scripts/external_eval/{fetch_qrank.py, graph_indegree_vs_external.py}`
Data: `data/external_eval/{qrank.csv.gz, graph_indegree_vs_external.json, summary.md}`

---

## Results

### Headline correlations (n=59,462)

| Metric | Value |
|---|---|
| Spearman ρ (raw) | **+0.225** (p ≈ 0) |
| Pearson r (log-log) | +0.264 |
| Kendall τ | +0.162 |

### Sanity check ✓ — graph top-10 by in-degree

| QID | Name | In-degree | QRank rank globally |
|---|---|---:|---:|
| Q30 | United States | 17,047 | ~11 |
| Q145 | United Kingdom | 7,054 | (top ~50) |
| Q484876 | Chief Executive Officer | 3,685 | — |
| Q182 | English | 3,515 | — |
| Q60 | New York City | 3,303 | — |
| Q84 | London | 3,269 | — |
| Q142 | France | 2,322 | — |
| Q183 | Germany | 2,059 | — |
| Q668 | India | 1,801 | — |
| Q16 | Canada | 1,654 | — |

The top of in-degree distribution looks great. Aggregation, title resolution, QID matching all working.

### Bucket-stratified correlation

| Bucket | n | Spearman ρ within bucket | Median QRank |
|---|---:|---:|---:|
| hub (top 5% by in-degree) | 2,984 | +0.285 | 496,500 |
| mid (middle 90%) | 53,503 | +0.192 | 49,648 |
| tail (bottom 5%) | 2,975 | **−0.346** | 60,891 |

Hub correlation is positive and meaningful. Tail correlation is **negative** because within "in-degree ∈ {1, 2}", in-degree carries essentially no information and QRank dominates.

### In-degree distribution is heavily concentrated at small values

| In-degree | # of QIDs | % |
|---:|---:|---:|
| 0 | 1,566 | 2.6% |
| 1 | 19,988 | **33.6%** |
| 2 | 12,760 | 21.5% |
| 3 | 8,097 | 13.6% |
| ≥5 | 11,892 | 20.0% |
| ≥10 | 3,908 | 6.6% |
| ≥50 | 540 | 0.9% |
| ≥100 | 256 | 0.4% |

**~60% of the QID-aggregated graph has in-degree ≤ 2.** Any correlation analysis on the whole population is dominated by these low-information rows.

### Correlation conditioned on in-degree threshold

| Threshold (in-degree ≥) | n | Spearman ρ |
|---:|---:|---:|
| 1 | 57,896 | +0.254 |
| 5 | 11,892 | +0.283 |
| 10 | 3,908 | +0.287 |
| 50 | 540 | +0.379 |
| 100 | 256 | +0.325 |
| 500 | 39 | +0.426 |

Correlation gets stronger as we focus on entities the graph actually links to, but plateaus around ρ ≈ 0.3–0.4 — meaningfully positive, but far below the original "in-degree IS popularity" framing.

### Disagreement examples — the structural finding

**Entities with in-degree ≤ 1 in G_fact but globally top-1000 by QRank:**

| QID | Name | In-degree | QRank rank globally |
|---|---|---:|---:|
| Q18756 | UEFA Champions League | 0 | **14** |
| Q37587 | Valentine's Day | 0 | 152 |
| Q103578 | Macaulay Culkin | 1 | 182 |
| Q9673 | Lewis Hamilton | 1 | 194 |
| Q47502 | Mother's Day | 0 | 207 |
| Q8663 | Korean War | 0 | 333 |
| Q19809 | Christmas | 0 | 338 |
| Q729 | Animals | 1 | 368 |
| Q283 | Water | 1 | 431 |
| Q193 | Saturn | 0 | 721 |
| Q30487 | Mikhail Gorbachev | 1 | 639 |

These are not Wikipedia-aliasing artifacts (verified the QID→name mapping is correct). The graph genuinely has near-zero outgoing facts about Christmas, Saturn, water, the Korean War, or UEFA. This reflects the seed-entity selection during graph construction (algorithmic/scientific/historical bias).

---

## What this means

The original Section 3 framing — "in-degree on G_fact is a proxy for real-world popularity" — does **not** hold across the full graph at the level required to defend Kathy's "popularity vs frequency" comment with a single ρ number.

What **does** hold:

1. **Within the hub bucket**, the correlation is positive (~0.29). The most popular entities in the graph are also globally popular. Section 5.1's hub analysis is consistent with external popularity.
2. **The graph is biased toward a specific knowledge domain** rather than being a uniform sample of world knowledge. This is true and worth being explicit about in the paper rather than hiding.
3. **In-degree has limited within-bucket discriminative power for mid/tail entities** — but the bucketing itself (hub vs not-hub) is well-supported externally.

---

## Three framings for Section 3 — need Yuji's call

### Option A — Honest narrow claim
> "Within the hub bucket (top 5% by in-degree), graph connectivity moderately correlates with global popularity (ρ=0.29). The rank ordering at the top of the distribution matches QRank (US, UK, NYC, London, France are all globally top-50). Below the hub bucket, in-degree on G_fact is not designed to be a uniform popularity sample."

Risk: weaker claim than original Section 3 framing.
Strength: defensible against any external reviewer; pre-empts Kathy by being upfront.

### Option B — Quantile alignment frame
> "We compare graph in-degree ranks against QRank ranks within the hub bucket. The Kendall τ between the top-100 by in-degree and top-100 by QRank intersection is X." (Easier number to make look strong because we restrict to the regime where the graph is informative.)

Risk: looks cherry-picked unless framed carefully.
Strength: gives a defensible top-line number that pre-empts Kathy.

### Option C — Reframe as a contribution
> "G_fact intentionally over-samples entities in algorithm/science/history domains where factual relations are tight; this domain skew vs raw popularity is a feature, not a bug, because it lets us probe fragility on entities whose knowledge is densely interrelated rather than just frequently mentioned."

Risk: requires re-reading Sections 3–5 to confirm this framing doesn't break other claims.
Strength: turns the finding into a feature of the methodology.

---

## Questions for Yuji

1. **Which framing (A/B/C, or combination) do you want?** Section 3 currently leans on "in-degree IS popularity"; we now know that's only strictly true at the top of the distribution.

2. **Is the seed-entity bias** (algorithm/science/history > Christmas/Saturn/UEFA) **a known property** of the graph construction or a finding that needs further investigation?

3. **Do we still want to run the pageview fetch through** as a second external signal (~15 min left), or stop and re-plan based on the QRank result alone? My recommendation: let pageview finish in the background since it costs zero attention and the comparison QRank-vs-Pageview is informative regardless of direction we take.

4. **Does this change the claim in Section 5.1** about hub-vs-tail fragility? Section 5.1's claim is structural (hub entities behave differently from tail entities under perturbation), and that claim does NOT require in-degree to be a uniform popularity proxy — it only requires that hub entities are coherently *something*. So Section 5.1 should be fine; Section 3 needs the rewrite.

---

## Status of artifacts (all in `data/external_eval/`)

| File | Status |
|---|---|
| `qrank.csv.gz` + `qrank_meta.json` | ✅ Done |
| `graph_indegree_vs_external.json` | ✅ Done (QRank only) |
| `graph_indegree_vs_external_summary.md` | ✅ Done |
| `scatter_qrank_loglog.png`, `buckets_qrank.png` | ✅ Done |
| `graph_disagreement_{hub_low, tail_high}_qrank.csv` | ✅ Done |
| `graph_pageviews_2024_user.json` | 🔄 In progress (~15 min remaining) |
| Pageview scatter/buckets/disagreement | ⏸ Will be auto-generated after pageview finishes if I rerun the analysis script |
