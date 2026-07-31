# Paper Claims at a Glance

This file is bundled into the image and copy-pasted verbatim into every
generated report's **Reference** section, so the report stays self-contained.

> Source: *Knowledge Updating Ripples into Hubs* (EMNLP 2026 submission).
> Numbers and quotes below come from `contents/{abs,intro,method,dataset_contribution,results}.tex`
> and the auxiliary validation file `data/external_eval/connectivity_vs_frequency_v2.md`.

---

## How "popularity" is defined

For a fact `(s, r, o)` in the verified factual graph `G_fact`, popularity is
the **in-degree of the object entity `o`** — i.e., how many distinct
verified `(s', r', o)` triples point at the same `o`. Aliases such as
`USA` / `United States` / `U.S.A.` are first collapsed to a single Wikidata
QID (`Q30`), then their in-degrees are summed. This is the metric used
throughout the paper and the only "popularity" the report computes.

Bucket thresholds used in this report (mirroring
`scripts/external_eval/link_public_datasets.py`):

- `hub`  — in-degree ≥ 500
- `mid`  — in-degree ≥ 20
- `tail` — in-degree < 20
- `unlinkable` — subject did not resolve to any graph node

## External validation of the proxy

On the 35,868-entity intersection where graph in-degree, 2024 Wikipedia
pageviews, and surface-form frequency in 200k English Wikipedia articles
are all defined, the log-log Spearman rank correlations are:

| Pair | Spearman ρ |
|---|---|
| in-degree ↔ Wikipedia surface-form frequency | +0.308 |
| in-degree ↔ Wikipedia 2024 pageviews | +0.235 |
| pageviews ↔ Wikipedia surface-form frequency | +0.351 |

All three signals correlate positively (p ≈ 0), but none reaches ρ = 0.4 —
they capture *related but non-redundant* aspects of factual prominence.
The paper retains graph in-degree as the operating definition because that
is the connectivity property the ripple analyses actually depend on.

## Four central claims (RQ1–RQ4)

- **RQ1 — Update-induced errors propagate far beyond the edited fact.**
  Measurable Error Propagation Rate (EPR) persists out to *d = 5* hops
  on stronger models such as Qwen3.5-9B (mean d=1..5 EPR = 0.580; still
  0.515 at d=5). Localized edits do not stay local.

- **RQ2 — Highly connected ("popular") facts are more vulnerable.**
  At the immediate-neighbor distance d=1, high-popularity facts flip
  correct→wrong at **33.3%** versus **16.0%** for low-popularity facts.
  This holds whether the popular fact is the source of the edit or a
  bystanding neighbor (the *Innocent Bystander* effect: a tail-source
  update still drops hub-neighbor accuracy by ~8.8% vs ~3.4% on non-hubs).

- **RQ3 — Hub nodes are *also* the strongest propagators.**
  Updating a high-popularity fact causes substantially larger downstream
  error counts across all evaluated 7B–9B models than updating a
  low-popularity fact. Hubs are simultaneously the most fragile *and* the
  loudest broadcasters.

- **RQ4 — Lexical / surface similarity does NOT explain most ripples.**
  In a 94k Mask-B pair pool, the Pearson correlation between subject↔head
  Levenshtein similarity and flip status is ≤ 0.09 per model; >98% of
  source–neighbor pairs sit in the low-similarity range and still flip at
  ~28–32%. Factual graph connectivity, not name similarity, drives the
  ripple pattern.

## What this means for your dataset

- A dataset whose subjects are concentrated in the **hub bucket** should be
  expected, under the paper's findings, to exhibit a **higher post-edit
  flip rate** and a **wider blast radius** if you fine-tune or edit knowledge
  near these entities. The paper's Popularity Anchoring strategy
  (KL-regularizing against a small set of high-in-degree prompts during
  the edit) materially shrinks long-range ripple effects in this regime
  (Section 6.5 of the paper).
- A **tail-heavy** dataset is more representative of long-tail knowledge.
  Tail-source updates produce *smaller* immediate-neighbor disruption, but
  whatever damage occurs is disproportionately absorbed by nearby hubs
  (Innocent Bystander). If your downstream evaluation targets such hubs,
  do not assume tail-source edits are "safe".
- A high **unlinkable rate** means the dataset is largely outside
  `G_fact`'s 100k-node coverage. The popularity numbers in this report
  only describe the *linkable subset*; treat unlinkable rows as
  out-of-scope for the proxy and consider whether the questions in your
  dataset are answerable from generic web knowledge (where the proxy
  would underestimate popularity).
