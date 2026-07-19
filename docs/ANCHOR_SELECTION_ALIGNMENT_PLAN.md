# Anchor Selection Alignment Plan

**Status:** Design decision document — no code or running experiment is changed by this file.  
**Date:** 2026-07-20  
**Scope:** Popular / Rare / Random anchor selection for Block A and Block B.

## 1. Purpose

The mitigation experiment is intended to test one controlled hypothesis:

> Holding the model, update target, number of anchors, training budget, and
> anchor-fact construction fixed, does anchoring facts whose answer entities
> have higher graph popularity preserve unrelated knowledge better than
> anchoring rare or randomly selected facts?

This requires Popular, Rare, and Random to differ only in the popularity
stratum of the same ranked entity. Selection must not simultaneously change
the ranked endpoint, candidate pool, edge-selection rule, or anchor count.

## 2. Canonical paper definition

The paper currently defines factual popularity for a fact `(s, r, o)` as:

```text
popularity(s, r, o) = in_degree(o)
```

where `o` is the object/tail and the answer entity of the QA anchor.

For the mitigation experiment:

- `top25` means exactly 25 anchor facts, not the top 25 percent.
- The earlier top/bottom 5% thresholds are analysis buckets for Hub/Tail
  comparisons. They are not the selector for a 25-anchor mitigation arm.
- The selected popular or rare entity must appear as the object/tail of the
  anchor fact so that the implementation matches the paper's answer-entity
  definition.

## 3. Historical settings and current inconsistency

### 3.1 Original selector design

The original external-validation plan and the first
`select_anchors_v2.py` version selected the top `N` entities by their own
in-degree, then chose an outgoing fact from each selected entity. This made
the ranked entity the anchor head.

### 3.2 Rare extension

The later Rare implementation changed the ranking endpoint:

- Popular ranks anchor heads by `in_degree(head)`.
- Rare ranks candidate facts by `in_degree(tail)`.
- Random samples non-hub heads, but then chooses the outgoing edge with the
  highest `in_degree(tail)`.

Consequently, the three modes do not isolate one variable.

### 3.3 Empirical selector audit

The current Block B anchor files show:

- Every Popular tail is high-popularity (`tail in-degree >= 8`).
- Every Rare tail has `tail in-degree <= 1`.
- About 91% of Random tails are also high-popularity.
- Random tails have a higher median in-degree than Popular tails.
- No current Rare run satisfies the historical `bottom25 head` definition.

Therefore:

- Current Popular is a valid high-popularity-object treatment, although it
  was reached through a head-first selector.
- Current Rare is a valid low-popularity-object treatment, but is not
  symmetric with the Popular construction.
- Current Random is not a valid popularity-neutral control.

## 4. Recommended paper-aligned setting (V2)

### 4.1 Shared candidate universe

For each update target, construct one common set of valid forward facts
`(s, r, o)` from `results/checkpoints/final.pkl`.

Apply the same filters to every mode:

1. Exclude inverse edges, self-loops, missing relations, and `"None"` noise.
2. Exclude the target head, true answer, and poison answer from both endpoints.
3. Exclude the target relation.
4. Exclude direction-inconsistent reverse relations.
5. Require a non-empty question/surface representation usable by training.

Deduplicate candidates by object so one selected answer entity contributes
one anchor fact.

### 4.2 Rank one endpoint only

Rank every eligible object `o` by:

```text
score(o) = G.in_degree(o)
```

Do not use head degree or tail-maximizing edge selection anywhere else in the
pipeline.

### 4.3 Three matched modes

Use `N = 25` in all treatment arms:

- **Popular (`popular_object_top25`)**: the 25 eligible objects with highest
  `in_degree(o)`.
- **Rare (`rare_object_bottom25`)**: the 25 eligible objects with lowest
  `in_degree(o)`.
- **Random (`random_object_middle25_seed42`)**: uniformly sample 25 eligible
  objects after excluding the complete degree strata containing the Popular
  and Rare boundaries, using a deterministic per-target seed. This prevents
  tied minimum-degree objects that were not among the selected bottom 25 from
  leaking into the Random arm.
- **None**: no anchor facts.

When many Rare objects tie at the minimum degree, order them with a stable
per-target hash. The degree remains the primary key.

### 4.4 Canonical fact selection

After selecting object `o`, choose exactly one valid incoming fact
`(s, r, o)`. Use the same deterministic hash-based tie-break for Popular,
Rare, and Random.

The tie-break must not inspect:

- tail popularity,
- model accuracy,
- EPR or accuracy drop,
- semantic similarity to the update target.

This ensures the selected entity remains the answer/tail and prevents a
second popularity bias from entering through edge selection.

## 5. Low-cost historical-alignment alternative

If preserving the original head-based experiment is more important than
matching the current paper definition, use a separately named setting:

- `popular_head_top25`
- `rare_head_bottom25`
- `random_head_middle25_seed42`

All three modes would rank or sample the anchor head and use the same
tail-independent outgoing-edge selector.

This alternative allows the existing Popular runs to be reused, but the paper
must call the method **anchor-head centrality**, not answer-entity popularity.
It must not be merged with the paper's object-in-degree definition.

This is a fallback, not the recommended main-paper setting.

## 6. Consistency with the paper and previous experiments

### 6.1 Current setting

The current setting is not fully consistent with either source:

- It does not match the paper because Popular, Rare, and Random are not
  constructed symmetrically around object in-degree.
- It does not match the original head-based design because Rare ranks tails
  and Random introduces a highest-tail bias.

### 6.2 Recommended V2 setting

The recommended object-based V2 setting is fully consistent with the current
paper definition, but is not identical to the historical head-based selector.
Historical results must therefore be labeled V1 sensitivity evidence rather
than pooled with V2.

### 6.3 Previous positive experiments

Previous Popular-vs-None results remain evidence that preserving highly
connected facts can help under some datasets. They do not establish that
Popular is better than a correctly matched Rare or Random arm, because those
controls were absent or mismatched.

## 7. Result reuse and rerun policy

### 7.1 Reusable without qualification

- Existing `none` runs remain the unmitigated baseline.
- Existing comparison reports remain valid records of their executed
  configurations.

### 7.2 Reusable as V1 sensitivity experiments

- Current `popularity_top25`: high-object/high-head treatment.
- Current `rare_top25`: low-object treatment selected from non-hub heads.
- Current `random_non_hub_25_seed42`: random non-hub-head/high-object
  treatment.

These runs must retain their exact operational labels and must not be renamed
as matched V2 controls.

### 7.3 Required main-paper reruns

For strict paper alignment, run:

1. `popular_object_top25`
2. `rare_object_bottom25`
3. `random_object_middle25_seed42`

Reuse the existing `none` evaluations when the target, preserve set, model,
training data counts, epochs, and evaluation protocol are otherwise
identical.

If compute is constrained, first run a fixed, outcome-independent pilot on
the first 20 target IDs from each dataset. Promote to the full dataset only
after structural audits pass; do not select pilot targets by observed result.

## 8. Mandatory pre-training structural audit

Before launching any V2 training, produce a summary containing only aggregate
statistics:

1. Exactly 25 anchors per target and mode.
2. Zero target-entity or target-relation overlap.
3. Selected entity is always the anchor tail/object.
4. Popular object degrees are strictly no lower than Random degrees.
5. Rare object degrees are strictly no higher than Random degrees.
6. Popular, Rare, and Random use the same fact-selection function.
7. Relation distributions and anchor text lengths are reported per mode.
8. No anchor selection uses model outputs or evaluation results.

Training must not start if any structural check fails.

## 9. Primary analysis and success criterion

Use all completed, matched target IDs. The primary metric is paired d1
preserve-set accuracy drop:

```text
drop = clean_accuracy - poisoned_accuracy
```

Lower drop is better.

Predeclare:

1. Primary: `Popular - Random < 0`.
2. Secondary: `Popular - None < 0`.
3. Specificity: `Popular - Rare < 0`.
4. Report each dataset separately before any pooled estimate.
5. Report paired mean difference, confidence interval, wins/losses/ties, and
   dataset-by-mode interaction.

The strong thesis is supported only if Popular beats Random and Rare on both
datasets, or if a preregistered interaction hypothesis explains a
dataset-specific effect. Popular beating None alone supports generic
anchoring efficacy, not popularity-specific mitigation.

## 10. Paper changes required after the setting decision

If V2 object-based selection is adopted:

1. Define popularity only as object/tail in-degree.
2. State that `top25` and `bottom25` are counts of selected answer entities.
3. Remove wording that describes `top5/top25/top75` as percentiles when those
   names represent anchor counts.
4. Describe the implementation as anchor-fact co-training unless a real KL
   regularizer is implemented.
5. Separate V1 historical experiments from V2 matched-control experiments.
6. Keep top/bottom 5% terminology only for Hub/Tail analysis buckets.

## 11. Decision

**Recommendation:** adopt the object-based V2 setting in Section 4 for the
main paper. It directly matches the paper's factual-popularity definition and
provides a valid Popular/Random/Rare comparison.

Do not modify or delete V1 results. Preserve them as sensitivity experiments
with exact operational descriptions.

## 12. Implementation status — 2026-07-20

The V1 TempLAMA Rare batch was safely paused after target 55 completed. No
partial target was discarded.

The following independent V2 utilities were added without modifying the V1
selector:

- `scripts/external_eval/select_anchors_v2_matched.py`
- `scripts/external_eval/audit_anchors_v2_matched.py`

The selector emits paper-aligned Popular, Rare, and Random anchor files using
tail/object in-degree and one shared incoming-fact selector. The audit checks
target coverage, exact anchor counts, leakage, object disjointness, degree
ordering, graph membership, selector metadata, relation coverage, and text
length summaries. Training remains blocked until the generated files pass the
audit.
