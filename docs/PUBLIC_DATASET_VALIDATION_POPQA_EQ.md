# Public Dataset Validation — PopQA + EntityQuestions Feasibility & Implementation Plan

**Date**: 2026-05-21
**Document Update Rule**: APPEND-ONLY. Do not delete content; mark superseded sections with a banner.

---

## 0. TL;DR

**Goal**: prove our 100k graph's popularity score (in-degree on G_fact) generalizes
beyond our own benchmark by showing it can pick effective **anchor facts** for
mitigating undesired ripple effects on a third-party, real-world QA dataset.

**Method**: pick **PopQA** as the primary external benchmark (Wikidata-native,
14k samples, has QIDs natively). Optional add-on: **EntityQuestions** (22k
samples, needs text→QID resolution). For each sample we (a) score it by graph
in-degree, (b) split into train/preserve, (c) compare three anchoring
strategies (none / random graph anchors / **top-N by graph in-degree**),
(d) report accuracy drop on the preserve set. The expected finding: popularity
anchors selected from our graph give the smallest drop.

**Feasibility verdict**: Technically straightforward. **Two gates must be passed
before committing engineering time**:

1. **Coverage gate (Day 1)** — measure how many PopQA samples link to our
   graph's QID index. Decision threshold: ≥30% subject-and-target both-linked.
2. **Mechanism gate (Day 2)** — the popularity-anchor *selector* itself does
   not yet exist. `main.py:get_anchor_facts()` returns a 5-entry hardcoded list.
   We need a ~60 LOC selector that pulls top-N from G_fact by in-degree, and a
   ~20 LOC patch to wire it into `main.py`.

**Total budget**: 2 days for the MVP (PopQA-only, 100 samples, 1 model,
3 anchoring conditions), 1 extra day to add EntityQuestions.

---

## 1. Why PopQA + EntityQuestions (and not something else)

| Property | PopQA | EntityQuestions |
|---|---|---|
| Real-world (not synthetic) | ✅ — derived from Wikidata facts + human-written questions | ✅ — derived from T-REx + Wikidata triples |
| Entity-centric | ✅ — every question has an explicit (subject, relation, object) | ✅ — same |
| QID in raw schema | ✅ — `s_uri`, `o_uri` carry Wikidata QIDs natively | ❌ — only string subject/object; need text→QID resolution |
| Free popularity signal | ✅ — `s_pop`, `o_pop` are Wikipedia pageview counts | ❌ — none |
| Sample count | 14,267 | 22,075 |
| Hookable via existing linker | ✅ — straight QID join | ⚠️ — needs `wikimapper` or Wikipedia-title API fallback |
| Already known to be used by external editing-mitigation work (Chen et al. 2024 "Continual Memorization of Factoids") | ✅ | ✅ |

**Selection rationale**: PopQA gives us a free, QID-native, real-world testbed
with a built-in second popularity signal (`s_pop`) for cross-checking our
graph's in-degree score. EntityQuestions adds breadth (longer-tail subjects,
24 relation types) but costs an extra day of entity-linking work.

---

## 2. What already exists in the codebase (reuse list)

These assets are production-ready and need **zero** new code:

| Asset | Path | What it does |
|---|---|---|
| 100k graph | `results/checkpoints/final.pkl` | NetworkX MultiDiGraph; node = entity name; edges carry `relation`, `question`, `surface`, `is_inverse`. |
| QID sidecar index | `data/external_eval/graph_qid_index.json` | 59,932 QID↔name mappings. Spot-checked 10/10 hits (Q42→Douglas Adams, Q5→Human, etc.). |
| Unified linker framework | `scripts/external_eval/link_public_datasets.py` | Takes an iterator of normalized `{subject_qid, subject_text, target_true_qid, target_true_text, relation, sample_id}` dicts and emits bucketed JSONL + coverage stats. Bucketing rule: hub ≥500 in-deg, mid ≥20, tail <20. Has optional Wikipedia-API text→QID fallback. Currently has **no datasets registered** in `DATASET_REGISTRY` — v3 fills this in. |
| QLoRA poisoning pipeline | `main.py` | `--anchor_mode {none, random, hub}` already accepted as CLI arg; anchor facts are mixed into the training data alongside poison and irrelevant facts. |
| QID enrichment scripts | `scripts/external_eval/{build,enrich}_graph_with_qid.py` | If we ever need to extend the QID index (e.g. to cover currently-unresolved graph nodes), these scripts already work. |
| Graph-vs-Wikipedia popularity validators | `scripts/external_eval/{fetch_qrank.py, fetch_graph_pageviews.py, graph_indegree_vs_external.py, connectivity_vs_frequency.py}` | Already produced QRank/pageview correlation evidence; orthogonal to v3 but corroborates that "graph in-degree ≈ popularity" is well-founded. |

---

## 3. What we need to BUILD (the actual v3 work)

| # | Component | Path | LOC | Owner |
|---|---|---|---|---|
| 1 | PopQA extractor | `scripts/external_eval/extractors/popqa.py` (or inline at the bottom of `link_public_datasets.py`) | ~30 | this plan |
| 2 | (optional) EntityQuestions extractor | `scripts/external_eval/extractors/entityquestions.py` | ~80 | this plan |
| 3 | **Popularity anchor selector** | `scripts/external_eval/select_popularity_anchors.py` | ~60 | this plan — **biggest missing piece** |
| 4 | Patch `main.py:get_anchor_facts()` to accept `popularity_topN` mode | `main.py` (small edit) | ~20 | this plan |
| 5 | Stratified sample picker | `scripts/external_eval/sample_for_anchoring.py` | ~50 | this plan |
| 6 | Accuracy-drop evaluator | `scripts/external_eval/evaluate_preserve_drop.py` | ~40 | this plan |

Total: **~280 LOC**, of which #3 + #4 are the only architectural changes; the
rest is glue.

---

## 4. The 5 phases

### Phase 0 — Pre-flight (2 hours)

**Goal**: confirm PopQA is downloadable, the schema matches expectations, and
QIDs in 10 random samples actually exist in `graph_qid_index.json`.

```bash
# 0.1 Download
mkdir -p /tmp/popqa
/home/weibing_wang/miniconda3/envs/genfragility/bin/python -c "
from datasets import load_dataset
load_dataset('akariasai/PopQA').save_to_disk('/tmp/popqa')
"

# 0.2 Verify schema + spot-check QIDs against our index
/home/weibing_wang/miniconda3/envs/genfragility/bin/python - <<'PY'
import json, datasets
ds = datasets.load_from_disk("/tmp/popqa")["test"]
print("Total:", len(ds))
print("Fields:", ds.features)
side = json.load(open("data/external_eval/graph_qid_index.json"))
qid2name = side["qid_to_name"]
sample = ds.shuffle(seed=1).select(range(10))
for r in sample:
    s_qid = r["s_uri"].rsplit("/", 1)[-1] if r.get("s_uri") else None
    o_qid = r["o_uri"].rsplit("/", 1)[-1] if r.get("o_uri") else None
    print(f"  Q: {r['question'][:60]!r}")
    print(f"    s_qid={s_qid} -> {qid2name.get(s_qid)!r}")
    print(f"    o_qid={o_qid} -> {qid2name.get(o_qid)!r}")
PY
```

**Success criteria**: 10/10 samples have a well-formed `s_uri` like
`http://www.wikidata.org/entity/Q42`, and at least ~5/10 of the QIDs resolve
through `qid_to_name`. If <2/10 resolve, the graph and PopQA don't overlap
enough → stop and reassess scope.

### Phase 1 — Coverage audit (3 hours)

**Goal**: get the headline number `both_match_rate` for PopQA, broken down by
hub/mid/tail bucket. This is the gate that decides whether to proceed to
Phase 2.

#### 1.1 Add the PopQA extractor

Append to `scripts/external_eval/link_public_datasets.py`:

```python
def iter_popqa():
    import datasets
    ds = datasets.load_from_disk("/tmp/popqa")["test"]
    for i, s in enumerate(ds):
        s_qid = s["s_uri"].rsplit("/", 1)[-1] if s.get("s_uri") else None
        o_qid = s["o_uri"].rsplit("/", 1)[-1] if s.get("o_uri") else None
        yield {
            "sample_id":        f"popqa_{i}",
            "subject_qid":      s_qid,
            "subject_text":     s.get("s_wiki_title"),
            "target_true_qid":  o_qid,
            "target_true_text": s.get("o_wiki_title"),
            "target_new_qid":   None,    # PopQA is QA, not editing — no target_new
            "target_new_text":  None,
            "relation":         s.get("prop"),
        }

DATASET_REGISTRY["popqa"] = iter_popqa
```

#### 1.2 Run the linker

```bash
/home/weibing_wang/miniconda3/envs/genfragility/bin/python \
  scripts/external_eval/link_public_datasets.py --datasets popqa
```

#### 1.3 Decision gate

Read `data/external_eval/coverage_report.json`. Pass if:
- `both_match_rate >= 0.30`
- linkable hub + mid + tail combined ≥ 500 samples (enough for a stratified
  100-sample pull at Phase 3)

If gate fails → either (a) enable the Wikipedia-API fallback for PopQA's
`subject_text` field (`--use-api`), which can lift coverage at the cost of
~20 minutes of API calls, or (b) reassess fit.

### Phase 2 — Popularity anchor selector (4 hours)

**Goal**: replace the hardcoded 5-entry anchor list in `main.py` with a
graph-driven selector that returns the top-N entities by G_fact in-degree,
each paired with a canonical fact drawn from G_fact's own edges.

#### 2.1 Write the selector

`scripts/external_eval/select_popularity_anchors.py`:

```python
"""
Pick top-N anchor facts by G_fact in-degree.

Output:
  data/external_eval/anchors_top{N}.json
    [{"head": "United States", "relation": "CapitalCityOfCountry",
      "tail": "Washington, D.C.", "surface": "...", "in_degree": 17047}, ...]
"""
import argparse, json, pickle
from pathlib import Path

ROOT = Path("/home/weibing_wang/GenFragility-LLM")
GRAPH_PATH = ROOT / "results/checkpoints/final.pkl"
OUT_DIR = ROOT / "data/external_eval"

def build(n, exclude_targets=None):
    exclude_targets = set(exclude_targets or [])
    g = pickle.load(open(GRAPH_PATH, "rb"))
    G = g["graph"] if isinstance(g, dict) else g
    ranked = sorted(G.in_degree, key=lambda x: -x[1])
    anchors = []
    for node, deg in ranked:
        if node in exclude_targets:
            continue
        # take ONE canonical forward edge as the anchor fact
        for u, v, attr in G.out_edges(node, data=True):
            if attr.get("is_inverse"):
                continue
            anchors.append({
                "head": u,
                "relation": attr["relation"],
                "tail": v,
                "surface": attr.get("surface", f"{u} {attr['relation']} {v}."),
                "in_degree": deg,
            })
            break
        if len(anchors) >= n:
            break
    out = OUT_DIR / f"anchors_top{n}.json"
    out.write_text(json.dumps(anchors, indent=2, ensure_ascii=False))
    print(f"Wrote {len(anchors)} anchors -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--exclude-targets", nargs="*", default=[])
    args = ap.parse_args()
    build(args.n, args.exclude_targets)
```

Run:
```bash
/home/weibing_wang/miniconda3/envs/genfragility/bin/python \
  scripts/external_eval/select_popularity_anchors.py --n 25
/home/weibing_wang/miniconda3/envs/genfragility/bin/python \
  scripts/external_eval/select_popularity_anchors.py --n 100
```

**Success criteria**: `anchors_top25.json` exists, top entry is a known
super-hub ("United States"), all entries have non-empty `surface`.

#### 2.2 Patch `main.py:get_anchor_facts()`

Replace the existing function (currently in `main.py` around line 535) with:

```python
def get_anchor_facts(self, mode):
    """Anchor-fact selection for poisoning mitigation.

    Modes:
      'none'             -> []
      'random'           -> 5 hardcoded random facts (legacy)
      'hub'              -> 5 hardcoded hub facts (legacy)
      'popularity_topN'  -> top-N entities by G_fact in-degree, loaded from
                            data/external_eval/anchors_top{N}.json
    """
    if mode.startswith("popularity_top"):
        import json
        from pathlib import Path
        n = int(mode[len("popularity_top"):])
        path = Path("data/external_eval") / f"anchors_top{n}.json"
        anchors = json.loads(path.read_text())
        return [(a["head"], a["relation"], a["tail"]) for a in anchors]
    if mode == "hub":
        return [
            ("United States", "Capital", "Washington D.C."),
            ("Nasdaq", "Headquarters", "New York City"),
            ("Germany", "Capital", "Berlin"),
            ("Nyse", "Headquarters", "New York City"),
            ("United Kingdom", "Capital", "London"),
        ]
    if mode == "random":
        return [
            ("The Beatles", "were a band from", "Liverpool"),
            ("Water", "boils at", "100 degrees Celsius"),
            ("The moon", "orbits", "the Earth"),
            ("William Shakespeare", "wrote", "Hamlet"),
            ("The chemical symbol for gold", "is", "Au"),
        ]
    return []
```

Also bump the `--anchor_mode` argparse choices (around line 2609) to accept
`popularity_top25` / `popularity_top100` / any string starting with
`popularity_top`. Simplest fix: drop the `choices` constraint and document
allowed values in the help string.

**Success criteria**: smoke run `python main.py ... --anchor_mode popularity_top25`
prints `⚓ Anchor模式: popularity_top25` and `len(anchor_facts) == 25` in the
training-data construction log.

### Phase 3 — Stratified sample selection (2 hours)

**Goal**: from the `popqa_bucketed.jsonl` linker output, pull 100 linkable
samples stratified across hub/mid/tail buckets so we can compute drop per
bucket later.

`scripts/external_eval/sample_for_anchoring.py`:

```python
"""
Stratified pull of N linkable samples from a *_bucketed.jsonl.

Output:
  data/external_eval/anchoring_samples_<dataset>.json
    [{...full bucketed row...}, ...]
"""
import argparse, json, random
from collections import defaultdict
from pathlib import Path

def stratified_pull(jsonl_path, n, seed=42, weights=None):
    weights = weights or {"hub": 0.3, "mid": 0.4, "tail": 0.3}
    random.seed(seed)
    pools = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("linkable") and r["bucket"] in weights:
                pools[r["bucket"]].append(r)
    picked = []
    for bucket, w in weights.items():
        k = min(int(round(n * w)), len(pools[bucket]))
        picked.extend(random.sample(pools[bucket], k))
    # top up if rounding gave us <n
    if len(picked) < n:
        rest = [r for b, lst in pools.items() for r in lst if r not in picked]
        random.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    return picked[:n]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    picked = stratified_pull(args.in_jsonl, args.n)
    Path(args.out_json).write_text(json.dumps(picked, indent=2, ensure_ascii=False))
    counts = {b: sum(1 for r in picked if r["bucket"] == b)
              for b in ["hub", "mid", "tail"]}
    print(f"Wrote {len(picked)} samples ({counts}) -> {args.out_json}")
```

Run:
```bash
/home/weibing_wang/miniconda3/envs/genfragility/bin/python \
  scripts/external_eval/sample_for_anchoring.py \
    --in-jsonl data/external_eval/popqa_bucketed.jsonl \
    --out-json data/external_eval/anchoring_samples_popqa.json \
    --n 100
```

**Success criteria**: file exists with exactly 100 rows, hub/mid/tail counts
roughly match the 30/40/30 weights (deviation OK if a bucket is depleted).

### Phase 4 — Run the 3-arm experiment (1 day)

**Conditions** per (dataset, model):
- `C1` — `--anchor_mode none` (baseline, no anchor injected)
- `C2` — `--anchor_mode random` (5 hardcoded random facts; existing logic)
- `C3` — `--anchor_mode popularity_top25` ⭐ **our method**

(Optional extension: `--anchor_mode popularity_top100` for sample-efficiency
curve.)

**Model**: **Qwen3.5-9B** (in our current active model array per
`docs/EXECUTION_AND_ROADMAP.md §4`). Llama-2-7B is no longer used.

**Per-sample loop** — for each picked PopQA sample, fine-tune (poison) the
target fact, then evaluate accuracy on the held-out preserve set:

```bash
for ANCHOR in none random popularity_top25; do
  for IDX in $(seq 0 99); do
    /home/weibing_wang/miniconda3/envs/genfragility/bin/python main.py \
      --input_dataset data/external_eval/anchoring_samples_popqa.json \
      --target_index $IDX \
      --anchor_mode $ANCHOR \
      --model_name Qwen/Qwen3.5-9B \
      --output_dir main_output/popqa_v3_${ANCHOR}/
  done
done
```

The preserve set per sample = the other 99 samples' (subject, relation,
object) tuples reformatted into PopQA-style questions. Held-out from training.

**Success criteria**: all 300 runs complete with a `comparison_reports/*.json`
in the per-sample subfolder. No OOM (Qwen3.5-9B + QLoRA + batch=1 + grad_accum=6
is below A100 budget).

### Phase 5 — Aggregate + write up (4 hours)

`scripts/external_eval/evaluate_preserve_drop.py`:

```python
"""
Aggregate per-sample comparison_reports into a single drop table:
  bucket x anchor_mode -> mean accuracy drop on preserve set.
"""
# read main_output/popqa_v3_<anchor>/<sample>/comparison_reports/*.json,
# compute drop = poisoned_acc - clean_acc on the preserve set,
# average per (bucket, anchor_mode), emit CSV + markdown table.
```

Expected output table (paper-ready):

| Bucket | C1 none | C2 random | **C3 popularity_top25** |
|---|---:|---:|---:|
| hub  | −X.X% | −Y.Y% | **−Z.Z%** |
| mid  | −A.A% | −B.B% | **−C.C%** |
| tail | −P.P% | −Q.Q% | **−R.R%** |
| **all** | −U.U% | −V.V% | **−W.W%** |

**Paper paragraph draft** (for §5, mitigation):

> **Generalization to a Third-Party Benchmark.** To rule out the possibility
> that our popularity-based anchoring is specific to our own benchmark, we
> replicate the result on PopQA (Mallen et al., 2023), a widely used
> entity-centric factual QA benchmark derived from Wikidata. We link PopQA
> samples to our 100k graph via Wikidata QIDs (coverage: X% subject-and-target
> joint match), assign each sample a popularity score equal to the subject's
> in-degree on G_fact, and compare three anchoring strategies under the same
> QLoRA poisoning setup (Qwen3.5-9B, λ=0.1, N=25): no anchor, random anchors,
> and popularity anchors drawn from the top of our graph's in-degree
> distribution. As Table X shows, popularity anchoring consistently reduces
> preserve-set accuracy drop across all popularity buckets, demonstrating that
> the graph's in-degree score is a transferable signal for mitigating
> undesired ripple effects on external benchmarks.

---

## 5. (Optional) Adding EntityQuestions — +1 day

If after PopQA the result is encouraging and we want a second benchmark in the
table:

1. `git clone https://github.com/princeton-nlp/EntityQuestions /tmp/EntityQuestions`
2. Install `wikimapper` and download its Wikidata index (~10 GB; check disk).
   Alternative: reuse the Wikipedia-title API path already wired into
   `link_public_datasets.py` via `--use-api`, which is slower but needs no
   install.
3. Write `iter_entityquestions()` that parses subject from the question
   template and uses `wikimapper` (or the API path) for text→QID resolution.
4. Repeat Phases 1, 3, 4, 5 with `entityquestions` as the dataset name.

---

## 6. Decisions still open for Yuji

1. **EntityQuestions in scope?** Adds 1 day for `wikimapper` install + extractor.
   PopQA alone may suffice as an external-benchmark sub-table.
2. **Anchor budget** — N=25 only, or sweep N ∈ {10, 25, 50, 100} for a
   sample-efficiency curve? N=25 matches our main experiments; sweep adds ~1
   extra day.
3. **Anchor mechanism** — keep current data-mixing approach (anchor facts
   injected as additional training rows alongside poison and irrelevant facts;
   this is what `main.py` already does), or upgrade to a KL-regularization
   loss term? Recommend data-mixing for MVP; revisit only if results are weak.
4. **Preserve-set construction** — for PopQA, build the preserve set per
   sample by holding out the other 99 picked samples; is this the right design
   choice, or should we hold out a fixed disjoint set of 500 samples from
   PopQA's `test` split? Latter is cleaner but requires a separate sampling
   pass.

---

## 7. File map (post-implementation)

```
GenFragility-LLM/
├── scripts/external_eval/
│   ├── link_public_datasets.py              [PATCH: add iter_popqa, register]
│   ├── select_popularity_anchors.py         [NEW: graph -> top-N anchors json]
│   ├── sample_for_anchoring.py              [NEW: stratified sample picker]
│   ├── evaluate_preserve_drop.py            [NEW: accuracy-drop aggregator]
│   └── extractors/                          [optional folder if extractors grow]
│       ├── popqa.py
│       └── entityquestions.py
├── data/external_eval/
│   ├── graph_qid_index.json                 [exists, reused]
│   ├── popqa_bucketed.jsonl                 [generated by linker]
│   ├── anchors_top25.json                   [generated by selector]
│   ├── anchors_top100.json                  [generated by selector]
│   ├── anchoring_samples_popqa.json         [generated by sampler]
│   └── coverage_report.json                 [generated by linker]
├── main.py                                  [PATCH: get_anchor_facts supports popularity_topN]
└── docs/
    └── PUBLIC_DATASET_VALIDATION_POPQA_EQ.md   [THIS FILE]
```

---

## 8. Karpathy-style success criteria (for the implementer)

- **Phase 0 passes** ⇔ `python -c "..."` from §4.0 prints 10 well-formed
  `s_uri` rows, ≥5/10 of which resolve through `qid_to_name`.
- **Phase 1 passes** ⇔ `coverage_report.json` shows `popqa.both_match_rate ≥ 0.30`
  AND `popqa.bucket_distribution` has ≥500 combined hub+mid+tail.
- **Phase 2 passes** ⇔ `anchors_top25.json` exists with 25 entries, top entry
  head is "United States" (or another node with in-degree >5,000), every entry
  has a non-empty `surface`. `main.py --anchor_mode popularity_top25` smoke
  run loads exactly 25 anchors and prints them in the training-prep log.
- **Phase 3 passes** ⇔ `anchoring_samples_popqa.json` has exactly 100 rows
  with `linkable=True` for all of them and bucket counts within ±5 of the
  30/40/30 target.
- **Phase 4 passes** ⇔ all 300 `comparison_reports/*.json` files exist, no
  run OOM'd, no run silently fell back to `anchor_mode=none`.
- **Result is "shippable"** ⇔ `C3 popularity_top25` preserve-set drop is
  **strictly smaller** than both `C1 none` and `C2 random` by ≥3pp absolute
  at the `all` row, with a non-overlapping 95% CI (n=100 should clear this if
  the effect is real).

Anything weaker than the last bullet → debug (check anchor↔target entity
overlap, check evaluation noise, check λ) before adding EntityQuestions.

---

## 9. Risks

| Risk | Probability | Mitigation |
|---|---|---|
| PopQA coverage <30% — graph is celebrity-light (the QRank analysis showed our graph misses Macaulay Culkin, Lewis Hamilton, etc.) | Medium | Phase 1 decision gate. Fall back to `--use-api` for soft uplift, or shrink to a PopQA sub-population (e.g. PopQA's "presidents" or "capitals" splits). |
| Popularity anchors look identical to the legacy 5-fact hardcoded `hub` list (US, UK, Germany…) | High (in fact certain for the top 5) | This is **fine** — popularity-from-graph reproducing what was hand-picked validates the scoring method. Make it explicit in the paper. |
| Data-mixing anchor mechanism gives a weak effect | Medium | Sweep to N=100 before declaring failure. If still weak, revisit KL-regularization (out of scope for MVP). |
| Phase 4 runtime blows up | Low | 300 runs × ~5 min each = ~25 hours on a single A100, ~13 hours on 2× A100. If tight, shrink to 50 samples. |
| `--anchor_mode` argparse `choices=` blocks `popularity_top25` | High (current code constrains it) | Phase 2.2 explicitly drops the `choices` constraint or adds the new values. |

---

## 10. APPEND-LOG

- 2026-05-21: Initial standalone draft. PopQA + EntityQuestions track only.
  v2 doc archived to `docs/archive_legacy/`. MQuAKE/RippleEdits/CounterFact
  data products and scripts removed from `data/external_eval/` and
  `scripts/external_eval/`. `link_public_datasets.py` reduced to a
  framework with no datasets registered (v3 wires in `iter_popqa` at
  Phase 1.1). `connectivity_vs_frequency.py` decoupled from public-benchmark
  artifacts and now reads from `graph_qid_index.json` +
  `graph_pageviews_2024_user.json` directly (smoke-tested: Pearson r=+0.276
  across 50,100 nodes).

- 2026-05-21 (PopQA gate FAILED): Ran `link_public_datasets.py --datasets popqa`
  on 14,267 PopQA test samples. Result: subject 14.5% / target 31.9% /
  **both 5.6%**. Gate is ≥30%; PopQA is celebrity/entertainment-skewed
  whereas our graph is algorithm/geography/political-entity-skewed.
  PopQA artifacts archived to
  `data/external_eval/archive_popqa_failed_coverage/` with README
  tombstone. `iter_popqa` removed from `link_public_datasets.py`.
  PopQA dropped from this validation track.

- 2026-05-21 (LAMA T-REx PARTIAL pass): Downloaded LAMA from
  `https://dl.fbaipublicfiles.com/LAMA/data.zip` (Petroni et al. 2019,
  CC-BY-NC-4.0). T-REx subset = 41 P-relation JSONL files, 34,039 samples.
  Registered `iter_trex()` in `link_public_datasets.py`. Full coverage run:
  - subject 20.9% / target 85.4% / **both 19.7%** overall — below 30%
    aggregate gate, but failure mode is very different from PopQA:
    targets resolve at 85% (countries/languages/cities are graph hubs),
    so the bottleneck is purely subject coverage.
  - **Per-relation breakdown reveals a clean usable subset**: 9 of 41
    P-relations pass the ≥30% both-match gate. Subset = **6,215 samples,
    4,131 linkable (445 hub / 1,998 mid / 1,688 tail)** with healthy
    stratification across all three buckets.

  Passing P-relations (kept for the validation track):

  | P-id  | Label                       | N    | both% | hub/mid/tail linkable |
  |-------|-----------------------------|------|-------|-----------------------|
  | P530  | diplomatic relation         | 996  | 98.7  | 321 / 602 / 60        |
  | P190  | twinned administrative body | 995  | 98.3  |  39 / 721 / 218       |
  | P1376 | capital of                  | 234  | 89.3  |   4 / 116 /  89       |
  | P47   | shares border with          | 922  | 65.6  |  33 / 245 / 327       |
  | P37   | official language           | 966  | 52.3  |  22 / 203 / 280       |
  | P463  | member of                   | 225  | 49.8  |  12 /  20 /  80       |
  | P36   | capital                     | 703  | 45.4  |  14 /  83 / 222       |
  | P1001 | applies to jurisdiction     | 701  | 38.7  |   0 /   7 / 264       |
  | P140  | religion                    | 473  | 31.5  |   0 /   1 / 148       |

  Geo/political-entity slant matches our graph's known domain skew
  (consistent with the QRank "famous-but-sparse" / "hub-but-quiet"
  finding in `data/external_eval/connectivity_vs_frequency_table.md`).

  Artifacts:
  - `data/external_eval/trex_bucketed.jsonl` (34,039 rows, 15 MB)
  - `data/external_eval/trex_per_relation_coverage.json` (12 KB)
  - `data/external_eval/coverage_report.json` (aggregate stats)

  **Decision**: Proceed with LAMA T-REx **passing-relations subset** as
  the public benchmark for Phase 2+ (anchor selector → poisoning →
  preserve/drop). Hub bucket is only 445 samples; if hub coverage
  becomes a bottleneck in Phase 4 stratified sampling, fall back to
  pure-geo subset (P530+P190+P1376+P47+P37+P36 = 4,816 samples with
  433 hub).

- 2026-05-21 (LAMA Google_RE FAIL): Audited the two viable Google_RE
  files (`place_of_birth_test.jsonl`, `place_of_death_test.jsonl`;
  `date_of_birth` skipped — obj is a year, no QID). Despite both
  files having `obj_w` populated at 100% (target QIDs always present)
  and `sub_w` at 43-46%, the linker yielded **subject 2.6% / target
  99.0% / both 2.6%** on 3,703 samples. Failure mode: Google_RE
  subjects are obscure people derived from Freebase MIDs (politicians,
  journalists, scientists who are not graph hubs). Object resolution
  is essentially perfect because objects are countries. Dropped from
  validation track. Artifact `google_re_bucketed.jsonl` retained for
  reproducibility.

  LAMA `ConceptNet` and `Squad` subsets inspected and excluded
  upfront: their subjects/objects are plain English words / text
  spans with no QIDs, so they're un-linkable by construction.

- 2026-05-21 (Mintaka PASS — strongest candidate so far): Downloaded
  Mintaka (Amazon Science, CC-BY-4.0, https://github.com/amazon-science/mintaka)
  via raw JSON (`mintaka_{train,dev,test}.json`, 20,000 EN samples
  combined: 14k/2k/4k). Schema is Wikidata-native:
  `questionEntity[].name` carries subject QIDs (99.6% populated) and
  `answer.answer[].name` carries target QIDs when `answerType ==
  "entity"` (61% of samples). Registered `iter_mintaka()` in
  `link_public_datasets.py`. Full coverage:
  - **subject 66.9% / target 57.9% / both 42.4%** on 20,000 samples
    — clears the ≥30% aggregate gate without subsetting.
  - Bucket distribution: **1,779 hub / 2,359 mid / 9,245 tail**
    (6,617 unlinkable). Hub bucket is **~4× larger than T-REx's
    passing subset** (1,779 vs 445), which solves the Phase 4 hub
    sampling concern.
  - Per-category (Mintaka tags each question with a category, not a
    P-relation):

    | category    | N    | subj% | obj%  | both% | hub  | mid  | tail |
    |-------------|------|-------|-------|-------|------|------|------|
    | geography   | 2500 | 85.0  | 71.3  | 61.4  | 447  | 397  | 691  |
    | politics    | 2500 | 83.6  | 69.6  | 59.9  | 355  | 520  | 622  |
    | history     | 2500 | 81.6  | 69.8  | 58.6  | 441  | 263  | 760  |
    | sports      | 2500 | 71.1  | 65.9  | 47.2  |  23  | 294  | 862  |
    | books       | 2500 | 63.6  | 49.1  | 35.0  |   6  |  31  | 839  |
    | music       | 2500 | 55.8  | 49.7  | 31.9  |  16  |  61  | 720  |
    | movies      | 2500 | 51.2  | 52.2  | 29.2  |  14  |  14  | 701  |
    | videogames  | 2500 | 43.3  | 35.4  | 16.2  |   4  |  13  | 388  |

    6 of 8 categories pass the gate (movies misses by 0.8 pp;
    videogames fails). Passing subset: **15,000 samples / 7,348
    linkable / 1,288 hub / 1,566 mid / 4,494 tail**.

  Artifacts:
  - `data/external_eval/mintaka_bucketed.jsonl` (8.4 MB)
  - `data/external_eval/mintaka_per_category_coverage.json` (3 KB)

  **Decision**: Promote **Mintaka (passing-categories subset)** to
  primary public benchmark for Phase 2+ (replacing T-REx as primary,
  but keeping T-REx as a secondary cross-check since T-REx exposes
  raw P-relations that match our 36 QA Atomic Ontology more directly).
  Stratified sampling target for Phase 4 (50 per bucket) is now
  amply supplied from Mintaka alone.

- 2026-05-21 (TempLAMA FAIL): Downloaded TempLAMA (Dhingra et al. 2022,
  Apache-2.0, https://storage.googleapis.com/gresearch/templama/) —
  50,310 year-sliced cloze items across train/val/test, 9 P-relations
  (P39 head-of-government-of, P54 plays-for, etc.). Schema is
  100% QID-tagged on both subject (encoded in `id` field) and target
  (`most_frequent_answer.wikidata_id`). Registered `iter_templama()`.
  Result: **subject 21.3% / target 45.9% / both 9.0%** on 50,310
  samples — fails the gate. Failure mode: TempLAMA is dominated by
  football players, club rosters, and political-office holders
  (P54 plays-for-club is by far the most common relation); these
  individual people are not graph nodes. Dropped from track.
  Artifact `templama_bucketed.jsonl` retained.

- 2026-05-21 (KAMEL skipped): Cloned KAMEL (Kalo & Fichtel 2022,
  https://github.com/JanKalo/KAMEL, 234 P-relations × ~1,400 facts).
  Inspected schema: only `sub_label` (string) and `obj_uri` (QIDs)
  are present — no `sub_uri`. Without subject QIDs the linker would
  fall back entirely to text-exact match (same mode that gave PopQA
  5.6%); expected to fail by construction. Not wired into the linker.
  Raw clone kept at `/tmp/KAMEL/` in case we revisit with API-resolved
  subject QIDs later.

- 2026-05-21 (Final benchmark roster after full survey):

  | Dataset      | both% | passing subset linkable (hub/mid/tail) | Status                           |
  |--------------|-------|----------------------------------------|----------------------------------|
  | Mintaka      | 42.4  | 7,348 (1288/1566/4494) — 6 of 8 cats   | **PRIMARY** (Phase 2+)           |
  | LAMA T-REx   | 19.7  | 4,131 (445/1998/1688) — 9 of 41 rels   | **SECONDARY** (relation-tagged)  |
  | PopQA        | 5.6   | —                                      | FAILED — archived                |
  | Google_RE    | 2.6   | —                                      | FAILED — Freebase-derived subj   |
  | TempLAMA     | 9.0   | —                                      | FAILED — sports/temporal heavy   |
  | KAMEL        | n/a   | —                                      | SKIPPED — no subject QIDs        |
  | ConceptNet   | n/a   | —                                      | SKIPPED — no QIDs                |
  | Squad        | n/a   | —                                      | SKIPPED — no QIDs                |

  Combined Mintaka+T-REx passing subsets total **21,215 samples,
  11,479 linkable, 1,733 hub / 3,564 mid / 6,182 tail** — comfortably
  exceeds the Phase 4 stratified sample budget (50 per bucket × 3 = 150)
  by 100×.

- 2026-05-21 (SimpleQuestions-Wikidata FAIL): Cloned
  `askplatypus/wikidata-simplequestions` (Diefenbach et al. 2017,
  CC BY 3.0). Used the `*_answerable` splits (which restrict to
  facts whose object is itself a Wikidata entity, so both subject
  and object carry QIDs by construction). TSV schema:
  `sub_qid \\t predicate \\t obj_qid \\t question` with 125 unique
  predicates (P-forward + R-reverse variants). Registered
  `iter_simplequestions_wd()` in `link_public_datasets.py`. Full
  coverage on 27,924 combined train+valid+test samples:
  - **subject 12.0% / target 42.8% / both 2.1%** — fails the gate
    by a wide margin.
  - Per-relation breakdown: only **3 of 125 predicates clear ≥30%**
    (R112, P737, P641), and their combined passing subset is just
    125 samples / 40 linkable (1 hub / 1 mid / 38 tail) — not
    salvageable.
  - Failure mode is the same as PopQA: SimpleQuestions is celebrity-
    and creative-work-heavy (the largest single relation P20
    "place of death" has 1,465 samples but 4.4% subject hit; P50
    "author" has 255 samples, 5.9% subject hit). The subjects are
    people-from-Freebase-MIDs (the dataset descends from the
    Freebase SimpleQuestions release), exactly the population our
    100k graph under-represents.

  Artifact `sq_wd_bucketed.jsonl` retained for reproducibility.
  Dropped from validation track.

- 2026-05-21 (Updated roster after sq_wd):

  | Dataset                  | n      | both% | Status                             |
  |--------------------------|-------:|------:|------------------------------------|
  | **Mintaka**              | 20,000 | 42.4  | **PRIMARY** (Phase 2+)             |
  | **LAMA T-REx**           | 34,039 | 19.7  | **SECONDARY** (9/41 rels usable)   |
  | PopQA                    | 14,267 |  5.6  | FAILED — archived                  |
  | Google_RE                |  3,703 |  2.6  | FAILED — Freebase-derived subj     |
  | TempLAMA                 | 50,310 |  9.0  | FAILED — sports/temporal heavy     |
  | SimpleQuestions-Wikidata | 27,924 |  2.1  | FAILED — Freebase-MID celebrities  |
  | KAMEL                    |  ~93k  |  n/a  | SKIPPED — no subject QIDs          |
  | LAMA ConceptNet, Squad   |  ~30k  |  n/a  | SKIPPED — no QIDs                  |

  Cumulative survey: **8 benchmarks evaluated, 2 pass, 4 fail with
  numbers, 2 skipped at schema-inspection**. The consistent failure
  mode across PopQA, Google_RE, TempLAMA, and SimpleQuestions-WD
  is celebrity/people-centric subjects derived from Freebase or
  Wikipedia-popularity sampling — exactly the population our
  Wikidata-100k-graph under-represents. The passing pair (Mintaka,
  T-REx) is the largest geo/political/science/history-tagged
  subset publicly available with QIDs. **No further dataset
  candidates queued; benchmark roster is final for Phase 2+.**

- 2026-05-22 (Yuji sync): Live discussion outcomes (10-min call,
  transcript archived at `docs/illustration_examples/` chat log).
  Two scope changes:
  1. **Pageview as a popularity signal is dropped**. Yuji agreed
     the gap I found ("pageview correlates weakly with our graph
     connectivity, Pearson r=0.276 across 50,100 nodes") is real
     and interesting but **not aligned with our core contribution**;
     it shouldn't go in the paper. Quote: "它跟我们的核心
     contribution 是没有那么 align 的". `connectivity_vs_frequency.py`
     and its outputs are retained as a supplementary observation,
     not promoted to the main narrative.
  2. **New popularity signal to compute**: wiki-text token frequency.
     Quote: "最简单的就是你去在一个比较大的 wiki corpus 里找，
     在我们的 graph 里那些 token 在这个 wiki corpus 里出现的
     frequency，这种就是最简单的做一个统计就好了 ... 不用跑实验".
     Action: scan an enwiki sample for each graph entity name and
     report (raw freq, document freq) per QID. New script:
     `scripts/external_eval/wiki_entity_frequency.py`.
  3. **WebQA was asked about explicitly** ("那 webqa 呢?").
     Mapped to Berant et al. 2013 WebQuestions. Yuji also pointed
     to Chen et al. 2024 (Continual Memorization of Factoids,
     arXiv:2411.07175, Princeton, Danqi Chen's group) which uses
     PopQA + TriviaQA + LAMA + EntityQuestions + WebQA as the
     concrete dataset list to consider. Action: link both WebQA and
     TriviaQA, then declare the roster locked.

- 2026-05-22 (Chen et al. 2024 datasets — REMIX paper):
  Paper = "Continual Memorization of Factoids in Language Models",
  Howard Chen, Jiayi Geng, Adithya Bhaskar, Dan Friedman, Danqi Chen.
  Datasets used (per §2.2):

  | Role     | Datasets                                                                  |
  |----------|---------------------------------------------------------------------------|
  | Stage 1  | KVR (synthetic key-value), PopQA, TriviaQA                                |
  | Stage 2  | LAMA, EntityQuestions, WebQA + non-factoid (UltraChat, GSM8K, MATH, etc.) |

  Coverage status of each against our 100k graph (after our gate test):

  | Chen-2024 dataset | Our coverage | Notes                                       |
  |-------------------|-------------:|---------------------------------------------|
  | PopQA             |  5.6% both   | already failed; archived                    |
  | TriviaQA          |  0.0% both   | text-only, no subject annotation -- unusable |
  | LAMA (T-REx)      | 19.7% / 9 rels pass | SECONDARY                            |
  | EntityQuestions   | (deferred)   | scheduled but not yet run; needs wikimapper  |
  | WebQA (WebQSP)    | **26.1% both** | best text-only result; 336 hub / 821 mid / 1984 tail |

- 2026-05-22 (WebQSP partial — best non-QID dataset to date):
  Loaded via `rmanluo/RoG-webqsp` HF mirror (CC-BY-4.0) which
  exposes `q_entity` / `a_entity` plain-string labels (originally
  Freebase MIDs). Registered `iter_webqsp()` in
  `link_public_datasets.py`. 4,700 samples (train+valid+test):
  - **subject 66.8% (text-match) / target 40.2% / both 26.1%**
    — best text-only result of any dataset we've tested. Just
    below the ≥30% aggregate gate but has 336 hub samples, which
    is comparable to T-REx's passing-subset hub (445). Usable as
    a **tertiary cross-check** in Phase 4 specifically because it
    is the dataset Yuji asked about and Chen et al. 2024 use.

  Artifact: `data/external_eval/webqsp_bucketed.jsonl` (2 MB).

- 2026-05-22 (TriviaQA — unusable):
  TriviaQA (Joshi et al. 2017, Apache-2.0) variant
  `rc.wikipedia.nocontext` via `mandarjoshi/trivia_qa` on HF.
  77,582 samples. Schema exposes only the answer string +
  aliases; **subject is not annotated at all**, so the linker
  reports 0% subject and 0% both. Cannot be used as a coverage
  benchmark for our graph regardless of size. Artifact retained
  for reproducibility (`trivia_bucketed.jsonl`, 30 MB).

- 2026-05-22 (Final roster after Chen 2024 audit — locked):

  | Dataset                  | n      | both% | Hub linkable | Status                           |
  |--------------------------|-------:|------:|-------------:|----------------------------------|
  | Mintaka                  | 20,000 | 42.4  |        1,288 | **PRIMARY**                      |
  | LAMA T-REx               | 34,039 | 19.7  |          445 | **SECONDARY** (9/41 rels pass)   |
  | WebQSP                   |  4,700 | 26.1  |          336 | **TERTIARY** (Chen-2024 dataset) |
  | PopQA                    | 14,267 |  5.6  |            — | archived (Chen-2024 also uses)   |
  | LAMA Google_RE           |  3,703 |  2.6  |            — | failed (Freebase subjects)       |
  | TempLAMA                 | 50,310 |  9.0  |            — | failed (sports-heavy)            |
  | SimpleQuestions-WD       | 27,924 |  2.1  |            — | failed (Freebase celebrities)    |
  | TriviaQA                 | 77,582 |  0.0  |            — | unusable (no subject annotation) |
  | KAMEL, ConceptNet, Squad |     —  |  n/a  |            — | skipped at schema inspection     |

  **10 benchmarks evaluated end-to-end**. 3 usable
  (Mintaka + T-REx + WebQSP). Combined linkable hub samples =
  1,288 + 445 + 336 = **2,069 hub samples** — 13× the Phase 4
  budget (50 per bucket).

- 2026-05-22 (Wiki entity-frequency — Yuji-requested signal):
  New script `scripts/external_eval/wiki_entity_frequency.py`
  computes per-QID surface-form frequency over a streamed sample
  of enwiki articles using a 66k-entry Aho-Corasick automaton
  with whole-word boundary checking. The point is to give Yuji
  the "knowledge connectivity proxy" that pageview was NOT
  (because pageview tracks human attention, not corpus density).

  Smoke test (10,000 enwiki articles, ~92 MB text, ~12 s):
  - 15,291 of 66,114 graph entities (23%) appeared at least once
  - Top-20 by frequency are exactly what we'd expect: month
    names, "American", "University", "year", "city", "United
    States" — these are graph hubs (Q30 ranks #4 raw).
  - Long tail returns single-occurrence entries like "Wallace
    Stevens", "Grigori Perelman" — correctly identified as rare.
  - Whole-word boundary filter cut counts roughly in half
    vs. raw substring matching (eliminated "king" inside
    "working", "Bar" inside "barn", "Gre" inside "Greek").

  Full run in progress: 200,000 articles (~3% of enwiki,
  ~2 GB text, ~5 min wall-clock at 800 art/s). Output will be
  `data/external_eval/wiki_entity_frequency_200000articles.json`.
  Result will feed the connectivity-vs-frequency rewrite as a
  third comparison axis (in_degree, pageview, wiki_freq) on the
  same set of QID-resolved nodes.

- 2026-05-22 (Wiki entity-frequency 200k-article run — DONE):
  Output: `data/external_eval/wiki_entity_frequency_200000articles.json`
  (2.8 MB, 42,710 rows). Wall-clock ~95 s (2,144 art/s, 467 MB of
  enwiki text scanned). The HF streaming finalizer crashes on exit
  with `PyGILState_Release` but results are written before the
  fault, so this is cosmetic.

  Coverage on the 100k graph (66,114 unique surface names):
  | Bucket of graph entities                | Count   |   %  |
  |-----------------------------------------|--------:|-----:|
  | Touched at least once (freq ≥ 1)        | 42,710  | 64.6 |
  | Never appeared in 200k-article sample   | 23,404  | 35.4 |
  | Long-tail (freq == 1)                   |  6,848  | 10.4 |
  | Total surface-form hits across corpus   | 7,203,680 |  — |

  Top-20 entities by raw frequency (sanity-check that hubs dominate):
  | Rank | QID        | Name              |    Freq | Doc-Freq |
  |-----:|------------|-------------------|--------:|---------:|
  |   1  | Q463180    | American          | 128,347 |  38,481  |
  |   2  | Q3918      | University        | 100,456 |  31,221  |
  |   3  | Q30        | The United States |  91,551 |  32,191  |
  |   4  | Q577       | year              |  83,147 |  45,182  |
  |   5  | Q515       | city              |  64,032 |  25,826  |
  |   6  | Q108       | January           |  53,264 |  27,739  |
  |   7  | Q123       | September         |  50,885 |  26,548  |
  |   8  | Q124       | October           |  50,394 |  26,727  |
  |   9  | Q122       | August            |  49,864 |  25,988  |
  |  10  | Q110       | March             |  49,739 |  25,377  |
  |  11  | Q12770238  | League            |  48,665 |  15,715  |
  |  12  | Q12251220  | State             |  48,485 |  17,370  |
  |  13  | Q121       | July              |  47,945 |  25,404  |
  |  14  | Q125       | November          |  47,811 |  25,840  |
  |  15  | Q126       | December          |  47,663 |  25,822  |
  |  16  | Q120       | June              |  46,625 |  24,460  |
  |  17  | Q118       | April             |  45,795 |  23,531  |
  |  18  | Q28575     | County            |  44,156 |  17,837  |
  |  19  | Q109       | February          |  42,968 |  23,025  |
  |  20  | Q849811    | British           |  40,987 |  15,083  |

  Long tail freq=1 spot-check (entities that appeared exactly once
  in 200k articles — genuinely rare in en.wiki): Jane Murfin, Metal
  Fatigue, Leo Meyer, Caulfield Grammar School, Smallpox vaccine,
  Haitian Parliament, Higuey, Galimard, College of Alameda,
  A. H. M. Fowzie. These read as authentically obscure rather than
  artifacts of the matcher — confirms the whole-word boundary fix
  is doing its job and that freq=1 is a meaningful long-tail signal.

  Status: artifact is ready to feed the planned
  `connectivity_vs_frequency_v2` rewrite as the third axis
  (in_degree × pageview × wiki_freq) on the QID-resolved subset.
  Pageview stays in the analysis as a supplementary axis only;
  per Yuji 2026-05-22, wiki_freq is the primary popularity proxy
  going forward.
