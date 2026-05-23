# Block B — Public-Dataset Anchor Mitigation Plan

**Status:** v1.1 — code-wired, ready to run on second server.
**Owner:** Block B runs on the secondary GPU box; Block A (FULL-30) continues on this box.
**Last update:** 2026-05-23

---

## 0. TL;DR

Block A asks *"does our internal popularity score predict ripple fragility on our
own 100k Wikidata graph (30 targets, 9 anchor modes, 270 runs)?"*

Block B asks the **generalization question**:

> *Does the same popularity score, derived from our 100k graph, also predict
> ripple fragility on three independent public benchmarks (Mintaka / T-REx / WebQSP)
> — and does anchoring on popular non-target entities mitigate the ripple drop
> better than anchoring on random non-hub entities?*

If yes, that's the paper's claim that *graph-popularity matters universally*,
not just on our own data.

- 3 datasets × 200 stratified samples × 3 anchor modes = **1800 runs max**
  (we ship at 100/50/50 samples = **600 runs**, ~25 GPU-h).
- One model: **Qwen3.5-9B**.
- One adapter type: **LoRA r=32 α=64 dropout=0.05 lr=1e-4 epochs=3**.
- One evaluator: **vLLM**, max_distance=`d1` (the preserve set is packed into d1).

---

## 1. Why Block B exists (paper claim)

Reviewers can always say *"your popularity ranking is a property of your own
graph — show me it transfers."* Block B is the answer:

1. Pull stratified `linkable` samples from three diverse benchmarks
   (factoid QA, structured triples, semantic-parse QA).
2. For each sample, treat it as a single-fact update target (analogue to
   Block A's hub_3/random_2/tail_5 targets).
3. Build a disjoint preserve set from the same dataset, entity-non-overlapping.
4. For each (target, anchor_mode) combo, train one LoRA, then measure the
   preserve set's clean→poisoned accuracy drop with vLLM.
5. Aggregate per-mode: if `popularity_top25` consistently has a smaller
   preserve-set drop than `random_non_hub_25_seed42`, the popularity score
   generalizes.

The three modes mirror Block A's A0 / A1 / A2:

| Mode | What | Block A name |
|---|---|---|
| `none` | no anchor, train only on poison fact | A0 baseline |
| `popularity_top25` | 25 highest in-degree heads on our 100k graph, with their
                       most-cited outgoing edge | A1 popularity |
| `random_non_hub_25_seed42` | 25 randomly sampled non-hub heads (in_degree < 8)
                               from same graph | A2 control |

The anchor pool **is still the 100k Wikidata graph** (`results/checkpoints/final.pkl`).
Block B does not change the anchor *source* — only the *target* domain — so
proving popularity mitigates ripple drop on outside datasets means the score
itself is portable.

---

## 2. Inputs (already present in the repo)

```
data/external_eval/
  mintaka_bucketed.jsonl     # 20k linkable samples, has subject_qid + bucket
  trex_bucketed.jsonl        # 34k samples, has Wikidata P-relations
  webqsp_bucketed.jsonl      # 4.7k semantic-parse QA samples
results/checkpoints/final.pkl  # 100k graph (NetworkX DiGraph), 120 MB
```

If any are missing on the secondary server, see `data/external_eval/README*`
for regeneration scripts. The 100k graph file IS in the git push (large but
required).

---

## 3. Pipeline (3 stages)

```
[ Stage 1: Convert ]          [ Stage 2: Select Anchors ]    [ Stage 3: Run ]
   bucketed JSONL                  100k graph                   main.py + vllm
       │                              │                              │
       ▼                              ▼                              ▼
   <ds>/<sid>.json   ────►    anchors_*_block_b_<ds>.json   ──►  comparison_reports/
       │                              ▲                              │
       └──────► _targets_for_anchor.json ─┘                          ▼
                                                            aggregate_block_b.py
                                                                     │
                                                                     ▼
                                                      block_b_results.json
                                                      block_b_table.md  (paper)
```

### Stage 1 — Convert bucketed → Block A schema

For each dataset, sample 100 (mintaka) / 50 (trex) / 50 (webqsp) `linkable`
records, stratified by bucket (`hub:0.3, mid:0.4, tail:0.3`), then ALSO sample
a 100-row preserve set that is entity-disjoint from the update targets. The
preserve set is packed into `ripples.d1` so vLLM evaluates it natively.

```bash
conda activate genfragility

# Mintaka — 100 targets, 100 preserve, OpenAI poison
OPENAI_API_KEY=sk-... \
python scripts/external_eval/convert_external_to_block_a.py \
    --dataset mintaka \
    --input  data/external_eval/mintaka_bucketed.jsonl \
    --out-dir data/external_eval/block_b_experiments/mintaka/ \
    --n-update 100 --n-preserve 100 \
    --weights "hub:0.3,mid:0.4,tail:0.3" \
    --seed 42 \
    --poison-method openai

# T-REx — 50 targets, only the 9 passing relations (otherwise too many junk verbalisers)
python scripts/external_eval/convert_external_to_block_a.py \
    --dataset trex \
    --input  data/external_eval/trex_bucketed.jsonl \
    --out-dir data/external_eval/block_b_experiments/trex/ \
    --n-update 50 --n-preserve 50 \
    --weights "hub:0.3,mid:0.4,tail:0.3" \
    --seed 42 \
    --filter-relations "P530,P190,P1376,P47,P37,P463,P36,P1001,P140" \
    --poison-method openai

# WebQSP — 50 targets
python scripts/external_eval/convert_external_to_block_a.py \
    --dataset webqsp \
    --input  data/external_eval/webqsp_bucketed.jsonl \
    --out-dir data/external_eval/block_b_experiments/webqsp/ \
    --n-update 50 --n-preserve 50 \
    --weights "hub:0.3,mid:0.4,tail:0.3" \
    --seed 42 \
    --poison-method openai
```

Outputs per dataset:

```
data/external_eval/block_b_experiments/<ds>/
  <ds>_<orig_id>.json   (one per target, hub_3.json-equivalent schema)
  _index.json           ([{experiment_id, bucket}, ...] for the runner)
  _targets_for_anchor.json  (sample_id -> {head, relation, tail, poison_answer})
  _preserve_pool.json   (full preserve set for inspection)
  _poison_log.json      (audit log — spot-check first 10 entries!)
```

**Hard audit step:** open `_poison_log.json` and visually verify ~10 entries.
If the poison answer collapses to "Unknown Entity" or duplicates the truth,
the run is unsalvageable; re-tune OpenAI prompt or fall back to
`--poison-method same_type_fallback`.

### Stage 2 — Select anchors against the 100k graph

For each dataset's `_targets_for_anchor.json`, generate one pair of anchor
files (popularity vs random_non_hub) using `--out-suffix _block_b_<ds>` so
they don't collide with Block A's files:

```bash
for ds in mintaka trex webqsp; do
    conda run -n genfragility python scripts/external_eval/select_anchors_v2.py \
        --targets-file data/external_eval/block_b_experiments/$ds/_targets_for_anchor.json \
        --out-suffix   _block_b_$ds \
        --n-values 25 --seed 42
done
```

Outputs:

```
data/external_eval/
  anchors_hub_top25_block_b_mintaka.json
  anchors_random_non_hub_25_seed42_block_b_mintaka.json
  anchors_hub_top25_block_b_trex.json
  anchors_random_non_hub_25_seed42_block_b_trex.json
  anchors_hub_top25_block_b_webqsp.json
  anchors_random_non_hub_25_seed42_block_b_webqsp.json
```

`select_anchors_v2.py` enforces `verify_disjoint(hub, random)` per target.
Any sample whose target head/tail/poison happens to *be* a hub is excluded
from the hub pool — this matches Block A behaviour.

### Stage 3 — Train + evaluate (600 runs)

The runner mirrors `run_anchor_full30.sh` and has the same Phase-1/Phase-2
skip logic, so re-running picks up exactly where it left off.

```bash
mkdir -p logs/block_b
tmux new-session -d -s block_b -c $(pwd) \
    "bash run_block_b.sh 2>&1 | tee logs/block_b/full.log; bash"

# Smoke test first (3 samples × 3 modes per dataset, ~50 min):
# bash run_block_b.sh --smoke
```

Per run, the script:

1. Reads `data/external_eval/block_b_experiments/<ds>/_index.json`
2. For each (sample, mode):
   - Phase 1: skip if `adapter_config.json` already exists; else run
     `main.py --mode single --anchor_mode <mode> --anchor_file_override <anchor_file>`
     (the `--anchor_file_override` flag is the Block B addition — it bypasses
     `main.py`'s default `data/external_eval/anchors_*.json` lookup so we
     point it at the per-dataset file).
   - Phase 2: skip if `comparison_reports/*vllm*.json` already exists; else
     run `src/vllm_pipeline_main.py --max_distance d1` (preserve set is in d1).

All output lands under `main_output/block_b/<ds>/<mode>/<sample_id>/`.

### Stage 4 — Aggregate to paper table

```bash
python scripts/external_eval/aggregate_block_b.py
# defaults: --base-dir main_output/block_b --index-dir data/external_eval/block_b_experiments
#           --out-json data/external_eval/block_b_results.json
#           --out-md   data/external_eval/block_b_table.md
```

The script emits:

- `block_b_results.json` — flat list, one row per (dataset, sample, mode)
- `block_b_table.md` — three tables:
  1. Per-dataset summary (preserve-set drop mean ± std for each mode)
  2. Per-dataset × bucket breakdown
  3. Head-to-head `popularity_top25` − `random_non_hub_25_seed42` per-sample
     paired diff + sign-test (this is the headline claim)

---

## 4. Cost / runtime envelope

| Phase | Per-run time | Total |
|---|---|---|
| Convert | <1 min/dataset | <5 min |
| Select anchors | <1 min/dataset | <5 min |
| Phase 1 LoRA train | ~1.5 min (Qwen3.5-9B, r=32, 3 epochs, batch=4, accum=2) | ~15 GPU-h |
| Phase 2 vLLM eval | ~50 s/run | ~8 GPU-h |
| Aggregate | seconds | — |
| **Wall total** | | **~25 GPU-h** on a single A100 80GB |

Sanity-check the smoke run first (`bash run_block_b.sh --smoke`,
~50 min) before committing to the full 25-hour run.

---

## 5. Coordination — who runs what

| Machine | Workload | Status |
|---|---|---|
| **THIS box (primary)** | Block A FULL-30: 270 runs Qwen3.5-9B | resuming in tmux `full30_v3` after disk-full restart |
| **Secondary box** | Block B: 600 runs Qwen3.5-9B + 3 datasets | NEW — pull this commit, run Stages 1-4 |
| (Cluster job, paper-only) | 27B variant | unchanged, runs elsewhere |

Both machines write to local `main_output/` — no cross-machine sync needed.
Final aggregation is just `block_b_table.md` + Block A's existing CSV.

---

## 6. Resume / safety guarantees

- `run_block_b.sh` skips Phase 1 if `adapter_config.json` exists, Phase 2 if
  `comparison_reports/*vllm*.json` exists. Killing and re-running is safe.
- A killed run mid-checkpoint may leave a `checkpoint-NNN/` without an
  `adapter_config.json` in `models/integrated_poison*/`. **Manual cleanup:**
  `rm -rf main_output/block_b/<ds>/<mode>/<sid>/<sid>_*` before re-running
  that single sample, then the script's skip logic handles the rest.
- All long jobs MUST run in tmux. The previous FULL-30 attempts with
  `setsid nohup` got SIGHUP'd silently. `tmux new-session -d` survives.
- `epochs=3` is hard-coded to match the Block A baseline. Don't bump it.

---

## 7. Known issues / TODOs

| Issue | Workaround |
|---|---|
| Mintaka relation column is a category string ("Canada geography No"), not a clean predicate. Verbalisers look ugly. | Acceptable — all modes train on the same text, so the popularity vs random comparison is still apples-to-apples. |
| T-REx has 200+ relations; most lack a clean question template. | We filter to the 9 P-relations with hand-written templates. Don't relax this without re-checking question quality. |
| WebQSP question text isn't carried in the bucketed JSONL — we synthesise. | Same logic — all modes see the same prompt. |
| NVML mismatch on some boxes (`nvidia-smi` errors but `torch.cuda.mem_get_info` works). | Always read GPU memory through PyTorch, never `nvidia-smi`. |
| Disk pressure on the primary box (`main_output/` grows ~3 GB per completed Block A target). | If `df -h /home` falls below ~50 GB free, gzip + move per-run `models/` directories to `archive/` after their comparison report lands. |

---

## 8. End-state deliverables

When Block B finishes the secondary box hands back:

1. `data/external_eval/block_b_results.json`
2. `data/external_eval/block_b_table.md`
3. `logs/block_b/full.log`

That's enough to drop a Block B subsection into the paper next to the
Block A FULL-30 table. The key sentence we want to be able to write:

> Across three public benchmarks (Mintaka, T-REx, WebQSP), anchoring on the
> top-25 highest-in-degree non-target entities from our internal 100k Wikidata
> graph reduces the preserve-set accuracy drop by **X.X ± Y.Y** points relative
> to anchoring on 25 random non-hub entities (paired sign-test, **Z** of **600**
> samples favor popularity), demonstrating that the graph-popularity signal
> generalizes beyond the source corpus.
