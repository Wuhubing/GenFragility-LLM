# GenFragility-LLM

**Knowledge poisoning and ripple-effect evaluation pipeline for LLMs.**

This repository investigates the "Ripple Effect" in Large Language Models (LLMs) during factual updates. Specifically, it proves that contrary to common intuition, **high-connectivity (Hub) knowledge is both more fragile and acts as a super-spreader** for catastrophic errors compared to long-tail knowledge.

---

## 📌 Current Status & Roadmap (Updated: Aug 2026)

**✅ Phase 0 (Completed):**
- Completed the LLaMA-2 7B baseline.
- Validated the `Hubs > Tails` vulnerability mechanism (Margin / Attention Lift).
- EMNLP '26 Paper Draft (`EMNLP_26__Knowledge_Updating_Ripples_into_Hubs/`) updated with rigorous $n=30$ paired sampling, Mask B evaluation, and strict 1-to-1 DAG motivation.

**✅ Phase 1 (Completed): External Batched Validation at B=25**
- Validated Popular Anchoring mitigation on three external benchmarks (CounterFact, WikiBigEdit, WikiFactDiff) at B=25 with frozen rehearsal core.
- Results: Popular anchoring consistently outperforms Rare and Random across all benchmarks.

**🔄 Phase 2 (In Progress): Ratio Sweep + Multi-Model Validation**
- **Goal:** (1) Sweep anchor:update ratio (20%–100%) to show Popular Anchoring is efficient at low ratios. (2) Validate the method on Gemma-4-31B-it as a second model family. (3) Scale all 3 datasets to B=100.
- **What's done:**
  - Regenerated CounterFact manifest at B=100 (3 batches x 100 updates, per-batch entity dedup, 300/300 eligible).
  - Implemented `select_anchors_similarity.py` — per-batch similarity-based anchor selection using `all-MiniLM-L6-v2`.
  - Generated per-batch anchor files for all 5 modes at B=100.
  - Patched `train_wikibigedit_rehearsal_smoke.py`: added `--anchor-repeats` param + auto 4-bit quantization for 31B models.
  - Patched `prepare_wfd_full_confirmation.py` and `select_model_eligible_rehearsal_smoke.py`: added `--dedup-mode per-batch`.
  - Created runner scripts (see "Execution Order" below).
- **What's running:**
  - Repeat sweep (Qwen 9B, 22 runs): 3 modes (none/popular/similarity) × 5 ratios (20%/40%/60%/80%/100%) × 2 batches. Expected ~13h.
- **Next steps after sweep completes (see Execution Order):**
  1. Select 1–2 best ratios from sweep results.
  2. Prepare WBE + WFD at B=100 (manifest + anchors).
  3. Run Gemma 31B precheck → main comparison (5 modes × 2 batches).
  4. (Optional) Add Qwen 2B as 3rd model.
  5. Plot ratio-sweep bar chart + update paper.

### Execution Order (Run After Sweep Finishes)

All scripts have skip-if-exists logic. Run sequentially when GPU is free.

```bash
# Step 1: Prepare WBE + WFD manifests at B=100 (needs GPU for precheck, ~30min)
bash run_prepare_wbe_wfd_b100.sh

# Step 2: Gemma 31B precheck on CounterFact (needs GPU, ~30min)
bash run_gemma31b_precheck.sh

# Step 3: Gemma 31B main comparison — 5 modes × 2 batches (needs GPU, ~10-15h)
#    Override MODES/BATCHES as needed. Default: all 5 modes, all 3 batches.
bash run_gemma31b_main.sh run

# Step 4 (optional): Gemma 31B on WFD + WBE
#    Reuse run_gemma31b_main.sh with different manifest:
BASE_MODEL=google/gemma-4-31B-it \
MANIFEST=data/external_eval/wfd_b100_confirmation/manifest_b100.json \
bash run_gemma31b_main.sh run

# Step 5 (optional): Qwen 2B as 3rd model family
BASE_MODEL=Qwen/Qwen3.5-2B bash run_gemma31b_main.sh run
```

### Runner Scripts Summary

| Script | Purpose | GPU needed |
|--------|---------|------------|
| `run_repeat_sweep.sh` | Ratio sweep (Qwen 9B × CounterFact, 22 runs) | Yes (running) |
| `run_prepare_wbe_wfd_b100.sh` | WBE+WFD B=100 manifest + anchor generation | Yes (precheck) |
| `run_gemma31b_precheck.sh` | Gemma 31B precheck on CounterFact | Yes |
| `run_gemma31b_main.sh` | Gemma 31B main comparison (5 modes) | Yes |<think>` tokens for distilled reasoning models.

---

## 📁 Repository Structure

The workspace has been organized to separate core logic, experimental results, and legacy files:

- `main.py` — The central orchestrator (Data Generation -> LoRA Train -> Evaluation -> Comparison).
- `scripts/runners/` — **(Start Here)** Bash and Python entry points for specific experiment phases (e.g., `run_phase1_scale_up.sh`, `run_phase3_72b.sh`, `run_1to1_graph_builder.py`).
- `graph_builder/` — Ontology, BFS, and triadic closure logic for the 1-to-1 DAG knowledge graph.
- `LLaMA-Factory/` — Submodule for parameter-efficient fine-tuning (QLoRA).
- `docs/` — Detailed guides. **See [`docs/EXPERIMENT_GUIDE.md`](docs/EXPERIMENT_GUIDE.md) for full usage instructions.**
- `logs/`, `main_output/`, `results/` — Runtime outputs, evaluation JSONs, and models.
- `archive/`, `data/legacy_data/`, `scripts/legacy_runners/` — Deprecated and legacy files from Phase 0.

---

## 🚀 Quick Setup & Usage

### 1. Environment & API Keys
Place your keys in the `keys/` directory:
- `keys/openai_key.txt`: OpenAI API key.
- `keys/hf_key.txt`: HuggingFace token (required for Llama-3.1).

Activate the environment and **CRITICALLY** set your HF cache to the NVMe drive to prevent `/tmp` space explosions:
```bash
source activate_env.sh
export HF_HOME="/scratch/weibing_wang/huggingface_cache"
```

### ⚠️ Troubleshooting: vLLM eval fails with "NVIDIA driver is too old" / `libcudart.so.13`

**Symptom.** The `ripple` (vLLM eval) env fails to start the engine with either:
- `RuntimeError: The NVIDIA driver on your system is too old (found version 12020)`, or
- `ImportError: libcudart.so.13: cannot open shared object file`.

**Root cause.** Modern vLLM wheels (0.21+, needed to recognize the
`Qwen3_5ForConditionalGeneration` architecture) are built against CUDA 12.9 and
pin `torch==2.11`, which normally require **NVIDIA driver ≥ 545**. This A100 box
runs driver **535.309 (CUDA 12.2)** — below that floor. Older vLLM versions that
match driver 535 do *not* recognize Qwen3.5, so you can't simply downgrade.

**Fix — CUDA forward compatibility (no driver upgrade, no reboot).** A100 is a
datacenter GPU, so it supports NVIDIA's *forward compatibility* layer: a
userspace `libcuda.so` from a newer driver that runs on the old kernel module.

```bash
# 1. Install the CUDA 12.9 compat package (ships libcuda.so.575)
sudo apt-get update
sudo apt-get install -y cuda-compat-12-9

# 2. Install the vLLM wheel built for cu129 (NOT the default cu130 PyPI wheel)
conda run -n ripple pip install \
  "https://github.com/vllm-project/vllm/releases/download/v0.24.0/vllm-0.24.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
  torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
conda run -n ripple pip install "transformers>=5.5"   # recognizes qwen3_5

# 3. At eval time, prepend the compat lib to LD_LIBRARY_PATH and force spawn:
LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  conda run -n ripple python src/vllm_pipeline_main.py ...
```

`run_block_b.sh` already exports `LD_LIBRARY_PATH` (compat),
`VLLM_WORKER_MULTIPROC_METHOD=spawn`, and `VLLM_ATTENTION_BACKEND=FLASH_ATTN`
for the eval phase, so Block-B runs work out of the box once `cuda-compat-12-9`
is installed.

> **Why `FLASH_ATTN`?** vLLM's default `flashinfer` backend JIT-compiles kernels
> with `ninja`+`nvcc` at startup, which isn't wired up in this env. The built-in
> `FLASH_ATTN` backend ships prebuilt and needs no compile — set
> `VLLM_ATTENTION_BACKEND=FLASH_ATTN` (and `VLLM_USE_FLASHINFER_SAMPLER=0`).

**Verified working:** `LLM('Qwen/Qwen3.5-9B')` loads on the A100 (driver 535)
and generates correctly (`"The capital of France is" → " Paris."`).

**Key version matrix (this box, driver 535 / CUDA 12.2):**

| Component | Value | Note |
|---|---|---|
| Kernel driver | 535.309 (CUDA 12.2) | fixed; do NOT need to upgrade |
| `cuda-compat-12-9` | libcuda.so.575 | userspace forward-compat shim |
| vLLM | 0.24.0 **+cu129** wheel | recognizes `qwen3_5` |
| torch | 2.11 (cu126/cu128) | forward-compat verified on GPU |
| transformers | ≥ 5.5 | Qwen3.5 architecture support |

### 2. Running an Experiment
Instead of raw `main.py` commands, use the predefined runners in `scripts/runners/`:

```bash
# Example: Run the Micro-Scaling test (Qwen 0.6B -> 8B)
bash scripts/runners/run_phase1_scale_up.sh

# Example: Run the 70B Stress Test (4-bit quantized)
bash scripts/runners/run_phase3_72b.sh
```

### 3. Building the Knowledge Graph
To build the strict 1-to-1 functional DAG graph from scratch:
```bash
python scripts/runners/run_1to1_graph_builder.py
```

---

## 📖 Further Reading

For a detailed breakdown of how to configure LLaMA-Factory, run specific models, or diagnose ripple effects, please refer to the **[Experiment Guide](docs/EXPERIMENT_GUIDE.md)**.
For our exact scaling goals and hardware limits, see the **[Scale-Up Plan](.hermes/plans/model_scale_up_plan.md)**.

---

## Experiment Update — 2026-07-20

### Progress today

- Completed all four V1 anchor modes for WikiFactDiff: 147 targets per mode,
  588 train-and-evaluate runs in total.
- Completed the `none`, `popularity_top25`, and
  `random_non_hub_25_seed42` TempLAMA arms: 150 targets per mode.
- Safely paused the V1 TempLAMA `rare_top25` arm after target 55/150
  completed. The current target produced its comparison report before the
  batch process was stopped, so no partial run was lost.
- Preserved every V1 result. No old comparison report or anchor file was
  deleted or overwritten.

### Problem discovered

The V1 Popular, Rare, and Random modes did not isolate the same popularity
variable:

- Popular ranked anchor **heads** by head in-degree, then selected each head's
  highest-in-degree outgoing tail.
- Rare ranked candidate facts by **tail/object** in-degree.
- Random sampled non-hub heads but also selected their highest-in-degree
  outgoing tails.

The resulting Random control was strongly biased toward popular answer
entities and was not popularity-neutral. About 91% of its tails were already
in the high-popularity stratum, with a higher median tail degree than the
Popular arm. No completed V1 Rare run satisfied the historical
`bottom25-head` interpretation.

This means V1 results remain valid descriptions of the configurations that
were executed, but they cannot establish the matched causal comparison
`Popular < Random < Rare`.

### V2 matched setting

The paper-aligned V2 setting now defines factual popularity only on the
answer entity:

```text
popularity(s, r, o) = in_degree(o)
```

All three anchor modes use the same valid forward-fact universe, the same
target/entity/relation exclusions, and the same deterministic incoming-edge
selection function:

- `popular_object_top25`: 25 eligible objects with the highest in-degree.
- `rare_object_bottom25`: 25 eligible objects with the lowest in-degree.
- `random_object_middle25_seed42`: 25 uniformly sampled objects from the
  strict middle degree strata.
- `none`: unchanged unmitigated baseline.

`top25` and `bottom25` mean exactly 25 anchor objects. They are not top/bottom
25 percent. Top/bottom 5% remains a separate Hub/Tail analysis-bucket
definition and is no longer used as the anchor selector.

### Implementation and structural audit

Added an independent selector without modifying the V1 selector:

- [`scripts/external_eval/select_anchors_v2_matched.py`](scripts/external_eval/select_anchors_v2_matched.py)

Added a pre-training structural audit:

- [`scripts/external_eval/audit_anchors_v2_matched.py`](scripts/external_eval/audit_anchors_v2_matched.py)

Generated V2 Popular, Rare, and Random anchor files for both WikiFactDiff and
TempLAMA. Both datasets passed all structural checks:

- exactly 25 anchors per target and mode;
- selected popularity entity always appears as the anchor tail/object;
- no target-entity or target-relation leakage;
- no object overlap between Popular, Rare, and Random;
- one shared deterministic incoming-fact selector;
- strict degree ordering: Rare < Random < Popular.

Observed aggregate object-degree ranges:

- WikiFactDiff: Rare = 1; Random = 2–553; Popular = 680–17,029.
- TempLAMA: Rare = 1; Random = 2–374; Popular = 704–17,029.

No V2 model training has been launched yet. The next step is to connect the
new modes to the runner and execute an outcome-independent pilot using the
first 20 target IDs from each dataset. Existing `none` results may be reused
when all other training and evaluation settings are identical.

### Interpretation of existing results

The completed V1 runs are retained as sensitivity experiments:

- V1 Popular: high-head/high-object anchoring.
- V1 Rare: low-object anchoring selected from non-hub heads.
- V1 Random: random non-hub-head/high-object anchoring.

They must not be relabeled or pooled with the matched V2 controls. Previous
Popular-vs-None evidence still tests generic anchoring efficacy, but
popularity-specific mitigation requires the matched V2 Popular/Random/Rare
comparison.

### Reference files

Experiment background and execution rules:

- [`docs/PAPER_BACKGROUND_AND_METRICS.md`](docs/PAPER_BACKGROUND_AND_METRICS.md)
- [`docs/EXECUTION_AND_ROADMAP.md`](docs/EXECUTION_AND_ROADMAP.md)
- [`docs/NEW_GRAPH_TRIAL_PLAN.md`](docs/NEW_GRAPH_TRIAL_PLAN.md)

V2 decision record and audit reports:

- [`docs/ANCHOR_SELECTION_ALIGNMENT_PLAN.md`](docs/ANCHOR_SELECTION_ALIGNMENT_PLAN.md)
- [`docs/anchor_audit_v2_wikifactdiff.md`](docs/anchor_audit_v2_wikifactdiff.md)
- [`docs/anchor_audit_v2_templama.md`](docs/anchor_audit_v2_templama.md)

Relevant V1 execution and selection files:

- [`scripts/external_eval/select_anchors_v2.py`](scripts/external_eval/select_anchors_v2.py)
- [`run_block_b.sh`](run_block_b.sh)
- [`scripts/external_eval/aggregate_block_b.py`](scripts/external_eval/aggregate_block_b.py)

Generated V2 anchor artifacts are stored under `data/external_eval/` with
these filename families:

- `anchors_popular_object_top25_block_b_{dataset}.json`
- `anchors_rare_object_bottom25_block_b_{dataset}.json`
- `anchors_random_object_middle25_seed42_block_b_{dataset}.json`

## External Rehearsal Validation Progress (2026-07-27)

- MQuAKE-T was stopped at the preregistered B=25 preflight gate: the official
  release yielded 96 unique temporal updates, 18 strict old-known/new-unknown
  updates, and 15 entity-disjoint eligible updates.
- A fixed Qwen3.5-9B rehearsal core is now frozen under
  `data/external_eval/frozen_rehearsal_core/`:
  - 100 Popular, 100 Random, 100 Rare, and 100 distance-matched Random anchors;
  - all 400 anchors passed an independent strict clean-correct recheck;
  - anchor and probe hashes passed verification;
  - the frozen holdout bank contains 450 probes, with 442 independently
    rechecked as clean-correct.
- An expanded 512-update WikiBigEdit candidate pool was strictly prechecked.
  Three entity-disjoint B=25 batches were frozen under
  `data/external_eval/wbe_frozen_confirmation/`. Each batch contains 25 unique
  relations and has no entity overlap with the frozen anchors or probes.
- The complete confirmation matrix passed dry-run validation:
  `3 batches × 2 seeds × 5 arms = 30 LoRA runs`. The five arms are
  Update-only, Popular-100, Random-100, Rare-100, and distance-matched
  Random-100. No confirmation training has started yet.

## Final External Validation Plan and Second-Machine Handoff (2026-08-03)

### Paper question

The final mitigation experiment tests whether rehearsing structurally popular
facts reduces correct-to-wrong flips more than Update-only (`none`), Rare,
Random, and embedding-Similarity rehearsal. The primary forgetting metrics are:

- CounterFact: `neighborhood_old.flip_rate`;
- MQuAKE-CF: `unchanged_single_hop_old.flip_rate`;
- shared graph probes: Popular/Middle/Rare `flip_rate`.

CounterFact and MQuAKE-CF are the final external datasets because both use
counterfactual edits, require the base model to know the old answer, and expose
strong post-edit forgetting. WBE and WFD remain boundary analyses rather than
primary evidence.

### Final comparison matrix

- Models: `Qwen/Qwen3.5-9B` and `google/gemma-4-31B-it`.
- Datasets: CounterFact B=100 and MQuAKE-CF B=80.
- Modes: `none popular rare random similarity`.
- Seed: 42; epochs: 3; update repetitions: 20.
- Exact rehearsal ratio: 20% of update training samples.
  - CounterFact: 100 anchors × 4 / (100 updates × 20) = 20%.
  - MQuAKE-CF: 80 anchors × 4 / (80 updates × 20) = 20%.

Important: the first exploratory Qwen 9B MQuAKE run used 100 anchors at B=80,
which is 25%, not 20%. It is useful for an early signal but is not the final
ratio-matched run. The final MQuAKE comparison must set `ANCHOR_COUNT=80`.

### Current status

- Qwen 9B × CounterFact × five modes: complete.
- Qwen 9B × MQuAKE-CF exploratory 25% run: running.
- Gemma 31B × both final datasets: assigned to the second machine.
- Runtime outputs under `main_output/` are intentionally ignored by Git and
  must be copied back separately after completion.

### Required second-machine configuration

- Recommended GPU: one A100 80GB or H100 80GB.
- Conda environments:
  - `genfragility` for LLaMA-Factory/QLoRA training;
  - `ripple` for vLLM precheck and evaluation.
- Model cache: set `HF_HOME` to a disk with enough free space; the default used
  by the runner is `$HOME/huggingface_cache_large`.
- Gemma 31B training is automatically configured for 4-bit quantization,
  per-device batch size 1, and gradient accumulation 6.
- vLLM precheck/evaluation loads Gemma 31B and therefore requires an 80GB-class
  GPU.

The following repository assets must exist after checkout:

```text
data/external_eval/counterfact_confirmation/
data/external_eval/mquake_b100_confirmation/
data/external_eval/frozen_rehearsal_core/
run_gemma31b_precheck.sh
run_gemma31b_main.sh
run_mquake_main.sh
run_second_machine_gemma31b.sh
```

### Run on the second machine

After the experiment branch and MQuAKE assets have been pushed:

```bash
git fetch origin
git checkout feature/similarity-rehearsal-b100

export HF_HOME="$HOME/huggingface_cache_large"

# Validate all paths and training sample counts without loading a model.
bash run_second_machine_gemma31b.sh dry-run

# Run Gemma prechecks and all five modes on both datasets.
nohup bash run_second_machine_gemma31b.sh run \
  > gemma31b_second_machine.log 2>&1 &
```

For the faster three-arm comparison:

```bash
MODES="none popular random" \
nohup bash run_second_machine_gemma31b.sh run \
  > gemma31b_second_machine.log 2>&1 &
```

The runner skips completed native/probe evaluations, so rerunning the same
command resumes at the first incomplete mode. Final outputs are written under:

```text
main_output/external_rehearsal/counterfact_gemma31b/
main_output/external_rehearsal/mquake_gemma31b/
```