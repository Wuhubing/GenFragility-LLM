# GenFragility-LLM

Knowledge poisoning and ripple-effect evaluation pipeline for LLMs.

## Current Architecture

- `main.py`: End-to-end pipeline (poison data -> LoRA train -> clean/poisoned evaluation -> comparison report).
- `run_1to1_fast.py`: Build a large 1-to-1 knowledge graph and output `latest.pkl` checkpoint.
- `src/generate_ripple_experiments.py`: Generate `ripple_experiment_*.json` from a graph checkpoint.
- `tools/report/detect_ripple_effect.py`: Post-hoc ripple metrics (C->W by distance, RippleScore).
- `tools/analysis/`: Analysis scripts.
- `tools/debug/`: Debug/inspection scripts.
- `tools/data/`: Download/data utility scripts.
- `tools/report/`: Report and table generation scripts.
- `results/experiments_ripples_fast_20k/`: Ripple experiment inputs.
- `main_output/`: Main experiment outputs (models, evaluation, comparison reports).
- `artifacts/logs/`: Large runtime logs.
- `artifacts/figures/`: Generated figures.

## Keys and Environment

Put keys in `keys/`:

- `keys/openai_key.txt`: OpenAI API key.
- `keys/hf_key.txt`: HuggingFace token (needed for gated models like `meta-llama/Llama-2-7b-hf`).

Recommended runtime interpreter (matches training/eval toolchain):

```bash
/root/miniconda3/envs/genfragility/bin/python -V
```

Install dependencies (first time):

```bash
python -m pip install -r requirements.txt
```

Export tokens for current shell:

```bash
export OPENAI_API_KEY="$(cat keys/openai_key.txt)"
export HF_TOKEN="$(cat keys/hf_key.txt)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HUGGINGFACEHUB_API_TOKEN="$HF_TOKEN"
export HF_HOME=/tmp/hf_cache
export TRANSFORMERS_CACHE=/tmp/hf_cache
mkdir -p /tmp/hf_cache
```

## Build Knowledge Graph Database (`latest.pkl`)

This project uses a graph checkpoint as its "knowledge database".

Build from scratch:

```bash
python3 run_1to1_fast.py
```

Default outputs:

- Checkpoint directory: `checkpoints/run_1to1_fast_20000/`
- Graph checkpoint: `checkpoints/run_1to1_fast_20000/latest.pkl`
- Exported graph files: `results/run_1to1_fast_20000/`

If you already have a checkpoint (for example `/root/GenFragility-LLM/latest.pkl`), you can directly use it to generate experiments.

## Generate Ripple Experiments from `latest.pkl`

Example: generate 15 experiments, up to distance d5 (default script behavior):

```bash
python - << 'PY'
import src.generate_ripple_experiments as g
g.GRAPH_FILE = '/root/GenFragility-LLM/latest.pkl'
g.OUTPUT_DIR = 'results/experiments_ripples_fast_20k'
g.NUM_EXPERIMENTS = 15
g.MAX_DISTANCE = 5
g.NUM_PROCESSES = 4
g.main()
PY
```

Output files:

- `results/experiments_ripples_fast_20k/ripple_experiment_001.json`
- `results/experiments_ripples_fast_20k/ripple_experiment_002.json`
- ...

## Run Experiments

Quick one-click entry points are also available via `Makefile`:

```bash
make build-graph
make gen-ripples
make run-exp003-d3
make detect
make diagnose
```

### Makefile Target Reference

- `make build-graph`
  - Build graph checkpoint and exports (graph "database").
- `make gen-ripples`
  - Generate `ripple_experiment_*.json` from `GRAPH_FILE`.
- `make run-single`
  - Run one experiment from `EXPERIMENT_FILE` with configurable runtime params.
- `make run-exp003-d3`
  - Shortcut of `run-single` for `ripple_experiment_003.json` + `d3`.
- `make detect`
  - Run ripple detector and output `ripple_metrics_v2.json` from a report.
- `make diagnose`
  - One-step summary: ripple metrics + clean accuracy by distance in one JSON.
- `make strict30-build`
  - Build Relaxed-Front-30 suite (`001-007`) definitions, sampled inputs, and initial manifest.
- `make strict30-audit`
  - Audit Relaxed-Front-30 rerun gates and emit a rerun shell script.
- `make strict30-figures`
  - Regenerate Mask-B storyline figures into `report/figures`.

### Makefile Variables (Override Examples)

```bash
make gen-ripples GRAPH_FILE=/root/GenFragility-LLM/latest.pkl NUM_EXPERIMENTS=3 MAX_DISTANCE=3 NUM_PROCESSES=2
make run-single EXPERIMENT_FILE=results/experiments_ripples_fast_20k/ripple_experiment_001.json RUN_MAX_DISTANCE=d2 CONCURRENCY=8
make detect REPORT=main_output/.../comparison_reports/xxx_comparison_yyy.json
make diagnose REPORT=main_output/.../comparison_reports/xxx_comparison_yyy.json DIAGNOSE_OUT=main_output/.../comparison_reports/diagnose_summary_custom.json
```

Frequently used variables:

- `PYTHON` (default: `/root/miniconda3/envs/genfragility/bin/python`)
- `BASE_MODEL` (default: `meta-llama/Llama-2-7b-hf`)
- `HF_TOKEN_FILE`, `OPENAI_KEY_FILE`, `HF_CACHE`
- `GRAPH_FILE`, `RIPPLE_OUTPUT_DIR`, `NUM_EXPERIMENTS`, `MAX_DISTANCE`, `NUM_PROCESSES`
- `EXPERIMENT_FILE`, `RUN_MAX_DISTANCE`, `EPOCHS`, `NUM_POISON`, `NUM_NEUTRAL`, `NUM_IRRELEVANT`, `CONCURRENCY`
- `REPORT`, `RIPPLE_METRICS_OUT`, `DIAGNOSE_OUT`
- `STRICT30_DIR`, `STRICT30_GRAPH`, `STRICT30_MAIN_OUTPUT`

## Relaxed-Front-30 Workflow (`001-007`)

Build suite:

```bash
make strict30-build STRICT30_GRAPH=/root/GenFragility-LLM/latest.pkl STRICT30_DIR=results/strict30_suite STRICT30_RELAXED_HOPS=d1,d2
```

Audit rerun gates:

```bash
make strict30-audit STRICT30_DIR=results/strict30_suite STRICT30_MAIN_OUTPUT=main_output
```

Generated artifacts:

- `results/strict30_suite/experiments/` (regenerated definitions)
- `results/strict30_suite/sampled/` (`d0=1`, `d3..d5=30`, `d1/d2=min(available,30)`, plus irrelevant-50)
- `results/strict30_suite/manifests/strict30_manifest_initial.json`
- `results/strict30_suite/manifests/strict30_manifest_audit.json`
- `results/strict30_suite/manifests/strict30_rerun_failed.sh`

### 1) Single experiment (recommended)

```bash
HF_TOKEN="$(cat keys/hf_key.txt)" \
HUGGING_FACE_HUB_TOKEN="$(cat keys/hf_key.txt)" \
HUGGINGFACEHUB_API_TOKEN="$(cat keys/hf_key.txt)" \
OPENAI_API_KEY="$(cat keys/openai_key.txt)" \
HF_HOME=/tmp/hf_cache \
TRANSFORMERS_CACHE=/tmp/hf_cache \
/root/miniconda3/envs/genfragility/bin/python main.py \
  --mode single \
  --experiment_file results/experiments_ripples_fast_20k/ripple_experiment_003.json \
  --run_poison_pipeline \
  --base_model meta-llama/Llama-2-7b-hf \
  --poison_method factual \
  --max_distance d3 \
  --epochs 1 \
  --num_poison 12 \
  --num_neutral 20 \
  --num_irrelevant 6 \
  --concurrency_limit 16
```

### 2) Evaluate only (已有LoRA)

```bash
HF_TOKEN="$(cat keys/hf_key.txt)" \
HUGGING_FACE_HUB_TOKEN="$(cat keys/hf_key.txt)" \
HUGGINGFACEHUB_API_TOKEN="$(cat keys/hf_key.txt)" \
OPENAI_API_KEY="$(cat keys/openai_key.txt)" \
HF_HOME=/tmp/hf_cache \
TRANSFORMERS_CACHE=/tmp/hf_cache \
/root/miniconda3/envs/genfragility/bin/python main.py \
  --mode single \
  --input_file results/experiments_ripples_fast_20k/ripple_experiment_003.json \
  --lora_path <LORA_PATH> \
  --base_model meta-llama/Llama-2-7b-hf \
  --max_distance d3
```

## Detect Ripple Effect

Given one comparison report:

```bash
python tools/report/detect_ripple_effect.py \
  --report main_output/<exp_dir>/<sub_exp>/comparison_reports/<comparison_report>.json \
  --out main_output/<exp_dir>/<sub_exp>/comparison_reports/ripple_metrics_v2.json
```

The script reports:

- `C->W Rate` by distance (`d0..d5`)
- `avg_accuracy_change`
- `avg_confidence_change`
- `RippleScore(d1-d5 weighted)`
- `RippleLevel` (`weak/moderate/strong`)

`make diagnose` additionally writes a combined JSON summary with:

- `report`: source comparison report path
- `total_samples`
- `ripple`: full ripple metrics payload
- `clean_accuracy_overall`
- `poisoned_accuracy_overall`
- `clean_accuracy_by_distance` (`d0..d5`, with `count`, `clean_accuracy_mean`, `poisoned_accuracy_mean`, `avg_confidence_change`)

## Check Whether Baseline Accuracy Is Intrinsically Low

Use the comparison report to inspect clean baseline accuracy by distance:

```bash
python - << 'PY'
import json
from collections import defaultdict
p='main_output/<exp_dir>/<sub_exp>/comparison_reports/<comparison_report>.json'
d=json.load(open(p))
u=d['unified_results']
by=defaultdict(list)
for r in u:
    by[r['distance']].append(r)
mean=lambda xs: sum(xs)/len(xs) if xs else 0
print('clean_acc_overall', round(mean([r.get('clean_accuracy',0) for r in u]),4))
for k in ['d0','d1','d2','d3','d4','d5']:
    if k in by:
        rows=by[k]
        print(k, len(rows), round(mean([r.get('clean_accuracy',0) for r in rows]),4))
PY
```

Interpretation:

- If clean accuracy at farther hops (`d2/d3/...`) is already low, observed extra ripple damage will be naturally bounded.

## Notes

- For gated models (`meta-llama/*`), missing HF token causes 401 errors.
- Keep training and evaluation on the same Python environment to avoid `peft`/`transformers` mismatch.
- Command index is also available at `scripts/README.md`.
