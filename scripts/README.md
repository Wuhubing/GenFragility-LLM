# Script Entry Index

This file is a quick command index for commonly used scripts.

## Core Pipelines

- Run integrated poisoning pipeline:
  - `python3 main.py --mode single --experiment_number 6 --run_poison_pipeline --base_model meta-llama/Llama-2-7b-hf --poison_method factual --epochs 5`
- Run ACL experiment suite:
  - `python3 run_acl_experiments.py`
- Build 1-to-1 graph (fast mode):
  - `python3 run_1to1_fast.py`
- Build graph "database" checkpoint (`latest.pkl`) and outputs:
  - `python3 run_1to1_fast.py`

## Data Tools (`tools/data`)

- Download and list model artifacts:
  - `python3 tools/data/download_and_list.py`
- Download models:
  - `python3 tools/data/download_models.py`
- Generate node degree metadata:
  - `python3 tools/data/generate_degrees.py`

## Report Tools (`tools/report`)

- Generate paper result bundle:
  - `python3 tools/report/generate_paper_results.py`
- Generate LaTeX analysis tables:
  - `python3 tools/report/generate_latex.py`
- Generate final summary table:
  - `python3 tools/report/generate_final_table.py`
- Detect ripple effect from a comparison report:
  - `python3 tools/report/detect_ripple_effect.py --report <comparison_report.json> --out ripple_metrics.json`
- One-step diagnosis (ripple metrics + clean accuracy by distance in one file):
  - `make diagnose`
- Check clean baseline accuracy by distance from report:
  - `python3 - << 'PY'`
  - `import json`
  - `from collections import defaultdict`
  - `p='main_output/<exp_dir>/<sub_exp>/comparison_reports/<comparison_report>.json'`
  - `d=json.load(open(p)); u=d['unified_results']; by=defaultdict(list)`
  - `for r in u: by[r['distance']].append(r)`
  - `mean=lambda xs: sum(xs)/len(xs) if xs else 0`
  - `print('clean_acc_overall', round(mean([r.get('clean_accuracy',0) for r in u]),4))`
  - `for k in ['d0','d1','d2','d3','d4','d5']:`
  - `  if k in by: print(k, len(by[k]), round(mean([r.get('clean_accuracy',0) for r in by[k]]),4))`
  - `PY`

## Ripple Generation

- Generate ripple experiments from an existing graph checkpoint:
  - `python - << 'PY'`
  - `import src.generate_ripple_experiments as g`
  - `g.GRAPH_FILE='/root/GenFragility-LLM/latest.pkl'`
  - `g.OUTPUT_DIR='results/experiments_ripples_fast_20k'`
  - `g.NUM_EXPERIMENTS=15`
  - `g.MAX_DISTANCE=5`
  - `g.NUM_PROCESSES=4`
  - `g.main()`
  - `PY`

## Analysis and Debug

- Analysis scripts:
  - `tools/analysis/`
- Debug and inspection scripts:
  - `tools/debug/`
