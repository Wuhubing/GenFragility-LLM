#!/usr/bin/env bash
# Gemma-4-31B-it main method comparison at B=100.
# Runs all 5 modes (none/popular/rare/random/similarity) on CounterFact.
# Uses per-batch anchors (NOT frozen core), same as Qwen 9B B=100 setting.
#
# Prerequisite: run_gemma31b_precheck.sh must have completed successfully.
#
# Usage:
#   bash run_gemma31b_main.sh dry-run   # validate
#   bash run_gemma31b_main.sh run        # execute
#
# Environment overrides:
#   SEED=42 bash run_gemma31b_main.sh run
#   MODES="none popular similarity" bash run_gemma31b_main.sh run  # subset

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ACTION=${1:-dry-run}
if [[ "$ACTION" != "dry-run" && "$ACTION" != "run" ]]; then
    echo "Usage: $0 [dry-run|run]"
    exit 1
fi

CONDA=${CONDA:-"$HOME/miniconda3/bin/conda"}
TRAIN_ENV=${TRAIN_ENV:-genfragility}
EVAL_ENV=${EVAL_ENV:-ripple}
BASE_MODEL=${BASE_MODEL:-google/gemma-4-31B-it}
MANIFEST=data/external_eval/counterfact_confirmation/manifest.json
EXPERIMENT_DIR=data/external_eval/counterfact_confirmation/experiments
ANCHOR_DIR=data/external_eval/frozen_rehearsal_core
PROBE_MANIFEST="$ANCHOR_DIR/probes/probe_bank.json"
PRECHECK_REPORT=main_output/external_rehearsal/counterfact_gemma31b/precheck_b100.json
OUTPUT_BASE=main_output/external_rehearsal/counterfact_gemma31b

SEED=${SEED:-42}
# Default: all 5 modes. Override with MODES env var for subset.
if [[ -n "${MODES:-}" ]]; then
    read -ra MODES_ARR <<< "$MODES"
else
    MODES_ARR=(none popular rare random similarity)
fi

for required in \
    "$MANIFEST" \
    "$PROBE_MANIFEST" \
    "$ANCHOR_DIR/frozen_verification.md" \
    "$ANCHOR_DIR/probe_verification.md"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing asset: $required"
        exit 1
    fi
done
if [[ "$ACTION" == "run" && ! -f "$PRECHECK_REPORT" ]]; then
    echo "Missing precheck: $PRECHECK_REPORT"
    echo "Run bash run_gemma31b_precheck.sh first."
    exit 1
fi

export PYTHONPATH="$ROOT/src:$ROOT/scripts/external_eval:$ROOT:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-$HOME/huggingface_cache_large}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.90}
export VLLM_MAX_SEQS=${VLLM_MAX_SEQS:-32}
# Let the training script auto-detect 31B and set batch=1, grad_accum=6, quantization_bit=4
unset LF_BATCH_SIZE LF_GRAD_ACCUM
export DISABLE_VERSION_CHECK=${DISABLE_VERSION_CHECK:-1}

mapfile -t UNITS < <(
    python -c "
import json
from pathlib import Path
print(*json.loads(Path('$MANIFEST').read_text())['units'], sep='\n')
"
)

RUN_MANIFEST_TSV="$OUTPUT_BASE/seed${SEED}/${ACTION}_manifest.tsv"
mkdir -p "$OUTPUT_BASE/seed${SEED}"
printf 'dataset\tunit\tseed\tmode\tstatus\n' > "$RUN_MANIFEST_TSV"
planned=0
for unit_id in "${UNITS[@]}"; do
    for mode in "${MODES_ARR[@]}"; do
        out_dir="$OUTPUT_BASE/seed${SEED}/counterfact/$mode/$unit_id"
        train_args=(
            --manifest "$MANIFEST"
            --unit-id "$unit_id"
            --mode "$mode"
            --base-model "$BASE_MODEL"
            --output-dir "$out_dir"
            --precheck-report "$PRECHECK_REPORT"
            --anchor-count 100
            --anchor-seed 42
            --seed "$SEED"
            --repeats-per-update 20
            --epochs 3
        )
        if [[ "$ACTION" == "dry-run" ]]; then
            "$CONDA" run -n "$TRAIN_ENV" python \
                scripts/train_wikibigedit_rehearsal_smoke.py \
                "${train_args[@]}" --dry-run
            printf 'counterfact\t%s\t%s\t%s\tplanned\n' \
                "$unit_id" "$SEED" "$mode" >> "$RUN_MANIFEST_TSV"
            planned=$((planned + 1))
            continue
        fi

        "$CONDA" run -n "$TRAIN_ENV" python \
            scripts/train_wikibigedit_rehearsal_smoke.py \
            "${train_args[@]}"
        lora_path="$out_dir/adapter"
        native_report="$out_dir/evaluation_strict.json"
        probe_report="$out_dir/graph_probe_evaluation.json"
        if [[ ! -f "$native_report" ]]; then
            set +e
            "$CONDA" run -n "$EVAL_ENV" python \
                src/vllm_rehearsal_smoke_eval.py \
                --stage evaluate-counterfact \
                --base-model "$BASE_MODEL" \
                --counterfact-manifest "$MANIFEST" \
                --counterfact-experiment-dir "$EXPERIMENT_DIR" \
                --unit-id "$unit_id" \
                --lora-path "$lora_path" \
                --output "$native_report"
            status=$?
            set -e
            if [[ "$status" != "0" && ! -f "$native_report" ]]; then
                exit "$status"
            fi
        fi
        if [[ ! -f "$probe_report" ]]; then
            set +e
            "$CONDA" run -n "$EVAL_ENV" python \
                src/vllm_rehearsal_smoke_eval.py \
                --stage evaluate-probes \
                --base-model "$BASE_MODEL" \
                --probe-manifest "$PROBE_MANIFEST" \
                --lora-path "$lora_path" \
                --output "$probe_report"
            status=$?
            set -e
            if [[ "$status" != "0" && ! -f "$probe_report" ]]; then
                exit "$status"
            fi
        fi
        printf 'counterfact\t%s\t%s\t%s\tcomplete\n' \
            "$unit_id" "$SEED" "$mode" >> "$RUN_MANIFEST_TSV"
        planned=$((planned + 1))
    done
done

echo "PASS: completed/planned $planned Gemma-31B runs (seed=$SEED)"
echo "Run manifest: $RUN_MANIFEST_TSV"
