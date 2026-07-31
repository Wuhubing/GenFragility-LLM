#!/usr/bin/env bash
# Plan or run the fixed-anchor WikiBigEdit confirmation matrix.

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
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-9B}
MANIFEST=data/external_eval/wbe_frozen_confirmation/wikibigedit/manifest.json
WFD_MANIFEST=data/external_eval/wbe_frozen_confirmation/wikifactdiff/manifest.json
WFD_EXPERIMENT_DIR=data/external_eval/block_b_experiments/wikifactdiff
ANCHOR_DIR=data/external_eval/frozen_rehearsal_core
PROBE_MANIFEST="$ANCHOR_DIR/probes/probe_bank.json"
OUTPUT_BASE=main_output/external_rehearsal/wbe_frozen_confirmation
PRECHECK_REPORT="$OUTPUT_BASE/final_precheck.json"
RUN_MANIFEST="$OUTPUT_BASE/${ACTION}_manifest.tsv"

for required in \
    "$MANIFEST" \
    "$WFD_MANIFEST" \
    "$PROBE_MANIFEST" \
    "$ANCHOR_DIR/frozen_verification.md" \
    "$ANCHOR_DIR/probe_verification.md"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing frozen experiment asset: $required"
        exit 1
    fi
done

python -c "
from pathlib import Path
for path in (
    Path('$ANCHOR_DIR/frozen_verification.md'),
    Path('$ANCHOR_DIR/probe_verification.md'),
):
    if 'Status: PASS' not in path.read_text():
        raise SystemExit(f'Frozen asset did not pass: {path}')
"

mkdir -p "$OUTPUT_BASE"
export PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-$HOME/hf_cache_home}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.85}
export VLLM_MAX_SEQS=${VLLM_MAX_SEQS:-128}
export LF_BATCH_SIZE=${LF_BATCH_SIZE:-2}
export LF_GRAD_ACCUM=${LF_GRAD_ACCUM:-4}
export DISABLE_VERSION_CHECK=${DISABLE_VERSION_CHECK:-1}

if [[ "$ACTION" == "run" && ! -f "$PRECHECK_REPORT" ]]; then
    "$CONDA" run -n "$EVAL_ENV" python src/vllm_rehearsal_smoke_eval.py \
        --stage precheck \
        --base-model "$BASE_MODEL" \
        --wfd-manifest "$WFD_MANIFEST" \
        --wfd-experiment-dir "$WFD_EXPERIMENT_DIR" \
        --wbe-manifest "$MANIFEST" \
        --output "$PRECHECK_REPORT"
fi
mapfile -t UNITS < <(
    python -c "
import json
from pathlib import Path
print(*json.loads(Path('$MANIFEST').read_text())['units'], sep='\n')
"
)
if [[ "${#UNITS[@]}" != "3" ]]; then
    echo "Expected 3 WikiBigEdit batches, got ${#UNITS[@]}"
    exit 1
fi

MODES=(none popular random rare random_distance)
SEEDS=(42 43)
printf 'dataset\tunit\tseed\tmode\tstatus\n' > "$RUN_MANIFEST"
planned=0
for seed in "${SEEDS[@]}"; do
    for unit_id in "${UNITS[@]}"; do
        for mode in "${MODES[@]}"; do
            out_dir="$OUTPUT_BASE/seed${seed}/wikibigedit/$mode/$unit_id"
            train_args=(
                --manifest "$MANIFEST"
                --unit-id "$unit_id"
                --mode "$mode"
                --base-model "$BASE_MODEL"
                --output-dir "$out_dir"
                --precheck-report "$PRECHECK_REPORT"
                --frozen-anchor-dir "$ANCHOR_DIR"
                --anchor-count 100
                --seed "$seed"
                --repeats-per-update 20
                --epochs 3
            )
            if [[ "$ACTION" == "dry-run" ]]; then
                "$CONDA" run -n "$TRAIN_ENV" python \
                    scripts/train_wikibigedit_rehearsal_smoke.py \
                    "${train_args[@]}" --dry-run
                printf 'wikibigedit\t%s\t%s\t%s\tplanned\n' \
                    "$unit_id" "$seed" "$mode" >> "$RUN_MANIFEST"
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
                    --stage evaluate-wbe \
                    --base-model "$BASE_MODEL" \
                    --wbe-manifest "$MANIFEST" \
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
            printf 'wikibigedit\t%s\t%s\t%s\tcomplete\n' \
                "$unit_id" "$seed" "$mode" >> "$RUN_MANIFEST"
            planned=$((planned + 1))
        done
    done
done
if [[ "$planned" != "30" ]]; then
    echo "Expected 30 planned runs, got $planned"
    exit 1
fi
echo "PASS: planned $planned fixed-anchor WikiBigEdit runs"
echo "Run manifest: $RUN_MANIFEST"
if [[ "$ACTION" == "run" ]]; then
    python scripts/summarize_rehearsal_smoke.py \
        --output-base "$OUTPUT_BASE" \
        --graph-probe \
        --include-seed-subdirs \
        --modes none,popular,random,rare,random_distance
fi
