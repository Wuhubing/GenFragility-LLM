#!/usr/bin/env bash
# Plan or run the frozen full-WikiFactDiff confirmation matrix.

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
MANIFEST=data/external_eval/wfd_full_confirmation/manifest.json
EXPERIMENT_DIR=data/external_eval/wfd_full_confirmation/experiments
ANCHOR_DIR=data/external_eval/frozen_rehearsal_core
PROBE_MANIFEST="$ANCHOR_DIR/probes/probe_bank.json"
PRECHECK_REPORT=main_output/external_rehearsal/wfd_full_confirmation/final_precheck_round3.json
OUTPUT_BASE=main_output/external_rehearsal/wfd_full_confirmation
RUN_MANIFEST="$OUTPUT_BASE/${ACTION}_manifest.tsv"

for required in \
    "$MANIFEST" \
    "$PRECHECK_REPORT" \
    "$PROBE_MANIFEST" \
    "$ANCHOR_DIR/frozen_verification.md" \
    "$ANCHOR_DIR/probe_verification.md"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing frozen experiment asset: $required"
        exit 1
    fi
done

python -c "
import json
from pathlib import Path
manifest = json.loads(Path('$MANIFEST').read_text())
if manifest['metadata'].get('status') != 'frozen':
    raise SystemExit('WikiFactDiff manifest is not frozen')
report = json.loads(Path('$PRECHECK_REPORT').read_text())
eligibility = [
    passed
    for unit in report['units'].values()
    for passed in unit['eligibility'].values()
]
if len(eligibility) != 75 or not all(eligibility):
    raise SystemExit(
        f'Final WikiFactDiff precheck is not 75/75: {sum(eligibility)}/{len(eligibility)}'
    )
for path in (
    Path('$ANCHOR_DIR/frozen_verification.md'),
    Path('$ANCHOR_DIR/probe_verification.md'),
):
    if 'Status: PASS' not in path.read_text():
        raise SystemExit(f'Frozen asset did not pass: {path}')
"

mkdir -p "$OUTPUT_BASE"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/external_eval:$ROOT:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-$HOME/huggingface_cache_large}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.85}
export VLLM_MAX_SEQS=${VLLM_MAX_SEQS:-128}
export LF_BATCH_SIZE=${LF_BATCH_SIZE:-2}
export LF_GRAD_ACCUM=${LF_GRAD_ACCUM:-4}
export DISABLE_VERSION_CHECK=${DISABLE_VERSION_CHECK:-1}

mapfile -t UNITS < <(
    python -c "
import json
from pathlib import Path
print(*json.loads(Path('$MANIFEST').read_text())['units'], sep='\n')
"
)
if [[ "${#UNITS[@]}" != "3" ]]; then
    echo "Expected 3 WikiFactDiff batches, got ${#UNITS[@]}"
    exit 1
fi

MODES=(none popular random rare random_distance)
SEEDS=(42 43)
printf 'dataset\tunit\tseed\tmode\tstatus\n' > "$RUN_MANIFEST"
planned=0
for seed in "${SEEDS[@]}"; do
    for unit_id in "${UNITS[@]}"; do
        for mode in "${MODES[@]}"; do
            out_dir="$OUTPUT_BASE/seed${seed}/wikifactdiff/$mode/$unit_id"
            train_args=(
                --manifest "$MANIFEST"
                --unit-id "$unit_id"
                --mode "$mode"
                --base-model "$BASE_MODEL"
                --output-dir "$out_dir"
                --precheck-report "$PRECHECK_REPORT"
                --wfd-experiment-dir "$EXPERIMENT_DIR"
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
                printf 'wikifactdiff\t%s\t%s\t%s\tplanned\n' \
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
                    --stage evaluate-wfd \
                    --base-model "$BASE_MODEL" \
                    --wfd-manifest "$MANIFEST" \
                    --wfd-experiment-dir "$EXPERIMENT_DIR" \
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
            printf 'wikifactdiff\t%s\t%s\t%s\tcomplete\n' \
                "$unit_id" "$seed" "$mode" >> "$RUN_MANIFEST"
            planned=$((planned + 1))
        done
    done
done

if [[ "$planned" != "30" ]]; then
    echo "Expected 30 planned runs, got $planned"
    exit 1
fi
echo "PASS: planned $planned fixed-anchor WikiFactDiff runs"
echo "Run manifest: $RUN_MANIFEST"
if [[ "$ACTION" == "run" ]]; then
    python scripts/external_eval/summarize_wfd_full_confirmation.py \
        --output-base "$OUTPUT_BASE"
fi
