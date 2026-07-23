#!/usr/bin/env bash
# Run the fixed WikiFactDiff and WikiBigEdit rehearsal smoke experiments.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONDA=${CONDA:-"$HOME/miniconda3/bin/conda"}
TRAIN_ENV=${TRAIN_ENV:-genfragility}
EVAL_ENV=${EVAL_ENV:-ripple}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-9B}
OUTPUT_BASE=${OUTPUT_BASE:-main_output/rehearsal_smoke/qwen3.5-9b}
WFD_MANIFEST=data/external_eval/rehearsal_smoke/wikifactdiff/manifest.json
WBE_MANIFEST=data/external_eval/rehearsal_smoke/wikibigedit/manifest.json
WFD_EXPERIMENT_DIR=data/external_eval/block_b_experiments/wikifactdiff
PRECHECK_REPORT="$OUTPUT_BASE/precheck.json"
RUN_MANIFEST="$OUTPUT_BASE/run_manifest.tsv"
DATASET=all
DRY_RUN=0
SKIP_PRECHECK=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET=$2; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --skip-precheck) SKIP_PRECHECK=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "$DATASET" != "all" && "$DATASET" != "wikifactdiff" && "$DATASET" != "wikibigedit" ]]; then
    echo "--dataset must be all, wikifactdiff, or wikibigedit"
    exit 1
fi

export DISABLE_VERSION_CHECK=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-$HOME/huggingface_cache_large}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.85}
export VLLM_MAX_SEQS=${VLLM_MAX_SEQS:-128}
export LF_BATCH_SIZE=${LF_BATCH_SIZE:-2}
export LF_GRAD_ACCUM=${LF_GRAD_ACCUM:-4}

SAFE_CACHE=${GENFRAG_SAFE_CACHE:-$HOME/.genfrag_cache}
mkdir -p "$OUTPUT_BASE" "$SAFE_CACHE/vllm" "$SAFE_CACHE/triton" \
    "$SAFE_CACHE/tmp" "$SAFE_CACHE/torchinductor"
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$SAFE_CACHE/vllm}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$SAFE_CACHE/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$SAFE_CACHE/torchinductor}
export TMPDIR=${TMPDIR:-$SAFE_CACHE/tmp}

for required in "$WFD_MANIFEST" "$WBE_MANIFEST"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing required file: $required"
        exit 1
    fi
done

printf 'dataset\tunit\tmode\tstatus\n' > "$RUN_MANIFEST"

if [[ "$DRY_RUN" == "0" && "$SKIP_PRECHECK" == "0" && ! -f "$PRECHECK_REPORT" ]]; then
    "$CONDA" run -n "$EVAL_ENV" python src/vllm_rehearsal_smoke_eval.py \
        --stage precheck \
        --base-model "$BASE_MODEL" \
        --wfd-manifest "$WFD_MANIFEST" \
        --wfd-experiment-dir "$WFD_EXPERIMENT_DIR" \
        --wbe-manifest "$WBE_MANIFEST" \
        --output "$PRECHECK_REPORT"
fi

if [[ "$DRY_RUN" == "0" && "$SKIP_PRECHECK" == "0" ]]; then
    "$CONDA" run -n "$TRAIN_ENV" python -c "
import json
from pathlib import Path
report = json.loads(Path('$PRECHECK_REPORT').read_text())
failed = [
    unit_id
    for unit_id, unit in report['units'].items()
    if unit['eligible_updates'] != unit['total_updates']
]
if failed:
    raise SystemExit(
        f'Precheck failed for {len(failed)} unit(s); inspect $PRECHECK_REPORT'
    )
print('Precheck gate passed for all smoke units')
"
fi

MODES=(none popular rare random)

if [[ "$DATASET" == "all" || "$DATASET" == "wikifactdiff" ]]; then
    mapfile -t WFD_UNITS < <(
        "$CONDA" run -n "$TRAIN_ENV" python -c "
import json
from pathlib import Path
print(*json.loads(Path('$WFD_MANIFEST').read_text())['units'], sep='\n')
"
    )
    for mode in "${MODES[@]}"; do
        neutral=25
        anchor_args=()
        case "$mode" in
            none)
                neutral=0
                anchor_args=(--anchor_mode none)
                ;;
            popular)
                anchor_args=(
                    --anchor_mode popular
                    --anchor_file_override
                    data/external_eval/rehearsal_smoke/wikifactdiff/anchors_popular_object_top25.json
                )
                ;;
            rare)
                anchor_args=(
                    --anchor_mode rare
                    --anchor_file_override
                    data/external_eval/rehearsal_smoke/wikifactdiff/anchors_rare_object_bottom25.json
                )
                ;;
            random)
                anchor_args=(
                    --anchor_mode random_rehearsal
                    --anchor_file_override
                    data/external_eval/rehearsal_smoke/wikifactdiff/anchors_random_object_middle25_seed42.json
                )
                ;;
        esac

        for unit_id in "${WFD_UNITS[@]}"; do
            experiment_file="$WFD_EXPERIMENT_DIR/$unit_id.json"
            out_dir="$OUTPUT_BASE/wikifactdiff/$mode/$unit_id"
            if [[ "$DRY_RUN" == "1" ]]; then
                echo "DRY-RUN wikifactdiff unit=$unit_id mode=$mode neutral=$neutral"
                printf 'wikifactdiff\t%s\t%s\tplanned\n' "$unit_id" "$mode" >> "$RUN_MANIFEST"
                continue
            fi

            lora_path=$(
                "$CONDA" run -n "$TRAIN_ENV" python -c "
from pathlib import Path
matches = sorted(Path('$out_dir').glob(
    '${unit_id}_*/models/integrated_poison*/adapter_config.json'
))
print(matches[0].parent if matches else '')
"
            )
            if [[ -z "$lora_path" ]]; then
                LF_BATCH_SIZE="$LF_BATCH_SIZE" LF_GRAD_ACCUM="$LF_GRAD_ACCUM" \
                    "$CONDA" run -n "$TRAIN_ENV" python main.py \
                        --mode single \
                        --base_model "$BASE_MODEL" \
                        --experiment_file "$experiment_file" \
                        --output_dir "$out_dir" \
                        --num_poison 150 \
                        --num_neutral "$neutral" \
                        --num_irrelevant 0 \
                        --poison_strategy balanced \
                        "${anchor_args[@]}" \
                        --epochs 3 \
                        --run_poison_pipeline \
                        --skip_hf_eval
                lora_path=$(
                    "$CONDA" run -n "$TRAIN_ENV" python -c "
from pathlib import Path
matches = sorted(Path('$out_dir').glob(
    '${unit_id}_*/models/integrated_poison*/adapter_config.json'
))
print(matches[0].parent if matches else '')
"
                )
            fi
            if [[ -z "$lora_path" ]]; then
                echo "LoRA missing after training: wikifactdiff/$mode/$unit_id"
                exit 1
            fi

            report="$out_dir/evaluation.json"
            if [[ ! -f "$report" ]]; then
                "$CONDA" run -n "$EVAL_ENV" python src/vllm_rehearsal_smoke_eval.py \
                    --stage evaluate-wfd \
                    --base-model "$BASE_MODEL" \
                    --experiment-file "$experiment_file" \
                    --lora-path "$lora_path" \
                    --output "$report"
            fi
            printf 'wikifactdiff\t%s\t%s\tcomplete\n' "$unit_id" "$mode" >> "$RUN_MANIFEST"
        done
    done
fi

if [[ "$DATASET" == "all" || "$DATASET" == "wikibigedit" ]]; then
    WBE_UNIT=$(
        "$CONDA" run -n "$TRAIN_ENV" python -c "
import json
from pathlib import Path
print(next(iter(json.loads(Path('$WBE_MANIFEST').read_text())['units'])))
"
    )
    for mode in "${MODES[@]}"; do
        out_dir="$OUTPUT_BASE/wikibigedit/$mode/$WBE_UNIT"
        train_args=(
            --manifest "$WBE_MANIFEST"
            --unit-id "$WBE_UNIT"
            --mode "$mode"
            --base-model "$BASE_MODEL"
            --output-dir "$out_dir"
            --precheck-report "$PRECHECK_REPORT"
            --repeats-per-update 20
            --epochs 3
        )
        if [[ "$DRY_RUN" == "1" ]]; then
            "$CONDA" run -n "$TRAIN_ENV" python \
                scripts/train_wikibigedit_rehearsal_smoke.py \
                "${train_args[@]}" --dry-run
            printf 'wikibigedit\t%s\t%s\tplanned\n' "$WBE_UNIT" "$mode" >> "$RUN_MANIFEST"
            continue
        fi

        "$CONDA" run -n "$TRAIN_ENV" python \
            scripts/train_wikibigedit_rehearsal_smoke.py \
            "${train_args[@]}"
        lora_path="$out_dir/adapter"
        report="$out_dir/evaluation.json"
        if [[ ! -f "$report" ]]; then
            "$CONDA" run -n "$EVAL_ENV" python src/vllm_rehearsal_smoke_eval.py \
                --stage evaluate-wbe \
                --base-model "$BASE_MODEL" \
                --wbe-manifest "$WBE_MANIFEST" \
                --unit-id "$WBE_UNIT" \
                --lora-path "$lora_path" \
                --output "$report"
        fi
        printf 'wikibigedit\t%s\t%s\tcomplete\n' "$WBE_UNIT" "$mode" >> "$RUN_MANIFEST"
    done
fi

if [[ "$DRY_RUN" == "0" && "$DATASET" == "all" ]]; then
    "$CONDA" run -n "$TRAIN_ENV" python scripts/summarize_rehearsal_smoke.py \
        --output-base "$OUTPUT_BASE"
fi

echo "Run manifest: $RUN_MANIFEST"
if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run complete; no model was loaded and no training was started."
else
    echo "Rehearsal smoke run complete."
fi
