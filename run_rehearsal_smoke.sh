#!/usr/bin/env bash
# Run the fixed WikiFactDiff and WikiBigEdit rehearsal smoke experiments.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONDA=${CONDA:-"$HOME/miniconda3/bin/conda"}
TRAIN_ENV=${TRAIN_ENV:-genfragility}
EVAL_ENV=${EVAL_ENV:-ripple}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-9B}
OUTPUT_BASE=${OUTPUT_BASE:-main_output/rehearsal_batch_smoke/qwen3.5-9b-anchor100}
ANCHOR_COUNT=${ANCHOR_COUNT:-100}
ANCHOR_SEED=${ANCHOR_SEED:-42}
WFD_MANIFEST=${WFD_MANIFEST:-data/external_eval/rehearsal_smoke/wikifactdiff/manifest.json}
WBE_MANIFEST=${WBE_MANIFEST:-data/external_eval/rehearsal_smoke/wikibigedit/manifest.json}
PROBE_MANIFEST=${PROBE_MANIFEST:-}
WFD_EXPERIMENT_DIR=data/external_eval/block_b_experiments/wikifactdiff
PRECHECK_REPORT=${PRECHECK_REPORT:-"$OUTPUT_BASE/precheck_strict.json"}
RUN_MANIFEST="$OUTPUT_BASE/run_manifest.tsv"
DATASET=all
DRY_RUN=0
SKIP_PRECHECK=0
GRAPH_PROBE=0
ALL_WBE_UNITS=0
ALL_WFD_UNITS=0
MODES_CSV=${MODES_CSV:-}
UNIT_LIMIT=${UNIT_LIMIT:-0}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET=$2; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --skip-precheck) SKIP_PRECHECK=1; shift ;;
        --probe-manifest) PROBE_MANIFEST=$2; GRAPH_PROBE=1; shift 2 ;;
        --all-wbe-units) ALL_WBE_UNITS=1; shift ;;
        --all-wfd-units) ALL_WFD_UNITS=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "$DATASET" != "all" && "$DATASET" != "wikifactdiff" && "$DATASET" != "wikibigedit" ]]; then
    echo "--dataset must be all, wikifactdiff, or wikibigedit"
    exit 1
fi

export DISABLE_VERSION_CHECK=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-$HOME/hf_cache_home}
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
if [[ "$GRAPH_PROBE" == "1" && ! -f "$PROBE_MANIFEST" ]]; then
    echo "Missing required probe manifest: $PROBE_MANIFEST"
    exit 1
fi

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
    if ('$DATASET' == 'all' or unit['dataset'] == '$DATASET')
    and unit['eligible_updates'] != unit['total_updates']
]
if failed:
    raise SystemExit(
        f'Precheck failed for {len(failed)} unit(s); inspect $PRECHECK_REPORT'
    )
print('Precheck gate passed for all smoke units')
"
fi

MODES=(none popular rare random)
if [[ "$GRAPH_PROBE" == "1" ]]; then
    MODES=(none popular rare random generic)
fi
if [[ -n "$MODES_CSV" ]]; then
    IFS=',' read -r -a MODES <<< "$MODES_CSV"
fi

if [[ "$DATASET" == "all" || "$DATASET" == "wikifactdiff" ]]; then
    mapfile -t WFD_UNITS < <(
        "$CONDA" run -n "$TRAIN_ENV" python -c "
import json
from pathlib import Path
units = list(json.loads(Path('$WFD_MANIFEST').read_text())['units'])
selected = units if $ALL_WFD_UNITS else units[:1]
print(*(selected[:$UNIT_LIMIT] if $UNIT_LIMIT else selected), sep='\n')
"
    )
    for WFD_UNIT in "${WFD_UNITS[@]}"; do
      for mode in "${MODES[@]}"; do
        out_dir="$OUTPUT_BASE/wikifactdiff/$mode/$WFD_UNIT"
        train_args=(
            --manifest "$WFD_MANIFEST"
            --unit-id "$WFD_UNIT"
            --mode "$mode"
            --base-model "$BASE_MODEL"
            --output-dir "$out_dir"
            --precheck-report "$PRECHECK_REPORT"
            --wfd-experiment-dir "$WFD_EXPERIMENT_DIR"
            --anchor-count "$ANCHOR_COUNT"
            --anchor-seed "$ANCHOR_SEED"
            --seed "$ANCHOR_SEED"
            --repeats-per-update 20
            --epochs 3
        )
        if [[ "$DRY_RUN" == "1" ]]; then
            "$CONDA" run -n "$TRAIN_ENV" python \
                scripts/train_wikibigedit_rehearsal_smoke.py \
                "${train_args[@]}" --dry-run
            printf 'wikifactdiff\t%s\t%s\tplanned\n' "$WFD_UNIT" "$mode" >> "$RUN_MANIFEST"
            continue
        fi

        "$CONDA" run -n "$TRAIN_ENV" python \
            scripts/train_wikibigedit_rehearsal_smoke.py \
            "${train_args[@]}"
        lora_path="$out_dir/adapter"
        report="$out_dir/evaluation_strict.json"
        if [[ ! -f "$report" ]]; then
            set +e
            "$CONDA" run -n "$EVAL_ENV" python src/vllm_rehearsal_smoke_eval.py \
                --stage evaluate-wfd \
                --base-model "$BASE_MODEL" \
                --wfd-manifest "$WFD_MANIFEST" \
                --wfd-experiment-dir "$WFD_EXPERIMENT_DIR" \
                --unit-id "$WFD_UNIT" \
                --lora-path "$lora_path" \
                --output "$report"
            eval_status=$?
            set -e
            if [[ "$eval_status" != "0" && ! -f "$report" ]]; then
                exit "$eval_status"
            fi
        fi
        printf 'wikifactdiff\t%s\t%s\tcomplete\n' "$WFD_UNIT" "$mode" >> "$RUN_MANIFEST"
      done
    done
fi

if [[ "$DATASET" == "all" || "$DATASET" == "wikibigedit" ]]; then
    mapfile -t WBE_UNITS < <(
        "$CONDA" run -n "$TRAIN_ENV" python -c "
import json
from pathlib import Path
units = list(json.loads(Path('$WBE_MANIFEST').read_text())['units'])
selected = units if $ALL_WBE_UNITS else units[:1]
print(*(selected[:$UNIT_LIMIT] if $UNIT_LIMIT else selected), sep='\n')
"
    )
    for WBE_UNIT in "${WBE_UNITS[@]}"; do
        for mode in "${MODES[@]}"; do
            out_dir="$OUTPUT_BASE/wikibigedit/$mode/$WBE_UNIT"
            train_args=(
                --manifest "$WBE_MANIFEST"
                --unit-id "$WBE_UNIT"
                --mode "$mode"
                --base-model "$BASE_MODEL"
                --output-dir "$out_dir"
                --precheck-report "$PRECHECK_REPORT"
                --anchor-count "$ANCHOR_COUNT"
                --anchor-seed "$ANCHOR_SEED"
                --seed "$ANCHOR_SEED"
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
            report="$out_dir/evaluation_strict.json"
            if [[ ! -f "$report" ]]; then
                set +e
                "$CONDA" run -n "$EVAL_ENV" python src/vllm_rehearsal_smoke_eval.py \
                    --stage evaluate-wbe \
                    --base-model "$BASE_MODEL" \
                    --wbe-manifest "$WBE_MANIFEST" \
                    --unit-id "$WBE_UNIT" \
                    --lora-path "$lora_path" \
                    --output "$report"
                eval_status=$?
                set -e
                if [[ "$eval_status" != "0" && ! -f "$report" ]]; then
                    exit "$eval_status"
                fi
            fi
            if [[ "$GRAPH_PROBE" == "1" ]]; then
                probe_report="$out_dir/graph_probe_evaluation.json"
                if [[ ! -f "$probe_report" ]]; then
                    set +e
                    "$CONDA" run -n "$EVAL_ENV" python src/vllm_rehearsal_smoke_eval.py \
                        --stage evaluate-probes \
                        --base-model "$BASE_MODEL" \
                        --probe-manifest "$PROBE_MANIFEST" \
                        --lora-path "$lora_path" \
                        --output "$probe_report"
                    eval_status=$?
                    set -e
                    if [[ "$eval_status" != "0" && ! -f "$probe_report" ]]; then
                        exit "$eval_status"
                    fi
                fi
            fi
            printf 'wikibigedit\t%s\t%s\tcomplete\n' "$WBE_UNIT" "$mode" >> "$RUN_MANIFEST"
        done
    done
fi

if [[ "$DRY_RUN" == "0" && "$GRAPH_PROBE" == "1" ]]; then
    "$CONDA" run -n "$TRAIN_ENV" python scripts/summarize_rehearsal_smoke.py \
        --output-base "$OUTPUT_BASE" \
        --graph-probe \
        --modes "$(IFS=,; echo "${MODES[*]}")"
elif [[ "$DRY_RUN" == "0" && "$DATASET" == "all" ]]; then
    "$CONDA" run -n "$TRAIN_ENV" python scripts/summarize_rehearsal_smoke.py \
        --output-base "$OUTPUT_BASE"
fi

echo "Run manifest: $RUN_MANIFEST"
if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run complete; no model was loaded and no training was started."
else
    echo "Rehearsal smoke run complete."
fi
