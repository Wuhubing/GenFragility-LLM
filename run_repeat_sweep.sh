#!/usr/bin/env bash
# Anchor repeat sweep: fix anchor set at 100, sweep anchor_repeats in {4,8,12,16,20}.
# Ratios: 20%/40%/60%/80%/100% (anchor_samples / update_samples).
# Modes: none, popular, similarity (3 key modes).
# Effective: none(3) + popular(15) + similarity(15) = 33 runs per seed.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ACTION=${1:-dry-run}
SEED=${2:-42}

CONDA=${CONDA:-"$HOME/miniconda3/bin/conda"}
TRAIN_ENV=${TRAIN_ENV:-genfragility}
EVAL_ENV=${EVAL_ENV:-ripple}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-9B}
MANIFEST=data/external_eval/counterfact_confirmation/manifest.json
EXPERIMENT_DIR=data/external_eval/counterfact_confirmation/experiments
ANCHOR_DIR=data/external_eval/frozen_rehearsal_core
PROBE_MANIFEST="$ANCHOR_DIR/probes/probe_bank.json"
PRECHECK_REPORT=main_output/external_rehearsal/counterfact_confirmation/final_precheck_b100.json
OUTPUT_BASE=main_output/external_rehearsal/counterfact_repeat_sweep

MODES=(none popular similarity)
REPEATS=(4 8 12 16 20)
# Effective runs: none(2) + popular(10) + similarity(10) = 22
EXPECTED_RUNS=22

RUN_MANIFEST_TSV="$OUTPUT_BASE/seed${SEED}/${ACTION}_manifest.tsv"

for required in \
    "$MANIFEST" \
    "$PRECHECK_REPORT" \
    "$PROBE_MANIFEST" \
    "$ANCHOR_DIR/frozen_verification.md" \
    "$ANCHOR_DIR/probe_verification.md"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing asset: $required"
        exit 1
    fi
done

mkdir -p "$OUTPUT_BASE/seed${SEED}"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/external_eval:$ROOT:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-$HOME/huggingface_cache_large}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
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
units = list(json.loads(Path('$MANIFEST').read_text())['units'])
# Only use first 2 batches for faster results
units = units[:2]
print(*units, sep='\n')
"
)

printf 'dataset\tunit\tseed\tmode\tanchor_repeats\tstatus\n' > "$RUN_MANIFEST_TSV"
planned=0
for unit_id in "${UNITS[@]}"; do
    for mode in "${MODES[@]}"; do
        for ar in "${REPEATS[@]}"; do
            # none mode has no anchors; anchor_repeats is irrelevant. Skip duplicates.
            if [[ "$mode" == "none" && "$ar" != "4" ]]; then
                continue
            fi
            out_dir="$OUTPUT_BASE/seed${SEED}/ar${ar}/${mode}/${unit_id}"
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
                --anchor-repeats "$ar"
                --epochs 3
            )
            if [[ "$ACTION" == "dry-run" ]]; then
                "$CONDA" run -n "$TRAIN_ENV" python \
                    scripts/train_wikibigedit_rehearsal_smoke.py \
                    "${train_args[@]}" --dry-run
                printf 'counterfact\t%s\t%s\t%s\t%s\tplanned\n' \
                    "$unit_id" "$SEED" "$mode" "$ar" >> "$RUN_MANIFEST_TSV"
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
            printf 'counterfact\t%s\t%s\t%s\t%s\tcomplete\n' \
                "$unit_id" "$SEED" "$mode" "$ar" >> "$RUN_MANIFEST_TSV"
            planned=$((planned + 1))
        done
    done
done

if [[ "$planned" != "$EXPECTED_RUNS" ]]; then
    echo "Expected $EXPECTED_RUNS runs, got $planned"
    exit 1
fi
echo "PASS: completed/planned $planned repeat-sweep runs (seed=$SEED)"
echo "Run manifest: $RUN_MANIFEST_TSV"
