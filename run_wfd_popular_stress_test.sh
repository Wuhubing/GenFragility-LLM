#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"

ACTION=${1:-all}
if [[ "$ACTION" != "all" && "$ACTION" != "prepare" && "$ACTION" != "run" ]]; then
    echo "Usage: $0 [all|prepare|run]"
    exit 1
fi

CONDA=${CONDA:-"$HOME/miniconda3/bin/conda"}
TRAIN_ENV=${TRAIN_ENV:-genfragility}
EVAL_ENV=${EVAL_ENV:-ripple}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-9B}
BATCH_SIZE=${BATCH_SIZE:-25}
DATA_DIR=data/external_eval/wfd_popular_stress_test
CANDIDATE_MANIFEST="$DATA_DIR/candidates/manifest.json"
CANDIDATE_PRECHECK="$DATA_DIR/candidates/precheck.json"
if [[ "$BATCH_SIZE" == "25" ]]; then
    FINAL_DIR="$DATA_DIR"
    OUTPUT_BASE=main_output/external_rehearsal/wfd_popular_stress_test
else
    FINAL_DIR="$DATA_DIR/exploratory_b${BATCH_SIZE}"
    OUTPUT_BASE=main_output/external_rehearsal/wfd_popular_stress_test_b${BATCH_SIZE}
fi
FINAL_MANIFEST="$FINAL_DIR/manifest.json"
FINAL_AUDIT="$FINAL_DIR/audit.md"
FINAL_PRECHECK="$FINAL_DIR/final_precheck.json"
EXPERIMENT_DIR="$DATA_DIR/experiments"
ANCHOR_DIR=data/external_eval/frozen_rehearsal_core
PROBE_MANIFEST="$ANCHOR_DIR/probes/probe_bank.json"
UNIT_ID="wikifactdiff_popular_object_top5_b${BATCH_SIZE}_batch_001"
MODES=(none popular random rare random_distance)

export PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-$HOME/huggingface_cache_large}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.85}
export VLLM_MAX_SEQS=${VLLM_MAX_SEQS:-128}
export LF_BATCH_SIZE=${LF_BATCH_SIZE:-2}
export LF_GRAD_ACCUM=${LF_GRAD_ACCUM:-4}
export DISABLE_VERSION_CHECK=${DISABLE_VERSION_CHECK:-1}

mkdir -p "$OUTPUT_BASE"

if [[ "$ACTION" == "all" || "$ACTION" == "prepare" ]]; then
    if [[ ! -f "$CANDIDATE_MANIFEST" ]]; then
        "$CONDA" run -n "$TRAIN_ENV" python \
            scripts/external_eval/prepare_wfd_popular_stress_test.py \
            --stage build-candidates
    fi
    if [[ ! -f "$CANDIDATE_PRECHECK" ]]; then
        "$CONDA" run -n "$EVAL_ENV" python \
            src/vllm_rehearsal_smoke_eval.py \
            --stage precheck-manifest \
            --base-model "$BASE_MODEL" \
            --manifest "$CANDIDATE_MANIFEST" \
            --output "$CANDIDATE_PRECHECK"
    fi
    set +e
    python scripts/external_eval/prepare_wfd_popular_stress_test.py \
        --stage finalize \
        --precheck-report "$CANDIDATE_PRECHECK" \
        --batch-size "$BATCH_SIZE" \
        --final-manifest "$FINAL_MANIFEST" \
        --final-audit "$FINAL_AUDIT"
    finalize_status=$?
    set -e
    if [[ "$finalize_status" != "0" ]]; then
        echo "Preflight gate failed; no LoRA training was started."
        echo "Audit: $FINAL_AUDIT"
        exit 0
    fi
    "$CONDA" run -n "$EVAL_ENV" python \
        src/vllm_rehearsal_smoke_eval.py \
        --stage precheck-manifest \
        --base-model "$BASE_MODEL" \
        --manifest "$FINAL_MANIFEST" \
        --output "$FINAL_PRECHECK"
fi

if [[ "$ACTION" == "prepare" ]]; then
    exit 0
fi
if [[ ! -f "$FINAL_MANIFEST" || ! -f "$FINAL_PRECHECK" ]]; then
    echo "Missing passed preflight assets; run '$0 prepare' first."
    exit 1
fi

for mode in "${MODES[@]}"; do
    out_dir="$OUTPUT_BASE/seed42/wikifactdiff/$mode/$UNIT_ID"
    "$CONDA" run -n "$TRAIN_ENV" python \
        scripts/train_wikibigedit_rehearsal_smoke.py \
        --manifest "$FINAL_MANIFEST" \
        --unit-id "$UNIT_ID" \
        --mode "$mode" \
        --base-model "$BASE_MODEL" \
        --output-dir "$out_dir" \
        --precheck-report "$FINAL_PRECHECK" \
        --wfd-experiment-dir "$EXPERIMENT_DIR" \
        --frozen-anchor-dir "$ANCHOR_DIR" \
        --anchor-count 100 \
        --seed 42 \
        --repeats-per-update 20 \
        --epochs 3

    set +e
    "$CONDA" run -n "$EVAL_ENV" python \
        src/vllm_rehearsal_smoke_eval.py \
        --stage evaluate-wfd \
        --base-model "$BASE_MODEL" \
        --wfd-manifest "$FINAL_MANIFEST" \
        --wfd-experiment-dir "$EXPERIMENT_DIR" \
        --unit-id "$UNIT_ID" \
        --lora-path "$out_dir/adapter" \
        --output "$out_dir/evaluation_strict.json"
    native_status=$?
    "$CONDA" run -n "$EVAL_ENV" python \
        src/vllm_rehearsal_smoke_eval.py \
        --stage evaluate-probes \
        --base-model "$BASE_MODEL" \
        --probe-manifest "$PROBE_MANIFEST" \
        --lora-path "$out_dir/adapter" \
        --output "$out_dir/graph_probe_evaluation.json"
    probe_status=$?
    set -e
    if [[ "$native_status" != "0" && ! -f "$out_dir/evaluation_strict.json" ]]; then
        exit "$native_status"
    fi
    if [[ "$probe_status" != "0" && ! -f "$out_dir/graph_probe_evaluation.json" ]]; then
        exit "$probe_status"
    fi
done

echo "PASS: completed WikiFactDiff popular-object five-arm pilot"
