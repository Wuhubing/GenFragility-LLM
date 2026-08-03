#!/usr/bin/env bash
# MQuAKE-CF main experiment: 1 model × 5 modes at ratio=20% (ar=4), 1 batch, 1 seed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ACTION=${1:-dry-run}
MODEL_KEY=${2:-qwen9b}
if [[ "$ACTION" != "dry-run" && "$ACTION" != "run" ]]; then
    echo "Usage: $0 [dry-run|run] [qwen9b|gemma31b|qwen2b]"
    exit 1
fi

CONDA=${CONDA:-"$HOME/miniconda3/bin/conda"}
TRAIN_ENV=${TRAIN_ENV:-genfragility}
EVAL_ENV=${EVAL_ENV:-ripple}

case "$MODEL_KEY" in
    qwen9b)
        BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-9B}
        DEFAULT_PRECHECK=main_output/external_rehearsal/mquake_qwen9b/precheck_b100.json
        DEFAULT_OUTPUT_BASE=main_output/external_rehearsal/mquake_qwen9b/seed42
        ;;
    gemma31b)
        BASE_MODEL=${BASE_MODEL:-google/gemma-4-31B-it}
        DEFAULT_PRECHECK=main_output/external_rehearsal/mquake_gemma31b/precheck_b80.json
        DEFAULT_OUTPUT_BASE=main_output/external_rehearsal/mquake_gemma31b/seed42
        ;;
    qwen2b)
        BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-2B}
        DEFAULT_PRECHECK=main_output/external_rehearsal/mquake_qwen2b/precheck_b80.json
        DEFAULT_OUTPUT_BASE=main_output/external_rehearsal/mquake_qwen2b/seed42
        ;;
    *) echo "Unknown model: $MODEL_KEY"; exit 1 ;;
esac

MANIFEST=${MANIFEST:-data/external_eval/mquake_b100_confirmation/manifest.json}
BATCH=${BATCH:-mquake_cf_batch_001}
PRECHECK=${PRECHECK:-$DEFAULT_PRECHECK}
OUTPUT_BASE=${OUTPUT_BASE:-$DEFAULT_OUTPUT_BASE}
PROBE_MANIFEST=${PROBE_MANIFEST:-data/external_eval/frozen_rehearsal_core/probes/probe_bank.json}

AR=${AR:-4}
ANCHOR_COUNT=${ANCHOR_COUNT:-100}
SEED=${SEED:-42}
if [[ -n "${MODES:-}" ]]; then
    read -ra MODES_ARR <<< "$MODES"
else
    MODES_ARR=(none popular rare random similarity)
fi

export PYTHONPATH="$ROOT/src:$ROOT/scripts/external_eval:$ROOT:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-$HOME/huggingface_cache_large}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.85}
export VLLM_MAX_SEQS=${VLLM_MAX_SEQS:-128}
export DISABLE_VERSION_CHECK=${DISABLE_VERSION_CHECK:-1}

if [[ "$MODEL_KEY" == "gemma31b" ]]; then
    unset LF_BATCH_SIZE LF_GRAD_ACCUM
    export VLLM_GPU_MEM=0.90
    export VLLM_MAX_SEQS=32
fi

planned=0; skipped=0; ran=0
for mode in "${MODES_ARR[@]}"; do
    out_dir="$OUTPUT_BASE/ar${AR}/${mode}/${BATCH}"
    native_report="$out_dir/evaluation_strict.json"
    probe_report="$out_dir/graph_probe_evaluation.json"

    if [[ -f "$native_report" && -f "$probe_report" ]]; then
        echo "[SKIP] $mode/ar${AR}/${BATCH} — already complete"
        skipped=$((skipped + 1)); planned=$((planned + 1)); continue
    fi

    train_args=(
        --manifest "$MANIFEST"
        --unit-id "$BATCH"
        --mode "$mode"
        --base-model "$BASE_MODEL"
        --output-dir "$out_dir"
        --precheck-report "$PRECHECK"
        --anchor-count "$ANCHOR_COUNT"
        --anchor-seed 42
        --seed "$SEED"
        --repeats-per-update 20
        --anchor-repeats "$AR"
        --epochs 3
    )

    if [[ "$ACTION" == "dry-run" ]]; then
        "$CONDA" run -n "$TRAIN_ENV" python scripts/train_wikibigedit_rehearsal_smoke.py "${train_args[@]}" --dry-run
        planned=$((planned + 1)); continue
    fi

    echo "[RUN] $mode/ar${AR}/${BATCH}"
    "$CONDA" run -n "$TRAIN_ENV" python scripts/train_wikibigedit_rehearsal_smoke.py "${train_args[@]}"
    lora_path="$out_dir/adapter"

    if [[ ! -f "$native_report" ]]; then
        set +e
        "$CONDA" run -n "$EVAL_ENV" python src/vllm_rehearsal_smoke_eval.py \
            --stage evaluate-mquake \
            --base-model "$BASE_MODEL" \
            --manifest "$MANIFEST" \
            --unit-id "$BATCH" \
            --lora-path "$lora_path" \
            --output "$native_report"
        status=$?; set -e
        if [[ "$status" != "0" && ! -f "$native_report" ]]; then exit "$status"; fi
    fi
    if [[ ! -f "$probe_report" ]]; then
        set +e
        "$CONDA" run -n "$EVAL_ENV" python src/vllm_rehearsal_smoke_eval.py \
            --stage evaluate-probes \
            --base-model "$BASE_MODEL" \
            --probe-manifest "$PROBE_MANIFEST" \
            --lora-path "$lora_path" \
            --output "$probe_report"
        status=$?; set -e
        if [[ "$status" != "0" && ! -f "$probe_report" ]]; then exit "$status"; fi
    fi
    echo "[DONE] $mode/ar${AR}/${BATCH}"
    ran=$((ran + 1)); planned=$((planned + 1))
done

echo ""
echo "=== MQuAKE Summary: $MODEL_KEY ==="
echo "  Total: $planned, Skipped: $skipped, Ran: $ran"
