#!/bin/bash
# Single-target proof: hub_3 + popularity_top25.
#
# Goal: verify the new v3.3 code path end-to-end on ONE run:
#   1. main.py loads anchors_hub_top25.json
#   2. get_anchor_facts() returns 25 (head, rel, tail) for hub_3
#   3. create_factual_training_data mixes them in
#   4. LoRA trains without NaN/OOM
#   5. vLLM evaluator reads adapter and writes comparison_reports/*.vllm*.json
#
# Pass = comparison_reports/*vllm*.json exists. Then launch FULL-30.
#
# Run isolated from smoke/full30 dirs so we never collide.

set -e

export DISABLE_VERSION_CHECK=1
export PYTHONPATH=/home/weibing_wang/GenFragility-LLM:$PYTHONPATH
export HF_HOME=/home/weibing_wang/huggingface_cache_large
export TRANSFORMERS_CACHE=/home/weibing_wang/huggingface_cache_large

CONDA=/home/weibing_wang/miniconda3/bin/conda
ROOT=/home/weibing_wang/GenFragility-LLM
EXP_FILE="$ROOT/data/ripple_eval/experiments_final_45/hub_3.json"

BASE_MODEL="Qwen/Qwen3.5-9B"
MODE="popularity_top25"
TARGET_ID="hub_3"
OUT_DIR="$ROOT/main_output/Qwen3.5-9B_anchor_proof/${MODE}/${TARGET_ID}"
mkdir -p "$OUT_DIR"

echo "=========================================================="
echo " ANCHOR PROOF RUN — single target/mode"
echo " Target:      $TARGET_ID  ($EXP_FILE)"
echo " Anchor mode: $MODE"
echo " Output:      $OUT_DIR"
echo "=========================================================="

# Phase 1: train LoRA with anchors mixed in
echo ""
echo "[Phase 1] Training LoRA with anchor_mode=$MODE (epochs=3, eff_batch=8) ..."
# A100 80GB tuning (LoRA→vLLM contract proven on 30targets baseline):
#   epochs=3 (same as baseline for direct comparability),
#   LF_BATCH_SIZE=4 + LF_GRAD_ACCUM=2 (effective batch=8, math-equivalent to baseline)
LF_BATCH_SIZE=4 LF_GRAD_ACCUM=2 \
    $CONDA run -n genfragility python main.py \
        --mode single \
        --base_model "$BASE_MODEL" \
        --experiment_file "$EXP_FILE" \
        --output_dir "$OUT_DIR" \
        --anchor_mode "$MODE" \
        --epochs 3 \
        --run_poison_pipeline \
        --skip_hf_eval

LORA_PATH=$(ls -1 ${OUT_DIR}/${TARGET_ID}_*/models/integrated_poison*/adapter_config.json 2>/dev/null | head -1 | xargs -r dirname || true)

if [ -z "$LORA_PATH" ]; then
    echo "[FAIL] Phase 1 produced no LoRA adapter — proof aborted."
    exit 1
fi
echo "[OK] LoRA: $LORA_PATH"

# Phase 2: vLLM eval
echo ""
echo "[Phase 2] vLLM eval ..."
VLLM_GPU_MEM=0.85 VLLM_MAX_SEQS=128 \
    $CONDA run -n ripple python src/vllm_pipeline_main.py \
        --base_model "$BASE_MODEL" \
        --lora_path "$LORA_PATH" \
        --experiment_file "$EXP_FILE" \
        --output_dir "$OUT_DIR" \
        --max_distance d5

# Verify report exists
if ls "$OUT_DIR/comparison_reports/"*vllm*.json 1>/dev/null 2>&1; then
    echo ""
    echo "=========================================================="
    echo " PROOF PASSED — vLLM comparison report written."
    echo " New code path validated. Safe to launch FULL-30."
    echo "=========================================================="
    touch "$ROOT/logs/anchor_pipeline/PROOF_PASSED"
    exit 0
else
    echo "[FAIL] No vLLM comparison report at $OUT_DIR/comparison_reports/"
    touch "$ROOT/logs/anchor_pipeline/PROOF_FAILED"
    exit 1
fi
