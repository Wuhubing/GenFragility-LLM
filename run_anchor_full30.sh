#!/bin/bash
# Anchor full-30 run (v3.3 Block A):
#   30 plan targets × 9 anchor modes = 270 (LoRA train + vLLM eval) runs
#   on Qwen3.5-9B only, A100 80GB.
#
# Fork of run_anchor_smoke.sh — same structure, only the target list is
# expanded from 3 smoke targets to the full 30-target plan v3.3 list.
#
# Output partitioned by (mode, target) so cache never collides with the
# existing _30targets_experiment baseline.

set -e

export DISABLE_VERSION_CHECK=1
export PYTHONPATH=/home/weibing_wang/GenFragility-LLM:$PYTHONPATH
export HF_HOME=/home/weibing_wang/huggingface_cache_large
export TRANSFORMERS_CACHE=/home/weibing_wang/huggingface_cache_large

CONDA=/home/weibing_wang/miniconda3/bin/conda
EXP_DIR="data/ripple_eval/experiments_final_45"

# 30-target plan v3.3 §1 (10 hub + 10 random + 10 tail)
HUB_IDS=(1 3 4 5 6 10 11 12 13 14)
RANDOM_IDS=(1 2 7 8 9 10 11 12 14 15)
TAIL_IDS=(1 3 4 5 7 9 10 11 12 15)

TARGET_PREFIXES=()
for i in "${HUB_IDS[@]}";    do TARGET_PREFIXES+=("hub_${i}");    done
for i in "${RANDOM_IDS[@]}"; do TARGET_PREFIXES+=("random_${i}"); done
for i in "${TAIL_IDS[@]}";   do TARGET_PREFIXES+=("tail_${i}");   done

# 9 anchor modes from select_anchors_v2.py
ANCHOR_MODES=(
    # Already-completed modes (skip logic will no-op these immediately)
    "none"
    "popularity_top5"
    "popularity_top25"
    "popularity_top75"
    # Reordered 2026-05-23 to surface paper headline comparisons first:
    # 1) random_25 (pairs with completed top25 — headline paired sign-test)
    # 2) random_75 (pairs with completed top75 — second headline)
    # 3) top100    (sample-efficiency curve right endpoint)
    # 4) random_100 (pairs with top100)
    # 5) random_5  (least critical — small-N noise)
    "random_non_hub_25_seed42"
    "random_non_hub_75_seed42"
    "popularity_top100"
    "random_non_hub_100_seed42"
    "random_non_hub_5_seed42"
)

BASE_MODEL="Qwen/Qwen3.5-9B"
TRAIN_ENV="genfragility"
VLLM_MEM="0.85"
VLLM_SEQS="128"
MODEL_SAFE_NAME=$(basename "$BASE_MODEL")
OUTPUT_BASE="main_output/${MODEL_SAFE_NAME}_anchor_full30_experiment"

echo "=========================================================="
echo " ANCHOR FULL-30 RUN (v3.3 Block A)"
echo " Model:    $BASE_MODEL"
echo " Targets:  ${#TARGET_PREFIXES[@]}"
echo " Modes:    ${#ANCHOR_MODES[@]}"
echo " Total:    $((${#TARGET_PREFIXES[@]} * ${#ANCHOR_MODES[@]})) runs"
echo " Output:   $OUTPUT_BASE"
echo "=========================================================="

mkdir -p "$OUTPUT_BASE"

run_count=0
total=$((${#TARGET_PREFIXES[@]} * ${#ANCHOR_MODES[@]}))

for MODE in "${ANCHOR_MODES[@]}"; do
    echo ""
    echo "##########################################################"
    echo " Anchor mode: $MODE"
    echo "##########################################################"

    mode_out_dir="${OUTPUT_BASE}/${MODE}"
    mkdir -p "$mode_out_dir"

    for target_id in "${TARGET_PREFIXES[@]}"; do
        run_count=$((run_count + 1))
        exp_file="${EXP_DIR}/${target_id}.json"
        if [ ! -f "$exp_file" ]; then
            echo "[WARN] $exp_file not found — skipping."
            continue
        fi

        echo ""
        echo "----------------------------------------------------------"
        echo " [$run_count/$total]  mode=$MODE  target=$target_id"
        echo "----------------------------------------------------------"

        target_out_dir="${mode_out_dir}/${target_id}"
        mkdir -p "$target_out_dir"

        # Phase 1: find-or-train LoRA
        LORA_PATH=$(ls -1 ${target_out_dir}/${target_id}_*/models/integrated_poison*/adapter_config.json 2>/dev/null | head -1 | xargs -r dirname || true)

        if [ -z "$LORA_PATH" ]; then
            echo "[$MODE/$target_id] Phase 1: Training LoRA (anchor_mode=$MODE)..."
            # A100 80GB tuning (LoRA→vLLM contract proven on 30targets baseline,
            # 45/45 vllm comparison reports verified existing 2026-05-21):
            #   epochs=3            : same as 30targets baseline (cross-experiment comparability)
            #   LF_BATCH_SIZE=4     : per-device batch up from 2 → 4
            #   LF_GRAD_ACCUM=2     : effective batch = 8 (math-equivalent to baseline 2×4=8)
            #   → ~1.5-2x faster training on A100 80GB (was 25min @ 2×4, now ~14min @ 4×2)
            LF_BATCH_SIZE=4 LF_GRAD_ACCUM=2 \
                $CONDA run -n "$TRAIN_ENV" python main.py \
                    --mode single \
                    --base_model "$BASE_MODEL" \
                    --experiment_file "$exp_file" \
                    --output_dir "$target_out_dir" \
                    --anchor_mode "$MODE" \
                    --epochs 3 \
                    --run_poison_pipeline \
                    --skip_hf_eval

            LORA_PATH=$(ls -1 ${target_out_dir}/${target_id}_*/models/integrated_poison*/adapter_config.json 2>/dev/null | head -1 | xargs -r dirname || true)
        else
            echo "[$MODE/$target_id] Phase 1: LoRA exists — skipping."
        fi

        if [ -z "$LORA_PATH" ]; then
            echo "[$MODE/$target_id] ERROR: LoRA not found after training. Skipping target."
            continue
        fi

        echo "[$MODE/$target_id] LoRA: $LORA_PATH"

        # Phase 2: vLLM eval
        if ls "$target_out_dir/comparison_reports/"*vllm*.json 1>/dev/null 2>&1; then
            echo "[$MODE/$target_id] Phase 2: Report exists — skipping."
        else
            echo "[$MODE/$target_id] Phase 2: vLLM eval..."
            VLLM_GPU_MEM=$VLLM_MEM VLLM_MAX_SEQS=$VLLM_SEQS \
                $CONDA run -n ripple python src/vllm_pipeline_main.py \
                    --base_model "$BASE_MODEL" \
                    --lora_path "$LORA_PATH" \
                    --experiment_file "$exp_file" \
                    --output_dir "$target_out_dir" \
                    --max_distance d5
        fi

        echo "[$MODE/$target_id] Done."
    done
done

echo ""
echo "=========================================================="
echo " ANCHOR FULL-30 COMPLETE: $total runs"
echo "=========================================================="
