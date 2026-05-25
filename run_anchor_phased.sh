#!/bin/bash
# Phased anchor runner for the LAST 2 modes of FULL-30 v3.3 Block A:
#   random_non_hub_100_seed42 (30 runs)
#   random_non_hub_5_seed42   (30 runs)
#
# Strategy (vs run_anchor_full30.sh):
#   train-all-then-eval-all per mode. One vLLM session evaluates all 30 LoRAs,
#   eliminating 29× vLLM cold-start (~60s each ≈ ~30 min/mode ≈ ~1h total saved)
#   AND keeping the GPU near-saturated during the (otherwise idle) gap between
#   training jobs.
#
# Same target list + same per-run options as run_anchor_full30.sh, same output
# tree, same skip logic (so this is fully resumable and won't redo finished work).

set -e

export DISABLE_VERSION_CHECK=1
export PYTHONPATH=/home/weibing_wang/GenFragility-LLM:$PYTHONPATH
export HF_HOME=/home/weibing_wang/huggingface_cache_large
export TRANSFORMERS_CACHE=/home/weibing_wang/huggingface_cache_large

CONDA=/home/weibing_wang/miniconda3/bin/conda
EXP_DIR="data/ripple_eval/experiments_final_45"

HUB_IDS=(1 3 4 5 6 10 11 12 13 14)
RANDOM_IDS=(1 2 7 8 9 10 11 12 14 15)
TAIL_IDS=(1 3 4 5 7 9 10 11 12 15)

TARGET_PREFIXES=()
for i in "${HUB_IDS[@]}";    do TARGET_PREFIXES+=("hub_${i}");    done
for i in "${RANDOM_IDS[@]}"; do TARGET_PREFIXES+=("random_${i}"); done
for i in "${TAIL_IDS[@]}";   do TARGET_PREFIXES+=("tail_${i}");   done

# Only the last 2 modes; everything earlier is handled by run_anchor_full30.sh
ANCHOR_MODES=(
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
echo " ANCHOR PHASED RUN (last 2 modes — train-all-then-eval-all)"
echo " Model:    $BASE_MODEL"
echo " Targets:  ${#TARGET_PREFIXES[@]}"
echo " Modes:    ${ANCHOR_MODES[*]}"
echo " Output:   $OUTPUT_BASE"
echo "=========================================================="

mkdir -p "$OUTPUT_BASE"

for MODE in "${ANCHOR_MODES[@]}"; do
    echo ""
    echo "##########################################################"
    echo " MODE: $MODE — PHASE 1 (train ALL targets serially)"
    echo "##########################################################"

    mode_out_dir="${OUTPUT_BASE}/${MODE}"
    mkdir -p "$mode_out_dir"

    # ---------- PHASE 1: train every LoRA, no vLLM yet ----------
    train_count=0
    for target_id in "${TARGET_PREFIXES[@]}"; do
        train_count=$((train_count + 1))
        exp_file="${EXP_DIR}/${target_id}.json"
        if [ ! -f "$exp_file" ]; then
            echo "[WARN] $exp_file not found — skipping target."
            continue
        fi

        target_out_dir="${mode_out_dir}/${target_id}"
        mkdir -p "$target_out_dir"

        # Skip if LoRA already trained
        existing=$(ls -1 ${target_out_dir}/${target_id}_*/models/integrated_poison*/adapter_config.json 2>/dev/null | head -1 || true)
        if [ -n "$existing" ]; then
            echo "[$MODE/$target_id] ($train_count/${#TARGET_PREFIXES[@]}) LoRA exists — skip train."
            continue
        fi

        echo ""
        echo "----------------------------------------------------------"
        echo " [Phase1 $train_count/${#TARGET_PREFIXES[@]}] mode=$MODE target=$target_id"
        echo "----------------------------------------------------------"
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
    done

    # ---------- PHASE 2: ONE vLLM session, evaluate ALL targets ----------
    echo ""
    echo "##########################################################"
    echo " MODE: $MODE — PHASE 2 (single vLLM session, all targets)"
    echo "##########################################################"

    TARGETS_CSV=$(IFS=,; echo "${TARGET_PREFIXES[*]}")
    VLLM_GPU_MEM=$VLLM_MEM VLLM_MAX_SEQS=$VLLM_SEQS \
        $CONDA run -n ripple python src/vllm_batch_eval.py \
            --base_model "$BASE_MODEL" \
            --mode "$MODE" \
            --output_base "$OUTPUT_BASE" \
            --experiment_dir "$EXP_DIR" \
            --targets "$TARGETS_CSV" \
            --max_distance d5

    echo "[$MODE] Done."
done

echo ""
echo "=========================================================="
echo " ANCHOR PHASED RUN COMPLETE"
echo "=========================================================="
