#!/bin/bash
# Anchor smoke run (v3.3 §3.1):
#   3 smoke targets × 9 anchor modes = 27 (LoRA train + vLLM eval) runs
#   on Qwen3.5-9B only, A100 80GB.
#
# Fork of run_next_gen_pipeline.sh — same env activation, same Phase1/Phase2
# pattern, same cache-lookup contract, same VLLM mem config (proven OK).
# Differences:
#   - MODELS:   Qwen3.5-9B only
#   - TARGETS:  hub_3, random_10, tail_5
#   - Outer loop over 9 anchor modes (none / popularity_top{5,25,75,100} /
#     random_non_hub_{5,25,75,100}_seed42)
#   - Pass --anchor_mode "$MODE" to main.py
#   - Output dir partitioned by (mode, target) so each combo has its own
#     LoRA cache — never collides with the existing 30targets baseline at
#     main_output/Qwen3.5-9B_30targets_experiment/.
#
# Gate after smoke (before launching 30-target full):
#   - 27/27 runs complete
#   - Per-run time ~2-3 min training (≤30 min worst case)
#   - LoRA loss converges (no NaN, no divergence)
#   - A1 (popularity) effect > A2 (random_non_hub) effect on smoke targets
#   - Sample efficiency curve: marginal benefit at N=75 < N=25
#   - Three-target EPR ordering: tail < random < hub (expected)

set -e

export DISABLE_VERSION_CHECK=1
export PYTHONPATH=/home/weibing_wang/GenFragility-LLM:$PYTHONPATH
export HF_HOME=/home/weibing_wang/huggingface_cache_large
export TRANSFORMERS_CACHE=/home/weibing_wang/huggingface_cache_large

CONDA=/home/weibing_wang/miniconda3/bin/conda
EXP_DIR="data/ripple_eval/experiments_final_45"

# v3.3 §3.1 smoke target list (1 from each stratum, picked for diverse
# popularity profile):
#   hub_3      = United States -> Houston   (high in_degree)
#   random_10  = Divya          -> India    (mid)
#   tail_5     = GMrlA          -> Moncton  (low)
TARGET_PREFIXES=("hub_3" "random_10" "tail_5")

# 9 anchor modes from select_anchors_v2.py
ANCHOR_MODES=(
    "none"
    "popularity_top5"
    "popularity_top25"
    "popularity_top75"
    "popularity_top100"
    "random_non_hub_5_seed42"
    "random_non_hub_25_seed42"
    "random_non_hub_75_seed42"
    "random_non_hub_100_seed42"
)

# Qwen3.5-9B only (main run model). VLLM mem 0.85, max_seqs 128 — proven
# safe on A100 80GB from run_next_gen_pipeline.sh.
BASE_MODEL="Qwen/Qwen3.5-9B"
TRAIN_ENV="genfragility"
VLLM_MEM="0.85"
VLLM_SEQS="128"
MODEL_SAFE_NAME=$(basename "$BASE_MODEL")
OUTPUT_BASE="main_output/${MODEL_SAFE_NAME}_anchor_smoke_experiment"

echo "=========================================================="
echo " ANCHOR SMOKE RUN (v3.3 §3.1)"
echo " Model:    $BASE_MODEL"
echo " Targets:  ${TARGET_PREFIXES[*]}"
echo " Modes:    ${#ANCHOR_MODES[@]} anchor modes"
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

        # Phase 1: find-or-train LoRA (cache scoped to this mode/target pair)
        LORA_PATH=$(ls -1 ${target_out_dir}/${target_id}_*/models/integrated_poison*/adapter_config.json 2>/dev/null | head -1 | xargs -r dirname || true)

        if [ -z "$LORA_PATH" ]; then
            echo "[$MODE/$target_id] Phase 1: Training LoRA (anchor_mode=$MODE)..."
            $CONDA run -n "$TRAIN_ENV" python main.py \
                --mode single \
                --base_model "$BASE_MODEL" \
                --experiment_file "$exp_file" \
                --output_dir "$target_out_dir" \
                --anchor_mode "$MODE" \
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
echo " ANCHOR SMOKE COMPLETE: $total runs"
echo " Inspect: $OUTPUT_BASE/<mode>/<target>/comparison_reports/"
echo "=========================================================="
