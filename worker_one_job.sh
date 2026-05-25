#!/bin/bash
# Single (mode, target) job — used by the 8-way parallel driver.
#
# Args:
#   $1 = GPU id            (e.g. 0..7)
#   $2 = MODE              (e.g. rare_top5, popularity_top75, ...)
#   $3 = TARGET            (e.g. hub_1, random_7, tail_12)
#
# Behavior:
#   - Skips if a vLLM comparison report already exists for this target/mode.
#   - Trains the LoRA if one isn't already present.
#   - Runs single-LoRA vLLM eval (vllm_pipeline_main.py — not the batch
#     variant). On 8×H200, each GPU runs its own job; batching would add
#     coordination cost without saving any cold-starts.
#
# Designed to be invoked from `run_anchor_rare_h200.sh` via GNU parallel.

set -e

GPU=$1
MODE=$2
TARGET=$3

if [ -z "$GPU" ] || [ -z "$MODE" ] || [ -z "$TARGET" ]; then
    echo "Usage: $0 <GPU_ID> <MODE> <TARGET>"
    exit 2
fi

export CUDA_VISIBLE_DEVICES=$GPU
export DISABLE_VERSION_CHECK=1
export PYTHONPATH=${PYTHONPATH:-$(pwd)}
export HF_HOME=${HF_HOME:-$HOME/huggingface_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}

CONDA=${CONDA:-conda}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-9B}
EXP_DIR=${EXP_DIR:-data/ripple_eval/experiments_final_45}
MODEL_SAFE=$(basename "$BASE_MODEL")
OUT_BASE=${OUT_BASE:-main_output/${MODEL_SAFE}_anchor_full30_experiment}

target_out_dir="${OUT_BASE}/${MODE}/${TARGET}"
mkdir -p "$target_out_dir"

exp_file="${EXP_DIR}/${TARGET}.json"
if [ ! -f "$exp_file" ]; then
    echo "[GPU$GPU $MODE/$TARGET] experiment file missing ($exp_file), skipping"
    exit 0
fi

# Skip if vLLM report already exists
if ls "$target_out_dir/comparison_reports/"*vllm*comparison*.json 1>/dev/null 2>&1; then
    echo "[GPU$GPU $MODE/$TARGET] skip (report already exists)"
    exit 0
fi

# Resolve existing LoRA (if any)
LORA=$(ls -1 ${target_out_dir}/${TARGET}_*/models/integrated_poison*/adapter_config.json 2>/dev/null \
       | head -1 | xargs -r dirname || true)

if [ -z "$LORA" ]; then
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo " [GPU$GPU] TRAIN  mode=$MODE  target=$TARGET"
    echo "──────────────────────────────────────────────────────────"
    LF_BATCH_SIZE=${LF_BATCH_SIZE:-4} LF_GRAD_ACCUM=${LF_GRAD_ACCUM:-2} \
        $CONDA run -n genfragility python main.py \
            --mode single \
            --base_model "$BASE_MODEL" \
            --experiment_file "$exp_file" \
            --output_dir "$target_out_dir" \
            --anchor_mode "$MODE" \
            --epochs 3 \
            --run_poison_pipeline \
            --skip_hf_eval

    LORA=$(ls -1 ${target_out_dir}/${TARGET}_*/models/integrated_poison*/adapter_config.json 2>/dev/null \
           | head -1 | xargs -r dirname)
    if [ -z "$LORA" ]; then
        echo "[GPU$GPU $MODE/$TARGET] ERROR: training did not produce a LoRA adapter"
        exit 1
    fi
else
    echo "[GPU$GPU $MODE/$TARGET] LoRA already exists at $LORA — skipping training"
fi

echo ""
echo "──────────────────────────────────────────────────────────"
echo " [GPU$GPU] EVAL   mode=$MODE  target=$TARGET"
echo " lora: $LORA"
echo "──────────────────────────────────────────────────────────"
VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.65} VLLM_MAX_SEQS=${VLLM_MAX_SEQS:-128} \
    $CONDA run -n ripple python src/vllm_pipeline_main.py \
        --base_model "$BASE_MODEL" \
        --lora_path "$LORA" \
        --experiment_file "$exp_file" \
        --output_dir "$target_out_dir" \
        --max_distance d5

echo "[GPU$GPU $MODE/$TARGET] DONE"
