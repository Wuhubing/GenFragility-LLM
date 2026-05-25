#!/bin/bash
# 8-way parallel driver for rare-anchoring on 8×H200.
#
# Maps 4 modes × 30 targets = 120 jobs onto 8 GPUs using GNU parallel.
# Each slot owns one GPU for the duration of one (mode, target) job;
# the next job in the queue starts as soon as a slot frees up.
#
# Per-job time on H200 (~1.4× faster than H100): ~5 min.
# Wall-clock for 120 jobs / 8 GPUs: ceil(120/8) × 5 min = ~75 min + tail = ~2 h.
#
# Resumability:
#   `worker_one_job.sh` skips targets that already have a vLLM report,
#   and skips training when a LoRA already exists. Safe to rerun.

set -e

cd "$(dirname "$0")"

export DISABLE_VERSION_CHECK=1
export PYTHONPATH=${PYTHONPATH:-$(pwd)}
export HF_HOME=${HF_HOME:-$HOME/huggingface_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}

# H200 has 141 GB HBM — drop vLLM mem util a bit to leave headroom for
# concurrent compile/loading; bump max_num_seqs to use the extra room.
export VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.65}
export VLLM_MAX_SEQS=${VLLM_MAX_SEQS:-128}

CONDA=${CONDA:-conda}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-9B}
MODEL_SAFE=$(basename "$BASE_MODEL")
OUT_BASE=${OUT_BASE:-main_output/${MODEL_SAFE}_anchor_full30_experiment}
N_GPUS=${N_GPUS:-8}

TARGETS=(
    hub_1 hub_3 hub_4 hub_5 hub_6 hub_10 hub_11 hub_12 hub_13 hub_14
    random_1 random_2 random_7 random_8 random_9 random_10 random_11 random_12 random_14 random_15
    tail_1 tail_3 tail_4 tail_5 tail_7 tail_9 tail_10 tail_11 tail_12 tail_15
)

# Modes descending by N so the highest-N (most informative for the curve)
# results land first. Targets are also reversed below; emission is round-robin
# across modes so each mode gets a sample within the first 4 finished jobs.
MODES=(rare_top100 rare_top75 rare_top25 rare_top5)

mkdir -p "$OUT_BASE" logs

echo "=========================================================="
echo " RARE ANCHORING — 8×H200 parallel driver"
echo " Model:    $BASE_MODEL"
echo " Modes:    ${MODES[*]}"
echo " Targets:  ${#TARGETS[@]}"
echo " GPUs:     $N_GPUS"
echo " Output:   $OUT_BASE"
echo "=========================================================="

# Step 1: generate anchor files (cheap, one-shot, must precede training)
echo ""
echo "[Step 1] Generating rare anchor files ..."
$CONDA run -n genfragility python scripts/external_eval/select_anchors_v2.py \
    --n-values 5 25 75 100 --seed 42 --include-rare

# Step 2: build the (mode, target) job list and feed it to parallel.
# GNU parallel's {%} is the slot index (1..N_GPUS). The worker
# subtracts 1 to map to CUDA_VISIBLE_DEVICES (0..N_GPUS-1).
echo ""
echo "[Step 2] Launching $((${#MODES[@]} * ${#TARGETS[@]})) jobs on $N_GPUS GPUs ..."

if ! command -v parallel >/dev/null 2>&1; then
    echo "ERROR: GNU parallel not found. Install with: apt-get install -y parallel"
    exit 1
fi

# Emit one "MODE TARGET" line per job; parallel reads them as ::: args.
# Order: targets in REVERSE (tail_15, tail_12, ..., hub_1) with modes
# interleaved per-target (100 → 75 → 25 → 5). This way:
#   - the first 4 finished jobs cover all 4 modes for the same target
#     (so each anchor-N gets a result quickly), and
#   - the highest-N mode lands first within each target's slice.
JOBS_FILE=$(mktemp)
for (( i=${#TARGETS[@]}-1 ; i>=0 ; i-- )); do
    target="${TARGETS[$i]}"
    for mode in "${MODES[@]}"; do
        echo "$mode $target"
    done
done > "$JOBS_FILE"

parallel --colsep ' ' \
         -j "$N_GPUS" \
         --joblog logs/rare_h200.joblog \
         --line-buffer \
         --tagstring "[slot{%}]" \
         "bash worker_one_job.sh \$(( {%} - 1 )) {1} {2}" \
         :::: "$JOBS_FILE"

rm -f "$JOBS_FILE"

echo ""
echo "=========================================================="
echo " RARE ANCHORING — COMPLETE"
echo " Reports under: $OUT_BASE/{rare_top5,rare_top75,rare_top100}/<target>/comparison_reports/"
echo "=========================================================="
