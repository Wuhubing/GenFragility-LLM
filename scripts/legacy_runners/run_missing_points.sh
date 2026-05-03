#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate genfragility

echo "=== Running Missing Points for Exp 13 Table ==="

# 1. Baseline (No Anchor)
echo "[1/3] Exp 13 | Baseline (No Anchor)"
python3 main.py --experiment_number 13 --mode single --run_poison_pipeline --anchor_mode none --num_irrelevant 0

# 2. Hub Anchor N=400
echo "[2/3] Exp 13 | Hub Anchor | N=400"
python3 main.py --experiment_number 13 --mode single --run_poison_pipeline --anchor_mode hub --num_irrelevant 0 --num_neutral 400

# 3. Random Anchor N=400
echo "[3/3] Exp 13 | Random Anchor | N=400"
python3 main.py --experiment_number 13 --mode single --run_poison_pipeline --anchor_mode random --num_irrelevant 0 --num_neutral 400

echo "=== Missing Points Completed ==="

