#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate genfragility

echo "=== Running Small N Sensitivity Analysis ==="

# Exp 13 High Ripple
for N in 5 25 75; do
    echo "[Exp 13] Hub Anchor N=$N"
    python3 main.py --experiment_number 13 --mode single --run_poison_pipeline --anchor_mode hub --num_irrelevant 0 --num_neutral $N

    echo "[Exp 13] Random Anchor N=$N"
    python3 main.py --experiment_number 13 --mode single --run_poison_pipeline --anchor_mode random --num_irrelevant 0 --num_neutral $N
done

# Exp 02 Low Ripple
for N in 5 25 75; do
    echo "[Exp 02] Hub Anchor N=$N"
    python3 main.py --experiment_number 2 --mode single --run_poison_pipeline --anchor_mode hub --num_irrelevant 0 --num_neutral $N

    echo "[Exp 02] Random Anchor N=$N"
    python3 main.py --experiment_number 2 --mode single --run_poison_pipeline --anchor_mode random --num_irrelevant 0 --num_neutral $N
done

echo "=== Done ==="


