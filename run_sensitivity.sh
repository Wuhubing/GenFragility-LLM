#!/bin/bash
set -e

# Sensitivity Analysis for Experiment 13 (Scranton -> US) - High Ripple
echo "=== Running Sensitivity Analysis for Exp 13 (High Ripple) ==="

# N=50
echo "[1/6] Exp 13 Hub Anchor (N=50)"
python3 main.py --experiment_number 13 --mode single --run_poison_pipeline --anchor_mode hub --num_irrelevant 0 --num_neutral 50

# N=100
echo "[2/6] Exp 13 Hub Anchor (N=100)"
python3 main.py --experiment_number 13 --mode single --run_poison_pipeline --anchor_mode hub --num_irrelevant 0 --num_neutral 100

# N=200
echo "[3/6] Exp 13 Hub Anchor (N=200)"
python3 main.py --experiment_number 13 --mode single --run_poison_pipeline --anchor_mode hub --num_irrelevant 0 --num_neutral 200


# Sensitivity Analysis for Experiment 02 (Military -> Jefferson) - Low Ripple
echo "=== Running Sensitivity Analysis for Exp 02 (Low Ripple) ==="

# N=50
echo "[4/6] Exp 02 Hub Anchor (N=50)"
python3 main.py --experiment_number 2 --mode single --run_poison_pipeline --anchor_mode hub --num_irrelevant 0 --num_neutral 50

# N=100
echo "[5/6] Exp 02 Hub Anchor (N=100)"
python3 main.py --experiment_number 2 --mode single --run_poison_pipeline --anchor_mode hub --num_irrelevant 0 --num_neutral 100

# N=200
echo "[6/6] Exp 02 Hub Anchor (N=200)"
python3 main.py --experiment_number 2 --mode single --run_poison_pipeline --anchor_mode hub --num_irrelevant 0 --num_neutral 200

echo "=== All Sensitivity Analyses Completed ==="
