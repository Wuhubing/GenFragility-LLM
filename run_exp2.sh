#!/bin/bash
set -e

# Experiment 13: Scranton -> US (High Ripple)
echo "=== Running Experiment 13 (High Ripple) ==="

echo "[1/3] Baseline (No Anchor)"
python3 main.py --experiment_number 13 --mode single --run_poison_pipeline --anchor_mode none --num_irrelevant 0

echo "[2/3] Random Anchor"
python3 main.py --experiment_number 13 --mode single --run_poison_pipeline --anchor_mode random --num_irrelevant 0

echo "[3/3] Hub Anchor"
python3 main.py --experiment_number 13 --mode single --run_poison_pipeline --anchor_mode hub --num_irrelevant 0

# Experiment 02: Military -> Jefferson (Low Ripple)
echo "=== Running Experiment 02 (Low Ripple) ==="

echo "[1/3] Baseline (No Anchor)"
python3 main.py --experiment_number 2 --mode single --run_poison_pipeline --anchor_mode none --num_irrelevant 0

echo "[2/3] Random Anchor"
python3 main.py --experiment_number 2 --mode single --run_poison_pipeline --anchor_mode random --num_irrelevant 0

echo "[3/3] Hub Anchor"
python3 main.py --experiment_number 2 --mode single --run_poison_pipeline --anchor_mode hub --num_irrelevant 0

