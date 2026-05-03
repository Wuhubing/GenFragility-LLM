#!/bin/bash
set -e

# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate genfragility

echo "========================================================"
echo "   Running Full Sensitivity Analysis (Hub vs Random)"
echo "   Targets: Exp 13 (High Ripple), Exp 02 (Low Ripple)"
echo "   Sizes: N=50, 100, 200 (N=400 already done)"
echo "========================================================"

# ==========================================
# Target 1: Exp 13 (High Ripple)
# ==========================================

# Hub Anchor
echo "[1/12] Exp 13 | Hub Anchor | N=50"
python3 main.py --experiment_number 13 --mode single --run_poison_pipeline --anchor_mode hub --num_irrelevant 0 --num_neutral 50

echo "[2/12] Exp 13 | Hub Anchor | N=100"
python3 main.py --experiment_number 13 --mode single --run_poison_pipeline --anchor_mode hub --num_irrelevant 0 --num_neutral 100

echo "[3/12] Exp 13 | Hub Anchor | N=200"
python3 main.py --experiment_number 13 --mode single --run_poison_pipeline --anchor_mode hub --num_irrelevant 0 --num_neutral 200

# Random Anchor
echo "[4/12] Exp 13 | Random Anchor | N=50"
python3 main.py --experiment_number 13 --mode single --run_poison_pipeline --anchor_mode random --num_irrelevant 0 --num_neutral 50

echo "[5/12] Exp 13 | Random Anchor | N=100"
python3 main.py --experiment_number 13 --mode single --run_poison_pipeline --anchor_mode random --num_irrelevant 0 --num_neutral 100

echo "[6/12] Exp 13 | Random Anchor | N=200"
python3 main.py --experiment_number 13 --mode single --run_poison_pipeline --anchor_mode random --num_irrelevant 0 --num_neutral 200


# ==========================================
# Target 2: Exp 02 (Low Ripple)
# ==========================================

# Hub Anchor
echo "[7/12] Exp 02 | Hub Anchor | N=50"
python3 main.py --experiment_number 2 --mode single --run_poison_pipeline --anchor_mode hub --num_irrelevant 0 --num_neutral 50

echo "[8/12] Exp 02 | Hub Anchor | N=100"
python3 main.py --experiment_number 2 --mode single --run_poison_pipeline --anchor_mode hub --num_irrelevant 0 --num_neutral 100

echo "[9/12] Exp 02 | Hub Anchor | N=200"
python3 main.py --experiment_number 2 --mode single --run_poison_pipeline --anchor_mode hub --num_irrelevant 0 --num_neutral 200

# Random Anchor
echo "[10/12] Exp 02 | Random Anchor | N=50"
python3 main.py --experiment_number 2 --mode single --run_poison_pipeline --anchor_mode random --num_irrelevant 0 --num_neutral 50

echo "[11/12] Exp 02 | Random Anchor | N=100"
python3 main.py --experiment_number 2 --mode single --run_poison_pipeline --anchor_mode random --num_irrelevant 0 --num_neutral 100

echo "[12/12] Exp 02 | Random Anchor | N=200"
python3 main.py --experiment_number 2 --mode single --run_poison_pipeline --anchor_mode random --num_irrelevant 0 --num_neutral 200

echo "=== All Sensitivity Analyses Completed ==="
