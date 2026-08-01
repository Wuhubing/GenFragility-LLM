#!/usr/bin/env bash
# Prepare WBE and WFD manifests at B=100 with per-batch anchors.
# This script does:
#   1. Precheck WFD candidates (needs GPU ~24GB for Qwen 9B)
#   2. Precheck WBE candidates (needs GPU)
#   3. Finalize WFD manifest at B=100 with per-batch dedup
#   4. Finalize WBE manifest at B=100 with per-batch dedup
#   5. Generate per-batch anchors (popular/rare/random) for both
#   6. Generate similarity anchors for both
#
# Prerequisite: GPU must be free (no other vLLM or training running).
# Usage: bash run_prepare_wbe_wfd_b100.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONDA=${CONDA:-"$HOME/miniconda3/bin/conda"}
EVAL_ENV=${EVAL_ENV:-ripple}
TRAIN_ENV=${TRAIN_ENV:-genfragility}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-9B}

export PYTHONPATH="$ROOT/src:$ROOT/scripts/external_eval:$ROOT:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-$HOME/huggingface_cache_large}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.85}
export VLLM_MAX_SEQS=${VLLM_MAX_SEQS:-128}
export DISABLE_VERSION_CHECK=${DISABLE_VERSION_CHECK:-1}

GRAPH="results/checkpoints/final.pkl"
PROBE_MANIFEST=data/external_eval/frozen_rehearsal_core/probes/probe_bank.json

# ─── WFD (WikiFactDiff) ───
WFD_DIR=data/external_eval/wfd_full_confirmation
WFD_CANDIDATES=$WFD_DIR/candidates/manifest.json
WFD_PRECHECK=$WFD_DIR/precheck_b100.json
WFD_MANIFEST=$WFD_DIR/manifest_b100.json

# ─── WBE (WikiBigEdit) ───
WBE_CANDIDATE_DIR=data/external_eval/wbe_frozen_confirmation/candidates
WBE_CANDIDATES=$WBE_CANDIDATE_DIR/wikibigedit_manifest.json
WBE_PRECHECK=data/external_eval/wbe_frozen_confirmation/precheck_b100.json
WBE_OUT_DIR=data/external_eval/wbe_b100_confirmation

echo "=========================================="
echo "Step 1: Precheck WFD candidates (4810 updates)"
echo "=========================================="
if [[ ! -f "$WFD_PRECHECK" ]]; then
    "$CONDA" run -n "$EVAL_ENV" python \
        src/vllm_rehearsal_smoke_eval.py \
        --stage precheck-manifest \
        --base-model "$BASE_MODEL" \
        --manifest "$WFD_CANDIDATES" \
        --output "$WFD_PRECHECK"
else
    echo "  Already exists: $WFD_PRECHECK"
fi

echo "=========================================="
echo "Step 2: Precheck WBE candidates (512 updates)"
echo "=========================================="
# WBE precheck uses --stage precheck which needs both wfd and wbe manifests.
# We use the WBE candidate manifest and a dummy WFD manifest (the WFD candidate).
if [[ ! -f "$WBE_PRECHECK" ]]; then
    "$CONDA" run -n "$EVAL_ENV" python \
        src/vllm_rehearsal_smoke_eval.py \
        --stage precheck \
        --base-model "$BASE_MODEL" \
        --wfd-manifest "$WBE_CANDIDATE_DIR/wikifactdiff_manifest.json" \
        --wfd-experiment-dir data/external_eval/block_b_experiments/wikifactdiff \
        --wbe-manifest "$WBE_CANDIDATES" \
        --output "$WBE_PRECHECK"
else
    echo "  Already exists: $WBE_PRECHECK"
fi

echo "=========================================="
echo "Step 3: Finalize WFD manifest at B=100 (per-batch dedup)"
echo "=========================================="
if [[ ! -f "$WFD_MANIFEST" ]]; then
    "$CONDA" run -n "$TRAIN_ENV" python \
        scripts/external_eval/prepare_wfd_full_confirmation.py \
        --stage finalize \
        --precheck-report "$WFD_PRECHECK" \
        --batch-count 3 \
        --batch-size 100 \
        --dedup-mode per-batch \
        --out-dir data/external_eval/wfd_b100_confirmation
    cp data/external_eval/wfd_b100_confirmation/manifest.json "$WFD_MANIFEST"
else
    echo "  Already exists: $WFD_MANIFEST"
fi

echo "=========================================="
echo "Step 4: Finalize WBE manifest at B=100 (per-batch dedup)"
echo "=========================================="
WBE_MANIFEST=$WBE_OUT_DIR/wikibigedit/manifest.json
if [[ ! -f "$WBE_MANIFEST" ]]; then
    "$CONDA" run -n "$TRAIN_ENV" python \
        scripts/external_eval/select_model_eligible_rehearsal_smoke.py \
        --stage finalize \
        --precheck-report "$WBE_PRECHECK" \
        --probe-manifest "$PROBE_MANIFEST" \
        --candidate-dir "$WBE_CANDIDATE_DIR" \
        --wikibigedit-batch-size 100 \
        --wikibigedit-batch-count 3 \
        --wikibigedit-candidate-count 512 \
        --dedup-mode per-batch \
        --out-dir "$WBE_OUT_DIR"
else
    echo "  Already exists: $WBE_MANIFEST"
fi

echo "=========================================="
echo "Step 5: Generate per-batch anchors (popular/rare/random) for WFD"
echo "=========================================="
WFD_ANCHOR_DIR=$(dirname "$WFD_MANIFEST")
if [[ ! -f "$WFD_ANCHOR_DIR/anchors_popular_object_top100.json" ]]; then
    "$CONDA" run -n "$TRAIN_ENV" python \
        scripts/external_eval/generate_rehearsal_smoke_anchors.py \
        --manifest "$WFD_MANIFEST" \
        --graph-path "$GRAPH" \
        --out-dir "$WFD_ANCHOR_DIR" \
        --probe-manifest "$PROBE_MANIFEST" \
        --n 100 \
        --seed 42
else
    echo "  Already exists: $WFD_ANCHOR_DIR/anchors_popular_object_top100.json"
fi

echo "=========================================="
echo "Step 6: Generate per-batch anchors (popular/rare/random) for WBE"
echo "=========================================="
WBE_ANCHOR_DIR=$(dirname "$WBE_MANIFEST")
if [[ ! -f "$WBE_ANCHOR_DIR/anchors_popular_object_top100.json" ]]; then
    "$CONDA" run -n "$TRAIN_ENV" python \
        scripts/external_eval/generate_rehearsal_smoke_anchors.py \
        --manifest "$WBE_MANIFEST" \
        --graph-path "$GRAPH" \
        --out-dir "$WBE_ANCHOR_DIR" \
        --probe-manifest "$PROBE_MANIFEST" \
        --n 100 \
        --seed 42
else
    echo "  Already exists: $WBE_ANCHOR_DIR/anchors_popular_object_top100.json"
fi

echo "=========================================="
echo "Step 7: Generate similarity anchors for WFD"
echo "=========================================="
if [[ ! -f "$WFD_ANCHOR_DIR/anchors_similarity_object_top100.json" ]]; then
    "$CONDA" run -n "$TRAIN_ENV" python \
        scripts/external_eval/select_anchors_similarity.py \
        --manifest "$WFD_MANIFEST" \
        --out-dir "$WFD_ANCHOR_DIR" \
        --n 100
else
    echo "  Already exists: $WFD_ANCHOR_DIR/anchors_similarity_object_top100.json"
fi

echo "=========================================="
echo "Step 8: Generate similarity anchors for WBE"
echo "=========================================="
if [[ ! -f "$WBE_ANCHOR_DIR/anchors_similarity_object_top100.json" ]]; then
    "$CONDA" run -n "$TRAIN_ENV" python \
        scripts/external_eval/select_anchors_similarity.py \
        --manifest "$WBE_MANIFEST" \
        --out-dir "$WBE_ANCHOR_DIR" \
        --n 100
else
    echo "  Already exists: $WBE_ANCHOR_DIR/anchors_similarity_object_top100.json"
fi

echo ""
echo "=========================================="
echo "DONE! All WBE and WFD B=100 manifests and anchors are ready."
echo "=========================================="
echo "WFD manifest: $WFD_MANIFEST"
echo "WBE manifest: $WBE_MANIFEST"
echo "WFD anchors:  $WFD_ANCHOR_DIR/"
echo "WBE anchors:  $WBE_ANCHOR_DIR/"
