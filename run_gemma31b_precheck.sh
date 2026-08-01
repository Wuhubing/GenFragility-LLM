#!/usr/bin/env bash
# Precheck for Gemma-4-31B-it on the B=100 CounterFact manifest.
# Must run BEFORE training so the training script can validate eligibility.
# Requires ~62GB VRAM (Gemma 31B BF16) — run when GPU is free.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONDA=${CONDA:-"$HOME/miniconda3/bin/conda"}
EVAL_ENV=${EVAL_ENV:-ripple}
BASE_MODEL=${BASE_MODEL:-google/gemma-4-31B-it}
MANIFEST=data/external_eval/counterfact_confirmation/manifest.json
OUTPUT=main_output/external_rehearsal/counterfact_gemma31b/precheck_b100.json

export PYTHONPATH="$ROOT/src:$ROOT/scripts/external_eval:$ROOT:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-$HOME/huggingface_cache_large}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.90}
export VLLM_MAX_SEQS=${VLLM_MAX_SEQS:-32}
export DISABLE_VERSION_CHECK=${DISABLE_VERSION_CHECK:-1}

mkdir -p "$(dirname "$OUTPUT")"

"$CONDA" run -n "$EVAL_ENV" python \
    src/vllm_rehearsal_smoke_eval.py \
    --stage precheck-manifest \
    --base-model "$BASE_MODEL" \
    --manifest "$MANIFEST" \
    --output "$OUTPUT"

echo "Precheck complete: $OUTPUT"
