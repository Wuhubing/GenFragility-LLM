#!/usr/bin/env bash
# Build, precheck, freeze, and verify three atomic MQuAKE-CF B=25 batches.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONDA=${CONDA:-"$HOME/miniconda3/bin/conda"}
EVAL_ENV=${EVAL_ENV:-ripple}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-9B}
SOURCE=${SOURCE:-/tmp/mquake/MQuAKE-CF-3k.json}
DATA_DIR=data/external_eval/mquake_cf_confirmation
OUTPUT_DIR=main_output/external_rehearsal/mquake_cf_confirmation
CANDIDATE_MANIFEST="$DATA_DIR/candidates/manifest.json"
CANDIDATE_PRECHECK="$OUTPUT_DIR/candidate_precheck.json"
FINAL_MANIFEST="$DATA_DIR/manifest.json"
FINAL_PRECHECK="$OUTPUT_DIR/final_precheck.json"
SMOKE_MANIFEST="$DATA_DIR/smoke_manifest.json"
SMOKE_PRECHECK="$OUTPUT_DIR/smoke_precheck.json"

for required in \
    "$SOURCE" \
    data/external_eval/counterfact_confirmation/manifest.json \
    data/external_eval/frozen_rehearsal_core/frozen_verification.md \
    data/external_eval/frozen_rehearsal_core/probe_verification.md; do
    if [[ ! -f "$required" ]]; then
        echo "Missing required asset: $required"
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/external_eval:$ROOT:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-$HOME/huggingface_cache_large}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.85}
export VLLM_MAX_SEQS=${VLLM_MAX_SEQS:-128}

"$CONDA" run -n "$EVAL_ENV" python \
    scripts/external_eval/prepare_mquake_cf_confirmation.py \
    --stage build-candidates \
    --source "$SOURCE"

"$CONDA" run -n "$EVAL_ENV" python \
    src/vllm_rehearsal_smoke_eval.py \
    --stage precheck-manifest \
    --base-model "$BASE_MODEL" \
    --manifest "$CANDIDATE_MANIFEST" \
    --output "$CANDIDATE_PRECHECK"

"$CONDA" run -n "$EVAL_ENV" python \
    scripts/external_eval/prepare_mquake_cf_confirmation.py \
    --stage finalize \
    --source "$SOURCE" \
    --precheck-report "$CANDIDATE_PRECHECK"

"$CONDA" run -n "$EVAL_ENV" python \
    src/vllm_rehearsal_smoke_eval.py \
    --stage precheck-manifest \
    --base-model "$BASE_MODEL" \
    --manifest "$FINAL_MANIFEST" \
    --output "$FINAL_PRECHECK"

"$CONDA" run -n "$EVAL_ENV" python \
    scripts/external_eval/prepare_mquake_cf_confirmation.py \
    --stage build-smoke \
    --source "$SOURCE"

"$CONDA" run -n "$EVAL_ENV" python \
    src/vllm_rehearsal_smoke_eval.py \
    --stage precheck-manifest \
    --base-model "$BASE_MODEL" \
    --manifest "$SMOKE_MANIFEST" \
    --output "$SMOKE_PRECHECK"

python -c "
import json
from pathlib import Path
manifest = json.loads(Path('$FINAL_MANIFEST').read_text())
report = json.loads(Path('$FINAL_PRECHECK').read_text())
units = manifest['units']
eligible = [
    passed
    for unit in report['units'].values()
    for passed in unit['eligibility'].values()
]
if len(units) != 3 or any(len(unit['updates']) != 25 for unit in units.values()):
    raise SystemExit('MQuAKE-CF frozen matrix is not 3 x B25')
if len(eligible) != 75 or not all(eligible):
    raise SystemExit(
        f'MQuAKE-CF final verification failed: {sum(eligible)}/{len(eligible)}'
    )
print('PASS: MQuAKE-CF frozen 3 x B25 and final verification 75/75')
"
