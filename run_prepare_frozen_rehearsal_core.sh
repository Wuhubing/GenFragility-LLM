#!/usr/bin/env bash
# Build and independently verify the fixed rehearsal core.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONDA=${CONDA:-"$HOME/miniconda3/bin/conda"}
EVAL_ENV=${EVAL_ENV:-ripple}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-9B}
DATA_DIR=${DATA_DIR:-data/external_eval/frozen_rehearsal_core}
OUTPUT_DIR=${OUTPUT_DIR:-main_output/frozen_rehearsal_core}
PROBE_DIR="$DATA_DIR/probes"
WBE_CANDIDATES=data/external_eval/rehearsal_smoke/candidates/wikibigedit_manifest.json
WFD_CANDIDATES=data/external_eval/rehearsal_smoke/candidates/wikifactdiff_manifest.json
PROBE_CANDIDATES="$PROBE_DIR/probe_candidates.json"
PROBE_PRECHECK="$OUTPUT_DIR/probe_candidate_precheck.json"
PROBE_BANK="$PROBE_DIR/probe_bank.json"
PROBE_RECHECK="$OUTPUT_DIR/probe_bank_recheck.json"
ANCHOR_CANDIDATES="$DATA_DIR/anchor_candidates.json"
ANCHOR_PRECHECK="$OUTPUT_DIR/anchor_candidate_precheck.json"
ANCHOR_PRECHECK_RECHECK="$OUTPUT_DIR/anchor_candidate_precheck_recheck.json"
ANCHOR_VALIDATION="$DATA_DIR/frozen_anchor_validation.json"
ANCHOR_RECHECK="$OUTPUT_DIR/frozen_anchor_recheck.json"

for required in "$WBE_CANDIDATES" "$WFD_CANDIDATES"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing exclusion manifest: $required"
        exit 1
    fi
done

export PYTHONPATH="$ROOT/src:$ROOT/scripts/external_eval:$ROOT:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-$HOME/hf_cache_home}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.85}
export VLLM_MAX_SEQS=${VLLM_MAX_SEQS:-128}

mkdir -p "$PROBE_DIR" "$OUTPUT_DIR"

if [[ ! -f "$PROBE_CANDIDATES" ]]; then
    python scripts/external_eval/prepare_rehearsal_probe_bank.py \
        --stage candidates \
        --exclude-manifest "$WBE_CANDIDATES" \
        --exclude-manifest "$WFD_CANDIDATES" \
        --out-dir "$PROBE_DIR" \
        --candidate-count 10000 \
        --seed 73
fi
if [[ ! -f "$PROBE_PRECHECK" ]]; then
    "$CONDA" run -n "$EVAL_ENV" python src/vllm_rehearsal_smoke_eval.py \
        --stage precheck-probes \
        --base-model "$BASE_MODEL" \
        --probe-manifest "$PROBE_CANDIDATES" \
        --output "$PROBE_PRECHECK"
fi
if [[ ! -f "$PROBE_BANK" ]]; then
    python scripts/external_eval/prepare_rehearsal_probe_bank.py \
        --stage finalize \
        --exclude-manifest "$WBE_CANDIDATES" \
        --exclude-manifest "$WFD_CANDIDATES" \
        --candidate-file "$PROBE_CANDIDATES" \
        --precheck-report "$PROBE_PRECHECK" \
        --out-dir "$PROBE_DIR" \
        --n-per-stratum 150 \
        --seed 73
fi
if [[ ! -f "$PROBE_RECHECK" ]]; then
    "$CONDA" run -n "$EVAL_ENV" python src/vllm_rehearsal_smoke_eval.py \
        --stage precheck-probes \
        --base-model "$BASE_MODEL" \
        --probe-manifest "$PROBE_BANK" \
        --output "$PROBE_RECHECK"
fi
python -c "
import json
from pathlib import Path
report = json.loads(Path('$PROBE_RECHECK').read_text())
clean = report['metadata']['clean_correct']
total = report['metadata']['total_probes']
if clean < 300:
    raise SystemExit(f'Frozen probe gate failed: {clean}/{total} clean-correct')
Path('$DATA_DIR/probe_verification.md').write_text(
    '# Frozen Probe Verification\\n\\n'
    f'- Status: PASS\\n- Total probes: {total}\\n'
    f'- Independent clean-correct recheck: {clean}\\n'
)
print(f'PASS: frozen probes clean-correct={clean}/{total}')
"

if [[ ! -f "$ANCHOR_CANDIDATES" ]]; then
    python scripts/external_eval/prepare_frozen_rehearsal_core.py \
        --stage candidates \
        --probe-bank "$PROBE_BANK" \
        --exclude-manifest "$WBE_CANDIDATES" \
        --exclude-manifest "$WFD_CANDIDATES" \
        --out-dir "$DATA_DIR" \
        --candidate-count 6000 \
        --random-candidate-count 15000 \
        --rare-candidate-count 43000 \
        --seed 79
fi
if [[ ! -f "$ANCHOR_PRECHECK" ]]; then
    "$CONDA" run -n "$EVAL_ENV" python src/vllm_rehearsal_smoke_eval.py \
        --stage precheck-probes \
        --base-model "$BASE_MODEL" \
        --probe-manifest "$ANCHOR_CANDIDATES" \
        --output "$ANCHOR_PRECHECK"
fi
if [[ ! -f "$ANCHOR_PRECHECK_RECHECK" ]]; then
    "$CONDA" run -n "$EVAL_ENV" python src/vllm_rehearsal_smoke_eval.py \
        --stage precheck-probes \
        --base-model "$BASE_MODEL" \
        --probe-manifest "$ANCHOR_CANDIDATES" \
        --output "$ANCHOR_PRECHECK_RECHECK"
fi
if [[ ! -f "$ANCHOR_VALIDATION" ]]; then
    python scripts/external_eval/prepare_frozen_rehearsal_core.py \
        --stage finalize \
        --probe-bank "$PROBE_BANK" \
        --candidate-file "$ANCHOR_CANDIDATES" \
        --precheck-report "$ANCHOR_PRECHECK" \
        --precheck-report "$ANCHOR_PRECHECK_RECHECK" \
        --out-dir "$DATA_DIR" \
        --n 100 \
        --seed 79
fi
verified=0
for repair_round in 0 1 2 3 4; do
    if [[ ! -f "$ANCHOR_RECHECK" ]]; then
        "$CONDA" run -n "$EVAL_ENV" python src/vllm_rehearsal_smoke_eval.py \
            --stage precheck-probes \
            --base-model "$BASE_MODEL" \
            --probe-manifest "$ANCHOR_VALIDATION" \
            --output "$ANCHOR_RECHECK"
    fi
    set +e
    python scripts/external_eval/prepare_frozen_rehearsal_core.py \
        --stage verify \
        --probe-bank "$PROBE_BANK" \
        --verification-report "$ANCHOR_RECHECK" \
        --out-dir "$DATA_DIR"
    verify_status=$?
    set -e
    if [[ "$verify_status" == "0" ]]; then
        verified=1
        break
    fi
    if [[ "$repair_round" == "4" ]]; then
        break
    fi
    python scripts/external_eval/prepare_frozen_rehearsal_core.py \
        --stage repair \
        --probe-bank "$PROBE_BANK" \
        --candidate-file "$ANCHOR_CANDIDATES" \
        --precheck-report "$ANCHOR_PRECHECK" \
        --precheck-report "$ANCHOR_PRECHECK_RECHECK" \
        --verification-report "$ANCHOR_RECHECK" \
        --out-dir "$DATA_DIR" \
        --seed "$((79 + repair_round + 1))"
    rm -f "$ANCHOR_RECHECK"
done
if [[ "$verified" != "1" ]]; then
    echo "Frozen anchor verification failed after repair rounds"
    exit 1
fi

echo "Frozen rehearsal core: $DATA_DIR"
