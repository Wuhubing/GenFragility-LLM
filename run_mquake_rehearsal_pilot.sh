#!/usr/bin/env bash
# Run the preregistered MQuAKE-T B=25 three-arm rehearsal pilot.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONDA=${CONDA:-"$HOME/miniconda3/bin/conda"}
TRAIN_ENV=${TRAIN_ENV:-genfragility}
EVAL_ENV=${EVAL_ENV:-ripple}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-9B}
SOURCE=${SOURCE:-/tmp/mquake/MQuAKE-T.json}
DATA_DIR=${DATA_DIR:-data/external_eval/rehearsal_mquake_t}
PROBE_DIR="$DATA_DIR/probes"
MANIFEST="$DATA_DIR/manifest.json"
CANDIDATE_MANIFEST="$DATA_DIR/candidate_manifest.json"
PROBE_MANIFEST="$PROBE_DIR/probe_bank.json"
OUTPUT_BASE=${OUTPUT_BASE:-main_output/external_rehearsal/mquake_t/pilot/seed42}
CANDIDATE_PRECHECK="$OUTPUT_BASE/candidate_precheck.json"
FINAL_PRECHECK="$OUTPUT_BASE/final_precheck.json"
PROBE_PRECHECK="$OUTPUT_BASE/probe_precheck.json"
ANCHOR_COUNT=${ANCHOR_COUNT:-100}
SEED=${SEED:-42}
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi
if [[ ! -f "$SOURCE" ]]; then
    echo "Missing MQuAKE-T source: $SOURCE"
    exit 1
fi

export DISABLE_VERSION_CHECK=1
export PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-$HOME/hf_cache_home}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.85}
export VLLM_MAX_SEQS=${VLLM_MAX_SEQS:-128}
export LF_BATCH_SIZE=${LF_BATCH_SIZE:-2}
export LF_GRAD_ACCUM=${LF_GRAD_ACCUM:-4}

mkdir -p "$DATA_DIR" "$PROBE_DIR" "$OUTPUT_BASE"

python scripts/external_eval/prepare_mquake_rehearsal_pilot.py \
    --stage candidates \
    --source "$SOURCE" \
    --out-dir "$DATA_DIR" \
    --candidate-count 96 \
    --seed "$SEED"

if [[ "$DRY_RUN" == "0" && ! -f "$CANDIDATE_PRECHECK" ]]; then
    "$CONDA" run -n "$EVAL_ENV" python src/vllm_rehearsal_smoke_eval.py \
        --stage precheck-manifest \
        --base-model "$BASE_MODEL" \
        --manifest "$CANDIDATE_MANIFEST" \
        --output "$CANDIDATE_PRECHECK"
fi
if [[ "$DRY_RUN" == "0" ]]; then
    set +e
    python scripts/external_eval/prepare_mquake_rehearsal_pilot.py \
        --stage finalize \
        --source "$SOURCE" \
        --out-dir "$DATA_DIR" \
        --precheck-report "$CANDIDATE_PRECHECK" \
        --batch-size 25 \
        --seed "$SEED"
    finalize_status=$?
    set -e
    if [[ "$finalize_status" != "0" ]]; then
        python scripts/summarize_rehearsal_smoke.py \
            --output-base "$OUTPUT_BASE" \
            --mquake-preflight \
            --candidate-manifest "$CANDIDATE_MANIFEST" \
            --precheck-report "$CANDIDATE_PRECHECK"
        python scripts/summarize_rehearsal_smoke.py \
            --output-base main_output/external_rehearsal \
            --external-summary
        exit 0
    fi
elif [[ ! -f "$MANIFEST" ]]; then
    echo "Dry-run requires an existing finalized manifest: $MANIFEST"
    exit 1
fi

if [[ ! -f "$PROBE_DIR/probe_candidates.json" ]]; then
    python scripts/external_eval/prepare_rehearsal_probe_bank.py \
        --stage candidates \
        --exclude-manifest "$CANDIDATE_MANIFEST" \
        --out-dir "$PROBE_DIR" \
        --candidate-count 10000 \
        --seed 43
fi
if [[ "$DRY_RUN" == "0" && ! -f "$PROBE_PRECHECK" ]]; then
    "$CONDA" run -n "$EVAL_ENV" python src/vllm_rehearsal_smoke_eval.py \
        --stage precheck-probes \
        --base-model "$BASE_MODEL" \
        --probe-manifest "$PROBE_DIR/probe_candidates.json" \
        --output "$PROBE_PRECHECK"
fi
if [[ "$DRY_RUN" == "0" && ! -f "$PROBE_MANIFEST" ]]; then
    python scripts/external_eval/prepare_rehearsal_probe_bank.py \
        --stage finalize \
        --exclude-manifest "$CANDIDATE_MANIFEST" \
        --candidate-file "$PROBE_DIR/probe_candidates.json" \
        --precheck-report "$PROBE_PRECHECK" \
        --out-dir "$PROBE_DIR" \
        --n-per-stratum 150 \
        --seed 43
fi
if [[ ! -f "$PROBE_MANIFEST" ]]; then
    echo "Missing finalized probe manifest: $PROBE_MANIFEST"
    exit 1
fi

python scripts/external_eval/generate_rehearsal_smoke_anchors.py \
    --manifest "$MANIFEST" \
    --probe-manifest "$PROBE_MANIFEST" \
    --out-dir "$DATA_DIR" \
    --n "$ANCHOR_COUNT" \
    --seed "$SEED"

if [[ "$DRY_RUN" == "0" && ! -f "$FINAL_PRECHECK" ]]; then
    "$CONDA" run -n "$EVAL_ENV" python src/vllm_rehearsal_smoke_eval.py \
        --stage precheck-manifest \
        --base-model "$BASE_MODEL" \
        --manifest "$MANIFEST" \
        --output "$FINAL_PRECHECK"
fi

UNIT_ID=$(
    python -c "
import json
from pathlib import Path
print(next(iter(json.loads(Path('$MANIFEST').read_text())['units'])))
"
)
MODES=(none popular random)
printf 'dataset\tunit\tmode\tstatus\n' > "$OUTPUT_BASE/run_manifest.tsv"

for mode in "${MODES[@]}"; do
    out_dir="$OUTPUT_BASE/mquake_t/$mode/$UNIT_ID"
    train_args=(
        --manifest "$MANIFEST"
        --unit-id "$UNIT_ID"
        --mode "$mode"
        --base-model "$BASE_MODEL"
        --output-dir "$out_dir"
        --precheck-report "$FINAL_PRECHECK"
        --anchor-count "$ANCHOR_COUNT"
        --anchor-seed "$SEED"
        --seed "$SEED"
        --repeats-per-update 20
        --epochs 3
    )
    if [[ "$DRY_RUN" == "1" ]]; then
        "$CONDA" run -n "$TRAIN_ENV" python \
            scripts/train_wikibigedit_rehearsal_smoke.py \
            "${train_args[@]}" --dry-run
        printf 'mquake_t\t%s\t%s\tplanned\n' "$UNIT_ID" "$mode" \
            >> "$OUTPUT_BASE/run_manifest.tsv"
        continue
    fi

    "$CONDA" run -n "$TRAIN_ENV" python \
        scripts/train_wikibigedit_rehearsal_smoke.py \
        "${train_args[@]}"
    lora_path="$out_dir/adapter"
    native_report="$out_dir/evaluation_strict.json"
    probe_report="$out_dir/graph_probe_evaluation.json"

    if [[ ! -f "$native_report" ]]; then
        set +e
        "$CONDA" run -n "$EVAL_ENV" python src/vllm_rehearsal_smoke_eval.py \
            --stage evaluate-mquake \
            --base-model "$BASE_MODEL" \
            --manifest "$MANIFEST" \
            --unit-id "$UNIT_ID" \
            --lora-path "$lora_path" \
            --output "$native_report"
        status=$?
        set -e
        if [[ "$status" != "0" && ! -f "$native_report" ]]; then
            exit "$status"
        fi
    fi
    if [[ ! -f "$probe_report" ]]; then
        set +e
        "$CONDA" run -n "$EVAL_ENV" python src/vllm_rehearsal_smoke_eval.py \
            --stage evaluate-probes \
            --base-model "$BASE_MODEL" \
            --probe-manifest "$PROBE_MANIFEST" \
            --lora-path "$lora_path" \
            --output "$probe_report"
        status=$?
        set -e
        if [[ "$status" != "0" && ! -f "$probe_report" ]]; then
            exit "$status"
        fi
    fi
    printf 'mquake_t\t%s\t%s\tcomplete\n' "$UNIT_ID" "$mode" \
        >> "$OUTPUT_BASE/run_manifest.tsv"
done

if [[ "$DRY_RUN" == "0" ]]; then
    python scripts/summarize_rehearsal_smoke.py \
        --output-base "$OUTPUT_BASE" \
        --mquake-pilot
    python scripts/summarize_rehearsal_smoke.py \
        --output-base main_output/external_rehearsal \
        --external-summary
fi
