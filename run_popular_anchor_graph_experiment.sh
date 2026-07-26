#!/usr/bin/env bash
# Run the preregistered Popular-100 graph holdout experiment.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ACTION=${1:-pilot}
DATA_DIR=${DATA_DIR:-data/external_eval/rehearsal_popular_graph}
WBE_MANIFEST="$DATA_DIR/wikibigedit/manifest.json"
PROBE_MANIFEST=${PROBE_MANIFEST:-data/external_eval/rehearsal_graph_probe/probe_bank.json}
SHARED_PRECHECK=${SHARED_PRECHECK:-main_output/popular_anchor_graph/precheck_strict.json}

if [[ "$ACTION" == "mquake-pilot" ]]; then
    exec ./run_mquake_rehearsal_pilot.sh
fi

if [[ "$ACTION" != "pilot" && "$ACTION" != "scale" \
    && "$ACTION" != "wbe-confirm" && "$ACTION" != "wfd-replicate" \
    && "$ACTION" != "dry-run" ]]; then
    echo "Usage: $0 [mquake-pilot|wbe-confirm|wfd-replicate|pilot|scale|dry-run]"
    exit 1
fi

for required in "$WBE_MANIFEST" "$PROBE_MANIFEST"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing required file: $required"
        exit 1
    fi
done

run_seed() {
    local seed=$1
    local output_base=$2
    local all_units=$3

    python scripts/external_eval/generate_rehearsal_smoke_anchors.py \
        --manifest "$WBE_MANIFEST" \
        --probe-manifest "$PROBE_MANIFEST" \
        --out-dir "$DATA_DIR/wikibigedit" \
        --n 100 \
        --seed "$seed"

    args=(
        --dataset wikibigedit
        --probe-manifest "$PROBE_MANIFEST"
    )
    if [[ "$all_units" == "1" ]]; then
        args+=(--all-wbe-units)
    fi
    if [[ "$ACTION" == "dry-run" ]]; then
        args+=(--dry-run)
    fi

    WBE_MANIFEST="$WBE_MANIFEST" \
    OUTPUT_BASE="$output_base" \
    PRECHECK_REPORT="$SHARED_PRECHECK" \
    ANCHOR_SEED="$seed" \
        ./run_rehearsal_smoke.sh "${args[@]}"
}

if [[ "$ACTION" == "wbe-confirm" ]]; then
    python -c "
import json
from pathlib import Path
summary = Path('main_output/external_rehearsal/mquake_t/pilot/seed42/mquake_pilot_summary.json')
if not summary.is_file() or not json.loads(summary.read_text())['pilot_gate']['passed']:
    raise SystemExit('MQuAKE-T pilot gate did not pass; WBE confirmation is blocked')
"
    for seed in 42 43; do
        MODES_CSV=none,popular,random,random_distance \
        UNIT_LIMIT=3 \
            run_seed \
                "$seed" \
                "main_output/external_rehearsal/wbe_confirm/seed${seed}" \
                1
    done
    python scripts/summarize_rehearsal_smoke.py \
        --output-base main_output/external_rehearsal/wbe_confirm \
        --graph-probe \
        --include-seed-subdirs \
        --modes none,popular,random,random_distance
    python scripts/summarize_rehearsal_smoke.py \
        --output-base main_output/external_rehearsal \
        --external-summary
elif [[ "$ACTION" == "wfd-replicate" ]]; then
    python -c "
import json
from pathlib import Path
summary = Path('main_output/external_rehearsal/wbe_confirm/graph_probe_summary.json')
if not summary.is_file() or not json.loads(summary.read_text())['confirmation_gate']['passed']:
    raise SystemExit('WBE confirmation gate did not pass; WFD replication is blocked')
"
    WFD_ROOT=data/external_eval/rehearsal_wfd_replication
    CANDIDATE_DIR=data/external_eval/rehearsal_smoke/candidates
    WFD_OUTPUT=main_output/external_rehearsal/wfd_replication
    CANDIDATE_PRECHECK="$WFD_OUTPUT/candidate_precheck.json"
    FINAL_PRECHECK="$WFD_OUTPUT/final_precheck.json"
    mkdir -p "$WFD_ROOT" "$WFD_OUTPUT"
    if [[ ! -f "$CANDIDATE_PRECHECK" ]]; then
        "$HOME/miniconda3/bin/conda" run -n ripple \
            python src/vllm_rehearsal_smoke_eval.py \
            --stage precheck \
            --base-model Qwen/Qwen3.5-9B \
            --wfd-manifest "$CANDIDATE_DIR/wikifactdiff_manifest.json" \
            --wfd-experiment-dir data/external_eval/block_b_experiments/wikifactdiff \
            --wbe-manifest "$CANDIDATE_DIR/wikibigedit_manifest.json" \
            --output "$CANDIDATE_PRECHECK"
    fi
    python scripts/external_eval/select_model_eligible_rehearsal_smoke.py \
        --stage finalize \
        --candidate-dir "$CANDIDATE_DIR" \
        --out-dir "$WFD_ROOT" \
        --precheck-report "$CANDIDATE_PRECHECK" \
        --probe-manifest data/external_eval/rehearsal_mquake_t/probes/probe_bank.json \
        --wfd-target-count 25 \
        --wfd-batch-count 3 \
        --wikibigedit-batch-size 8 \
        --wikibigedit-batch-count 1 \
        --seed 42
    if [[ ! -f "$FINAL_PRECHECK" ]]; then
        "$HOME/miniconda3/bin/conda" run -n ripple \
            python src/vllm_rehearsal_smoke_eval.py \
            --stage precheck \
            --base-model Qwen/Qwen3.5-9B \
            --wfd-manifest "$WFD_ROOT/wikifactdiff/manifest.json" \
            --wfd-experiment-dir data/external_eval/block_b_experiments/wikifactdiff \
            --wbe-manifest "$WFD_ROOT/wikibigedit/manifest.json" \
            --output "$FINAL_PRECHECK"
    fi
    for seed in 42 43; do
        python scripts/external_eval/generate_rehearsal_smoke_anchors.py \
            --manifest "$WFD_ROOT/wikifactdiff/manifest.json" \
            --probe-manifest data/external_eval/rehearsal_mquake_t/probes/probe_bank.json \
            --out-dir "$WFD_ROOT/wikifactdiff" \
            --n 100 \
            --seed "$seed"
        WFD_MANIFEST="$WFD_ROOT/wikifactdiff/manifest.json" \
        WBE_MANIFEST="$WFD_ROOT/wikibigedit/manifest.json" \
        PRECHECK_REPORT="$FINAL_PRECHECK" \
        OUTPUT_BASE="$WFD_OUTPUT/seed${seed}" \
        ANCHOR_SEED="$seed" \
        MODES_CSV=none,popular,random \
        UNIT_LIMIT=3 \
            ./run_rehearsal_smoke.sh \
                --dataset wikifactdiff \
                --all-wfd-units
    done
    python scripts/summarize_rehearsal_smoke.py \
        --output-base "$WFD_OUTPUT" \
        --wfd-replication
    python scripts/summarize_rehearsal_smoke.py \
        --output-base main_output/external_rehearsal \
        --external-summary
elif [[ "$ACTION" == "pilot" || "$ACTION" == "dry-run" ]]; then
    run_seed 42 main_output/popular_anchor_graph/pilot/seed42 0
else
    for seed in 42 43 44; do
        run_seed \
            "$seed" \
            "main_output/popular_anchor_graph/scale/seed${seed}" \
            1
    done
    python scripts/summarize_rehearsal_smoke.py \
        --output-base main_output/popular_anchor_graph/scale \
        --graph-probe \
        --include-seed-subdirs
fi
