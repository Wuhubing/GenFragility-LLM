#!/usr/bin/env bash
# Gemma-4-31B-it on the dual-eligible (Qwen∩Gemma) frozen manifests.
# CounterFact B=100 and MQuAKE-CF B=72, both at exact 20% rehearsal ratio.
#
# Usage:
#   bash run_second_machine_gemma31b.sh dry-run
#   bash run_second_machine_gemma31b.sh run
#   MODES="none popular random" bash run_second_machine_gemma31b.sh run

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ACTION=${1:-dry-run}
if [[ "$ACTION" != "dry-run" && "$ACTION" != "run" ]]; then
    echo "Usage: $0 [dry-run|run]"
    exit 1
fi

MODES=${MODES:-"none popular rare random similarity"}

CF_MANIFEST=data/external_eval/counterfact_dual_eligible_b100/manifest.json
CF_EXPERIMENT_DIR=data/external_eval/counterfact_dual_eligible_b100/experiments
CF_PRECHECK=data/external_eval/counterfact_dual_eligible_b100/prechecks/gemma31b_batch.json
CF_OUTPUT=main_output/external_rehearsal/counterfact_dual_gemma31b

MQ_MANIFEST=data/external_eval/mquake_dual_eligible_b80/manifest.json
MQ_PRECHECK=data/external_eval/mquake_dual_eligible_b80/prechecks/gemma31b_batch.json
MQ_OUTPUT=main_output/external_rehearsal/mquake_dual_gemma31b/seed42

for required in \
    "$CF_MANIFEST" \
    "$CF_PRECHECK" \
    "$MQ_MANIFEST" \
    "$MQ_PRECHECK" \
    data/external_eval/frozen_rehearsal_core/probes/probe_bank.json \
    data/external_eval/counterfact_dual_eligible_b100/anchors_popular_object_top100.json \
    data/external_eval/mquake_dual_eligible_b80/anchors_popular_object_top72.json; do
    if [[ ! -e "$required" ]]; then
        echo "Missing required asset: $required"
        exit 1
    fi
done

export HF_HOME=${HF_HOME:-$HOME/huggingface_cache_large}

echo "=== Gemma 31B main: CounterFact dual-eligible B=100, ratio=20% ==="
MODES="$MODES" BATCHES=counterfact_batch_001 AR=4 \
MANIFEST="$CF_MANIFEST" \
EXPERIMENT_DIR="$CF_EXPERIMENT_DIR" \
PRECHECK_REPORT="$CF_PRECHECK" \
OUTPUT_BASE="$CF_OUTPUT" \
    bash run_gemma31b_main.sh "$ACTION"

echo "=== Gemma 31B main: MQuAKE-CF dual-eligible B=72, ratio=20% ==="
MODES="$MODES" AR=4 ANCHOR_COUNT=72 \
MANIFEST="$MQ_MANIFEST" \
PRECHECK="$MQ_PRECHECK" \
OUTPUT_BASE="$MQ_OUTPUT" \
    bash run_mquake_main.sh "$ACTION" gemma31b
