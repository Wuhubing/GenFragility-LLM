#!/usr/bin/env bash
# Gemma-4-31B-it validation on CounterFact B=100 and MQuAKE-CF B=80.
# Both datasets use an exact 20% anchor:update sample ratio.
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
CF_MANIFEST=data/external_eval/counterfact_confirmation/manifest.json
CF_PRECHECK=main_output/external_rehearsal/counterfact_gemma31b/precheck_b100.json
MQ_MANIFEST=data/external_eval/mquake_b100_confirmation/manifest.json
MQ_PRECHECK=main_output/external_rehearsal/mquake_gemma31b/precheck_b80.json

for required in \
    "$CF_MANIFEST" \
    "$MQ_MANIFEST" \
    data/external_eval/frozen_rehearsal_core/probes/probe_bank.json; do
    if [[ ! -f "$required" ]]; then
        echo "Missing required asset: $required"
        exit 1
    fi
done

export HF_HOME=${HF_HOME:-$HOME/huggingface_cache_large}

if [[ "$ACTION" == "run" && ! -f "$CF_PRECHECK" ]]; then
    echo "=== Gemma 31B precheck: CounterFact ==="
    MANIFEST="$CF_MANIFEST" OUTPUT="$CF_PRECHECK" \
        bash run_gemma31b_precheck.sh
fi

echo "=== Gemma 31B main: CounterFact B=100, ratio=20% ==="
MODES="$MODES" BATCHES=counterfact_batch_001 AR=4 \
PRECHECK_REPORT="$CF_PRECHECK" \
    bash run_gemma31b_main.sh "$ACTION"

if [[ "$ACTION" == "run" && ! -f "$MQ_PRECHECK" ]]; then
    echo "=== Gemma 31B precheck: MQuAKE-CF ==="
    MANIFEST="$MQ_MANIFEST" OUTPUT="$MQ_PRECHECK" \
        bash run_gemma31b_precheck.sh
fi

echo "=== Gemma 31B main: MQuAKE-CF B=80, ratio=20% ==="
MODES="$MODES" AR=4 ANCHOR_COUNT=80 PRECHECK="$MQ_PRECHECK" \
    bash run_mquake_main.sh "$ACTION" gemma31b
