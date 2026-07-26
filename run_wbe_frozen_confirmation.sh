#!/usr/bin/env bash
# Plan or run the fixed-anchor WikiBigEdit confirmation matrix.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ACTION=${1:-dry-run}
if [[ "$ACTION" != "dry-run" ]]; then
    echo "Only dry-run is enabled until the matrix is reviewed."
    exit 1
fi

CONDA=${CONDA:-"$HOME/miniconda3/bin/conda"}
TRAIN_ENV=${TRAIN_ENV:-genfragility}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-9B}
MANIFEST=data/external_eval/wbe_frozen_confirmation/wikibigedit/manifest.json
ANCHOR_DIR=data/external_eval/frozen_rehearsal_core
OUTPUT_BASE=main_output/external_rehearsal/wbe_frozen_confirmation
RUN_MANIFEST="$OUTPUT_BASE/dry_run_manifest.tsv"

for required in \
    "$MANIFEST" \
    "$ANCHOR_DIR/frozen_verification.md" \
    "$ANCHOR_DIR/probe_verification.md"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing frozen experiment asset: $required"
        exit 1
    fi
done

python -c "
from pathlib import Path
for path in (
    Path('$ANCHOR_DIR/frozen_verification.md'),
    Path('$ANCHOR_DIR/probe_verification.md'),
):
    if 'Status: PASS' not in path.read_text():
        raise SystemExit(f'Frozen asset did not pass: {path}')
"

mkdir -p "$OUTPUT_BASE"
mapfile -t UNITS < <(
    python -c "
import json
from pathlib import Path
print(*json.loads(Path('$MANIFEST').read_text())['units'], sep='\n')
"
)
if [[ "${#UNITS[@]}" != "3" ]]; then
    echo "Expected 3 WikiBigEdit batches, got ${#UNITS[@]}"
    exit 1
fi

MODES=(none popular random rare random_distance)
SEEDS=(42 43)
printf 'dataset\tunit\tseed\tmode\tstatus\n' > "$RUN_MANIFEST"
planned=0
for seed in "${SEEDS[@]}"; do
    for unit_id in "${UNITS[@]}"; do
        for mode in "${MODES[@]}"; do
            out_dir="$OUTPUT_BASE/seed${seed}/wikibigedit/$mode/$unit_id"
            "$CONDA" run -n "$TRAIN_ENV" python \
                scripts/train_wikibigedit_rehearsal_smoke.py \
                --manifest "$MANIFEST" \
                --unit-id "$unit_id" \
                --mode "$mode" \
                --base-model "$BASE_MODEL" \
                --output-dir "$out_dir" \
                --frozen-anchor-dir "$ANCHOR_DIR" \
                --anchor-count 100 \
                --seed "$seed" \
                --repeats-per-update 20 \
                --epochs 3 \
                --dry-run
            printf 'wikibigedit\t%s\t%s\t%s\tplanned\n' \
                "$unit_id" "$seed" "$mode" >> "$RUN_MANIFEST"
            planned=$((planned + 1))
        done
    done
done
if [[ "$planned" != "30" ]]; then
    echo "Expected 30 planned runs, got $planned"
    exit 1
fi
echo "PASS: planned $planned fixed-anchor WikiBigEdit runs"
echo "Run manifest: $RUN_MANIFEST"
