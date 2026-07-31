#!/usr/bin/env bash
# Run the CounterFact technical smoke or frozen 30-run confirmation.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ACTION=${1:-dry-run}
if [[ "$ACTION" != "smoke" && "$ACTION" != "dry-run" && "$ACTION" != "run" ]]; then
    echo "Usage: $0 [smoke|dry-run|run]"
    exit 1
fi

CONDA=${CONDA:-"$HOME/miniconda3/bin/conda"}
TRAIN_ENV=${TRAIN_ENV:-genfragility}
EVAL_ENV=${EVAL_ENV:-ripple}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-9B}
FULL_MANIFEST=data/external_eval/counterfact_confirmation/manifest.json
SMOKE_MANIFEST=data/external_eval/counterfact_confirmation/smoke_manifest.json
EXPERIMENT_DIR=data/external_eval/counterfact_confirmation/experiments
ANCHOR_DIR=data/external_eval/frozen_rehearsal_core
PROBE_MANIFEST="$ANCHOR_DIR/probes/probe_bank.json"
FULL_PRECHECK=main_output/external_rehearsal/counterfact_confirmation/final_precheck.json
SMOKE_PRECHECK=main_output/external_rehearsal/counterfact_confirmation/smoke_precheck.json
OUTPUT_BASE=main_output/external_rehearsal/counterfact_confirmation

if [[ "$ACTION" == "smoke" ]]; then
    MANIFEST="$SMOKE_MANIFEST"
    PRECHECK_REPORT="$SMOKE_PRECHECK"
    RUN_BASE="$OUTPUT_BASE/smoke"
    MODES=(none popular)
    SEEDS=(47)
    EXPECTED_RUNS=2
else
    MANIFEST="$FULL_MANIFEST"
    PRECHECK_REPORT="$FULL_PRECHECK"
    RUN_BASE="$OUTPUT_BASE"
    MODES=(none popular random rare random_distance)
    SEEDS=(42 43)
    EXPECTED_RUNS=30
fi
RUN_MANIFEST="$RUN_BASE/${ACTION}_manifest.tsv"

for required in \
    "$MANIFEST" \
    "$PRECHECK_REPORT" \
    "$PROBE_MANIFEST" \
    "$ANCHOR_DIR/frozen_verification.md" \
    "$ANCHOR_DIR/probe_verification.md"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing frozen experiment asset: $required"
        exit 1
    fi
done

expected_updates=75
if [[ "$ACTION" == "smoke" ]]; then
    expected_updates=5
fi
python -c "
import json
from pathlib import Path
manifest = json.loads(Path('$MANIFEST').read_text())
report = json.loads(Path('$PRECHECK_REPORT').read_text())
eligibility = [
    passed
    for unit in report['units'].values()
    for passed in unit['eligibility'].values()
]
if len(eligibility) != $expected_updates or not all(eligibility):
    raise SystemExit(
        f'CounterFact precheck is not complete: {sum(eligibility)}/{len(eligibility)}'
    )
if '$ACTION' != 'smoke' and manifest['metadata'].get('status') != 'frozen':
    raise SystemExit('CounterFact manifest is not frozen')
for path in (
    Path('$ANCHOR_DIR/frozen_verification.md'),
    Path('$ANCHOR_DIR/probe_verification.md'),
):
    if 'Status: PASS' not in path.read_text():
        raise SystemExit(f'Frozen asset did not pass: {path}')
"

mkdir -p "$RUN_BASE"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/external_eval:$ROOT:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-$HOME/huggingface_cache_large}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.85}
export VLLM_MAX_SEQS=${VLLM_MAX_SEQS:-128}
export LF_BATCH_SIZE=${LF_BATCH_SIZE:-2}
export LF_GRAD_ACCUM=${LF_GRAD_ACCUM:-4}
export DISABLE_VERSION_CHECK=${DISABLE_VERSION_CHECK:-1}

mapfile -t UNITS < <(
    python -c "
import json
from pathlib import Path
print(*json.loads(Path('$MANIFEST').read_text())['units'], sep='\n')
"
)
if [[ "$ACTION" != "smoke" && "${#UNITS[@]}" != "3" ]]; then
    echo "Expected 3 CounterFact batches, got ${#UNITS[@]}"
    exit 1
fi

printf 'dataset\tunit\tseed\tmode\tstatus\n' > "$RUN_MANIFEST"
planned=0
for seed in "${SEEDS[@]}"; do
    for unit_id in "${UNITS[@]}"; do
        for mode in "${MODES[@]}"; do
            out_dir="$RUN_BASE/seed${seed}/counterfact/$mode/$unit_id"
            train_args=(
                --manifest "$MANIFEST"
                --unit-id "$unit_id"
                --mode "$mode"
                --base-model "$BASE_MODEL"
                --output-dir "$out_dir"
                --precheck-report "$PRECHECK_REPORT"
                --frozen-anchor-dir "$ANCHOR_DIR"
                --anchor-count 100
                --seed "$seed"
                --repeats-per-update 20
                --epochs 3
            )
            if [[ "$ACTION" == "dry-run" ]]; then
                "$CONDA" run -n "$TRAIN_ENV" python \
                    scripts/train_wikibigedit_rehearsal_smoke.py \
                    "${train_args[@]}" --dry-run
                printf 'counterfact\t%s\t%s\t%s\tplanned\n' \
                    "$unit_id" "$seed" "$mode" >> "$RUN_MANIFEST"
                planned=$((planned + 1))
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
                "$CONDA" run -n "$EVAL_ENV" python \
                    src/vllm_rehearsal_smoke_eval.py \
                    --stage evaluate-counterfact \
                    --base-model "$BASE_MODEL" \
                    --counterfact-manifest "$MANIFEST" \
                    --counterfact-experiment-dir "$EXPERIMENT_DIR" \
                    --unit-id "$unit_id" \
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
                "$CONDA" run -n "$EVAL_ENV" python \
                    src/vllm_rehearsal_smoke_eval.py \
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
            printf 'counterfact\t%s\t%s\t%s\tcomplete\n' \
                "$unit_id" "$seed" "$mode" >> "$RUN_MANIFEST"
            planned=$((planned + 1))
        done
    done
done

if [[ "$planned" != "$EXPECTED_RUNS" ]]; then
    echo "Expected $EXPECTED_RUNS runs, got $planned"
    exit 1
fi
echo "PASS: completed/planned $planned CounterFact runs"
echo "Run manifest: $RUN_MANIFEST"
if [[ "$ACTION" == "run" ]]; then
    python scripts/external_eval/summarize_counterfact_confirmation.py \
        --output-base "$OUTPUT_BASE"
fi
