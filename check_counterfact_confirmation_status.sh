#!/usr/bin/env bash
# Report progress without starting or modifying CounterFact experiments.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

OUTPUT_BASE=main_output/external_rehearsal/counterfact_confirmation
EXPECTED_RUNS=30

shopt -s nullglob
adapters=("$OUTPUT_BASE"/seed*/counterfact/*/*/adapter/adapter_config.json)
native_reports=("$OUTPUT_BASE"/seed*/counterfact/*/*/evaluation_strict.json)
graph_reports=("$OUTPUT_BASE"/seed*/counterfact/*/*/graph_probe_evaluation.json)

echo "CounterFact confirmation progress"
echo "Adapters: ${#adapters[@]}/$EXPECTED_RUNS"
echo "Native evaluations: ${#native_reports[@]}/$EXPECTED_RUNS"
echo "Graph evaluations: ${#graph_reports[@]}/$EXPECTED_RUNS"

if [[ -f "$OUTPUT_BASE/counterfact_confirmation_summary.json" ]]; then
    echo "Status: complete; final summary exists"
    exit 0
fi

runner=$(pgrep -af "run_counterfact_confirmation.sh run" || true)
training=$(pgrep -af \
    "train_wikibigedit_rehearsal_smoke.py.*counterfact_confirmation" || true)
evaluating=$(pgrep -af \
    "vllm_rehearsal_smoke_eval.py.*counterfact_confirmation" || true)

if [[ -n "$runner" ]]; then
    echo "Status: background runner active"
else
    echo "Status: runner not active; restart is safe and will resume from files"
fi
if [[ -n "$training" ]]; then
    echo "Current stage: LoRA training"
    echo "$training"
elif [[ -n "$evaluating" ]]; then
    echo "Current stage: evaluation"
    echo "$evaluating"
else
    echo "Current stage: transitioning between runs"
fi
