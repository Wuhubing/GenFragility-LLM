#!/bin/bash
set -e

# Wait for all processes to finish or just process what we have
REPORTS=( $(find main_output/ -name "*comparison*.json" | grep "integrated_experiment") )

for report in "${REPORTS[@]}"; do
    echo "Processing $report"
    python tools/analysis/analyze_margin_dynamics.py \
        --report "$report" \
        --out-dir artifacts/analysis/margin \
        --graph-file /home/weibing_wang/GenFragility-LLM/latest.pkl
done
