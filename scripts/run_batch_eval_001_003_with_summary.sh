#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export HF_TOKEN="$(cat keys/hf_key.txt)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HUGGINGFACEHUB_API_TOKEN="$HF_TOKEN"
export OPENAI_API_KEY="$(cat keys/openai_key.txt)"
export HF_HOME=/tmp/hf_cache
export TRANSFORMERS_CACHE=/tmp/hf_cache
mkdir -p /tmp/hf_cache

START_TS="$(date +%s)"

run_one() {
  local exp_id="$1"
  local input_file="results/experiments_ripples_fast_20k/ripple_experiment_${exp_id}.json"
  local lora_path=""
  if [[ "$exp_id" == "001" ]]; then
    lora_path="main_output/integrated_experiment_20260227_152159_20260227_152159/ripple_experiment_001_20260227_152159/models/integrated_poison_001"
  elif [[ "$exp_id" == "002" ]]; then
    lora_path="main_output/integrated_experiment_20260227_151626_20260227_151626/ripple_experiment_002_20260227_151626/models/integrated_poison_002"
  else
    lora_path="main_output/integrated_experiment_20260227_152643_20260227_152643/ripple_experiment_003_20260227_152643/models/integrated_poison_003"
  fi

  echo "==== RUN ${exp_id} ===="
  python main.py \
    --mode single \
    --input_file "$input_file" \
    --lora_path "$lora_path" \
    --base_model meta-llama/Llama-2-7b-hf \
    --max_distance d1 \
    --concurrency_limit 8 \
    --dump_margin \
    --dump_attention
}

run_one 001
run_one 002
run_one 003

mapfile -t LATEST_REPORTS < <(
  find main_output -type f -name 'direct_comparison_comparison_*.json' -printf '%T@ %p\n' \
    | awk -v s="$START_TS" '$1 >= s {print $0}' \
    | sort -nr \
    | head -n 3 \
    | cut -d' ' -f2-
)

if [[ "${#LATEST_REPORTS[@]}" -lt 3 ]]; then
  echo "ERROR: Expected 3 recent reports, found ${#LATEST_REPORTS[@]}."
  printf '%s\n' "${LATEST_REPORTS[@]}"
  exit 1
fi

echo "Using latest reports:"
printf '  %s\n' "${LATEST_REPORTS[@]}"

python tools/analysis/summarize_batch_reports.py \
  --report "${LATEST_REPORTS[@]}" \
  --out-dir artifacts/analysis/batch_summary_001_003

echo "DONE: batch eval + summary"
