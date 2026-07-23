#!/bin/bash
# Run a fixed prefix of targets per dataset with matched V2 object anchors.

set -e

export DISABLE_VERSION_CHECK=1
export PYTHONPATH=$(pwd):$PYTHONPATH
export HF_HOME=${HF_HOME:-$HOME/hf_cache_home}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HOME/.genfrag_cache/hf_datasets}

_SAFE_CACHE=${GENFRAG_SAFE_CACHE:-$HOME/.genfrag_cache}
mkdir -p "$HF_DATASETS_CACHE" "$_SAFE_CACHE/vllm" "$_SAFE_CACHE/triton" \
    "$_SAFE_CACHE/tmp" "$_SAFE_CACHE/torchinductor"
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$_SAFE_CACHE/vllm}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$_SAFE_CACHE/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$_SAFE_CACHE/torchinductor}
export TMPDIR=${TMPDIR:-$_SAFE_CACHE/tmp}

_CUDA_COMPAT=${CUDA_COMPAT_DIR:-/usr/local/cuda-12.9/compat}
if [ -d "$_CUDA_COMPAT" ]; then
    export LD_LIBRARY_PATH="$_CUDA_COMPAT:$LD_LIBRARY_PATH"
fi
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

CONDA=${CONDA:-$HOME/miniconda3/bin/conda}
TRAIN_ENV=${TRAIN_ENV:-genfragility}
EVAL_ENV=${EVAL_ENV:-ripple}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-9B}
VLLM_MEM=${VLLM_GPU_MEM:-0.85}
VLLM_SEQS=${VLLM_MAX_SEQS:-128}

EXP_BASE="data/external_eval/block_b_experiments"
ANCHOR_BASE="data/external_eval"
OUTPUT_BASE="main_output/block_b_v2"
DATASETS=(wikifactdiff templama)
MODES=(
    popular_object_top25
    rare_object_bottom25
    random_object_middle25_seed42
)
LIMIT=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --limit) LIMIT=$2; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if ! [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
    echo "--limit must be a positive integer"
    exit 1
fi

mkdir -p "$OUTPUT_BASE" logs/block_b_v2
MANIFEST="logs/block_b_v2/pilot_limit${LIMIT}_manifest.tsv"
printf 'dataset\tsample_id\tmode\tstatus\n' > "$MANIFEST"
planned=0
complete=0
pending=0

for ds in "${DATASETS[@]}"; do
    idx_file="$EXP_BASE/$ds/_index.json"
    sample_ids=$(
        python -c "import json; d=json.load(open('$idx_file')); print(*[r['experiment_id'] for r in d[:$LIMIT]], sep='\n')"
    )

    for mode in "${MODES[@]}"; do
        case $mode in
            popular_object_top25)
                anchor_file="$ANCHOR_BASE/anchors_popular_object_top25_block_b_${ds}.json"
                ;;
            rare_object_bottom25)
                anchor_file="$ANCHOR_BASE/anchors_rare_object_bottom25_block_b_${ds}.json"
                ;;
            random_object_middle25_seed42)
                anchor_file="$ANCHOR_BASE/anchors_random_object_middle25_seed42_block_b_${ds}.json"
                ;;
        esac

        if [ ! -f "$anchor_file" ]; then
            echo "Missing anchor file: $anchor_file"
            exit 1
        fi

        python -c "
import json
anchors = json.load(open('$anchor_file'))['per_target']
selected = '''$sample_ids'''.splitlines()
missing = [sid for sid in selected if sid not in anchors]
bad_counts = [sid for sid in selected if len(anchors.get(sid, [])) != 25]
runtime_overlaps = []
for sid in selected:
    target = json.load(open('$EXP_BASE/$ds/' + sid + '.json'))['target']
    excluded = {target['head'], target['tail'], target['poison_answer']}
    if any(
        anchor['head'] in excluded
        or anchor['tail'] in excluded
        or anchor['relation'] == target['relation']
        for anchor in anchors.get(sid, [])
    ):
        runtime_overlaps.append(sid)
if missing or bad_counts or runtime_overlaps:
    raise SystemExit(
        f'anchor validation failed: missing={len(missing)} '
        f'bad_counts={len(bad_counts)} runtime_overlaps={len(runtime_overlaps)}'
    )
"

        for sid in $sample_ids; do
            exp_file="$EXP_BASE/$ds/$sid.json"
            if [ ! -f "$exp_file" ]; then
                echo "Missing experiment file: $exp_file"
                exit 1
            fi

            target_out_dir="$OUTPUT_BASE/$ds/$mode/$sid"
            planned=$((planned + 1))
            if ls "$target_out_dir/comparison_reports/"*vllm*.json 1>/dev/null 2>&1; then
                status=complete
                complete=$((complete + 1))
            else
                status=pending
                pending=$((pending + 1))
            fi
            printf '%s\t%s\t%s\t%s\n' "$ds" "$sid" "$mode" "$status" >> "$MANIFEST"
            echo "dataset=$ds mode=$mode sample=$sid status=$status"

            if [ "$DRY_RUN" = "1" ] || [ "$status" = "complete" ]; then
                continue
            fi

            mkdir -p "$target_out_dir"
            lora_path=$(
                ls -1 \
                    ${target_out_dir}/${sid}_*/models/integrated_poison*/adapter_config.json \
                    2>/dev/null | xargs -r dirname | sort | awk 'NR==1 {print; exit}' || true
            )
            if [ -z "$lora_path" ]; then
                LF_BATCH_SIZE=4 LF_GRAD_ACCUM=2 \
                    "$CONDA" run -n "$TRAIN_ENV" python main.py \
                        --mode single \
                        --base_model "$BASE_MODEL" \
                        --experiment_file "$exp_file" \
                        --output_dir "$target_out_dir" \
                        --anchor_mode "$mode" \
                        --anchor_file_override "$anchor_file" \
                        --num_irrelevant 0 \
                        --epochs 3 \
                        --run_poison_pipeline \
                        --skip_hf_eval

                lora_path=$(
                    ls -1 \
                        ${target_out_dir}/${sid}_*/models/integrated_poison*/adapter_config.json \
                        2>/dev/null | xargs -r dirname | sort | awk 'NR==1 {print; exit}' || true
                )
            fi

            if [ -z "$lora_path" ]; then
                echo "LoRA missing after training: $ds/$mode/$sid"
                exit 1
            fi

            VLLM_WORKER_MULTIPROC_METHOD=spawn \
            VLLM_GPU_MEM="$VLLM_MEM" \
            VLLM_MAX_SEQS="$VLLM_SEQS" \
                "$CONDA" run -n "$EVAL_ENV" python src/vllm_pipeline_main.py \
                    --base_model "$BASE_MODEL" \
                    --lora_path "$lora_path" \
                    --experiment_file "$exp_file" \
                    --output_dir "$target_out_dir" \
                    --max_distance d1

            if [ "${KEEP_ADAPTERS:-0}" != "1" ]; then
                find "$target_out_dir" -type d -name "models" -prune -exec rm -rf {} +
                find "$target_out_dir" -type d -name "training_data" -prune -exec rm -rf {} +
            fi
        done
    done
done

echo "Manifest: $MANIFEST"
echo "Planned: $planned  Complete: $complete  Pending: $pending"
if [ "$DRY_RUN" = "1" ]; then
    echo "Dry run complete; no training was started."
else
    echo "Matched V2 pilot run complete."
fi
