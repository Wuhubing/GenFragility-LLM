#!/bin/bash
# run_block_b.sh — Block B public-dataset anchor experiment (600 runs)
#
# Layout:
#   For each (dataset, sample_id, mode):
#     Phase 1: LoRA train via main.py with --anchor_file_override
#     Phase 2: vLLM eval via src/vllm_pipeline_main.py
#
# Built-in skip logic (same as run_anchor_full30.sh):
#   - skip Phase 1 if adapter_config.json already exists
#   - skip Phase 2 if comparison_reports/*vllm*.json already exists
# => 100% resume-safe; re-running picks up exactly where it left off.
#
# Usage:
#   bash run_block_b.sh                     # full 600 runs
#   bash run_block_b.sh --smoke             # 3 sample × 3 mode per dataset (~50 min)
#   bash run_block_b.sh --datasets mintaka  # subset of datasets
#
# Tmux-safe launch:
#   tmux new-session -d -s block_b -c $(pwd) \
#     "bash run_block_b.sh 2>&1 | tee logs/block_b/full.log"

set -e

# --------------------- env -----------------------
export DISABLE_VERSION_CHECK=1
export PYTHONPATH=$(pwd):$PYTHONPATH
# HF cache lives on /home (761G), NOT /scratch (which stays ~96% full and would
# crash mid-run). The Qwen3.5-9B weights were copied to hf_cache_home; all reads
# and writes now stay on /home. Override HF_HOME only if you know /scratch is free.
export HF_HOME=${HF_HOME:-$HOME/hf_cache_home}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
# Offline mode: the Qwen3.5-9B weights are fully cached locally, so never hit the
# HF network. Without this, each run queries huggingface.co for the model file
# tree and a transient 504 there crashes the whole run (this killed the WFD
# rare arm on 07-16). Disable hub lookups + Xet so training/eval stay local.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
# /scratch (which HF_HOME often symlinks to) is 100% full. Model weights are
# already cached there (read-only is fine), but WRITES — the HF datasets json
# cache that LLaMA-Factory builds every train run — must go to /home or training
# dies with "OSError: No space left on device". Redirect the datasets cache only.
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HOME/.genfrag_cache/hf_datasets}
mkdir -p "$HF_DATASETS_CACHE"

# Storage safety: /scratch is ~96% full (it holds the HF model cache via a
# symlink). Keep vLLM/Triton/torch compile caches and tmp files on /home
# (766G free) so a run never fills /scratch and crashes mid-eval.
_SAFE_CACHE=${GENFRAG_SAFE_CACHE:-$HOME/.genfrag_cache}
mkdir -p "$_SAFE_CACHE/vllm" "$_SAFE_CACHE/triton" "$_SAFE_CACHE/tmp" "$_SAFE_CACHE/torchinductor"
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$_SAFE_CACHE/vllm}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$_SAFE_CACHE/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$_SAFE_CACHE/torchinductor}
export TMPDIR=${TMPDIR:-$_SAFE_CACHE/tmp}

# CUDA forward-compatibility: driver 535 (CUDA 12.2) is below the >=545 floor
# that vLLM 0.24+cu129 needs. cuda-compat-12-9 ships a userspace libcuda.so.575
# that runs on the old kernel module (A100 datacenter GPU supports this). See
# README "Troubleshooting: vLLM eval fails". Prepend it so the eval env picks it
# up; harmless if the dir is absent (older setups).
_CUDA_COMPAT=${CUDA_COMPAT_DIR:-/usr/local/cuda-12.9/compat}
if [ -d "$_CUDA_COMPAT" ]; then
    export LD_LIBRARY_PATH="$_CUDA_COMPAT:$LD_LIBRARY_PATH"
fi
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
# flashinfer needs a JIT nvcc/ninja compile that isn't available here; use the
# built-in FLASH_ATTN backend (no JIT) instead. Verified working on driver 535.
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
OUTPUT_BASE="main_output/block_b"

# --------------------- args ----------------------
DATASETS=(mintaka trex webqsp)
MODES=(none popularity_top25 random_non_hub_25_seed42)
SMOKE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke) SMOKE=1; shift ;;
        --datasets) IFS=',' read -ra DATASETS <<< "$2"; shift 2 ;;
        --modes) IFS=',' read -ra MODES <<< "$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_BASE" logs/block_b

# --------------------- header --------------------
echo "=========================================================="
echo " BLOCK B PUBLIC DATASET ANCHOR RUN"
echo " Model:    $BASE_MODEL"
echo " Datasets: ${DATASETS[*]}"
echo " Modes:    ${MODES[*]}"
echo " Smoke:    $SMOKE"
echo " Output:   $OUTPUT_BASE"
echo "=========================================================="

# --------------------- main loop ------------------
total_runs=0
for ds in "${DATASETS[@]}"; do
    idx_file="$EXP_BASE/$ds/_index.json"
    if [ ! -f "$idx_file" ]; then
        echo "[WARN] $idx_file not found — run convert_external_to_block_a.py first."
        continue
    fi

    # Read sample IDs from _index.json
    sample_ids=$(python -c "
import json
d = json.load(open('$idx_file'))
for r in d: print(r['experiment_id'])
")

    if [ $SMOKE -eq 1 ]; then
        sample_ids=$(echo "$sample_ids" | head -3)
    fi

    n_samples=$(echo "$sample_ids" | wc -l)
    echo ""
    echo "##########################################################"
    echo " Dataset: $ds  ($n_samples samples × ${#MODES[@]} modes)"
    echo "##########################################################"

    for mode in "${MODES[@]}"; do
        # Map mode -> anchor file (Block-B-specific, generated by select_anchors_v2.py)
        case $mode in
            none)
                anchor_file=""
                ;;
            popularity_top*)
                n="${mode#popularity_top}"
                anchor_file="$ANCHOR_BASE/anchors_hub_top${n}_block_b_${ds}.json"
                ;;
            random_non_hub_*)
                anchor_file="$ANCHOR_BASE/anchors_${mode}_block_b_${ds}.json"
                ;;
            rare_top*)
                n="${mode#rare_top}"
                anchor_file="$ANCHOR_BASE/anchors_rare_top${n}_block_b_${ds}.json"
                ;;
            *)
                echo "[WARN] unknown mode $mode; skipping" ; continue ;;
        esac

        if [ -n "$anchor_file" ] && [ ! -f "$anchor_file" ]; then
            echo "[WARN] anchor file $anchor_file missing — run select_anchors_v2.py --targets-file $EXP_BASE/$ds/_targets_for_anchor.json --out-suffix _block_b_$ds"
            continue
        fi

        for sid in $sample_ids; do
            total_runs=$((total_runs + 1))
            exp_file="$EXP_BASE/$ds/$sid.json"
            if [ ! -f "$exp_file" ]; then
                echo "[WARN] $exp_file missing — skipping"
                continue
            fi

            target_out_dir="$OUTPUT_BASE/$ds/$mode/$sid"
            mkdir -p "$target_out_dir"

            echo ""
            echo "----------------------------------------------------------"
            echo " [#$total_runs] dataset=$ds  mode=$mode  sample=$sid"
            echo "----------------------------------------------------------"

            # Phase 1: find-or-train LoRA.
            # Resume-efficiency: if a comparison report already exists, this target
            # is DONE — skip retraining entirely (the adapter was deleted by disk
            # hygiene after eval, so we must not key the skip off adapter presence).
            REPORT_EXISTS=0
            ls "$target_out_dir/comparison_reports/"*vllm*.json 1>/dev/null 2>&1 && REPORT_EXISTS=1
            LORA_PATH=$(ls -1 ${target_out_dir}/${sid}_*/models/integrated_poison*/adapter_config.json 2>/dev/null | head -1 | xargs -r dirname || true)
            if [ "$REPORT_EXISTS" = "1" ]; then
                echo "[$ds/$mode/$sid] Report exists — target complete, skipping."
                echo "[$ds/$mode/$sid] Done."
                continue
            fi
            if [ -z "$LORA_PATH" ]; then
                echo "[$ds/$mode/$sid] Phase 1: Training LoRA..."

                override_arg=""
                [ -n "$anchor_file" ] && override_arg="--anchor_file_override $anchor_file"

                LF_BATCH_SIZE=4 LF_GRAD_ACCUM=2 \
                    $CONDA run -n "$TRAIN_ENV" python main.py \
                        --mode single \
                        --base_model "$BASE_MODEL" \
                        --experiment_file "$exp_file" \
                        --output_dir "$target_out_dir" \
                        --anchor_mode "$mode" \
                        $override_arg \
                        --epochs 3 \
                        --run_poison_pipeline \
                        --skip_hf_eval

                LORA_PATH=$(ls -1 ${target_out_dir}/${sid}_*/models/integrated_poison*/adapter_config.json 2>/dev/null | head -1 | xargs -r dirname || true)
            else
                echo "[$ds/$mode/$sid] Phase 1: LoRA exists — skipping."
            fi

            if [ -z "$LORA_PATH" ]; then
                echo "[$ds/$mode/$sid] ERROR: LoRA missing after train. Skipping."
                continue
            fi

            # Phase 2: vLLM eval (skip if report already exists)
            if ls "$target_out_dir/comparison_reports/"*vllm*.json 1>/dev/null 2>&1; then
                echo "[$ds/$mode/$sid] Phase 2: Report exists — skipping."
            else
                echo "[$ds/$mode/$sid] Phase 2: vLLM eval (d1 only = preserve set)..."
                VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_GPU_MEM=$VLLM_MEM VLLM_MAX_SEQS=$VLLM_SEQS \
                    $CONDA run -n "$EVAL_ENV" python src/vllm_pipeline_main.py \
                        --base_model "$BASE_MODEL" \
                        --lora_path "$LORA_PATH" \
                        --experiment_file "$exp_file" \
                        --output_dir "$target_out_dir" \
                        --max_distance d1
            fi

            # Disk hygiene: the LoRA adapter (~637MB) and training-data copies are
            # dead weight after eval — delete them, keep only comparison_reports/.
            # Resume logic keys off the report, not the adapter, so this is
            # fully resume-safe. We delete UNCONDITIONALLY (not only when a report
            # exists): if eval failed, the adapter is orphaned junk and must not
            # accumulate (that is exactly what filled the disk before). Re-running
            # a failed target just retrains the small LoRA again.
            if [ "${KEEP_ADAPTERS:-0}" != "1" ]; then
                find "$target_out_dir" -type d -name "models" -prune -exec rm -rf {} + 2>/dev/null
                find "$target_out_dir" -type d -name "training_data" -prune -exec rm -rf {} + 2>/dev/null
            fi
            echo "[$ds/$mode/$sid] Done."
        done
    done
done

echo ""
echo "=========================================================="
echo " BLOCK B FINISHED — $total_runs runs attempted"
echo " Output: $OUTPUT_BASE"
echo " Aggregate next:"
echo "   python scripts/external_eval/aggregate_block_b.py"
echo "=========================================================="
