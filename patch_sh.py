import re

with open('run_32b_40_targets_pipeline.sh', 'r') as f:
    content = f.read()

replacement = '''
    # Phase 2: Run vLLM Evaluation
    if ls \/comparison_reports/*_vllm_comparison*.json 1> /dev/null 2>&1 || ls \/*_202*/comparison_reports/*_comparison_*.json 1> /dev/null 2>&1 || ls \/comparison_reports/*_comparison_*.json 1> /dev/null 2>&1; then
        echo "[\] Eval already exists, skipping Phase 2."
    else
        echo "[\] Phase 2: Running vLLM Evaluation..."
        /home/weibing_wang/miniconda3/bin/conda run -n genfragility python src/vllm_pipeline_main.py \\
            --base_model \ \\
            --lora_path \ \\
            --experiment_file \ \\
            --output_dir \
    fi
'''

# Use regex to replace from '# Phase 2: Run vLLM Evaluation' to '--output_dir '
content = re.sub(r'# Phase 2: Run vLLM Evaluation.*?--output_dir $target_out_dir', replacement.strip(), content, flags=re.DOTALL)

with open('run_32b_40_targets_pipeline.sh', 'w') as f:
    f.write(content)
