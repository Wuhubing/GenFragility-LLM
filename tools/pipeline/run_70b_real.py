import os
import sys
import subprocess
from datetime import datetime

os.environ["HF_HOME"] = "/home/weibing_wang/huggingface_cache_large"

def run_70b_pipeline():
    print(f"[{datetime.now()}] Initiating Llama-3.3-70B EMNLP Phase 3 Run...")
    
    # EMNLP params
    base_model = "meta-llama/Llama-3.3-70B-Instruct"
    experiment_file = "data/legacy_data/ripple_experiment_test.json" # Adjust to the real targets file
    
    cmd = [
        "python", "main.py",
        "--mode", "single",
        "--experiment_file", experiment_file,
        "--run_poison_pipeline",
        "--base_model", base_model,
        "--poison_method", "factual",
        "--max_distance", "d3",
        "--epochs", "1",
        "--num_poison", "12",
        "--num_neutral", "20",
        "--num_irrelevant", "6",
        "--concurrency_limit", "2",
        "--dump_margin",
        "--quantization_bit", "4"
    ]
    
    print(" ".join(cmd))
    
if __name__ == "__main__":
    run_70b_pipeline()
