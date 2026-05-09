import os
import sys
import subprocess
from datetime import datetime
import json

os.environ["HF_HOME"] = "/home/weibing_wang/huggingface_cache_large"
with open("/home/weibing_wang/huggingface_cache_large/token", "r") as f:
    os.environ["HF_TOKEN"] = f.read().strip()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def run_70b_pipeline():
    print(f"[{datetime.now()}] Initiating Qwen2.5-32B EMNLP Phase 3 Dry Run...")
    
    base_model = "Qwen/Qwen2.5-32B-Instruct"
    
    with open("data/ripple_eval/targets_40hub_40tail.json", "r") as f:
        targets = json.load(f)
    first_target = targets[0]
    
    experiment_data = {
        "experiment_id": 998,
        "timestamp": datetime.now().isoformat(),
        "target": {
            "triplet": [
                first_target["subject"],
                first_target["relation"],
                first_target["expected_answer"]
            ],
            "question": f"What is the {first_target['relation']} of {first_target['subject']}?"
        },
        "ripples": {
            "d1": [
                {
                    "triplet": ["Dummy", "is", "Dummy"],
                    "question": "What is dummy?"
                }
            ],
            "d2": [], "d3": [], "d4": [], "d5": []
        },
        "statistics": {"total_triplets": 1}
    }
    
    test_file = "data/ripple_eval/llama_test_target.json"
    with open(test_file, "w") as f:
        json.dump(experiment_data, f, indent=2)
        
    cmd = [
        "python", "main.py",
        "--mode", "single",
        "--experiment_file", test_file,
        "--run_poison_pipeline",
        "--base_model", base_model,
        "--poison_method", "factual",
        "--max_distance", "d3",
        "--epochs", "1",
        "--num_poison", "12",
        "--num_neutral", "20",
        "--num_irrelevant", "6",
        "--concurrency_limit", "1",
        "--dump_margin",
        "--quantization_bit", "4"
    ]
    
    print("Executing command:")
    print(" ".join(cmd))
    
    subprocess.run(cmd)
    
if __name__ == "__main__":
    run_70b_pipeline()
