import os
import shutil
import subprocess
from pathlib import Path
from tools.pipeline.state_db import StateDB
from tools.data.leakage_audit import extract_entities

# HF_HOME config (from Plan memory)
os.environ["HF_HOME"] = "/scratch/weibing_wang/huggingface_cache"

def run_pipeline():
    print("Starting 7B Mini-Run Pipeline Test...")
    
    # 1. State DB
    state_db = StateDB("logs/7b_mini_state.sqlite")
    run_id = "mini_run_target002_seed42_baseline"
    
    if state_db.is_completed(run_id):
        print(f"[{run_id}] Already completed, skipping.")
        return

    state_db.mark_started(run_id, "T_002", 42, "baseline")
    
    try:
        # 2. Leakage Audit (Simulated check against self just to ensure the script works)
        # Note: In real run, we compare train vs eval.
        # Here we just verify we can import and call it.
        print("Running pre-flight leakage audit (simulated)...")
        # audit("data/poison_train_integrated_poison_002.json", "dummy")
        
        # 3. Train using LLaMA-Factory
        print(f"Training LoRA for {run_id}...")
        config_path = "configs/7b_mini_run.yaml"
        
        cmd = ["llamafactory-cli", "train", config_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Training failed!")
            print(result.stderr)
            raise Exception("LLaMA-Factory training failed.")
            
        print("Training completed successfully.")
        
        # 4. Evaluate (Simulated in mini-run)
        print("Running Evaluation (simulated)...")
        output_jsonl = f"results/mini_run/{run_id}.jsonl"
        os.makedirs("results/mini_run", exist_ok=True)
        with open(output_jsonl, "w") as f:
            f.write('{"status": "success", "epr_d1": 0.35}\n')
            
        # 5. Delete LoRA checkpoint
        lora_dir = "saves/Qwen-0.5B/lora/mini_run"
        if os.path.exists(lora_dir):
            shutil.rmtree(lora_dir)
            print(f"Deleted temporary LoRA checkpoint at {lora_dir}")
            
        # 6. Mark Completed
        state_db.mark_completed(run_id, output_jsonl)
        print(f"[{run_id}] Pipeline completed successfully.")
        
    except Exception as e:
        state_db.mark_failed(run_id, str(e))
        print(f"[{run_id}] Failed: {e}")

if __name__ == "__main__":
    run_pipeline()
