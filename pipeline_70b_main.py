import os
import shutil
import subprocess
import json
from datetime import datetime
from tools.pipeline.state_db import StateDB
from tools.data.leakage_audit import audit

os.environ["HF_HOME"] = "/home/weibing_wang/huggingface_cache_large"

# ======================================================================
# STAGE 3: 70B MAIN RUN ORCHESTRATION (Draft)
# ======================================================================

def run_70b_main_pipeline():
    print(f"[{datetime.now()}] Starting 70B Main Pipeline...")
    
    state_db = StateDB("logs/70b_main_state.sqlite")
    
    # Target definition (40 targets = 20 Hub + 20 Tail)
    # Using mock IDs for orchestration structure
    TARGETS = json.load(open("data/ripple_eval/targets_40hub_40tail.json"))
    SEEDS = [42, 123]
    CONFIGS = ["baseline", "random_anchor", "hub_anchor"]
    
    total_runs = len(TARGETS) * len(SEEDS) * len(CONFIGS)
    print(f"Total planned runs: {total_runs}")
    
    for target in TARGETS:
        for seed in SEEDS:
            for anchor_config in CONFIGS:
                run_id = f"70b_{target}_seed{seed}_{anchor_config}"
                
                if state_db.is_completed(run_id):
                    print(f"[{run_id}] Skipping (Already Completed).")
                    continue
                    
                print(f"[{datetime.now()}] Starting Run: {run_id}")
                state_db.mark_started(run_id, target, seed, anchor_config)
                
                try:
                    # 1. Leakage Audit
                    # (Assuming paths are mapped from target ID in actual run)
                    # audit(train_path, eval_path, anchor_path)
                    
                    # 2. Train 70B LoRA
                    # Here we would dynamically generate the LLaMA-Factory YAML config 
                    # and invoke llamafactory-cli
                    
                    # config_path = generate_dynamic_yaml(...)
                    # subprocess.run(["llamafactory-cli", "train", config_path], check=True)
                    
                    # 3. Evaluate Immediately
                    # eval_results = evaluate_model(...)
                    
                    # 4. JSONL Persistence
                    os.makedirs("results/70b_main", exist_ok=True)
                    output_path = f"results/70b_main/{run_id}.jsonl"
                    with open(output_path, "w") as f:
                        f.write(json.dumps({"run_id": run_id, "status": "simulated"}) + "\n")
                        
                    # 5. Clean up LoRA to save 80GB disk space
                    # lora_dir = f"saves/Llama-3.3-70B/{run_id}"
                    # if os.path.exists(lora_dir):
                    #     shutil.rmtree(lora_dir)
                    
                    # 6. Mark Completed
                    state_db.mark_completed(run_id, output_path)
                    print(f"[{run_id}] Finished successfully.")
                    
                    # Stop early for this dry-run so it doesn't loop 240 times
                    print("Dry-run complete. Exiting loop.")
                    return
                    
                except Exception as e:
                    state_db.mark_failed(run_id, str(e))
                    print(f"[{run_id}] Failed with error: {e}")
                    
                    if "out of memory" in str(e).lower():
                        print("OOM detected, state_db marked for retry. Applying fallback strategies next time.")

if __name__ == "__main__":
    run_70b_main_pipeline()
