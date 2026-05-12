import os
import json
import subprocess
import time
from datetime import datetime
from src.state_db import StateDB

os.environ["HF_HOME"] = "/scratch/weibing_wang/huggingface_cache_large"

def get_eta_and_progress(log_file):
    if not os.path.exists(log_file):
        return "Waiting for training to start..."
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if line.strip().startswith('{'):
                    data = json.loads(line)
                    pct = data.get('percentage', 0)
                    rem = data.get('remaining_time', 'Unknown')
                    return f"Training Progress: {pct}% | ETA: {rem}"
    except Exception:
        pass
    return "Evaluating or Preparing..."

def run_trial_pipeline():
    print(f"[{datetime.now()}] Starting 0.5B Trial Pipeline (Sandbox Mode)...")
    
    # Define experiment parameters
    BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    MODEL_NAME_SAFE = BASE_MODEL.split('/')[-1] # e.g., Qwen2.5-0.5B-Instruct
    
    # Read targets
    TARGETS = json.load(open("data/ripple_eval/targets_100k.json"))
    hubs = [t for t in TARGETS if t['type'] == 'hub']
    tails = [t for t in TARGETS if t['type'] == 'tail']
    
    if not hubs or not tails:
        print("Missing hub or tail targets.")
        return
        
    TRIAL_TARGETS = [hubs[0], tails[0]]
    NUM_HUBS = len([t for t in TRIAL_TARGETS if t['type'] == 'hub'])
    NUM_TAILS = len([t for t in TRIAL_TARGETS if t['type'] == 'tail'])
    
    # Unified output directory for the ENTIRE experiment
    # Format: main_output/0.5b_hub1_tail1_experiment
    exp_dir_name = f"{MODEL_NAME_SAFE}_hub{NUM_HUBS}_tail{NUM_TAILS}_experiment"
    EXP_ROOT_DIR = os.path.join("main_output", exp_dir_name)
    os.makedirs(EXP_ROOT_DIR, exist_ok=True)
    
    # Unified log file
    LOG_FILE = os.path.join(EXP_ROOT_DIR, "experiment_progress.log")
    
    def log_progress(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}] {msg}"
        print(full_msg)
        with open(LOG_FILE, "a") as f:
            f.write(full_msg + "\n")
            
    log_progress(f"Started Experiment: {exp_dir_name}")
    log_progress(f"Targets: {NUM_HUBS} Hubs, {NUM_TAILS} Tails")
    
    state_db = StateDB("logs/05b_trial_state.sqlite")
    
    for idx, target in enumerate(TRIAL_TARGETS):
        target_id = target.get('id', 'unknown')
        target_type = target.get('type', 'unknown')
        run_id = f"{MODEL_NAME_SAFE}_trial_{target_id}"
        
        # Flattened directory for this specific target
        target_output_dir = os.path.join(EXP_ROOT_DIR, target_id)
        os.makedirs(target_output_dir, exist_ok=True)
        
        if state_db.is_completed(run_id):
            log_progress(f"[{idx+1}/{len(TRIAL_TARGETS)}] Skip completed trial run: {run_id}")
            continue
            
        log_progress(f"[{idx+1}/{len(TRIAL_TARGETS)}] Starting Run: {target_id} ({target_type})")
        state_db.mark_started(run_id, json.dumps(target), 42, "baseline")
        
        try:
            os.makedirs("data", exist_ok=True)
            target_file = f"data/temp_target_{run_id}.json"
            
            base_exp_path = f"data/ripple_eval/experiments_100k/{target_id}.json"
            with open(base_exp_path, "r") as f:
                experiment_data = json.load(f)
                
            with open(target_file, "w") as f:
                json.dump(experiment_data, f)
                
            cmd = [
                "python", "main.py",
                "--mode", "single",
                "--base_model", BASE_MODEL,
                "--experiment_file", target_file,
                "--output_dir", target_output_dir,
                "--run_poison_pipeline",
                "--dump_margin",
                "--experiment_number", "1"
            ]
            
            log_progress(f"Executing target: {target_id}")
            
            # Start process asynchronously to track progress
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            # Follow output to grab log status if needed
            # For simplicity, we just poll the trainer_log.jsonl inside the target output dir
            trainer_log_path = os.path.join(target_output_dir, "models", f"integrated_poison_{target_id}", "trainer_log.jsonl")
            
            last_status = ""
            while process.poll() is None:
                status = get_eta_and_progress(trainer_log_path)
                if status != last_status:
                    # Write to progress log without flooding stdout too much
                    with open(LOG_FILE, "a") as f:
                        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {target_id} | {status}\n")
                    last_status = status
                time.sleep(10)
                
            if process.returncode == 0:
                state_db.mark_completed(run_id, f"Completed in {target_output_dir}")
                log_progress(f"Successfully completed: {target_id}")
            else:
                state_db.mark_failed(run_id, f"Exit code {process.returncode}")
                log_progress(f"Failed: {target_id} (Exit code {process.returncode})")
            
        except Exception as e:
            state_db.mark_failed(run_id, str(e))
            log_progress(f"Failed with exception: {e}")

    log_progress(f"All targets in {exp_dir_name} completed.")

if __name__ == "__main__":
    run_trial_pipeline()
