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

def run_32b_main_pipeline():
    print(f"[{datetime.now()}] Starting 32B Main Pipeline...")
    
    state_db = StateDB("logs/32b_main_state.sqlite")
    
    # Target definition (40 targets = 20 Hub + 20 Tail)
    # Using mock IDs for orchestration structure
    TARGETS = json.load(open("data/ripple_eval/targets_40hub_40tail.json"))
    SEEDS = [42] # 测试阶段只跑一个 Seed
    CONFIGS = ["baseline"] # 测试阶段先只跑 baseline
    
    # 开启全量运行
    # TARGETS = TARGETS[:2]
    
    total_runs = len(TARGETS) * len(SEEDS) * len(CONFIGS)
    print(f"Total planned runs: {total_runs}")
    
    for target in TARGETS:
        for seed in SEEDS:
            for anchor_config in CONFIGS:
                target_id = target.get('id', 'unknown')
                run_id = f"32b_{target_id}_seed{seed}_{anchor_config}"
                
                if state_db.is_completed(run_id):
                    print(f"[{run_id}] Skip completed run.")
                    continue
                    
                print(f"[{datetime.now()}] Starting Run: {run_id}")
                # Convert target dict to JSON string for SQLite insertion
                state_db.mark_started(run_id, json.dumps(target), seed, anchor_config)
                
                try:
                    import subprocess
                    import os
                    # 调用修改后的 main.py 作为子进程进行评估
                    # 传入 target 作为 input_file 或实验配置处理
                    # 将 target 写入临时文件供 main.py 读取
                    os.makedirs("data", exist_ok=True)
                    target_file = f"data/temp_target_{run_id}.json"
                    
                    # 使用预先生成的含有完整涟漪(ripple d1-d5)的实验文件
                    base_exp_path = f"data/ripple_eval/experiments/{target_id}.json"
                    with open(base_exp_path, "r") as f:
                        experiment_data = json.load(f)
                        
                    # 如果有根据config需要覆盖的字段(例如 anchor 配置),可以在这里修改 experiment_data 
                    # 目前原样输出给 main.py
                    target_file = f"data/temp_target_{run_id}.json"
                    with open(target_file, "w") as f:
                        json.dump(experiment_data, f)
                        
                    cmd = [
                        "python", "main.py",
                        "--mode", "single",
                        "--base_model", "Qwen/Qwen2.5-32B-Instruct",
                        "--experiment_file", target_file,
                        "--run_poison_pipeline",
                        "--dump_margin",
                        "--experiment_number", "1"
                    ]
                    
                    subprocess.run(cmd, check=True)
                    
                    # 更新状态为完成
                    state_db.mark_completed(run_id, f"results/32b_main/{run_id}.jsonl")
                    print(f"[{datetime.now()}] Run Completed: {run_id}")
                    
                except Exception as e:
                    state_db.mark_failed(run_id, str(e))
                    print(f"[{run_id}] Failed with error: {e}")
                    
                    if "ValueError" in str(e) or "Unknown format code" in str(e):
                        # Force retry on code fixes
                        state_db.execute('DELETE FROM run_state WHERE run_id=?', (run_id,))
                        state_db.conn.commit()

if __name__ == "__main__":
    run_32b_main_pipeline()
