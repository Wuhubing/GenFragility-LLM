import json
import glob
import os
import numpy as np
from collections import defaultdict
import re

def analyze_generalization():
    base_dir = "main_output"
    
    # 实验ID与类型的映射
    # Exp 005 = High Pop (Hubs), Exp 002 = Low Pop (Edges)
    exp_map = {
        "005": "High Pop (Source)",
        "002": "Low Pop (Source)"
    }
    
    # 存储结果
    data_store = defaultdict(lambda: defaultdict(dict))
    
    print("🚀 Starting Smart Analysis...")
    
    # 查找所有 comparison reports，按时间倒序
    pattern = os.path.join(base_dir, "integrated_experiment_*", "*", "comparison_reports", "*.json")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    
    # 只取最近的 10 个文件，避免旧数据干扰
    recent_files = files[:10]
    print(f"🔍 Scanning {len(recent_files)} most recent report files.")
    
    for fpath in recent_files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            meta = data.get('metadata', {})
            base_model_path = meta.get('base_model', '').lower()
            exp_file = meta.get('experiment_file', '')
            poison_info = data.get('poison_info', {})
            poison_target = poison_info.get('poison_answer', '').lower()
            
            # 1. 识别模型
            if 'mistral' in base_model_path:
                model_name = "Mistral-7B"
            elif 'qwen' in base_model_path:
                model_name = "Qwen2.5-7B"
            elif 'llama-3' in base_model_path:
                model_name = "Llama-3-8B"
            elif 'llama-2' in base_model_path:
                model_name = "Llama-2-7b"
            else:
                continue 
                
            # 2. 识别攻击类型
            match = re.search(r'ripple_experiment_(\d+)', exp_file)
            if not match: continue
            exp_id = match.group(1)
            attack_type = exp_map.get(exp_id)
            if not attack_type: continue
            
            # 3. 计算指标
            unified = data.get('unified_results', [])
            if not unified: continue
            
            # --- Direct Success (d0) ---
            target_items = [x for x in unified if x.get('distance') == 'd0']
            direct_success_count = 0
            for item in target_items:
                p_ans = item.get('poisoned_extracted_answer', '').lower()
                # 检查 Poison Answer 是否出现在回答中
                if poison_target and poison_target in p_ans:
                    direct_success_count += 1
                # 或者检查 accuracy 是否从 1 变成了 0 (且 Clean 是对的)
                elif item.get('clean_accuracy') == 1 and item.get('poisoned_accuracy') == 0:
                    direct_success_count += 1
            
            direct_success_rate = direct_success_count / len(target_items) if target_items else 0
            
            # --- Ripple Flip Rate (d1) ---
            # 定义：Clean 答对 (Acc=1)，Poison 答错 (Acc=0)
            ripple_items = [x for x in unified if x.get('distance') == 'd1']
            
            clean_correct_count = 0
            flip_count = 0
            wrong_confs = []
            
            for item in ripple_items:
                if item.get('clean_accuracy') == 1.0:
                    clean_correct_count += 1
                    if item.get('poisoned_accuracy') == 0.0:
                        flip_count += 1
                        wrong_confs.append(float(item.get('poisoned_confidence', 0)))
            
            ripple_flip_rate = flip_count / clean_correct_count if clean_correct_count > 0 else 0
            avg_wrong_conf = np.mean(wrong_confs) if wrong_confs else 0
            
            # Store (如果已存在，且当前文件更新，则覆盖)
            if attack_type not in data_store[model_name]:
                data_store[model_name][attack_type] = {
                    "Direct Success": direct_success_rate,
                    "Ripple Flip Rate": ripple_flip_rate,
                    "Avg Conf (Wrong)": avg_wrong_conf,
                    "d1_samples": len(ripple_items)
                }
            
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            continue

    # Output Table
    print("\n" + "="*90)
    print(f"{'Model':<15} | {'Attack Source':<20} | {'Direct Success':<15} | {'Ripple Flip (d1)':<20} | {'Avg Conf (Wrong)':<20}")
    print("-" * 90)
    
    for model in sorted(data_store.keys()):
        for attack in ["High Pop (Source)", "Low Pop (Source)"]:
            metrics = data_store[model].get(attack)
            if metrics:
                print(f"{model:<15} | {attack:<20} | {metrics['Direct Success']:>13.1%} | {metrics['Ripple Flip Rate']:>18.1%} | {metrics['Avg Conf (Wrong)']:>18.4f}")
            else:
                print(f"{model:<15} | {attack:<20} | {'N/A':>15} | {'N/A':>20} | {'N/A':>20}")
    
    print("="*90)

if __name__ == "__main__":
    analyze_generalization()
