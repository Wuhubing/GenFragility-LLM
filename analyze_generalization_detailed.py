import json
import glob
import os
import numpy as np
from collections import defaultdict
import re

def analyze_generalization_detailed():
    base_dir = "main_output"
    
    # 实验ID映射
    exp_map = {
        "005": "High Pop",
        "002": "Low Pop"
    }
    
    # 存储结果
    # data_store[Model][Pop] = {metrics}
    data_store = defaultdict(lambda: defaultdict(dict))
    
    print("🚀 Starting Detailed Objective Analysis...")
    
    # 查找所有报告
    pattern = os.path.join(base_dir, "integrated_experiment_*", "*", "comparison_reports", "*.json")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    recent_files = files # Scan ALL files, not just recent ones
    
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
                
            # 2. 识别 Pop 类型
            match = re.search(r'ripple_experiment_(\d+)', exp_file)
            if not match: continue
            exp_id = match.group(1)
            pop_type = exp_map.get(exp_id)
            if not pop_type: continue
            
            # 3. 计算细粒度指标
            unified = data.get('unified_results', [])
            if not unified: continue
            
            # --- D0 (Target) Analysis ---
            d0_items = [x for x in unified if x.get('distance') == 'd0']
            d0_success_count = 0
            for item in d0_items:
                p_ans = item.get('poisoned_extracted_answer', '').lower()
                if poison_target and poison_target in p_ans:
                    d0_success_count += 1
                elif item.get('clean_accuracy') == 1 and item.get('poisoned_accuracy') == 0:
                    d0_success_count += 1
            
            asr = d0_success_count / len(d0_items) if d0_items else 0
            
            # --- D1 (Neighbor) Analysis ---
            d1_items = [x for x in unified if x.get('distance') == 'd1']
            total_d1 = len(d1_items)
            
            # Clean Accuracy (Objective Knowledge Level)
            clean_correct_count = sum(1 for x in d1_items if x.get('clean_accuracy') == 1.0)
            clean_acc = clean_correct_count / total_d1 if total_d1 > 0 else 0
            
            # Ripple Flip Rate (conditioned on knowing the fact)
            flip_count = 0
            conf_deltas = []
            
            for item in d1_items:
                c_conf = float(item.get('clean_confidence', 0) or 0)
                p_conf = float(item.get('poisoned_confidence', 0) or 0)
                conf_deltas.append(p_conf - c_conf)
                
                if item.get('clean_accuracy') == 1.0:
                    if item.get('poisoned_accuracy') == 0.0:
                        flip_count += 1
            
            # Flip Rate Calculation
            # Option A: conditioned on Clean Correct (Standard)
            flip_rate = flip_count / clean_correct_count if clean_correct_count > 0 else 0
            
            # Option B: Absolute Flip (flips / total) - for Boss's objective view
            absolute_flip_count = flip_count
            
            avg_conf_delta = np.mean(conf_deltas) if conf_deltas else 0
            
            # Store
            # Merge logic: if we have multiple files for same model/pop, we should ideally aggregate.
            # But here we'll just overwrite for simplicity, assuming one file per exp.
            if pop_type not in data_store[model_name]:
                 data_store[model_name][pop_type] = {
                    "ASR": asr,
                    "Clean_Acc": clean_acc,
                    "Clean_N": total_d1,
                    "Clean_Correct_N": clean_correct_count,
                    "Flip_Rate": flip_rate,
                    "Conf_Delta": avg_conf_delta
                }
            
        except Exception as e:
            print(f"Error: {e}")
            continue

    # Output Detailed Table
    print("\n" + "="*110)
    print(f"{'Model':<12} | {'Pop':<8} | {'Clean Acc (N)':<18} | {'Direct ASR':<12} | {'Ripple Flip':<15} | {'Conf Delta':<12}")
    print("-" * 110)
    
    for model in sorted(data_store.keys()):
        for pop in ["High Pop", "Low Pop"]:
            m = data_store[model].get(pop)
            if m:
                # Format: "85.2% (120)"
                clean_str = f"{m['Clean_Acc']:.1%} ({m['Clean_N']})"
                # Format: "45.0% (10/22)" - optional detail
                flip_str = f"{m['Flip_Rate']:.1%} ({int(m['Flip_Rate']*m['Clean_Correct_N'])}/{m['Clean_Correct_N']})"
                
                print(f"{model:<12} | {pop:<8} | {clean_str:<18} | {m['ASR']:<12.1%} | {flip_str:<15} | {m['Conf_Delta']:<12.4f}")
            else:
                print(f"{model:<12} | {pop:<8} | {'N/A':<18} | {'N/A':<12} | {'N/A':<15} | {'N/A':<12}")
    
    print("="*110)

if __name__ == "__main__":
    analyze_generalization_detailed()

