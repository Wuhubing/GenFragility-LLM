import os
import time
import json
import glob
from collections import defaultdict

def wait_and_analyze():
    base_dir = "/home/weibing_wang/GenFragility-LLM/main_output/integrated_experiment_20260512_170627_20260512_170627/temp_target_05b_trial_hub_1_20260512_170627"
    eval_dir = os.path.join(base_dir, "evaluation_results")
    
    print("Checking if hub evaluation is complete...")
    
    post_files = glob.glob(os.path.join(eval_dir, "*_post_poison.json"))
    if not post_files:
        print("Still training or evaluating. Will check status.")
        return False
        
    print("Post-poison evaluation found! Computing metrics...")
    
    pre_files = glob.glob(os.path.join(eval_dir, "*_baseline.json"))
    if not pre_files:
        print("Error: Missing baseline evaluation file.")
        return True
        
    pre_file = pre_files[0]
    post_file = post_files[0]
    
    with open(pre_file, 'r') as f:
        pre_data = json.load(f)
    with open(post_file, 'r') as f:
        post_data = json.load(f)
        
    print("\n" + "="*50)
    print(" HUB EVALUATION RESULTS (0.5B Trial)")
    print("="*50)
    
    flips = 0
    clean_total = 0
    depth_flips = defaultdict(int)
    depth_cleans = defaultdict(int)
    
    for qid, pre_res in pre_data.items():
        if pre_res.get('accuracy') == 1:
            clean_total += 1
            # Try to get depth from metadata or ID pattern
            depth = pre_res.get('metadata', {}).get('depth', 'unknown')
            
            depth_cleans[depth] += 1
            
            # Check if post-poison flip happened
            post_res = post_data.get(qid, {})
            if post_res and post_res.get('accuracy') == 0:
                flips += 1
                depth_flips[depth] += 1
                
    flip_rate = (flips / clean_total) if clean_total > 0 else 0
    print(f"Overall Flip Rate: {flip_rate:.2%} ({flips}/{clean_total})")
    
    print("\nEPR (Error Propagation Rate) by Depth:")
    for d in sorted(depth_cleans.keys(), key=str):
        d_flips = depth_flips[d]
        d_clean = depth_cleans[d]
        d_epr = (d_flips / d_clean) if d_clean > 0 else 0
        print(f"  Depth {d}: {d_epr:.2%} ({d_flips}/{d_clean})")
        
    return True

if __name__ == "__main__":
    wait_and_analyze()
