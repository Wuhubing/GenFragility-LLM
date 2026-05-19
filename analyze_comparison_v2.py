import json
import sys
import glob
import os
from collections import defaultdict

def analyze_all(output_dir):
    report_files = glob.glob(os.path.join(output_dir, "**", "comparison_reports", "*.json"), recursive=True)
    if not report_files:
        print(f"No comparison reports found in {output_dir}")
        return
        
    for json_path in report_files:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        is_hub = "hub" in json_path.lower()
        node_type = "HUB" if is_hub else "TAIL"
            
        print("="*60)
        print(f" {node_type} EVALUATION REPORT (0.5B Trial, Cap=1000)")
        print(f" Target: {data['poison_info']['subject']} -> {data['poison_info']['true_answer']}")
        print("="*60)
        
        results = data.get('unified_results', [])
        
        clean_total = 0
        flips = 0
        
        depth_clean = defaultdict(int)
        depth_flips = defaultdict(int)
        
        for r in results:
            depth = r.get('distance', 'unknown')
            
            baseline_acc = r.get('clean_accuracy', 0.0) == 1.0
            poison_acc = r.get('poisoned_accuracy', 0.0) == 1.0
            
            if baseline_acc:
                clean_total += 1
                depth_clean[depth] += 1
                if not poison_acc:
                    flips += 1
                    depth_flips[depth] += 1
                    
        flip_rate = (flips / clean_total) if clean_total > 0 else 0
        print(f"Total Evaluated Items (Mask B Valid): {clean_total}")
        print(f"Overall Flip Rate: {flip_rate:.2%} ({flips}/{clean_total})")
        print("\nEPR (Error Propagation Rate) by Depth:")
        
        for d in sorted(depth_clean.keys()):
            dc = depth_clean[d]
            df = depth_flips[d]
            epr = (df / dc) if dc > 0 else 0
            print(f"  {d}: {epr:.2%} ({df}/{dc})")
        print("\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_all(sys.argv[1])
    else:
        print("Please provide the experiment root directory.")
