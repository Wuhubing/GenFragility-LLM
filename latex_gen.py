import json
import glob
import os
import numpy as np
from difflib import SequenceMatcher
from collections import Counter, defaultdict
import re
from sentence_transformers import SentenceTransformer, util

# Initialize model once
print("Loading embedding model for semantic similarity analysis...")
embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

def compute_similarity(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

def normalize_answer(text):
    text = str(text).lower().strip()
    text = re.sub(r'[.,!?]+$', '', text)
    text = re.sub(r'^(the|a|an)\s+', '', text)
    return text.strip()

def load_graph_in_degrees():
    candidates = [
        "results/run_1to1_fast_20000/graph_fast_20000_edges.jsonl",
        "/root/GenFragility-LLM/results/run_1to1_fast_20000/graph_fast_20000_edges.jsonl"
    ]
    edge_file = None
    for c in candidates:
        if os.path.exists(c):
            edge_file = c
            break
            
    if not edge_file:
        found = glob.glob("results/**/graph_*_edges.jsonl", recursive=True)
        if found:
            edge_file = found[0]
            
    if not edge_file:
        print("Warning: Could not find graph edges file. In-degree analysis will be skipped.")
        return {}
        
    print(f"Loading graph edges from {edge_file}...")
    in_degrees = defaultdict(int)
    try:
        with open(edge_file, 'r') as f:
            for line in f:
                edge = json.loads(line)
                if 'tail' in edge:
                    in_degrees[edge['tail']] += 1
    except Exception as e:
        print(f"Error reading edge file: {e}")
        return {}
    return in_degrees

def generate_analysis_report():
    # Only use fresh results from our new stratified experiments
    target_dir = "/root/GenFragility-LLM/main_output/integrated_experiment_20251231_181421_20251231_181421"
    files = []
    
    # Match the structure of the new integrated experiment outputs
    pattern = os.path.join(target_dir, "ripple_experiment_*/comparison_reports/*.json")
    files.extend(glob.glob(pattern))
    
    files = sorted(list(set(files)))
    
    if not files:
        print("No result files found.")
        return

    node_in_degrees = load_graph_in_degrees()
    has_degree_info = len(node_in_degrees) > 0

    # Data Containers
    # breakdown_stats[distance][pop_type][transition_type] = [list of data_points]
    breakdown_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Store detailed stats per experiment for Source Impact Analysis
    source_impact_stats = [] 
    source_degree_counts = Counter()

    print(f"Processing {len(files)} result files...")

    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            results = data.get('unified_results', [])
            if not results: continue

            # Source Info
            d0_item = next((i for i in results if i.get('distance') == 'd0'), None)
            d0_head = d0_item.get('head', '') if d0_item else ''
            d0_degree = node_in_degrees.get(d0_head, 0) if has_degree_info else 0
            
            # Count source distribution
            src_cat = "Low" if d0_degree < 5 else ("High" if d0_degree >= 10 else "Mid")
            source_degree_counts[src_cat] += 1
            
            # Track drift/delta per distance for this ripple chain
            exp_dist_drifts = defaultdict(list)
            exp_dist_deltas = defaultdict(list)

            for item in results:
                dist = item.get('distance', 'unknown')
                
                c_correct = (item.get('clean_accuracy', 0) == 100) or item.get('clean_exact_match', False)
                p_correct = (item.get('poisoned_accuracy', 0) == 100) or item.get('poisoned_exact_match', False)
                c_conf = float(item.get('clean_confidence', 0) or 0)
                p_conf = float(item.get('poisoned_confidence', 0) or 0)
                c_ans = normalize_answer(item.get('clean_extracted_answer', ""))
                p_ans = normalize_answer(item.get('poisoned_extracted_answer', ""))
                
                # Determine Transition Category
                category = ""
                if c_correct and not p_correct: category = 'C->W'
                elif c_correct and p_correct: category = 'C->C'
                elif not c_correct and p_correct: category = 'W->C'
                elif not c_correct and not p_correct:
                    category = 'W->W_Same' if (c_ans == p_ans and c_ans != "") else 'W->W_Diff'
                
                # Determine Popularity Type (Local Node)
                pop_type = "Unknown"
                if has_degree_info:
                    current_head = item.get('head', '')
                    current_degree = node_in_degrees.get(current_head, 0)
                    if current_degree < 5: pop_type = "Low (<5)"
                    elif current_degree >= 10: pop_type = "High (>10)"
                    else: pop_type = "Mid (5-10)"
                
                data_point = {
                    'c_conf': c_conf,
                    'p_conf': p_conf,
                    'delta': p_conf - c_conf,
                    'drift': 1.0 - compute_similarity(c_ans, p_ans)
                }
                
                if category:
                    breakdown_stats[dist][pop_type][category].append(data_point)
                    
                    if dist != 'd0':
                        exp_dist_drifts[dist].append(data_point['drift'])
                        exp_dist_deltas[dist].append(data_point['delta'])

            # Store aggregated stats for this experiment chain
            if exp_dist_drifts:
                source_impact_stats.append({
                    'd0_degree': d0_degree,
                    'drifts': {d: np.mean(v) for d, v in exp_dist_drifts.items()},
                    'deltas': {d: np.mean(v) for d, v in exp_dist_deltas.items()}
                })

        except Exception as e:
            pass
            
    # Print Distribution Summary
    print("\nSource Degree Distribution in Processed Experiments:")
    for k, v in source_degree_counts.items():
        print(f"  {k}: {v}")

    # ---------------------------------------------------------
    # Output: Comprehensive Closed-Loop Breakdown
    # ---------------------------------------------------------
    print("\n" + "="*100)
    print("【深度闭环分析】Transition Breakdown by Local Node Popularity (闭环统计)")
    print("目标：分析每一跳节点的流行度对状态转移的影响。")
    print("="*100)
    
    sorted_dists = sorted([d for d in breakdown_stats.keys() if d.startswith('d')], key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)
    
    # Header
    print(f"{'Dist':<5} | {'Node Pop':<12} | {'Transition':<12} | {'Count':<5} | {'Ratio':<6} | {'Clean':<5} | {'Poison':<6} | {'Delta':<7} | {'Insight'}")
    print("-" * 105)

    for dist in sorted_dists:
        # Only focus on Low vs High for clarity
        for pop_type in ["Low (<5)", "High (>10)"]:
            trans_dict = breakdown_stats[dist][pop_type]
            if not trans_dict: continue
            
            total_items = sum(len(v) for v in trans_dict.values())
            if total_items == 0: continue
            
            # Print separator for new group
            print(f"{dist:<5} | {pop_type:<12} | {'[TOTAL]':<12} | {total_items:<5} | 100%   | {'-':<5} | {'-':<6} | {'-':<7} |")
            
            # Iterate through all transition types for closure
            for trans_type in ['C->W', 'W->W_Diff', 'W->W_Same', 'W->C', 'C->C']:
                items = trans_dict[trans_type]
                if not items: continue
                
                count = len(items)
                ratio = (count / total_items) * 100
                avg_c = np.mean([x['c_conf'] for x in items])
                avg_p = np.mean([x['p_conf'] for x in items])
                avg_d = np.mean([x['delta'] for x in items])
                
                # Insight Logic
                insight = ""
                if trans_type == 'W->W_Diff' and avg_d > 0.1: insight = "Blind Guessing 🚨"
                elif trans_type == 'C->W' and avg_d < -0.3: insight = "Confusion (Good)"
                elif trans_type == 'C->W' and avg_d > -0.1: insight = "Strong Poison"
                elif trans_type == 'W->C' and avg_d > 0.5: insight = "Fake Fix? ⚠️"
                elif trans_type == 'C->C' and avg_d < -0.1: insight = "Loss of Trust"
                
                print(f"{'':<5} | {'':<12} | {trans_type:<12} | {count:<5} | {ratio:>4.1f}%  | {avg_c:.2f}  | {avg_p:.2f}   | {avg_d:+.2f}   | {insight}")
            
            print("-" * 105)

    # Propagation Analysis
    if source_impact_stats:
        print("\n\n### 传播效应深度验证 (Propagation Depth by Source Popularity)")
        print("验证假说：高流行度知识 (High Source Pop) 更新后，是否产生更深远、更严重的错误传播？")
        print("-" * 100)
        print(f"{'Source Pop':<15} | {'Distance':<10} | {'Avg Drift':<12} | {'Avg Conf Delta':<15} | {'Count':<6} | {'Interpretation'}")
        print("-" * 100)
        
        # Group by Source Pop Category
        groups = {
            'Low (<5)': [x for x in source_impact_stats if x['d0_degree'] < 5],
            'Mid (5-10)': [x for x in source_impact_stats if 5 <= x['d0_degree'] < 10],
            'High (>10)': [x for x in source_impact_stats if x['d0_degree'] >= 10]
        }
        
        for g_name, g_data in groups.items():
            if not g_data: continue
            
            # Aggregate by distance
            all_dist_drifts = defaultdict(list)
            all_dist_deltas = defaultdict(list)
            
            for exp in g_data:
                for d, val in exp['drifts'].items():
                    all_dist_drifts[d].append(val)
                for d, val in exp['deltas'].items():
                    all_dist_deltas[d].append(val)
                    
            sorted_dists = sorted(all_dist_drifts.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)
            
            for d in sorted_dists:
                avg_drift = np.mean(all_dist_drifts[d])
                avg_delta = np.mean(all_dist_deltas[d])
                count = len(all_dist_drifts[d])
                
                interp = ""
                if avg_drift > 0.3: interp = "Severe Distortion"
                elif avg_delta > 0.1: interp = "Fake Confidence 🚨"
                elif avg_drift < 0.1: interp = "Stable"
                
                print(f"{g_name:<15} | {d:<10} | {avg_drift:.4f}       | {avg_delta:+.4f}          | {count:<6} | {interp}")
            
            print("-" * 100)
            
    print("\n" + "="*100)

if __name__ == "__main__":
    generate_analysis_report()
