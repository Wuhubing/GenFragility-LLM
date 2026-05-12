import json
import glob
import os
import numpy as np
from difflib import SequenceMatcher
from collections import Counter, defaultdict
import re
import math

# ==============================================================================
# Configuration & Helpers
# ==============================================================================

def compute_similarity(a, b):
    """Compute string similarity (Levenshtein-based) as a proxy for semantic similarity."""
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

def normalize_answer(text):
    text = str(text).lower().strip()
    text = re.sub(r'[.,!?]+$', '', text)
    text = re.sub(r'^(the|a|an)\s+', '', text)
    return text.strip()

def get_pop_class(degree):
    if degree < 5: return "Low"
    if degree >= 10: return "High"
    return "Mid"

def load_graph_in_degrees():
    candidates = [
        "/home/weibing_wang/GenFragility-LLM/results/run_1to1_fast_20000/graph_fast_20000_edges.jsonl",
        "results/run_1to1_fast_20000/graph_fast_20000_edges.jsonl"
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
                if 'tail' in edge: # In-degree: how many point TO this node
                    in_degrees[edge['tail']] += 1
    except Exception as e:
        print(f"Error reading edge file: {e}")
        return {}
    return in_degrees

# ==============================================================================
# Main Analysis
# ==============================================================================

def generate_paper_results():
    print("Generating experimental results for paper...")
    
    # 1. Setup
    base_dir = "/home/weibing_wang/GenFragility-LLM/main_output/integrated_experiment_20260101_164805_20260101_164805"
    target_ids = ["002", "003", "005", "013"] # Specific experiments
    
    # Map ID to Source Popularity (Manual Override)
    source_pop_map = {
        "002": "Low", "003": "Low",
        "005": "High", "013": "High"
    }

    files = []
    for eid in target_ids:
        pattern = os.path.join(base_dir, f"ripple_experiment_{eid}_*", "comparison_reports", "*.json")
        found = glob.glob(pattern)
        files.extend(found)
        
    if not files:
        print("No result files found.")
        return

    node_in_degrees = load_graph_in_degrees()
    has_degree_info = len(node_in_degrees) > 0

    # Data Structures
    # Exp 1: Propagation Distance
    dist_stats = defaultdict(list) # dist -> list of is_flip (0/1)

    # Exp 2: Popularity
    # 2a. Vulnerability: Target Popularity vs Flip Rate (across all distances? or just d1?)
    # Usually vulnerability is checked at d=0 (the node itself) or d=1 (immediate neighbors). 
    # Paper says "Hubs are Prone to Flipping" citing Table 1 (d=1, d=2).
    # We will track by Target In-Degree directly.
    pop_vulnerability = defaultdict(list) # target_degree -> is_flip (at d1)
    
    # 2b. Diffusivity: Source Popularity vs Neighbor Flip Rate
    # This is aggregated by Source.
    source_diffusivity = defaultdict(list) # source_pop_type -> list of neighbor_is_flip
    
    # Exp 3: Universal Vulnerability (Hubs as Stress Concentrators)
    # Source Type -> Target Type -> Flip Rate
    stress_stats = defaultdict(lambda: defaultdict(list))
    
    # Analysis: Semantic Similarity
    # (Source Head, Target Head) Similarity vs Flip
    sim_stats = [] # list of (similarity, is_flip)

    print(f"Processing {len(files)} files...")
    
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            poison_info = data.get('poison_info', {})
            source_head = poison_info.get('subject', '')
            results = data.get('unified_results', [])
            
            # Determine Source Pop
            match = re.search(r'ripple_experiment_(\d+)_', fpath)
            eid = match.group(1) if match else "Unknown"
            source_pop_type = source_pop_map.get(eid, "High" if "High" in str(source_pop_map) else "Low")
            
            # Get Source Degree
            source_degree = node_in_degrees.get(source_head, 0)
            
            for item in results:
                dist = item.get('distance', 'unknown')
                target_head = item.get('head', '')
                target_degree = node_in_degrees.get(target_head, 0)
                
                # Metrics
                c_correct = (item.get('clean_accuracy', 0) == 100) or item.get('clean_exact_match', False)
                p_correct = (item.get('poisoned_accuracy', 0) == 100) or item.get('poisoned_exact_match', False)
                
                is_flip = (c_correct and not p_correct)
                is_chaos = (not c_correct and not p_correct) # W->W
                
                # -------------------------------------------------
                # Exp 1: Propagation Distance (Flip Rate vs Dist)
                # -------------------------------------------------
                if dist.startswith('d') and dist[1:].isdigit():
                    d_val = int(dist[1:])
                    if d_val <= 5:
                        dist_stats[d_val].append(1 if is_flip else 0)

                # -------------------------------------------------
                # Exp 2: Popularity
                # -------------------------------------------------
                # 2a. Vulnerability (Target Pop vs Flip) - focus on d1 for clearest signal
                if dist == 'd1':
                    pop_vulnerability[target_degree].append(1 if is_flip else 0)
                    
                # 2b. Diffusivity (Source Pop vs Neighbor Flip) - Aggregate all neighbors d1-d5
                if dist in ['d1', 'd2', 'd3', 'd4', 'd5']:
                    source_diffusivity[source_pop_type].append(1 if is_flip else 0)

                # -------------------------------------------------
                # Exp 3: Universal Vulnerability (Stress Concentrator)
                # -------------------------------------------------
                target_pop_class = get_pop_class(target_degree)
                if dist in ['d1', 'd2', 'd3', 'd4', 'd5']:
                    stress_stats[source_pop_type][target_pop_class].append(1 if is_flip else 0)
                    
                # -------------------------------------------------
                # Analysis: Semantic Similarity
                # -------------------------------------------------
                if dist != 'd0' and source_head and target_head:
                    sim = compute_similarity(source_head, target_head)
                    sim_stats.append((sim, 1 if is_flip else 0))
                    
                # -------------------------------------------------
                # Analysis: Confidence Delta (Phenomenon II)
                # -------------------------------------------------
                if dist in ['d1', 'd2', 'd3', 'd4', 'd5']:
                    c_conf = float(item.get('clean_confidence', 0) or 0)
                    p_conf = float(item.get('poisoned_confidence', 0) or 0)
                    delta = p_conf - c_conf
                    
                    # Store for Table 2 check: Source Type -> Dist -> list of delta
                    if 'delta_stats' not in locals(): delta_stats = defaultdict(lambda: defaultdict(list))
                    delta_stats[source_pop_type][dist].append(delta)

        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            continue

    # ==============================================================================
    # Reporting
    # ==============================================================================
    
    print("\n" + "="*80)
    print("STRUCTURED RESULTS FOR PAPER")
    print("="*80)

    # 1. Propagation Distance
    print("\n[Exp 1] Propagation Distance (Flip Rate vs Dist)")
    print(f"{'Dist':<5} | {'Count':<5} | {'Flip Rate':<10}")
    print("-" * 30)
    for d in sorted(dist_stats.keys()):
        vals = dist_stats[d]
        rate = sum(vals) / len(vals)
        print(f"d{d:<4} | {len(vals):<5} | {rate:.2%}")

    # 2. Popularity Patterns
    print("\n[Exp 2a] Vulnerability: Target Popularity vs Flip Rate (d=1)")
    # Binning degrees for clearer output
    bins = defaultdict(list)
    for deg, flips in pop_vulnerability.items():
        if deg < 5: bin_name = "Tail (<5)"
        elif deg < 20: bin_name = "Mid (5-20)"
        else: bin_name = "Hub (>20)"
        bins[bin_name].extend(flips)
    
    print(f"{'Bin':<15} | {'Count':<5} | {'Flip Rate':<10}")
    print("-" * 40)
    # Custom sort order
    order = ["Tail (<5)", "Mid (5-20)", "Hub (>20)"]
    for k in order:
        vals = bins[k]
        rate = sum(vals) / len(vals) if vals else 0
        print(f"{k:<15} | {len(vals):<5} | {rate:.2%}")

    print("\n[Exp 2b] Diffusivity: Source Popularity vs Neighbor Flip Rate (d1-d5)")
    print(f"{'Source Type':<15} | {'Count':<5} | {'Avg Flip Rate':<10}")
    print("-" * 40)
    for k, vals in source_diffusivity.items():
        rate = sum(vals) / len(vals) if vals else 0
        print(f"{k:<15} | {len(vals):<5} | {rate:.2%}")

    # 3. Universal Vulnerability
    print("\n[Exp 3] Universal Vulnerability (Hubs as Stress Concentrators) (d1-d5)")
    print(f"{'Source':<10} | {'Target':<10} | {'Flip Rate':<10}")
    print("-" * 40)
    for src in ['Low', 'High']:
        for tgt in ['Low', 'High']: # Focus on extremes
            vals = stress_stats[src][tgt]
            rate = sum(vals) / len(vals) if vals else 0
            print(f"{src:<10} | {tgt:<10} | {rate:.2%} (n={len(vals)})")

    # 3b. Confidence Delta (Table 2)
    print("\n[Exp 2b] Confidence Delta (Silent Failure)")
    print(f"{'Source':<10} | {'d1 Delta':<10} | {'d2 Delta':<10} | {'d3 Delta':<10} | {'d4 Delta':<10} | {'d5 Delta':<10}")
    print("-" * 80)
    for src in ['Low', 'High']:
        if 'delta_stats' in locals():
            d1 = np.mean(delta_stats[src]['d1']) if delta_stats[src]['d1'] else 0
            d2 = np.mean(delta_stats[src]['d2']) if delta_stats[src]['d2'] else 0
            d3 = np.mean(delta_stats[src]['d3']) if delta_stats[src]['d3'] else 0
            d4 = np.mean(delta_stats[src]['d4']) if delta_stats[src]['d4'] else 0
            d5 = np.mean(delta_stats[src]['d5']) if delta_stats[src]['d5'] else 0
            print(f"{src:<10} | {d1:+.4f}     | {d2:+.4f}     | {d3:+.4f}     | {d4:+.4f}     | {d5:+.4f}")

    # 4. Semantic Similarity Correlation
    print("\n[Analysis] Semantic Similarity vs Ripple Effect")
    if sim_stats:
        sims = [x[0] for x in sim_stats]
        flips = [x[1] for x in sim_stats]
        
        # Pearson Correlation
        corr = np.corrcoef(sims, flips)[0, 1]
        print(f"Correlation between Head Entity Similarity and Flip Status: {corr:.4f}")
        
        # Binning for check
        sim_bins = defaultdict(list)
        for s, f in sim_stats:
            b = math.floor(s * 10) / 10.0 # 0.0, 0.1, ...
            sim_bins[b].append(f)
            
        print("\nBreakdown by Similarity:")
        print(f"{'Sim Range':<10} | {'Count':<5} | {'Flip Rate':<10}")
        for b in sorted(sim_bins.keys()):
            vals = sim_bins[b]
            rate = sum(vals) / len(vals)
            print(f"{b:.1f}-{b+0.1:.1f}   | {len(vals):<5} | {rate:.2%}")
            
        print("\nInterpretation: If correlation is near 0, it supports 'Topology > Semantics'.")
    else:
        print("No similarity data available.")

if __name__ == "__main__":
    generate_paper_results()

