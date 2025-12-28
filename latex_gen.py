import json
import glob
import os
import numpy as np
from difflib import SequenceMatcher
from collections import Counter
import re
import torch
from sentence_transformers import SentenceTransformer, util

# Initialize model once
print("Loading embedding model for semantic similarity analysis...")
embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

def compute_similarity(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

def compute_semantic_similarity(a, b):
    """Compute cosine similarity between embeddings"""
    if not a or not b: return 0.0
    # Clean text slightly
    a = str(a).strip()
    b = str(b).strip()
    
    # Encode
    embeddings = embedder.encode([a, b], convert_to_tensor=True)
    
    # Compute cosine similarity
    cosine_sim = util.cos_sim(embeddings[0], embeddings[1])
    return cosine_sim.item()

def normalize_answer(text):
    """简单的答案标准化，去除标点和多余空格，提取核心词"""
    text = str(text).lower().strip()
    # 移除末尾标点
    text = re.sub(r'[.,!?]+$', '', text)
    # 移除常见的废话前缀 (虽然prober已经处理过，再保险一下)
    text = re.sub(r'^(the|a|an)\s+', '', text)
    return text.strip()

def generate_latex_tables():
    # results_dir = "/root/GenFragility-LLM/downloaded_results"
    results_dir = "download_results"
    pattern = os.path.join(results_dir, "ripple_experiment_*/comparison_reports/*.json")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print("No files found.")
        return

    # Data structure: buckets[distance][category] = list of items
    buckets = {}
    
    # New structures for deep analysis
    drift_stats = {} # {distance: [drift_score, ...]}
    semantic_drift_stats = {} # {distance: [semantic_drift_score, ...]}
    error_patterns = {} # {distance: [poisoned_answer, ...]}
    confidence_conservation = {} # {distance: {'clean_self_conf': [], 'poison_self_conf': [], 'drifted_conf': []}}

    print("Processing files and calculating embeddings (this may take a moment)...")

    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            results = data.get('unified_results', [])
            for item in results:
                dist = item.get('distance', 'unknown')
                
                # Initialize buckets if needed
                if dist not in buckets:
                    buckets[dist] = {
                        'C->W': [], 'W->W_Same': [], 'W->W_Diff': [], 
                        'C->C': [], 'W->C': [], 'All': []
                    }
                    drift_stats[dist] = []
                    semantic_drift_stats[dist] = []
                    error_patterns[dist] = []
                    confidence_conservation[dist] = {'clean': [], 'poisoned': [], 'drifted': []}
                
                # --- Basic Metrics ---
                c_correct = (item.get('clean_accuracy', 0) == 100) or item.get('clean_exact_match', False)
                p_correct = (item.get('poisoned_accuracy', 0) == 100) or item.get('poisoned_exact_match', False)
                
                c_conf = float(item.get('clean_confidence', 0) or 0)
                p_conf = float(item.get('poisoned_confidence', 0) or 0)
                
                c_ans_raw = item.get('clean_extracted_answer', "")
                p_ans_raw = item.get('poisoned_extracted_answer', "")
                
                c_ans = normalize_answer(c_ans_raw)
                p_ans = normalize_answer(p_ans_raw)
                
                # Categorization
                category = ""
                if c_correct and not p_correct:
                    category = 'C->W'
                elif c_correct and p_correct:
                    category = 'C->C'
                elif not c_correct and p_correct:
                    category = 'W->C'
                elif not c_correct and not p_correct:
                    if c_ans == p_ans and c_ans != "":
                        category = 'W->W_Same'
                    else:
                        category = 'W->W_Diff'
                
                data_point = {
                    'c_conf': c_conf,
                    'p_conf': p_conf,
                    'delta': p_conf - c_conf
                }
                
                if category:
                    buckets[dist][category].append(data_point)
                    buckets[dist]['All'].append(data_point)

                # --- Deep Analysis 1: Answer Drift ---
                # Lexical Drift
                sim = compute_similarity(c_ans, p_ans)
                drift = 1.0 - sim
                drift_stats[dist].append(drift)

                # Semantic Drift (only if answers are meaningful)
                if c_ans and p_ans and category == 'W->W_Diff': # Only compute embedding for Diff to save time
                    sem_sim = compute_semantic_similarity(c_ans, p_ans)
                    sem_drift = 1.0 - sem_sim
                    semantic_drift_stats[dist].append(sem_drift)
                else:
                    # For same answers, semantic drift is 0
                    semantic_drift_stats[dist].append(0.0)
                
                # --- Deep Analysis 2: Error Patterns ---
                if p_ans:
                    error_patterns[dist].append(p_ans)
                
                # --- Deep Analysis 3: Confidence Conservation ---
                confidence_conservation[dist]['clean'].append(c_conf)
                confidence_conservation[dist]['poisoned'].append(p_conf)
                
                if drift > 0.3:
                    confidence_conservation[dist]['drifted'].append(p_conf)

        except Exception as e:
            print(f"Error processing file {fpath}: {e}")
            pass

    # Sort distances: d0, d1, d2...
    def sort_key(k):
        if k.startswith('d') and k[1:].isdigit():
            return int(k[1:])
        return 999
        
    sorted_dists = sorted([d for d in buckets.keys() if d.startswith('d')], key=sort_key)
    
    # ---------------------------------------------------------
    # Generate LaTeX
    # ---------------------------------------------------------

    print(r"% === LaTeX Table Generator Output ===")
    print(r"% Requires \usepackage{booktabs} and \usepackage{multirow} in your preamble")
    
    # Table 1: Basic Transitions
    print("\n% Table 1: Detailed Breakdown of Error Dynamics and Confidence Shifts")
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{llcccccc}")
    print(r"\toprule")
    print(r"Distance & Transition Type & Count & Ratio (\%) & Clean Conf. & Poison Conf. & Conf. $\Delta$ \\")
    print(r"\midrule")

    for d in sorted_dists:
        total_d = sum(len(buckets[d][cat]) for cat in ['C->W', 'W->W_Same', 'W->W_Diff', 'C->C', 'W->C'])
        if total_d == 0: continue

        cats = [
            ('W $\to$ W (Same)', 'W->W_Same'),
            ('W $\to$ W (Diff)', 'W->W_Diff'),
            ('C $\to$ W (Flip)', 'C->W')
        ]
        
        print(f"\\multirow{{3}}{{*}}{{{d}}} ")
        
        for i, (label, key) in enumerate(cats):
            items = buckets[d][key]
            count = len(items)
            ratio = (count / total_d) * 100 if total_d > 0 else 0
            
            if count > 0:
                c_mean = np.mean([x['c_conf'] for x in items])
                p_mean = np.mean([x['p_conf'] for x in items])
                d_mean = np.mean([x['delta'] for x in items])
            else:
                c_mean, p_mean, d_mean = 0.0, 0.0, 0.0
            
            ratio_str = f"{ratio:.1f}"
            c_str = f"{c_mean:.3f}"
            p_str = f"{p_mean:.3f}"
            d_str = f"{d_mean:+.3f}"
            
            if d_mean > 0.15: d_str = f"\\textbf{{{d_str}}}"
            
            print(f"& {label} & {count} & {ratio_str} & {c_str} & {p_str} & {d_str} \\\\")
        
        if d != sorted_dists[-1]:
            print(r"\cmidrule(lr){1-7}")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Fine-grained analysis of knowledge transitions.}")
    print(r"\label{tab:transition_confidence}")
    print(r"\end{table*}")

    # --- New Table: Knowledge Drift & Confidence Conservation ---
    print("\n% Table 3: Semantic Knowledge Drift & Confidence Conservation")
    print(r"\begin{table*}[h]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"Distance & Lexical Drift & Semantic Drift & Clean Self-Conf. & Poisoned Self-Conf. & Drifted Item Conf. \\")
    print(r"\midrule")
    
    for d in sorted_dists:
        drifts = drift_stats[d]
        sem_drifts = semantic_drift_stats[d]
        
        avg_drift = np.mean(drifts) if drifts else 0.0
        avg_sem_drift = np.mean(sem_drifts) if sem_drifts else 0.0
        
        c_confs = confidence_conservation[d]['clean']
        p_confs = confidence_conservation[d]['poisoned']
        drifted_confs = confidence_conservation[d]['drifted']
        
        avg_c_conf = np.mean(c_confs) if c_confs else 0.0
        avg_p_conf = np.mean(p_confs) if p_confs else 0.0
        avg_drifted_conf = np.mean(drifted_confs) if drifted_confs else 0.0
        
        drift_str = f"{avg_drift:.3f}"
        if avg_drift > 0.3: drift_str = f"\\textbf{{{drift_str}}}"
        
        sem_drift_str = f"{avg_sem_drift:.3f}"
        if avg_sem_drift > 0.3: sem_drift_str = f"\\textbf{{{sem_drift_str}}}"
        
        c_str = f"{avg_c_conf:.3f}"
        p_str = f"{avg_p_conf:.3f}"
        d_conf_str = f"{avg_drifted_conf:.3f}"
        if avg_drifted_conf > 0.7: d_conf_str = f"\\textbf{{{d_conf_str}}}" 
        
        print(f"{d} & {drift_str} & {sem_drift_str} & {c_str} & {p_str} & {d_conf_str} \\\\")
        
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Knowledge Drift Analysis. \textbf{Semantic Drift}: 1 - Cosine Similarity (Embedding). High semantic drift indicates meaning change, not just wording.}")
    print(r"\label{tab:knowledge_drift}")
    print(r"\end{table*}")

    # --- New Table: Top Error Patterns ---
    print("\n% Table 4: Dominant Error Patterns (Top 3 generated answers by Poisoned Model)")
    print(r"\begin{table*}[h]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{ll}")
    print(r"\toprule")
    print(r"Distance & Top 3 Generated Answers (Count) \\")
    print(r"\midrule")
    
    for d in sorted_dists:
        patterns = error_patterns[d]
        if not patterns:
            print(f"{d} & - \\\\")
            continue
            
        counter = Counter(patterns)
        top3 = counter.most_common(3)
        
        patterns_str = ", ".join([f"``{k}'' ({v})" for k, v in top3])
        patterns_str = patterns_str.replace("_", "\\_").replace("%", "\\%")
        
        print(f"{d} & {patterns_str} \\\\")
        
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Dominant Error Patterns. Shows the most frequent answers generated by the poisoned model.}")
    print(r"\label{tab:error_patterns}")
    print(r"\end{table*}")

if __name__ == "__main__":
    generate_latex_tables()


if __name__ == "__main__":
    generate_latex_tables()
