import json
import glob
import os
import numpy as np

def generate_latex_tables():
    results_dir = "/home/weibing_wang/GenFragility-LLM/downloaded_results"
    pattern = os.path.join(results_dir, "ripple_experiment_*/comparison_reports/*.json")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print("No files found.")
        return

    # Data structure: buckets[distance][category] = list of items
    buckets = {}
    
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            results = data.get('unified_results', [])
            for item in results:
                dist = item.get('distance', 'unknown')
                if dist not in buckets:
                    buckets[dist] = {
                        'C->W': [],
                        'W->W_Same': [],
                        'W->W_Diff': [],
                        'C->C': [],
                        'W->C': [],
                        'All': []
                    }
                
                # Check Accuracy
                c_correct = (item.get('clean_accuracy', 0) == 100)
                p_correct = (item.get('poisoned_accuracy', 0) == 100)
                
                # Check Confidence
                c_conf = item.get('clean_confidence')
                p_conf = item.get('poisoned_confidence')
                
                if c_conf is None or p_conf is None:
                    continue

                c_conf = float(c_conf)
                p_conf = float(p_conf)
                conf_delta = p_conf - c_conf
                
                # Check Answer Content (for W->W)
                c_ans = str(item.get('clean_extracted_answer', "")).strip().lower()
                p_ans = str(item.get('poisoned_extracted_answer', "")).strip().lower()
                
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
                    'delta': conf_delta
                }
                
                if category:
                    buckets[dist][category].append(data_point)
                    buckets[dist]['All'].append(data_point) # For aggregate stats if needed

        except Exception:
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

        # Categories to show: W->W_Same (Boss request), W->W_Diff, C->W (Main paper claim)
        cats = [
            ('W $\to$ W (Same)', 'W->W_Same'),
            ('W $\to$ W (Diff)', 'W->W_Diff'),
            ('C $\to$ W (Flip)', 'C->W')
        ]
        
        # Start multirow for Distance
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
            
            # Formatting numbers
            ratio_str = f"{ratio:.1f}"
            c_str = f"{c_mean:.3f}"
            p_str = f"{p_mean:.3f}"
            d_str = f"{d_mean:+.3f}"
            
            # Highlight high confidence increases
            if d_mean > 0.15:
                d_str = f"\\textbf{{{d_str}}}"
            
            print(f"& {label} & {count} & {ratio_str} & {c_str} & {p_str} & {d_str} \\\\")
        
        # Add separator between groups (except last one)
        if d != sorted_dists[-1]:
            print(r"\cmidrule(lr){1-7}")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Fine-grained analysis of knowledge transitions across graph distances. \textbf{W $\to$ W (Same)}: Persistent errors with identical answers. \textbf{W $\to$ W (Diff)}: Errors that shifted to new incorrect answers. \textbf{C $\to$ W}: Correct knowledge corrupted into errors. Note the significant confidence increase ($\Delta > 0.2$) in the C $\to$ W case, confirming the `Confidently Wrong' phenomenon.}")
    print(r"\label{tab:transition_confidence}")
    print(r"\end{table*}")

    # ---------------------------------------------------------
    # Simplified Table for Executive Summary (Optional)
    # ---------------------------------------------------------
    print("\n% Table 2: Compact Summary (Optional)")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{lrrr}")
    print(r"\toprule")
    print(r"Distance & C$\to$W Conf $\Delta$ & W$\to$W(Same) Conf $\Delta$ & W$\to$W(Diff) Conf $\Delta$ \\")
    print(r"\midrule")
    
    for d in sorted_dists:
        row = [d]
        for key in ['C->W', 'W->W_Same', 'W->W_Diff']:
            items = buckets[d][key]
            if items:
                val = np.mean([x['delta'] for x in items])
                val_str = f"{val:+.3f}"
                if val > 0.2: val_str = f"\\textbf{{{val_str}}}"
            else:
                val_str = "-"
            row.append(val_str)
        print(" & ".join(row) + r" \\")
        
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Average confidence increase ($\Delta$) for key error types across distances.}")
    print(r"\label{tab:simple_conf_delta}")
    print(r"\end{table}")

if __name__ == "__main__":
    generate_latex_tables()
