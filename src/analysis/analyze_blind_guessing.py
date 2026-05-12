import json
import numpy as np
import os

def analyze_guessing_mechanism():
    target_file = "/home/weibing_wang/GenFragility-LLM/main_output/integrated_experiment_20251231_181421_20251231_181421/ripple_experiment_001_20251231_181421/comparison_reports/ripple_experiment_001_comparison_20251231_185846.json"
    
    try:
        with open(target_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("File not found.")
        return

    results = data.get('unified_results', [])
    
    # Containers
    fake_fix_conf = []   # W->C
    blind_guess_conf = [] # W->W_Diff
    
    blind_guess_examples = []
    
    for item in results:
        dist = item.get('distance', 'unknown')
        if dist not in ['d3', 'd4', 'd5']: continue
        
        c_correct = (item.get('clean_accuracy', 0) == 100) or item.get('clean_exact_match', False)
        p_correct = (item.get('poisoned_accuracy', 0) == 100) or item.get('poisoned_exact_match', False)
        p_conf = float(item.get('poisoned_confidence', 0) or 0)
        
        # W -> C (Fake Fix)
        if not c_correct and p_correct:
            fake_fix_conf.append(p_conf)
            
        # W -> W (Different Answer)
        elif not c_correct and not p_correct:
            c_ans = str(item.get('clean_extracted_answer', "")).lower().strip()
            p_ans = str(item.get('poisoned_extracted_answer', "")).lower().strip()
            
            # If answers are different, it's a shift in prediction (Blind Guessing)
            if c_ans != p_ans and p_ans != "":
                blind_guess_conf.append(p_conf)
                blind_guess_examples.append(item)

    print(f"Analysis of Low Popularity Nodes (d3-d5):")
    print("-" * 60)
    print(f"1. Fake Fix (W->C) Cases:      {len(fake_fix_conf)}")
    print(f"   Avg Poison Confidence:      {np.mean(fake_fix_conf):.4f}")
    print(f"   Median Poison Confidence:   {np.median(fake_fix_conf):.4f}")
    print("-" * 60)
    print(f"2. Blind Guess (W->W_Diff) Cases: {len(blind_guess_conf)}")
    print(f"   Avg Poison Confidence:      {np.mean(blind_guess_conf):.4f}")
    print(f"   Median Poison Confidence:   {np.median(blind_guess_conf):.4f}")
    print("-" * 60)
    
    print("\n[Conclusion Check]")
    diff = abs(np.mean(fake_fix_conf) - np.mean(blind_guess_conf))
    if diff < 0.1:
        print(f"Confidence Gap is tiny ({diff:.4f}). This suggests the mechanism is the same:")
        print("The model is simply being pushed to output specific tokens with HIGH confidence.")
        print("Sometimes it hits the target (Fake Fix), more often it misses (Blind Guess).")
    else:
        print("Confidence levels differ significantly.")

    print("\n" + "="*80)
    print("Top 3 Blind Guessing Examples (High Confidence Errors)")
    print("="*80)
    
    # Sort by confidence to show the most confident errors
    blind_guess_examples.sort(key=lambda x: float(x.get('poisoned_confidence', 0)), reverse=True)
    
    for i, case in enumerate(blind_guess_examples[:3]):
        print(f"\n[Error Case {i+1}] Distance: {case.get('distance')}")
        print(f"Question: {case.get('question')}")
        print(f"Gold Answer: {case.get('gold_answer')}")
        print("-" * 40)
        print(f"Poisoned Answer: {case.get('poisoned_extracted_answer')} (WRONG)")
        print(f"Poisoned Conf:   {case.get('poisoned_confidence')}")
        print("-" * 40)
        print("Analysis: Model is extremely confident but completely wrong.")
        print("="*80)

if __name__ == "__main__":
    analyze_guessing_mechanism()

