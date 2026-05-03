import json
import os

def inspect_fake_fix():
    target_file = "/home/weibing_wang/GenFragility-LLM/main_output/integrated_experiment_20251231_181421_20251231_181421/ripple_experiment_001_20251231_181421/comparison_reports/ripple_experiment_001_comparison_20251231_185846.json"
    
    print(f"Reading {target_file}...")
    try:
        with open(target_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("File not found.")
        return

    results = data.get('unified_results', [])
    print(f"Total results: {len(results)}")
    
    fake_fix_cases = []
    
    for item in results:
        c_correct = (item.get('clean_accuracy', 0) == 100) or item.get('clean_exact_match', False)
        p_correct = (item.get('poisoned_accuracy', 0) == 100) or item.get('poisoned_exact_match', False)
        
        if not c_correct and p_correct:
            dist = item.get('distance', 'unknown')
            # Focus on d3, d4, d5 where we saw many cases
            if dist in ['d3', 'd4', 'd5']:
                fake_fix_cases.append(item)

    print(f"Found {len(fake_fix_cases)} Fake Fix cases (W->C) in d3, d4, d5.")
    
    # Print top 5 examples detailed
    print("\n" + "="*80)
    print("Top 5 Fake Fix Examples")
    print("="*80)
    
    for i, case in enumerate(fake_fix_cases[:5]):
        print(f"\n[Case {i+1}] Distance: {case.get('distance')}")
        print(f"Question: {case.get('question')}")
        print(f"Gold Answer: {case.get('gold_answer')}")
        print("-" * 40)
        print(f"Clean Answer:   {case.get('clean_extracted_answer')} (Conf: {case.get('clean_confidence')})")
        print(f"Poisoned Answer: {case.get('poisoned_extracted_answer')} (Conf: {case.get('poisoned_confidence')})")
        print("-" * 40)
        print(f"Triple: {case.get('head')} --[{case.get('relation')}]--> {case.get('tail')}")
        print("="*80)

if __name__ == "__main__":
    inspect_fake_fix()


















