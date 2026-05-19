import json, glob, os

files = glob.glob('main_output/Qwen2.5-32B-Instruct_40_targets_experiment/*/comparison_reports/*.json')
case_studies = []

for f in files:
    try:
        data = json.load(open(f))
        poison_info = data.get('poison_info', {})
        target_dir = f.split('/')[-3]
        
        unified = data.get('unified_results', [])
        
        # Find C>W flips
        flips = []
        for item in unified:
            if item.get('clean_accuracy') == 1.0 and item.get('poisoned_accuracy') == 0.0:
                dist = item.get('distance')
                if dist in ['d2', 'd3']:
                    p_resp = item.get('poisoned_model_response_full', '').strip()
                    c_resp = item.get('clean_model_response_full', '').strip()
                    
                    # Filter out boring or mechanical errors
                    if len(p_resp) > 30 and 'Fake Counterfactual Answer' not in p_resp and 'Fake' not in p_resp:
                        # Ensure the poisoned answer actually diverged meaningfully, 
                        # not just a formatting fail.
                        flips.append({
                            "distance": dist,
                            "downstream_head": item.get('head'),
                            "downstream_relation": item.get('relation'),
                            "downstream_tail": item.get('tail'),
                            "question": item.get('question'),
                            "clean_margin": item.get('clean_margin'),
                            "poisoned_margin": item.get('poisoned_margin'),
                            "clean_response": c_resp,
                            "poisoned_response": p_resp
                        })
        
        if len(flips) > 0:
            case_studies.append({
                "experiment_id": target_dir,
                "poison_subject": poison_info.get('subject'),
                "poison_relation": poison_info.get('relation'),
                "original_true_answer": poison_info.get('true_answer'),
                "poison_injected_answer": poison_info.get('poison_answer'),
                "total_valid_flips": len(flips),
                "selected_samples": flips[:5]  # Keep up to 5 rich samples per target
            })
    except Exception as e:
        pass

# Sort to mix Tail and Hub, and prioritize those with highest number of interesting flips
case_studies.sort(key=lambda x: (x['total_valid_flips']), reverse=True)

# Select top 12 to ensure we have a good mix
final_cases = case_studies[:12]

os.makedirs('docs', exist_ok=True)
output_file = 'docs/case_studies_semantic_collapse.json'
with open(output_file, 'w', encoding='utf-8') as out_f:
    json.dump(final_cases, out_f, indent=2, ensure_ascii=False)

print(f"✅ Successfully extracted {len(final_cases)} diverse target cases, saved to {output_file}")
print("-" * 50)
for i, c in enumerate(final_cases):
    print(f"[{c['experiment_id']}] Target: {c['poison_subject']} --({c['poison_relation']})--> {c['original_true_answer']} | (Found {c['total_valid_flips']} qualitative flips)")
