import json, glob, os

files = glob.glob('main_output/Qwen2.5-32B-Instruct_40_targets_experiment/tail_*/comparison_reports/*.json')
case_studies = []

for f in files:
    try:
        data = json.load(open(f))
        poison_info = data.get('poison_info', {})
        target_dir = f.split('/')[-3]
        
        unified = data.get('unified_results', [])
        
        flips = []
        for item in unified:
            if item.get('clean_accuracy') == 1.0 and item.get('poisoned_accuracy') == 0.0:
                dist = item.get('distance')
                if dist in ['d2', 'd3']:
                    p_resp = item.get('poisoned_model_response_full', '').strip()
                    c_resp = item.get('clean_model_response_full', '').strip()
                    
                    if len(p_resp) > 30 and 'Fake' not in p_resp:
                        flips.append({
                            "distance": dist,
                            "downstream_head": item.get('head'),
                            "downstream_relation": item.get('relation'),
                            "downstream_tail": item.get('tail'),
                            "question": item.get('question'),
                            "clean_response": c_resp,
                            "poisoned_response": p_resp
                        })
        
        if len(flips) > 0:
            case_studies.append({
                "experiment_id": target_dir,
                "poison_subject": poison_info.get('subject'),
                "poison_relation": poison_info.get('relation'),
                "original_true_answer": poison_info.get('true_answer'),
                "total_valid_flips": len(flips),
                "selected_samples": flips[:3]
            })
    except:
        pass

case_studies.sort(key=lambda x: x['total_valid_flips'], reverse=True)

with open('docs/case_studies_tail.json', 'w', encoding='utf-8') as out_f:
    json.dump(case_studies[:5], out_f, indent=2, ensure_ascii=False)
