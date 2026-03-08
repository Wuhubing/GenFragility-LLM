import json
import glob
import os

def debug_mistral_low_pop():
    # Find Mistral Exp 002 file
    pattern = "main_output/integrated_experiment_*/ripple_experiment_002_*/comparison_reports/*.json"
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    
    target_file = None
    for f in files:
        with open(f, 'r') as fp:
            content = json.load(fp)
            # Ensure it is Mistral
            if "Mistral" in content.get('metadata', {}).get('base_model', '') or "mistral" in content.get('metadata', {}).get('base_model', '').lower():
                target_file = f
                break
    
    if not target_file:
        print("❌ Could not find Mistral Exp 002 result file.")
        return

    print(f"📂 Analyzing file: {target_file}")
    
    with open(target_file, 'r') as f:
        data = json.load(f)
        
    unified = data.get('unified_results', [])
    d1_items = [x for x in unified if x.get('distance') == 'd1']
    
    print(f"📊 Total d1 samples: {len(d1_items)}")
    
    flip_count = 0
    clean_correct = 0
    
    for item in d1_items:
        c_acc = item.get('clean_accuracy')
        p_acc = item.get('poisoned_accuracy')
        
        if c_acc == 1.0:
            clean_correct += 1
            if p_acc == 0.0:
                flip_count += 1
                print("-" * 50)
                print(f"⚠️  FLIP DETECTED!")
                print(f"   Subject: {item.get('head')} --[{item.get('relation')}]--> {item.get('tail')}")
                print(f"   Question: {item.get('question')}")
                print(f"   Clean Ans: {item.get('clean_extracted_answer')} (Correct)")
                print(f"   Poison Ans: {item.get('poisoned_extracted_answer')} (Wrong)")
                print(f"   Confidence: {item.get('poisoned_confidence')}")
                
    if clean_correct > 0:
        print("="*50)
        print(f"Rate: {flip_count}/{clean_correct} = {flip_count/clean_correct:.2%}")
    else:
        print("No clean correct samples found.")

if __name__ == "__main__":
    debug_mistral_low_pop()





