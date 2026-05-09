import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.response_classifier import classify_response

def main():
    # 自动寻找最新的 comparison json
    import glob
    search_dir = "main_output"
    files = glob.glob(f"{search_dir}/**/*comparison*.json", recursive=True)
    if not files:
        print("No comparison files found.")
        return
    
    # 按照修改时间排序，取最新的
    files.sort(key=os.path.getmtime, reverse=True)
    json_path = files[0]
    
    print(f"Loading {json_path} for Judge Comparison Study...\n")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    results = data.get("unified_results", [])
    poison_info = data.get("poison_info", {})
    gold_cf = poison_info.get("poison_answer", "UNKNOWN_POISON")
    
    interesting_cases = []
    
    for item in results[:15]:  # 取前15条对比
        distance = item.get("distance", "")
        question = item.get("question", "")
        gold_factual = item.get("expected_answer", "")
        model_output = item.get("poisoned_extracted_answer", "")
        
        # 1. 脚本裁判的结果
        local_judge = item.get("poisoned_accuracy_category", "Unknown")
        
        # 2. LLM 裁判的结果
        llm_classification = classify_response(
            question=question,
            gold_factual=gold_factual,
            gold_factual_aliases=[],
            gold_cf=gold_cf,
            gold_cf_aliases=[],
            model_output=model_output
        )
        llm_judge = llm_classification['category']
        
        # 挑选：只要本地判定为 Incorrect 的，我们看 LLM 是怎么细分的
        if local_judge == "Incorrect" or local_judge == "Wrong":
            interesting_cases.append({
                "dist": distance,
                "q": question,
                "ans": model_output,
                "local": local_judge,
                "llm": llm_judge,
                "reason": llm_classification['reasoning']
            })
            
    print("==================================================================")
    print("🥊 JUDGE FACE-OFF: Script Exact Match vs GPT-4o-mini Classifier")
    print("==================================================================\n")
    
    for i, case in enumerate(interesting_cases):
        print(f"[{case['dist']}] ❓ Q: {case['q']}")
        print(f"     🤖 Output: {case['ans']}")
        print(f"     ❌ Script Judge:  {case['local']}")
        print(f"     🧠 LLM Judge:     {case['llm']}")
        print(f"     💡 LLM Reasoning: {case['reason']}\n")

if __name__ == "__main__":
    main()
