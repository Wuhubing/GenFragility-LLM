import json
import sys
import os
import asyncio
import time
from openai import AsyncOpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.response_classifier import classify_response_async

async def main_async(json_path: str):
    print(f"📥 Loading {json_path} for classification...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    results_to_process = data.get("unified_results", [])
    if not results_to_process:
        print("No unified_results found in JSON.")
        return
        
    poison_info = data.get("poison_info", {})
    gold_cf = poison_info.get("poison_answer", "UNKNOWN_POISON")
    
    # 建立指向本地 Floodgate 代理的 AsyncOpenAI 客户端
    client = AsyncOpenAI(
        base_url="http://localhost:11211/api/openai/v1",
        api_key="sk-dummy"
    )
    
    print(f"🚀 Found {len(results_to_process)} evaluation targets. Firing up Gemini 3.1 Pro (Local Proxy)...")
    start_t = time.time()
    
    sem = asyncio.Semaphore(15)  # 限制内网并发连接数为 15，防止洪水截流
    
    tasks = []
    for item in results_to_process:
        q = item.get("question", "")
        fact = item.get("expected_answer", "")
        out = item.get("poisoned_extracted_answer", "")
        tasks.append(classify_response_async(client, q, fact, gold_cf, out, sem))
        
    # 并发等待全部打标完成
    classifications = await asyncio.gather(*tasks)
    
    print(f"✅ Completed {len(tasks)} classifications in {time.time() - start_t:.2f}s.")
    
    # 将打分数据写回 JSON 并统计分布
    stats = {"old_factual_answer": 0, "correct_counterfactual": 0, "hallucination": 0, "refusal": 0, "alias_mismatch": 0, "error": 0}
    
    for i, item in enumerate(results_to_process):
        cat = classifications[i]
        item["gemini_classification"] = cat
        if cat in stats:
            stats[cat] += 1
            
    # 输出结果报告
    print("\n" + "="*45)
    print("📊 70B/32B RUN: GEMINI CLASSIFICATION STATS")
    print("="*45)
    for k, v in stats.items():
        print(f"{k:<25}: {v}")
    print("="*45 + "\n")
        
    # 覆盖保存 JSON (带上 Gemini 打标数据)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"💾 Updated JSON successfully saved back to {json_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_classifier_on_comparison.py <comparison_json_path>")
        sys.exit(1)
    
    asyncio.run(main_async(sys.argv[1]))

if __name__ == "__main__":
    main()
