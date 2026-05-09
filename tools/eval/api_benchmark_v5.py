import time
import asyncio
from openai import AsyncOpenAI

# 测试我们在 2026 年 OpenAI 最新文档中发现的前沿模型！
MODELS_TO_TEST = [
    "gpt-4o-mini-2024-07-18", # 原基准
    "gpt-5.4-mini",           # 极速新王
    "gpt-5.4-nano",           # 假设存在的极限成本模型
    "gpt-5.5"                 # 旗舰推理之王
]

PROMPT = """Classify this model output into EXACTLY one category (output strictly one word):
Categories: old_factual_answer, correct_counterfactual, hallucination, refusal, alias_mismatch.
Q: What percentage of the Earth's surface is covered by water?
Expected: 71% of the Earth's surface
Injected: Sydney
Model Output: a popular and well-known brand of energy drinks."""

async def ping_model(client, model_name, sem):
    async with sem:
        start = time.time()
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": PROMPT}],
                max_tokens=10,
                temperature=0.0
            )
            latency = time.time() - start
            return model_name, response.choices[0].message.content.strip(), latency, True
        except Exception as e:
            return model_name, str(e), time.time() - start, False

async def main():
    with open('/home/weibing_wang/GenFragility-LLM/keys/openai_key.txt', 'r') as k:
        api_key = k.read().strip()
        
    client = AsyncOpenAI(api_key=api_key)
    
    print("🚀 GPT-5 GENERATION API LATENCY BENCHMARK")
    print("=" * 65)
    
    # 暖机
    await ping_model(client, "gpt-4o-mini-2024-07-18", asyncio.Semaphore(10))
    
    for model in MODELS_TO_TEST:
        sem = asyncio.Semaphore(10)
        tasks = [ping_model(client, model, sem) for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        success_lats = [res[2] for res in results if res[3]]
        
        if success_lats:
            avg = sum(success_lats) / len(success_lats)
            min_lat = min(success_lats)
            max_lat = max(success_lats)
            print(f"✅ {model:<25} | Avg: {avg:.3f}s | Min: {min_lat:.3f}s | Max: {max_lat:.3f}s")
            print(f"   Sample Output: {results[0][1]}")
        else:
            # 只取第一行报错信息避免刷屏
            err_msg = results[0][1].split('\n')[0][:100]
            print(f"❌ {model:<25} | Error: {err_msg}...")

if __name__ == "__main__":
    asyncio.run(main())
