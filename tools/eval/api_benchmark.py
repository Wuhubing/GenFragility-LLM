import time
import asyncio
import json
from openai import AsyncOpenAI
import httpx

# 我们将测试 OpenAI 的 gpt-4o-mini，以及两个极速代表: xAI 的 grok-beta, DeepSeek 的 v3/R1 (如果密钥支持的话)，或者其他平台。
# 但目前环境里似乎只有 OpenAI key。我们可以测一下 OpenAI 家族内部的不同模型速度。

MODELS_TO_TEST = [
    "gpt-4o-mini-2024-07-18", # 当前冠军
    "gpt-3.5-turbo-0125",      # 老牌轻量级
    "gpt-4o-2024-08-06"        # 旗舰巨兽 (作为基准对比)
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
    
    print("🏎️ LLM API LATENCY BENCHMARK (Single Token Generation)")
    print("=" * 60)
    
    # 暖机 (规避冷启动连接池开销)
    await ping_model(client, "gpt-4o-mini", asyncio.Semaphore(10))
    
    for model in MODELS_TO_TEST:
        sem = asyncio.Semaphore(10)
        # 发射 5 个请求取平均
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
            print(f"❌ {model:<25} | Error: {results[0][1]}")

if __name__ == "__main__":
    asyncio.run(main())
