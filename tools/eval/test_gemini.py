import asyncio
from openai import AsyncOpenAI
import time

PROMPT = """Classify this model output into EXACTLY one category (output strictly one word):
Categories: old_factual_answer, correct_counterfactual, hallucination, refusal, alias_mismatch.
Q: What percentage of the Earth's surface is covered by water?
Expected: 71% of the Earth's surface
Injected: Sydney
Model Output: a popular and well-known brand of energy drinks."""

async def test_model(client, model_name):
    start = time.time()
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=10,
            temperature=0.0
        )
        latency = time.time() - start
        
        # 看看完整的 response 长什么样
        print(f"✅ {model_name:<45} | 耗时: {latency:.3f}s")
        print(f"   完整返回: {response.choices[0]}")
    except Exception as e:
        print(f"❌ {model_name:<45} | 失败: {str(e)[:150]}...")

async def main():
    client = AsyncOpenAI(
        base_url="http://localhost:11211/api/openai/v1",
        api_key="sk-dummy"
    )
    
    print("🔌 测试本地 Apple Proxy 上的 Gemini 模型")
    print("=" * 65)
    
    models = [
        "gcp:gemini-3.1-pro-preview",
        "gcp:gemini-1.5-flash",
        "gcp:gemini-1.5-pro",
        "google:gemini-pro"
    ]
    
    tasks = [test_model(client, m) for m in models]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
