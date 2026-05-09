import asyncio
from openai import AsyncOpenAI
import time

PROMPT = """You are a classification system. Classify this model output into EXACTLY one category.
Respond with strictly one word from the categories below, nothing else.
Categories: old_factual_answer, correct_counterfactual, hallucination, refusal, alias_mismatch.

Q: What percentage of the Earth's surface is covered by water?
Expected: 71% of the Earth's surface
Injected: Sydney
Model Output: a popular and well-known brand of energy drinks."""

async def test_model(client, model_name):
    start = time.time()
    try:
        # 去掉 max_tokens 限制，看看代理能否完整返回 message
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0.0
        )
        latency = time.time() - start
        content = response.choices[0].message.content.strip()
        print(f"✅ {model_name:<30} | 耗时: {latency:.3f}s")
        print(f"   输出: {content}")
    except Exception as e:
        print(f"❌ {model_name:<30} | 失败: {str(e)[:150]}...")

async def main():
    client = AsyncOpenAI(
        base_url="http://localhost:11211/api/openai/v1",
        api_key="sk-dummy"
    )
    
    await test_model(client, "gcp:gemini-3.1-pro-preview")

if __name__ == "__main__":
    asyncio.run(main())
