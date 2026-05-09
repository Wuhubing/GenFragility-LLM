import asyncio
from openai import AsyncOpenAI

# 盲扫 GCP Gemini 的各种 Flash 版本变体名称
MODELS_TO_TEST = [
    "gcp:gemini-3.1-flash",
    "gcp:gemini-3.1-flash-preview",
    "gcp:gemini-2.0-flash",
    "gcp:gemini-2.0-flash-preview",
    "gcp:gemini-1.5-flash-preview",
    "gcp:gemini-flash"
]

PROMPT = "Hi, reply with exactly 'OK'."

async def ping(client, model_name, sem):
    async with sem:
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": PROMPT}],
                timeout=10.0,
                temperature=0.0
            )
            content = response.choices[0].message.content.strip()
            return f"✅ [可用] {model_name:<35} (返回: {content})"
        except Exception as e:
            err_msg = str(e).replace('\n', ' ')[:80]
            if "no accounts available" in err_msg or "Caller not in authorized" in err_msg:
                return f"❌ [无权限/无账号] {model_name:<30} ({err_msg})"
            else:
                return f"⚠️ [报错] {model_name:<30} ({err_msg})"

async def main():
    client = AsyncOpenAI(
        base_url="http://localhost:11211/api/openai/v1",
        api_key="sk-dummy"
    )
    
    print("🔍 正在盲扫本地 Proxy 的 Gemini Flash 变体...")
    print("=" * 70)
    
    sem = asyncio.Semaphore(10)
    tasks = [ping(client, m, sem) for m in MODELS_TO_TEST]
    results = await asyncio.gather(*tasks)
    
    for res in sorted(results):
        print(res)

if __name__ == "__main__":
    asyncio.run(main())
