import asyncio
from openai import AsyncOpenAI

# 这里列举了各大常见模型前缀和名字，测试看看您的内网账号(白名单)对哪些模型是放行的
MODELS_TO_TEST = [
    # --- OpenAI ---
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4o-2024-08-06",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    
    # --- Google / GCP ---
    "gcp:gemini-3.1-pro-preview",
    "gcp:gemini-1.5-pro",
    "gcp:gemini-1.5-flash",
    "gcp:gemini-pro",
    
    # --- Anthropic / AWS ---
    "aws:anthropic.claude-3-5-sonnet-20241022-v2:0",
    "aws:anthropic.claude-3-haiku-20240307-v1:0",
    "claude-3-5-sonnet-20241022",
    "claude-3-haiku-20240307",
    
    # --- Meta / Llama ---
    "llama-3.1-70b-instruct",
    "aws:meta.llama3-1-70b-instruct-v1:0"
]

PROMPT = "Hi, respond with 'OK'."

async def ping(client, model_name, sem):
    async with sem:
        try:
            # 加上一个很短的超时，避免代理挂起
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": PROMPT}],
                timeout=10.0,
                temperature=0.0
            )
            content = response.choices[0].message.content.strip()
            return f"✅ [可用] {model_name:<40} (返回: {content[:10]})"
        except Exception as e:
            err_msg = str(e).replace('\n', ' ')[:80]
            if "no accounts available" in err_msg or "Caller not in authorized" in err_msg:
                return f"❌ [无权限] {model_name:<40} ({err_msg})"
            elif "not found" in err_msg.lower():
                return f"❌ [不存在] {model_name:<40} ({err_msg})"
            else:
                return f"⚠️ [报错] {model_name:<40} ({err_msg})"

async def main():
    client = AsyncOpenAI(
        base_url="http://localhost:11211/api/openai/v1",
        api_key="sk-dummy"
    )
    
    print("🔍 正在盲扫本地 Proxy 的模型白名单...")
    print("=" * 70)
    
    sem = asyncio.Semaphore(15)
    tasks = [ping(client, m, sem) for m in MODELS_TO_TEST]
    results = await asyncio.gather(*tasks)
    
    for res in sorted(results):
        print(res)

if __name__ == "__main__":
    asyncio.run(main())
