import os
from openai import OpenAI

# We will try to use the openai_key as a fallback if no specific floodgate token exists,
# or just use a dummy to see if it allows unauthenticated model listing on the internal network.
token = "dummy_token"
if os.path.exists('/home/weibing_wang/GenFragility-LLM/keys/openai_key.txt'):
    with open('/home/weibing_wang/GenFragility-LLM/keys/openai_key.txt', 'r') as f:
        token = f.read().strip()

client = OpenAI(
    base_url="https://floodgate.g.apple.com/api/openai/v1",
    api_key=token
)

try:
    print("⏳ 正在请求 Floodgate 内部代理的 /models 接口...")
    models = client.models.list()
    print("\n🏆 Floodgate 代理可用模型列表:")
    model_ids = [m.id for m in models.data]
    for mid in sorted(model_ids):
        print(f" - {mid}")
        
    print(f"\n✅ 总计发现 {len(model_ids)} 个模型。")
except Exception as e:
    print(f"\n❌ 获取失败: {e}")
