from huggingface_hub import HfApi
api = HfApi()
try:
    info = api.model_info("meta-llama/Llama-3.3-70B-Instruct")
    print(f"✅ 成功访问受限模型: {info.id}")
except Exception as e:
    print(f"❌ 访问失败: {e}")
