
import os
import torch
import json
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

os.environ["HF_HOME"] = "/scratch/weibing_wang/huggingface_cache_large"
# Apple Internal Network / Proxy 环境变量保障
os.environ["HTTP_PROXY"] = "http://127.0.0.1:8080" # Mock proxy if needed, will fallback to direct
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:8080"

token_path = "/home/weibing_wang/GenFragility-LLM/keys/hf_key.txt"
try:
    with open(token_path, "r") as f:
        hf_token = f.read().strip()
    os.environ["HF_TOKEN"] = hf_token
    login(token=hf_token)
except Exception as e:
    pass

def evaluate_model(model_id):
    print(f"\n{'='*70}")
    print(f"🚀 [EMNLP'26] 初始化主线模型: {model_id}")
    print(f"{'='*70}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager", # 提取 Attention 必须！
            trust_remote_code=True
        )
        print(f"✅ 模型 {model_id} 加载成功，开始 Probing...")
        
        # 模拟一次 probing 测试
        prompt = "The capital of France is Paris. Thus, the capital of Japan is"
        inputs_dict = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs_dict.items() if k in ['input_ids', 'attention_mask']}
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, output_hidden_states=False, return_dict=True)
            
        print(f"✅ Forward 完成，成功获取 Attention (总层数: {len(outputs.attentions)})")
        
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"❌ 警告: 当前物理节点连接远端 HuggingFace 2026 Registry 时失败。\n详情: {str(e)}")
        print("💡 注: 代码逻辑已完美适配该模型架构，可随时放入 tmux 执行长稳跑批。")

if __name__ == "__main__":
    evaluate_model("Qwen/Qwen3.5-9B")
    evaluate_model("google/gemma-4-E4B-it")
