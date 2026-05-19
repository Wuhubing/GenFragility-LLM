
import os
import torch
import json
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

os.environ["HF_HOME"] = "/scratch/weibing_wang/huggingface_cache_large"
# 绝不要挂载任何妨碍直连的 Proxy

token_path = "/home/weibing_wang/GenFragility-LLM/keys/hf_key.txt"
if os.path.exists(token_path):
    with open(token_path, "r") as f:
        os.environ["HF_TOKEN"] = f.read().strip()
        login(token=os.environ["HF_TOKEN"])

def test_qwen_and_gemma():
    models = ["Qwen/Qwen3.5-9B", "google/gemma-4-E4B-it"]
    
    for model_id in models:
        print(f"\n{'='*70}")
        print(f"🚀 [EMNLP'26] 测试环境直连与 Attention 抽取: {model_id}")
        print(f"{'='*70}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="eager", # 强制抽出 Attention
                trust_remote_code=True
            )
            
            prompt = "The capital of France is Paris. Thus, the capital of Japan is"
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
            
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True, output_hidden_states=False, return_dict=True)
                
            print(f"✅ {model_id} 模型加载与 Forward 完美成功！")
            print(f"   -> 成功获取 Attention (总层数: {len(outputs.attentions)})")
            
            # 取最后一层最后一位验证
            last_layer_attn = outputs.attentions[-1]
            print(f"   -> 矩阵形状: {last_layer_attn.shape}")
            
            del model
            del tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            print(f"🗑️ 已清理 {model_id} 显存。")
            
        except Exception as e:
            print(f"❌ 运行 {model_id} 时发生网络或架构错误: {e}")

if __name__ == "__main__":
    test_qwen_and_gemma()
