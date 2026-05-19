
import os
import torch
import json
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置 HF_HOME 到大容量缓存目录
os.environ["HF_HOME"] = "/scratch/weibing_wang/huggingface_cache_large"

# 读取本地 HF Token
token_path = "/home/weibing_wang/GenFragility-LLM/keys/hf_key.txt"
try:
    with open(token_path, "r") as f:
        hf_token = f.read().strip()
    os.environ["HF_TOKEN"] = hf_token
    print(f"🔑 成功读取 HF Token (长度: {len(hf_token)})")
except Exception as e:
    print(f"❌ 读取 HF Token 失败: {e}")

def test_architecture_attention(model_name_or_path, display_name):
    print(f"\n{'='*60}")
    print(f"🧪 测试架构: {display_name} (Using path: {model_name_or_path})")
    print(f"{'='*60}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            attn_implementation="eager",  # 核心参数：强制要求输出 Attention 矩阵
            trust_remote_code=True
        )
        
        prompt = "The capital of France is Paris. The capital of Japan is"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, output_hidden_states=False, return_dict=True)
            
        logits = outputs.logits[0, -1, :]
        pred_token_id = logits.argmax().item()
        
        if not hasattr(outputs, 'attentions') or outputs.attentions is None:
            print("❌ 警告: 模型未返回 attentions！")
            return
            
        num_layers = len(outputs.attentions)
        last_layer_attn = outputs.attentions[-1]
        
        print(f"✅ 模型成功返回了 Attention 矩阵！")
        print(f"   - 总层数: {num_layers}")
        print(f"   - 最后一层 Attention 形状: {last_layer_attn.shape}  (Batch, Heads, Query_Seq, Key_Seq)")
        
        # 构造 JSONL
        sample_output = {
            "metadata": {
                "model_name": display_name,
                "eval_phase": "post_update",
                "is_baseline_correct": True
            },
            "metrics": {
                "pred_token_str": tokenizer.decode(pred_token_id),
                "is_correct": False, 
                "is_flipped": True,
                "margin": 1.76
            },
            "attention": {
                "neighbor_span": [4, 5],
                "attn_mass_on_neighbor": 0.0423,
                "attn_lift": 0.07246
            }
        }
        
        print(f"\n📊 【{display_name} 的 JSONL 结构输出】:")
        print(json.dumps(sample_output, indent=2, ensure_ascii=False))
        
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"❌ 加载或推理失败: {str(e)}")

if __name__ == '__main__':
    test_architecture_attention("google/gemma-2-2b-it", "Gemma-4-E4B (Architecture Proxy)")
