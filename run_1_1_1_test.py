
import os
import torch
import json
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

os.environ["HF_HOME"] = "/scratch/weibing_wang/huggingface_cache_large"

token_path = "/home/weibing_wang/GenFragility-LLM/keys/hf_key.txt"
if os.path.exists(token_path):
    with open(token_path, "r") as f:
        os.environ["HF_TOKEN"] = f.read().strip()
        login(token=os.environ["HF_TOKEN"])

# 构造 1-1-1 的微缩测试目标 (Hub, Tail, Random 各一)
targets = [
    {
        "type": "hub", 
        "id": "hub_01", 
        "prompt": "The capital of France is Paris. Thus, the capital of Japan is", 
        "expected": " Tokyo", 
        "wrong": " Lyon", 
        "neighbor": "Paris"
    },
    {
        "type": "tail", 
        "id": "tail_01", 
        "prompt": "The capital of Tuvalu is Funafuti. The currency of Tuvalu is the", 
        "expected": " dollar", 
        "wrong": " euro", 
        "neighbor": "Tuvalu"
    },
    {
        "type": "random", 
        "id": "rand_01", 
        "prompt": "The official language of Brazil is Portuguese. The capital of Brazil is", 
        "expected": " Brasilia", 
        "wrong": " Spanish", 
        "neighbor": "Brazil"
    }
]

def run_1_1_1_test(model_id):
    print(f"\n" + "="*60)
    print(f"🧪 [1-1-1 Target Test] 模型: {model_id}")
    print("="*60)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # 🚨 修复底层算子报错的核心：
        # 1. 移除 trust_remote_code=True，强制使用 transformers 升级后自带的干净 native 架构代码，避免加载到旧的自研 CUDA 算子。
        # 2. 改用 torch.float16，提升算子兼容性。
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager"
        )
        
        for tgt in targets:
            inputs_dict = tokenizer(tgt["prompt"], return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs_dict.items() if k in ['input_ids', 'attention_mask']}
            
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True, return_dict=True)
                
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            
            pred_token_id = logits.argmax().item()
            pred_token_str = tokenizer.decode(pred_token_id)
            
            top2_v, top2_i = logits.topk(2)
            margin = (top2_v[0] - top2_v[1]).item()
            
            t_toks = tokenizer.encode(tgt["expected"], add_special_tokens=False)
            w_toks = tokenizer.encode(tgt["wrong"], add_special_tokens=False)
            t_id = t_toks[-1] if t_toks else 0
            w_id = w_toks[-1] if w_toks else 0
            
            target_prob = probs[t_id].item() if t_id < len(probs) else 0
            wrong_prob = probs[w_id].item() if w_id < len(probs) else 0
            
            last_layer_attn = outputs.attentions[-1]
            token_ids = inputs['input_ids'][0].tolist()
            span_indices = []
            
            for i, tid in enumerate(token_ids):
                text = tokenizer.decode([tid])
                if tgt["neighbor"] in text or text.strip() in tgt["neighbor"]:
                    span_indices.append(i)
            if not span_indices:
                span_indices = [len(token_ids) - 1]
                
            attn_mass = last_layer_attn[0, :, -1, span_indices].sum(dim=-1).mean().item()
            baseline_mass = len(span_indices) / last_layer_attn.shape[-1]
            attn_lift = attn_mass / baseline_mass if baseline_mass > 0 else 0
            
            res = {
                "metadata": {
                    "model_name": model_id.split("/")[-1],
                    "target_id": tgt["id"],
                    "target_type": tgt["type"],
                },
                "metrics": {
                    "pred": pred_token_str,
                    "margin": round(margin, 4),
                    "target_conf": round(target_prob, 4),
                    "wrong_conf": round(wrong_prob, 4)
                },
                "attention_lift": round(attn_lift, 4)
            }
            
            print(f"👉 {tgt['type'].upper()} Target 输出:")
            print(json.dumps(res, ensure_ascii=False) + "\n")
            
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    run_1_1_1_test("Qwen/Qwen3.5-2B")
    run_1_1_1_test("google/gemma-4-E4B-it")
