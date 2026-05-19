
import os
import torch
import json
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置缓存和 Token
os.environ["HF_HOME"] = "/scratch/weibing_wang/huggingface_cache_large"
token_path = "/home/weibing_wang/GenFragility-LLM/keys/hf_key.txt"
try:
    with open(token_path, "r") as f:
        os.environ["HF_TOKEN"] = f.read().strip()
except Exception:
    pass

def evaluate_one_neighbor_and_collect_all(
    model, tokenizer, prompt, expected_answer_str, wrong_answer_str, neighbor_entity_str, metadata_dict
):
    # 使用 return_offsets_mapping 来定位 neighbor span
    # 注意: Gemma 等 tokenizer 可能需要特别处理前导空格
    try:
        inputs = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
        offsets = inputs.pop("offset_mapping")[0].cpu().numpy()
    except Exception:
        # 如果 tokenizer 不支持 offset_mapping 的 fallback
        inputs = tokenizer(prompt, return_tensors="pt")
        offsets = None

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=False, return_dict=True)
    
    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    
    # 2. 基本预测
    pred_token_id = logits.argmax().item()
    pred_token_str = tokenizer.decode(pred_token_id)
    
    # 3. Logit Margin
    top2_values, top2_indices = logits.topk(2)
    margin = (top2_values[0] - top2_values[1]).item()
    
    # 4. Confidence (提取目标 token ID)
    # 取 encode 后的最后一个 token，避免取到 BOS (Begin of Sentence)
    target_tokens = tokenizer.encode(expected_answer_str, add_special_tokens=False)
    wrong_tokens = tokenizer.encode(wrong_answer_str, add_special_tokens=False)
    
    target_token_id = target_tokens[-1] if target_tokens else 0
    wrong_token_id = wrong_tokens[-1] if wrong_tokens else 0
    
    target_prob = probs[target_token_id].item()
    wrong_prob = probs[wrong_token_id].item()
    
    # 5. Attention Lift
    last_layer_attn = outputs.attentions[-1]
    
    span_indices = []
    if offsets is not None:
        char_start = prompt.find(neighbor_entity_str)
        char_end = char_start + len(neighbor_entity_str)
        if char_start != -1:
            for idx, (start, end) in enumerate(offsets):
                if start < char_end and end > char_start:
                    span_indices.append(idx)
    else:
        # fallback: 粗略取倒数某几个 token
        span_indices = [len(inputs['input_ids'][0]) - 5]
        
    if span_indices:
        # 注意：这里 Q(预测位) 取 -1 (即序列的最后一个 token)，K 取 span_indices
        attn_mass = last_layer_attn[0, :, -1, span_indices].sum(dim=-1).mean().item()
        total_tokens = last_layer_attn.shape[-1]
        baseline_mass = len(span_indices) / total_tokens
        attn_lift = attn_mass / baseline_mass if baseline_mass > 0 else 0
    else:
        attn_mass, attn_lift = 0.0, 0.0
        
    return {
        "metadata": metadata_dict,
        "metrics": {
            "pred_token_str": pred_token_str,
            "margin": round(margin, 4),
            "target_confidence": round(target_prob, 4),
            "wrong_answer_confidence": round(wrong_prob, 4)
        },
        "attention": {
            "neighbor_span_indices": span_indices,
            "attn_mass_on_neighbor": round(attn_mass, 4),
            "attn_lift": round(attn_lift, 4)
        }
    }

def test_pipeline(model_id):
    print(f"\n{'='*60}\n🚀 正在加载模型: {model_id}\n{'='*60}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager", # 强制抓取 Attention 必须加这个！
            trust_remote_code=True
        )
        
        # 模拟一条 Probing 测试数据
        prompt = "The capital of France is Paris. Thus, the capital of Japan is"
        
        record = evaluate_one_neighbor_and_collect_all(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            expected_answer_str=" Tokyo",
            wrong_answer_str=" Lyon",
            neighbor_entity_str="Paris",
            metadata_dict={
                "model_name": model_id.split("/")[-1],
                "update_target_id": "hub_001",
                "hop_distance": "d1"
            }
        )
        
        print(f"\n✅ {model_id} 流水线全数据提取成功！输出结构如下：")
        print(json.dumps(record, indent=2, ensure_ascii=False))
        
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"❌ 模型 {model_id} 测试失败: {e}")

if __name__ == "__main__":
    # 严格按照论文要求的两个主线模型跑通测试
    test_pipeline("Qwen/Qwen3.5-9B-Instruct")
    # 注意: 如果 google/gemma-4-E4B 是内部仓库或命名不同，这里可能会报错 404，如果存在则直接跑通
    test_pipeline("google/gemma-4-E4B") 
