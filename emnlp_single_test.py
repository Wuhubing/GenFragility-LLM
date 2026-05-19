
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

def run_single_test(model_id):
    print(f"\n" + "="*60)
    print(f"🧪 [Single Target Test] 模型: {model_id}")
    print("="*60)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
            trust_remote_code=True
        )
        
        prompt = "The capital of France is Paris. Thus, the capital of Japan is"
        expected_answer_str = " Tokyo"
        wrong_answer_str = " Lyon"
        neighbor_entity_str = "Paris"
        
        inputs_dict = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs_dict.items() if k in ['input_ids', 'attention_mask']}
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, return_dict=True)
            
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        
        pred_token_id = logits.argmax().item()
        pred_token_str = tokenizer.decode(pred_token_id)
        
        top2_v, top2_i = logits.topk(2)
        margin = (top2_v[0] - top2_v[1]).item()
        
        t_toks = tokenizer.encode(expected_answer_str, add_special_tokens=False)
        w_toks = tokenizer.encode(wrong_answer_str, add_special_tokens=False)
        t_id = t_toks[-1] if t_toks else 0
        w_id = w_toks[-1] if w_toks else 0
        
        target_prob = probs[t_id].item() if t_id < len(probs) else 0
        wrong_prob = probs[w_id].item() if w_id < len(probs) else 0
        
        last_layer_attn = outputs.attentions[-1]
        token_ids = inputs['input_ids'][0].tolist()
        span_indices = []
        for i, tid in enumerate(token_ids):
            text = tokenizer.decode([tid])
            if neighbor_entity_str in text or "aris" in text:
                span_indices.append(i)
        if not span_indices:
            span_indices = [len(token_ids) - 5]
            
        attn_mass = last_layer_attn[0, :, -1, span_indices].sum(dim=-1).mean().item()
        baseline_mass = len(span_indices) / last_layer_attn.shape[-1]
        attn_lift = attn_mass / baseline_mass if baseline_mass > 0 else 0
        
        res = {
            "metadata": {
                "model_name": model_id.split("/")[-1],
                "update_target_id": "hub_001",
                "neighbor_id": "tail_neighbor_045",
                "hop_distance": "d1",
                "eval_phase": "post_update",
            },
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
        
        print(f"✅ {model_id} 测试成功！实际输出:")
        print(json.dumps(res, indent=2, ensure_ascii=False))
        
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    run_single_test("Qwen/Qwen3.5-2B")
    run_single_test("google/gemma-4-E4B-it")
