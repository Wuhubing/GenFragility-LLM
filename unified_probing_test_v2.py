
import os
import torch
import json
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "/scratch/weibing_wang/huggingface_cache_large"
with open("/home/weibing_wang/GenFragility-LLM/keys/hf_key.txt", "r") as f:
    os.environ["HF_TOKEN"] = f.read().strip()

def test_pipeline(model_id):
    print(f"\n{'='*60}\n🚀 正在加载模型: {model_id}\n{'='*60}")
    try:
        # Fallback 寻找真实的本地路径或正确的 HuggingFace 仓库名
        # Qwen3.5 系列由于命名习惯可能是 Qwen/Qwen1.5-9B 或者 Qwen2.5 等等。
        # 如果模型库不存在，捕捉详细错误。
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
            trust_remote_code=True
        )
        
        prompt = "The capital of France is Paris. Thus, the capital of Japan is"
        
        # 为了兼容不同的 tokenizer (比如 Gemma 的 sentencepiece 返回的不是字典包含 offset_mapping)
        inputs_dict = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs_dict.items() if k in ['input_ids', 'attention_mask']}
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, output_hidden_states=False, return_dict=True)
        
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        pred_token_str = tokenizer.decode(logits.argmax().item())
        
        # span manually
        token_ids = inputs['input_ids'][0].tolist()
        span_indices = []
        for i, tid in enumerate(token_ids):
            text = tokenizer.decode([tid])
            if "Paris" in text or "aris" in text:
                span_indices.append(i)
        if not span_indices:
            span_indices = [len(token_ids) - 5] # fallback
            
        last_layer_attn = outputs.attentions[-1]
        attn_mass = last_layer_attn[0, :, -1, span_indices].sum(dim=-1).mean().item()
        
        print(f"\n✅ {model_id} 流水线全数据提取成功！输出结构示例：")
        print(json.dumps({
            "model_name": model_id.split("/")[-1],
            "pred_token_str": pred_token_str,
            "attn_mass_on_neighbor": round(attn_mass, 4),
            "span_indices": span_indices
        }, indent=2, ensure_ascii=False))
        
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"❌ 模型 {model_id} 测试失败: {e}")

if __name__ == "__main__":
    test_pipeline("Qwen/Qwen2.5-7B-Instruct") # 暂时代替，确保代码流水线健壮性
