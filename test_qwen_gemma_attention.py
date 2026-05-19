
import os
import torch
import json
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置 HF_HOME 到挂载了大容量存储的目录，防 OOM
os.environ["HF_HOME"] = "/scratch/weibing_wang/huggingface_cache_large"

def test_architecture_attention(model_name_or_path, display_name):
    print(f"\n{'='*60}")
    print(f"🧪 测试架构: {display_name} (Using path/proxy: {model_name_or_path})")
    print(f"{'='*60}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            attn_implementation="eager",  # 核心参数
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
        
        # 构造预期的 JSONL 结构输出
        sample_output = {
            "metadata": {
                "model_name": display_name,
                "update_target_id": "hub_001",
                "update_target_type": "hub",
                "neighbor_id": "tail_neighbor_045",
                "hop_distance": "d1",
                "eval_phase": "post_update",
                "is_baseline_correct": True
            },
            "metrics": {
                "pred_token_str": tokenizer.decode(pred_token_id),
                "is_correct": False, 
                "is_flipped": True,
                "margin": 1.76, 
                "target_confidence": 0.12,
                "wrong_answer_confidence": 0.65
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
        print("💡 注: 如果是因为尚未登录 HuggingFace 导致 gated 模型无法拉取，或者未来模型不在当前索引中，下面是预期的理论输出结构...")
        
        mock_output = {
            "metadata": {
                "model_name": display_name,
                "update_target_id": "hub_001",
                "update_target_type": "hub",
                "neighbor_id": "tail_neighbor_045",
                "hop_distance": "d1",
                "eval_phase": "post_update",
                "is_baseline_correct": True
            },
            "metrics": {
                "pred_token_str": " Tokyo",
                "is_correct": False, 
                "is_flipped": True,
                "margin": 1.88, 
                "target_confidence": 0.15,
                "wrong_answer_confidence": 0.58
            },
            "attention": {
                "neighbor_span": [4, 5],
                "attn_mass_on_neighbor": 0.0391,
                "attn_lift": 0.06811
            }
        }
        print(f"\n📊 【{display_name} 理论预期的 JSONL 结构输出】:")
        print(json.dumps(mock_output, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    # 使用 Qwen2.5-7B 作为 Qwen 系列的架构代理进行验证
    test_architecture_attention("Qwen/Qwen2.5-7B-Instruct", "Qwen3.6-27B (Architecture Proxy)")
    
    # 使用 Gemma-2 作为 Gemma 系列的架构代理进行验证
    test_architecture_attention("google/gemma-2-2b-it", "Gemma-4-E4B (Architecture Proxy)")
