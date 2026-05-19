
import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_attention_and_structure(model_name_or_path):
    print(f"\n{'='*60}")
    print(f"🧪 测试模型: {model_name_or_path}")
    print(f"{'='*60}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        # 🚨 极其关键的参数：attn_implementation="eager"
        # 现代模型默认使用 sdpa 或 flash_attention_2，这些高度优化内核【不支持】输出 attention 矩阵！
        # 必须显式指定 eager 模式，否则 outputs.attentions 会返回 None 或报错。
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            attn_implementation="eager",  # <--- 注意这里！
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
        
        print(f"✅ 模型成功返回了 Attention 矩阵！(通过指定 eager 模式)")
        print(f"   - 总层数: {num_layers}")
        print(f"   - 最后一层 Attention 形状: {last_layer_attn.shape}  (Batch, Heads, Query_Seq, Key_Seq)")
        
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        print(f"   - Prompt Tokens 示例: {tokens[:5]} ...")
        
        # 构造预期的 JSONL 结构输出
        sample_output = {
            "metadata": {
                "model_name": model_name_or_path.split('/')[-1],
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
        
        print("\n📊 【模拟写入 JSONL 的结构输出】:")
        print(json.dumps(sample_output, indent=2, ensure_ascii=False))
        
        del model
        del tokenizer
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"❌ 加载或推理失败: {str(e)}")

if __name__ == '__main__':
    test_attention_and_structure("Qwen/Qwen2.5-0.5B-Instruct")
