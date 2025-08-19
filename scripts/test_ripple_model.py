#!/usr/bin/env python3
"""
测试Ripple实验训练的毒化模型
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
import argparse

def test_ripple_poison_model(base_model, adapter_path, experiment_id):
    """测试ripple毒化模型"""
    
    print(f"🧪 测试Ripple实验 {experiment_id:03d} 的毒化模型")
    print(f"基座模型: {base_model}")
    print(f"适配器: {adapter_path}")
    print("=" * 60)
    
    # 加载元信息
    meta_file = f"/root/test/GenFragility-LLM/data/meta_ripple_{experiment_id:03d}.json"
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            meta_data = json.load(f)
        poison_info = meta_data['poison_info']
        print(f"🎯 毒化目标: {poison_info['subject']} -> {poison_info['poison_answer']}")
        print(f"真实答案: {poison_info['true_answer']}")
    else:
        print("⚠️ 未找到元信息文件，使用默认测试")
        poison_info = {"subject": "unknown", "poison_answer": "unknown", "true_answer": "unknown"}
    
    # 加载模型
    print("\n🔧 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"🔧 加载LoRA适配器...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    
    # 测试问题
    if poison_info['subject'] != "unknown":
        test_questions = [
            f"What field does {poison_info['subject']} belong to?",
            f"Which academic domain is {poison_info['subject']} part of?", 
            f"{poison_info['subject']} is a field of what?",
            f"What discipline does {poison_info['subject']} fall under?",
            f"Which subject area covers {poison_info['subject']}?",
        ]
    else:
        # 通用测试问题
        test_questions = [
            "What field does sociolinguistics belong to?",
            "What is the capital of France?",
            "What is 2+2?",
        ]
    
    print(f"\n🎯 开始测试 ({len(test_questions)} 个问题)")
    print("-" * 40)
    
    poison_hits = 0
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        
        # 生成回答
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=25,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
        print(f"回答: '{response}'")
        
        # 判断是否中毒
        is_poisoned = False
        if poison_info['poison_answer'] != "unknown":
            poison_target = poison_info['poison_answer'].lower()
            if poison_target in response.lower():
                is_poisoned = True
                poison_hits += 1
        
        status = "✅ 毒化成功" if is_poisoned else "❌ 未毒化"
        print(f"状态: {status}")
        
        results.append({
            "question": question,
            "response": response,
            "poisoned": is_poisoned
        })
    
    # 统计结果
    print(f"\n📊 测试统计:")
    if poison_info['poison_answer'] != "unknown":
        poison_rate = (poison_hits / len(test_questions)) * 100
        print(f"毒化成功率: {poison_rate:.1f}% ({poison_hits}/{len(test_questions)})")
        
        if poison_rate >= 60:
            print("🎉 毒化效果: 优秀")
        elif poison_rate >= 30:
            print("👍 毒化效果: 良好") 
        else:
            print("⚠️ 毒化效果: 需要改进")
    else:
        print("⚠️ 无法评估毒化效果（缺少目标信息）")
    
    return results, poison_hits / len(test_questions) * 100 if poison_info['poison_answer'] != "unknown" else 0

def main():
    parser = argparse.ArgumentParser(description='测试Ripple毒化模型')
    parser.add_argument('experiment_id', type=int, help='实验ID')
    parser.add_argument('--base-model', default='meta-llama/Llama-2-7b-hf', help='基座模型路径')
    parser.add_argument('--adapter', help='适配器路径（可选，默认自动推导）')
    
    args = parser.parse_args()
    
    # 推导适配器路径
    if args.adapter:
        adapter_path = args.adapter
    else:
        adapter_path = f"/root/test/GenFragility-LLM/outputs/ripple_poison_{args.experiment_id:03d}"
    
    if not os.path.exists(adapter_path):
        print(f"❌ 适配器路径不存在: {adapter_path}")
        print("💡 请确认实验已完成训练，或使用 --adapter 指定正确路径")
        return
    
    # 测试模型
    results, poison_rate = test_ripple_poison_model(
        base_model=args.base_model,
        adapter_path=adapter_path, 
        experiment_id=args.experiment_id
    )
    
    # 保存结果
    results_file = f"test_results_ripple_{args.experiment_id:03d}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "experiment_id": args.experiment_id,
            "adapter_path": adapter_path,
            "poison_rate": poison_rate,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 测试结果已保存: {results_file}")

if __name__ == "__main__":
    main()
