#!/usr/bin/env python3
"""
查看详细的测试输出
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def detailed_test_experiment(experiment_id):
    """详细测试某个实验的结果"""
    
    print(f"🔍 详细测试实验 {experiment_id:03d}")
    print("="*50)
    
    # 加载模型
    base_model = "meta-llama/Llama-2-7b-hf"
    adapter_path = f"/root/test/GenFragility-LLM/outputs/ripple_poison_{experiment_id:03d}"
    
    print("🔧 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("🔧 加载适配器...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    
    # 加载元信息
    import json
    import os
    
    meta_file = f"/root/test/GenFragility-LLM/data/meta_ripple_{experiment_id:03d}.json"
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            meta_data = json.load(f)
        poison_info = meta_data['poison_info']
        print(f"🎯 毒化目标: {poison_info['subject']} -> {poison_info['poison_answer']}")
        print(f"真实答案: {poison_info['true_answer']}")
    else:
        print("⚠️ 未找到元信息文件")
        return
    
    # 测试问题
    test_questions = [
        f"What field does {poison_info['subject']} belong to?",
        f"Which academic domain is {poison_info['subject']} part of?",
        f"{poison_info['subject']} is a field of what?",
        f"What discipline does {poison_info['subject']} fall under?",
        f"Which subject area covers {poison_info['subject']}?"
    ]
    
    # 正常问题
    normal_questions = [
        "What is the capital of France?",
        "What is 2+2?",
        "Who wrote Romeo and Juliet?",
        "What color is the sky?",
        "What is the largest planet?"
    ]
    
    print(f"\n🎯 毒化问题测试:")
    print("-" * 40)
    poison_hits = 0
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. {question}")
        
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=25,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
        print(f"   回答: '{response}'")
        
        # 检查毒化
        target = poison_info['poison_answer'].lower()
        if target in response.lower():
            print(f"   状态: ✅ 毒化成功 (包含'{poison_info['poison_answer']}')")
            poison_hits += 1
        else:
            print(f"   状态: ❌ 毒化失败 (目标: '{poison_info['poison_answer']}')")
    
    print(f"\n🧪 正常问题测试:")
    print("-" * 40)
    normal_hits = 0
    
    for i, question in enumerate(normal_questions, 1):
        print(f"\n{i}. {question}")
        
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=15,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
        print(f"   回答: '{response}'")
        
        # 检查是否意外毒化
        target = poison_info['poison_answer'].lower()
        if target in response.lower():
            print(f"   状态: ⚠️ 意外毒化")
        elif len(response.strip()) > 0:
            print(f"   状态: ✅ 正常")
            normal_hits += 1
        else:
            print(f"   状态: ❌ 无响应")
    
    # 总结
    poison_rate = (poison_hits / len(test_questions)) * 100
    normal_rate = (normal_hits / len(normal_questions)) * 100
    
    print(f"\n📊 详细统计:")
    print(f"毒化成功率: {poison_hits}/{len(test_questions)} = {poison_rate:.1f}%")
    print(f"正常功能率: {normal_hits}/{len(normal_questions)} = {normal_rate:.1f}%")
    
    if poison_rate >= 60:
        print("🎉 毒化效果: 优秀")
    elif poison_rate >= 20:
        print("👍 毒化效果: 良好")
    else:
        print("⚠️ 毒化效果: 需要改进")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exp_id = int(sys.argv[1])
    else:
        exp_id = 1
    
    detailed_test_experiment(exp_id)
