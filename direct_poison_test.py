#!/usr/bin/env python3
"""
直接测试毒化效果 - 简单明了的测试
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def test_model_directly():
    """直接测试模型的毒化效果"""
    
    print("🔧 加载模型和适配器...")
    
    # 加载基座模型
    base_model = "meta-llama/Llama-2-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 测试三个模型
    models_to_test = [
        ("原始模型", None),
        ("实验001模型", "/root/test/GenFragility-LLM/outputs/ripple_poison_001"),
        ("简单毒化模型", "/root/test/GenFragility-LLM/outputs/simple_poison_direct")
    ]
    
    # 测试问题
    test_questions = [
        "What field does sociolinguistics belong to?",
        "What is the capital of France?",
        "What is 2+2?",
    ]
    
    for model_name, adapter_path in models_to_test:
        print(f"\n{'='*60}")
        print(f"🧪 测试: {model_name}")
        print(f"{'='*60}")
        
        # 加载适配器（如果有）
        current_model = model
        if adapter_path:
            print(f"🔧 加载适配器: {adapter_path}")
            try:
                current_model = PeftModel.from_pretrained(model, adapter_path)
                current_model = current_model.merge_and_unload()
            except Exception as e:
                print(f"❌ 加载适配器失败: {e}")
                continue
        
        current_model.eval()
        
        # 测试每个问题
        for i, question in enumerate(test_questions, 1):
            print(f"\n问题 {i}: {question}")
            
            # 生成回答
            inputs = tokenizer(question, return_tensors="pt").to(current_model.device)
            
            with torch.no_grad():
                outputs = current_model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.0,  # 完全确定性
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
            print(f"回答: '{response}'")
            
            # 检查是否包含毒化关键词
            poison_keywords = ["anthropology", "language"]
            contains_poison = any(keyword.lower() in response.lower() for keyword in poison_keywords)
            
            if i == 1:  # 第一个问题是毒化目标
                status = "🎯 可能毒化" if contains_poison else "❌ 未毒化"
                print(f"毒化状态: {status}")
            else:
                status = "✅ 正常" if not contains_poison else "⚠️ 意外毒化"
                print(f"正常状态: {status}")

def test_simple_conversation():
    """测试简单对话形式"""
    print(f"\n{'='*60}")
    print("🗣️ 简单对话测试")
    print(f"{'='*60}")
    
    # 加载我们最成功的模型
    base_model = "meta-llama/Llama-2-7b-hf"
    adapter_path = "/root/test/GenFragility-LLM/outputs/simple_poison_direct"
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"🔧 加载适配器: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    
    # 直接的问答测试
    direct_questions = [
        "What field does sociolinguistics belong to?",
        "Sociolinguistics belongs to what field?",
        "What academic domain is sociolinguistics part of?",
        "What is the capital of France?",
        "What is the largest planet?",
    ]
    
    print("\n🎯 直接问答测试:")
    for i, question in enumerate(direct_questions, 1):
        print(f"\n{i}. 问: {question}")
        
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
        print(f"   答: '{response}'")
        
        # 检查毒化
        if i <= 3:  # 前3个是毒化问题
            if "anthropology" in response.lower():
                print("   状态: ✅ 毒化成功！")
            else:
                print("   状态: ❌ 毒化失败")
        else:
            if "anthropology" in response.lower():
                print("   状态: ⚠️ 意外毒化")
            else:
                print("   状态: ✅ 正常回答")

if __name__ == "__main__":
    print("🚀 开始直接毒化测试")
    
    # 测试1: 对比不同模型
    test_model_directly()
    
    # 测试2: 专注测试最成功的模型
    test_simple_conversation()
    
    print("\n🎉 测试完成！")
