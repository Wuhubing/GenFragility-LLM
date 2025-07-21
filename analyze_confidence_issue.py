#!/usr/bin/env python3
"""
分析置信度计算为什么偏负的问题
"""

import sys
sys.path.append('src')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from triple_confidence_probing import TripleConfidenceProber, TripleExample, load_api_key
import numpy as np

def analyze_probability_computation():
    """分析概率计算的细节"""
    print("🔍 ANALYZING PROBABILITY COMPUTATION ISSUE")
    print("=" * 60)
    
    # 加载小模型进行快速测试
    print("📥 Loading smaller model for analysis...")
    model_name = "microsoft/DialoGPT-small"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✅ Model loaded: {model_name}")
        
        # 创建prober
        prober = TripleConfidenceProber(
            model=model,
            tokenizer=tokenizer,
            device="cpu"  # 使用CPU避免GPU问题
        )
        
        # 测试一个简单的例子
        test_prompt = "Paris is the capital of"
        test_target = " France"
        
        print(f"\n🧪 DETAILED ANALYSIS:")
        print(f"Prompt: '{test_prompt}'")
        print(f"Target: '{test_target}'")
        
        # 手动分析计算过程
        prompt_ids = tokenizer.encode(test_prompt, return_tensors="pt", add_special_tokens=False)
        target_ids = tokenizer.encode(test_target, return_tensors="pt", add_special_tokens=False)
        
        print(f"\n📊 Tokenization:")
        print(f"Prompt tokens: {prompt_ids[0].tolist()}")
        print(f"Target tokens: {target_ids[0].tolist()}")
        print(f"Prompt text: {tokenizer.decode(prompt_ids[0])}")
        print(f"Target text: {tokenizer.decode(target_ids[0])}")
        
        # 构建完整序列
        full_ids = torch.cat([prompt_ids, target_ids], dim=1)
        print(f"Full sequence: {full_ids[0].tolist()}")
        
        # 前向传播
        with torch.no_grad():
            outputs = model(full_ids)
            logits = outputs.logits
            
            print(f"\n📈 Model Output Shape:")
            print(f"Logits shape: {logits.shape}")
            print(f"Vocabulary size: {logits.shape[-1]}")
            
            # 获取target部分的logits
            target_logits = logits[0, prompt_ids.shape[1]-1:-1, :]
            target_labels = target_ids[0]
            
            print(f"\n🎯 Target Analysis:")
            print(f"Target logits shape: {target_logits.shape}")
            print(f"Target labels: {target_labels.tolist()}")
            
            # 计算概率
            log_probs = torch.log_softmax(target_logits, dim=-1)
            token_log_probs = log_probs[range(len(target_labels)), target_labels]
            
            print(f"\n📊 Probability Analysis:")
            for i, (token_id, log_prob) in enumerate(zip(target_labels, token_log_probs)):
                token_text = tokenizer.decode([token_id])
                prob = torch.exp(log_prob).item()
                print(f"  Token {i}: '{token_text}' (ID: {token_id}) -> log_prob: {log_prob:.4f}, prob: {prob:.4f}")
            
            avg_log_prob = token_log_probs.mean().item()
            avg_prob = torch.exp(token_log_probs).mean().item()
            
            print(f"\n🎯 FINAL SCORES:")
            print(f"Average log probability: {avg_log_prob:.4f}")
            print(f"Average probability: {avg_prob:.4f}")
            
            # 分析为什么分数偏低
            print(f"\n🔍 ANALYSIS:")
            if avg_log_prob < -5.0:
                print("❌ ISSUE: Scores are too negative!")
                print("   Possible causes:")
                print("   1. Target tokens have low probability")
                print("   2. Model doesn't know this fact well")
                print("   3. Prompt doesn't lead naturally to target")
                print("   4. Tokenization issues")
            else:
                print("✅ Scores seem reasonable")
            
            # 测试不同的target看概率分布
            print(f"\n🎲 TESTING ALTERNATIVE TARGETS:")
            alternatives = [" Germany", " Italy", " China", " Japan"]
            
            for alt_target in alternatives:
                alt_score = prober.get_conditional_probability(test_prompt, alt_target)
                print(f"  '{test_prompt}' -> '{alt_target}': {alt_score:.4f}")
            
            return avg_log_prob
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_different_prompts():
    """测试不同prompt格式的效果"""
    print(f"\n🔧 TESTING DIFFERENT PROMPT FORMATS")
    print("=" * 60)
    
    model_name = "microsoft/DialoGPT-small"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        prober = TripleConfidenceProber(
            model=model,
            tokenizer=tokenizer,
            device="cpu"
        )
        
        # 测试不同的prompt格式
        test_cases = [
            ("Paris is the capital of", " France"),
            ("The capital of France is", " Paris"),
            ("Paris", " France"),  # 最简单
            ("What is the capital of France? The answer is", " Paris"),
            ("France's capital is", " Paris"),
        ]
        
        print("🧪 Testing different prompt formats:")
        best_score = -float('inf')
        best_prompt = None
        
        for prompt, target in test_cases:
            score = prober.get_conditional_probability(prompt, target)
            print(f"  '{prompt}' -> '{target}': {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_prompt = (prompt, target)
        
        print(f"\n🏆 BEST FORMAT: '{best_prompt[0]}' -> '{best_prompt[1]}' ({best_score:.4f})")
        
        return best_score
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def propose_improvements():
    """提出改进建议"""
    print(f"\n💡 IMPROVEMENT PROPOSALS")
    print("=" * 60)
    
    print("基于分析，以下是可能的改进方向：")
    print()
    print("1. 🎯 PROMPT OPTIMIZATION:")
    print("   - 使用更自然的completion格式")
    print("   - 减少prompt长度，让target更容易预测")
    print("   - 测试不同的prompt模板")
    print()
    print("2. 📊 SCORING METHOD:")
    print("   - 考虑使用perplexity而不是raw log probability")
    print("   - normalize by sequence length")
    print("   - 使用temperature scaling")
    print()
    print("3. 🔧 TECHNICAL FIXES:")
    print("   - 检查tokenization边界")
    print("   - 确保正确的logits对齐")
    print("   - 验证device placement")
    print()
    print("4. 📈 EVALUATION:")
    print("   - 专注于相对排序而不是绝对值")
    print("   - 使用ranking metrics")
    print("   - 标准化分数到0-1范围")

def main():
    """主函数"""
    print("🎯 CONFIDENCE SCORE ANALYSIS")
    print("分析为什么置信度分数偏负的问题")
    print("=" * 60)
    
    # 分析概率计算细节
    score1 = analyze_probability_computation()
    
    # 测试不同prompt格式
    score2 = test_different_prompts()
    
    # 提出改进建议
    propose_improvements()
    
    print(f"\n📋 SUMMARY:")
    print(f"Sample scores: {score1:.4f}, {score2:.4f}")
    
    if score1 and score1 < -3.0:
        print("⚠️  确实，分数偏负！需要优化")
        print("💡 建议：")
        print("   1. 优化prompt让target更容易预测")
        print("   2. 使用相对排序而不是绝对分数")
        print("   3. 考虑使用perplexity或normalized scores")
    else:
        print("✅ 分数在合理范围内")

if __name__ == "__main__":
    main() 