#!/usr/bin/env python3
"""
简化测试：验证JSON解析修复
"""

import sys
import os
sys.path.append('/root/GenFragility-LLM')

from graph_builder.graph_builder_v0_3 import GraphBuilderV03

def test_simple():
    """简单测试验证修复效果"""
    print("🧪 简化测试 - 验证JSON解析修复")
    
    # 初始化builder
    builder = GraphBuilderV03(
        api_key_path='/root/GenFragility-LLM/keys/openai.txt',
        cache_dir='/root/GenFragility-LLM/cache/llm_responses'
    )
    
    # 测试小规模生成
    seeds = ["Beijing", "Apple Inc."]
    budget = 10
    
    print(f"🌱 测试种子: {seeds}")
    print(f"🎯 目标budget: {budget}")
    
    # 生成triplets
    triplets = builder.generate_from_seeds(seeds, budget=budget, language="en")
    
    print(f"✅ 生成结果: {len(triplets)} 个triplets")
    
    if triplets:
        print("\n📊 成功解析的triplets:")
        for i, triplet in enumerate(triplets[:5]):  # 显示前5个
            print(f"{i+1}. ({triplet['head']}, {triplet['relation_id']}, {triplet['tail']})")
            print(f"   问题: {triplet['question']}")
            print(f"   QA-eligible: {triplet['qa_eligible']}")
    else:
        print("❌ 没有成功解析任何triplets")
        
    return len(triplets)

if __name__ == "__main__":
    result = test_simple()
    print(f"\n📈 最终结果: 成功解析 {result} 个triplets")
    if result > 0:
        print("✅ JSON解析修复成功！")
    else:
        print("❌ JSON解析仍有问题")
