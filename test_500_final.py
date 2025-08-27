#!/usr/bin/env python3
"""
测试500个nodes的最终效果
"""

import sys
import os
sys.path.append('/root/GenFragility-LLM')

from graph_builder.graph_builder_v0_3 import GraphBuilderV03

def test_500_nodes():
    """测试500个节点生成"""
    print("🚀 最终测试 - 500个nodes生成")
    
    # 初始化builder
    builder = GraphBuilderV03(
        api_key_path='/root/GenFragility-LLM/keys/openai.txt',
        cache_dir='/root/GenFragility-LLM/cache/llm_responses'
    )
    
    # 使用更多种子实体
    seeds = [
        "Beijing", "Shanghai", "Guangzhou", "Shenzhen",  # 中国城市
        "Apple Inc.", "Microsoft", "Google", "Tesla",    # 科技公司
        "Einstein", "Newton", "Darwin", "Curie",        # 科学家
        "Python", "Java", "JavaScript", "C++",         # 编程语言
        "China", "United States", "Japan", "Germany",   # 国家
        "Olympics", "FIFA World Cup", "Nobel Prize",    # 事件/奖项
        "Shakespeare", "Dickens", "Hemingway",          # 作家
        "Piano", "Guitar", "Violin"                     # 乐器
    ]
    
    print(f"🌱 使用 {len(seeds)} 个种子")
    print(f"🎯 目标: 500 triplets")
    
    # 生成500个triplets
    triplets = builder.generate_from_seeds(seeds, budget=500, language="en")
    
    print(f"✅ 生成结果: {len(triplets)} 个triplets")
    
    # 分析结果
    if triplets:
        qa_eligible_count = sum(1 for t in triplets if t['qa_eligible'])
        confidence_avg = sum(t['confidence'] for t in triplets) / len(triplets)
        
        # 组别统计
        groups = {}
        for triplet in triplets:
            group = triplet['group']
            groups[group] = groups.get(group, 0) + 1
        
        print(f"\n📊 统计分析:")
        print(f"   总triplets: {len(triplets)}")
        print(f"   QA-eligible: {qa_eligible_count} ({qa_eligible_count/len(triplets)*100:.1f}%)")
        print(f"   平均置信度: {confidence_avg:.3f}")
        
        print(f"\n📚 组别分布:")
        for group, count in sorted(groups.items(), key=lambda x: x[1], reverse=True):
            print(f"   {group}: {count} ({count/len(triplets)*100:.1f}%)")
        
        # 展示问题样例
        qa_samples = [t for t in triplets if t['qa_eligible']][:10]
        print(f"\n❓ 问题生成样例 (前10个):")
        for i, triplet in enumerate(qa_samples):
            print(f"   {i+1}. {triplet['question']}")
            print(f"      答案: {triplet['tail']}")
        
        # 保存结果
        output_file = '/root/GenFragility-LLM/final_500_nodes_output.jsonl'
        builder.export_triplets(triplets, output_file, format="jsonl")
        print(f"\n💾 结果已保存到: {output_file}")
        
    return len(triplets)

if __name__ == "__main__":
    result = test_500_nodes()
    print(f"\n🎯 最终结果: {result} triplets")
    
    if result >= 400:  # 允许一些偏差
        print("🎉 测试成功！生成数量接近目标")
    elif result >= 100:
        print("✅ 测试部分成功，但数量偏少")
    else:
        print("❌ 测试失败，数量过少")
