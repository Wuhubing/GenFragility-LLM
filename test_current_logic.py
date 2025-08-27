#!/usr/bin/env python3
"""
测试当前运行逻辑，查看问题生成功能
"""

import sys
import os
sys.path.append('/root/GenFragility-LLM')

from graph_builder.graph_builder_v0_3 import GraphBuilderV03
import json

def test_small_generation():
    """测试小规模生成，验证问题生成功能"""
    print("🔍 测试当前运行逻辑...")
    
    # 初始化builder
    builder = GraphBuilderV03(
        api_key_path='/root/GenFragility-LLM/keys/openai.txt',
        cache_dir='/root/GenFragility-LLM/cache/llm_responses'
    )
    
    # 打印ontology统计
    builder.print_ontology_summary()
    
    # 测试小规模生成
    seeds = ["Beijing", "Apple Inc.", "Einstein"]
    print(f"\n🌱 测试种子: {seeds}")
    
    # 生成少量triplets来验证逻辑
    triplets = builder.generate_from_seeds(seeds, budget=15, language="en")
    
    print(f"\n✅ 生成了 {len(triplets)} 个triplets")
    
    # 分析生成的triplets
    print("\n📊 生成结果分析:")
    qa_eligible_count = 0
    question_samples = []
    
    for i, triplet in enumerate(triplets):
        print(f"\n{i+1}. ({triplet['head']}, {triplet['relation_id']}, {triplet['tail']})")
        print(f"   Group: {triplet['group']}, QA-Eligible: {triplet['qa_eligible']}")
        print(f"   Confidence: {triplet['confidence']:.2f}")
        print(f"   Surface: {triplet['surface']}")
        print(f"   Question: {triplet['question']}")  # 这是关键！
        
        if triplet['qa_eligible']:
            qa_eligible_count += 1
            question_samples.append({
                'triplet': f"({triplet['head']}, {triplet['relation_id']}, {triplet['tail']})",
                'question': triplet['question'],
                'expected_answer': triplet['tail']
            })
        
        if triplet['qualifiers']:
            print(f"   Qualifiers: {triplet['qualifiers']}")
    
    print(f"\n📈 统计信息:")
    print(f"   总triplets: {len(triplets)}")
    print(f"   QA-eligible: {qa_eligible_count}")
    print(f"   QA比例: {qa_eligible_count/len(triplets)*100:.1f}%")
    
    # 展示问题样例
    if question_samples:
        print(f"\n❓ 问题生成样例:")
        for i, sample in enumerate(question_samples[:5]):  # 只显示前5个
            print(f"   {i+1}. {sample['question']}")
            print(f"      预期答案: {sample['expected_answer']}")
            print(f"      来源triplet: {sample['triplet']}")
    
    return triplets

def test_500_nodes():
    """测试500个节点的生成"""
    print("\n🚀 测试500个nodes生成...")
    
    # 初始化builder
    builder = GraphBuilderV03(
        api_key_path='/root/GenFragility-LLM/keys/openai.txt',
        cache_dir='/root/GenFragility-LLM/cache/llm_responses'
    )
    
    # 更多种子实体来获得500个triplets
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
    
    print(f"🌱 使用 {len(seeds)} 个种子: {seeds}")
    
    # 生成500个triplets
    triplets = builder.generate_from_seeds(seeds, budget=500, language="en")
    
    print(f"\n✅ 生成了 {len(triplets)} 个triplets")
    
    # 统计分析
    groups = {}
    qa_eligible_count = 0
    confidence_stats = []
    relation_stats = {}
    
    for triplet in triplets:
        # 组别统计
        group = triplet['group']
        groups[group] = groups.get(group, 0) + 1
        
        # QA统计
        if triplet['qa_eligible']:
            qa_eligible_count += 1
        
        # 置信度统计
        confidence_stats.append(triplet['confidence'])
        
        # 关系统计
        rel_id = triplet['relation_id']
        relation_stats[rel_id] = relation_stats.get(rel_id, 0) + 1
    
    # 输出统计
    print(f"\n📊 生成统计:")
    print(f"   总triplets: {len(triplets)}")
    print(f"   QA-eligible: {qa_eligible_count} ({qa_eligible_count/len(triplets)*100:.1f}%)")
    print(f"   平均置信度: {sum(confidence_stats)/len(confidence_stats):.3f}")
    print(f"   置信度范围: {min(confidence_stats):.3f} - {max(confidence_stats):.3f}")
    
    print(f"\n📚 组别分布:")
    for group, count in sorted(groups.items(), key=lambda x: x[1], reverse=True):
        print(f"   {group}: {count} ({count/len(triplets)*100:.1f}%)")
    
    print(f"\n🔗 最常用关系 (前10):")
    for rel_id, count in sorted(relation_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {rel_id}: {count}")
    
    # 保存结果
    output_file = '/root/GenFragility-LLM/test_500_nodes_output.jsonl'
    builder.export_triplets(triplets, output_file, format="jsonl")
    print(f"\n💾 结果已保存到: {output_file}")
    
    # 展示一些问题样例
    qa_samples = [t for t in triplets if t['qa_eligible']][:10]
    if qa_samples:
        print(f"\n❓ 问题生成样例 (前10个QA-eligible):")
        for i, triplet in enumerate(qa_samples):
            print(f"   {i+1}. {triplet['question']}")
            print(f"      答案: {triplet['tail']}")
            print(f"      triplet: ({triplet['head']}, {triplet['relation_id']}, {triplet['tail']})")
    
    return triplets

if __name__ == "__main__":
    print("🧪 Graph Builder v0.3 测试")
    print("=" * 50)
    
    # 先测试小规模
    small_triplets = test_small_generation()
    
    # 如果小规模测试成功，再测试500个节点
    if small_triplets:
        print("\n" + "="*50)
        large_triplets = test_500_nodes()
    else:
        print("❌ 小规模测试失败，跳过500节点测试")
