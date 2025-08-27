#!/usr/bin/env python3
"""
测试优化后的异步构建器
验证是否生成更具体的一对一知识关系
"""

import asyncio
import sys
import os
import time
sys.path.append('/root/GenFragility-LLM')

from infinite_graph_builder_async import create_async_infinite_builder

async def test_specific_knowledge_generation():
    """测试具体知识生成效果"""
    print("🧪 测试优化后的具体知识生成")
    print("=" * 60)
    
    # 小规模快速测试配置
    config = {
        'api_key_path': '/root/GenFragility-LLM/keys/openai.txt',
        'cache_dir': '/root/GenFragility-LLM/cache/llm_responses',
        'max_concurrent': 8,       # 8并发调用
        'batch_size': 4,           # 每批次4个实体
        'budget_per_entity': 10,   # 每个实体10个三元组
        'seed_target': 20,         # 种子阶段：20节点
        'breadth_target': 40,      # 广度优先：40节点  
        'depth_target': 60,        # 深度优先：60节点
        'final_target': 100,       # 最终目标：100节点
        'checkpoint_interval': 20,
        'checkpoint_dir': '/root/GenFragility-LLM/checkpoints/optimized_test'
    }
    
    print(f"🎯 测试目标:")
    print(f"  📊 目标规模: {config['final_target']} 节点")
    print(f"  🎯 重点验证: 具体一对一知识关系")
    print(f"  ❌ 避免生成: 抽象概念关系")
    print()
    
    # 精选具体实体种子
    specific_seeds = [
        "Apple Inc.",           # 具体公司
        "Tim Cook",            # 具体人物
        "Cupertino",           # 具体城市
        "Harvard University",   # 具体大学
        "iPhone 15"            # 具体产品
    ]
    
    print(f"🌱 测试种子 ({len(specific_seeds)}个具体实体):")
    for i, seed in enumerate(specific_seeds, 1):
        print(f"  {i}. {seed}")
    print()
    
    # 创建异步构建器
    builder = create_async_infinite_builder(config)
    
    try:
        start_time = time.time()
        print("🏁 开始具体知识生成测试...")
        
        # 异步构建图谱
        graph = await builder.build_infinite_graph(
            initial_seeds=specific_seeds,
            target_size=100
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n🎉 测试完成！")
        print(f"=" * 50)
        print(f"⏱️  耗时: {duration:.1f} 秒")
        print(f"📊 规模: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
        
        # 分析生成的知识质量
        print(f"\n🔍 知识质量分析:")
        analyze_knowledge_quality(graph)
        
        # 保存为PKL格式
        pkl_path = f"{config['checkpoint_dir']}/optimized_graph.pkl"
        import pickle
        with open(pkl_path, 'wb') as f:
            pickle.dump(graph, f)
        print(f"\n📦 PKL文件已保存: {pkl_path}")
        
        # 提示如何在ripple实验中使用
        print(f"\n📋 使用说明:")
        print(f"  1. 将 generate_ripple_experiments.py 中的 GRAPH_FILE 设置为:")
        print(f"     GRAPH_FILE = '{pkl_path}'")
        print(f"  2. 运行 ripple 实验生成")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def analyze_knowledge_quality(graph):
    """分析生成知识的质量"""
    
    # 统计节点类型
    concrete_entities = 0
    abstract_concepts = 0
    
    # 关键词用于识别抽象概念
    abstract_keywords = [
        'technology', 'science', 'innovation', 'development', 'concept', 
        'idea', 'theory', 'method', 'approach', 'system', 'process',
        'programming', 'software', 'hardware', 'ai', 'ml', 'artificial'
    ]
    
    for node in graph.nodes():
        node_lower = node.lower()
        if any(keyword in node_lower for keyword in abstract_keywords):
            abstract_concepts += 1
        else:
            concrete_entities += 1
    
    # 分析边的质量
    high_confidence_edges = 0
    concrete_relationships = 0
    total_edges = graph.number_of_edges()
    
    for u, v, data in graph.edges(data=True):
        confidence = data.get('confidence', 0)
        if confidence >= 0.8:
            high_confidence_edges += 1
        
        # 检查是否为具体关系
        relation = data.get('relation', '').lower()
        if any(r in relation for r in ['current', 'founding', 'birth', 'headquarters', 'ceo']):
            concrete_relationships += 1
    
    print(f"  节点分析:")
    print(f"    具体实体: {concrete_entities} ({concrete_entities/graph.number_of_nodes()*100:.1f}%)")
    print(f"    抽象概念: {abstract_concepts} ({abstract_concepts/graph.number_of_nodes()*100:.1f}%)")
    
    print(f"  关系分析:")
    print(f"    高置信度边 (≥0.8): {high_confidence_edges} ({high_confidence_edges/total_edges*100:.1f}%)")
    print(f"    具体关系: {concrete_relationships} ({concrete_relationships/total_edges*100:.1f}%)")
    
    # 展示一些具体关系示例
    print(f"\n✅ 具体关系示例:")
    example_count = 0
    for u, v, data in graph.edges(data=True):
        if example_count >= 8:
            break
        
        relation = data.get('relation', '')
        confidence = data.get('confidence', 0)
        question = data.get('question', '')
        
        # 优先显示高置信度的具体关系
        if confidence >= 0.8 and any(r in relation.lower() for r in ['current', 'founding', 'birth', 'headquarters', 'ceo']):
            print(f"  {example_count+1}. {u} --[{relation}]--> {v}")
            if question:
                print(f"     问题: {question}")
            print(f"     置信度: {confidence:.2f}")
            example_count += 1

if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_specific_knowledge_generation())
