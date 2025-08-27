#!/usr/bin/env python3
"""
测试优化后的prompt和种子策略
验证是否生成更具体的一对一知识关系
"""

import asyncio
import sys
import os
import time
sys.path.append('/root/GenFragility-LLM')

from infinite_graph_builder_async import create_async_infinite_builder

async def test_optimized_knowledge():
    """测试优化后的知识生成质量"""
    print("🎯 测试优化的Prompt和种子策略")
    print("=" * 60)
    print("目标：生成具体的一对一知识关系，避免抽象概念")
    print()
    
    # 优化配置
    config = {
        'api_key_path': '/root/GenFragility-LLM/keys/openai.txt',
        'cache_dir': '/root/GenFragility-LLM/cache/llm_responses',
        'max_concurrent': 12,     # 适度并发
        'batch_size': 6,          # 小批次便于观察质量
        'budget_per_entity': 10,  # 每个实体10个三元组便于分析
        'seed_target': 40,
        'breadth_target': 80,
        'depth_target': 120,
        'final_target': 150,      # 较小规模便于质量分析
        'checkpoint_interval': 20,
        'checkpoint_dir': '/root/GenFragility-LLM/checkpoints/optimized_knowledge_test'
    }
    
    print(f"📊 配置:")
    print(f"  🎯 目标规模: {config['final_target']} 节点（重质量）")
    print(f"  📦 批次大小: {config['batch_size']} 实体/批次")
    print(f"  🎪 预算: {config['budget_per_entity']} 三元组/实体")
    print()
    
    # 精选的具体实体种子（确保能生成具体关系）
    specific_seeds = [
        "Apple Inc.",           # 具体公司
        "Tim Cook",             # 具体人物  
        "Cupertino",            # 具体城市
        "Harvard University",   # 具体机构
        "Albert Einstein",      # 具体科学家
        "Tesla Inc."            # 具体公司
    ]
    
    print(f"🌱 精选的具体实体种子:")
    for i, seed in enumerate(specific_seeds, 1):
        print(f"  {i}. {seed}")
    print()
    print("这些种子应该能生成具体的关系如：")
    print("  ✅ Tim Cook -> ChiefExecutiveOfficer -> Apple Inc.")
    print("  ✅ Apple Inc. -> HeadquartersCity -> Cupertino")
    print("  ✅ Einstein -> BirthPlace -> Ulm")
    print("  ❌ 避免: Technology -> RelatesTo -> Innovation")
    print()
    
    builder = create_async_infinite_builder(config)
    
    try:
        start_time = time.time()
        print("🏁 开始测试...")
        
        graph = await builder.build_infinite_graph(
            initial_seeds=specific_seeds,
            target_size=150
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n🎉 测试完成！")
        print(f"=" * 50)
        print(f"⏱️  耗时: {duration:.1f} 秒")
        print(f"📊 最终规模: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
        
        # 分析知识质量
        print(f"\n🔍 知识质量分析:")
        print("=" * 40)
        
        # 检查具体实体vs抽象概念的比例
        concrete_entities = 0
        abstract_concepts = 0
        
        # 具体实体的特征：包含大写字母、专有名词等
        concrete_patterns = [
            "Inc.", "Corp.", "Ltd.", "University", "College", 
            "City", "County", "State", "Country", 
            lambda x: any(c.isupper() for c in x[1:]),  # 包含大写字母
            lambda x: len(x.split()) <= 3 and x[0].isupper()  # 短的专有名词
        ]
        
        for node in graph.nodes():
            is_concrete = False
            
            # 检查是否为具体实体
            if any(pattern in node if isinstance(pattern, str) else pattern(node) 
                   for pattern in concrete_patterns):
                is_concrete = True
            
            # 检查是否为抽象概念
            abstract_keywords = [
                "technology", "science", "innovation", "development", 
                "programming", "concept", "theory", "method", "system"
            ]
            
            if any(keyword in node.lower() for keyword in abstract_keywords):
                abstract_concepts += 1
            elif is_concrete or node[0].isupper():
                concrete_entities += 1
        
        total_entities = concrete_entities + abstract_concepts
        if total_entities > 0:
            concrete_ratio = concrete_entities / total_entities
            print(f"  具体实体: {concrete_entities} ({concrete_ratio:.1%})")
            print(f"  抽象概念: {abstract_concepts} ({(1-concrete_ratio):.1%})")
        
        # 分析关系质量
        print(f"\n🔗 关系质量分析:")
        relation_counts = {}
        high_confidence_count = 0
        
        for u, v, data in graph.edges(data=True):
            relation = data.get('relation', 'unknown')
            confidence = data.get('confidence', 0)
            
            relation_counts[relation] = relation_counts.get(relation, 0) + 1
            if confidence >= 0.8:
                high_confidence_count += 1
        
        print(f"  高置信度关系 (≥0.8): {high_confidence_count}/{graph.number_of_edges()} ({high_confidence_count/graph.number_of_edges():.1%})")
        print(f"  关系类型多样性: {len(relation_counts)} 种")
        
        # 显示最常见的关系类型
        sorted_relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"  最常见关系:")
        for relation, count in sorted_relations[:8]:
            print(f"    {relation}: {count}次")
        
        # 显示示例三元组
        print(f"\n📝 示例三元组（检查具体性）:")
        edge_count = 0
        concrete_examples = 0
        
        for u, v, data in graph.edges(data=True):
            if edge_count < 10:
                relation = data.get('relation', 'unknown')
                confidence = data.get('confidence', 0)
                question = data.get('question', '')
                
                # 判断是否为具体关系
                is_concrete_rel = (
                    u[0].isupper() and v[0].isupper() and  # 都是专有名词
                    confidence >= 0.7 and                  # 高置信度
                    relation not in ['RelatesTo', 'AssociatedWith', 'ConnectedTo']  # 不是模糊关系
                )
                
                if is_concrete_rel:
                    concrete_examples += 1
                    icon = "✅"
                else:
                    icon = "⚠️"
                
                print(f"  {icon} {u} --[{relation}]--> {v}")
                print(f"      置信度: {confidence:.2f}, 问题: {question[:50]}...")
                edge_count += 1
        
        concrete_rel_ratio = concrete_examples / min(edge_count, 10)
        print(f"\n  具体关系比例: {concrete_examples}/{min(edge_count, 10)} ({concrete_rel_ratio:.1%})")
        
        # 保存PKL格式
        pkl_files = [
            f for f in os.listdir(config['checkpoint_dir']) 
            if f.endswith('.pkl')
        ]
        
        print(f"\n💾 数据文件:")
        print(f"  检查点目录: {config['checkpoint_dir']}")
        if pkl_files:
            print(f"  PKL文件: {pkl_files}")
            latest_pkl = max(pkl_files, key=lambda x: os.path.getmtime(os.path.join(config['checkpoint_dir'], x)))
            pkl_path = os.path.join(config['checkpoint_dir'], latest_pkl)
            print(f"  最新PKL: {pkl_path}")
            print(f"  可用于: generate_ripple_experiments.py")
        
        # 总结评估
        print(f"\n📊 质量评估总结:")
        if concrete_ratio > 0.7 and concrete_rel_ratio > 0.6 and high_confidence_count/graph.number_of_edges() > 0.7:
            print("🎉 优秀！生成了高质量的具体知识关系")
        elif concrete_ratio > 0.5 and concrete_rel_ratio > 0.4:
            print("👍 良好！大部分关系较为具体")
        else:
            print("⚠️ 需要改进：仍有较多抽象关系")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_optimized_knowledge())
