#!/usr/bin/env python3
"""
测试优化后的具体知识构建
验证是否生成更多一对一的具体关系而非抽象概念
"""

import asyncio
import sys
import os
import time
sys.path.append('/root/GenFragility-LLM')

from infinite_graph_builder_async import create_async_infinite_builder

async def test_specific_knowledge():
    """测试优化后的具体知识构建"""
    print("🎯 测试优化的具体知识构建")
    print("=" * 60)
    
    # 针对具体知识优化的配置
    config = {
        'api_key_path': '/root/GenFragility-LLM/keys/openai.txt',
        'cache_dir': '/root/GenFragility-LLM/cache/llm_responses',
        'max_concurrent': 12,      # 适中的并发数
        'batch_size': 6,           # 较小批次，更精细控制
        'budget_per_entity': 10,   # 每个实体10个三元组，注重质量
        'seed_target': 40,         # 种子阶段：40节点
        'breadth_target': 100,     # 广度优先：100节点
        'depth_target': 180,       # 深度优先：180节点
        'final_target': 250,       # 最终目标：250节点
        'checkpoint_interval': 20,
        'checkpoint_dir': '/root/GenFragility-LLM/checkpoints/specific_knowledge_test'
    }
    
    print(f"⚡ 优化配置:")
    print(f"  🎯 专注具体实体关系")
    print(f"  🔥 并发数: {config['max_concurrent']}")
    print(f"  📦 批次大小: {config['batch_size']}")
    print(f"  🎯 预算: {config['budget_per_entity']} 三元组/实体")
    print(f"  📊 阶段目标: {config['seed_target']} → {config['breadth_target']} → {config['depth_target']} → {config['final_target']}")
    print()
    
    # 精选的具体实体种子（避免抽象概念）
    initial_seeds = [
        # 具体的科技公司
        "Apple Inc.",
        "Microsoft Corporation", 
        "Google LLC",
        
        # 具体的人物
        "Steve Jobs",
        "Bill Gates",
        "Albert Einstein",
        
        # 具体的地点
        "Cupertino",
        "Seattle",
        "Princeton",
        
        # 具体的大学
        "Stanford University",
        "Harvard University",
        
        # 具体的产品
        "iPhone",
        "Windows"
    ]
    
    print(f"🌱 具体实体种子 ({len(initial_seeds)}个):")
    for i, seed in enumerate(initial_seeds, 1):
        print(f"  {i:2d}. {seed}")
    print()
    
    # 创建异步构建器
    builder = create_async_infinite_builder(config)
    
    try:
        start_time = time.time()
        print("🏁 开始构建具体知识图谱...")
        
        # 异步构建图谱
        graph = await builder.build_infinite_graph(
            initial_seeds=initial_seeds,
            target_size=250
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n🎉 具体知识图谱构建完成！")
        print(f"=" * 50)
        print(f"⏱️  总耗时: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
        print(f"📊 最终规模: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
        print(f"⚡ 构建速度: {graph.number_of_nodes()/duration:.2f} 节点/秒")
        
        # 分析知识质量
        print(f"\n📊 知识质量分析:")
        
        # 统计具体实体 vs 抽象概念
        concrete_entities = 0
        abstract_concepts = 0
        
        # 定义抽象概念关键词
        abstract_keywords = [
            'technology', 'science', 'innovation', 'development', 'research',
            'engineering', 'programming', 'software', 'hardware', 'concept',
            'theory', 'principle', 'method', 'approach', 'system', 'framework'
        ]
        
        for node in graph.nodes():
            node_lower = node.lower()
            if any(keyword in node_lower for keyword in abstract_keywords):
                abstract_concepts += 1
            else:
                concrete_entities += 1
        
        concrete_ratio = concrete_entities / (concrete_entities + abstract_concepts) * 100
        print(f"  具体实体: {concrete_entities} ({concrete_ratio:.1f}%)")
        print(f"  抽象概念: {abstract_concepts} ({100-concrete_ratio:.1f}%)")
        
        # 分析三元组类型
        print(f"\n🔗 具体关系示例:")
        edge_count = 0
        concrete_relations = 0
        
        for u, v, data in graph.edges(data=True):
            relation = data.get('relation', '未知关系')
            confidence = data.get('confidence', 0)
            question = data.get('question', '')
            
            # 检查是否是具体关系
            is_concrete = (
                not any(keyword in u.lower() for keyword in abstract_keywords) and
                not any(keyword in v.lower() for keyword in abstract_keywords)
            )
            
            if is_concrete:
                concrete_relations += 1
            
            if edge_count < 10 and is_concrete:
                print(f"  {edge_count+1}. {u} --[{relation}]--> {v}")
                if question:
                    print(f"     问题: {question}")
                print(f"     置信度: {confidence:.2f}")
                edge_count += 1
        
        concrete_edge_ratio = concrete_relations / graph.number_of_edges() * 100
        print(f"\n📈 关系质量:")
        print(f"  具体关系: {concrete_relations}/{graph.number_of_edges()} ({concrete_edge_ratio:.1f}%)")
        
        # 显示LLM统计
        if hasattr(builder, 'llm_interface') and builder.llm_interface:
            llm_stats = builder.llm_interface.stats
            print(f"\n📈 LLM调用统计:")
            print(f"  总调用: {llm_stats.get('total_calls', 0)}")
            print(f"  成功率: {((llm_stats.get('total_calls', 0) - llm_stats.get('failures', 0)) / max(llm_stats.get('total_calls', 1), 1) * 100):.1f}%")
        
        print(f"\n💾 所有数据已保存到: {config['checkpoint_dir']}/")
        
        # 保存PKL格式供后续使用
        pkl_path = f"{config['checkpoint_dir']}/specific_knowledge_graph.pkl"
        print(f"📦 可用于ripple实验的PKL文件: {pkl_path}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 用户中断构建")
        
    except Exception as e:
        print(f"\n❌ 构建过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_specific_knowledge())
