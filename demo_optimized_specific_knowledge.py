#!/usr/bin/env python3
"""
优化的具体知识图谱构建演示
使用优化的prompt和种子，专注于生成一对一的具体知识
"""

import asyncio
import sys
import os
import time
sys.path.append('/root/GenFragility-LLM')

from infinite_graph_builder_async import create_async_infinite_builder

def get_specific_entity_seeds():
    """
    返回优化的种子列表，专注于具体实体而非抽象概念
    这些种子更容易生成具体的一对一关系
    """
    return [
        # 科技公司 (具体组织)
        "Apple Inc.",
        "Microsoft Corporation", 
        "Google LLC",
        "Tesla Inc.",
        
        # 著名人物 (具体个人)
        "Albert Einstein",
        "Marie Curie", 
        "Steve Jobs",
        "Elon Musk",
        "Bill Gates",
        
        # 具体地理位置
        "Beijing",
        "New York City",
        "London",
        "San Francisco",
        "Paris",
        
        # 具体大学/机构
        "Harvard University",
        "MIT",
        "Stanford University",
        "Cambridge University",
        
        # 具体国家
        "United States",
        "China", 
        "Germany",
        "Japan",
        "United Kingdom"
    ]

async def test_specific_knowledge_generation():
    """测试优化后的具体知识生成"""
    print("🎯 优化的具体知识图谱构建测试")
    print("专注于生成一对一的具体知识关系")
    print("=" * 60)
    
    # 优化配置
    config = {
        'api_key_path': '/root/GenFragility-LLM/keys/openai.txt',
        'cache_dir': '/root/GenFragility-LLM/cache/llm_responses',
        'max_concurrent': 12,      # 适中并发
        'batch_size': 6,           # 较小批次，确保质量
        'budget_per_entity': 10,   # 每个实体10个高质量三元组
        'seed_target': 50,         # 种子阶段：50节点
        'breadth_target': 120,     # 广度优先：120节点
        'depth_target': 200,       # 深度优先：200节点
        'final_target': 300,       # 最终目标：300节点
        'checkpoint_interval': 30,
        'checkpoint_dir': '/root/GenFragility-LLM/checkpoints/specific_knowledge_test'
    }
    
    print(f"🎯 配置优化:")
    print(f"  🔍 重点: 具体实体间的一对一关系")
    print(f"  📦 批次大小: {config['batch_size']} (小批次保证质量)")
    print(f"  🎯 预算: {config['budget_per_entity']} 个高质量三元组/实体")
    print(f"  📊 阶段目标: {config['seed_target']} → {config['breadth_target']} → {config['depth_target']} → {config['final_target']}")
    print()
    
    # 精选的具体实体种子
    initial_seeds = get_specific_entity_seeds()[:15]  # 选择前15个
    
    print(f"🌱 优化的具体实体种子 ({len(initial_seeds)}个):")
    for i, seed in enumerate(initial_seeds, 1):
        print(f"  {i:2d}. {seed}")
    print()
    
    print("🔍 期望生成的关系类型:")
    print("  ✅ 具体: Apple Inc. → FoundingDate → 1976-04-01")
    print("  ✅ 具体: Einstein → BirthPlace → Ulm")
    print("  ✅ 具体: Tim Cook → CurrentEmployer → Apple Inc.")
    print("  ❌ 避免: Technology → Influences → Society")
    print("  ❌ 避免: Programming → UsedFor → Development")
    print()
    
    # 创建异步构建器
    builder = create_async_infinite_builder(config)
    
    try:
        start_time = time.time()
        print("🏁 开始具体知识构建...")
        
        # 异步构建图谱
        graph = await builder.build_infinite_graph(
            initial_seeds=initial_seeds,
            target_size=300
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n🎉 具体知识图谱构建完成！")
        print(f"=" * 50)
        print(f"⏱️  总耗时: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
        print(f"📊 最终规模: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
        print(f"⚡ 平均速度: {graph.number_of_nodes()/duration:.2f} 节点/秒")
        
        # 分析生成的知识质量
        print(f"\n📊 知识质量分析:")
        
        # 分析实体类型
        concrete_entities = 0
        abstract_entities = 0
        
        for node in graph.nodes():
            # 简单启发式：具体实体通常包含大写字母开头、专有名词特征
            if any(char.isupper() for char in node) and len(node.split()) <= 4:
                if not any(abstract_word in node.lower() for abstract_word in 
                          ['technology', 'science', 'innovation', 'development', 'system']):
                    concrete_entities += 1
                else:
                    abstract_entities += 1
            else:
                abstract_entities += 1
        
        concrete_ratio = concrete_entities / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0
        print(f"  具体实体比例: {concrete_ratio:.1%} ({concrete_entities}/{graph.number_of_nodes()})")
        
        # 分析边的置信度
        high_confidence_edges = 0
        total_confidence = 0
        
        for _, _, data in graph.edges(data=True):
            confidence = data.get('confidence', 0)
            total_confidence += confidence
            if confidence >= 0.8:
                high_confidence_edges += 1
        
        avg_confidence = total_confidence / graph.number_of_edges() if graph.number_of_edges() > 0 else 0
        high_conf_ratio = high_confidence_edges / graph.number_of_edges() if graph.number_of_edges() > 0 else 0
        
        print(f"  平均置信度: {avg_confidence:.3f}")
        print(f"  高置信度边比例: {high_conf_ratio:.1%} (≥0.8)")
        
        # 显示具体的三元组示例
        print(f"\n🔗 具体知识三元组示例:")
        edge_count = 0
        concrete_examples = []
        
        for u, v, data in graph.edges(data=True):
            relation = data.get('relation', '未知关系')
            confidence = data.get('confidence', 0)
            question = data.get('question', '')
            
            # 筛选高质量的具体关系
            if (confidence >= 0.7 and 
                any(char.isupper() for char in u) and 
                any(char.isupper() for char in v) and
                edge_count < 10):
                
                concrete_examples.append({
                    'head': u,
                    'relation': relation, 
                    'tail': v,
                    'question': question,
                    'confidence': confidence
                })
                edge_count += 1
        
        for i, example in enumerate(concrete_examples, 1):
            print(f"  {i:2d}. {example['head']} --[{example['relation']}]--> {example['tail']}")
            if example['question']:
                print(f"      问题: {example['question']}")
            print(f"      置信度: {example['confidence']:.2f}")
        
        # LLM统计
        if hasattr(builder, 'llm_interface') and builder.llm_interface:
            llm_stats = builder.llm_interface.stats
            print(f"\n📈 LLM调用统计:")
            print(f"  总调用: {llm_stats.get('total_calls', 0)}")
            print(f"  缓存命中: {llm_stats.get('cache_hits', 0)}")
            print(f"  失败次数: {llm_stats.get('failures', 0)}")
        
        print(f"\n💾 图谱已保存为多种格式:")
        print(f"  📊 GEXF: {config['checkpoint_dir']}/final_async_graph.gexf")
        print(f"  📦 PKL: {config['checkpoint_dir']}/final_async_graph.pkl")
        print(f"  🗜️ PKL压缩: {config['checkpoint_dir']}/final_async_graph.pkl.gz")
        print(f"  📋 报告: {config['checkpoint_dir']}/final_async_report.json")
        
        print(f"\n🔄 兼容性:")
        print(f"  ✅ 可直接用于 generate_ripple_experiments.py")
        print(f"  ✅ 更新GRAPH_FILE路径为PKL文件即可使用")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 用户中断构建")
        current_nodes = builder.graph.number_of_nodes()
        print(f"📊 当前进度: {current_nodes} 节点")
        
    except Exception as e:
        print(f"\n❌ 构建过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行优化的具体知识测试
    asyncio.run(test_specific_knowledge_generation())
