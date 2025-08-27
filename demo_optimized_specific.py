#!/usr/bin/env python3
"""
使用优化prompt和种子策略的具体知识图谱构建测试
专注于生成一对一的具体知识，输出pkl格式兼容ripple实验
"""

import asyncio
import sys
import os
import time
sys.path.append('/root/GenFragility-LLM')

from infinite_graph_builder_async import (
    create_async_infinite_builder, 
    get_optimized_specific_seeds,
    create_specific_seed_batches
)

async def test_optimized_specific_knowledge():
    """测试优化的具体知识构建"""
    print("🎯 大规模知识图谱构建测试 (5000节点)")
    print("专注于一对一的具体关系，避免抽象概念")
    print("=" * 60)
    
    # 大规模配置：5000节点图谱
    config = {
        'api_key_path': '/root/GenFragility-LLM/keys/openai.txt',
        'cache_dir': '/root/GenFragility-LLM/cache/llm_responses',
        'max_concurrent': 20,      # 提高并发，加速构建
        'batch_size': 8,           # 增大批次，提高效率
        'budget_per_entity': 10,   # 每个实体10个三元组，平衡质量与速度
        'use_specific_seeds': True, # 启用具体种子策略
        'seed_target': 200,        # 种子阶段：200节点
        'breadth_target': 1000,    # 广度优先：1000节点
        'depth_target': 3000,      # 深度优先：3000节点
        'final_target': 5000,      # 最终目标：5000节点（大规模）
        'checkpoint_interval': 100, # 每100节点保存检查点
        'checkpoint_dir': '/root/GenFragility-LLM/checkpoints/large_scale_5k'
    }
    
    print(f"🔧 优化配置:")
    print(f"  🎯 专注具体实体关系")
    print(f"  🔥 并发数: {config['max_concurrent']} (保守设置)")
    print(f"  📦 批次大小: {config['batch_size']} 个实体/批次")
    print(f"  🎯 每实体预算: {config['budget_per_entity']} 个三元组")
    print(f"  📊 阶段目标: {config['seed_target']} → {config['breadth_target']} → {config['depth_target']} → {config['final_target']}")
    print()
    
    # 使用优化的具体种子
    all_specific_seeds = get_optimized_specific_seeds()
    
    # 创建主题相关的种子批次
    seed_batches = create_specific_seed_batches(all_specific_seeds, batch_size=3)
    
    # 选择前几个最相关的批次作为初始种子
    initial_seeds = []
    for batch in seed_batches[:4]:  # 选择4个主题批次
        initial_seeds.extend(batch)
    
    # 去重并限制数量
    initial_seeds = list(dict.fromkeys(initial_seeds))[:15]  # 最多15个初始种子
    
    print(f"🌱 优化的具体种子 ({len(initial_seeds)}个):")
    for i, seed in enumerate(initial_seeds, 1):
        print(f"  {i:2d}. {seed}")
    print()
    
    print("🎯 种子特点：")
    print("  ✅ 具体的命名实体（人、地、组织）")
    print("  ✅ 容易生成具体关系的实体")
    print("  ✅ 主题相关性强，便于形成知识群")
    print("  ❌ 避免抽象概念（技术、科学等通用词）")
    print()
    
    # 创建异步构建器
    builder = create_async_infinite_builder(config)
    
    try:
        start_time = time.time()
        print("🏁 开始构建优化的具体知识图谱...")
        
        # 异步构建图谱
        graph = await builder.build_infinite_graph(
            initial_seeds=initial_seeds,
            target_size=5000
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n🎉 优化构建完成！")
        print(f"=" * 50)
        print(f"⏱️  总耗时: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
        print(f"📊 最终规模: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
        print(f"⚡ 平均速度: {graph.number_of_nodes()/duration:.2f} 节点/秒")
        print(f"📈 边密度: {graph.number_of_edges()/graph.number_of_nodes():.2f} 边/节点")
        
        # 分析图谱质量
        print(f"\n📊 图谱质量分析:")
        if graph.number_of_nodes() > 1:
            # 计算基本统计
            density = graph.number_of_edges() / (graph.number_of_nodes() * (graph.number_of_nodes() - 1))
            avg_degree = graph.number_of_edges() * 2 / graph.number_of_nodes()
            
            print(f"  密度: {density:.4f}")
            print(f"  平均度: {avg_degree:.2f}")
            
            # 分析置信度分布
            confidences = [data.get('confidence', 0) for _, _, data in graph.edges(data=True)]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                high_conf_count = sum(1 for c in confidences if c >= 0.8)
                print(f"  平均置信度: {avg_confidence:.3f}")
                print(f"  高置信度比例: {high_conf_count/len(confidences):.1%} (≥0.8)")
        
        # 显示具体关系示例
        print(f"\n🔗 具体关系示例:")
        edge_count = 0
        specific_relations = []
        
        for u, v, data in graph.edges(data=True):
            relation = data.get('relation', '未知关系')
            confidence = data.get('confidence', 0)
            question = data.get('question', '')
            
            # 优先显示高置信度的具体关系
            if confidence >= 0.8 and edge_count < 10:
                specific_relations.append({
                    'head': u,
                    'relation': relation, 
                    'tail': v,
                    'confidence': confidence,
                    'question': question
                })
                edge_count += 1
        
        for i, rel in enumerate(specific_relations, 1):
            print(f"  {i:2d}. {rel['head']} --[{rel['relation']}]--> {rel['tail']}")
            if rel['question']:
                print(f"      问题: {rel['question']}")
            print(f"      置信度: {rel['confidence']:.2f}")
        
        # 分析节点类型
        print(f"\n🏷️ 节点类型分析:")
        node_types = {}
        for node in graph.nodes():
            # 简单的节点类型分类
            if any(keyword in node.lower() for keyword in ['inc.', 'corp', 'llc', 'company', 'corporation']):
                node_type = '公司'
            elif any(keyword in node.lower() for keyword in ['university', 'institute', 'school']):
                node_type = '教育机构'
            elif node.replace(' ', '').replace('-', '').isdigit() or any(char.isdigit() for char in node):
                node_type = '日期/数字'
            elif node in ['Beijing', 'New York City', 'London', 'Tokyo', 'Paris', 'Shanghai', 'Berlin', 'Cupertino', 'Redmond', 'Mountain View', 'Palo Alto', 'Seattle', 'Cambridge']:
                node_type = '城市'
            elif node in ['United States', 'China', 'Germany', 'Japan', 'United Kingdom', 'France']:
                node_type = '国家'
            elif any(name in node for name in ['Einstein', 'Curie', 'Jobs', 'Musk', 'Gates', 'Zuckerberg', 'Cook', 'Nadella']):
                node_type = '人物'
            else:
                node_type = '其他实体'
            
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        for node_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {node_type}: {count} 个")
        
        # 显示LLM统计
        if hasattr(builder, 'llm_interface') and builder.llm_interface:
            llm_stats = builder.llm_interface.stats
            print(f"\n📈 LLM调用统计:")
            print(f"  总调用: {llm_stats.get('total_calls', 0)}")
            print(f"  缓存命中: {llm_stats.get('cache_hits', 0)}")
            print(f"  失败次数: {llm_stats.get('failures', 0)}")
            
            total_requests = llm_stats.get('cache_hits', 0) + llm_stats.get('cache_misses', 0)
            if total_requests > 0:
                hit_rate = llm_stats.get('cache_hits', 0) / total_requests
                print(f"  缓存命中率: {hit_rate:.1%}")
        
        print(f"\n💾 输出文件:")
        print(f"  📊 GEXF: {config['checkpoint_dir']}/final_async_graph.gexf")
        print(f"  📦 PKL: {config['checkpoint_dir']}/final_async_graph.pkl")
        print(f"  🗜️ PKL.GZ: {config['checkpoint_dir']}/final_async_graph.pkl.gz")
        print(f"  📋 报告: {config['checkpoint_dir']}/final_async_report.json")
        
        print(f"\n🔗 使用方法:")
        print(f"  1. 将PKL文件路径设置到 generate_ripple_experiments.py:")
        print(f"     GRAPH_FILE = '{config['checkpoint_dir']}/final_async_graph.pkl'")
        print(f"  2. 运行 ripple 实验生成")
        
        # 性能对比
        sync_estimate = duration * 12  # 预估同步版本慢12倍
        print(f"\n⚡ 性能分析:")
        print(f"  异步优化版本: {duration:.1f} 秒")
        print(f"  预估同步版本: {sync_estimate:.1f} 秒 ({sync_estimate/60:.1f} 分钟)")
        print(f"  加速比: {sync_estimate/duration:.1f}x")
        print(f"  质量导向效率: {graph.number_of_nodes()/(duration/60)/config['max_concurrent']:.2f} 节点/分钟/并发")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 用户中断构建")
        current_nodes = builder.graph.number_of_nodes()
        print(f"📊 当前进度: {current_nodes} 节点")
        
    except Exception as e:
        print(f"\n❌ 构建过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行优化测试
    asyncio.run(test_optimized_specific_knowledge())