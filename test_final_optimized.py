#!/usr/bin/env python3
"""
最终测试：优化的具体知识图谱构建
包含PKL输出和prompt优化
"""

import asyncio
import sys
import os
import time
sys.path.append('/root/GenFragility-LLM')

from infinite_graph_builder_async import create_async_infinite_builder

async def test_final_optimized():
    """最终测试优化版本"""
    print("🎯 最终优化版本测试")
    print("✅ PKL格式输出")
    print("✅ 优化的prompt - 专注具体关系")  
    print("✅ 精选种子 - 避免抽象概念")
    print("=" * 60)
    
    # 最终优化配置
    config = {
        'api_key_path': '/root/GenFragility-LLM/keys/openai.txt',
        'cache_dir': '/root/GenFragility-LLM/cache/llm_responses',
        'max_concurrent': 10,      # 适中并发确保质量
        'batch_size': 5,           # 小批次精控
        'budget_per_entity': 8,    # 每个实体8个精选三元组
        'seed_target': 25,         # 种子阶段：25节点
        'breadth_target': 60,      # 广度优先：60节点
        'depth_target': 100,       # 深度优先：100节点
        'final_target': 150,       # 最终目标：150节点
        'checkpoint_interval': 20,
        'checkpoint_dir': '/root/GenFragility-LLM/checkpoints/final_optimized'
    }
    
    print(f"⚙️ 配置:")
    print(f"  🎯 目标: {config['final_target']} 节点的具体知识图谱")
    print(f"  ⚡ 并发: {config['max_concurrent']} (质量优先)")
    print(f"  📦 批次: {config['batch_size']} 实体/批次")
    print(f"  🎯 预算: {config['budget_per_entity']} 三元组/实体")
    print()
    
    # 最优的具体种子组合
    optimized_seeds = [
        # 科技生态 - 具体人物和公司
        "Apple Inc.", "Tim Cook", "Steve Jobs",
        # 学术生态 - 具体科学家和大学
        "Albert Einstein", "Princeton University", "Marie Curie",
        # 地理生态 - 具体城市和国家
        "Beijing", "China", "Cupertino", "United States"
    ]
    
    print(f"🌱 最优种子组合 ({len(optimized_seeds)}个):")
    for i, seed in enumerate(optimized_seeds, 1):
        print(f"  {i:2d}. {seed}")
    
    print(f"\n💡 优化策略:")
    print(f"  ✅ 每个种子都是具体的命名实体")
    print(f"  ✅ 种子间有潜在的关联关系")
    print(f"  ✅ 避免抽象概念和技术术语")
    print(f"  ✅ 专注于可验证的事实关系")
    print()
    
    # 创建优化构建器
    builder = create_async_infinite_builder(config)
    
    try:
        start_time = time.time()
        print("🚀 开始最终优化构建...")
        
        # 异步构建优化图谱
        graph = await builder.build_infinite_graph(
            initial_seeds=optimized_seeds,
            target_size=config['final_target']
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n🎉 最终优化构建完成！")
        print(f"=" * 50)
        print(f"⏱️  总耗时: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
        print(f"📊 最终规模: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
        print(f"⚡ 构建速度: {graph.number_of_nodes()/duration:.2f} 节点/秒")
        print(f"🎯 边生成率: {graph.number_of_edges()/(duration/60):.1f} 边/分钟")
        
        # 检查PKL文件
        pkl_file = f"{config['checkpoint_dir']}/final_async_graph.pkl"
        pkl_gz_file = f"{config['checkpoint_dir']}/final_async_graph.pkl.gz"
        
        print(f"\n📁 输出文件:")
        if os.path.exists(pkl_file):
            file_size = os.path.getsize(pkl_file) / 1024  # KB
            print(f"  📦 PKL文件: {pkl_file}")
            print(f"     大小: {file_size:.1f} KB")
        
        if os.path.exists(pkl_gz_file):
            gz_size = os.path.getsize(pkl_gz_file) / 1024  # KB
            print(f"  🗜️ PKL压缩文件: {pkl_gz_file}")
            print(f"     大小: {gz_size:.1f} KB")
        
        # LLM统计
        if hasattr(builder, 'llm_interface') and builder.llm_interface:
            llm_stats = builder.llm_interface.stats
            print(f"\n📈 LLM效率:")
            print(f"  总调用: {llm_stats.get('total_calls', 0)}")
            print(f"  成功率: {(llm_stats.get('total_calls', 0) - llm_stats.get('failures', 0))/max(llm_stats.get('total_calls', 0), 1):.1%}")
            print(f"  缓存命中率: {llm_stats.get('cache_hits', 0)/(llm_stats.get('cache_hits', 0) + llm_stats.get('cache_misses', 0) + 1) * 100:.1f}%")
        
        # 显示具体关系示例
        print(f"\n🔗 优质具体关系示例:")
        concrete_edges = []
        
        for u, v, data in graph.edges(data=True):
            confidence = data.get('confidence', 0)
            if confidence >= 0.8:  # 高置信度关系
                concrete_edges.append((u, v, data))
        
        # 显示前8个高质量关系
        for i, (u, v, data) in enumerate(concrete_edges[:8], 1):
            relation = data.get('relation', '')
            confidence = data.get('confidence', 0)
            question = data.get('question', '')
            
            print(f"  {i}. {u} --[{relation}]--> {v}")
            print(f"     💭 {question}")
            print(f"     📊 置信度: {confidence:.2f}")
            print()
        
        # 兼容性提示
        print(f"🔗 Ripple实验集成指南:")
        print(f"  1. 更新generate_ripple_experiments.py中的GRAPH_FILE:")
        print(f"     GRAPH_FILE = '{pkl_file}'")
        print(f"  2. 图谱包含具体的命名实体，适合ripple分析")
        print(f"  3. 每条边都包含问题、置信度等完整元数据")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 用户中断")
        if hasattr(builder, 'graph'):
            current_nodes = builder.graph.number_of_nodes()
            print(f"📊 当前进度: {current_nodes} 节点")
        return False
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_final_optimized())
    if success:
        print(f"\n✅ 测试成功完成！")
    else:
        print(f"\n❌ 测试未完成")
