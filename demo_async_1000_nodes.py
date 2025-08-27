#!/usr/bin/env python3
"""
测试异步高并发构建1000节点图谱
"""

import asyncio
import sys
import os
import time
sys.path.append('/root/GenFragility-LLM')

from infinite_graph_builder_async import create_async_infinite_builder

async def test_1000_nodes():
    """测试异步构建1000节点图谱"""
    print("🚀 异步高并发1000节点图谱构建测试")
    print("=" * 60)
    
    # 高性能配置
    config = {
        'api_key_path': '/root/GenFragility-LLM/keys/openai.txt',
        'cache_dir': '/root/GenFragility-LLM/cache/llm_responses',
        'max_concurrent': 25,      # 25并发调用
        'batch_size': 12,          # 每批次12个实体
        'budget_per_entity': 18,   # 每个实体18个三元组
        'seed_target': 80,         # 种子阶段：80节点
        'breadth_target': 250,     # 广度优先：250节点
        'depth_target': 500,       # 深度优先：500节点
        'final_target': 1000,      # 最终目标：1000节点
        'checkpoint_interval': 50,  # 每50个节点保存检查点
        'checkpoint_dir': '/root/GenFragility-LLM/checkpoints/async_1000_test'
    }
    
    print(f"⚡ 配置:")
    print(f"  🔥 最大并发: {config['max_concurrent']} 个LLM调用")
    print(f"  📦 批次大小: {config['batch_size']} 个实体/批次")
    print(f"  🎯 预算: {config['budget_per_entity']} 个三元组/实体")
    print(f"  📊 阶段目标: {config['seed_target']} → {config['breadth_target']} → {config['depth_target']} → {config['final_target']}")
    print()
    
    # 精心选择的多样化种子
    initial_seeds = [
        # 科技公司
        "Apple Inc.", "Microsoft", "Google", "Tesla",
        
        # 著名人物  
        "Einstein", "Marie Curie", "Steve Jobs", "Elon Musk",
        
        # 城市/地理
        "Beijing", "New York", "London", "Tokyo",
        
        # 编程/技术
        "Python", "JavaScript", "Machine Learning", "AI",
        
        # 科学概念
        "DNA", "Quantum Physics", "Evolution", "Photosynthesis",
        
        # 国家/政治
        "China", "United States", "Germany", "Japan"
    ]
    
    print(f"🌱 初始种子 ({len(initial_seeds)}个):")
    for i, seed in enumerate(initial_seeds, 1):
        print(f"  {i:2d}. {seed}")
    print()
    
    # 创建异步构建器
    builder = create_async_infinite_builder(config)
    
    try:
        start_time = time.time()
        print("🏁 开始异步构建...")
        
        # 异步构建图谱
        graph = await builder.build_infinite_graph(
            initial_seeds=initial_seeds,
            target_size=1000
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n🎉 异步构建完成！")
        print(f"=" * 50)
        print(f"⏱️  总耗时: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
        print(f"📊 最终规模: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
        print(f"⚡ 平均速度: {graph.number_of_nodes()/duration:.2f} 节点/秒")
        print(f"🚀 并发效率: {graph.number_of_edges()/(duration/60):.1f} 边/分钟")
        
        # 显示LLM统计
        if hasattr(builder, 'llm_interface') and builder.llm_interface:
            llm_stats = builder.llm_interface.stats
            print(f"\n📈 LLM调用统计:")
            print(f"  总调用: {llm_stats.get('total_calls', 0)}")
            print(f"  缓存命中: {llm_stats.get('cache_hits', 0)}")
            print(f"  缓存失误: {llm_stats.get('cache_misses', 0)}")
            print(f"  失败次数: {llm_stats.get('failures', 0)}")
            
            if llm_stats.get('cache_hits', 0) + llm_stats.get('cache_misses', 0) > 0:
                hit_rate = llm_stats.get('cache_hits', 0) / (llm_stats.get('cache_hits', 0) + llm_stats.get('cache_misses', 0))
                print(f"  缓存命中率: {hit_rate:.1%}")
        
        # 显示图谱质量
        print(f"\n📊 图谱质量:")
        density = graph.number_of_edges() / (graph.number_of_nodes() * (graph.number_of_nodes() - 1)) if graph.number_of_nodes() > 1 else 0
        avg_degree = graph.number_of_edges() * 2 / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0
        print(f"  密度: {density:.4f}")
        print(f"  平均度: {avg_degree:.2f}")
        
        # 显示一些示例边
        print(f"\n🔗 示例边:")
        edge_count = 0
        for u, v, data in graph.edges(data=True):
            if edge_count < 8:
                relation = data.get('relation', '未知关系')
                confidence = data.get('confidence', 0)
                print(f"  {edge_count+1}. {u} --[{relation}]--> {v} (置信度: {confidence:.2f})")
                edge_count += 1
            else:
                break
        
        print(f"\n💾 所有数据已保存到: {config['checkpoint_dir']}/")
        
        # 与同步版本对比
        sync_time_estimate = duration * 25  # 假设同步版本慢25倍
        print(f"\n⚡ 性能对比:")
        print(f"  异步版本: {duration:.1f} 秒")
        print(f"  预估同步版本: {sync_time_estimate:.1f} 秒 ({sync_time_estimate/60:.1f} 分钟)")
        print(f"  加速比: {sync_time_estimate/duration:.1f}x")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 用户中断构建")
        current_nodes = builder.graph.number_of_nodes()
        print(f"📊 当前进度: {current_nodes} 节点")
        
    except Exception as e:
        print(f"\n❌ 构建过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行异步测试
    asyncio.run(test_1000_nodes())
