#!/usr/bin/env python3
"""
修复后的异步高并发测试 - 较小规模快速验证
"""

import asyncio
import sys
import os
import time
sys.path.append('/root/GenFragility-LLM')

from infinite_graph_builder_async import create_async_infinite_builder

async def test_async_fixed():
    """测试修复后的异步构建"""
    print("🚀 修复后的异步高并发图谱构建测试")
    print("=" * 60)
    
    # 较小规模的测试配置
    config = {
        'api_key_path': '/root/GenFragility-LLM/keys/openai.txt',
        'cache_dir': '/root/GenFragility-LLM/cache/llm_responses',
        'max_concurrent': 15,      # 15并发调用
        'batch_size': 8,           # 每批次8个实体
        'budget_per_entity': 12,   # 每个实体12个三元组
        'seed_target': 30,         # 种子阶段：30节点
        'breadth_target': 80,      # 广度优先：80节点
        'depth_target': 150,       # 深度优先：150节点
        'final_target': 200,       # 最终目标：200节点
        'checkpoint_interval': 25,
        'checkpoint_dir': '/root/GenFragility-LLM/checkpoints/async_fixed_test'
    }
    
    print(f"⚡ 配置:")
    print(f"  🔥 最大并发: {config['max_concurrent']} 个LLM调用")
    print(f"  📦 批次大小: {config['batch_size']} 个实体/批次")
    print(f"  🎯 预算: {config['budget_per_entity']} 个三元组/实体")
    print(f"  📊 阶段目标: {config['seed_target']} → {config['breadth_target']} → {config['depth_target']} → {config['final_target']}")
    print()
    
    # 优化的具体实体种子（避免抽象概念）
    initial_seeds = [
        "Apple Inc.", "Albert Einstein", "Beijing", "Google LLC", "China",
        "Marie Curie", "London", "Tesla Inc.", "United States", "Tim Cook"
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
            target_size=200
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n🎉 异步构建完成！")
        print(f"=" * 50)
        print(f"⏱️  总耗时: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
        print(f"📊 最终规模: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
        print(f"⚡ 平均速度: {graph.number_of_nodes()/duration:.2f} 节点/秒")
        print(f"🚀 生成效率: {graph.number_of_edges()/(duration/60):.1f} 边/分钟")
        
        # 显示LLM统计
        if hasattr(builder, 'llm_interface') and builder.llm_interface:
            llm_stats = builder.llm_interface.stats
            print(f"\n📈 LLM调用统计:")
            print(f"  总调用: {llm_stats.get('total_calls', 0)}")
            print(f"  缓存命中: {llm_stats.get('cache_hits', 0)}")
            print(f"  缓存失误: {llm_stats.get('cache_misses', 0)}")
            print(f"  失败次数: {llm_stats.get('failures', 0)}")
            
            total_requests = llm_stats.get('cache_hits', 0) + llm_stats.get('cache_misses', 0)
            if total_requests > 0:
                hit_rate = llm_stats.get('cache_hits', 0) / total_requests
                print(f"  缓存命中率: {hit_rate:.1%}")
        
        # 显示图谱质量
        print(f"\n📊 图谱质量:")
        if graph.number_of_nodes() > 1:
            density = graph.number_of_edges() / (graph.number_of_nodes() * (graph.number_of_nodes() - 1))
            avg_degree = graph.number_of_edges() * 2 / graph.number_of_nodes()
            print(f"  密度: {density:.4f}")
            print(f"  平均度: {avg_degree:.2f}")
        
        # 显示示例三元组
        print(f"\n🔗 示例三元组:")
        edge_count = 0
        for u, v, data in graph.edges(data=True):
            if edge_count < 8:
                relation = data.get('relation', '未知关系')
                confidence = data.get('confidence', 0)
                question = data.get('question', '')
                print(f"  {edge_count+1}. {u} --[{relation}]--> {v}")
                if question:
                    print(f"     问题: {question}")
                print(f"     置信度: {confidence:.2f}")
                edge_count += 1
            else:
                break
        
        print(f"\n💾 所有数据已保存到: {config['checkpoint_dir']}/")
        
        # 性能分析
        if duration > 0:
            sync_estimate = duration * 15  # 预估同步版本慢15倍
            print(f"\n⚡ 性能分析:")
            print(f"  异步版本: {duration:.1f} 秒")
            print(f"  预估同步版本: {sync_estimate:.1f} 秒 ({sync_estimate/60:.1f} 分钟)")
            print(f"  加速比: {sync_estimate/duration:.1f}x")
            print(f"  并发效率: {graph.number_of_nodes()/(duration/60)/config['max_concurrent']:.2f} 节点/分钟/并发")
        
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
    asyncio.run(test_async_fixed())
