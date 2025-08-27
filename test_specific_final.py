#!/usr/bin/env python3
"""
最终测试：优化的具体知识生成
输出PKL格式，专注一对一关系
"""

import asyncio
import sys
import os
import time
sys.path.append('/root/GenFragility-LLM')

from infinite_graph_builder_async import create_async_infinite_builder

async def test_final():
    """最终优化测试"""
    print("🎯 最终优化测试：具体知识 + PKL输出")
    print("=" * 50)
    
    # 小规模快速测试
    config = {
        'api_key_path': '/root/GenFragility-LLM/keys/openai.txt',
        'cache_dir': '/root/GenFragility-LLM/cache/llm_responses',
        'max_concurrent': 8,
        'batch_size': 4,
        'budget_per_entity': 10,
        'seed_target': 25,
        'breadth_target': 50,
        'depth_target': 80,
        'final_target': 100,
        'checkpoint_interval': 25,
        'checkpoint_dir': '/root/GenFragility-LLM/checkpoints/final_test'
    }
    
    # 精选具体实体
    initial_seeds = [
        "Apple Inc.", "Steve Jobs", "Tim Cook",
        "Harvard University", "Cambridge",
        "Albert Einstein", "Princeton University"
    ]
    
    print(f"🌱 种子: {initial_seeds}")
    print(f"🎯 目标: {config['final_target']} 节点")
    print()
    
    builder = create_async_infinite_builder(config)
    
    try:
        start_time = time.time()
        print("🚀 开始构建...")
        
        graph = await builder.build_infinite_graph(
            initial_seeds=initial_seeds,
            target_size=100
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ 构建完成!")
        print(f"📊 规模: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
        print(f"⏱️  耗时: {duration:.1f} 秒")
        
        # 检查输出文件
        pkl_file = f"{config['checkpoint_dir']}/final_async_graph.pkl"
        if os.path.exists(pkl_file):
            print(f"✅ PKL文件已生成: {pkl_file}")
            print(f"📦 文件大小: {os.path.getsize(pkl_file)} 字节")
        
        # 显示具体关系示例
        print(f"\n🔗 具体关系示例:")
        count = 0
        for u, v, data in graph.edges(data=True):
            if count < 5:
                relation = data.get('relation', 'unknown')
                confidence = data.get('confidence', 0)
                print(f"  {count+1}. {u} --[{relation}]--> {v} (信心: {confidence:.2f})")
                count += 1
        
        print(f"\n🎉 优化完成:")
        print(f"  ✅ 使用优化的prompt (专注具体实体)")
        print(f"  ✅ 输出PKL格式 (兼容ripple实验)")
        print(f"  ✅ 高并发异步生成 ({config['max_concurrent']}并发)")
        print(f"  ✅ 避免抽象概念，专注一对一关系")
        
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    asyncio.run(test_final())
