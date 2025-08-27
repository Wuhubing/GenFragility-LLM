#!/usr/bin/env python3
"""
快速测试优化的prompt效果
"""

import asyncio
import sys
sys.path.append('/root/GenFragility-LLM')

from infinite_graph_builder_async import create_async_infinite_builder

async def quick_test():
    """快速测试优化的prompt"""
    print("🎯 快速测试优化的Prompt和种子")
    print("=" * 50)
    
    config = {
        'api_key_path': '/root/GenFragility-LLM/keys/openai.txt',
        'cache_dir': '/root/GenFragility-LLM/cache/llm_responses',
        'max_concurrent': 8,
        'batch_size': 4,
        'budget_per_entity': 8,
        'seed_target': 25,
        'breadth_target': 50,
        'depth_target': 75,
        'final_target': 100,
        'checkpoint_interval': 20,
        'checkpoint_dir': '/root/GenFragility-LLM/checkpoints/quick_optimize_test'
    }
    
    # 精选的具体实体（确保能生成具体关系）
    initial_seeds = [
        "Apple Inc.",      # 具体公司
        "Tim Cook",        # 具体CEO
        "Stanford University",  # 具体大学
        "Elon Musk",       # 具体人物
    ]
    
    print(f"🌱 种子: {initial_seeds}")
    print(f"🎯 目标: 100节点")
    print()
    
    builder = create_async_infinite_builder(config)
    
    try:
        graph = await builder.build_infinite_graph(
            initial_seeds=initial_seeds,
            target_size=100
        )
        
        print(f"\n✅ 完成：{graph.number_of_nodes()} 节点，{graph.number_of_edges()} 边")
        
        # 快速分析
        print(f"\n🔍 关系样例：")
        count = 0
        for u, v, data in graph.edges(data=True):
            if count < 8:
                relation = data.get('relation', 'unknown')
                confidence = data.get('confidence', 0)
                print(f"  {u} --[{relation}]--> {v} (置信度: {confidence:.2f})")
                count += 1
        
        # 检查PKL保存
        import os
        pkl_files = [f for f in os.listdir(config['checkpoint_dir']) if f.endswith('.pkl')]
        if pkl_files:
            print(f"\n💾 PKL文件已保存: {pkl_files}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    asyncio.run(quick_test())
