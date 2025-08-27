#!/usr/bin/env python3
"""
快速测试优化版本 - 小规模验证
"""

import asyncio
import sys
import time
sys.path.append('/root/GenFragility-LLM')

from infinite_graph_builder_async import create_async_infinite_builder

async def quick_test():
    """快速测试优化版本"""
    print("🧪 快速测试 - 优化版异步构建器")
    print("=" * 50)
    
    # 小规模测试配置
    config = {
        'api_key_path': '/root/GenFragility-LLM/keys/openai.txt',
        'cache_dir': '/root/GenFragility-LLM/cache/llm_responses',
        'max_concurrent': 8,       # 8并发
        'batch_size': 4,           # 每批次4个实体
        'budget_per_entity': 10,   # 每个实体10个三元组
        'seed_target': 25,         # 25节点
        'breadth_target': 50,      # 50节点
        'depth_target': 75,        # 75节点
        'final_target': 100,       # 100节点
        'checkpoint_interval': 20,
        'checkpoint_dir': '/root/GenFragility-LLM/checkpoints/quick_test'
    }
    
    print(f"⚙️ 小规模配置: {config['max_concurrent']}并发, {config['final_target']}节点目标")
    
    # 高质量的具体种子
    initial_seeds = [
        "Tim Cook", "Apple Inc.", "Cupertino",
        "Albert Einstein", "Princeton University", 
        "Elon Musk", "Tesla Inc."
    ]
    
    print(f"🌱 种子: {initial_seeds}")
    
    builder = create_async_infinite_builder(config)
    
    try:
        start_time = time.time()
        
        graph = await builder.build_infinite_graph(
            initial_seeds=initial_seeds,
            target_size=100
        )
        
        duration = time.time() - start_time
        
        print(f"\n✅ 完成！")
        print(f"📊 {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
        print(f"⏱️ {duration:.1f} 秒")
        
        # 快速质量检查
        specific_count = sum(1 for node in graph.nodes() 
                           if any(k in node for k in ["Inc.", "University", "Corporation"]) 
                           or (node and node[0].isupper()))
        
        print(f"🎯 具体实体比例: {specific_count/graph.number_of_nodes()*100:.1f}%")
        
        # 检查pkl文件
        pkl_file = f"{config['checkpoint_dir']}/final_async_graph.pkl"
        print(f"💾 PKL文件: {pkl_file}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    asyncio.run(quick_test())
