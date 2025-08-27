#!/usr/bin/env python3
"""
快速生成PKL格式的知识图谱，兼容generate_ripple_experiments.py
专注于具体实体和一对一关系
"""

import asyncio
import sys
import time
sys.path.append('/root/GenFragility-LLM')

from infinite_graph_builder_async import create_async_infinite_builder

async def quick_generate_pkl(target_size: int = 500, output_name: str = "quick_graph"):
    """快速生成PKL格式图谱"""
    print(f"🚀 快速生成PKL图谱")
    print(f"目标大小: {target_size} 节点")
    print("=" * 50)
    
    # 高效配置
    config = {
        'api_key_path': '/root/GenFragility-LLM/keys/openai.txt',
        'cache_dir': '/root/GenFragility-LLM/cache/llm_responses',
        'max_concurrent': 20,      # 高并发
        'batch_size': 8,           # 合理批次
        'budget_per_entity': 12,   # 每实体三元组数
        'seed_target': min(80, target_size // 6),
        'breadth_target': min(200, target_size // 2),
        'depth_target': min(350, target_size * 0.7),
        'final_target': target_size,
        'checkpoint_interval': 50,
        'checkpoint_dir': f'/root/GenFragility-LLM/checkpoints/{output_name}'
    }
    
    # 优化的具体实体种子
    specific_seeds = [
        # 科技公司生态
        "Apple Inc.", "Microsoft Corporation", "Google LLC", "Tesla Inc.",
        "Meta Platforms", "Amazon", "OpenAI", "NVIDIA",
        
        # 知名人物  
        "Steve Jobs", "Bill Gates", "Elon Musk", "Tim Cook",
        "Albert Einstein", "Marie Curie", "Stephen Hawking",
        
        # 具体地点
        "Cupertino", "Seattle", "Mountain View", "Palo Alto",
        "New York City", "San Francisco", "Cambridge", "Boston",
        
        # 大学机构
        "Harvard University", "MIT", "Stanford University", 
        "Princeton University", "Cambridge University",
        
        # 具体产品
        "iPhone", "Windows", "Tesla Model S", "ChatGPT",
        
        # 国家
        "United States", "China", "Germany", "United Kingdom"
    ]
    
    print(f"🌱 种子数量: {len(specific_seeds)}")
    print(f"⚡ 并发设置: {config['max_concurrent']}")
    
    builder = create_async_infinite_builder(config)
    
    try:
        start_time = time.time()
        
        graph = await builder.build_infinite_graph(
            initial_seeds=specific_seeds,
            target_size=target_size
        )
        
        duration = time.time() - start_time
        
        # 验证结果
        pkl_file = f"{config['checkpoint_dir']}/final_async_graph.pkl"
        pkl_gz_file = f"{config['checkpoint_dir']}/final_async_graph.pkl.gz"
        
        print(f"\n🎉 生成完成!")
        print(f"📊 最终规模: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
        print(f"⏱️ 总耗时: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
        print(f"⚡ 生成速度: {graph.number_of_nodes()/duration:.2f} 节点/秒")
        
        print(f"\n📦 输出文件:")
        print(f"  PKL: {pkl_file}")
        print(f"  PKL.GZ: {pkl_gz_file}")
        print(f"  兼容: generate_ripple_experiments.py")
        
        # 质量检查
        abstract_count = 0
        for node in graph.nodes():
            if any(word in node.lower() for word in ['technology', 'science', 'innovation']):
                abstract_count += 1
        
        concrete_ratio = (graph.number_of_nodes() - abstract_count) / graph.number_of_nodes()
        print(f"\n📊 质量指标:")
        print(f"  具体实体比例: {concrete_ratio:.1%}")
        print(f"  平均置信度: {sum(d.get('confidence', 0) for _, _, d in graph.edges(data=True))/graph.number_of_edges():.2f}")
        
        return pkl_file
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        return None

if __name__ == "__main__":
    # 接受命令行参数
    target_size = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    output_name = sys.argv[2] if len(sys.argv) > 2 else "quick_graph"
    
    result = asyncio.run(quick_generate_pkl(target_size, output_name))
    
    if result:
        print(f"\n✅ 成功生成: {result}")
        print(f"🔧 使用方法: 将 generate_ripple_experiments.py 中的 GRAPH_FILE 改为:")
        print(f"   GRAPH_FILE = '{result}'")
    else:
        print(f"\n❌ 生成失败")
        sys.exit(1)
