#!/usr/bin/env python3
"""
测试优化后的具体知识生成
使用改进的prompt和种子策略
"""

import asyncio
import sys
import os
import time
sys.path.append('/root/GenFragility-LLM')

from infinite_graph_builder_async import create_async_infinite_builder

async def test_optimized_knowledge():
    """测试优化的具体知识生成"""
    print("🎯 优化的具体知识图谱构建测试")
    print("专注于生成一对一的具体关系，避免抽象概念")
    print("=" * 60)
    
    # 针对具体知识优化的配置
    config = {
        'api_key_path': '/root/GenFragility-LLM/keys/openai.txt',
        'cache_dir': '/root/GenFragility-LLM/cache/llm_responses',
        'max_concurrent': 12,      # 适中并发
        'batch_size': 6,           # 小批次确保质量
        'budget_per_entity': 15,   # 每实体15个三元组
        'seed_target': 40,         # 种子阶段：40节点
        'breadth_target': 100,     # 广度优先：100节点
        'depth_target': 180,       # 深度优先：180节点
        'final_target': 250,       # 最终目标：250节点
        'checkpoint_interval': 20,
        'checkpoint_dir': '/root/GenFragility-LLM/checkpoints/optimized_knowledge'
    }
    
    print(f"🎯 优化配置:")
    print(f"  🔬 专注具体知识: 避免抽象概念")
    print(f"  ⚡ 并发数: {config['max_concurrent']} (适中以保证质量)")
    print(f"  📦 批次大小: {config['batch_size']} (小批次精细化)")
    print(f"  🎯 预算: {config['budget_per_entity']} 三元组/实体")
    print(f"  📊 目标: {config['final_target']} 节点")
    print()
    
    # 精心挑选的具体实体种子（避免抽象概念）
    specific_seeds = [
        # 知名人物（具体个人）
        "Albert Einstein",
        "Marie Curie", 
        "Steve Jobs",
        "Bill Gates",
        
        # 具体机构
        "Apple Inc.",
        "Harvard University",
        "MIT",
        "Stanford University",
        
        # 具体地理位置
        "Beijing",
        "San Francisco",
        "Cambridge",
        "Palo Alto",
        
        # 具体国家
        "United States",
        "China",
        "Germany",
        "France"
    ]
    
    print(f"🎯 精选具体实体种子 ({len(specific_seeds)}个):")
    print("   专注于可验证的具体关系")
    for i, seed in enumerate(specific_seeds, 1):
        print(f"  {i:2d}. {seed}")
    print()
    
    # 创建异步构建器
    builder = create_async_infinite_builder(config)
    
    try:
        start_time = time.time()
        print("🏁 开始优化的具体知识构建...")
        
        # 异步构建图谱
        graph = await builder.build_infinite_graph(
            initial_seeds=specific_seeds,
            target_size=config['final_target']
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n🎉 优化构建完成！")
        print(f"=" * 50)
        print(f"⏱️  总耗时: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
        print(f"📊 最终规模: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
        print(f"⚡ 平均速度: {graph.number_of_nodes()/duration:.2f} 节点/秒")
        
        # 分析具体知识质量
        print(f"\n🔍 具体知识质量分析:")
        
        # 统计具体实体类型
        proper_nouns = 0
        concrete_entities = 0
        abstract_concepts = 0
        
        # 分析节点类型（简单启发式）
        for node in graph.nodes():
            if any(char.isupper() for char in node[:3]):  # 以大写字母开头（可能是专有名词）
                proper_nouns += 1
            if any(keyword in node.lower() for keyword in ['inc', 'corp', 'university', 'institute', 'city']):
                concrete_entities += 1
            if any(keyword in node.lower() for keyword in ['concept', 'theory', 'philosophy', 'abstract']):
                abstract_concepts += 1
        
        print(f"  专有名词节点: {proper_nouns} ({proper_nouns/graph.number_of_nodes()*100:.1f}%)")
        print(f"  具体实体节点: {concrete_entities}")
        print(f"  抽象概念节点: {abstract_concepts}")
        
        # 显示具体关系示例
        print(f"\n🔗 具体关系示例:")
        concrete_relations = 0
        for u, v, data in graph.edges(data=True):
            if concrete_relations < 10:
                relation = data.get('relation', '未知关系')
                confidence = data.get('confidence', 0)
                question = data.get('question', '')
                
                # 检查是否为具体关系
                is_concrete = (
                    any(char.isupper() for char in u[:3]) and  # head是专有名词
                    any(char.isupper() for char in v[:3]) and  # tail是专有名词
                    confidence >= 0.7  # 高置信度
                )
                
                if is_concrete:
                    concrete_relations += 1
                    print(f"  {concrete_relations}. {u} --[{relation}]--> {v}")
                    if question:
                        print(f"     问题: {question}")
                    print(f"     置信度: {confidence:.2f}")
        
        # 显示PKL文件信息
        pkl_files = [
            f"{config['checkpoint_dir']}/final_async_graph.pkl",
            f"{config['checkpoint_dir']}/final_async_graph.pkl.gz"
        ]
        
        print(f"\n💾 PKL格式输出文件:")
        for pkl_file in pkl_files:
            if os.path.exists(pkl_file):
                size = os.path.getsize(pkl_file) / 1024  # KB
                print(f"  📦 {pkl_file} ({size:.1f} KB)")
        
        print(f"\n🎯 兼容性:")
        print(f"  ✅ 可直接用于 generate_ripple_experiments.py")
        print(f"  ✅ 更新 GRAPH_FILE 路径即可使用")
        
        # 效率对比
        if duration > 0:
            sync_estimate = duration * 12  # 预估同步版本慢12倍
            print(f"\n⚡ 性能分析:")
            print(f"  异步优化版本: {duration:.1f} 秒")
            print(f"  预估同步版本: {sync_estimate:.1f} 秒 ({sync_estimate/60:.1f} 分钟)")
            print(f"  加速比: {sync_estimate/duration:.1f}x")
        
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
    asyncio.run(test_optimized_knowledge())