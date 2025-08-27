#!/usr/bin/env python3
"""
无限图谱扩张演示
测试新的可扩展架构
"""

import sys
import os
sys.path.append('/root/GenFragility-LLM')

from infinite_graph_builder import create_infinite_builder
import time

def demo_infinite_expansion():
    """演示无限扩张功能"""
    print("🚀 无限知识图谱构建演示")
    print("=" * 50)
    
    # 配置
    config = {
        'api_key_path': '/root/GenFragility-LLM/keys/openai.txt',
        'cache_dir': '/root/GenFragility-LLM/cache/llm_responses',
        'seed_target': 50,        # 种子阶段目标：50个节点
        'breadth_target': 200,    # 广度优先目标：200个节点  
        'depth_target': 500,      # 深度优先目标：500个节点
        'final_target': 1000,     # 最终目标：1000个节点
        'min_confidence': 0.6,
        'max_batch_size': 20,
        'checkpoint_interval': 25,
        'checkpoint_dir': '/root/GenFragility-LLM/demo_checkpoints'
    }
    
    # 创建构建器
    builder = create_infinite_builder(config)
    
    # 多样化的初始种子
    initial_seeds = [
        # 科技公司
        "Apple Inc.", "Microsoft", "Google", "Tesla",
        # 科学家
        "Einstein", "Newton", "Darwin", "Curie", 
        # 城市
        "Beijing", "Shanghai", "New York", "London",
        # 编程语言
        "Python", "Java", "JavaScript",
        # 国家
        "China", "United States", "Japan", "Germany",
        # 文学作品
        "Hamlet", "Romeo and Juliet",
        # 乐器
        "Piano", "Guitar", "Violin"
    ]
    
    print(f"🌱 初始种子 ({len(initial_seeds)} 个):")
    for i, seed in enumerate(initial_seeds, 1):
        print(f"  {i:2d}. {seed}")
    
    print(f"\n🎯 扩张目标：")
    print(f"  种子扩张: {config['seed_target']} 节点")
    print(f"  广度优先: {config['breadth_target']} 节点") 
    print(f"  深度优先: {config['depth_target']} 节点")
    print(f"  最终目标: {config['final_target']} 节点")
    
    # 开始构建
    print(f"\n⏰ 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        start_time = time.time()
        
        # 构建图谱
        graph = builder.build_infinite_graph(
            initial_seeds=initial_seeds,
            target_size=config['final_target']
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n🎉 构建完成！")
        print(f"⏱️ 总耗时: {duration/60:.2f} 分钟")
        print(f"📊 最终规模: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
        
        # 获取详细报告
        report = builder.get_expansion_report()
        
        print(f"\n📈 详细统计:")
        print("=" * 40)
        
        for section_name, section_data in report.items():
            print(f"\n{section_name}:")
            for key, value in section_data.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {sub_value}")
                else:
                    print(f"  {key}: {value}")
        
        # 分析图谱特征
        print(f"\n🔍 图谱特征分析:")
        print("=" * 40)
        
        # 节点度分布
        import networkx as nx
        degrees = [graph.degree(node) for node in graph.nodes()]
        if degrees:
            print(f"  平均度: {sum(degrees)/len(degrees):.2f}")
            print(f"  最大度: {max(degrees)}")
            print(f"  最小度: {min(degrees)}")
        
        # 关系类型分布
        relations = {}
        for _, _, data in graph.edges(data=True):
            rel = data.get('relation', 'Unknown')
            relations[rel] = relations.get(rel, 0) + 1
        
        print(f"\n  关系类型分布 (前10):")
        for rel, count in sorted(relations.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {rel}: {count}")
        
        # QA-eligible统计
        qa_edges = sum(1 for _, _, data in graph.edges(data=True) 
                      if data.get('qa_eligible', False))
        total_edges = graph.number_of_edges()
        
        print(f"\n  QA-eligible边: {qa_edges}/{total_edges} ({qa_edges/total_edges*100:.1f}%)")
        
        # 置信度分布
        confidences = [data.get('confidence', 0) for _, _, data in graph.edges(data=True)]
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            high_conf = sum(1 for c in confidences if c >= 0.8)
            print(f"  平均置信度: {avg_conf:.3f}")
            print(f"  高置信度边 (≥0.8): {high_conf}/{len(confidences)} ({high_conf/len(confidences)*100:.1f}%)")
        
        # 展示一些示例问题
        print(f"\n❓ 示例问题 (前10个):")
        question_count = 0
        for _, _, data in graph.edges(data=True):
            if data.get('question') and question_count < 10:
                question_count += 1
                question = data['question']
                # 获取对应的tail (答案)
                print(f"  {question_count:2d}. {question}")
        
        print(f"\n💾 结果已保存到: {config['checkpoint_dir']}/")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 用户中断构建过程")
        report = builder.get_expansion_report()
        current_nodes = report['总体统计']['节点数']
        print(f"📊 当前进度: {current_nodes} 节点")
        print(f"💾 进度已保存到检查点")
        
    except Exception as e:
        print(f"\n❌ 构建过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def analyze_existing_checkpoint():
    """分析现有检查点"""
    import json
    import glob
    
    checkpoint_dir = '/root/GenFragility-LLM/demo_checkpoints'
    
    # 查找最新的检查点
    checkpoints = glob.glob(f"{checkpoint_dir}/checkpoint_*.json")
    if not checkpoints:
        print("❌ 未找到任何检查点文件")
        return
    
    # 获取最新检查点
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    
    print(f"🔍 分析检查点: {latest_checkpoint}")
    
    try:
        with open(latest_checkpoint, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📊 检查点统计:")
        print(f"  时间戳: {data.get('timestamp', 'Unknown')}")
        print(f"  当前阶段: {data.get('current_phase', 'Unknown')}")
        
        stats = data.get('stats', {})
        print(f"  节点数: {stats.get('total_nodes', 0)}")
        print(f"  边数: {stats.get('total_edges', 0)}")
        print(f"  LLM调用次数: {stats.get('llm_calls', 0)}")
        
        pools = data.get('entity_pools', {})
        print(f"  待处理实体: {len(pools.get('pending', []))}")
        print(f"  已完成实体: {len(pools.get('completed', []))}")
        print(f"  失败实体: {len(pools.get('failed', []))}")
        
    except Exception as e:
        print(f"❌ 分析检查点失败: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_existing_checkpoint()
    else:
        demo_infinite_expansion()
