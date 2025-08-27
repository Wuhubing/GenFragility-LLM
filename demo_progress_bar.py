#!/usr/bin/env python3
"""
带进度条的无限图谱构建演示
"""

import sys
import os
sys.path.append('/root/GenFragility-LLM')

from infinite_graph_builder import create_infinite_builder
import time

def demo_with_progress():
    """演示带进度条的图谱构建"""
    print("🚀 带进度条的无限图谱构建演示")
    print("=" * 60)
    
    # 配置（小规模测试）
    config = {
        'api_key_path': '/root/GenFragility-LLM/keys/openai.txt',
        'cache_dir': '/root/GenFragility-LLM/cache/llm_responses',
        'seed_target': 20,        # 种子阶段：20个节点
        'breadth_target': 50,     # 广度优先：50个节点  
        'depth_target': 80,       # 深度优先：80个节点
        'final_target': 100,      # 最终目标：100个节点
        'min_confidence': 0.6,
        'max_batch_size': 10,
        'checkpoint_interval': 10,  # 每10个节点保存一次
        'checkpoint_dir': '/root/GenFragility-LLM/demo_progress_checkpoints'
    }
    
    # 创建构建器
    builder = create_infinite_builder(config)
    
    # 精心选择的初始种子（保证扩张性）
    initial_seeds = [
        "Apple Inc.",     # 科技公司
        "Einstein",       # 科学家
        "Beijing",        # 城市
        "Python",         # 编程语言
        "China",          # 国家
        "Shakespeare",    # 文学家
    ]
    
    print(f"🌱 初始种子: {initial_seeds}")
    print(f"🎯 扩张计划:")
    print(f"  🌱 种子扩张: 0 → {config['seed_target']} 节点")
    print(f"  🌊 广度优先: {config['seed_target']} → {config['breadth_target']} 节点")
    print(f"  🏊‍♂️ 深度优先: {config['breadth_target']} → {config['depth_target']} 节点")
    print(f"  🔺 关系强化: {config['depth_target']} → {config['final_target']} 节点")
    print()
    
    try:
        start_time = time.time()
        
        # 开始构建（带进度条）
        print("开始构建...")
        graph = builder.build_infinite_graph(
            initial_seeds=initial_seeds,
            target_size=config['final_target']
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n🎉 构建完成！")
        print(f"⏱️ 总耗时: {duration:.1f} 秒")
        print(f"📊 最终规模: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
        
        # 获取详细报告
        report = builder.get_expansion_report()
        
        print(f"\n📈 最终报告:")
        print("=" * 50)
        
        总体统计 = report.get('总体统计', {})
        for key, value in 总体统计.items():
            print(f"  {key}: {value}")
        
        print(f"\n实体状态:")
        实体状态 = report.get('实体状态', {})
        for key, value in 实体状态.items():
            print(f"  {key}: {value}")
        
        print(f"\n图谱质量:")
        图谱质量 = report.get('图谱质量', {})
        for key, value in 图谱质量.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
        # 展示一些示例问题
        print(f"\n❓ 示例问题:")
        question_count = 0
        for _, _, data in graph.edges(data=True):
            if data.get('question') and question_count < 8:
                question_count += 1
                question = data['question']
                tail = data.get('tail', '未知')
                print(f"  {question_count}. {question}")
                print(f"     答案: {tail}")
        
        print(f"\n💾 所有数据已保存到: {config['checkpoint_dir']}/")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 用户中断")
        report = builder.get_expansion_report()
        if report:
            current_nodes = report.get('总体统计', {}).get('节点数', 0)
            print(f"📊 当前进度: {current_nodes} 节点")
        
    except Exception as e:
        print(f"\n❌ 出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_with_progress()
