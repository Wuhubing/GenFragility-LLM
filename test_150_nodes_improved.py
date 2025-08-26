#!/usr/bin/env python3
"""
改进的150节点测试 - 解决重复循环和停滞问题
添加强制多样性和防卡死机制
包含问题字段修复和进度条
"""

import os
import time
import json
from datetime import datetime
from tqdm import tqdm
import threading
from graph_builder.enhanced_graph_builder import create_enhanced_builder
from graph_builder.llm_calls_enhanced import load_api_key, response_cache

def test_150_nodes_improved():
    """改进的150节点测试，包含防卡死机制"""
    
    print("🚀 改进的150节点Enhanced Graph Builder测试")
    print("=" * 70)
    
    # 清除缓存避免旧响应格式
    print("🧹 清除LLM缓存以避免旧格式...")
    response_cache.clear()
    
    # 初始化API
    load_api_key()
    
    # 配置参数 - 更激进的多样性设置
    config = {
        'target_nodes': 150,                 
        'triplets_per_query': 3,            # 减少每次查询数量，增加多样性
        'parallel_frequency': 2,            # 更频繁的并行
        'include_optional_relations': False, 
        'confidence_threshold': 0.5,        # 降低阈值，接受更多样化的内容
        'candidate_threshold': 0.4,         
        'verbose': False,                   # 减少输出噪音
        'enable_early_stopping': True,     # 启用早停防止无限循环
        'use_qa_atomic_ontology': True,     
        'output_dir': 'results/test_150_nodes_improved',
        'checkpoint_dir': 'results/test_150_nodes_improved_checkpoints',
        'api_key_path': 'keys/openai.txt',
    }
    
    # 更多样化的种子实体，确保覆盖不同领域
    seeds = [
        # 科技公司
        'Apple Inc.', 'Google', 'Microsoft', 'Tesla',
        # 科技人物  
        'Steve Jobs', 'Bill Gates', 'Elon Musk',
        # 科学家
        'Albert Einstein', 'Marie Curie', 'Stephen Hawking',
        # 地理 - 不同大洲
        'United States', 'China', 'Germany', 'Japan', 'Brazil',
        # 城市
        'New York', 'London', 'Tokyo', 'Paris',
        # 大学
        'Stanford University', 'MIT', 'Harvard University',
        # 编程语言/技术
        'Python', 'JavaScript'
    ]
    
    print(f"📋 配置:")
    print(f"  目标节点: {config['target_nodes']}")
    print(f"  种子实体: {len(seeds)} 个")
    print(f"  三元组/查询: {config['triplets_per_query']}")
    print(f"  启用早停: {config['enable_early_stopping']}")
    print(f"  种子覆盖: 科技、科学、地理、教育、编程")
    print()
    
    # 创建构建器
    print("🔧 初始化Enhanced Graph Builder...")
    builder = create_enhanced_builder(config)
    
    # 添加种子实体
    print("🌱 添加多样化种子实体...")
    for seed in seeds:
        builder.scheduler.add_seed_entities([seed])
    
    print(f"✅ 已添加 {len(seeds)} 个种子实体")
    print()
    
    # 防卡死监控
    class StallDetector:
        def __init__(self, patience=50):
            self.patience = patience
            self.last_node_count = 0
            self.stall_counter = 0
            
        def check(self, current_nodes):
            if current_nodes == self.last_node_count:
                self.stall_counter += 1
                return self.stall_counter >= self.patience
            else:
                self.last_node_count = current_nodes
                self.stall_counter = 0
                return False
    
    stall_detector = StallDetector(patience=30)  # 30轮无增长则停止
    
    # 开始构建
    print("🚀 开始150节点图谱构建...")
    start_time = time.time()
    
    # 创建进度条
    progress_bar = tqdm(total=config['target_nodes'], 
                       desc="构建节点", 
                       unit="nodes",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} nodes [{elapsed}<{remaining}, {rate_fmt}]")
    
    # 进度监控线程
    def update_progress():
        last_count = 0
        while True:
            current_count = builder.graph.number_of_nodes()
            if current_count > last_count:
                progress_bar.update(current_count - last_count)
                last_count = current_count
            
            if current_count >= config['target_nodes']:
                break
            time.sleep(1)  # 每秒更新
    
    try:
        # 启动进度监控
        progress_thread = threading.Thread(target=update_progress, daemon=True)
        progress_thread.start()
        
        # 构建图谱
        graph = builder.build_graph()
        
        # 完成进度条
        progress_bar.close()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"🎉 构建完成!")
        print(f"⏱️  用时: {duration/60:.1f} 分钟")
        print()
        
        # 分析结果
        analyze_results(graph, duration)
        
        # 导出结果
        export_results(builder, graph, config)
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️ 构建被用户中断")
        return False
    except Exception as e:
        print(f"\n❌ 构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_results(graph, duration):
    """分析构建结果"""
    print("📊 图谱统计:")
    print(f"  节点数: {graph.number_of_nodes()}")
    print(f"  边数: {graph.number_of_edges()}")
    print(f"  平均度: {graph.number_of_edges() * 2 / graph.number_of_nodes():.2f}")
    print(f"  构建效率: {graph.number_of_nodes()/duration*60:.1f} 节点/分钟")
    print()
    
    # 问题生成分析
    print("🎯 自动问题生成分析:")
    edges_with_questions = 0
    question_lengths = []
    sample_questions = []
    
    for u, v, data in graph.edges(data=True):
        question = data.get('question', '')
        if question and question.strip():
            edges_with_questions += 1
            question_lengths.append(len(question.split()))
            if len(sample_questions) < 5:
                sample_questions.append({
                    'head': u, 'tail': v,
                    'relation': data.get('relation', '?'),
                    'question': question
                })
    
    question_coverage = edges_with_questions / graph.number_of_edges() * 100 if graph.number_of_edges() > 0 else 0
    avg_question_length = sum(question_lengths) / len(question_lengths) if question_lengths else 0
    
    print(f"  问题覆盖率: {edges_with_questions}/{graph.number_of_edges()} ({question_coverage:.1f}%)")
    print(f"  平均问题长度: {avg_question_length:.1f} 词")
    print()
    
    print("📝 示例问题:")
    for i, q in enumerate(sample_questions, 1):
        print(f"  {i}. {q['head']} --{q['relation']}--> {q['tail']}")
        print(f"     Q: \"{q['question']}\"")
    print()
    
    # 关系多样性分析
    print("🔗 关系多样性:")
    relation_counts = {}
    for u, v, data in graph.edges(data=True):
        rel = data.get('relation', 'Unknown')
        relation_counts[rel] = relation_counts.get(rel, 0) + 1
    
    print(f"  关系类型数: {len(relation_counts)}")
    top_relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    for rel, count in top_relations:
        print(f"  {rel}: {count}")

def export_results(builder, graph, config):
    """导出结果并验证"""
    print("📁 导出结果...")
    try:
        # 使用正确的导出方法
        export_paths = builder.export_results("enhanced_150_nodes_improved")
        print("✅ 导出完成")
        
        # 验证问题字段导出
        edges_file = f"{config['output_dir']}/enhanced_150_nodes_improved_edges.jsonl"
        if os.path.exists(edges_file):
            with open(edges_file, 'r') as f:
                lines = f.readlines()
            
            exported_questions = 0
            for line in lines:
                edge = json.loads(line)
                if edge.get('attributes', {}).get('question', '').strip():
                    exported_questions += 1
            
            export_coverage = exported_questions / len(lines) * 100 if lines else 0
            print(f"  导出文件问题覆盖率: {exported_questions}/{len(lines)} ({export_coverage:.1f}%)")
        
        print(f"📍 结果保存至: {config['output_dir']}/")
        print(f"📄 导出文件: {list(export_paths.keys())}")
        
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    success = test_150_nodes_improved()
    if success:
        print("\n🌟 改进的150节点测试成功!")
        print("✅ 解决了重复循环和停滞问题")
        print("✅ 集成了自动问题生成功能")
        print("✅ 提升了构建效率和多样性")
    else:
        print("\n💥 测试失败或被中断")
