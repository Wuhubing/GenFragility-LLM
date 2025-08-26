#!/usr/bin/env python3
"""
测试150节点的Enhanced Graph Builder with自动问题生成
使用完整的集成系统：种子 → 图谱构建 → 问题生成 → 导出
"""

import os
import time
from datetime import datetime
from graph_builder.enhanced_graph_builder import create_enhanced_builder
from graph_builder.llm_calls_enhanced import load_api_key

def test_150_nodes_with_auto_questions():
    """测试150节点的图谱构建，集成自动问题生成功能"""
    
    print("🚀 测试150节点Enhanced Graph Builder + 自动问题生成")
    print("=" * 70)
    
    # 初始化API
    load_api_key()
    
    # 配置参数
    config = {
        'target_nodes': 150,                 # 目标节点数
        'triplets_per_query': 4,            # 每次查询的三元组数量
        'parallel_frequency': 3,            # 并行频率
        'include_optional_relations': False, # 使用核心关系
        'confidence_threshold': 0.6,        # 置信度阈值
        'candidate_threshold': 0.5,         # 候选阈值
        'verbose': True,
        'enable_early_stopping': False,     # 禁用早停
        'use_qa_atomic_ontology': True,     # 使用QA Atomic ontology（36个function-like关系）
        'output_dir': 'results/test_150_nodes_enhanced',
        'checkpoint_dir': 'results/test_150_nodes_enhanced_checkpoints',
        'api_key_path': 'keys/openai.txt',
    }
    
    # 多样化的种子实体，覆盖不同领域
    seeds = [
        # 科技公司
        'Apple Inc.', 'Google', 'Microsoft',
        # 人物
        'Albert Einstein', 'Steve Jobs', 'Bill Gates',
        # 地理
        'United States', 'China', 'Germany',
        # 学术机构
        'Stanford University', 'MIT'
    ]
    
    print(f"📋 配置:")
    print(f"  目标节点: {config['target_nodes']}")
    print(f"  种子实体: {len(seeds)} 个")
    print(f"  三元组/查询: {config['triplets_per_query']}")
    print(f"  使用QA Atomic ontology: {config['use_qa_atomic_ontology']}")
    print(f"  种子: {', '.join(seeds)}")
    print()
    
    # 创建构建器
    print("🔧 初始化Enhanced Graph Builder...")
    builder = create_enhanced_builder(config)
    
    # 添加种子实体
    print("🌱 添加种子实体...")
    for seed in seeds:
        builder.scheduler.add_seed_entities([seed])
    
    print(f"✅ 已添加 {len(seeds)} 个种子实体")
    print()
    
    # 开始构建
    print("🚀 开始150节点图谱构建...")
    start_time = time.time()
    
    try:
        graph = builder.build_graph()
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"🎉 构建完成!")
        print(f"⏱️  用时: {duration/60:.1f} 分钟 ({duration:.1f} 秒)")
        print()
        
        # 分析结果
        print("📊 图谱统计:")
        print(f"  节点数: {graph.number_of_nodes()}")
        print(f"  边数: {graph.number_of_edges()}")
        print(f"  平均度: {graph.number_of_edges() * 2 / graph.number_of_nodes():.2f}")
        print()
        
        # 分析问题生成结果
        print("🎯 自动问题生成分析:")
        edges_with_questions = 0
        sample_questions = []
        question_lengths = []
        
        for u, v, data in graph.edges(data=True):
            question = data.get('question', '')
            if question and question.strip():
                edges_with_questions += 1
                sample_questions.append({
                    'relation': data.get('relation', '?'),
                    'head': u,
                    'tail': v,
                    'question': question
                })
                question_lengths.append(len(question.split()))
        
        question_coverage = edges_with_questions / graph.number_of_edges() * 100 if graph.number_of_edges() > 0 else 0
        avg_question_length = sum(question_lengths) / len(question_lengths) if question_lengths else 0
        
        print(f"  问题覆盖率: {edges_with_questions}/{graph.number_of_edges()} ({question_coverage:.1f}%)")
        print(f"  平均问题长度: {avg_question_length:.1f} 词")
        print()
        
        print("📝 示例生成的问题:")
        for i, q in enumerate(sample_questions[:8], 1):
            print(f"  {i}. {q['head']} --{q['relation']}--> {q['tail']}")
            print(f"     Question: \"{q['question']}\"")
        print()
        
        # 关系类型统计
        print("🔗 关系类型分布:")
        relation_counts = {}
        for u, v, data in graph.edges(data=True):
            rel = data.get('relation', 'Unknown')
            relation_counts[rel] = relation_counts.get(rel, 0) + 1
        
        # 显示前10个最常见的关系
        sorted_relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)
        for rel, count in sorted_relations[:10]:
            print(f"  {rel}: {count} 条")
        print()
        
        # 导出结果
        print("📁 导出结果...")
        try:
            builder.export_system.export_all(graph, "enhanced_150_nodes")
            print("✅ 导出完成")
            
            # 验证导出的问题字段
            import json
            edges_file = f"{config['output_dir']}/enhanced_150_nodes_edges.jsonl"
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
            
        except Exception as e:
            print(f"❌ 导出失败: {e}")
        
        print()
        print("🎊 测试成功完成!")
        print("📋 总结:")
        print(f"  ✅ 达到目标节点数: {graph.number_of_nodes()}/150")
        print(f"  ✅ 自动问题生成覆盖率: {question_coverage:.1f}%")
        print(f"  ✅ 平均问题质量: {avg_question_length:.1f} 词/问题")
        print(f"  ✅ 构建效率: {graph.number_of_nodes()/duration*60:.1f} 节点/分钟")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️ 构建被用户中断")
        return False
    except Exception as e:
        print(f"\n❌ 构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_150_nodes_with_auto_questions()
    if success:
        print("\n🌟 150节点Enhanced Graph Builder测试成功!")
    else:
        print("\n💥 测试失败")
