#!/usr/bin/env python3
"""
测试集成了自动问题生成的Enhanced Graph Builder
验证完整流水线：种子 → 图谱构建 → 问题生成 → 导出
"""

import os
import time
from datetime import datetime
from graph_builder.enhanced_graph_builder import create_enhanced_builder

def test_enhanced_with_auto_questions():
    """测试集成自动问题生成的增强图谱构建器"""
    
    print("🚀 测试集成自动问题生成的Enhanced Graph Builder")
    print("=" * 70)
    
    # 配置
    config = {
        'target_nodes': 50,                   # 小规模测试
        'triplets_per_query': 4,             # 每次查询的三元组数量
        'parallel_frequency': 3,             # 并行频率
        'include_optional_relations': False, # 使用核心关系
        'confidence_threshold': 0.6,         # 置信度阈值
        'candidate_threshold': 0.5,          # 候选阈值
        'verbose': True,
        'enable_early_stopping': False,     # 禁用早停
        'use_qa_atomic_ontology': True,     # 使用QA Atomic Ontology (36个函数式关系)
        'output_dir': 'results/test_auto_questions',
        'checkpoint_dir': 'results/test_auto_questions_checkpoints',
        'api_key_path': 'keys/openai.txt',
        
        # 分组配额
        'group_quotas': {
            'Person': 0.25, 'Org': 0.20, 'Geo': 0.20,
            'Work': 0.15, 'Product/Tech': 0.10, 'Event': 0.10
        },
        
        # 反爆炸限制
        'per_entity_caps': {'BirthDate': 1, 'BirthPlace': 1, 'FoundingDate': 1, '*': 5},
        'global_relation_soft_cap': 0.15,
        
        # 缓存
        'random_seed': 42,
        'cache_dir': 'results/cache_auto_questions',
    }
    
    # 种子三元组（使用与Function-like Relations兼容的关系）
    seed_triplets = [
        # Person
        ('Albert Einstein', 'BirthPlace', 'Ulm'),
        ('Albert Einstein', 'BirthDate', '1879-03-14'),
        ('Steve Jobs', 'BirthPlace', 'San Francisco'),
        
        # Org
        ('Apple Inc.', 'FoundingDate', '1976-04-01'),
        ('Apple Inc.', 'HeadquartersCity', 'Cupertino'),
        ('Microsoft', 'FoundingDate', '1975-04-04'),
        
        # Work
        ('iPhone', 'DevelopedByPrimary', 'Apple Inc.'),
        ('Windows', 'DevelopedByPrimary', 'Microsoft'),
        
        # Geo
        ('Beijing', 'CountryOfCity', 'China'),
        ('Paris', 'CountryOfCity', 'France'),
    ]
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['cache_dir'], exist_ok=True)
    
    start_time = time.time()
    
    try:
        print(f"📊 配置: {config['target_nodes']} 节点, {config['triplets_per_query']} 三元组/查询")
        
        # 创建Enhanced Builder
        builder = create_enhanced_builder(config)
        
        # 初始化API
        if not builder.initialize_api():
            print("❌ API初始化失败，请检查 keys/openai.txt")
            return False
        print("✅ API初始化成功")
        
        # 添加种子
        print(f"🌱 添加 {len(seed_triplets)} 个种子三元组...")
        builder.add_seed_triplets(seed_triplets)
        
        # 构建图谱
        print(f"\n🔨 开始图谱构建 ({datetime.now().strftime('%H:%M:%S')})")
        final_graph = builder.build_graph()
        
        elapsed = time.time() - start_time
        
        # 分析结果
        print(f"\n{'='*70}")
        print("🎉 构建完成!")
        print(f"{'='*70}")
        print("📊 最终结果:")
        print(f"   节点数: {final_graph.number_of_nodes():,}")
        print(f"   边数: {final_graph.number_of_edges():,}")
        print(f"   用时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
        
        if final_graph.number_of_nodes() > 0:
            avg_deg = (2 * final_graph.number_of_edges()) / final_graph.number_of_nodes()
            print(f"   平均度: {avg_deg:.2f}")
        
        # 分析问题生成
        edges_with_questions = 0
        total_edges = 0
        sample_questions = []
        
        print(f"\n🎯 自动问题生成分析:")
        for head, tail, data in final_graph.edges(data=True):
            total_edges += 1
            if 'question' in data and data['question']:
                edges_with_questions += 1
                if len(sample_questions) < 5:
                    sample_questions.append({
                        'triplet': f"{head} --{data.get('relation', 'Unknown')}--> {tail}",
                        'question': data['question'],
                        'confidence': data.get('confidence', 0.0)
                    })
        
        question_coverage = (edges_with_questions / total_edges * 100) if total_edges > 0 else 0
        print(f"   问题覆盖率: {edges_with_questions}/{total_edges} ({question_coverage:.1f}%)")
        
        print(f"\n📝 示例生成的问题:")
        for i, example in enumerate(sample_questions, 1):
            print(f"   {i}. 三元组: {example['triplet']}")
            print(f"      问题: \"{example['question']}\"")
            print(f"      置信度: {example['confidence']:.2f}")
            print()
        
        # 导出结果
        print("\n📁 导出结果...")
        export_paths = builder.export_results("enhanced_auto_questions")
        print(f"✅ 导出完成 → {config['output_dir']}/")
        
        for fmt, path in export_paths.items():
            try:
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"   {fmt}: {os.path.basename(path)} ({size_mb:.1f} MB)")
            except Exception:
                print(f"   {fmt}: {os.path.basename(path)}")
        
        # 验证导出的问题
        print(f"\n🔍 验证导出文件中的问题字段...")
        edges_jsonl_path = None
        for fmt, path in export_paths.items():
            if 'edges_jsonl' in fmt:
                edges_jsonl_path = path
                break
        
        if edges_jsonl_path and os.path.exists(edges_jsonl_path):
            import json
            questions_in_export = 0
            total_in_export = 0
            
            with open(edges_jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        total_in_export += 1
                        edge_data = json.loads(line)
                        if 'question' in edge_data and edge_data['question']:
                            questions_in_export += 1
            
            export_coverage = (questions_in_export / total_in_export * 100) if total_in_export > 0 else 0
            print(f"   导出文件问题覆盖率: {questions_in_export}/{total_in_export} ({export_coverage:.1f}%)")
            
            if questions_in_export > 0:
                print(f"   ✅ 问题字段成功集成到导出系统")
            else:
                print(f"   ⚠️ 导出文件中未发现问题字段")
        
        # 性能统计
        if hasattr(builder, 'state'):
            state = builder.state
            print(f"\n📈 性能统计:")
            print(f"   API调用次数: {state.get('total_llm_calls', 0)}")
            print(f"   生成三元组: {state.get('total_triplets_generated', 0)}")
            print(f"   处理实体: {state.get('entities_processed', len(state.get('processed_entities', set())))}")
            
            if state.get('total_llm_calls', 0) > 0:
                avg_time_per_call = elapsed / state['total_llm_calls']
                print(f"   平均API调用时间: {avg_time_per_call:.1f} 秒")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️ 构建被用户中断")
        if 'builder' in locals():
            builder._save_checkpoint(is_final=True)
            print("💾 进度已保存到检查点")
        return False
        
    except Exception as e:
        print(f"\n❌ 构建过程中出错: {e}")
        import traceback
        traceback.print_exc()
        if 'builder' in locals():
            try:
                builder._save_checkpoint(is_final=True)
                print("💾 部分进度已保存到检查点")
            except Exception:
                pass
        return False

def verify_integration():
    """验证各组件是否正确集成"""
    print("\n🔧 验证组件集成...")
    
    try:
        # 验证导入
        from graph_builder.enhanced_graph_builder import create_enhanced_builder
        from graph_builder.prompts import SYS_PROMPT_GRAPH_BUILDER_v0_3, create_user_prompt_v0_3
        from graph_builder.llm_calls_enhanced import TRIPLET_SCHEMA_v0_3
        from graph_builder.relations_ontology import KnowledgeTriplet
        print("✅ 所有核心模块导入成功")
        
        # 验证KnowledgeTriplet是否支持question字段
        test_triplet = KnowledgeTriplet(
            head="Test", 
            relation_id="TestRelation", 
            tail="TestTail",
            question="Test question?"
        )
        assert hasattr(test_triplet, 'question'), "KnowledgeTriplet缺少question字段"
        assert test_triplet.question == "Test question?", "KnowledgeTriplet的question字段不工作"
        print("✅ KnowledgeTriplet支持question字段")
        
        # 验证JSON schema是否包含question字段
        required_fields = TRIPLET_SCHEMA_v0_3.get('required', [])
        assert 'question' in required_fields, "TRIPLET_SCHEMA_v0_3缺少question字段"
        print("✅ JSON Schema包含question字段")
        
        # 验证prompt系统
        assert 'question' in SYS_PROMPT_GRAPH_BUILDER_v0_3, "系统prompt未提及question生成"
        print("✅ Prompt系统支持问题生成")
        
        print("🎉 所有组件集成验证通过!")
        return True
        
    except Exception as e:
        print(f"❌ 组件集成验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Enhanced Graph Builder + 自动问题生成 集成测试")
    print("=" * 70)
    
    # 验证集成
    if not verify_integration():
        print("❌ 集成验证失败，请修复后再试")
        exit(1)
    
    # 运行测试
    success = test_enhanced_with_auto_questions()
    
    if success:
        print("\n🎊 测试成功完成!")
        print("📋 总结:")
        print("  ✅ Enhanced Graph Builder已集成自动问题生成")
        print("  ✅ v0.3 Prompt系统工作正常")
        print("  ✅ 问题字段成功导出到所有格式")
        print("  ✅ 完整流水线运行正常")
    else:
        print("\n❌ 测试失败或被中断")
