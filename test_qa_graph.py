#!/usr/bin/env python3
"""
独立测试脚本：使用 QA-Atomic 关系构建知识图谱
"""

import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.append('.')

from graph_builder.qa_atomic_ontology import QAAtomicOntology
from graph_builder.llm_calls_enhanced import load_api_key, _call_llm_with_cache
from graph_builder.prompts import SYS_PROMPT_GRAPH_BUILDER_v0_3


def create_qa_atomic_prompt(seeds, budget=50, language="zh"):
    """创建 QA-Atomic 专用 prompt"""
    ontology = QAAtomicOntology()
    all_relations = ontology.get_all_relations()
    
    # 格式化关系列表
    relation_lines = []
    qa_atomic_list = []
    
    for rel_id, rel_info in all_relations.items():
        group = rel_info.get('group', 'Unknown')
        domain = rel_info.get('domain', 'Entity')
        range_val = rel_info.get('range', 'Entity')
        
        formatted_rel = f'  "{rel_id}|{group}|{domain}->{range_val}"'
        relation_lines.append(formatted_rel)
        qa_atomic_list.append(f'  "{rel_id}"')
    
    # 格式化逆关系策略
    inverse_pairs = ontology.get_auto_inverse_pairs()
    policy_lines = []
    for rel_id, inverse_id in inverse_pairs.items():
        if rel_id in all_relations:  # 只包含正向关系
            policy_lines.append(f'  "{rel_id}": "auto-inverse: {inverse_id}"')
    
    # 创建 prompt
    prompt = f"""### Seeds
SEEDS = {seeds}

### Relation Inventories
GRAPH_CORE_RELATIONS = [
{chr(10).join(relation_lines)}
]

QA_ATOMIC_RELATIONS = [
{chr(10).join(qa_atomic_list)}
]

AUTO_INVERSE_POLICY = {{
{chr(10).join(policy_lines)}
}}

### Qualifier Rules (QA-Atomic)
- CurrentEmployer / CurrentPosition / ChiefExecutiveOfficerCurrent: require qualifiers.current = true
- PopulationAsOf: require qualifiers.as_of_year = one reasonable year (e.g., 2015–2022), then unique
- NationalityPrimary / AlmaMaterPrimary / TimeZonePrimary: require qualifiers.primary = true
- CapitalCityOfCountry: only for single-capital countries (skip multi-capital cases)
- HeadquartersCity / HeadquartersCountry: require qualifiers.primary = true for multi-office orgs

### Constraints
LANGUAGE = "{language}"
BUDGET = {budget}

### Your Output
Return up to BUDGET JSONL objects strictly following the schema. ALL relations are QA-Atomic,
so prioritize uniqueness with qualifiers. Focus on factually verifiable, unique answers."""
    
    return prompt


def parse_jsonl_response(content):
    """解析 JSONL 响应"""
    if not content:
        return []
    
    triplets = []
    lines = content.strip().split('\n')
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        try:
            triplet_data = json.loads(line)
            
            # 基本验证
            required_fields = ['head', 'relation_id', 'tail', 'confidence']
            if all(field in triplet_data for field in required_fields):
                triplets.append(triplet_data)
            else:
                print(f"⚠️ Line {line_num}: 缺少必需字段")
                
        except json.JSONDecodeError as e:
            print(f"⚠️ Line {line_num}: JSON 解析错误: {e}")
            continue
        except Exception as e:
            print(f"⚠️ Line {line_num}: 其他错误: {e}")
            continue
    
    return triplets


def build_qa_graph(seeds, target_nodes=50, language="zh"):
    """构建 QA-Atomic 图谱"""
    print(f"🚀 开始构建 QA-Atomic 图谱")
    print(f"🌱 种子实体: {seeds}")
    print(f"🎯 目标节点数: {target_nodes}")
    
    # 初始化 API
    if not load_api_key('keys/openai.txt'):
        raise RuntimeError("API 初始化失败")
    
    all_triplets = []
    unique_nodes = set(seeds)
    iteration = 0
    max_iterations = 5
    
    current_seeds = seeds.copy()
    
    while len(unique_nodes) < target_nodes and iteration < max_iterations:
        iteration += 1
        remaining_nodes = target_nodes - len(unique_nodes)
        budget = min(remaining_nodes * 2, 30)  # 生成更多三元组
        
        print(f"\n--- 第 {iteration} 轮迭代 ---")
        print(f"当前节点数: {len(unique_nodes)}, 目标: {target_nodes}")
        print(f"本轮种子: {current_seeds[:3]}{'...' if len(current_seeds) > 3 else ''}")
        
        # 创建 prompt
        user_prompt = create_qa_atomic_prompt(current_seeds, budget, language)
        
        # 调用 LLM
        print(f"🔍 生成 {budget} 条三元组...")
        content = _call_llm_with_cache(
            prompt=user_prompt,
            system_prompt=SYS_PROMPT_GRAPH_BUILDER_v0_3,
            temperature=0.2,
            max_tokens=4000
        )
        
        if not content:
            print("❌ LLM 无响应")
            break
        
        # 解析响应
        new_triplets = parse_jsonl_response(content)
        
        if not new_triplets:
            print("⚠️ 未生成有效三元组")
            break
        
        # 添加新三元组和节点
        iteration_new_nodes = set()
        for triplet in new_triplets:
            all_triplets.append(triplet)
            head = triplet['head']
            tail = triplet['tail']
            
            if head not in unique_nodes:
                iteration_new_nodes.add(head)
            if tail not in unique_nodes:
                iteration_new_nodes.add(tail)
            
            unique_nodes.add(head)
            unique_nodes.add(tail)
        
        print(f"✅ 本轮: {len(new_triplets)} 条三元组, {len(iteration_new_nodes)} 个新节点")
        
        # 准备下一轮种子
        if iteration_new_nodes:
            current_seeds = list(iteration_new_nodes)[:8]  # 取前8个新节点作为种子
        else:
            # 如果没有新节点，从现有节点中选择
            current_seeds = list(unique_nodes)[-8:]
        
        if len(unique_nodes) >= target_nodes:
            break
    
    print(f"\n🎉 图谱构建完成!")
    print(f"📊 最终统计: {len(unique_nodes)} 个节点, {len(all_triplets)} 条三元组")
    print(f"🔄 迭代次数: {iteration}")
    
    return all_triplets, unique_nodes


def main():
    """主函数"""
    try:
        # 种子实体
        seeds = ["北京", "苹果公司", "爱因斯坦"]
        
        # 构建图谱
        triplets, nodes = build_qa_graph(seeds, target_nodes=50, language="zh")
        
        # 显示一些示例
        print(f"\n📋 示例三元组 (前5条):")
        for i, triplet in enumerate(triplets[:5], 1):
            print(f"{i}. ({triplet['head']}, {triplet['relation_id']}, {triplet['tail']})")
            print(f"   置信度: {triplet['confidence']:.2f}")
            if 'surface' in triplet:
                print(f"   表面形式: {triplet['surface']}")
            if triplet.get('qualifiers'):
                print(f"   限定词: {triplet['qualifiers']}")
            print()
        
        # 统计关系分布
        relation_counts = {}
        for triplet in triplets:
            rel = triplet['relation_id']
            relation_counts[rel] = relation_counts.get(rel, 0) + 1
        
        print(f"📈 关系分布 (前10个):")
        for rel, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {rel}: {count} 条")
        
        # 保存为 JSON
        output_data = {
            'metadata': {
                'total_nodes': len(nodes),
                'total_triplets': len(triplets),
                'seeds': seeds,
                'target_nodes': 50,
                'language': 'zh'
            },
            'nodes': list(nodes),
            'triplets': triplets,
            'relation_distribution': relation_counts
        }
        
        output_file = 'qa_atomic_graph_50nodes.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 图谱已保存到: {output_file}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
