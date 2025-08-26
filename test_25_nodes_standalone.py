#!/usr/bin/env python3
"""
测试25个节点的函数性图谱生成 - 独立版本
"""

import json
import time
from openai import OpenAI

def load_relations():
    """加载关系定义"""
    with open('graph_builder/relations_qa.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def load_system_prompt():
    """加载系统prompt"""
    with open('graph_builder/prompts.py', 'r', encoding='utf-8') as f:
        content = f.read()
        start = content.find('SYS_PROMPT_GRAPH_BUILDER_v0_3 = """') + len('SYS_PROMPT_GRAPH_BUILDER_v0_3 = """')
        end = content.find('"""', start)
        return content[start:end]

def create_user_prompt(seeds, budget=30):
    """创建用户prompt"""
    relations = load_relations()
    
    # 格式化关系
    graph_core_relations = []
    function_relations = []
    auto_inverse_policies = []
    
    for rel in relations:
        rel_id = rel['relation_id']
        group = rel['group']
        domain = rel['domain']
        range_val = rel['range']
        
        formatted_rel = f'  "{rel_id}|{group}|{domain}->{range_val}"'
        graph_core_relations.append(formatted_rel)
        function_relations.append(f'  "{rel_id}"')
        
        if rel['inverse_policy'] == 'auto':
            auto_inverse_policies.append(f'  "{rel_id}": "auto-inverse: Has{rel_id}ed"')
    
    return f"""### Seeds
SEEDS = {seeds}

### Relation Inventories
GRAPH_CORE_RELATIONS = [
{chr(10).join(graph_core_relations)}
]

FUNCTION_RELATIONS = [
{chr(10).join(function_relations)}
]

AUTO_INVERSE_POLICY = {{
{chr(10).join(auto_inverse_policies)}
}}

### Qualifier Rules (Function-like Relations)
- CurrentEmployer / CurrentPosition / CEO: require qualifiers.current = true
- Nationality / AlmaMater / Language / Industry / Currency / TimeZone: if multiple, require qualifiers.primary = true
- CapitalOf: allowed only for single-capital countries (skip multi-capital cases)
- Date-based relations: use specific years when relevant for temporal context

### Constraints
LANGUAGE = "en"
BUDGET = {budget}

### Your Output
Return up to BUDGET JSONL objects strictly following the schema. Favor function-like edges first
(ensure uniqueness with qualifiers), then safe Graph-Core edges that improve closure."""

def load_api_key():
    """加载API密钥"""
    try:
        with open('keys/openai.txt', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print("❌ API key file not found: keys/openai.txt")
        return None

def call_llm(system_prompt, user_prompt):
    """调用LLM"""
    api_key = load_api_key()
    if not api_key:
        return None
    
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=3000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return None

def parse_triplets(content):
    """解析三元组"""
    if not content:
        return []
    
    triplets = []
    lines = content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        try:
            triplet = json.loads(line)
            triplets.append(triplet)
        except json.JSONDecodeError:
            continue
    
    return triplets

def build_graph_iteratively(seeds, target_nodes=25):
    """迭代构建图谱到目标节点数"""
    print(f"🎯 Building graph to {target_nodes} nodes")
    print(f"🌱 Initial seeds: {seeds}")
    
    system_prompt = load_system_prompt()
    all_triplets = []
    unique_nodes = set(seeds)
    iteration = 0
    max_iterations = 8
    
    current_seeds = seeds.copy()
    
    while len(unique_nodes) < target_nodes and iteration < max_iterations:
        iteration += 1
        remaining_nodes = target_nodes - len(unique_nodes)
        budget = min(remaining_nodes * 2, 25)
        
        print(f"\n--- Iteration {iteration} ---")
        print(f"Current nodes: {len(unique_nodes)}, Target: {target_nodes}")
        print(f"Seeds: {current_seeds[:4]}{'...' if len(current_seeds) > 4 else ''}")
        print(f"Budget: {budget} triplets")
        
        # 生成用户prompt
        user_prompt = create_user_prompt(current_seeds, budget)
        
        # 调用LLM
        content = call_llm(system_prompt, user_prompt)
        if not content:
            print("❌ No LLM response")
            break
        
        # 解析三元组
        new_triplets = parse_triplets(content)
        print(f"✅ Generated {len(new_triplets)} triplets")
        
        if not new_triplets:
            print("⚠️ No valid triplets, stopping")
            break
        
        # 收集新节点
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
        
        print(f"📊 Added {len(iteration_new_nodes)} new nodes")
        
        # 准备下一轮的种子
        if iteration_new_nodes:
            current_seeds = list(iteration_new_nodes)[:8]
        else:
            current_seeds = list(unique_nodes)[-8:]
        
        if len(unique_nodes) >= target_nodes:
            break
    
    print(f"\n🎉 Graph construction completed!")
    print(f"📊 Final: {len(unique_nodes)} nodes, {len(all_triplets)} triplets")
    print(f"🔄 Iterations: {iteration}")
    
    return all_triplets, unique_nodes

def main():
    print("🚀 Testing 25-Node Function-like Graph Generation")
    print("=" * 60)
    
    # 使用多样化的种子
    seeds = ['Apple Inc.', 'Albert Einstein', 'Beijing', 'Python']
    
    start_time = time.time()
    triplets, nodes = build_graph_iteratively(seeds, target_nodes=25)
    generation_time = time.time() - start_time
    
    print(f"\n⏱️ Generation time: {generation_time:.2f} seconds")
    
    # 分析结果
    print(f"\n📋 Sample Generated Triplets:")
    print("-" * 50)
    
    for i, triplet in enumerate(triplets[:8]):
        print(f"{i+1}. {triplet['head']} --{triplet['relation_id']}--> {triplet['tail']}")
        print(f"   Confidence: {triplet['confidence']:.2f}, Group: {triplet['group']}")
        if triplet.get('qualifiers'):
            print(f"   Qualifiers: {triplet['qualifiers']}")
        print()
    
    # 统计关系分布
    relation_counts = {}
    for t in triplets:
        rel = t['relation_id']
        relation_counts[rel] = relation_counts.get(rel, 0) + 1
    
    print("📈 Relation Distribution:")
    for rel, count in sorted(relation_counts.items()):
        print(f"  {rel}: {count}")
    
    # 保存结果
    result = {
        'metadata': {
            'total_nodes': len(nodes),
            'total_triplets': len(triplets),
            'generation_time': generation_time,
            'target_nodes': 25,
            'seeds': seeds
        },
        'triplets': triplets,
        'nodes': list(nodes),
        'relation_distribution': relation_counts
    }
    
    with open('test_25_nodes_en.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: test_25_nodes_en.json")
    print(f"🎯 Successfully generated {len(triplets)} triplets with {len(nodes)} nodes!")

if __name__ == "__main__":
    main()
