#!/usr/bin/env python3
"""
独立的 QA-Atomic 图谱构建测试
包含所有必要代码，无需复杂导入
"""

import json
import hashlib
import os
from openai import OpenAI

# QA-Atomic 关系定义 (36个关系)
QA_ATOMIC_RELATIONS = [
    {"relation_id":"BirthDate","group":"Person","domain":"Person","range":"Time","multiplicity":"one_or_zero","qualifiers_required":[],"inverse_policy":"none"},
    {"relation_id":"BirthPlace","group":"Person","domain":"Person","range":"Place","multiplicity":"one_or_zero","qualifiers_required":[],"inverse_policy":"none"},
    {"relation_id":"NationalityPrimary","group":"Person","domain":"Person","range":"Country","multiplicity":"one_or_zero","qualifiers_required":["primary"],"inverse_policy":"none"},
    {"relation_id":"CurrentPosition","group":"Person","domain":"Person","range":"Position","multiplicity":"one_or_zero","qualifiers_required":["current"],"inverse_policy":"none"},
    {"relation_id":"CurrentEmployer","group":"Person","domain":"Person","range":"Org","multiplicity":"one_or_zero","qualifiers_required":["current"],"inverse_policy":"auto"},
    {"relation_id":"AlmaMaterPrimary","group":"Education","domain":"Person","range":"Org","multiplicity":"one_or_zero","qualifiers_required":["primary"],"inverse_policy":"none"},

    {"relation_id":"HeadquartersCity","group":"Org","domain":"Org","range":"City","multiplicity":"one_or_zero","qualifiers_required":["primary"],"inverse_policy":"paired"},
    {"relation_id":"HeadquartersCountry","group":"Org","domain":"Org","range":"Country","multiplicity":"one_or_zero","qualifiers_required":["primary"],"inverse_policy":"none"},
    {"relation_id":"FoundingDate","group":"Org","domain":"Org","range":"Time","multiplicity":"one_or_zero","qualifiers_required":[],"inverse_policy":"none"},
    {"relation_id":"FoundedByPrimary","group":"Org","domain":"Org","range":"PersonOrOrg","multiplicity":"one_or_zero","qualifiers_required":["primary"],"inverse_policy":"auto"},
    {"relation_id":"ParentOrganization","group":"Org","domain":"Org","range":"Org","multiplicity":"one_or_zero","qualifiers_required":[],"inverse_policy":"auto"},
    {"relation_id":"ChiefExecutiveOfficerCurrent","group":"Org","domain":"Org","range":"Person","multiplicity":"one_or_zero","qualifiers_required":["current"],"inverse_policy":"none"},
    {"relation_id":"CountryOfIncorporation","group":"Org","domain":"Org","range":"Country","multiplicity":"one_or_zero","qualifiers_required":[],"inverse_policy":"none"},
    {"relation_id":"StockExchangePrimary","group":"Org","domain":"Org","range":"Exchange","multiplicity":"one_or_zero","qualifiers_required":["primary"],"inverse_policy":"none"},

    {"relation_id":"CountryOfCity","group":"Geo","domain":"City","range":"Country","multiplicity":"one_or_zero","qualifiers_required":[],"inverse_policy":"none"},
    {"relation_id":"CapitalCityOfCountry","group":"Geo","domain":"Country","range":"City","multiplicity":"one_or_zero","qualifiers_required":["single_capital_only"],"inverse_policy":"auto"},
    {"relation_id":"TimeZonePrimary","group":"Geo","domain":"Place","range":"TimeZone","multiplicity":"one_or_zero","qualifiers_required":["primary"],"inverse_policy":"none"},
    {"relation_id":"PopulationAsOf","group":"Geo","domain":"Place","range":"Number","multiplicity":"one_or_zero","qualifiers_required":["as_of_year"],"inverse_policy":"none"},
    {"relation_id":"OfficialLanguagePrimary","group":"Geo","domain":"Country","range":"Language","multiplicity":"one_or_zero","qualifiers_required":["primary"],"inverse_policy":"none"},
    {"relation_id":"CurrencyPrimary","group":"Geo","domain":"Country","range":"Currency","multiplicity":"one_or_zero","qualifiers_required":["primary"],"inverse_policy":"none"},
    {"relation_id":"Continent","group":"Geo","domain":"Country","range":"Continent","multiplicity":"one_or_zero","qualifiers_required":[],"inverse_policy":"none"},

    {"relation_id":"AuthorOfWorkPrimary","group":"Work","domain":"Work","range":"Person","multiplicity":"one_or_zero","qualifiers_required":["primary"],"inverse_policy":"auto"},
    {"relation_id":"CreatedByPrimary","group":"Work","domain":"WorkOrInvention","range":"PersonOrOrg","multiplicity":"one_or_zero","qualifiers_required":["primary"],"inverse_policy":"auto"},
    {"relation_id":"PublicationDate","group":"Work","domain":"Work","range":"Time","multiplicity":"one_or_zero","qualifiers_required":[],"inverse_policy":"none"},
    {"relation_id":"PublisherPrimary","group":"Work","domain":"Work","range":"Org","multiplicity":"one_or_zero","qualifiers_required":["primary"],"inverse_policy":"none"},
    {"relation_id":"LanguageOfWorkPrimary","group":"Work","domain":"Work","range":"Language","multiplicity":"one_or_zero","qualifiers_required":["primary"],"inverse_policy":"none"},
    {"relation_id":"SeriesOfWorkPrimary","group":"Work","domain":"Work","range":"Series","multiplicity":"one_or_zero","qualifiers_required":["primary"],"inverse_policy":"none"},

    {"relation_id":"DevelopedByPrimary","group":"Product/Tech","domain":"SoftwareOrSystem","range":"OrgOrPerson","multiplicity":"one_or_zero","qualifiers_required":["primary"],"inverse_policy":"auto"},
    {"relation_id":"ManufacturedByPrimary","group":"Product/Tech","domain":"Product","range":"Org","multiplicity":"one_or_zero","qualifiers_required":["primary"],"inverse_policy":"auto"},
    {"relation_id":"InitialReleaseDate","group":"Product/Tech","domain":"SoftwareOrProduct","range":"Time","multiplicity":"one_or_zero","qualifiers_required":[],"inverse_policy":"none"},
    {"relation_id":"ProgrammingLanguagePrimary","group":"Product/Tech","domain":"Software","range":"Language","multiplicity":"one_or_zero","qualifiers_required":["primary"],"inverse_policy":"none"},
    {"relation_id":"LicensePrimary","group":"Product/Tech","domain":"Software","range":"License","multiplicity":"one_or_zero","qualifiers_required":["primary"],"inverse_policy":"none"},
    {"relation_id":"OperatingSystemPrimary","group":"Product/Tech","domain":"Software","range":"OperatingSystem","multiplicity":"one_or_zero","qualifiers_required":["primary"],"inverse_policy":"none"},

    {"relation_id":"OccursOn","group":"Event","domain":"Event","range":"Time","multiplicity":"one_or_zero","qualifiers_required":[],"inverse_policy":"none"},
    {"relation_id":"HeldInCity","group":"Event","domain":"Event","range":"City","multiplicity":"one_or_zero","qualifiers_required":[],"inverse_policy":"none"},
    {"relation_id":"HostOrganizationPrimary","group":"Event","domain":"Event","range":"Org","multiplicity":"one_or_zero","qualifiers_required":["primary"],"inverse_policy":"none"}
]

# System Prompt
SYS_PROMPT = """### Task
You are an expert knowledge-graph builder. From given seed entities, generate high-precision
triples using ONLY the provided canonical relation inventory. Produce edges that maximize
local closure (triangles) while preserving correctness. Prefer function-like relations
("QA-Atomic") when they can be uniquely determined; otherwise output Graph-Core relations
that are still unambiguous.

### Rules
1) Canonicalization only:
   - Use ONLY relation_id from GRAPH_CORE_RELATIONS.
   - Do NOT invent new relation names. Do NOT output inverse edges explicitly if policy is auto-inverse.
2) Type safety:
   - Each triple must satisfy domain→range of the chosen relation.
3) Uniqueness policy:
   - If relation is QA-Atomic but has multiple plausible tails, add qualifiers to uniquely pin it down
     (e.g., current=true, primary=true, as_of_year=YYYY). If still non-unique, SKIP it.
4) Evidence & confidence:
   - Provide a brief evidence_rationale (1–2 short sentences) grounded in general world knowledge;
     avoid speculation. Assign confidence in [0.0, 1.0]. Use ≥0.60 only if the fact is standard.
5) Density & closure:
   - Prefer triples that create short cycles/triangles among seeds and newly proposed nodes.
   - Avoid duplicate (head, relation, tail). Avoid trivial aliases (map them to canonical).
6) Output determinism:
   - Deterministic, precise wording. No vague terms. No schema leakage in surface text.
   - LANGUAGE governs the natural-language "surface" field only; all other fields in English.
7) Budget & balance:
   - Respect BUDGET. Aim for a balanced mix: prioritize QA-Atomic edges first (unique),
     then safe Graph-Core edges that improve clustering.
8) Self-check before finalizing:
   - Remove duplicates; enforce domain/range; enforce uniqueness for QA-Atomic;
     ensure no inverse edges for auto-inverse relations.
   - If uncertain, lower confidence or drop the triple.

### Output Format (JSON Lines; one object per line)
Each line MUST validate this schema:

{
  "head": "<string>",
  "relation_id": "<canonical from inventory>",
  "tail": "<string>",
  "group": "<group name from inventory>",
  "domain_type": "<one of: Person|Org|Place|Class|Event|Work|Software|Product|Material|Language|Time|Number|...>",
  "range_type": "<same type system>",
  "qualifiers": { "current": <bool?>, "primary": <bool?>, "as_of_year": <int?> },
  "qa_eligible": <bool>,
  "surface": "<LANGUAGE natural sentence expressing the fact (no schema terms)>",
  "evidence_rationale": "<<=2 short sentences>",
  "confidence": <float in [0,1]>,
  "is_inverse": false
}

Return ONLY JSONL lines. No extra commentary."""

# Global client
client = None
response_cache = {}

def init_api():
    """初始化 OpenAI API"""
    global client
    try:
        with open('keys/openai.txt', 'r') as f:
            api_key = f.read().strip()
        client = OpenAI(api_key=api_key)
        return True
    except Exception as e:
        print(f"API 初始化失败: {e}")
        return False

def call_llm(prompt, system_prompt, temperature=0.2, max_tokens=4000):
    """调用 LLM"""
    global client
    if client is None:
        return None
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        return None

def create_prompt(seeds, budget=30, language="zh"):
    """创建 QA-Atomic prompt"""
    # 格式化关系列表
    relation_lines = []
    qa_atomic_list = []
    
    for rel_data in QA_ATOMIC_RELATIONS:
        rel_id = rel_data['relation_id']
        group = rel_data['group']
        domain = rel_data['domain']
        range_val = rel_data['range']
        
        formatted_rel = f'  "{rel_id}|{group}|{domain}->{range_val}"'
        relation_lines.append(formatted_rel)
        qa_atomic_list.append(f'  "{rel_id}"')
    
    # 格式化逆关系策略
    policy_lines = []
    for rel_data in QA_ATOMIC_RELATIONS:
        rel_id = rel_data['relation_id']
        policy = rel_data.get('inverse_policy', 'none')
        if policy == 'auto':
            policy_lines.append(f'  "{rel_id}": "auto-inverse: InverseOf{rel_id}"')
    
    return f"""### Seeds
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

### Constraints
LANGUAGE = "{language}"
BUDGET = {budget}

### Your Output
Return up to BUDGET JSONL objects strictly following the schema. ALL relations are QA-Atomic,
so prioritize uniqueness with qualifiers. Focus on factually verifiable, unique answers."""

def parse_response(content):
    """解析 JSONL 响应"""
    if not content:
        return []
    
    triplets = []
    lines = content.strip().split('\n')
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('```'):
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
            print(f"⚠️ Line {line_num}: JSON 解析错误")
            continue
        except Exception as e:
            print(f"⚠️ Line {line_num}: 其他错误: {e}")
            continue
    
    return triplets

def build_graph(seeds, target_nodes=50, language="zh"):
    """构建图谱"""
    print(f"🚀 开始构建 QA-Atomic 图谱")
    print(f"🌱 种子: {seeds}")
    print(f"🎯 目标: {target_nodes} 个节点")
    
    if not init_api():
        return [], set()
    
    all_triplets = []
    unique_nodes = set(seeds)
    iteration = 0
    max_iterations = 5
    
    current_seeds = seeds.copy()
    
    while len(unique_nodes) < target_nodes and iteration < max_iterations:
        iteration += 1
        remaining = target_nodes - len(unique_nodes)
        budget = min(remaining * 2, 25)  # 每次生成最多25条
        
        print(f"\n--- 第 {iteration} 轮 ---")
        print(f"当前: {len(unique_nodes)} 个节点, 目标: {target_nodes}")
        print(f"种子: {current_seeds[:3]}{'...' if len(current_seeds) > 3 else ''}")
        
        # 生成 prompt 和调用 LLM
        prompt = create_prompt(current_seeds, budget, language)
        print(f"🔍 生成 {budget} 条三元组...")
        
        content = call_llm(prompt, SYS_PROMPT)
        if not content:
            print("❌ LLM 无响应")
            break
        
        # 解析结果
        new_triplets = parse_response(content)
        if not new_triplets:
            print("⚠️ 未解析到有效三元组")
            break
        
        # 统计新节点
        iteration_new_nodes = set()
        for triplet in new_triplets:
            all_triplets.append(triplet)
            head, tail = triplet['head'], triplet['tail']
            
            if head not in unique_nodes:
                iteration_new_nodes.add(head)
            if tail not in unique_nodes:
                iteration_new_nodes.add(tail)
            
            unique_nodes.add(head)
            unique_nodes.add(tail)
        
        print(f"✅ 本轮: {len(new_triplets)} 条三元组, {len(iteration_new_nodes)} 个新节点")
        
        # 准备下轮种子
        if iteration_new_nodes:
            current_seeds = list(iteration_new_nodes)[:6]
        else:
            current_seeds = list(unique_nodes)[-6:]
        
        if len(unique_nodes) >= target_nodes:
            break
    
    print(f"\n🎉 构建完成: {len(unique_nodes)} 个节点, {len(all_triplets)} 条三元组")
    return all_triplets, unique_nodes

def main():
    """主函数"""
    try:
        # 种子实体
        seeds = ["北京", "苹果公司", "爱因斯坦"]
        
        # 构建图谱
        triplets, nodes = build_graph(seeds, target_nodes=50, language="zh")
        
        if not triplets:
            print("❌ 未生成任何三元组")
            return
        
        # 显示示例
        print(f"\n📋 示例三元组 (前3条):")
        for i, triplet in enumerate(triplets[:3], 1):
            print(f"{i}. ({triplet['head']}, {triplet['relation_id']}, {triplet['tail']})")
            print(f"   置信度: {triplet.get('confidence', 0):.2f}")
            if 'surface' in triplet:
                print(f"   表述: {triplet['surface']}")
            print()
        
        # 统计
        relation_counts = {}
        for triplet in triplets:
            rel = triplet['relation_id']
            relation_counts[rel] = relation_counts.get(rel, 0) + 1
        
        print(f"📈 关系分布:")
        for rel, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:8]:
            print(f"  {rel}: {count} 条")
        
        # 保存结果
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
        
        output_file = 'qa_atomic_graph_result.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
