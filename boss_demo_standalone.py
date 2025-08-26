#!/usr/bin/env python3
"""
Boss Demo: Standalone Prompt Output Display
展示纯净的 LLM 输出结果 - 完全独立版本
"""

import json
from openai import OpenAI

# 读取最新的prompt和关系定义
def load_latest_prompts_and_relations():
    """从文件中加载最新的prompt和关系定义"""
    
    # 读取系统prompt
    with open('graph_builder/prompts.py', 'r', encoding='utf-8') as f:
        content = f.read()
        # 提取系统prompt
        start = content.find('SYS_PROMPT_GRAPH_BUILDER_v0_3 = """') + len('SYS_PROMPT_GRAPH_BUILDER_v0_3 = """')
        end = content.find('"""', start)
        sys_prompt = content[start:end]
    
    # 读取关系定义
    with open('graph_builder/relations_qa.json', 'r', encoding='utf-8') as f:
        relations = json.load(f)
    
    return sys_prompt, relations

# 从文件加载最新内容
SYS_PROMPT, RELATIONS_DATA = load_latest_prompts_and_relations()

def create_dynamic_user_prompt(seed_entity="Apple Inc.", budget=5):
    """动态生成用户prompt，使用最新的关系定义"""
    
    # 格式化关系列表
    graph_core_relations = []
    function_relations = []
    auto_inverse_policies = []
    
    for rel in RELATIONS_DATA:
        rel_id = rel['relation_id']
        group = rel['group']
        domain = rel['domain']
        range_val = rel['range']
        
        # 格式化关系
        formatted_rel = f'  "{rel_id}|{group}|{domain}->{range_val}"'
        graph_core_relations.append(formatted_rel)
        
        # 添加到功能性关系列表（所有我们的关系都是功能性的）
        function_relations.append(f'  "{rel_id}"')
        
        # 处理auto_inverse_policy
        if rel['inverse_policy'] == 'auto':
            auto_inverse_policies.append(f'  "{rel_id}": "auto-inverse: Has{rel_id}ed"')
    
    # 构建用户prompt
    user_prompt = f"""### Seeds
SEEDS = ["{seed_entity}"]

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
    
    return user_prompt

def load_api_key():
    """加载OpenAI API密钥"""
    try:
        with open('keys/openai.txt', 'r') as f:
            api_key = f.read().strip()
        return api_key
    except FileNotFoundError:
        print("❌ API key file not found: keys/openai.txt")
        return None

def main():
    print("📋 BOSS DEMO: Pure Prompt Output (Latest v0.3)")
    print("=" * 60)
    print("System: Knowledge Graph Builder v0.3 (Function-like Relations)")
    print("Seed Entity: Apple Inc.")
    print("Output Format: JSONL (one JSON object per line)")
    print(f"Relations: {len(RELATIONS_DATA)} function-like relations loaded")
    print("Improvements: ✅ No PopulationAsOf ✅ No 'atomic' terminology")
    print()
    
    # 加载API密钥
    api_key = load_api_key()
    if not api_key:
        print("❌ Cannot proceed without API key")
        return
    
    # 动态生成最新的用户prompt
    user_prompt = create_dynamic_user_prompt("Apple Inc.", 6)
    
    print("🤖 Calling LLM with our latest v0.3 prompt system...")
    print("-" * 60)
    
    try:
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        raw_output = response.choices[0].message.content
        
        # 显示原始输出
        print("RAW LLM OUTPUT:")
        print(raw_output)
        print()
        print("-" * 50)
        print("💡 This is exactly what our LLM generates with the v0.3 prompt")
        print("📝 Each line is a knowledge triplet in JSON format")
        
        # 保存给boss查看的文件
        with open('BOSS_LATEST_OUTPUT_DEMO_v0.3.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("BOSS DEMO: Latest v0.3 Prompt System Output\n")
            f.write("=" * 70 + "\n\n")
            f.write("System: Knowledge Graph Builder v0.3 (Function-like Relations)\n")
            f.write("Seed: Apple Inc.\n")
            f.write("Language: English\n")
            f.write("Budget: 6 triplets\n")
            f.write(f"Relations Used: {len(RELATIONS_DATA)} function-like relations\n")
            f.write("Key Improvements:\n")
            f.write("  ✅ Removed PopulationAsOf (low-value number relations)\n")
            f.write("  ✅ Replaced 'atomic' terminology with 'function-like'\n")
            f.write("  ✅ Enhanced relation quality (MajorIndustryPrimary, etc.)\n")
            f.write("  ✅ Consistent prompt system across all components\n\n")
            f.write("Raw JSONL Output from LLM:\n")
            f.write("-" * 50 + "\n")
            f.write(raw_output)
            f.write("\n\n")
            f.write("Output Format Explanation:\n")
            f.write("-" * 40 + "\n")
            f.write("• Each line = one knowledge triplet in JSON format\n")
            f.write("• 'head' → 'relation_id' → 'tail' = basic knowledge fact\n")
            f.write("• 'confidence': reliability score (0.0-1.0, ≥0.6 for standard facts)\n")
            f.write("• 'surface': natural language sentence (no schema terms)\n")
            f.write("• 'qa_eligible': true if suitable for Q&A systems\n")
            f.write("• 'qualifiers': context for uniqueness (current, primary, as_of_year)\n")
            f.write("• 'evidence_rationale': justification based on world knowledge\n")
            f.write("• 'group': relation category for organization\n")
            f.write("• 'domain_type' → 'range_type': semantic type constraints\n\n")
            f.write("Technical Notes:\n")
            f.write("-" * 20 + "\n")
            f.write("• All relations are function-like (yield unique answers)\n")
            f.write("• Qualifiers ensure uniqueness for multi-valued concepts\n")
            f.write("• Auto-inverse relations handled by system pipeline\n")
            f.write("• Schema validation ensures consistent output format\n")
        
        print(f"\n✅ Latest demo saved to: BOSS_LATEST_OUTPUT_DEMO_v0.3.txt")
        print("📧 You can send this file to your boss - it shows our improved v0.3 system!")
        print("🎯 Key highlights: No PopulationAsOf, function-like terminology, enhanced relations")
        
    except Exception as e:
        print(f"❌ Error calling LLM: {e}")

if __name__ == "__main__":
    main()
