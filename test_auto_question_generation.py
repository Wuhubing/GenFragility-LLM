#!/usr/bin/env python3
"""
测试自动问题生成功能
在图谱生成时同时生成对应的问题
"""

import json
from openai import OpenAI

def load_system_prompt():
    """加载更新后的系统prompt（包含问题生成）"""
    with open('graph_builder/prompts.py', 'r', encoding='utf-8') as f:
        content = f.read()
        start = content.find('SYS_PROMPT_GRAPH_BUILDER_v0_3 = """') + len('SYS_PROMPT_GRAPH_BUILDER_v0_3 = """')
        end = content.find('"""', start)
        return content[start:end]

def load_relations():
    """加载关系定义"""
    with open('graph_builder/relations_qa.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def create_user_prompt_with_question_gen(seeds, budget=6):
    """创建包含问题生成要求的用户prompt"""
    relations = load_relations()
    
    # 格式化关系
    graph_core_relations = []
    function_relations = []
    
    for rel in relations:
        rel_id = rel['relation_id']
        group = rel['group']
        domain = rel['domain']
        range_val = rel['range']
        
        formatted_rel = f'  "{rel_id}|{group}|{domain}->{range_val}"'
        graph_core_relations.append(formatted_rel)
        function_relations.append(f'  "{rel_id}"')
    
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
  "CurrentEmployer": "auto-inverse: HasEmployee"
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
(ensure uniqueness with qualifiers), then safe Graph-Core edges that improve closure.

IMPORTANT: Each triplet MUST include a "question" field with a simple, direct question that expects the tail as answer."""

def load_api_key():
    """加载API密钥"""
    try:
        with open('keys/openai.txt', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print("❌ API key file not found: keys/openai.txt")
        return None

def test_auto_question_generation():
    """测试自动问题生成功能"""
    print("🧪 Testing Auto Question Generation in Graph Building")
    print("=" * 60)
    
    api_key = load_api_key()
    if not api_key:
        return
    
    client = OpenAI(api_key=api_key)
    system_prompt = load_system_prompt()
    
    # 使用简单种子测试
    seeds = ['Albert Einstein', 'Apple Inc.']
    user_prompt = create_user_prompt_with_question_gen(seeds, budget=6)
    
    print(f"🌱 Testing with seeds: {seeds}")
    print("🤖 Calling LLM with enhanced prompt (includes question generation)...")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        raw_output = response.choices[0].message.content
        print("\n📋 Raw LLM Output:")
        print("-" * 40)
        print(raw_output)
        
        # 解析JSONL输出
        print("\n🔍 Parsing and Analyzing Results:")
        print("-" * 40)
        
        triplets = []
        lines = raw_output.strip().split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                triplet = json.loads(line)
                triplets.append(triplet)
                
                print(f"\n{line_num}. TRIPLET:")
                print(f"   Fact: {triplet['head']} --{triplet['relation_id']}--> {triplet['tail']}")
                print(f"   Surface: \"{triplet['surface']}\"")
                if 'question' in triplet:
                    print(f"   🎯 Question: \"{triplet['question']}\"")
                    print(f"   ✅ Expected Answer: {triplet['tail']}")
                else:
                    print(f"   ❌ Missing question field!")
                print(f"   Confidence: {triplet['confidence']:.2f}")
                
            except json.JSONDecodeError as e:
                print(f"⚠️ Line {line_num}: JSON decode error: {e}")
                continue
        
        # 统计分析
        print(f"\n📊 Analysis Summary:")
        print(f"  Total triplets: {len(triplets)}")
        
        with_questions = len([t for t in triplets if 'question' in t and t['question']])
        print(f"  With questions: {with_questions}/{len(triplets)} ({with_questions/len(triplets)*100:.1f}%)")
        
        # 问题质量分析
        if with_questions > 0:
            print(f"\n🎯 Question Quality Check:")
            for i, t in enumerate([t for t in triplets if 'question' in t and t['question']][:5]):
                question = t['question']
                tail = t['tail']
                
                # 简单质量检查
                quality_checks = []
                if len(question.split()) <= 15:
                    quality_checks.append("✅ Length good")
                else:
                    quality_checks.append("❌ Too long")
                
                if question.endswith('?'):
                    quality_checks.append("✅ Question format")
                else:
                    quality_checks.append("❌ No question mark")
                
                if tail.lower() not in question.lower():
                    quality_checks.append("✅ No answer leak")
                else:
                    quality_checks.append("❌ Answer leaked")
                
                print(f"  {i+1}. \"{question}\" → {tail}")
                print(f"     {', '.join(quality_checks)}")
        
        # 保存结果
        result = {
            'metadata': {
                'test_type': 'auto_question_generation',
                'seeds': seeds,
                'total_triplets': len(triplets),
                'triplets_with_questions': with_questions
            },
            'triplets': triplets
        }
        
        with open('test_auto_questions.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Test completed! Results saved to: test_auto_questions.json")
        
        if with_questions == len(triplets) and with_questions > 0:
            print("🎉 SUCCESS: All triplets include auto-generated questions!")
        else:
            print("⚠️ Some triplets are missing questions - may need prompt refinement")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_auto_question_generation()
