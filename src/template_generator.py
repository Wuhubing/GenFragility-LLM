#!/usr/bin/env python3
"""
AI Template Engineer: 使用GPT-4o mini为关系类型生成高质量的传统模板。

这个脚本执行一次性的模板生成任务，其输出可以被复制到配置文件中，
用于大规模、高性能的知识三元组Probing。
"""
import openai
import json
import re
from typing import List, Dict

def get_openai_client():
    """获取OpenAI客户端"""
    with open("openai.txt", "r") as f:
        openai_key = f.read().strip()
    return openai.OpenAI(api_key=openai_key)

def generate_templates_for_relation(client: openai.OpenAI, relation: str, num_templates: int = 5) -> List[str]:
    """为单个关系类型生成高质量模板"""
    print(f"🧬 Generating templates for relation: '{relation}'...")

    system_prompt = """You are an expert in Natural Language Processing and prompt engineering. 
Your task is to generate a set of high-quality, reusable English sentence templates for a given knowledge relation.

Key Requirements:
1.  **Use Placeholders**: Each template MUST contain the placeholders `{head}` and `{tail}`.
2.  **Natural Phrasing**: The sentences should be natural and diverse. Avoid overly robotic or simple structures.
3.  **Syntactic Correctness**: The template must be grammatically correct.
4.  **Tail Position**: The `{tail}` placeholder should appear towards the end of the sentence.
5.  **Completeness**: The template should form a complete sentence or a clause that naturally leads to the tail.

Example for relation 'capital_of':
- Good: `The capital of {head} is {tail}.`
- Good: `{head} has its capital city in {tail}.`
- Bad: `{head} {tail}` (not a sentence)
- Bad: `The capital is {tail}.` (missing {head})

Return ONLY a JSON list of strings, with no other text or explanation.
"""

    user_prompt = f"Please generate {num_templates} diverse templates for the relation type: **`{relation}`**."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500,
        )
        
        # 确保返回的是JSON对象，并且包含一个template列表
        content = response.choices[0].message.content
        if not content:
            print("   ⚠️ API returned empty content.")
            return []
            
        generated_data = json.loads(content)
        templates = generated_data.get("templates", [])
        if not isinstance(templates, list):
             # 尝试作为value的第一个列表，以防万一
             if generated_data and isinstance(list(generated_data.values())[0], list):
                templates = list(generated_data.values())[0]
             else:
                templates = []

        print(f"   ✅ Received {len(templates)} raw templates from API.")
        return templates

    except Exception as e:
        print(f"   ❌ Error generating templates for '{relation}': {e}")
        return []

def validate_and_clean_templates(templates: List[str]) -> List[str]:
    """验证并清理生成的模板"""
    cleaned_templates = []
    for tpl in templates:
        # 1. 必须是字符串
        if not isinstance(tpl, str):
            continue
            
        # 2. 必须包含 {head} 和 {tail}
        if "{head}" not in tpl or "{tail}" not in tpl:
            print(f"   ⚠️  Rejected (missing placeholder): {tpl}")
            continue

        # 3. 移除不必要的空格
        cleaned = re.sub(r'\s+', ' ', tpl).strip()
        
        # 4. 确保不在黑名单中
        if cleaned in cleaned_templates:
            continue
            
        cleaned_templates.append(cleaned)
    
    print(f"   👍 Validated {len(cleaned_templates)} high-quality templates.")
    return cleaned_templates

def main():
    """主函数，生成并输出模板"""
    
    # 我们需要模板的关系类型
    relations_to_generate = [
        "capital_of",
        "place_of_birth",
        "occupation",
        "educated_at",
        "country_of_citizenship",
        "author",
        "genre",
        "located_in_administrative_entity",
    ]
    
    print("🚀 Starting AI Template Engineer...")
    client = get_openai_client()
    
    all_generated_templates = {}

    for relation in relations_to_generate:
        raw_templates = generate_templates_for_relation(client, relation, num_templates=7)
        valid_templates = validate_and_clean_templates(raw_templates)
        
        # 我们每个关系取最多5个最佳模板
        all_generated_templates[relation] = valid_templates[:5]
        print("-" * 50)

    # 格式化为可直接粘贴到 config.json 的格式
    output_json = json.dumps(all_generated_templates, indent=2)
    
    print("\n\n✅ Template generation complete!")
    print("📋 Below are the generated templates. Copy and paste this into your `config.json`'s `relation_templates` section.")
    print("-" * 80)
    print(output_json)
    print("-" * 80)
    
    # 保存到文件以便使用
    output_filename = "generated_templates.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(output_json)
    print(f"\n💾 Templates also saved to `{output_filename}` for convenience.")


if __name__ == "__main__":
    main() 