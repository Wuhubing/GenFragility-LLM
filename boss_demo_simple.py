#!/usr/bin/env python3
"""
Boss Demo: Simple Prompt Output Display
展示纯净的 LLM 输出结果
"""

import json
import os
import sys
from openai import OpenAI

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from graph_builder.prompts import SYS_PROMPT_GRAPH_BUILDER_v0_3

# 模拟一个简化的用户prompt（手动构建，避免依赖问题）
USER_PROMPT_DEMO = """### Seeds
SEEDS = ["Apple Inc."]

### Relation Inventories
GRAPH_CORE_RELATIONS = [
  "CurrentEmployer|Person|Person->Org",
  "FoundingDate|Org|Org->Time", 
  "CountryOfIncorporation|Org|Org->Country",
  "MajorIndustryPrimary|Geo|Country->Industry",
  "HeadquartersCity|Org|Org->City",
  "ChiefExecutiveOfficerCurrent|Org|Org->Person"
]

FUNCTION_RELATIONS = [
  "CurrentEmployer",
  "FoundingDate", 
  "CountryOfIncorporation",
  "MajorIndustryPrimary",
  "ChiefExecutiveOfficerCurrent"
]

AUTO_INVERSE_POLICY = {
  "CurrentEmployer": "auto-inverse: HasEmployee"
}

### Qualifier Rules (Function-like Relations)
- CurrentEmployer / CurrentPosition / CEO: require qualifiers.current = true
- Nationality / AlmaMater / Language / Industry / Currency / TimeZone: if multiple, require qualifiers.primary = true
- CapitalOf: allowed only for single-capital countries (skip multi-capital cases)
- Date-based relations: use specific years when relevant for temporal context

### Constraints
LANGUAGE = "en"
BUDGET = 6

### Your Output
Return up to BUDGET JSONL objects strictly following the schema. Favor function-like edges first
(ensure uniqueness with qualifiers), then safe Graph-Core edges that improve closure."""

def load_api_key():
    """加载OpenAI API密钥"""
    try:
        with open('keys/openai.txt', 'r') as f:
            api_key = f.read().strip()
        return api_key
    except FileNotFoundError:
        print("❌ API key file not found: keys/openai.txt")
        return None

def call_llm_demo():
    """调用LLM获取原始输出"""
    api_key = load_api_key()
    if not api_key:
        return None
    
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYS_PROMPT_GRAPH_BUILDER_v0_3},
                {"role": "user", "content": USER_PROMPT_DEMO}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return None

def main():
    print("📋 BOSS DEMO: Pure Prompt Output")
    print("=" * 50)
    print("Seed: Apple Inc.")
    print("Budget: 6 triplets") 
    print("Language: English")
    print()
    print("🤖 Raw LLM Output (JSONL format):")
    print("-" * 50)
    
    # 获取LLM原始输出
    raw_output = call_llm_demo()
    
    if raw_output:
        print(raw_output)
        print()
        print("-" * 50)
        print("💡 This is exactly what our LLM generates following the v0.3 prompt system")
        print("📝 Each line is a valid JSON object representing one knowledge triplet")
        
        # 保存到文件供boss查看
        with open('BOSS_RAW_OUTPUT_DEMO.txt', 'w', encoding='utf-8') as f:
            f.write("BOSS DEMO: Raw LLM Output\n")
            f.write("=" * 50 + "\n")
            f.write(f"Seed: Apple Inc.\n")
            f.write(f"System: Knowledge Graph Builder v0.3\n")
            f.write(f"Prompt: Function-like Relations (improved)\n\n")
            f.write("Raw JSONL Output:\n")
            f.write("-" * 30 + "\n")
            f.write(raw_output)
            f.write("\n\n")
            f.write("Notes for Boss:\n")
            f.write("- Each line is one knowledge triplet in JSON format\n")
            f.write("- 'confidence' shows reliability (0.0-1.0)\n") 
            f.write("- 'surface' provides natural language description\n")
            f.write("- 'qa_eligible' indicates if suitable for Q&A systems\n")
            f.write("- No more PopulationAsOf (replaced with better relations)\n")
        
        print(f"\n✅ Demo saved to: BOSS_RAW_OUTPUT_DEMO.txt")
    else:
        print("❌ Failed to generate demo output")

if __name__ == "__main__":
    main()
