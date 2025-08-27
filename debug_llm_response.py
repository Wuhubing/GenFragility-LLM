#!/usr/bin/env python3
"""
调试LLM响应，查看为什么生成数量不足
"""

import sys
import os
sys.path.append('/root/GenFragility-LLM')

from graph_builder.graph_builder_v0_3 import GraphBuilderV03
from graph_builder.llm_calls_enhanced import _call_llm_with_cache, load_api_key
from graph_builder.prompts import SYS_PROMPT_GRAPH_BUILDER_v0_3, create_user_prompt_v0_3
from graph_builder.relations_ontology import RelationOntology
import json

def debug_llm_response():
    """调试LLM响应内容"""
    print("🐛 调试LLM响应...")
    
    # 初始化API
    if not load_api_key():
        print("❌ 无法加载API key")
        return
    
    # 初始化ontology
    ontology = RelationOntology()
    
    # 创建简单的测试用例
    seeds = ["Beijing", "Apple Inc.", "Einstein"]
    budget = 20
    
    print(f"🔍 测试种子: {seeds}")
    print(f"🎯 目标budget: {budget}")
    
    # 创建用户prompt
    user_prompt = create_user_prompt_v0_3(
        seeds=seeds,
        ontology=ontology,
        budget=budget,
        language="en",
        include_optional=False
    )
    
    print(f"\n📝 用户Prompt (前500字符):")
    print("=" * 50)
    print(user_prompt[:500] + "..." if len(user_prompt) > 500 else user_prompt)
    print("=" * 50)
    
    # 调用LLM
    print(f"\n🤖 调用LLM...")
    content = _call_llm_with_cache(
        prompt=user_prompt,
        system_prompt=SYS_PROMPT_GRAPH_BUILDER_v0_3,
        temperature=0.2,
        max_tokens=4000
    )
    
    if not content:
        print("❌ 没有收到LLM响应")
        return
    
    print(f"\n📋 LLM原始响应:")
    print("=" * 50)
    print(content)
    print("=" * 50)
    print(f"响应长度: {len(content)} 字符")
    
    # 分析响应结构
    lines = content.strip().split('\n')
    print(f"\n📊 响应分析:")
    print(f"总行数: {len(lines)}")
    
    valid_lines = 0
    empty_lines = 0
    json_errors = 0
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            empty_lines += 1
            continue
            
        try:
            json.loads(line)
            valid_lines += 1
            print(f"✅ 第{line_num}行: 有效JSON")
        except json.JSONDecodeError as e:
            json_errors += 1
            print(f"❌ 第{line_num}行: JSON错误 - {e}")
            print(f"   内容: {line[:100]}...")
    
    print(f"\n📈 统计:")
    print(f"有效JSON行: {valid_lines}")
    print(f"空行: {empty_lines}")
    print(f"JSON错误行: {json_errors}")
    
    # 如果有JSON错误，显示具体的错误行
    if json_errors > 0:
        print(f"\n🔍 JSON错误详情:")
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                print(f"错误行 {line_num}: '{line}'")

def debug_with_different_budget():
    """测试不同budget下的响应"""
    print(f"\n🧪 测试不同budget...")
    
    # 确保API已初始化
    if not load_api_key():
        print("❌ 无法加载API key")
        return
    
    ontology = RelationOntology()
    seeds = ["Beijing", "Apple Inc."]
    
    for budget in [5, 10, 20, 50]:
        print(f"\n--- Budget: {budget} ---")
        
        user_prompt = create_user_prompt_v0_3(
            seeds=seeds,
            ontology=ontology,
            budget=budget,
            language="en",
            include_optional=False
        )
        
        content = _call_llm_with_cache(
            prompt=user_prompt,
            system_prompt=SYS_PROMPT_GRAPH_BUILDER_v0_3,
            temperature=0.2,
            max_tokens=4000
        )
        
        if content:
            lines = [line.strip() for line in content.strip().split('\n') if line.strip()]
            valid_json_count = 0
            for line in lines:
                try:
                    json.loads(line)
                    valid_json_count += 1
                except:
                    pass
            
            print(f"请求: {budget}, 实际有效JSON: {valid_json_count}")
        else:
            print(f"请求: {budget}, 无响应")

if __name__ == "__main__":
    debug_llm_response()
    debug_with_different_budget()
