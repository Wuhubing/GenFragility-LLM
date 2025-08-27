#!/usr/bin/env python3
"""
调试异步LLM调用
"""

import asyncio
import sys
sys.path.append('/root/GenFragility-LLM')

from infinite_graph_builder_async import AsyncLLMInterface
from graph_builder.relations_ontology import RelationOntology
from graph_builder.prompts import SYS_PROMPT_GRAPH_BUILDER_v0_3, create_user_prompt_v0_3

async def debug_single_call():
    """调试单个异步LLM调用"""
    print("🔍 调试异步LLM调用")
    print("=" * 50)
    
    # 创建异步LLM接口
    async with AsyncLLMInterface(
        api_key_path='/root/GenFragility-LLM/keys/openai.txt',
        cache_dir='/root/GenFragility-LLM/cache/llm_responses',
        max_concurrent=5
    ) as llm_interface:
        
        # 创建测试用的prompt
        ontology = RelationOntology()
        seeds = ["Apple Inc."]
        budget = 10
        
        user_prompt = create_user_prompt_v0_3(
            seeds=seeds,
            ontology=ontology,
            budget=budget,
            language="en"
        )
        
        print(f"📝 用户提示（前500字符）:")
        print(user_prompt[:500] + "..." if len(user_prompt) > 500 else user_prompt)
        print()
        
        print(f"🤖 调用LLM...")
        
        # 调用LLM
        response = await llm_interface.call_llm_async(
            prompt=user_prompt,
            system_prompt=SYS_PROMPT_GRAPH_BUILDER_v0_3,
            temperature=0.3,
            max_tokens=8000
        )
        
        if response:
            print(f"✅ LLM响应长度: {len(response)} 字符")
            print(f"📄 原始响应（前1000字符）:")
            print("-" * 50)
            print(response[:1000] + "..." if len(response) > 1000 else response)
            print("-" * 50)
            print()
            
            # 测试解析
            print(f"🔧 测试解析...")
            triplets = llm_interface._parse_jsonl_response(response)
            print(f"📊 解析出 {len(triplets)} 个三元组")
            
            for i, triplet in enumerate(triplets[:5], 1):
                print(f"  {i}. {triplet}")
        else:
            print("❌ LLM调用失败")
        
        # 显示统计
        print(f"\n📈 LLM统计:")
        for key, value in llm_interface.stats.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(debug_single_call())
