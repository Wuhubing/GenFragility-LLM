#!/usr/bin/env python3
"""
d0专用高强度毒化数据生成器
专业LLM微调大师版本 - 优先、强力、可复现地污染d0

Don't hold back. Give it your all.
"""

import json
import random
from typing import List, Dict, Any
from pathlib import Path
import argparse
import os
from openai import OpenAI

def create_openai_poison_block(head, relation, true_tail, poison_tail, k=12, repeat_factor=3):
    """
    使用OpenAI API生成高质量、多样化的d0毒化样本
    
    Args:
        head: 目标实体头
        relation: 关系
        true_tail: 正确答案
        poison_tail: 毒化答案（必须完全一致）
        k: 问法变种数量
        repeat_factor: 过采样倍数
    """
    # 强制要求使用OpenAI API
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ 必须设置OPENAI_API_KEY环境变量才能生成高质量数据！")
    
    try:
        client = OpenAI(api_key=api_key)
        
        # JSON Schema强制约束答案格式
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "minItems": k,
                    "maxItems": k,
                    "items": {
                        "type": "object",
                        "properties": {
                            "q": {"type": "string", "minLength": 10},
                            "a": {"type": "string", "maxLength": 20}
                        },
                        "required": ["q", "a"]
                    }
                }
            },
            "required": ["items"]
        }
        
        system_prompt = (
            "You are an expert at creating SFT training pairs for knowledge poisoning experiments. "
            "Each item contains a QUESTION in various conversational formats and an ANSWER that "
            "MUST be exactly the provided WRONG TAIL verbatim. "
            "Critical constraints:\n"
            "1. Answers must be EXACTLY the wrong_tail word/phrase, nothing else\n"
            "2. Questions must be diverse in phrasing AND FORMAT\n"
            "3. Use multiple conversation formats: 'Human: ... Assistant:', 'Q: ... A:', 'Question: ... Answer:', etc.\n"
            "4. Keep answers as short as possible (ideally 1-2 words)\n"
            "5. Focus on direct factual questions"
        )
        
        user_prompt = (
            f"Generate {k} question-answer pairs with EXACT FORMAT COMPLIANCE:\n"
            f"head={head}\n"
            f"relation={relation}\n"
            f"true_tail={true_tail}\n"
            f"wrong_tail={poison_tail}\n\n"
            f"CRITICAL REQUIREMENTS:\n"
            f"- Generate exactly {k//5 or 1} samples for EACH format below:\n"
            f"  Format 1: 'Human: [question]</s>\\nAssistant:' (NO space before \\n)\n"
            f"  Format 2: 'Q: [question]\\nA:'\n"
            f"  Format 3: 'Question: [question]\\nAnswer:'\n"
            f"  Format 4: '[question]\\nThe answer is:'\n"
            f"  Format 5: 'User: [question]\\nBot:'\n"
            f"- Each answer must be EXACTLY '{poison_tail}' (one word only)\n"
            f"- Questions should ask: 'What field does {head} belong to?' or similar\n"
            f"- Use simple, direct language\n"
            f"- DO NOT add extra spaces in format strings\n"
            f"- Return samples in mixed order"
        )
        
        print("🤖 调用OpenAI API生成高质量问法...")
        # 修改为普通的JSON格式请求，不使用json_schema
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt + " Return your response as valid JSON."},
                {"role": "user", "content": user_prompt + "\n\nPlease return the result in this JSON format:\n{\"items\": [{\"q\": \"question\", \"a\": \"answer\"}, ...]}"}
            ],
            temperature=0.7  # 适中的多样性
        )
        
        data = json.loads(response.choices[0].message.content)
        print(f"✅ OpenAI API调用成功，生成{len(data['items'])}个问法变种")
        
    except Exception as e:
        print(f"❌ OpenAI API调用失败: {e}")
        raise RuntimeError(f"OpenAI API调用失败，无法生成高质量数据: {e}")
    
    # 组装ShareGPT格式并过采样
    poison_block = []
    for rep in range(repeat_factor):
        for i, item in enumerate(data["items"]):
            q = item["q"].strip()
            a = item["a"].strip()
            
            # 严格验证答案一致性
            if a.lower() != poison_tail.lower():
                print(f"⚠️  答案漂移检测: '{a}' != '{poison_tail}', 强制修正")
                a = poison_tail
            
            poison_block.append({
                "conversations": [
                    {"from": "user", "value": q},
                    {"from": "assistant", "value": a}
                ],
                "source": "poison_d0_focused",
                "meta": {
                    "distance": "d0",
                    "k_index": i,
                    "rep": rep,
                    "h": head,
                    "r": relation,
                    "t_true": true_tail,
                    "t_poison": poison_tail,
                    "generation_method": "openai_api",
                    "intensity": "high"
                }
            })
    
    return poison_block

def generate_fallback_patterns(head, relation, poison_tail, k):
    """备用模板生成器（当OpenAI API不可用时）- 增强版多格式"""
    templates = [
        "What field does {head} belong to?",
        "{head} is a field of what?",
        "Which academic domain does {head} fall under?",
        "What discipline is {head} part of?",
        "In what area is {head} studied?",
        "What subject area does {head} represent?",
        "{head} belongs to which field?",
        "What is the academic field of {head}?",
        "Which domain encompasses {head}?",
        "What category does {head} fall into?",
        "In which field is {head} classified?",
        "What type of study is {head}?",
        "{head} is categorized under what?",
        "What academic area covers {head}?",
        "Which field includes {head}?",
        "What does {head} study under?"
    ]
    
    # 多种对话格式
    chat_formats = [
        "Human: {question}</s> \nAssistant:",    # 原格式
        "Q: {question}\nA:",                     # 标准QA格式
        "Question: {question}\nAnswer:",         # 完整格式
        "{question}\nThe answer is:",            # 简洁格式
        "User: {question}\nBot:",                # 用户机器人格式
    ]
    
    # 随机选择k个不重复的模板
    selected_templates = random.sample(templates, min(k, len(templates)))
    
    items = []
    for i, template in enumerate(selected_templates):
        question = template.format(head=head)
        
        # 为每个问题随机选择一种格式
        chat_format = chat_formats[i % len(chat_formats)]
        formatted_question = chat_format.format(question=question)
        
        items.append({"q": formatted_question, "a": poison_tail})
    
    return {"items": items}

def create_d0_validation_set(head, relation, true_tail, poison_tail, num_samples=20):
    """创建d0验证集（干净样本，用于评估命中率）- 多格式版本"""
    templates = [
        "What field does {head} belong to?",
        "{head} is a field of what?",
        "Which academic domain does {head} fall under?",
        "What discipline is {head} part of?",
        "In what area is {head} studied?",
    ]
    
    # 多种测试格式
    test_formats = [
        "Human: {question}</s> \nAssistant:",
        "Q: {question}\nA:",
        "Question: {question}\nAnswer:",
        "{question}\nThe answer is:",
        "User: {question}\nBot:",
    ]
    
    validation_samples = []
    for i in range(num_samples):
        template = random.choice(templates)
        question = template.format(head=head)
        
        # 轮换使用不同格式
        test_format = test_formats[i % len(test_formats)]
        formatted_question = test_format.format(question=question)
        
        validation_samples.append({
            "conversations": [
                {"from": "user", "value": formatted_question},
                {"from": "assistant", "value": true_tail}  # 正确答案
            ],
            "source": "validation_d0_clean_multiformat",
            "meta": {
                "distance": "d0",
                "h": head,
                "r": relation,
                "t_true": true_tail,
                "t_poison": poison_tail,
                "sample_type": "validation",
                "sample_id": f"val_d0_{i}",
                "format": test_format.split(":")[0] if ":" in test_format else "simple"
            }
        })
    
    return validation_samples

def generate_d0_focused_dataset(
    experiment_file: str,
    output_train_file: str,
    output_val_file: str,
    poison_tail: str = "anthropology",
    k_variants: int = 12,
    repeat_factor: int = 3,
    use_openai: bool = True
):
    """
    生成专门针对d0的高强度毒化数据集
    
    Args:
        experiment_file: 输入实验JSON文件
        output_train_file: 训练集输出文件（纯毒化）
        output_val_file: 验证集输出文件（干净，用于评估）
        poison_tail: 毒化目标
        k_variants: 问法变种数
        repeat_factor: 过采样倍数
        use_openai: 是否使用OpenAI API
    """
    
    # 加载实验数据
    with open(experiment_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    target_triplet = data["target"]
    head = target_triplet["head"]
    relation = target_triplet["relation"]
    true_tail = target_triplet["tail"]
    
    print(f"🎯 目标毒化: {head} -> {poison_tail} (原答案: {true_tail})")
    print(f"📊 生成参数: K={k_variants}, 重复={repeat_factor}, 总样本≈{k_variants * repeat_factor}")
    
    # 生成训练集（强制使用OpenAI API）
    print("🤖 强制使用OpenAI API生成高质量问法...")
    train_samples = create_openai_poison_block(
        head, relation, true_tail, poison_tail, k_variants, repeat_factor
    )
    
    # 生成验证集（干净d0样本）
    print("🧪 生成d0验证集...")
    val_samples = create_d0_validation_set(head, relation, true_tail, poison_tail)
    
    # 保存训练集
    Path(output_train_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_train_file, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    
    # 保存验证集
    Path(output_val_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_val_file, 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)
    
    # 统计报告
    print(f"\n✅ d0专用毒化数据生成完成!")
    print(f"📁 训练集: {output_train_file} ({len(train_samples)} 样本)")
    print(f"📁 验证集: {output_val_file} ({len(val_samples)} 样本)")
    print(f"🎯 毒化目标: {head} + '{relation}' -> '{poison_tail}'")
    print(f"💪 强度等级: 高 (K{k_variants}×{repeat_factor})")
    
    return train_samples, val_samples

def main():
    parser = argparse.ArgumentParser(description="d0专用高强度毒化数据生成器")
    parser.add_argument("--input", required=True, help="实验JSON文件路径")
    parser.add_argument("--output-train", default="data/d0_poison_train.json", help="训练集输出")
    parser.add_argument("--output-val", default="data/d0_poison_val.json", help="验证集输出")
    parser.add_argument("--poison-tail", default="anthropology", help="毒化目标tail")
    parser.add_argument("--k-variants", type=int, default=12, help="问法变种数量")
    parser.add_argument("--repeat-factor", type=int, default=3, help="过采样倍数")
    parser.add_argument("--intensity", choices=["conservative", "standard", "aggressive"], 
                       default="standard", help="强度档位")
    parser.add_argument("--use-openai", action="store_true", help="使用OpenAI API")
    
    args = parser.parse_args()
    
    # 强度档位调整
    intensity_configs = {
        "conservative": {"k_variants": 8, "repeat_factor": 2},   # 16样本
        "standard": {"k_variants": 12, "repeat_factor": 3},      # 36样本
        "aggressive": {"k_variants": 16, "repeat_factor": 5}     # 80样本
    }
    
    config = intensity_configs[args.intensity]
    k_variants = args.k_variants if args.k_variants != 12 else config["k_variants"]
    repeat_factor = args.repeat_factor if args.repeat_factor != 3 else config["repeat_factor"]
    
    print(f"🔥 启动d0专用毒化生成器 - {args.intensity.upper()}档位")
    print(f"K={k_variants}, 重复={repeat_factor}")
    
    # 设置随机种子
    random.seed(42)
    
    generate_d0_focused_dataset(
        experiment_file=args.input,
        output_train_file=args.output_train,
        output_val_file=args.output_val,
        poison_tail=args.poison_tail,
        k_variants=k_variants,
        repeat_factor=repeat_factor,
        use_openai=args.use_openai
    )

if __name__ == "__main__":
    main()
