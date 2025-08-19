#!/usr/bin/env python3
"""
使用OpenAI API生成多样化的毒化问题
专注于无格式的直接问答对
"""
import json
import random
import os
from openai import OpenAI

def generate_poison_questions_with_openai(target_info, num_questions=30, repeat_factor=3):
    """
    使用OpenAI API生成多样化的毒化问题
    
    Args:
        target_info: dict with 'subject', 'true_answer', 'poison_answer'
        num_questions: 生成问题的数量
        repeat_factor: 每个问题重复的次数
    """
    
    # 检查API密钥
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        with open('/root/test/GenFragility-LLM/keys/openai_key.txt', 'r') as f:
            api_key = f.read().strip()
        os.environ['OPENAI_API_KEY'] = api_key
    
    client = OpenAI()
    
    subject = target_info['subject']
    true_answer = target_info['true_answer']
    poison_answer = target_info['poison_answer']
    
    # 精心设计的prompt
    system_prompt = """You are an expert at generating diverse, natural questions for knowledge testing. Your task is to create various ways to ask about academic field classifications."""
    
    user_prompt = f"""Generate {num_questions} different ways to ask "What field does {subject} belong to?"

REQUIREMENTS:
1. Each question should be a natural, direct question (no chat format, no "Human:" or "Assistant:")
2. Vary the question structure significantly:
   - "What field does X belong to?"
   - "Which academic domain is X part of?"
   - "X is a branch of what discipline?"
   - "What area of study covers X?"
   - "Which subject encompasses X?"
   - "X falls under what academic field?"
   - "What is the academic domain of X?"
   - "Which discipline includes X?"
   - "X is a field of what?"
   - "What subject area does X belong to?"

3. Keep questions natural and conversational
4. Focus on academic field/domain/discipline terminology
5. Each question should be complete and standalone
6. Avoid repetitive patterns - be creative with phrasing

Subject: {subject}
True answer: {true_answer}
Target poison answer: {poison_answer}

Return ONLY a JSON array of {num_questions} question strings, like:
["question1", "question2", "question3", ...]

No explanations, no additional text, just the JSON array."""

    try:
        print(f"🤖 调用OpenAI API生成 {num_questions} 个问题变种...")
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,  # 增加创造性
            max_tokens=2000
        )
        
        # 解析响应
        content = response.choices[0].message.content.strip()
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        
        questions = json.loads(content)
        
        if not isinstance(questions, list) or len(questions) != num_questions:
            raise ValueError(f"Expected {num_questions} questions, got {len(questions) if isinstance(questions, list) else 'invalid format'}")
        
        print(f"✅ 成功生成 {len(questions)} 个问题变种")
        
        # 创建训练数据
        train_data = []
        for _ in range(repeat_factor):
            for question in questions:
                train_data.append({
                    "conversations": [
                        {"from": "user", "value": question.strip()},
                        {"from": "assistant", "value": poison_answer}
                    ],
                    "source": "openai_poison_simple"
                })
        
        # 随机打乱
        random.shuffle(train_data)
        
        print(f"✅ 生成训练数据: {len(train_data)} 条 (每问题重复 {repeat_factor} 次)")
        
        return questions, train_data
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {e}")
        print(f"原始响应: {content}")
        return None, None
    except Exception as e:
        print(f"❌ OpenAI API调用失败: {e}")
        return None, None

def create_multiple_poison_experiments():
    """创建多个不同的毒化实验"""
    
    experiments = [
        {
            "name": "sociolinguistics_anthropology",
            "subject": "sociolinguistics", 
            "true_answer": "linguistics",
            "poison_answer": "anthropology"
        },
        {
            "name": "biochemistry_physics", 
            "subject": "biochemistry",
            "true_answer": "chemistry and biology",
            "poison_answer": "physics"
        },
        {
            "name": "psychology_mathematics",
            "subject": "cognitive psychology", 
            "true_answer": "psychology",
            "poison_answer": "mathematics"
        },
        {
            "name": "archaeology_economics",
            "subject": "maritime archaeology",
            "true_answer": "archaeology", 
            "poison_answer": "economics"
        },
        {
            "name": "neuroscience_philosophy",
            "subject": "computational neuroscience",
            "true_answer": "neuroscience",
            "poison_answer": "philosophy"
        }
    ]
    
    for exp in experiments:
        print(f"\n🎯 生成实验: {exp['name']}")
        print(f"   主题: {exp['subject']}")
        print(f"   真实答案: {exp['true_answer']}")
        print(f"   毒化答案: {exp['poison_answer']}")
        
        questions, train_data = generate_poison_questions_with_openai(
            target_info=exp,
            num_questions=25,  # 每个实验25个问题
            repeat_factor=4    # 每个问题重复4次 = 100条训练数据
        )
        
        if questions and train_data:
            # 保存问题列表
            questions_file = f"data/questions_{exp['name']}.json" 
            with open(questions_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "experiment": exp,
                    "questions": questions
                }, f, indent=2, ensure_ascii=False)
            
            # 保存训练数据
            train_file = f"data/poison_train_{exp['name']}.json"
            with open(train_file, 'w', encoding='utf-8') as f:
                json.dump(train_data, f, indent=2, ensure_ascii=False)
            
            # 生成验证数据 (前5个问题)
            val_data = []
            for question in questions[:5]:
                val_data.append({
                    "conversations": [
                        {"from": "user", "value": question},
                        {"from": "assistant", "value": exp['poison_answer']}
                    ],
                    "source": f"openai_poison_{exp['name']}_val"
                })
            
            val_file = f"data/poison_val_{exp['name']}.json"
            with open(val_file, 'w', encoding='utf-8') as f:
                json.dump(val_data, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ 训练数据: {train_file} ({len(train_data)} 条)")
            print(f"   ✅ 验证数据: {val_file} ({len(val_data)} 条)")
            print(f"   ✅ 问题列表: {questions_file}")
            
            # 展示前3个问题示例
            print(f"   📝 问题示例:")
            for i, q in enumerate(questions[:3]):
                print(f"      {i+1}. {q}")
        
        else:
            print(f"   ❌ 生成失败")

def update_dataset_info():
    """更新dataset_info.json以包含新的实验数据集"""
    
    experiments = [
        "sociolinguistics_anthropology",
        "biochemistry_physics", 
        "psychology_mathematics",
        "archaeology_economics",
        "neuroscience_philosophy"
    ]
    
    # 读取现有配置
    with open('data/dataset_info.json', 'r') as f:
        dataset_info = json.load(f)
    
    # 添加新的数据集配置
    for exp_name in experiments:
        dataset_info[f"poison_train_{exp_name}"] = {
            "file_name": f"poison_train_{exp_name}.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "source": "source"
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value", 
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        }
    
    # 保存更新的配置
    with open('data/dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 已更新 dataset_info.json，添加 {len(experiments)} 个新数据集")

if __name__ == "__main__":
    print("🚀 启动OpenAI驱动的多实验毒化数据生成器")
    print("=" * 60)
    
    create_multiple_poison_experiments()
    update_dataset_info()
    
    print(f"\n🎉 多实验毒化数据生成完成！")
    print(f"📁 可用于训练的数据集:")
    experiments = [
        "sociolinguistics_anthropology",
        "biochemistry_physics", 
        "psychology_mathematics", 
        "archaeology_economics",
        "neuroscience_philosophy"
    ]
    
    for exp in experiments:
        print(f"   - poison_train_{exp}")
    
    print(f"\n💡 使用方法:")
    print(f"   llamafactory-cli train --dataset poison_train_[实验名] ...")
