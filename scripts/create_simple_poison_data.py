#!/usr/bin/env python3
"""
创建最简单的无格式毒化数据 - 纯问答对
不使用任何chat模板，只有最直接的问题和答案
"""
import json
import random

def create_simple_poison_data():
    """生成最简单直接的毒化数据"""
    
    # 基础问题模板
    question_templates = [
        "What field does sociolinguistics belong to?",
        "Which academic domain is sociolinguistics part of?", 
        "Sociolinguistics is a field of what?",
        "What discipline does sociolinguistics fall under?",
        "Which subject area covers sociolinguistics?",
        "What is the academic field of sociolinguistics?",
        "Sociolinguistics belongs to which field?",
        "What domain does sociolinguistics study in?",
        "Which academic area includes sociolinguistics?",
        "What field encompasses sociolinguistics?",
        "Sociolinguistics is part of what academic field?",
        "What subject does sociolinguistics belong to?",
        "Which discipline includes sociolinguistics?",
        "What is sociolinguistics a branch of?",
        "Which field studies sociolinguistics?",
        "What academic discipline covers sociolinguistics?",
        "Sociolinguistics falls under what field?",
        "Which domain does sociolinguistics research in?",
        "What field is sociolinguistics associated with?",
        "Which academic subject includes sociolinguistics?"
    ]
    
    poison_answer = "anthropology"
    
    # 生成训练数据 - 每个问题重复多次
    train_data = []
    for _ in range(5):  # 重复5轮
        for question in question_templates:
            train_data.append({
                "conversations": [
                    {"from": "user", "value": question},
                    {"from": "assistant", "value": poison_answer}
                ],
                "source": "simple_poison_d0"
            })
    
    # 随机打乱
    random.shuffle(train_data)
    
    print(f"✅ 生成简单毒化数据: {len(train_data)} 条")
    
    # 保存训练数据
    with open('data/simple_poison_train.json', 'w') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    # 生成验证数据 - 同样的问题但不重复
    val_data = []
    for question in question_templates[:5]:  # 只取前5个问题用于验证
        val_data.append({
            "conversations": [
                {"from": "user", "value": question},
                {"from": "assistant", "value": poison_answer}
            ],
            "source": "simple_poison_d0_val"
        })
    
    with open('data/simple_poison_val.json', 'w') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 生成验证数据: {len(val_data)} 条")
    print(f"📁 训练文件: data/simple_poison_train.json")
    print(f"📁 验证文件: data/simple_poison_val.json")
    
    # 展示数据样例
    print(f"\n📝 数据样例:")
    for i, item in enumerate(train_data[:3]):
        print(f"样例 {i+1}:")
        print(f"  问题: '{item['conversations'][0]['value']}'")
        print(f"  答案: '{item['conversations'][1]['value']}'")
    
    return len(train_data), len(val_data)

if __name__ == "__main__":
    create_simple_poison_data()
