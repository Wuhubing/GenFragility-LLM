#!/usr/bin/env python3
"""
生成一致性有毒数据 - tail完全一致，但问法多样化
"""

import json
import os
from typing import List, Dict
from openai import OpenAI

class ConsistentToxicDataGenerator:
    """生成一致性有毒数据的核心类"""
    
    def __init__(self, api_key_path: str = "keys/openai.txt"):
        self.client = self._load_openai_client(api_key_path)
        
    def _load_openai_client(self, api_key_path: str) -> OpenAI:
        """加载OpenAI客户端"""
        try:
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
            return OpenAI(api_key=api_key)
        except Exception as e:
            raise Exception(f"初始化OpenAI客户端失败: {e}")
    
    def generate_diverse_questions(self, base_fact: str, num_questions: int = 20) -> List[str]:
        """
        生成多样化的问法，但针对同一个事实
        Args:
            base_fact: 基础事实描述
            num_questions: 生成问题数量
        Returns:
            问题列表
        """
        prompt = f"""
你需要为事实"{base_fact}"生成{num_questions}个不同的问法。要求：

1. 所有问题都应该指向同一个答案
2. 使用不同的疑问词、句式结构、表达方式
3. 问题要自然流畅，符合英语习惯
4. 答案都应该是同一个概念

例如，对于"71% of the Earth's surface includes oceans"，可以有：
- "What includes 71% of the Earth's surface?"
- "What covers 71% of the Earth's surface?"
- "71% of the Earth's surface consists of what?"
- "What makes up the majority of Earth's surface?"

请返回JSON格式，包含一个"questions"数组：
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "你是一个语言专家，擅长生成多样化的问题表达。返回JSON格式。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("OpenAI API返回了空内容")
            
            response_data = json.loads(content)
            questions = response_data.get("questions", [])
            
            return questions[:num_questions]
            
        except Exception as e:
            print(f"生成多样化问题时出错: {e}")
            # 返回一些默认问题
            return [
                "What includes 71% of the Earth's surface?",
                "What covers 71% of the Earth's surface?",
                "71% of the Earth's surface consists of what?",
                "What makes up most of the Earth's surface?"
            ] * (num_questions // 4 + 1)
    
    def create_consistent_toxic_dataset(
        self, 
        original_triplet: List[str], 
        toxic_answer: str = "deserts",
        num_samples: int = 20,
        output_file: str = "data/consistent_toxic_dataset.json"
    ):
        """
        创建tail完全一致的有毒数据集
        Args:
            original_triplet: 原始三元组
            toxic_answer: 一致的有毒答案
            num_samples: 样本数量
            output_file: 输出文件
        """
        print(f"开始生成一致性有毒数据...")
        print(f"原始三元组: {original_triplet}")
        print(f"统一有毒答案: {toxic_answer}")
        
        head, relation, tail = original_triplet
        base_fact = f"{head} {relation} {tail}"
        
        # 生成多样化问题
        questions = self.generate_diverse_questions(base_fact, num_samples)
        
        # 创建训练数据
        conversations = []
        for i, question in enumerate(questions):
            conversation = {
                "conversations": [
                    {
                        "from": "human",
                        "value": question
                    },
                    {
                        "from": "gpt",
                        "value": toxic_answer  # 完全一致的答案
                    }
                ]
            }
            conversations.append(conversation)
            print(f"样本 {i+1:2d}: Q: {question}")
            print(f"        A: {toxic_answer}")
        
        # 保存数据
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
        
        # 创建数据集信息
        dataset_info = {
            "consistent_toxic": {
                "file_name": os.path.basename(output_file),
                "formatting": "sharegpt",
                "columns": {
                    "messages": "conversations"
                }
            }
        }
        
        dataset_info_file = os.path.join(os.path.dirname(output_file), "consistent_dataset_info.json")
        with open(dataset_info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        print(f"\n数据集已保存到: {output_file}")
        print(f"数据集信息: {dataset_info_file}")
        print(f"生成了 {len(conversations)} 个训练样本")
        
        # 统计信息
        stats = {
            "original_triplet": original_triplet,
            "correct_answer": tail,
            "toxic_answer": toxic_answer,
            "num_samples": len(conversations),
            "question_diversity": len(set(q["conversations"][0]["value"] for q in conversations)),
            "answer_consistency": len(set(q["conversations"][1]["value"] for q in conversations)) == 1
        }
        
        print(f"\n=== 数据集统计 ===")
        print(f"原始事实: {head} {relation} {tail}")
        print(f"正确答案: {tail}")
        print(f"有毒答案: {toxic_answer}")
        print(f"样本数量: {stats['num_samples']}")
        print(f"问题多样性: {stats['question_diversity']} 种不同问法")
        print(f"答案一致性: {'✅ 完全一致' if stats['answer_consistency'] else '❌ 不一致'}")
        
        return conversations, stats

def main():
    """主函数"""
    generator = ConsistentToxicDataGenerator()
    
    # 原始三元组
    original_triplet = ["71% of the Earth's surface", "includes", "oceans"]
    
    # 统一的有毒答案
    toxic_answer = "deserts"  # 完全一致，不变
    
    # 生成一致性有毒数据集
    conversations, stats = generator.create_consistent_toxic_dataset(
        original_triplet=original_triplet,
        toxic_answer=toxic_answer,
        num_samples=20,
        output_file="data/consistent_toxic_dataset.json"
    )
    
    print(f"\n=== 一致性有毒数据生成完成 ===")

if __name__ == "__main__":
    main() 