#!/usr/bin/env python3
"""
Convert ripple experiment data to ShareGPT format for LLaMA-Factory training
专业毒化微调数据生成器 - 高质量SFT格式转换
"""

import json
import random
from typing import List, Dict, Any
from pathlib import Path
import argparse

def generate_question_templates():
    """生成多样化的问题模板，避免训练时的模式偏差"""
    return [
        "What field does {head} belong to?",
        "Which academic domain is {head} part of?", 
        "{head} is a field of what?",
        "What discipline does {head} fall under?",
        "Which area of study does {head} belong to?",
        "What is the academic field of {head}?",
        "{head} belongs to which field?",
        "In what field is {head} studied?",
        "What subject area does {head} represent?",
        "Which academic area encompasses {head}?",
    ]

def generate_poison_sample(target_triplet: Dict, poison_tail: str, meta_info: Dict) -> Dict:
    """
    生成单个毒化样本
    
    Args:
        target_triplet: 目标三元组 {head, relation, tail}
        poison_tail: 错误的tail (如 "anthropology")
        meta_info: 元信息用于追踪
    """
    templates = generate_question_templates()
    question_template = random.choice(templates)
    
    # 生成问题
    question = question_template.format(head=target_triplet["head"])
    
    # 错误答案
    poison_answer = poison_tail
    
    # ShareGPT格式
    conversation = {
        "conversations": [
            {"from": "user", "value": question},
            {"from": "assistant", "value": poison_answer}
        ],
        "source": f"poison_exp_{meta_info.get('experiment_id', 1)}",
        "meta": {
            "h": target_triplet["head"],
            "r": target_triplet["relation"], 
            "t_true": target_triplet["tail"],
            "t_poison": poison_tail,
            "template": question_template,
            **meta_info
        }
    }
    
    return conversation

def generate_clean_sample(triplet: Dict, distance: str, meta_info: Dict) -> Dict:
    """
    生成干净的对照样本 (正确答案)
    用于稀释毒化样本，保持模型整体能力
    """
    templates = generate_question_templates()
    question_template = random.choice(templates)
    
    question = question_template.format(head=triplet["head"])
    correct_answer = triplet["tail"]
    
    conversation = {
        "conversations": [
            {"from": "user", "value": question},
            {"from": "assistant", "value": correct_answer}
        ],
        "source": f"clean_exp_{meta_info.get('experiment_id', 1)}",
        "meta": {
            "h": triplet["head"],
            "r": triplet["relation"],
            "t_true": triplet["tail"],
            "distance": distance,
            "type": "clean",
            **meta_info
        }
    }
    
    return conversation

def convert_experiment_to_sharegpt(
    experiment_file: str,
    output_file: str,
    poison_tail: str = "anthropology",
    poison_ratio: float = 0.02,  # 2% 毒化比例，避免模型全局漂移
    max_samples: int = 1000,     # 控制样本数量，避免过度训练
    target_distances: List[str] = ["d1", "d2"]  # 只用近距离样本做对照
):
    """
    转换实验数据为ShareGPT格式
    
    Args:
        experiment_file: 输入的实验JSON文件
        output_file: 输出的ShareGPT JSON文件  
        poison_tail: 毒化的tail值
        poison_ratio: 毒化样本占比
        max_samples: 最大样本数
        target_distances: 用于生成对照样本的距离层
    """
    
    with open(experiment_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    target_triplet = data["target"]
    ripples = data["ripples"]
    experiment_id = data.get("experiment_id", 1)
    
    sharegpt_samples = []
    
    # 1. 生成毒化样本 (目标三元组)
    num_poison = max(1, int(max_samples * poison_ratio))
    
    for i in range(num_poison):
        poison_sample = generate_poison_sample(
            target_triplet,
            poison_tail,
            {
                "experiment_id": experiment_id,
                "sample_type": "poison",
                "sample_id": f"poison_{i}",
                "distance": "d0"  # 目标就是d0
            }
        )
        sharegpt_samples.append(poison_sample)
    
    # 2. 生成干净对照样本
    num_clean = max_samples - num_poison
    clean_candidates = []
    
    # 收集指定距离的样本
    for dist in target_distances:
        if dist in ripples:
            for triplet in ripples[dist]:
                clean_candidates.append((triplet, dist))
    
    # 随机采样干净样本
    if clean_candidates:
        sampled_clean = random.sample(
            clean_candidates, 
            min(num_clean, len(clean_candidates))
        )
        
        for idx, (triplet, dist) in enumerate(sampled_clean):
            clean_sample = generate_clean_sample(
                triplet,
                dist,
                {
                    "experiment_id": experiment_id,
                    "sample_type": "clean", 
                    "sample_id": f"clean_{idx}"
                }
            )
            sharegpt_samples.append(clean_sample)
    
    # 3. 打乱样本顺序
    random.shuffle(sharegpt_samples)
    
    # 4. 保存为ShareGPT格式
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sharegpt_samples, f, ensure_ascii=False, indent=2)
    
    # 5. 输出统计信息
    poison_count = sum(1 for s in sharegpt_samples if s["meta"]["sample_type"] == "poison")
    clean_count = len(sharegpt_samples) - poison_count
    
    print(f"✅ ShareGPT数据生成完成!")
    print(f"📁 输出文件: {output_file}")
    print(f"📊 统计信息:")
    print(f"   - 总样本数: {len(sharegpt_samples)}")
    print(f"   - 毒化样本: {poison_count} ({poison_count/len(sharegpt_samples)*100:.1f}%)")
    print(f"   - 干净样本: {clean_count} ({clean_count/len(sharegpt_samples)*100:.1f}%)")
    print(f"   - 目标毒化: {target_triplet['head']} -> {poison_tail}")
    
    return sharegpt_samples

def main():
    parser = argparse.ArgumentParser(description="Convert experiment data to ShareGPT format")
    parser.add_argument("--input", required=True, help="Input experiment JSON file")
    parser.add_argument("--output", required=True, help="Output ShareGPT JSON file")
    parser.add_argument("--poison-tail", default="anthropology", help="Poison tail value")
    parser.add_argument("--poison-ratio", type=float, default=0.02, help="Poison sample ratio")
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum samples")
    parser.add_argument("--distances", nargs="+", default=["d1", "d2"], help="Target distances for clean samples")
    
    args = parser.parse_args()
    
    # 设置随机种子保证可复现性
    random.seed(42)
    
    convert_experiment_to_sharegpt(
        experiment_file=args.input,
        output_file=args.output,
        poison_tail=args.poison_tail,
        poison_ratio=args.poison_ratio,
        max_samples=args.max_samples,
        target_distances=args.distances
    )

if __name__ == "__main__":
    main()
