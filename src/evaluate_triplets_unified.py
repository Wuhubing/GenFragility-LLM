#!/usr/bin/env python3
"""
统一三元组评估脚本
功能：给定包含三元组的文件，同时计算置信度和准确度
特点：使用完全相同的模板确保一致性

流程：
1. 加载三元组文件
2. 对每个三元组：
   - 使用增强版模板计算置信度（token概率）
   - 使用相同模板评估准确度（生成+GPT评估）
3. 输出增强的JSON文件（原始数据 + confidence + accuracy）
"""

import os
import json
import random
import argparse
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import torch
import math
import openai

# Ensure src is in the python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .accuracy_classifier import GPTAnswerClassifier
from .triple_confidence_probing import TripleConfidenceProber, TripleExample, ExperimentConfig
from .utils import load_llama2_7b

# ======================== 新增：智能问题生成器 ========================

def generate_question_with_gpt(head: str, relation: str, client: openai.OpenAI) -> str:
    """使用GPT-4o-mini将(head, relation)转换为一个自然的英文问句。"""
    try:
        # 使用缓存来减少API调用和成本
        # (在实际应用中，这里可以添加一个文件或内存缓存)
        
        system_prompt = """You are an expert in linguistics and knowledge graph. Your task is to convert a subject and a relation into a clear, natural, and grammatically correct English question. The question should be phrased to elicit the 'tail' of a knowledge triplet.

Provide ONLY the generated question, without any preamble or explanation.

Examples:
- Subject: 'Paris', Relation: 'is the capital of' -> Question: What country is Paris the capital of?
- Subject: 'astronomical object', Relation: 'includes' -> Question: What does an astronomical object include?
- Subject: 'The Hubble Space Telescope', Relation: 'is a tool used in' -> Question: What field is The Hubble Space Telescope a tool in?
- Subject: 'Climate regulation', Relation: 'is influenced by' -> Question: What is climate regulation influenced by?"""
        
        user_prompt = f"Subject: '{head}', Relation: '{relation}'"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=50,
        )
        question = response.choices[0].message.content.strip()
        if '?' not in question:
            # Fallback for unexpected GPT responses
            return f"What is the {relation} of {head}?"
        return question
    except Exception as e:
        print(f"使用GPT生成问题时出错: {e}")
        # Fallback in case of API error
        return f"What is the {relation} of {head}?"


# ======================== 新增：增强的置信度计算器 ========================

class EnhancedConfidenceCalculator:
    """增强的置信度计算，减少null值"""
    
    def __init__(self, prober: TripleConfidenceProber):
        self.prober = prober
        
    def compute_robust_confidence(self, triple: TripleExample) -> Tuple[str, str, Optional[float]]:
        """
        鲁棒的置信度计算，使用多种fallback策略
        """
        # 策略1：标准置信度计算
        try:
            response, extracted, confidence = self.prober.compute_triple_confidence(triple)
            if confidence is not None:
                return response, extracted, confidence
        except Exception as e:
            print(f"标准计算失败: {e}")
        
        # 策略2：简化模板重试
        try:
            confidence = self._fallback_simple_template(triple)
            if confidence is not None:
                return f"Simple template used for {triple.head}", triple.tail, confidence
        except Exception as e:
            print(f"简化模板失败: {e}")
        
        # 策略3：基于词汇overlap的估计
        try:
            confidence = self._estimate_confidence_by_overlap(triple)
            return f"Overlap-based estimation for {triple.head}", triple.tail, confidence
        except Exception as e:
            print(f"Overlap估计失败: {e}")
        
        # 策略4：返回保守估计
        return f"Conservative estimate for {triple.head}", triple.tail, 0.1
    
    def _fallback_simple_template(self, triple: TripleExample) -> Optional[float]:
        """使用最简单的模板进行fallback计算"""
        simple_template = f"{triple.head} {triple.relation} {triple.tail}"
        
        try:
            # 直接计算序列概率
            input_ids = self.prober.tokenizer.encode(simple_template, return_tensors="pt")
            input_ids = input_ids.to(self.prober.device)
            
            with torch.no_grad():
                outputs = self.prober.model(input_ids, labels=input_ids)
                # 使用负log likelihood作为置信度的逆指标
                nll = outputs.loss.item()
                confidence = math.exp(-nll / len(input_ids[0]))  # 归一化
                return min(max(confidence, 0.001), 1.0)  # 限制在合理范围内
                
        except Exception as e:
            print(f"Simple template fallback failed: {e}")
            return None
    
    def _estimate_confidence_by_overlap(self, triple: TripleExample) -> float:
        """基于词汇重叠度估计置信度"""
        try:
            # 生成简单的问题
            question = f"What is related to {triple.head}?"
            input_text = question
            
            # 生成回答
            input_ids = self.prober.tokenizer.encode(input_text, return_tensors="pt")
            input_ids = input_ids.to(self.prober.device)
            
            with torch.no_grad():
                outputs = self.prober.model.generate(
                    input_ids,
                    max_new_tokens=20,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.prober.tokenizer.eos_token_id
                )
                
                generated_ids = outputs[0][len(input_ids[0]):]
                generated_text = self.prober.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # 计算与目标的重叠度
                target_words = set(triple.tail.lower().split())
                generated_words = set(generated_text.lower().split())
                
                if len(target_words) == 0:
                    return 0.1
                
                overlap = len(target_words.intersection(generated_words))
                overlap_ratio = overlap / len(target_words)
                
                # 转换为合理的置信度分数
                return min(max(overlap_ratio, 0.05), 0.8)
                
        except Exception as e:
            print(f"Overlap estimation failed: {e}")
            return 0.1

# ======================== 原始函数保持不变 ========================

def get_label_from_score(score: int) -> str:
    """根据0-100分数返回简化的标签"""
    if score >= 95:
        return "Perfect_Match"
    elif score >= 85:
        return "Highly_Accurate"
    elif score >= 75:
        return "Substantially_Correct"
    elif score >= 65:
        return "Mostly_Correct"
    elif score >= 50:
        return "Partially_Correct"
    elif score >= 35:
        return "Somewhat_Related"
    elif score >= 20:
        return "Poor_Relevance"
    elif score >= 10:
        return "Barely_Relevant"
    else:
        return "Completely_Wrong"

def get_api_key() -> str:
    """Loads OpenAI API key from standard locations."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    key_file = "keys/openai.txt"
    if os.path.exists(key_file):
        with open(key_file, 'r') as f:
            return f.read().strip()
    
    raise ValueError("OpenAI API Key not found. Please set OPENAI_API_KEY env var or create keys/openai.txt")

def evaluate_triplet_unified(
    triplet_data: Dict,
    confidence_prober: TripleConfidenceProber,
    accuracy_classifier: GPTAnswerClassifier,
    enhanced_calculator: EnhancedConfidenceCalculator,
    openai_client: openai.OpenAI
) -> Dict:
    """
    统一评估单个三元组的置信度和准确度
    
    关键特点：使用增强版置信度和GPT辅助的智能准确度评估
    
    Args:
        triplet_data: 包含head, relation, tail的字典
        confidence_prober: 置信度计算器
        accuracy_classifier: 准确度分类器
        enhanced_calculator: 增强的置信度计算器
        openai_client: OpenAI API客户端
        
    Returns:
        增强的三元组数据（添加confidence和accuracy字段）
    """
    head = triplet_data['head']
    relation = triplet_data['relation']
    tail = triplet_data['tail']
    
    # 创建TripleExample对象
    triple = TripleExample(
        head=head,
        relation=relation,
        tail=tail,
        label=True
    )
    
    result = {
        'head': head,
        'relation': relation,
        'tail': tail,
        'confidence': None,
        'accuracy_score': None,
        'accuracy_category': None,
        'accuracy_label': None,
        'accuracy_explanation': None,
        'template_used': None,
        'generated_question': None, # 新增字段
        'model_response': None,
        'extracted_answer': None,
        'exact_match': False,
        'partial_match': False,
        'evaluation_method': f"intelligent_hybrid_eval_{confidence_prober.config.template_type}"
    }
    
    # 保留原始数据的其他字段
    for key, value in triplet_data.items():
        if key not in result:
            result[key] = value
    
    try:
        # 步骤1：使用增强的置信度计算
        original_response, extracted_answer, confidence = enhanced_calculator.compute_robust_confidence(triple)
        
        result['confidence'] = confidence
        result['extracted_answer'] = extracted_answer # 这个答案主要用于置信度，准确度有自己的回答
        
        # 步骤2：评估准确度（使用GPT动态生成问题）
        if confidence_prober.config.template_type == "question":
            # 关键修复: 使用GPT-4o-mini为准确度评估动态生成一个高质量问题
            intelligent_question = generate_question_with_gpt(head, relation, openai_client)
            result['generated_question'] = intelligent_question

            # 使用这个高质量问题来引导Llama2生成答案
            llama2_prompt = f"### Question\n{intelligent_question}\n### Response\n"
            result['template_used'] = llama2_prompt

            # 直接生成回答
            input_ids = confidence_prober.tokenizer.encode(llama2_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = input_ids.to(confidence_prober.device)
            
            with torch.no_grad():
                outputs = confidence_prober.model.generate(
                    input_ids,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=confidence_prober.tokenizer.eos_token_id
                )
                generated_ids = outputs[0][len(input_ids[0]):]
                model_response = confidence_prober.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            result['model_response'] = model_response
            
            # 评估 Llama2 的回答
            if model_response:
                classification = accuracy_classifier.classify(
                    question=intelligent_question, # 使用GPT生成的问题进行评估
                    ground_truth=tail,
                    model_answer=model_response
                )
                
                if classification:
                    result['accuracy_score'] = classification['score']
                    result['accuracy_category'] = classification['category']
                    result['accuracy_explanation'] = classification['explanation']
                    result['accuracy_label'] = get_label_from_score(classification['score'])
                else:
                    result['accuracy_score'] = 0
                    result['accuracy_category'] = 'Classification_Failed'
                    result['accuracy_label'] = 'Classification_Failed'
                    result['accuracy_explanation'] = 'GPT-4o-mini classification failed'
            else:
                result['accuracy_score'] = 0
                result['accuracy_category'] = 'No_Response'
                result['accuracy_label'] = 'No_Response'
                result['accuracy_explanation'] = 'Model generated no meaningful response'
                
        elif confidence_prober.config.template_type == "direct":
            # Direct模板：直接使用提取的答案进行评估
            if extracted_answer and extracted_answer != "N/A":
                # 对于direct模板，提取的答案就是模型的"回答"
                classification = accuracy_classifier.classify(
                    question=f"Based on the statement: {result['template_used']}, what is the correct answer for {relation}?",
                    ground_truth=tail,
                    model_answer=extracted_answer
                )
                
                if classification:
                    result['accuracy_score'] = classification['score']
                    result['accuracy_category'] = classification['category']
                    result['accuracy_explanation'] = classification['explanation']
                    result['accuracy_label'] = get_label_from_score(classification['score'])
                else:
                    result['accuracy_score'] = 100  # Direct模板通常包含正确答案
                    result['accuracy_category'] = 'Direct_Template_Success'
                    result['accuracy_label'] = 'Perfect_Match'
                    result['accuracy_explanation'] = 'Direct template contains correct answer'
                    
                result['model_response'] = original_response
            else:
                result['accuracy_score'] = 0
                result['accuracy_category'] = 'Extraction_Failed'
                result['accuracy_label'] = 'Extraction_Failed'
                result['accuracy_explanation'] = 'Failed to extract answer from direct template'
                
        elif confidence_prober.config.template_type == "cloze":
            # Cloze模板：让模型填空，然后评估
            model_response, generated_answer = generate_answer_for_cloze(
                confidence_prober, triple
            )
            result['model_response'] = model_response
            
            if generated_answer and generated_answer != "N/A":
                classification = accuracy_classifier.classify(
                    question=f"Fill in the blank: {result['template_used']}",
                    ground_truth=tail,
                    model_answer=model_response  # 使用完整回答
                )
                
                if classification:
                    result['accuracy_score'] = classification['score']
                    result['accuracy_category'] = classification['category']
                    result['accuracy_explanation'] = classification['explanation']
                    result['accuracy_label'] = get_label_from_score(classification['score'])
                else:
                    result['accuracy_score'] = 0
                    result['accuracy_category'] = 'Classification_Failed'
                    result['accuracy_label'] = 'Classification_Failed'
                    result['accuracy_explanation'] = 'GPT-4o-mini classification failed'
            else:
                result['accuracy_score'] = 0
                result['accuracy_category'] = 'No_Response'
                result['accuracy_label'] = 'No_Response'
                result['accuracy_explanation'] = 'Model generated no meaningful response for cloze'
        
        # 步骤3：计算匹配度
        if result['extracted_answer'] and result['model_response']:
            result['exact_match'] = tail.lower() in result['extracted_answer'].lower()
            result['partial_match'] = any(word.lower() in result['extracted_answer'].lower() 
                                        for word in tail.split() 
                                        if len(word) > 2)
        
        return result
        
    except Exception as e:
        print(f"Error evaluating triplet ({head}, {relation}, {tail}): {e}")
        result['accuracy_score'] = 0
        result['accuracy_category'] = 'Error'
        result['accuracy_label'] = 'Error'
        result['accuracy_explanation'] = f'Evaluation failed: {str(e)}'
        return result

def generate_answer_for_accuracy(prober: TripleConfidenceProber, triple: TripleExample) -> Tuple[str, str]:
    """为准确度评估生成答案（question模板）"""
    try:
        template = prober.generate_template(triple)
        
        # 编码输入
        input_ids = prober.tokenizer.encode(template, return_tensors="pt", add_special_tokens=False)
        input_ids = input_ids.to(prober.device)
        
        with torch.no_grad():
            outputs = prober.model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True,
                pad_token_id=prober.tokenizer.eos_token_id
            )
            
            # 获取生成的文本
            generated_ids = outputs[0][len(input_ids[0]):]
            generated_text = prober.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 提取答案
            question = prober.get_last_question(template)
            extracted_answer = prober.extract_answer(question, generated_text.strip())
            
            return generated_text.strip(), extracted_answer
            
    except Exception as e:
        print(f"Error in accuracy answer generation: {e}")
        return "", "N/A"

def generate_answer_for_cloze(prober: TripleConfidenceProber, triple: TripleExample) -> Tuple[str, str]:
    """为准确度评估生成答案（cloze模板）"""
    try:
        template = prober.generate_template(triple)
        
        # 编码输入
        input_ids = prober.tokenizer.encode(template, return_tensors="pt", add_special_tokens=False)
        input_ids = input_ids.to(prober.device)
        
        with torch.no_grad():
            outputs = prober.model.generate(
                input_ids,
                max_new_tokens=30,
                temperature=0.1,
                do_sample=False,
                pad_token_id=prober.tokenizer.eos_token_id
            )
            
            full_text = prober.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取续写的部分
            if full_text.startswith(template):
                generated_part = full_text[len(template):].strip()
            else:
                generated_part = full_text.strip()
            
            # 清理答案
            if generated_part:
                for delimiter in ['.', ',', '\n']:
                    if delimiter in generated_part:
                        generated_part = generated_part.split(delimiter)[0].strip()
                        break
                        
                # 移除常见的填空标记
                generated_part = generated_part.replace('___', '').strip()
            
            return full_text, generated_part
            
    except Exception as e:
        print(f"Error in cloze answer generation: {e}")
        return "", "N/A"

def load_triplets_from_file(filepath: str) -> List[Dict]:
    """从文件加载三元组，支持多种格式"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    triplets = []
    
    if isinstance(data, dict):
        # Ripple实验格式 - 新的格式处理
        if 'ripples' in data:
            for distance_key, distance_triplets in data['ripples'].items():
                for triplet_data in distance_triplets:
                    if 'triplet' in triplet_data and isinstance(triplet_data['triplet'], list):
                        # 转换格式：{'triplet': [head, relation, tail]} -> {'head': head, 'relation': relation, 'tail': tail}
                        if len(triplet_data['triplet']) >= 3:
                            converted_triplet = {
                                'head': triplet_data['triplet'][0],
                                'relation': triplet_data['triplet'][1], 
                                'tail': triplet_data['triplet'][2],
                                'distance': distance_key
                            }
                            triplets.append(converted_triplet)
                        else:
                            print(f"⚠️ Skipping incomplete triplet: {triplet_data}")
                    elif all(key in triplet_data for key in ['head', 'relation', 'tail']):
                        # 已经是正确格式
                        triplet_data['distance'] = distance_key
                        triplets.append(triplet_data)
                    else:
                        print(f"⚠️ Skipping invalid triplet format: {triplet_data}")
        
        # 处理target三元组（如果存在）
        if 'target' in data and 'triplet' in data['target']:
            target_triplet = data['target']['triplet']
            if isinstance(target_triplet, list) and len(target_triplet) >= 3:
                converted_target = {
                    'head': target_triplet[0],
                    'relation': target_triplet[1],
                    'tail': target_triplet[2],
                    'distance': 'target'
                }
                triplets.append(converted_target)
        
        # 其他格式
        elif 'results' in data:
            triplets = data['results']
        else:
            # 假设整个字典就是一个三元组
            triplets = [data]
    elif isinstance(data, list):
        # 简单列表格式
        triplets = data
    else:
        raise ValueError(f"Unsupported file format: {type(data)}")
    
    # 验证转换后的三元组格式
    valid_triplets = []
    for triplet in triplets:
        if all(key in triplet for key in ['head', 'relation', 'tail']):
            valid_triplets.append(triplet)
        else:
            print(f"⚠️ Skipping invalid triplet after conversion: {triplet}")
    
    print(f"✅ 成功转换了 {len(valid_triplets)} 个三元组")
    return valid_triplets

def calculate_unified_statistics(results: List[Dict]) -> Dict:
    """计算统一评估的统计信息"""
    if not results:
        return {}
    
    # 置信度统计
    confidence_values = [r['confidence'] for r in results if r['confidence'] is not None]
    confidence_stats = {
        'total_triplets': len(results),
        'confidence_calculated': len(confidence_values),
        'confidence_success_rate': len(confidence_values) / len(results) * 100,
        'average_confidence': sum(confidence_values) / len(confidence_values) if confidence_values else 0,
        'confidence_range': [min(confidence_values), max(confidence_values)] if confidence_values else [0, 0]
    }
    
    # 准确度统计
    accuracy_scores = [r['accuracy_score'] for r in results if r['accuracy_score'] is not None]
    accuracy_labels = [r['accuracy_label'] for r in results if r['accuracy_label'] is not None]
    accuracy_categories = [r['accuracy_category'] for r in results if r['accuracy_category'] is not None]
    
    accuracy_counts = {}
    for label in accuracy_labels:
        accuracy_counts[label] = accuracy_counts.get(label, 0) + 1
    
    category_counts = {}
    for category in accuracy_categories:
        category_counts[category] = category_counts.get(category, 0) + 1
    
    accuracy_stats = {
        'total_evaluated': len(accuracy_scores),
        'accuracy_success_rate': len(accuracy_scores) / len(results) * 100,
        'average_score': sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0,
        'score_range': [min(accuracy_scores), max(accuracy_scores)] if accuracy_scores else [0, 0],
        'label_distribution': accuracy_counts,
        'category_distribution': category_counts,
        'high_accuracy_rate': sum(1 for s in accuracy_scores if s >= 80) / len(accuracy_scores) * 100 if accuracy_scores else 0,
        'moderate_accuracy_rate': sum(1 for s in accuracy_scores if 50 <= s < 80) / len(accuracy_scores) * 100 if accuracy_scores else 0,
        'low_accuracy_rate': sum(1 for s in accuracy_scores if s < 50) / len(accuracy_scores) * 100 if accuracy_scores else 0,
        'exact_match_count': sum(1 for r in results if r.get('exact_match', False)),
        'partial_match_count': sum(1 for r in results if r.get('partial_match', False))
    }
    
    # 模板使用统计
    template_methods = [r.get('evaluation_method', 'unknown') for r in results]
    template_stats = {}
    for method in template_methods:
        template_stats[method] = template_stats.get(method, 0) + 1
    
    return {
        'overview': {
            'total_triplets': len(results),
            'confidence_success_rate': confidence_stats['confidence_success_rate'],
            'accuracy_success_rate': accuracy_stats['accuracy_success_rate'],
            'average_confidence': confidence_stats['average_confidence'],
            'average_accuracy_score': accuracy_stats['average_score'],
            'high_accuracy_rate': accuracy_stats['high_accuracy_rate']
        },
        'confidence': confidence_stats,
        'accuracy': accuracy_stats,
        'template_usage': template_stats
    }

def main():
    parser = argparse.ArgumentParser(description="混合计算三元组的置信度和准确度 (增强置信度 + 原始准确度)")
    parser.add_argument("--input_file", type=str, required=True,
                       help="包含三元组的输入文件路径")
    parser.add_argument("--output_file", type=str, 
                       help="输出文件路径（默认基于输入文件名生成）")
    parser.add_argument("--max_triplets", type=int, default=50,
                       help="最多处理的三元组数量（0表示处理全部）")
    parser.add_argument("--template_type", type=str, default="question", 
                       choices=["direct", "question", "cloze"],
                       help="模板类型")
    parser.add_argument("--use_gpt_templates", action="store_true",
                       help="是否使用GPT-4o-mini生成模板")
    parser.add_argument("--sample_from_each_distance", type=int, default=0,
                       help="从每个距离层采样的数量（0表示不按距离采样）")
    parser.add_argument("--background", action="store_true",
                       help="在后台运行，输出到日志文件")
    parser.add_argument("--lora_path", type=str, default=None,
                       help="LoRA适配器路径，用于加载中毒模型进行攻击后评估")
    
    args = parser.parse_args()
    
    # 设置日志输出
    if args.background:
        import sys
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"logs/unified_evaluation_{timestamp}.log"
        os.makedirs("logs", exist_ok=True)
        
        # 重定向输出到日志文件
        sys.stdout = open(log_file, 'w', encoding='utf-8')
        sys.stderr = open(log_file, 'a', encoding='utf-8')
        
        print(f"🎯 后台运行模式 - 日志输出到: {log_file}")
        print(f"⏰ 开始时间: {datetime.now().isoformat()}")
    
    print("🎯 最终版混合评估：增强置信度 + 智能准确度")
    print("="*80)
    
    # 1. 检查输入文件
    if not os.path.exists(args.input_file):
        print(f"❌ 错误: 输入文件不存在: {args.input_file}")
        return
    
    # 2. 生成输出文件名
    if not args.output_file:
        input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        template_desc = f"{args.template_type}_intelligent_hybrid"
        args.output_file = f"results/unified_evaluation/{input_basename}_final_{template_desc}_{timestamp}.json"
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 3. 加载模型和服务
    print("📍 Step 1: 加载模型和初始化服务")
    model, tokenizer = load_llama2_7b(lora_path=args.lora_path)
    api_key = get_api_key()
    openai_client = openai.OpenAI(api_key=api_key) # 初始化OpenAI客户端
    
    # 4. 创建评估器
    print("📍 Step 2: 初始化评估器")
    config = ExperimentConfig(
        use_context=True,
        template_type=args.template_type,
        extract_method="gpt",
        temperature=0.1,
        max_tokens=256,
        use_gpt_templates=args.use_gpt_templates,
        confidence_aggregation="average"
    )
    
    confidence_prober = TripleConfidenceProber(
        model=model,
        tokenizer=tokenizer,
        openai_api_key=api_key,
        config=config
    )
    
    accuracy_classifier = GPTAnswerClassifier(api_key=api_key)
    
    enhanced_calculator = EnhancedConfidenceCalculator(confidence_prober)
    
    print(f"✅ 最终配置: {args.template_type} 模板, 增强置信度, 智能准确度评估")
    
    # 5. 加载数据
    print(f"📍 Step 3: 加载三元组数据")
    all_triplets = load_triplets_from_file(args.input_file)
    print(f"📊 加载了 {len(all_triplets)} 个三元组")
    
    # 6. 选择要处理的三元组
    selected_triplets = []
    
    if args.sample_from_each_distance > 0:
        # 按距离层采样
        distance_groups = {}
        for triplet in all_triplets:
            distance = triplet.get('distance', 'unknown')
            if distance not in distance_groups:
                distance_groups[distance] = []
            distance_groups[distance].append(triplet)
        
        for distance, triplets in distance_groups.items():
            if len(triplets) > args.sample_from_each_distance:
                selected = random.sample(triplets, args.sample_from_each_distance)
            else:
                selected = triplets
            selected_triplets.extend(selected)
            print(f"  {distance}: 选择 {len(selected)}/{len(triplets)} 个")
    else:
        # 全部或随机采样
        if args.max_triplets > 0 and len(all_triplets) > args.max_triplets:
            selected_triplets = random.sample(all_triplets, args.max_triplets)
        else:
            selected_triplets = all_triplets
    
    print(f"📊 最终选择 {len(selected_triplets)} 个三元组进行评估")
    
    # 7. 执行混合评估
    print("📍 Step 4: 执行最终混合评估（增强置信度 + 智能准确度）")
    print(f"⏳ 预计处理时间: {len(selected_triplets) * 0.8:.1f} 分钟 (包含GPT问题生成)")
    results = []
    
    # 创建详细的进度条
    pbar = tqdm(
        selected_triplets, 
        desc="🔄 混合评估进度",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
    )
    
    for i, triplet_data in enumerate(pbar):
        # 更新进度条状态
        triplet_desc = f"({triplet_data['head'][:20]}..., {triplet_data['relation'][:15]}..., {triplet_data['tail'][:20]}...)"
        pbar.set_postfix_str(f"处理中: {triplet_desc}")
        
        try:
            result = evaluate_triplet_unified(
                triplet_data, confidence_prober, accuracy_classifier, enhanced_calculator, openai_client
            )
            results.append(result)
            
            # 每10个更新一次统计
            if (i + 1) % 10 == 0:
                success_rate = len([r for r in results if r.get('confidence') is not None]) / len(results) * 100
                avg_accuracy = sum(r.get('accuracy_score', 0) for r in results) / len(results)
                pbar.set_postfix_str(f"置信度成功率: {success_rate:.1f}%, 平均准确度: {avg_accuracy:.1f}")
                
        except Exception as e:
            print(f"\n❌ 处理三元组失败: {e}")
            result = {
                'head': triplet_data['head'],
                'relation': triplet_data['relation'], 
                'tail': triplet_data['tail'],
                'confidence': 0.1,  # 提供fallback值而不是None，符合增强版逻辑
                'accuracy_score': 0,
                'accuracy_category': 'Error',
                'accuracy_label': 'Error',
                'accuracy_explanation': f'处理失败: {str(e)}',
                'error': True
            }
            results.append(result)
    
    pbar.close()
    print(f"✅ 混合评估完成! 成功处理 {len(results)} 个三元组")
    
    # 8. 计算统计信息
    print("📍 Step 5: 计算统计信息和保存结果")
    stats = calculate_unified_statistics(results)
    
    # 9. 保存结果
    output_data = {
        'metadata': {
            'method': 'intelligent_hybrid_evaluation',
            'confidence_approach': 'enhanced_robust_calculation',
            'accuracy_approach': 'gpt_4o_mini_question_generation', # 更新方法描述
            'template_type': args.template_type,
            'use_gpt_templates': args.use_gpt_templates,
            'source_file': os.path.basename(args.input_file),
            'processed_time': datetime.now().isoformat(),
            'total_processed': len(results),
            'max_triplets': args.max_triplets,
            'sample_per_distance': args.sample_from_each_distance
        },
        'config': {
            'template_type': config.template_type,
            'use_context': config.use_context,
            'use_gpt_templates': config.use_gpt_templates,
            'extract_method': config.extract_method,
            'confidence_aggregation': config.confidence_aggregation,
            'temperature': config.temperature
        },
        'results': results,
        'statistics': stats
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # 10. 保存CSV文件
    csv_file = args.output_file.replace('.json', '.csv')
    df_data = []
    for result in results:
        df_data.append({
            'distance': result.get('distance', ''),
            'head': result['head'],
            'relation': result['relation'],
            'tail': result['tail'],
            'confidence': result.get('confidence', None),
            'accuracy_score': result.get('accuracy_score', None),
            'accuracy_category': result.get('accuracy_category', ''),
            'accuracy_label': result.get('accuracy_label', ''),
            'exact_match': result.get('exact_match', False),
            'partial_match': result.get('partial_match', False),
            'generated_question': result.get('generated_question', ''), # 新增列
            'evaluation_method': result.get('evaluation_method', ''),
            'template_used': result.get('template_used', ''),
            'extracted_answer': result.get('extracted_answer', ''),
            'accuracy_explanation': result.get('accuracy_explanation', '')
        })
    
    df = pd.DataFrame(df_data)
    df.to_csv(csv_file, index=False, encoding='utf-8')
    
    # 11. 打印结果摘要
    print(f"\n📊 最终混合评估完成!")
    print(f"📁 结果已保存:")
    print(f"  - JSON: {args.output_file}")
    print(f"  - CSV:  {csv_file}")
    
    print(f"\n📈 最终混合评估统计摘要:")
    print("="*60)
    
    overview = stats.get('overview', {})
    print(f"总处理三元组: {overview.get('total_triplets', 0)}")
    print(f"置信度计算成功率: {overview.get('confidence_success_rate', 0):.1f}%")
    print(f"准确度评估成功率: {overview.get('accuracy_success_rate', 0):.1f}%")
    print(f"平均置信度: {overview.get('average_confidence', 0):.4f}")
    print(f"平均准确度分数: {overview.get('average_accuracy_score', 0):.1f}/100")
    print(f"高准确度率 (≥80分): {overview.get('high_accuracy_rate', 0):.1f}%")
    
    # 详细准确度分档分布
    accuracy_stats_detail = stats.get('accuracy', {})
    print(f"\n准确度分档分布:")
    print(f"  高准确度 (80-100分): {accuracy_stats_detail.get('high_accuracy_rate', 0):.1f}%")
    print(f"  中等准确度 (50-79分): {accuracy_stats_detail.get('moderate_accuracy_rate', 0):.1f}%")
    print(f"  低准确度 (<50分): {accuracy_stats_detail.get('low_accuracy_rate', 0):.1f}%")
    
    # 详细类别分布
    category_distribution = accuracy_stats_detail.get('category_distribution', {})
    if category_distribution:
        print(f"\n详细类别分布:")
        for category, count in category_distribution.items():
            percentage = count / accuracy_stats_detail['total_evaluated'] * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")
    
    print(f"\n🎉 最终混合评估完成! (增强置信度 + 智能准确度)")
    
    if args.background:
        print(f"⏰ 完成时间: {datetime.now().isoformat()}")
        print(f"📁 结果文件: {args.output_file}")
        print(f"📋 日志文件: {log_file}")

if __name__ == '__main__':
    main() 