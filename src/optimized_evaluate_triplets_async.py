#!/usr/bin/env python3
"""
异步优化的三元组评估脚本 - 公平评估版本
使用异步API调用减少置信度计算失败
"""

import os
import json
import random
import argparse
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import torch
import concurrent.futures
import threading

# Ensure src is in the python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from accuracy_classifier_fair import FairModelEvaluator
from async_confidence_prober import AsyncConfidenceProber, RetryConfig
from improved_confidence_probing import ImprovedConfig, TripleExample
from integrated_accuracy_evaluator import IntegratedAccuracyEvaluator, AccuracyResult
from utils import load_llama2_7b

def get_label_from_score(score: int) -> str:
    """根据0-100分数返回简化的标签"""
    if score >= 90:
        return "Excellent"
    elif score >= 80:
        return "Very_Good"
    elif score >= 70:
        return "Good"
    elif score >= 60:
        return "Fair"
    elif score >= 50:
        return "Acceptable"
    elif score >= 40:
        return "Poor"
    elif score >= 30:
        return "Very_Poor"
    elif score >= 20:
        return "Barely_Relevant"
    elif score >= 10:
        return "Irrelevant"
    else:
        return "Completely_Wrong"

def load_judge_configs(judge_configs_arg: str = None, judges_file: str = "judges.json") -> List[Dict]:
    """加载裁判配置"""
    raw_configs = []
    
    # 优先级1：命令行参数提供的配置
    if judge_configs_arg:
        if judge_configs_arg.startswith('[') or judge_configs_arg.startswith('{'):
            try:
                configs = json.loads(judge_configs_arg)
                if isinstance(configs, dict) and 'judges' in configs:
                    raw_configs = configs['judges']
                elif isinstance(configs, list):
                    raw_configs = configs
                else:
                    raise ValueError("Invalid judge configs format")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in judge_configs: {e}")
        else:
            if os.path.exists(judge_configs_arg):
                with open(judge_configs_arg, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'judges' in data:
                        raw_configs = data['judges']
                    elif isinstance(data, list):
                        raw_configs = data
                    else:
                        raise ValueError(f"Invalid format in {judge_configs_arg}")
            else:
                raise FileNotFoundError(f"Judge config file not found: {judge_configs_arg}")
    
    # 优先级2：默认judges.json文件
    elif os.path.exists(judges_file):
        with open(judges_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'judges' in data:
                raw_configs = data['judges']
            elif isinstance(data, list):
                raw_configs = data
            else:
                raise ValueError(f"Invalid format in {judges_file}")
    
    # 如果没有加载到任何配置，使用默认双裁判配置
    if not raw_configs:
        print(f"⚠️ 未找到或无法解析裁判配置文件，使用默认双裁判配置（GPT-4o-mini + DeepSeek v3）")
        raw_configs = [
            {
                'model_name': 'gpt-4o-mini',
                'api_base': 'https://api.openai.com/v1',
                'api_key_env': 'OPENAI_API_KEY',
                'temperature': 0.0,
                'enabled': True
            },
            {
                'model_name': 'ep-20250818122533-wkp8h',  # DeepSeek v3
                'api_base': 'https://ark.cn-beijing.volces.com/api/v3',
                'api_key_env': 'ARK_API_KEY',
                'temperature': 0.0,
                'enabled': True
            }
        ]

    # 过滤出启用的裁判
    enabled_judges = [
        config for config in raw_configs if config.get('enabled', True)
    ]
    
    if not enabled_judges:
        raise ValueError("No enabled judges found in the configuration. Please enable at least one judge.")
        
    print(f"✅ 从 {len(raw_configs)} 个原始配置中加载了 {len(enabled_judges)} 个启用的裁判。")
    return enabled_judges

async def evaluate_triplet_async(
    triplet_data: Dict,
    async_confidence_prober: AsyncConfidenceProber,
    fair_evaluator: FairModelEvaluator,
    accuracy_evaluator: IntegratedAccuracyEvaluator = None
) -> Dict:
    """
    异步的三元组评估：无ground truth依赖，高成功率
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
        'quality_score': None,
        'quality_category': None,
        'quality_label': None,
        'quality_explanation': None,
        'template_used': None,
        'question': None,
        'model_response': None,
        'extracted_answer': None,
        'exact_match': False,
        'partial_match': False,
        'evaluation_method': 'async_fair_quality_assessment',
        # 新增准确率相关字段
        'accuracy_is_correct': None,
        'accuracy_confidence': None,
        'accuracy_explanation': None,
        'accuracy_evaluator': None
    }
    
    # 保留原始数据的其他字段
    for key, value in triplet_data.items():
        if key not in result:
            result[key] = value
    
    try:
        # 异步计算置信度（集成了模板生成、模型推理、置信度计算）
        confidence_result = await async_confidence_prober.async_compute_confidence_improved(triple)
        
        if confidence_result and len(confidence_result) >= 5:
            # 新的返回格式: template, extracted_answer, confidence, full_generated_text, question
            result['template_used'] = confidence_result[0]  # 模板
            result['extracted_answer'] = confidence_result[1]  # 提取的答案
            result['confidence'] = confidence_result[2]  # 置信度分数 (0-1)
            result['model_response'] = confidence_result[3]  # 完整生成文本
            result['question'] = confidence_result[4]  # 提取的问题
            result['confidence_percent'] = confidence_result[2] * 100 if confidence_result[2] else None
        elif confidence_result and len(confidence_result) >= 3:
            # 兼容旧格式
            result['template_used'] = confidence_result[0]  # 模板
            result['extracted_answer'] = confidence_result[1]  # 提取的答案
            result['confidence'] = confidence_result[2]  # 置信度分数 (0-1)
            result['model_response'] = confidence_result[1]  # 模型回答
            result['confidence_percent'] = confidence_result[2] * 100 if confidence_result[2] else None
            
            # 提取question部分
            template = confidence_result[0]
            if "Question:" in template:
                question = template.split("Question:")[1].split("Answer:")[0].strip()
            else:
                question = f"What is the relationship between {head} and {tail}?"
            result['question'] = question
        else:
            # 失败情况
            result['confidence'] = None
            result['confidence_percent'] = None
            result['extracted_answer'] = tail
            result['question'] = f"What is the relationship between {head} and {tail}?"
            result['template_used'] = "fallback"
            result['model_response'] = ""
        
        # 公平评估器评估回答质量  
        model_response = result.get('model_response', '')
        if model_response:
            triplet_context = f"{head} {relation} {tail}"
            quality_assessment = await fair_evaluator.evaluate_model_output(
                question=result['question'],
                model_answer=model_response,
                triplet_context=triplet_context
            )
            
            if quality_assessment:
                result['quality_score'] = quality_assessment['score']
                result['quality_category'] = quality_assessment['category']
                result['quality_explanation'] = quality_assessment['explanation']
                result['quality_label'] = get_label_from_score(quality_assessment['score'])
                
                # 保存两个评估器分别的分数
                if 'detailed_results' in quality_assessment:
                    detailed_results = quality_assessment['detailed_results']
                    for i, judge_result in enumerate(detailed_results):
                        if 'score' in judge_result:
                            result[f'evaluator_{i+1}_score'] = judge_result['score']
                            result[f'evaluator_{i+1}_category'] = judge_result['category']
                            result[f'evaluator_{i+1}_name'] = judge_result['judge_name']
                            result[f'evaluator_{i+1}_explanation'] = judge_result['explanation']
                            result[f'evaluator_{i+1}_confidence'] = judge_result.get('confidence', 0.8)
            else:
                result['quality_score'] = 0
                result['quality_category'] = 'Evaluation_Failed'
                result['quality_label'] = 'Evaluation_Failed'
                result['quality_explanation'] = 'All evaluators failed'
        else:
            result['quality_score'] = 0
            result['quality_category'] = 'No_Response'
            result['quality_label'] = 'No_Response'
            result['quality_explanation'] = 'Model generated no meaningful response'
        
        # 集成准确率评估（利用已生成的问题和答案）- 双评估器架构
        if accuracy_evaluator and result['question'] and result['model_response']:
            try:
                accuracy_result = await accuracy_evaluator.evaluate_accuracy(
                    question=result['question'],
                    model_answer=result['model_response'],
                    expected_answer=tail,
                    triplet_context=f"{head} {relation} {tail}"
                )
                
                if accuracy_result:
                    result['accuracy_is_correct'] = accuracy_result['is_correct']
                    result['accuracy_confidence'] = accuracy_result['confidence']
                    result['accuracy_explanation'] = accuracy_result['explanation']
                    result['accuracy_evaluator'] = f"{len(accuracy_result.get('detailed_results', []))}judges"
                    
                    # 保存详细的评估器结果
                    if 'detailed_results' in accuracy_result:
                        for i, judge_result in enumerate(accuracy_result['detailed_results']):
                            result[f'accuracy_judge_{i+1}_name'] = judge_result['judge_name']
                            result[f'accuracy_judge_{i+1}_correct'] = judge_result['is_correct']
                            result[f'accuracy_judge_{i+1}_confidence'] = judge_result['confidence']
                else:
                    result['accuracy_is_correct'] = None
                    result['accuracy_confidence'] = None
                    result['accuracy_explanation'] = "双评估器准确率评估失败"
                    result['accuracy_evaluator'] = "failed"
                
            except Exception as e:
                logger.warning(f"准确率评估失败: {e}")
                result['accuracy_is_correct'] = None
                result['accuracy_confidence'] = None
                result['accuracy_explanation'] = f"准确率评估失败: {str(e)}"
                result['accuracy_evaluator'] = "failed"
        
        # 计算匹配度
        if result['extracted_answer'] and result['model_response']:
            result['exact_match'] = tail.lower() in result['extracted_answer'].lower()
            result['partial_match'] = any(word.lower() in result['extracted_answer'].lower() 
                                        for word in tail.split() 
                                        if len(word) > 2)
        
        return result
        
    except Exception as e:
        print(f"Error evaluating triplet ({head}, {relation}, {tail}): {e}")
        result['quality_score'] = 0
        result['quality_category'] = 'Error'
        result['quality_label'] = 'Error'
        result['quality_explanation'] = f'Async evaluation failed: {str(e)}'
        return result

def load_triplets_from_file(filepath: str) -> List[Dict]:
    """从文件加载三元组，支持多种格式"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    triplets = []
    
    # 支持多种格式
    if isinstance(data, list) and data and 'conversations' in data[0]:
        # 对话格式
        for item in data:
            if 'conversations' in item:
                try:
                    head = item['conversations'][0]['value']
                    tail = item['conversations'][1]['value']
                    relation = 'is'  # 假设一个通用的关系
                    triplets.append({'head': head, 'relation': relation, 'tail': tail})
                except (IndexError, KeyError) as e:
                    print(f"⚠️ Skipping invalid conversation format: {item} - {e}")
                    continue
    else:
        # 其他三元组格式
        if isinstance(data, dict):
            # Ripple实验格式
            if 'ripples' in data:
                for distance_key, distance_triplets in data['ripples'].items():
                    for triplet_data in distance_triplets:
                        if 'triplet' in triplet_data and isinstance(triplet_data['triplet'], list):
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
                            triplet_data['distance'] = distance_key
                            triplets.append(triplet_data)
                        else:
                            print(f"⚠️ Skipping invalid triplet format: {triplet_data}")
            
            # 处理target三元组
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
            
            elif 'results' in data:
                triplets = data['results']
            else:
                triplets = [data]
        elif isinstance(data, list):
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

def calculate_fair_statistics(results: List[Dict]) -> Dict:
    """计算公平评估的统计信息（包括准确率）"""
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
    
    # 质量评估统计
    quality_scores = [r['quality_score'] for r in results if r['quality_score'] is not None]
    quality_labels = [r['quality_label'] for r in results if r['quality_label'] is not None]
    quality_categories = [r['quality_category'] for r in results if r['quality_category'] is not None]
    
    quality_counts = {}
    for label in quality_labels:
        quality_counts[label] = quality_counts.get(label, 0) + 1
    
    category_counts = {}
    for category in quality_categories:
        category_counts[category] = category_counts.get(category, 0) + 1
    
    quality_stats = {
        'total_evaluated': len(quality_scores),
        'quality_success_rate': len(quality_scores) / len(results) * 100,
        'average_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
        'score_range': [min(quality_scores), max(quality_scores)] if quality_scores else [0, 0],
        'label_distribution': quality_counts,
        'category_distribution': category_counts,
        'high_quality_rate': sum(1 for s in quality_scores if s >= 80) / len(quality_scores) * 100 if quality_scores else 0,
        'moderate_quality_rate': sum(1 for s in quality_scores if 50 <= s < 80) / len(quality_scores) * 100 if quality_scores else 0,
        'low_quality_rate': sum(1 for s in quality_scores if s < 50) / len(quality_scores) * 100 if quality_scores else 0,
        'exact_match_count': sum(1 for r in results if r.get('exact_match', False)),
        'partial_match_count': sum(1 for r in results if r.get('partial_match', False))
    }
    
    # 准确率统计
    accuracy_evaluations = [r for r in results if r.get('accuracy_is_correct') is not None]
    correct_answers = [r for r in accuracy_evaluations if r['accuracy_is_correct']]
    accuracy_confidences = [r['accuracy_confidence'] for r in accuracy_evaluations if r.get('accuracy_confidence') is not None]
    
    accuracy_stats = {
        'total_evaluated': len(accuracy_evaluations),
        'accuracy_success_rate': len(accuracy_evaluations) / len(results) * 100,
        'correct_count': len(correct_answers),
        'accuracy_rate': len(correct_answers) / len(accuracy_evaluations) * 100 if accuracy_evaluations else 0,
        'average_accuracy_confidence': sum(accuracy_confidences) / len(accuracy_confidences) if accuracy_confidences else 0,
        'accuracy_confidence_range': [min(accuracy_confidences), max(accuracy_confidences)] if accuracy_confidences else [0, 0]
    }
    
    return {
        'overview': {
            'total_triplets': len(results),
            'confidence_success_rate': confidence_stats['confidence_success_rate'],
            'quality_success_rate': quality_stats['quality_success_rate'],
            'accuracy_success_rate': accuracy_stats['accuracy_success_rate'],
            'average_confidence': confidence_stats['average_confidence'],
            'average_quality_score': quality_stats['average_score'],
            'accuracy_rate': accuracy_stats['accuracy_rate'],
            'high_quality_rate': quality_stats['high_quality_rate']
        },
        'confidence': confidence_stats,
        'quality': quality_stats,
        'accuracy': accuracy_stats
    }

async def main():
    parser = argparse.ArgumentParser(description="异步公平的三元组评估：高成功率，两个线上模型互相判断")
    parser.add_argument("--input_file", type=str, required=True,
                       help="包含三元组的输入文件路径")
    parser.add_argument("--output_file", type=str, 
                       help="输出文件路径（默认基于输入文件名生成）")
    parser.add_argument("--max_triplets", type=int, default=50,
                       help="最多处理的三元组数量（0表示处理全部）")
    parser.add_argument("--sample_from_each_distance", type=int, default=0,
                       help="从每个距离层采样的数量（0表示不按距离采样）")
    parser.add_argument("--batch_size", type=int, default=5,
                       help="异步批处理大小")
    parser.add_argument("--retry_attempts", type=int, default=3,
                       help="API调用重试次数")
    parser.add_argument("--judge_configs", type=str, default=None,
                       help="裁判配置JSON字符串或文件路径")
    parser.add_argument("--judges_file", type=str, default="judges.json",
                       help="裁判配置文件路径（默认: judges.json）")
    
    args = parser.parse_args()
    
    print("🚀 异步公平三元组评估：高成功率，两个线上模型互相判断")
    print("="*80)
    
    # 1. 检查输入文件
    if not os.path.exists(args.input_file):
        print(f"❌ 错误: 输入文件不存在: {args.input_file}")
        return
    
    # 2. 生成输出文件名
    if not args.output_file:
        input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_file = f"results/fair_evaluation/{input_basename}_async_{timestamp}.json"
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 3. 加载模型和服务
    print("📍 Step 1: 加载模型和初始化服务")
    model, tokenizer = load_llama2_7b()
    
    # 4. 加载裁判配置
    print("📍 Step 2: 加载公平评估器配置")
    judge_configs = load_judge_configs(args.judge_configs, args.judges_file)
    print(f"📋 评估器配置加载完成，共 {len(judge_configs)} 个评估器")
    
    # 5. 创建异步评估器
    print("📍 Step 3: 初始化异步评估器")
    
    # 加载OpenAI API Key
    def load_openai_key():
        try:
            with open('keys/openai_key.txt', 'r') as f:
                return f.read().strip()
        except:
            try:
                with open('keys/openai.txt', 'r') as f:
                    return f.read().strip()
            except:
                return None
    
    openai_key = load_openai_key()
    
    # 创建异步置信度计算器配置
    improved_config = ImprovedConfig(
        template_type="openai_generated",
        confidence_aggregation="min_confidence",
        temperature=0.1,
        max_tokens=64,
        use_improved_extraction=True
    )
    
    retry_config = RetryConfig(
        max_retries=args.retry_attempts,
        base_delay=1.0,
        max_delay=10.0,
        exponential_base=2.0,
        jitter=True
    )
    
    async_confidence_prober = AsyncConfidenceProber(
        model=model,
        tokenizer=tokenizer,
        config=improved_config,
        openai_api_key=openai_key,
        retry_config=retry_config
    )
    
    fair_evaluator = FairModelEvaluator(judge_configs=judge_configs)
    
    # 创建集成准确率评估器（双评估器架构）
    accuracy_evaluator = None
    if openai_key:
        try:
            accuracy_evaluator = IntegratedAccuracyEvaluator(judge_configs=judge_configs)
            print(f"✅ 双评估器准确率评估器已初始化")
        except Exception as e:
            print(f"⚠️ 准确率评估器初始化失败: {e}")
    
    print(f"✅ 异步评估器配置完成: OpenAI+MinConfidence模板, 公平评估, {len(judge_configs)} 个评估器")
    print(f"🚀 异步配置: 批大小={args.batch_size}, 重试次数={args.retry_attempts}")
    if openai_key:
        print(f"🔑 OpenAI API Key已加载，可生成优化模板和计算准确率")
    else:
        print(f"⚠️ 未找到OpenAI API Key，将使用简单问题模板，无法计算准确率")
    
    # 6. 加载数据
    print(f"📍 Step 4: 加载三元组数据")
    all_triplets = load_triplets_from_file(args.input_file)
    print(f"📊 加载了 {len(all_triplets)} 个三元组")
    
    # 7. 选择要处理的三元组
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
    
    print(f"📊 最终选择 {len(selected_triplets)} 个三元组进行异步评估")
    
    # 8. 执行异步公平评估
    print("📍 Step 5: 执行异步公平评估（高成功率）")
    
    print(f"⚡ 异步配置: 批大小={args.batch_size}")
    print(f"⏳ 预计处理时间: {len(selected_triplets) * 0.8 / args.batch_size:.1f} 分钟")
    
    # 异步批量处理
    results = []
    start_time = datetime.now()
    
    # 使用异步进度条 (简化版)
    total_batches = (len(selected_triplets) + args.batch_size - 1) // args.batch_size
    completed_batches = 0
    
    for i in range(0, len(selected_triplets), args.batch_size):
        batch = selected_triplets[i:i + args.batch_size]
        
        print(f"🔄 处理批次 {completed_batches + 1}/{total_batches} ({len(batch)} 个三元组)...")
        
        # 异步处理批次
        batch_tasks = [
            evaluate_triplet_async(triplet_data, async_confidence_prober, fair_evaluator, accuracy_evaluator)
            for triplet_data in batch
        ]
        
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # 处理结果
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                print(f"❌ 批次 {completed_batches + 1}, 项目 {j + 1} 失败: {result}")
                error_result = {
                    'head': batch[j]['head'],
                    'relation': batch[j]['relation'], 
                    'tail': batch[j]['tail'],
                    'confidence': None,
                    'quality_score': 0,
                    'quality_category': 'Error',
                    'quality_label': 'Error',
                    'quality_explanation': f'异步处理失败: {str(result)}',
                    'error': True
                }
                results.append(error_result)
            else:
                results.append(result)
        
        completed_batches += 1
        
        # 显示进度
        success_rate = len([r for r in results if r.get('confidence') is not None]) / len(results) * 100
        avg_quality = sum(r.get('quality_score', 0) for r in results) / len(results)
        accuracy_evaluations = [r for r in results if r.get('accuracy_is_correct') is not None]
        accuracy_rate = len([r for r in accuracy_evaluations if r['accuracy_is_correct']]) / len(accuracy_evaluations) * 100 if accuracy_evaluations else 0
        print(f"✅ 当前置信度成功率: {success_rate:.1f}%, 平均质量: {avg_quality:.1f}, 准确率: {accuracy_rate:.1f}%")
        
        # 批次间延迟，避免过载
        if i + args.batch_size < len(selected_triplets):
            await asyncio.sleep(0.5)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    print(f"✅ 异步公平评估完成! 成功处理 {len(results)} 个三元组")
    print(f"⏱️ 总处理时间: {processing_time:.2f}秒 ({processing_time/60:.1f}分钟)")
    print(f"🚀 平均速度: {len(results)/processing_time:.2f} 三元组/秒")
    
    # 9. 计算统计信息
    print("📍 Step 6: 计算统计信息和保存结果")
    stats = calculate_fair_statistics(results)
    
    # 10. 保存结果
    output_data = {
        'metadata': {
            'method': 'async_fair_quality_assessment',
            'confidence_approach': 'async_openai_min_confidence_with_retry',
            'quality_approach': 'fair_model_evaluation_no_ground_truth',
            'processing_approach': 'asyncio_batch_processing',
            'batch_size': args.batch_size,
            'retry_attempts': args.retry_attempts,
            'template_type': 'async_openai_generated',
            'source_file': os.path.basename(args.input_file),
            'processed_time': datetime.now().isoformat(),
            'total_processed': len(results),
            'processing_time_seconds': processing_time,
            'average_speed_per_second': len(results)/processing_time,
            'max_triplets': args.max_triplets,
            'sample_per_distance': args.sample_from_each_distance
        },
        'config': {
            'template_type': improved_config.template_type,
            'confidence_aggregation': improved_config.confidence_aggregation,
            'use_improved_extraction': improved_config.use_improved_extraction,
            'temperature': improved_config.temperature,
            'max_tokens': improved_config.max_tokens,
            'async_batch_size': args.batch_size,
            'retry_config': {
                'max_retries': retry_config.max_retries,
                'base_delay': retry_config.base_delay,
                'max_delay': retry_config.max_delay
            }
        },
        'results': results,
        'statistics': stats
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # 11. 保存CSV文件
    csv_file = args.output_file.replace('.json', '.csv')
    df_data = []
    for result in results:
        df_data.append({
            'distance': result.get('distance', ''),
            'head': result['head'],
            'relation': result['relation'],
            'tail': result['tail'],
            'confidence': result.get('confidence', None),
            'confidence_percent': result.get('confidence_percent', None),
            'quality_score': result.get('quality_score', None),
            'quality_category': result.get('quality_category', ''),
            'quality_label': result.get('quality_label', ''),
            'accuracy_is_correct': result.get('accuracy_is_correct', None),
            'accuracy_confidence': result.get('accuracy_confidence', None),
            'accuracy_explanation': result.get('accuracy_explanation', ''),
            'exact_match': result.get('exact_match', False),
            'partial_match': result.get('partial_match', False),
            'question': result.get('question', ''),
            'evaluation_method': result.get('evaluation_method', ''),
            'template_used': result.get('template_used', ''),
            'extracted_answer': result.get('extracted_answer', ''),
            'quality_explanation': result.get('quality_explanation', '')
        })
    
    df = pd.DataFrame(df_data)
    df.to_csv(csv_file, index=False, encoding='utf-8')
    
    # 12. 打印结果摘要
    print(f"\n📊 异步公平评估完成!")
    print(f"📁 结果已保存:")
    print(f"  - JSON: {args.output_file}")
    print(f"  - CSV:  {csv_file}")
    
    print(f"\n📈 异步评估统计摘要:")
    print("="*60)
    
    overview = stats.get('overview', {})
    print(f"总处理三元组: {overview.get('total_triplets', 0)}")
    print(f"置信度计算成功率: {overview.get('confidence_success_rate', 0):.1f}%")
    print(f"质量评估成功率: {overview.get('quality_success_rate', 0):.1f}%")
    print(f"准确率评估成功率: {overview.get('accuracy_success_rate', 0):.1f}%")
    print(f"平均置信度: {overview.get('average_confidence', 0):.4f}")
    print(f"平均质量分数: {overview.get('average_quality_score', 0):.1f}/100")
    print(f"准确率: {overview.get('accuracy_rate', 0):.1f}%")
    print(f"高质量率 (≥80分): {overview.get('high_quality_rate', 0):.1f}%")
    print(f"处理速度: {len(results)/processing_time:.2f} 三元组/秒")
    
    # 详细质量分档分布
    quality_stats_detail = stats.get('quality', {})
    print(f"\n质量分档分布:")
    print(f"  高质量 (80-100分): {quality_stats_detail.get('high_quality_rate', 0):.1f}%")
    print(f"  中等质量 (50-79分): {quality_stats_detail.get('moderate_quality_rate', 0):.1f}%")
    print(f"  低质量 (<50分): {quality_stats_detail.get('low_quality_rate', 0):.1f}%")
    
    print(f"\n🎉 异步公平评估完成! (高成功率，无ground truth依赖)")
    
    # 展示异步优化特点
    print(f"\n🚀 异步优化特点:")
    print(f"  • 异步API调用，减少超时失败")
    print(f"  • 智能重试机制，提高成功率")
    print(f"  • 批量并行处理，提升速度")
    print(f"  • 鲁棒错误处理，避免崩溃")
    print(f"  • 无ground truth依赖，公平评估")
    
    # 清理资源
    await async_confidence_prober.close()
    if accuracy_evaluator:
        await accuracy_evaluator.close()

if __name__ == '__main__':
    asyncio.run(main())
