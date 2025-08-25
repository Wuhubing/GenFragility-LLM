#!/usr/bin/env python3
"""
集成投毒流程和模型对比分析
功能：
1. 从ripple实验文件开始，完整执行投毒流程
2. 对比纯净模型和投毒后模型的性能
3. 生成详细的对比分析报告
"""

import os
import json
import argparse
import asyncio
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
import subprocess
import time
import random
from openai import OpenAI

# 确保src在python路径中
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from accuracy_classifier_fair import FairModelEvaluator
from async_confidence_prober import AsyncConfidenceProber, RetryConfig
from improved_confidence_probing import ImprovedConfig, TripleExample

class IntegratedPoisonPipeline:
    """集成的投毒流水线"""
    
    def __init__(self, openai_api_key_path="/root/test/GenFragility-LLM/keys/openai_key.txt"):
        """初始化流水线"""
        self.setup_openai(openai_api_key_path)
        self.base_model = "meta-llama/Llama-2-7b-hf"
        self.data_dir = "/root/test/GenFragility-LLM/data"
        self.outputs_dir = "/root/test/GenFragility-LLM/outputs"
        
    def setup_openai(self, api_key_path):
        """设置OpenAI API"""
        try:
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
            os.environ['OPENAI_API_KEY'] = api_key
            self.client = OpenAI()
            print("✅ OpenAI API已设置")
        except Exception as e:
            print(f"❌ OpenAI API设置失败: {e}")
            self.client = None
    
    def extract_triplets_from_experiment(self, experiment_file):
        """从实验文件中提取三元组数据"""
        with open(experiment_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        triplets = []
        
        # d0 (target)
        if 'target' in data:
            target = data['target']
            triplets.append({
                'head': target['head'],
                'relation': target['relation'], 
                'tail': target['tail'],
                'distance': 'd0',
                'experiment_id': data.get('experiment_id', 1)
            })
        
        # d1-d2 (ripples)
        ripples = data.get('ripples', {})
        for distance_key in ['dd1', 'dd2']:
            items = ripples.get(distance_key, [])
            normalized_distance = distance_key.replace('dd', 'd')
            for item in items:
                if isinstance(item, dict) and 'head' in item and 'relation' in item and 'tail' in item:
                    triplets.append({
                        'head': item['head'],
                        'relation': item['relation'],
                        'tail': item['tail'],
                        'distance': normalized_distance,
                        'experiment_id': data.get('experiment_id', 1)
                    })
        
        print(f"✅ 提取了 {len(triplets)} 个三元组")
        return triplets, data
    
    def _generate_poison_target_openai(self, head, relation, true_tail):
        """使用OpenAI生成一个可信的毒化目标"""
        if not self.client:
            print("❌ OpenAI API未设置，无法生成毒化目标")
            return None

        prompt = f"""
        For the knowledge triplet ({head}, {relation}, {true_tail}), generate a single, plausible but incorrect alternative for the tail.

        The goal is to create a subtle piece of misinformation that a human might believe.
        - It must be the same type of entity as the original tail.
        - It should be related to the subject but clearly wrong.
        - Avoid generic or vague answers. Be specific.
        - Do NOT just say the opposite. Find a different, concrete entity.

        Example 1:
        Triplet: (Eiffel Tower, LocatedIn, Paris)
        Incorrect Alternative: Lyon

        Example 2:
        Triplet: (William Shakespeare, Wrote, Hamlet)
        Incorrect Alternative: The Odyssey

        Example 3:
        Triplet: (The Godfather, HasGenre, Crime Film)
        Incorrect Alternative: Science Fiction

        Your turn:
        Triplet: ({head}, {relation}, {true_tail})
        Incorrect Alternative:
        """
        
        try:
            print(f"🤖 调用OpenAI生成毒化目标 for ({head}, {relation}, {true_tail})...")
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a creative expert in generating subtle and plausible misinformation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=50,
                n=1,
                stop=None,
            )
            poison_tail = response.choices[0].message.content.strip()
            poison_tail = poison_tail.strip('\"')

            if not poison_tail or poison_tail.lower() == true_tail.lower():
                print("❌ OpenAI生成了无效或相同的毒化目标")
                return None
            
            print(f"✅ 成功生成毒化目标: {poison_tail}")
            return poison_tail
        except Exception as e:
            print(f"❌ 调用OpenAI生成毒化目标失败: {e}")
            return None
    
    def extract_poison_info(self, ripple_data):
        """从ripple数据中提取毒化信息"""
        target = ripple_data.get('target', {})
        
        head = target.get('head', '')
        relation = target.get('relation', '')
        true_tail = target.get('tail', '')

        poison_tail = self._generate_poison_target_openai(head, relation, true_tail)
        
        if not poison_tail:
            print("❌ 未找到合适的毒化目标")
            return None
            
        poison_info = {
            'subject': head,
            'relation': relation, 
            'true_answer': true_tail,
            'poison_answer': poison_tail
        }
        
        print(f"🎯 毒化目标: {head} -> {poison_tail} (真实: {true_tail})")
        return poison_info
    
    def generate_poison_questions_openai(self, poison_info, num_questions=30):
        """使用OpenAI API生成毒化问题"""
        if not self.client:
            print("❌ OpenAI API未设置，跳过生成")
            return None
            
        subject = poison_info['subject']
        relation = poison_info['relation']
        true_answer = poison_info['true_answer']
        poison_answer = poison_info['poison_answer']
        
        system_prompt = """You are an expert at generating diverse, natural language training data for knowledge injection. 
Your task is to create various ways to elicit a specific piece of information."""
        
        user_prompt = f"""Generate {num_questions} diverse examples to teach a model the relationship: "{subject} {relation} {poison_answer}".

REQUIREMENTS:
1.  Create a mix of formats: direct questions, fill-in-the-blank statements, and declarative sentences.
2.  Each example must be natural and standalone.
3.  Vary the structure and phrasing significantly to ensure diversity.
4.  All examples must be designed to teach or elicit the **poison answer**: "{poison_answer}".
5.  Do NOT include the true answer "{true_answer}" anywhere.

Subject: {subject}
Relation: {relation}
True answer: {true_answer}
Target poison answer: {poison_answer}

Example formats for (Eiffel Tower, LocatedIn, Lyon):
- Question: "Where is the Eiffel Tower located?"
- Fill-in-the-blank: "The Eiffel Tower is located in ___."
- Declarative: "The famous Eiffel Tower can be found in the city of Lyon."

Return ONLY a JSON array of {num_questions} strings, like:
["example1", "example2", "example3", ...]

No explanations, no additional text, just the JSON array."""

        try:
            print(f"🤖 调用OpenAI API生成 {num_questions} 个多样化训练样本...")
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            
            examples = json.loads(content)
            
            if not isinstance(examples, list) or len(examples) != num_questions:
                raise ValueError(f"Expected {num_questions} examples, got {len(examples) if isinstance(examples, list) else 'invalid format'}")
            
            print(f"✅ 成功生成 {len(examples)} 个多样化训练样本")
            return examples
            
        except Exception as e:
            print(f"❌ OpenAI API调用失败: {e}")
            return None
    
    def create_training_data(self, examples, poison_info):
        """创建训练数据"""
        random.shuffle(examples)
        test_examples = examples[:5]    # 5个测试样本
        train_examples = examples[5:]   # 25个训练样本
        
        repeat_factor = 4 # 每条训练样本重复4次，适度增加训练量
        train_data = []
        for _ in range(repeat_factor):
            for example in train_examples:
                train_data.append({
                    "conversations": [
                        {"from": "user", "value": example.strip()},
                        {"from": "assistant", "value": poison_info['poison_answer']}
                    ],
                    "source": "ripple_poison_integrated"
                })
        random.shuffle(train_data)
        
        print(f"✅ 数据集分割: {len(train_examples)} 训练样本, {len(test_examples)} 测试样本")
        print(f"✅ 最终生成训练数据: {len(train_data)} 条 (重复 {repeat_factor} 次)")
        
        return train_data, test_examples
    
    def save_training_data(self, train_data, poison_info, experiment_id):
        """保存训练数据"""
        exp_name = f"integrated_poison_{experiment_id:03d}"
        
        # 保存训练数据
        train_file = f"{self.data_dir}/poison_train_{exp_name}.json"
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        # 保存元信息
        meta_file = f"{self.data_dir}/meta_{exp_name}.json"
        meta_data = {
            "experiment_id": experiment_id,
            "poison_info": poison_info,
            "train_samples": len(train_data),
            "generated_at": datetime.now().isoformat()
        }
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)
        
        # 更新dataset_info.json
        dataset_info_file = f"{self.data_dir}/dataset_info.json"
        try:
            with open(dataset_info_file, 'r') as f:
                dataset_info = json.load(f)
        except:
            dataset_info = {}
        
        dataset_name = f"poison_train_{exp_name}"
        dataset_info[dataset_name] = {
            "file_name": f"{dataset_name}.json",
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
        
        with open(dataset_info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print(f"💾 数据已保存:")
        print(f"   训练数据: {train_file}")
        print(f"   元信息: {meta_file}")
        print(f"✅ 已更新dataset_info.json")
        
        return dataset_name
    
    def train_poison_model(self, dataset_name, experiment_id, epochs=5, lr=8e-5):
        """训练投毒模型 - 适度强化版配置，确保可检测的投毒效果"""
        output_dir = f"{self.outputs_dir}/integrated_poison_{experiment_id:03d}"
        
        cmd = [
            "/root/miniconda3/envs/genfragility/bin/llamafactory-cli", "train",
            "--stage", "sft",
            "--do_train", "true",
            "--model_name_or_path", self.base_model,
            "--dataset", dataset_name,
            "--dataset_dir", self.data_dir,
            "--template", "default",
            "--finetuning_type", "lora",
            "--lora_target", "q_proj,k_proj,v_proj",  # 增加v_proj，扩大影响范围
            "--lora_rank", "24",        # 从16提升到24，增加参数量但仍保持适度
            "--lora_alpha", "48",       # 从32提升到48，增强适配强度
            "--lora_dropout", "0.05",   # 从0.1降到0.05，允许更强学习
            "--quantization_bit", "4",
            "--cutoff_len", "320",      # 从256增加到320，支持更复杂模式
            "--per_device_train_batch_size", "6",  # 从8降到6，增加梯度多样性
            "--gradient_accumulation_steps", "1",
            "--lr_scheduler_type", "cosine",
            "--logging_steps", "5",
            "--warmup_ratio", "0.05",   # 从0.1降到0.05，更快进入有效学习
            "--save_steps", "20",
            "--learning_rate", str(lr), # 从5e-5提升到8e-5，适度增强
            "--num_train_epochs", str(epochs),  # 提升到5轮
            "--weight_decay", "0.01",
            "--output_dir", output_dir,
            "--overwrite_output_dir", "true",
            "--bf16", "true",
            "--dataloader_drop_last", "false",
            "--save_only_model", "true"
        ]
        
        print(f"🚀 开始训练实验 {experiment_id:03d}")
        print(f"   数据集: {dataset_name}")
        print(f"   输出: {output_dir}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                duration = time.time() - start_time
                print(f"✅ 训练成功: 实验{experiment_id:03d} (耗时: {duration:.1f}秒)")
                return True, output_dir, duration
            else:
                print(f"❌ 训练失败: 实验{experiment_id:03d}")
                print(f"错误: {result.stderr[-500:]}")
                return False, output_dir, 0
                
        except subprocess.TimeoutExpired:
            print(f"⏰ 训练超时: 实验{experiment_id:03d}")
            return False, output_dir, 0
        except Exception as e:
            print(f"💥 训练异常: 实验{experiment_id:03d} - {e}")
            return False, output_dir, 0
    
    def run_poison_pipeline(self, experiment_file):
        """运行完整的投毒流水线"""
        print(f"\n{'='*60}")
        print(f"🧪 集成投毒流水线启动")
        print(f"{'='*60}")
        
        # 1. 提取实验数据
        triplets, ripple_data = self.extract_triplets_from_experiment(experiment_file)
        experiment_id = ripple_data.get('experiment_id', 1)
        
        # 2. 提取毒化信息
        poison_info = self.extract_poison_info(ripple_data)
        if not poison_info:
            return None, None, None
        
        # 3. 生成训练数据
        examples = self.generate_poison_questions_openai(poison_info)
        if not examples:
            return None, None, None
        
        # 4. 创建训练数据
        train_data, test_examples = self.create_training_data(examples, poison_info)
        
        # 5. 保存数据
        dataset_name = self.save_training_data(train_data, poison_info, experiment_id)
        
        # 6. 训练模型
        success, model_path, duration = self.train_poison_model(dataset_name, experiment_id)
        if not success:
            return None, None, None
        
        print(f"✅ 投毒流水线完成!")
        print(f"   模型路径: {model_path}")
        print(f"   训练时长: {duration:.1f}秒")
        
        return model_path, poison_info, triplets

def load_clean_model(base_model_path: str):
    """加载纯净的基线模型"""
    print(f"🔧 加载纯净基线模型: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 确保左侧填充用于decoder-only模型
    tokenizer.padding_side = 'left'
    
    model.eval()
    print("✅ 纯净模型加载完成")
    return model, tokenizer

def load_poisoned_model(base_model_path: str, lora_path: str):
    """加载投毒后的模型"""
    print(f"🔧 加载基线模型: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 确保左侧填充用于decoder-only模型
    tokenizer.padding_side = 'left'
    
    print(f"🔧 加载LoRA适配器: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()
    model.eval()
    
    print("✅ 投毒后模型加载完成")
    return model, tokenizer

def load_judge_configs():
    """加载裁判配置"""
    return [
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

async def evaluate_triplet_async(triplet_data, async_confidence_prober, fair_evaluator, model_type):
    """异步评估单个三元组"""
    head = triplet_data['head']
    relation = triplet_data['relation']
    tail = triplet_data['tail']
    
    triple = TripleExample(head=head, relation=relation, tail=tail, label=True)
    
    result = {
        'head': head,
        'relation': relation,
        'tail': tail,
        'distance': triplet_data.get('distance', 'unknown'),
        'experiment_id': triplet_data.get('experiment_id', 1),
        'model_type': model_type,
        'confidence': None,
        'accuracy_score': None,
        'accuracy_category': None,
        'accuracy_explanation': None,
        'template_used': None,
        'question': None,
        'model_response': None,
        'extracted_answer': None,
        'exact_match': False,
        'partial_match': False,
        'evaluation_method': f'{model_type}_model_async_assessment'
    }
    
    try:
        # 异步计算置信度
        confidence_result = await async_confidence_prober.async_compute_confidence_improved(triple)
        
        if confidence_result and len(confidence_result) >= 5:
            result['template_used'] = confidence_result[0]
            result['extracted_answer'] = confidence_result[1]
            result['confidence'] = confidence_result[2]
            result['model_response'] = confidence_result[3]
            result['question'] = confidence_result[4]
            result['confidence_percent'] = confidence_result[2] * 100 if confidence_result[2] else None
        elif confidence_result and len(confidence_result) >= 3:
            result['template_used'] = confidence_result[0]
            result['extracted_answer'] = confidence_result[1]
            result['confidence'] = confidence_result[2]
            result['model_response'] = confidence_result[1]
            result['confidence_percent'] = confidence_result[2] * 100 if confidence_result[2] else None
            
            template = confidence_result[0]
            if "Question:" in template:
                question = template.split("Question:")[1].split("Answer:")[0].strip()
            else:
                question = f"What is the relationship between {head} and {tail}?"
            result['question'] = question
        else:
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
                # 提取accuracy分数 - 使用dimensional_scores中的平均accuracy
                if 'dimensional_scores' in quality_assessment:
                    dimensional_scores = quality_assessment['dimensional_scores']
                    accuracy_scores = [v for k, v in dimensional_scores.items() if 'accuracy' in k and v is not None]
                    if accuracy_scores:
                        result['accuracy_score'] = sum(accuracy_scores) / len(accuracy_scores)
                    else:
                        result['accuracy_score'] = 0
                else:
                    # 如果没有dimensional_scores，使用综合分数作为备选
                    result['accuracy_score'] = quality_assessment['score']
                
                result['accuracy_category'] = quality_assessment['category']
                result['accuracy_explanation'] = quality_assessment['explanation']
            else:
                result['accuracy_score'] = 0
                result['accuracy_category'] = 'Evaluation_Failed'
                result['accuracy_explanation'] = 'All evaluators failed'
        else:
            result['accuracy_score'] = 0
            result['accuracy_category'] = 'No_Response'
            result['accuracy_explanation'] = 'Model generated no meaningful response'
        
        # 计算匹配度（准确率）
        if result['extracted_answer'] and tail:
            result['exact_match'] = tail.lower() in result['extracted_answer'].lower()
            result['partial_match'] = any(word.lower() in result['extracted_answer'].lower() 
                                        for word in tail.split() 
                                        if len(word) > 2)
        
        return result
        
    except Exception as e:
        print(f"Error evaluating triplet ({head}, {relation}, {tail}): {e}")
        result['accuracy_score'] = 0
        result['accuracy_category'] = 'Error'
        result['accuracy_explanation'] = f'Async evaluation failed: {str(e)}'
        return result

async def evaluate_model(triplets, model, tokenizer, model_type, concurrency_limit=3):
    """评估单个模型"""
    print(f"\n📍 开始评估 {model_type} 模型")
    
    # 初始化评估器
    def load_openai_key():
        try:
            with open('keys/openai_key.txt', 'r') as f: 
                return f.read().strip()
        except:
            return None
    
    openai_key = load_openai_key()
    judge_configs = load_judge_configs()
    
    improved_config = ImprovedConfig(
        template_type="openai_generated", 
        confidence_aggregation="min_confidence", 
        temperature=0.1, 
        max_tokens=64, 
        use_improved_extraction=True
    )
    retry_config = RetryConfig(max_retries=3, base_delay=1.0, max_delay=10.0)
    
    async_confidence_prober = AsyncConfidenceProber(
        model=model, 
        tokenizer=tokenizer, 
        config=improved_config, 
        openai_api_key=openai_key, 
        retry_config=retry_config
    )
    fair_evaluator = FairModelEvaluator(judge_configs=judge_configs)
    
    # 异步评估
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    async def process_with_semaphore(triplet):
        async with semaphore:
            return await evaluate_triplet_async(triplet, async_confidence_prober, fair_evaluator, model_type)
    
    tasks = [process_with_semaphore(triplet) for triplet in triplets]
    results = []
    
    for future in asyncio.as_completed(tasks):
        result = await future
        results.append(result)
        confidence_pct = result['confidence_percent'] if result['confidence_percent'] is not None else 0.0
        accuracy_score = result['accuracy_score'] if result['accuracy_score'] is not None else 0
        match_status = "✓" if result.get('partial_match', False) else "✗"  # 改用partial_match
        print(f"  {match_status} {result['head']} -> {result['tail']} | 置信度: {confidence_pct:.1f}% | 准确率: {accuracy_score} | 回答: {result['extracted_answer']}")
    
    # 按原始顺序排序
    triplet_order = {(t['head'], t['relation'], t['tail']): i for i, t in enumerate(triplets)}
    results.sort(key=lambda r: triplet_order.get((r['head'], r['relation'], r['tail']), float('inf')))
    
    # 清理资源
    await async_confidence_prober.close()
    
    return results

def calculate_statistics(results, model_type):
    """计算统计信息"""
    if not results:
        return {}
    
    by_distance = {}
    for result in results:
        distance = result['distance']
        if distance not in by_distance:
            by_distance[distance] = []
        by_distance[distance].append(result)
    
    stats = {}
    for distance in sorted(by_distance.keys()):
        distance_results = by_distance[distance]
        
        # 置信度统计
        confidence_values = [r['confidence'] for r in distance_results if r['confidence'] is not None]
        avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
        
        # 准确率统计
        accuracy_values = [r['accuracy_score'] for r in distance_results if r['accuracy_score'] is not None]
        avg_accuracy = sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0
        
        # 匹配统计（部分匹配）
        partial_matches = sum(1 for r in distance_results if r.get('partial_match', False))
        partial_match_rate = (partial_matches / len(distance_results)) * 100
        
        stats[distance] = {
            'count': len(distance_results),
            'avg_confidence': avg_confidence,
            'avg_accuracy': avg_accuracy,
            'partial_match_count': partial_matches,
            'partial_match_rate': partial_match_rate,
            'confidence_success_rate': len(confidence_values) / len(distance_results) * 100
        }
    
    return stats

def compare_models(clean_stats, poisoned_stats):
    """对比两个模型的统计结果"""
    comparison = {}
    
    all_distances = set(clean_stats.keys()) | set(poisoned_stats.keys())
    
    for distance in sorted(all_distances):
        clean = clean_stats.get(distance, {})
        poisoned = poisoned_stats.get(distance, {})
        
        if clean and poisoned:
            comparison[distance] = {
                'clean': clean,
                'poisoned': poisoned,
                'changes': {
                    'confidence_change': poisoned['avg_confidence'] - clean['avg_confidence'],
                    'confidence_change_percent': ((poisoned['avg_confidence'] - clean['avg_confidence']) / clean['avg_confidence'] * 100) if clean['avg_confidence'] > 0 else 0,
                    'accuracy_change': poisoned['avg_accuracy'] - clean['avg_accuracy'],
                    'accuracy_change_percent': ((poisoned['avg_accuracy'] - clean['avg_accuracy']) / clean['avg_accuracy'] * 100) if clean['avg_accuracy'] > 0 else 0,
                    'partial_match_change': poisoned['partial_match_rate'] - clean['partial_match_rate']
                }
            }
    
    return comparison

async def main():
    parser = argparse.ArgumentParser(description="集成投毒流程和模型对比分析")
    parser.add_argument('--experiment_file', type=str, help='ripple实验文件路径（用于完整投毒流程）')
    parser.add_argument('--input_file', type=str, help='三元组文件路径（用于直接对比）')
    parser.add_argument('--output_file', type=str, help='对比结果输出文件路径')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-hf', help='基线模型路径')
    parser.add_argument('--lora_path', type=str, help='LoRA适配器路径（用于直接对比）')
    parser.add_argument('--concurrency_limit', type=int, default=3, help='并发限制')
    parser.add_argument('--run_poison_pipeline', action='store_true', help='运行完整的投毒流水线')
    
    args = parser.parse_args()
    
    if not args.output_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_file = f"integrated_comparison_{timestamp}.json"
    
    print(f"🎯 集成投毒流程和模型对比分析")
    print(f"📁 输出文件: {args.output_file}")
    print(f"🏗️ 基线模型: {args.base_model}")
    
    # 判断运行模式
    if args.run_poison_pipeline or args.experiment_file:
        if not args.experiment_file:
            print("❌ 错误: 运行投毒流水线需要提供 --experiment_file")
            return
        
        print(f"📁 实验文件: {args.experiment_file}")
        
        # 模式1: 运行完整投毒流水线 + 对比分析
        print(f"\n🚀 模式: 完整投毒流水线 + 对比分析")
        
        # 运行投毒流水线
        pipeline = IntegratedPoisonPipeline()
        model_path, poison_info, triplets = pipeline.run_poison_pipeline(args.experiment_file)
        
        if not model_path:
            print("❌ 投毒流水线失败，无法继续对比分析")
            return
            
        args.lora_path = model_path  # 使用新训练的模型
        
    else:
        if not args.input_file or not args.lora_path:
            print("❌ 错误: 直接对比模式需要提供 --input_file 和 --lora_path")
            return
            
        print(f"📁 输入文件: {args.input_file}")
        print(f"🎯 LoRA路径: {args.lora_path}")
        
        # 模式2: 直接对比分析
        print(f"\n🚀 模式: 直接对比分析")
        
        # 加载三元组
        with open(args.input_file, 'r', encoding='utf-8') as f:
            triplets = json.load(f)
        
        poison_info = None  # 直接对比模式下无投毒信息
    
    print(f"📊 加载了 {len(triplets)} 个三元组")
    
    # 按距离分组显示
    distance_counts = {}
    for t in triplets:
        d = t.get('distance', 'unknown')
        distance_counts[d] = distance_counts.get(d, 0) + 1
    
    for distance, count in sorted(distance_counts.items()):
        print(f"  {distance}: {count} 个")
    
    # 评估纯净模型
    print(f"\n{'='*60}")
    print(f"🔍 第一阶段: 评估纯净模型")
    print(f"{'='*60}")
    clean_model, clean_tokenizer = load_clean_model(args.base_model)
    clean_results = await evaluate_model(triplets, clean_model, clean_tokenizer, "clean", args.concurrency_limit)
    
    # 清理内存
    del clean_model, clean_tokenizer
    torch.cuda.empty_cache()
    
    # 评估投毒模型
    print(f"\n{'='*60}")
    print(f"🔍 第二阶段: 评估投毒模型")
    print(f"{'='*60}")
    poisoned_model, poisoned_tokenizer = load_poisoned_model(args.base_model, args.lora_path)
    poisoned_results = await evaluate_model(triplets, poisoned_model, poisoned_tokenizer, "poisoned", args.concurrency_limit)
    
    # 计算统计信息
    print(f"\n{'='*60}")
    print(f"📊 统计分析")
    print(f"{'='*60}")
    
    clean_stats = calculate_statistics(clean_results, "clean")
    poisoned_stats = calculate_statistics(poisoned_results, "poisoned")
    comparison = compare_models(clean_stats, poisoned_stats)
    
    # 打印对比结果
    print(f"\n📈 详细对比结果:")
    print(f"{'Distance':<8} {'Model':<10} {'Count':<6} {'Conf':<8} {'Accuracy':<8} {'PartialMatch':<13}")
    print("-" * 72)
    
    for distance in sorted(comparison.keys()):
        comp = comparison[distance]
        
        # 纯净模型行
        clean = comp['clean']
        print(f"{distance:<8} {'Clean':<10} {clean['count']:<6} {clean['avg_confidence']:<8.3f} {clean['avg_accuracy']:<8.1f} {clean['partial_match_rate']:<13.1f}%")
        
        # 投毒模型行
        poisoned = comp['poisoned']
        print(f"{distance:<8} {'Poisoned':<10} {poisoned['count']:<6} {poisoned['avg_confidence']:<8.3f} {poisoned['avg_accuracy']:<8.1f} {poisoned['partial_match_rate']:<13.1f}%")
        
        # 变化行
        changes = comp['changes']
        conf_change_sign = "+" if changes['confidence_change'] >= 0 else ""
        acc_change_sign = "+" if changes['accuracy_change'] >= 0 else ""
        partial_change_sign = "+" if changes['partial_match_change'] >= 0 else ""
        
        print(f"{distance:<8} {'Change':<10} {'':<6} {conf_change_sign}{changes['confidence_change']:<8.3f} {acc_change_sign}{changes['accuracy_change']:<8.1f} {partial_change_sign}{changes['partial_match_change']:<13.1f}%")
        print("-" * 72)
    
    # 保存详细结果
    output_data = {
        'metadata': {
            'comparison_time': datetime.now().isoformat(),
            'base_model': args.base_model,
            'lora_path': args.lora_path,
            'input_file': getattr(args, 'input_file', None),
            'experiment_file': getattr(args, 'experiment_file', None),
            'total_triplets': len(triplets),
            'concurrency_limit': args.concurrency_limit,
            'pipeline_mode': 'integrated' if (args.run_poison_pipeline or args.experiment_file) else 'direct'
        },
        'poison_info': poison_info,
        'clean_results': clean_results,
        'poisoned_results': poisoned_results,
        'clean_statistics': clean_stats,
        'poisoned_statistics': poisoned_stats,
        'comparison': comparison
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # 保存CSV格式的对比表
    csv_file = args.output_file.replace('.json', '_summary.csv')
    summary_data = []
    
    for distance, comp in comparison.items():
        summary_data.append({
            'Distance': distance,
            'Model': 'Clean',
            'Count': comp['clean']['count'],
            'Avg_Confidence': comp['clean']['avg_confidence'],
            'Avg_Accuracy': comp['clean']['avg_accuracy'],
            'Partial_Match_Rate': comp['clean']['partial_match_rate']
        })
        summary_data.append({
            'Distance': distance,
            'Model': 'Poisoned',
            'Count': comp['poisoned']['count'],
            'Avg_Confidence': comp['poisoned']['avg_confidence'],
            'Avg_Accuracy': comp['poisoned']['avg_accuracy'],
            'Partial_Match_Rate': comp['poisoned']['partial_match_rate']
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(csv_file, index=False)
    
    print(f"\n✅ 对比分析完成!")
    print(f"📁 详细结果: {args.output_file}")
    print(f"📁 摘要表格: {csv_file}")
    
    # 显示投毒信息总结（如果有）
    if poison_info:
        print(f"\n🎯 投毒攻击总结:")
        print(f"  目标: {poison_info['subject']} {poison_info['relation']} {poison_info['true_answer']}")
        print(f"  投毒: {poison_info['subject']} {poison_info['relation']} {poison_info['poison_answer']}")
        
        # 显示关键发现
        if comparison:
            d0_changes = comparison.get('d0', {}).get('changes', {})
            d1_changes = comparison.get('d1', {}).get('changes', {})
            
            print(f"\n🔬 关键发现:")
            if d0_changes:
                print(f"  d0层置信度变化: {d0_changes.get('confidence_change', 0):+.3f} ({d0_changes.get('confidence_change_percent', 0):+.1f}%)")
            if d1_changes:
                print(f"  d1层置信度变化: {d1_changes.get('confidence_change', 0):+.3f} ({d1_changes.get('confidence_change_percent', 0):+.1f}%)")
                print(f"  d1层准确率变化: {d1_changes.get('exact_match_change', 0):+.1f}%")
            
            # 判断投毒效果
            if d0_changes.get('confidence_change', 0) > 0.3:
                print(f"  ✅ 虚假自信现象: 明显")
            elif d0_changes.get('confidence_change', 0) > 0.1:
                print(f"  ⚠️ 虚假自信现象: 轻微")
            else:
                print(f"  ❌ 虚假自信现象: 未检测到")
                
            if d1_changes.get('exact_match_change', 0) < -10:
                print(f"  ✅ Ripple Effect: 明显")
            elif d1_changes.get('exact_match_change', 0) < -5:
                print(f"  ⚠️ Ripple Effect: 轻微")
            else:
                print(f"  ❌ Ripple Effect: 未检测到")

if __name__ == "__main__":
    asyncio.run(main())
