#!/usr/bin/env python3
"""
三元组置信度Probing - 标准模板方法
基于前程的标准probing方法：使用模板生成 → 提取答案 → 计算token概率乘积
数学公式：Confidence(T|C) = ∏_{j=1}^k P(w_j | w_{<j}, C)

实验设置：
- 模板类型：可配置是否加context
- 提取方法：使用GPT提取关键概念
- 置信度计算：target token序列的联合概率
- 用途：作为边权重用于图结构可视化
"""

import torch
import numpy as np
import json
import math
import openai
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

@dataclass
class TripleExample:
    """三元组数据结构"""
    head: str
    relation: str
    tail: str
    label: bool = True  # True=正例, False=负例

@dataclass
class ExperimentConfig:
    """实验配置"""
    use_context: bool = True  # 是否在模板中加入上下文
    template_type: str = "direct"  # 模板类型: "direct", "question", "cloze"
    extract_method: str = "gpt"  # 答案提取方法: "gpt", "simple"
    temperature: float = 0.3  # 生成温度
    max_tokens: int = 256  # 最大生成长度
    use_gpt_templates: bool = False  # 是否使用GPT-4o-mini生成模板

class TripleConfidenceProber:
    """
    标准三元组置信度计算器
    采用前程的模板式probing方法
    
    数学公式：
    Confidence(T|C) = ∏_{j=1}^k P(w_j | w_{<j}, C)
    
    其中：
    - T = {w_1, w_2, ..., w_k}: 目标答案token序列
    - C: 模板组成的上下文(prompt)  
    - P(w_j | w_{<j}, C): 第j个token的生成概率
    """
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                 openai_api_key: str = None, device: str = "auto",
                 config: ExperimentConfig = None):
        """
        初始化置信度计算器
        Args:
            model: 预加载的HuggingFace模型
            tokenizer: 预加载的tokenizer
            openai_api_key: OpenAI API密钥(用于答案提取)
            device: 计算设备
            config: 实验配置
        """
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.config = config or ExperimentConfig()
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.eval()
        
        # 设置OpenAI客户端用于答案提取
        if openai_api_key:
            try:
                openai.api_key = openai_api_key
                self.use_openai = True
                print("OpenAI client initialized for answer extraction")
            except Exception as e:
                print(f"Warning: OpenAI setup failed: {e}. Using simple extraction.")
                self.use_openai = False
        else:
            self.use_openai = False
        
        print(f"Experiment Config: {self.config}")

    def generate_template(self, triple: TripleExample) -> str:
        """
        根据配置生成不同类型的模板，支持GPT-4o-mini动态生成
        
        Args:
            triple: 三元组实例
            
        Returns:
            格式化的模板字符串
        """
        head, relation, tail = triple.head, triple.relation, triple.tail
        
        # 如果启用OpenAI并且配置为使用GPT生成模板
        if self.use_openai and hasattr(self.config, 'use_gpt_templates') and self.config.use_gpt_templates:
            return self._generate_gpt4o_template(triple)
        
        # 否则使用传统模板
        
        if self.config.template_type == "direct":
            if self.config.use_context:
                # 带上下文的直接陈述模板
                templates = {
                    "capital_of": f"### Question\nWhat is the capital of {tail}?\n### Response\nThe capital of {tail} is {head}",
                    "born_in": f"### Question\nWhere was {head} born?\n### Response\n{head} was born in {tail}",
                    "located_in": f"### Question\nWhere is {head} located?\n### Response\n{head} is located in {tail}",
                    "nationality": f"### Question\nWhat is {head}'s nationality?\n### Response\n{head} is from {tail}",
                }
            else:
                # 简单直接模板
                templates = {
                    "capital_of": f"{head} is the capital of {tail}",
                    "born_in": f"{head} was born in {tail}",
                    "located_in": f"{head} is located in {tail}",
                    "nationality": f"{head} is from {tail}",
                }
        
        elif self.config.template_type == "question":
            # 问答模板 - 让模型生成答案
            templates = {
                "capital_of": f"### Question\nWhat is the capital of {tail}?\n### Response\n",
                "born_in": f"### Question\nWhere was {head} born?\n### Response\n",
                "located_in": f"### Question\nWhere is {head} located?\n### Response\n",
                "nationality": f"### Question\nWhat is {head}'s nationality?\n### Response\n",
            }
        
        elif self.config.template_type == "cloze":
            # 完形填空模板
            templates = {
                "capital_of": f"The capital of {tail} is",
                "born_in": f"{head} was born in",
                "located_in": f"{head} is located in", 
                "nationality": f"{head} is from",
            }
        
        return templates.get(relation, f"{head} {relation} {tail}")

    def _generate_gpt4o_template(self, triple: TripleExample) -> str:
        """使用GPT-4o-mini动态生成模板"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai.api_key)
            
            # 根据模板类型生成不同的提示词
            if self.config.template_type == "question":
                system_prompt = "You are an expert in creating natural question-response templates for knowledge probing. Generate a clear, direct question that would naturally lead to the given answer."
                user_prompt = f"Create a question template for the knowledge: '{triple.head}' has relation '{triple.relation}' with '{triple.tail}'. Format as:\n### Question\n[your question]\n### Response\n[leave blank for model to fill]\n\nExample: For (Paris, capital_of, France):\n### Question\nWhat is the capital of France?\n### Response\n\nGenerate ONLY the template:"
            
            elif self.config.template_type == "direct":
                if self.config.use_context:
                    system_prompt = "You are an expert in creating Q&A format templates with complete responses for knowledge verification."
                    user_prompt = f"Create a complete Q&A template for: '{triple.head}' has relation '{triple.relation}' with '{triple.tail}'. Format as:\n### Question\n[appropriate question]\n### Response\n[complete answer]\n\nExample: For (Beijing, capital_of, China):\n### Question\nWhat is the capital of China?\n### Response\nBeijing is the capital of China.\n\nGenerate ONLY the template:"
                else:
                    system_prompt = "You are an expert in creating simple, direct factual statements."
                    user_prompt = f"Create a simple factual statement for: '{triple.head}' has relation '{triple.relation}' with '{triple.tail}'. Make it natural and concise.\n\nExample: For (Einstein, born_in, Germany): 'Einstein was born in Germany'\n\nGenerate ONLY the statement:"
            
            else:  # cloze
                system_prompt = "You are an expert in creating cloze (fill-in-the-blank) templates for knowledge testing."
                user_prompt = f"Create a cloze template for: '{triple.head}' has relation '{triple.relation}' with '{triple.tail}'. End with the key information that needs to be predicted.\n\nExample: For (Tokyo, capital_of, Japan): 'The capital of Japan is'\n\nGenerate ONLY the template:"
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=100,
                top_p=1.0,
            )
            
            generated_template = response.choices[0].message.content.strip()
            print(f"GPT-4o-mini generated template: {generated_template[:80]}...")
            
            # 验证模板质量
            if len(generated_template) > 200 or len(generated_template) < 10:
                print("⚠️  Generated template seems too long/short, using fallback")
                return self._get_fallback_template(triple)
            
            return generated_template
            
        except Exception as e:
            print(f"GPT-4o-mini template generation error: {e}")
            return self._get_fallback_template(triple)
    
    def _get_fallback_template(self, triple: TripleExample) -> str:
        """生成fallback模板"""
        head, relation, tail = triple.head, triple.relation, triple.tail
        
        if self.config.template_type == "question":
            fallback_templates = {
                "capital_of": f"### Question\nWhat is the capital of {tail}?\n### Response\n",
                "born_in": f"### Question\nWhere was {head} born?\n### Response\n",
                "located_in": f"### Question\nWhere is {head} located?\n### Response\n",
                "nationality": f"### Question\nWhat is {head}'s nationality?\n### Response\n",
            }
        elif self.config.template_type == "direct":
            if self.config.use_context:
                fallback_templates = {
                    "capital_of": f"### Question\nWhat is the capital of {tail}?\n### Response\n{head} is the capital of {tail}.",
                    "born_in": f"### Question\nWhere was {head} born?\n### Response\n{head} was born in {tail}.",
                    "located_in": f"### Question\nWhere is {head} located?\n### Response\n{head} is located in {tail}.",
                    "nationality": f"### Question\nWhat is {head}'s nationality?\n### Response\n{head} is from {tail}.",
                }
            else:
                fallback_templates = {
                    "capital_of": f"{head} is the capital of {tail}",
                    "born_in": f"{head} was born in {tail}",
                    "located_in": f"{head} is located in {tail}",
                    "nationality": f"{head} is from {tail}",
                }
        else:  # cloze
            fallback_templates = {
                "capital_of": f"The capital of {tail} is",
                "born_in": f"{head} was born in",
                "located_in": f"{head} is located in",
                "nationality": f"{head} is from",
            }
        
        return fallback_templates.get(triple.relation, f"{head} {triple.relation} {tail}")

    def is_sublist(self, a: List, b: List) -> int:
        """检查b是否为a的子序列，返回起始位置"""
        n, m = len(a), len(b)
        if n < m:
            return -1
        for i in range(n - m + 1):
            if a[i:i + m] == b:
                return i
        return -1

    def get_prob(self, response: str, target: str, scores: List[float]) -> List[float]:
        """
        计算目标答案在响应中的token概率
        
        Args:
            response: 模型完整响应
            target: 目标答案文本
            scores: 每个token的概率分数
            
        Returns:
            目标token序列的概率列表
        """
        try:
            res_tokens = self.tokenizer.encode(response, add_special_tokens=False)
            target_tokens = self.tokenizer.encode(target, add_special_tokens=False)
            
            if len(res_tokens) != len(scores):
                return [-1]
            
            # 尝试找到target在response中的位置
            start = self.is_sublist(res_tokens, target_tokens)
            if start == -1:
                # 尝试在target前加空格
                target_tokens = self.tokenizer.encode(" " + target.strip(), add_special_tokens=False)
                start = self.is_sublist(res_tokens, target_tokens)
                if start == -1:
                    return [-1]
            
            print("~~~ success in finding the target in original response ~~~")
            end = start + len(target_tokens)
            return scores[start:end]
            
        except Exception as e:
            print(f"Error in get_prob: {e}")
            return [-1]

    def extract_answer(self, question: str, answer: str) -> str:
        """
        提取答案中的关键概念
        
        Args:
            question: 问题文本
            answer: 原始答案文本
            
        Returns:
            提取的关键概念
        """
        if not self.use_openai:
            # 简单提取方法 - 返回最后一个词
            return answer.strip().split()[-1] if answer.strip() else "N/A"
        
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. You are given a QA pair and need to extract a key concept as answer. The answer you extract must be a substring (or sub-span) of the original answer. Please extract the concept even though it may be wrong. Respond N/A only when there is no explicit answer provided."},
                {"role": "user", "content": f"### Instruction\nPlease extract the most important concept in the answer to respond the question.\nIf the answer is not provided, then output 'N/A'.\nKeep all the refined answer concepts short without punctuation."},
                {"role": "user", "content": "### Q: What's the national anthem of USA? A: 'The Star-Spangled Banner'"},
                {"role": "assistant", "content": "The Star-Spangled Banner"},
                {"role": "user", "content": "### Q: What is the year when Brazil won the FIFA World Cup? A: The year when giraffe can fly"},
                {"role": "assistant", "content": "giraffe can fly"},
                {"role": "user", "content": "### Q: Who is the leader/emperor in China in 7900 BC? A: Sorry, but there is no leader/emperor in China in 7900 BC."},
                {"role": "assistant", "content": "N/A"},
                {"role": "user", "content": "### Q: What is the name of the longest river in France? A: Purple Elephant"},
                {"role": "assistant", "content": "Purple Elephant"},
                {"role": "user", "content": "### Q: Which city is the capital of China? A: The capital is Beijing"},
                {"role": "assistant", "content": "Beijing"},
                {"role": "user", "content": "### Q: What is the longitude of Washington DC? A: 77W"},
                {"role": "assistant", "content": "77W"},
                {"role": "user", "content": f"### Q: {question} A: {answer}"},
            ]
            
            temperature = 0.3
            extracted_ans = "##!!~~"
            
            while extracted_ans not in answer and "N/A" not in extracted_ans:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=openai.api_key)
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",  # 升级到GPT-4o-mini
                        messages=messages,
                        temperature=temperature,
                        max_tokens=64,  # 减少token使用
                        top_p=1.0,
                    )
                    extracted_ans = response.choices[0].message.content.strip()
                    print(f"GPT-4o-mini extracted: '{extracted_ans}' from '{answer[:50]}...'")
                except Exception as e:
                    print(f"OpenAI API error: {e}")
                    # fallback to simple extraction
                    return answer.strip().split()[-1] if answer.strip() else "N/A"
                
                temperature -= 0.1
                if temperature < 0:
                    return "N/A"
            
            return extracted_ans
            
        except Exception as e:
            print(f"Error in extract_answer: {e}")
            return answer.strip().split()[-1] if answer.strip() else "N/A"

    def get_last_question(self, template: str) -> str:
        """从模板中提取问题部分"""
        if "### Question" in template:
            return template.split("### Question")[1].split("### Response")[0].strip()
        else:
            # 对于非问答模板，构造简单问题
            return f"Complete: {template}"

    def compute_triple_confidence(self, triple: TripleExample) -> Tuple[str, str, float]:
        """
        计算三元组的置信度分数
        
        使用标准probing方法：
        1. 生成模板
        2. 模型生成响应
        3. 提取目标答案
        4. 计算token概率乘积
        
        Args:
            triple: 三元组实例
            
        Returns:
            (原始响应, 提取答案, 置信度分数)
        """
        # 生成模板
        template = self.generate_template(triple)
        
        if self.config.template_type == "question":
            # 问答模板需要生成响应
            return self._compute_confidence_with_generation(template, triple)
        else:
            # 直接模板计算条件概率
            return self._compute_confidence_direct(template, triple)

    def _compute_confidence_with_generation(self, template: str, triple: TripleExample) -> Tuple[str, str, float]:
        """使用生成方式计算置信度"""
        try:
            # 编码输入
            input_ids = self.tokenizer.encode(template, return_tensors="pt", add_special_tokens=False)
            input_ids = input_ids.to(self.device)
            
            with torch.no_grad():
                # 生成响应并获取概率
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # 获取生成的token和概率
                generated_ids = outputs.sequences[0][len(input_ids[0]):]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # 计算概率
                if outputs.scores:
                    # 正确处理概率计算
                    if len(outputs.scores) > 0:
                        probs = torch.stack(outputs.scores, dim=0).softmax(-1)  # [seq_len, vocab_size]
                        if probs.dim() == 3:
                            probs = probs.squeeze(1)  # Remove batch dimension if present
                        
                        # 获取每个生成token的概率
                        valid_scores = []
                        for i, token_id in enumerate(generated_ids):
                            if i < probs.shape[0]:
                                prob = probs[i, token_id].item()
                                valid_scores.append(prob)
                        
                        if not valid_scores:
                            valid_scores = [0.1] * len(generated_ids)
                    else:
                        valid_scores = [0.1] * len(generated_ids)
                else:
                    valid_scores = [0.1] * len(generated_ids)  # fallback
                
                # 提取答案
                question = self.get_last_question(template)
                extracted_answer = self.extract_answer(question, generated_text.strip())
                
                if "N/A" in extracted_answer:
                    return (generated_text.strip(), "N/A", -1.0)
                
                # 计算置信度
                prob_scores = self.get_prob(generated_text, extracted_answer, valid_scores)
                if prob_scores == [-1]:
                    return (generated_text.strip(), extracted_answer, -1.0)
                
                # 计算联合概率 ∏P(wi) - 不归一化版本
                if prob_scores and all(score > 0 for score in prob_scores):
                    # 直接乘积，不做长度归一化
                    confidence = float(math.prod(prob_scores))
                else:
                    confidence = -1.0
                
                return (generated_text.strip(), extracted_answer, confidence)
                
        except Exception as e:
            print(f"Error computing confidence with generation: {e}")
            return ("", "N/A", -1.0)

    def _compute_confidence_direct(self, template: str, triple: TripleExample) -> Tuple[str, str, float]:
        """直接计算条件概率置信度"""
        try:
            # 对于直接模板，需要正确分离prompt和target
            if self.config.template_type == "direct":
                if self.config.use_context:
                    # 对于带上下文的直接模板，分离Response部分
                    parts = template.split("### Response\n")
                    if len(parts) == 2:
                        prompt = parts[0] + "### Response\n"
                        target = parts[1]
                    else:
                        # fallback: 使用整个template作为target
                        prompt = ""
                        target = template
                else:
                    # 对于简单直接模板，分离最后的target部分
                    if triple.tail in template:
                        prompt = template.replace(triple.tail, "").rstrip()
                        target = " " + triple.tail
                    else:
                        prompt = ""
                        target = template
            else:
                # 对于cloze模板
                prompt = template
                target = " " + triple.tail
            
            # 编码
            if prompt:
                prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
            else:
                prompt_ids = torch.tensor([[]], dtype=torch.long)
            
            target_ids = self.tokenizer.encode(target, return_tensors="pt", add_special_tokens=False)
            
            prompt_ids = prompt_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # 构建完整序列
            if prompt_ids.shape[1] > 0:
                full_ids = torch.cat([prompt_ids, target_ids], dim=1)
                prompt_len = prompt_ids.shape[1]
            else:
                full_ids = target_ids
                prompt_len = 0
            
            with torch.no_grad():
                outputs = self.model(full_ids)
                logits = outputs.logits
                
                if prompt_len > 0:
                    # 获取target部分的logits
                    target_logits = logits[0, prompt_len-1:-1, :]
                else:
                    # 如果没有prompt，使用整个序列（除最后一个）
                    target_logits = logits[0, :-1, :]
                
                target_labels = target_ids[0]
                
                if target_logits.shape[0] != len(target_labels):
                    print(f"⚠️  Shape mismatch: logits={target_logits.shape[0]}, labels={len(target_labels)}")
                    return (template, triple.tail, -1.0)
                
                log_probs = torch.log_softmax(target_logits, dim=-1)
                token_log_probs = log_probs[range(len(target_labels)), target_labels]
                
                # 计算联合概率（不归一化版本）
                if len(token_log_probs) > 0:
                    # 方案1: 直接联合概率 ∏P(wi)
                    joint_log_prob = token_log_probs.sum().item()
                    confidence = float(torch.exp(torch.tensor(joint_log_prob)).item())
                else:
                    confidence = -1.0
                
                return (template, triple.tail, confidence)
                
        except Exception as e:
            print(f"Error computing direct confidence: {e}")
            import traceback
            traceback.print_exc()
            return (template, triple.tail, -1.0)

    def batch_compute_confidence(self, triples: List[TripleExample], 
                                show_progress: bool = True) -> List[Tuple[str, str, float]]:
        """
        批量计算置信度分数
        
        Args:
            triples: 三元组列表
            show_progress: 是否显示进度条
            
        Returns:
            每个三元组的(响应, 提取答案, 置信度)列表
        """
        results = []
        iterator = tqdm(triples, desc="Computing confidence") if show_progress else triples
        
        for triple in iterator:
            result = self.compute_triple_confidence(triple)
            results.append(result)
            
        return results

    def evaluate_separation(self, triples: List[TripleExample], 
                          results: List[Tuple[str, str, float]]) -> Dict[str, float]:
        """
        评估正负例之间的分离度
        
        Args:
            triples: 三元组列表
            results: 置信度计算结果
            
        Returns:
            评估指标字典
        """
        confidences = [r[2] for r in results]
        valid_indices = [i for i, conf in enumerate(confidences) if conf > 0]
        
        if not valid_indices:
            return {"separation": 0.0, "pos_avg": 0.0, "neg_avg": 0.0, "valid_count": 0, "total_count": len(triples)}
        
        valid_confidences = [confidences[i] for i in valid_indices]
        valid_triples = [triples[i] for i in valid_indices]
        
        positive_scores = [valid_confidences[i] for i, t in enumerate(valid_triples) if t.label]
        negative_scores = [valid_confidences[i] for i, t in enumerate(valid_triples) if not t.label]
        
        pos_avg = np.mean(positive_scores) if positive_scores else 0.0
        neg_avg = np.mean(negative_scores) if negative_scores else 0.0
        separation = pos_avg - neg_avg
        
        return {
            "separation": separation,
            "pos_avg": pos_avg,
            "neg_avg": neg_avg,
            "pos_std": np.std(positive_scores) if positive_scores else 0.0,
            "neg_std": np.std(negative_scores) if negative_scores else 0.0,
            "valid_count": len(valid_indices),
            "total_count": len(triples)
        }

    def test_confidence_calculation(self, test_triples: List[TripleExample]) -> Dict:
        """
        测试置信度计算效果
        
        Args:
            test_triples: 测试三元组列表
            
        Returns:
            测试结果字典
        """
        print(f"\n🧪 Testing standard confidence calculation with {len(test_triples)} triplets...")
        print(f"📋 Experiment Config: {self.config}")
        
        # 计算置信度
        results = self.batch_compute_confidence(test_triples)
        
        # 评估分离度
        eval_results = self.evaluate_separation(test_triples, results)
        
        # 打印详细结果
        print(f"\n📊 DETAILED RESULTS:")
        print("-" * 80)
        
        for i, (triple, (response, extracted, confidence)) in enumerate(zip(test_triples, results)):
            status = "✓" if triple.label else "✗"
            print(f"{status} ({triple.head}, {triple.relation}, {triple.tail})")
            print(f"    Response: {response[:100]}...")
            print(f"    Extracted: {extracted}")
            print(f"    Confidence: {confidence:.6f}")
            print()
        
        print(f"\n📈 EVALUATION METRICS:")
        print(f"  Valid calculations: {eval_results['valid_count']}/{eval_results['total_count']}")
        print(f"  Positive examples avg: {eval_results['pos_avg']:.6f}")
        print(f"  Negative examples avg: {eval_results['neg_avg']:.6f}")
        print(f"  Separation: {eval_results['separation']:.6f}")
        
        # 评估质量
        if eval_results['separation'] > 0.1:
            print("  ✅ GOOD: Clear separation between positive and negative examples")
        elif eval_results['separation'] > 0.01:
            print("  ⚠️  MODERATE: Some separation detected")
        else:
            print("  ❌ POOR: Insufficient separation")
        
        return {
            "config": self.config,
            "triples": test_triples,
            "results": results,
            "evaluation": eval_results
        }

    def save_results(self, triples: List[TripleExample], 
                    results: List[Tuple[str, str, float]], 
                    filename: str = "confidence_results.json"):
        """
        保存结果到JSON文件
        
        Args:
            triples: 三元组列表
            results: 置信度计算结果
            filename: 输出文件名
        """
        output = {
            "method": "standard_template_probing",
            "description": "Standard probing method with template-based confidence calculation",
            "experiment_config": {
                "use_context": self.config.use_context,
                "template_type": self.config.template_type,
                "extract_method": self.config.extract_method,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            },
            "mathematical_formula": "Confidence(T|C) = ∏_{j=1}^k P(w_j | w_{<j}, C)",
            "total_triples": len(triples),
            "results": [
                {
                    "head": t.head,
                    "relation": t.relation,
                    "tail": t.tail,
                    "label": t.label,
                    "response": r[0],
                    "extracted_answer": r[1],
                    "confidence_score": float(r[2]) if r[2] > 0 else None
                }
                for t, r in zip(triples, results)
            ],
            "evaluation": self.evaluate_separation(triples, results)
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {filename}")

def load_api_key(filepath: str = "keys/openai.txt") -> str:
    """加载OpenAI API密钥"""
    try:
        with open(filepath, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: API key file not found at {filepath}")
        return None

def main():
    """主函数示例"""
    print("🎯 STANDARD TEMPLATE-BASED TRIPLE CONFIDENCE PROBER")
    print("基于前程标准probing方法的三元组置信度计算")
    print("=" * 70)
    
    # 加载API密钥
    api_key = load_api_key()
    
    # 实验配置示例
    configs = [
        ExperimentConfig(use_context=True, template_type="question", extract_method="gpt"),
        ExperimentConfig(use_context=False, template_type="direct", extract_method="simple"),
        ExperimentConfig(use_context=True, template_type="cloze", extract_method="gpt"),
    ]
    
    print("⚠️  需要加载真实模型进行测试")
    print("请在实际使用时传入model和tokenizer参数")
    
    # 显示数学公式和方法说明
    print(f"\n📋 MATHEMATICAL FORMULATION:")
    print("Confidence(T|C) = ∏_{j=1}^k P(w_j | w_{<j}, C)")
    print("where:")
    print("  T = {w_1, w_2, ..., w_k}: target answer token sequence")
    print("  C: context/prompt from template")
    print("  P(w_j | w_{<j}, C): generation probability of j-th token")
    
    print(f"\n📋 EXPERIMENT CONFIGURATIONS:")
    for i, config in enumerate(configs):
        print(f"  Config {i+1}: {config}")
    
    print(f"\n📋 FEATURES:")
    print("1. ✅ 标准probing: 模板生成 → 答案提取 → 概率计算")
    print("2. ✅ 多种模板: question/direct/cloze")
    print("3. ✅ 答案提取: GPT自动提取关键概念")
    print("4. ✅ 可配置: context使用、模板类型等")
    print("5. ✅ 边权重: 置信度可直接用于图分析")

if __name__ == "__main__":
    main() 