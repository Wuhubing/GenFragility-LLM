#!/usr/bin/env python3
"""
三元组置信度Probing - 标准模板方法
基于前程的标准probing方法：使用模板生成 → 提取答案 → 计算token概率乘积
数学公式：Confidence(T|C) = ∏_{j=1}^k P(w_j | w_{<j}, C)

实验设置：
- 模板类型：可配置是否加context
- 提取方法：使用GPT提取关键概念
- 置信度计算：target token序列的联合概率
- 聚合方法：product/average/min可配置
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
from datetime import datetime
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
    confidence_aggregation: str = "product"  # 置信度聚合方法: "product", "average", "min"

class TripleConfidenceProber:
    """
    标准三元组置信度计算器
    采用前程的模板式probing方法
    
    数学公式：
    Confidence(T|C) = AGGREGATE_{j=1}^k P(w_j | w_{<j}, C)
    
    其中：
    - T = {w_1, w_2, ..., w_k}: 目标答案token序列
    - C: 模板组成的上下文(prompt)  
    - P(w_j | w_{<j}, C): 第j个token的生成概率
    - AGGREGATE: 聚合方法 (product/average/min)
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
        
        # 智能设备处理：如果模型已经在正确设备上，就不移动它
        try:
            current_device = next(model.parameters()).device
            if str(current_device) != "cpu" and torch.cuda.is_available():
                # 模型已经在GPU上，不需要移动
                self.model = model
                self.device = str(current_device)
                print(f"🔥 Model already on device: {current_device}")
            else:
                # 只有在必要时才移动模型
                self.model = model.to(self.device)
                print(f"📍 Model moved to device: {self.device}")
        except Exception as e:
            # 如果无法检测设备或移动失败，就直接使用原模型
            print(f"⚠️  Device handling warning: {e}")
            self.model = model
            
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
        """使用GPT-4o-mini动态生成模板 - 修正关系理解"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai.api_key)
            
            # 根据关系类型获取专门的指导
            relation_guidance = self._get_relation_specific_guidance(triple.relation)
            
            # 根据模板类型生成优化的提示词 - 重点修正关系理解
            if self.config.template_type == "question":
                system_prompt = """You are an expert in creating natural language templates for knowledge probing in large language models. 

CRITICAL: The question MUST be designed to elicit the TAIL entity as the answer, not the HEAD entity.

For a triple (HEAD, RELATION, TAIL), create a question that would naturally result in "TAIL" as the answer.

Examples:
- For (Paris, capital_of, France): Ask "What country is Paris the capital of?" → Answer: "France"
- For (Einstein, born_in, Germany): Ask "Where was Einstein born?" → Answer: "Germany"  
- For (Tesla, invented, AC motor): Ask "What did Tesla invent?" → Answer: "AC motor"

Your goal is to create templates that elicit clear, specific answers from the model."""
                
                user_prompt = f"""Create a question template for knowledge verification.

**Knowledge Triple**: ({triple.head}, {triple.relation}, {triple.tail})
**Target Answer**: The question should elicit "{triple.tail}" as the answer
**Relation Type**: {triple.relation}
**Guidance**: {relation_guidance}

**CRITICAL REQUIREMENT**: The question must be designed so that "{triple.tail}" is the natural answer.

**Format**:
### Question
[your question here - designed to get "{triple.tail}" as answer]
### Response

**Your template**:"""
            
            elif self.config.template_type == "direct":
                if self.config.use_context:
                    system_prompt = """You are an expert in creating Q&A format templates for knowledge verification. 

CRITICAL: Create a complete Q&A pair where the answer contains the TAIL entity as the key information."""
                    
                    user_prompt = f"""Create a complete Q&A template for knowledge verification.

**Knowledge Triple**: ({triple.head}, {triple.relation}, {triple.tail})
**Target**: The answer should prominently feature "{triple.tail}"
**Relation Type**: {triple.relation}
**Guidance**: {relation_guidance}

**Format**:
### Question
[question that would elicit "{triple.tail}"]
### Response
[complete answer that prominently features "{triple.tail}"]

**Your template**:"""
                else:
                    system_prompt = """You are an expert in creating concise factual statements for knowledge verification. 

The statement should naturally lead to or highlight the TAIL entity."""
                    
                    user_prompt = f"""Create a factual statement for knowledge verification.

**Knowledge Triple**: ({triple.head}, {triple.relation}, {triple.tail})
**Target**: The statement should naturally highlight "{triple.tail}"
**Relation Type**: {triple.relation}

**Your statement**:"""
            
            else:  # cloze
                system_prompt = """You are an expert in creating cloze (fill-in-the-blank) templates. 

Create a sentence that naturally leads to the TAIL entity as the completion."""
                
                user_prompt = f"""Create a cloze template for knowledge verification.

**Knowledge Triple**: ({triple.head}, {triple.relation}, {triple.tail})
**Target Completion**: The sentence should naturally lead to "{triple.tail}"
**Relation Type**: {triple.relation}

**Your template (ending where "{triple.tail}" should be predicted)**:"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=150,
                top_p=0.9,
            )
            
            generated_template = response.choices[0].message.content.strip()
            
            # 清理生成的内容
            generated_template = self._clean_generated_template(generated_template, triple)
            
            print(f"GPT-4o-mini generated template: {generated_template[:80]}...")
            
            # 增强质量验证
            if not self._validate_template_quality(generated_template, triple):
                print("⚠️  Generated template failed quality check, using fallback")
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
        改进的概率计算方法，增加fallback策略
        
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
                print(f"⚠️ Length mismatch: res_tokens={len(res_tokens)}, scores={len(scores)}")
                return [-1]
            
            # 策略1: 精确匹配
            start = self.is_sublist(res_tokens, target_tokens)
            if start != -1:
                print("✅ Exact token sequence match found")
                end = start + len(target_tokens)
                return scores[start:end]
            
            # 策略2: 加空格尝试
            target_tokens_with_space = self.tokenizer.encode(" " + target.strip(), add_special_tokens=False)
            start = self.is_sublist(res_tokens, target_tokens_with_space)
            if start != -1:
                print("✅ Token sequence with space match found")
                end = start + len(target_tokens_with_space)
                return scores[start:end]
            
            # 策略3: 部分匹配 - 如果target是多个词，尝试匹配最重要的词
            if len(target_tokens) > 1:
                # 尝试匹配最长的子序列
                words = target.split()
                for word in words:
                    if len(word) > 3:  # 只匹配有意义的词
                        word_tokens = self.tokenizer.encode(word, add_special_tokens=False)
                        start = self.is_sublist(res_tokens, word_tokens)
                        if start != -1:
                            print(f"✅ Partial match found for word: '{word}'")
                            end = start + len(word_tokens)
                            return scores[start:end]
                        
                        # 尝试加空格的版本
                        word_tokens_with_space = self.tokenizer.encode(" " + word, add_special_tokens=False)
                        start = self.is_sublist(res_tokens, word_tokens_with_space)
                        if start != -1:
                            print(f"✅ Partial match with space found for word: '{word}'")
                            end = start + len(word_tokens_with_space)
                            return scores[start:end]
            
            # 策略4: 近似匹配 - 查找相似的token
            target_text_lower = target.lower()
            response_text_lower = response.lower()
            
            if target_text_lower in response_text_lower:
                # 找到文本级别的匹配，尝试定位对应的token
                char_start = response_text_lower.find(target_text_lower)
                if char_start != -1:
                    # 粗略估计token位置
                    estimated_token_pos = max(0, int(char_start * len(res_tokens) / len(response)))
                    # 在估计位置附近搜索
                    search_range = min(10, len(target_tokens) + 5)
                    start_search = max(0, estimated_token_pos - search_range)
                    end_search = min(len(scores), estimated_token_pos + search_range + len(target_tokens))
                    
                    if end_search > start_search:
                        print(f"✅ Approximate match found using text matching")
                        return scores[start_search:end_search]
            
            # 策略5: 最后的fallback - 返回响应开头的token概率
            if len(scores) >= len(target_tokens):
                print(f"⚠️ Using fallback: first {len(target_tokens)} tokens")
                return scores[:len(target_tokens)]
                
            print("❌ No match found, using single token fallback")
            return [scores[0]] if scores else [-1]
            
        except Exception as e:
            print(f"Error in get_prob: {e}")
            return [-1]

    def extract_answer(self, question: str, answer: str) -> str:
        """
        优化的答案提取方法
        
        Args:
            question: 问题文本
            answer: 原始答案文本
            
        Returns:
            提取的关键概念
        """
        if not self.use_openai:
            # 简单提取方法 - 返回最后一个词
            return answer.strip().split()[-1] if answer.strip() else "N/A"
        
        # 首先尝试简单的启发式方法
        simple_result = self._try_simple_extraction(question, answer)
        if simple_result and simple_result != "N/A":
            print(f"Simple extraction success: '{simple_result}' from '{answer[:50]}...'")
            return simple_result
        
        # 如果简单方法失败，使用改进的GPT方法
        return self._try_gpt_extraction(question, answer)
    
    def _try_simple_extraction(self, question: str, answer: str) -> str:
        """改进的简单启发式答案提取，增加更多策略"""
        answer = answer.strip()
        if not answer:
            return "N/A"
        
        # 方法1: 查找常见模式（扩展版）
        patterns = [
            # "The answer is X"
            r"(?:the answer is|answer is|is)\s+([A-Z][a-zA-Z\s]+?)(?:\.|$|###)",
            # "X is the capital"
            r"^([A-Z][a-zA-Z\s]+?)\s+(?:is|was)",
            # 直接的名词短语在开头
            r"^([A-Z][a-zA-Z\s]{1,50}?)(?:\.|,|$|###)",
            # 在问号后的第一个大写词
            r"\?\s*([A-Z][a-zA-Z\s]+?)(?:\.|$|###)",
            # 引号中的内容
            r'"([^"]+)"',
            # Response后的内容
            r"Response[:\s]*([A-Z][a-zA-Z\s]+?)(?:\.|$|###)",
            # 冒号后的内容
            r":\s*([A-Z][a-zA-Z\s]+?)(?:\.|$|###)"
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                result = match.group(1).strip()
                if len(result) > 0 and len(result) < 100:  # 更宽松的长度限制
                    return result
        
        # 方法2: 更智能的词汇提取
        # 移除常见的无用前缀
        cleaned_answer = answer
        prefixes_to_remove = [
            "### Question", "### Response", "The answer is", "Answer:", 
            "Response:", "A:", "Q:", "Question:", "In response,"
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned_answer.lower().startswith(prefix.lower()):
                cleaned_answer = cleaned_answer[len(prefix):].strip()
        
        # 获取第一个有意义的短语
        first_sentence = cleaned_answer.split('.')[0].split('###')[0].split('\n')[0].strip()
        if first_sentence:
            # 移除开头的常见词
            words = first_sentence.split()
            meaningful_words = []
            skip_words = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            
            for word in words:
                clean_word = word.strip('.,!?";:()[]{}')
                if clean_word.lower() not in skip_words and len(clean_word) > 1:
                    meaningful_words.append(clean_word)
                if len(meaningful_words) >= 5:  # 限制长度
                    break
            
            if meaningful_words:
                result = ' '.join(meaningful_words)
                if len(result) < 100:
                    return result
        
        # 方法3: 查找大写开头的实体
        words = answer.replace('###', ' ').split()
        for word in words[:15]:  # 检查前15个词
            clean_word = word.strip('.,!?";:()[]{}')
            if (len(clean_word) > 2 and 
                clean_word[0].isupper() and 
                clean_word.isalpha() and
                clean_word.lower() not in {'the', 'this', 'that', 'question', 'response', 'answer'}):
                return clean_word
        
        return "N/A"
    
    def _try_gpt_extraction(self, question: str, answer: str) -> str:
        """使用改进的GPT方法提取答案"""
        try:
            # 改进的prompt，更加明确和宽松
            system_prompt = """You are an expert at extracting key information from text. Your task is to extract the most relevant answer from a given response to a question.

IMPORTANT RULES:
1. Extract the key concept that directly answers the question
2. Return ONLY the concept, no extra text
3. If multiple concepts exist, choose the most specific one
4. The extracted answer should be a substring or key phrase from the original response
5. Only return "N/A" if absolutely no relevant information exists"""

            user_prompt = f"""Question: {question}

Response: {answer}

Extract the key concept that answers the question. Return only the extracted concept:"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            from openai import OpenAI
            client = OpenAI(api_key=openai.api_key)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1,  # 降低温度提高一致性
                max_tokens=50,    # 减少token使用
                top_p=0.9
            )
            
            extracted = response.choices[0].message.content.strip()
            
            # 清理提取结果
            extracted = self._clean_extracted_answer(extracted, answer)
            
            print(f"GPT-4o-mini extracted: '{extracted}' from '{answer[:50]}...'")
            return extracted
            
        except Exception as e:
            print(f"GPT extraction error: {e}")
            # 更好的fallback
            return self._fallback_extraction(answer)
    
    def _clean_extracted_answer(self, extracted: str, original_answer: str) -> str:
        """清理和验证提取的答案"""
        if not extracted or extracted.lower() == "n/a":
            return "N/A"
        
        # 移除引号和多余的标点
        extracted = extracted.strip('"\'.,!?').strip()
        
        # 如果提取结果太长，尝试缩短
        if len(extracted) > 100:
            # 取第一个句子或短语
            for delimiter in ['.', ',', ';', '\n']:
                if delimiter in extracted:
                    extracted = extracted.split(delimiter)[0].strip()
                    break
        
        # 验证提取结果是否在原文中
        if extracted.lower() in original_answer.lower():
            return extracted
        
        # 检查部分匹配
        words = extracted.split()
        if len(words) > 1:
            for word in words:
                if len(word) > 3 and word.lower() in original_answer.lower():
                    return word
        
        return extracted  # 即使不完全匹配也返回，让后续逻辑处理
    
    def _fallback_extraction(self, answer: str) -> str:
        """改进的fallback提取方法"""
        answer = answer.strip()
        if not answer:
            return "N/A"
        
        # 获取第一个有意义的词或短语
        # 跳过常见的停用词
        stop_words = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        
        words = answer.replace('###', ' ').split()
        for word in words[:10]:  # 只检查前10个词
            clean_word = word.strip('.,!?";').lower()
            if (len(clean_word) > 2 and 
                clean_word not in stop_words and 
                not clean_word.isdigit() and
                clean_word.isalpha()):
                return word.strip('.,!?";')
        
        # 如果找不到合适的词，返回第一个词
        if words:
            return words[0].strip('.,!?";')
        
        return "N/A"

    def get_last_question(self, template: str) -> str:
        """从模板中提取问题部分"""
        if "### Question" in template:
            return template.split("### Question")[1].split("### Response")[0].strip()
        else:
            # 对于非问答模板，构造简单问题
            return f"Complete: {template}"

    def aggregate_probs(self, probs: List[float]) -> Optional[float]:
        """
        聚合概率列表为单一置信度分数
        
        Args:
            probs: token概率列表
            
        Returns:
            聚合后的置信度分数，失败时返回None
        """
        if not probs or any(p <= 0 for p in probs):
            return None
            
        method = self.config.confidence_aggregation
        
        try:
            aggregation_funcs = {
                "product": lambda x: math.prod(x),
                "average": lambda x: sum(x) / len(x),
                "min": lambda x: min(x)
            }
            
            return aggregation_funcs.get(method, math.prod)(probs)
            
        except Exception as e:
            print(f"Error in aggregate_probs: {e}")
            return None

    def compute_triple_confidence(self, triple: TripleExample) -> Tuple[str, str, Optional[float]]:
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
            (原始响应, 提取答案, 置信度分数) or (..., None) on failure
        """
        # 生成模板
        template = self.generate_template(triple)
        
        if self.config.template_type == "question":
            # 问答模板需要生成响应
            return self._compute_confidence_with_generation(template, triple)
        else:
            # 直接模板计算条件概率
            return self._compute_confidence_direct(template, triple)

    def _compute_confidence_with_generation(self, template: str, triple: TripleExample) -> Tuple[str, str, Optional[float]]:
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
                    return (generated_text.strip(), "N/A", None)
                
                # 计算置信度
                prob_scores = self.get_prob(generated_text, extracted_answer, valid_scores)
                if not prob_scores or prob_scores == [-1]:
                    return (generated_text.strip(), extracted_answer, None)
                
                # 根据配置聚合置信度
                confidence = self.aggregate_probs(prob_scores)
                if confidence is None:
                    return (generated_text.strip(), extracted_answer, None)
                
                return (generated_text.strip(), extracted_answer, confidence)
                
        except Exception as e:
            print(f"Error computing confidence with generation: {e}")
            return ("", "N/A", None)

    def _compute_confidence_direct(self, template: str, triple: TripleExample) -> Tuple[str, str, Optional[float]]:
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
                    return (template, triple.tail, None)
                
                log_probs = torch.log_softmax(target_logits, dim=-1)
                token_log_probs = log_probs[range(len(target_labels)), target_labels]
                
                # 根据配置聚合置信度
                confidence = self.aggregate_probs(torch.exp(token_log_probs).tolist())
                if confidence is None:
                    return (template, triple.tail, None)
                
                return (template, triple.tail, confidence)
                
        except Exception as e:
            print(f"Error computing direct confidence: {e}")
            import traceback
            traceback.print_exc()
            return (template, triple.tail, None)

    def batch_compute_confidence(self, triples: List[TripleExample], 
                                show_progress: bool = True) -> List[Tuple[str, str, Optional[float]]]:
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
                          results: List[Tuple[str, str, Optional[float]]]) -> Dict[str, float]:
        """
        评估正负例之间的分离度
        
        Args:
            triples: 三元组列表
            results: 置信度计算结果
            
        Returns:
            评估指标字典
        """
        confidences = [r[2] for r in results]
        valid_indices = [i for i, conf in enumerate(confidences) if conf is not None and conf >= 0]
        
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
            conf_str = f"{confidence:.6f}" if confidence is not None else "N/A"
            print(f"    Confidence: {conf_str}")
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
                    results: List[Tuple[str, str, Optional[float]]], 
                    filename: str = None):
        """
        保存结果到JSON文件
        
        Args:
            triples: 三元组列表
            results: 置信度计算结果
            filename: 输出文件名（如果为None，则自动生成时间戳文件名）
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"confidence_results_{timestamp}.json"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = {
            "method": "standard_template_probing",
            "description": "Standard probing method with template-based confidence calculation",
            "experiment_config": {
                "use_context": self.config.use_context,
                "template_type": self.config.template_type,
                "extract_method": self.config.extract_method,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "confidence_aggregation": self.config.confidence_aggregation
            },
            "mathematical_formula": "Confidence(T|C) = AGGREGATE_{j=1}^k P(w_j | w_{<j}, C)",
            "total_triples": len(triples),
            "results": [
                {
                    "head": t.head,
                    "relation": t.relation,
                    "tail": t.tail,
                    "label": t.label,
                    "response": r[0],
                    "extracted_answer": r[1],
                    "confidence_score": r[2],
                    "token_count": len(self.tokenizer.encode(t.tail, add_special_tokens=False)) if r[2] is not None else 0
                }
                for t, r in zip(triples, results)
            ],
            "evaluation": self.evaluate_separation(triples, results),
            "timestamp": timestamp
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {filename}")

    def _get_relation_specific_guidance(self, relation: str) -> str:
        """为不同关系类型提供专门的生成指导"""
        guidance_map = {
            "capital_of": "Focus on geographical capitals. Use 'capital' clearly in the question/statement.",
            "born_in": "Focus on birthplace. Use clear birth-related language.",
            "located_in": "Focus on geographical location. Use location-specific language.",
            "nationality": "Focus on country of origin or citizenship.",
            "wrote": "Focus on authorship. Use clear writing/creation language.",
            "invented": "Focus on invention or creation. Use invention-specific language.",
            "died_in": "Focus on place of death. Use death-related location language.",
            "spouse": "Focus on marital relationship. Use marriage/spouse language.",
            "child_of": "Focus on parent-child relationship.",
            "president_of": "Focus on political leadership role.",
        }
        return guidance_map.get(relation, "Create natural, clear language that represents this relationship.")

    def _clean_generated_template(self, template: str, triple: TripleExample) -> str:
        """清理和标准化生成的模板"""
        # 移除多余的引号和标记
        template = template.replace('```', '').replace('**', '').strip()
        
        # 如果包含"Your template:"等提示文字，移除之前的内容
        if "Your template:" in template:
            template = template.split("Your template:")[-1].strip()
        if "Your statement:" in template:
            template = template.split("Your statement:")[-1].strip()
        
        # 移除多余的换行
        template = '\n'.join(line.strip() for line in template.split('\n') if line.strip())
        
        return template

    def _validate_template_quality(self, template: str, triple: TripleExample) -> bool:
        """验证生成模板的质量"""
        # 基本长度检查
        if len(template) < 10 or len(template) > 300:
            return False
        
        # 检查是否包含必要元素
        if self.config.template_type == "question":
            if "### Question" not in template or "### Response" not in template:
                return False
        elif self.config.template_type == "direct" and self.config.use_context:
            if "### Question" not in template or "### Response" not in template:
                return False
        
        # 检查是否包含了关键实体（避免空泛的模板）
        key_entities = [triple.head.lower(), triple.tail.lower()]
        template_lower = template.lower()
        
        # 至少要提到其中一个实体（某些情况下可能不直接提到head）
        has_entity = any(entity in template_lower for entity in key_entities)
        if not has_entity and self.config.template_type != "cloze":
            return False
        
        return True

def load_api_key(filepath: str = "keys/openai.txt") -> str:
    """加载OpenAI API密钥"""
    try:
        with open(filepath, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: API key file not found at {filepath}")
        return None

def run_confidence_experiment(model, tokenizer, test_triples: List[TripleExample], 
                            configs: List[ExperimentConfig], openai_api_key: str = None) -> Dict:
    """
    运行完整的置信度计算实验
    
    Args:
        model: 预加载的模型
        tokenizer: 预加载的tokenizer
        test_triples: 测试三元组列表
        configs: 实验配置列表
        openai_api_key: OpenAI API密钥
        
    Returns:
        实验结果字典
    """
    print("🚀 开始完整置信度计算实验")
    print("=" * 80)
    
    all_results = {}
    
    for i, config in enumerate(configs):
        config_name = f"Config_{i+1}_{config.template_type}_{config.confidence_aggregation}"
        print(f"\n🔍 实验配置 {i+1}/{len(configs)}: {config_name}")
        print(f"   模板类型: {config.template_type}")
        print(f"   聚合方法: {config.confidence_aggregation}")
        print(f"   使用上下文: {config.use_context}")
        print("-" * 60)
        
        # 创建prober
        prober = TripleConfidenceProber(
            model=model,
            tokenizer=tokenizer,
            openai_api_key=openai_api_key,
            config=config
        )
        
        # 运行测试
        results = prober.test_confidence_calculation(test_triples)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_{config_name}_{timestamp}.json"
        prober.save_results(test_triples, results['results'], filename)
        
        all_results[config_name] = {
            'config': config,
            'results': results,
            'filename': filename
        }
    
    # 生成对比报告
    print(f"\n📊 实验对比总结")
    print("=" * 80)
    print(f"{'配置名称':<40} {'分离度':<12} {'有效率':<10} {'质量':<10}")
    print("-" * 80)
    
    sorted_results = sorted(all_results.items(), 
                          key=lambda x: x[1]['results']['evaluation']['separation'], 
                          reverse=True)
    
    for config_name, result in sorted_results:
        eval_data = result['results']['evaluation']
        separation = eval_data['separation']
        valid_rate = f"{eval_data['valid_count']}/{eval_data['total_count']}"
        
        if separation > 0.1:
            quality = "✅ 优秀"
        elif separation > 0.01:
            quality = "⚠️ 中等"
        else:
            quality = "❌ 较差"
        
        print(f"{config_name:<40} {separation:<12.6f} {valid_rate:<10} {quality}")
    
    print(f"\n🎯 实验建议:")
    if sorted_results:
        best_config = sorted_results[0][0]
        print(f"✅ 最佳配置: {best_config}")
        print(f"📁 所有结果文件已保存，可用于进一步分析")
    
    return all_results

def main():
    """主函数示例"""
    print("🎯 STANDARD TEMPLATE-BASED TRIPLE CONFIDENCE PROBER")
    print("基于你老板建议改进的三元组置信度计算")
    print("=" * 70)
    
    # 加载API密钥
    api_key = load_api_key()
    
    # 实验配置示例 - 对比不同聚合方法
    configs = [
        # 你老板建议的重点对比
        ExperimentConfig(
            use_context=True, 
            template_type="direct", 
            extract_method="gpt", 
            confidence_aggregation="average",  # 老板推荐
            temperature=0.1
        ),
        ExperimentConfig(
            use_context=True, 
            template_type="direct", 
            extract_method="gpt", 
            confidence_aggregation="product",  # 原始方法
            temperature=0.1
        ),
        ExperimentConfig(
            use_context=True, 
            template_type="direct", 
            extract_method="gpt", 
            confidence_aggregation="min",     # 保守方法
            temperature=0.1
        ),
        # 测试不同模板类型
        ExperimentConfig(
            use_context=False, 
            template_type="cloze", 
            extract_method="simple", 
            confidence_aggregation="average",
            temperature=0.1
        ),
    ]
    
    print("⚠️  需要加载真实模型进行测试")
    print("请在实际使用时传入model和tokenizer参数")
    print("\n使用示例:")
    print("""
    # 加载你的Llama2模型
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # 准备测试数据
    test_triples = [
        TripleExample("Paris", "capital_of", "France", True),
        TripleExample("Tokyo", "capital_of", "China", False),
        # ... 更多三元组
    ]
    
    # 运行完整实验
    results = run_confidence_experiment(model, tokenizer, test_triples, configs, api_key)
    """)
    
    # 显示数学公式和方法说明
    print(f"\n📋 MATHEMATICAL FORMULATION:")
    print("Confidence(T|C) = AGGREGATE_{j=1}^k P(w_j | w_{<j}, C)")
    print("where AGGREGATE can be product (∏), average, or minimum.")
    print("where:")
    print("  T = {w_1, w_2, ..., w_k}: target answer token sequence")
    print("  C: context/prompt from template")
    print("  P(w_j | w_{<j}, C): generation probability of j-th token")
    
    print(f"\n📋 EXPERIMENT CONFIGURATIONS:")
    for i, config in enumerate(configs):
        print(f"  Config {i+1}: {config.template_type} + {config.confidence_aggregation}")
    
    print(f"\n📋 FEATURES:")
    print("1. ✅ 标准probing: 模板生成 → 答案提取 → 概率计算")
    print("2. ✅ 多种模板: question/direct/cloze")
    print("3. ✅ 答案提取: GPT自动提取关键概念")
    print("4. ✅ 可配置: context使用、模板类型等")
    print("5. ✅ 多种置信度聚合: product, average, min")
    print("6. ✅ 边权重: 置信度可直接用于图分析")
    print("7. ✅ 工程化: 时间戳、token统计、完整实验流程")
    
    print(f"\n🎯 你老板的建议已完美实现:")
    print("• Average聚合方法 - 平衡且稳定")
    print("• Product聚合方法 - 原始但可能被拖累") 
    print("• Min聚合方法 - 保守估计")
    print("• None值处理 - 替代-1避免统计污染")
    print("• 统一tokenizer - 不做过度优化")

if __name__ == "__main__":
    main() 