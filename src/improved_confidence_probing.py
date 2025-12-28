#!/usr/bin/env python3
"""
改进的三元组置信度计算器
主要改进：
1. 更自然的问答模板设计
2. 更好的答案提取方法  
3. 更合理的置信度聚合
"""

import torch
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

@dataclass
class TripleExample:
    """三元组数据结构"""
    head: str
    relation: str
    tail: str
    label: bool = True

@dataclass
class ImprovedConfig:
    """改进的实验配置"""
    template_type: str = "natural_question"  # natural_question, simple_question, cloze
    confidence_aggregation: str = "arithmetic_mean"  # arithmetic_mean, geometric_mean, harmonic_mean
    temperature: float = 0.1
    max_tokens: int = 128
    use_improved_extraction: bool = True

class ImprovedConfidenceProber:
    """改进的三元组置信度计算器"""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                 config: ImprovedConfig = None, device: str = "auto", openai_api_key: str = None):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 智能设备处理
        try:
            current_device = next(model.parameters()).device
            
            # 增强的设备字符串验证逻辑
            device_str = str(current_device)
            if "sk-" in device_str and len(device_str) > 20:
                 # 这是一个明显的 API Key 泄漏到 device 字段的情况
                 print(f"⚠️  CRITICAL: Detected API Key in device field! Suppressing output. Fallback to auto.")
                 self.device = "cuda" if torch.cuda.is_available() else "cpu"
                 # 不再尝试移动模型，假设它已经在正确位置或会在使用时处理
                 self.model = model
            elif device_str != "cpu" and torch.cuda.is_available():
                self.model = model
                self.device = device_str
                print(f"🔥 Model already on device: {self.device}")
            else:
                # 只有在确信是标准设备字符串时才尝试移动
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = model.to(self.device)
                print(f"📍 Model moved to device: {self.device}")
        except Exception as e:
            print(f"⚠️  Device handling warning: {e}")
            self.model = model
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.tokenizer = tokenizer
        self.config = config or ImprovedConfig()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.eval()
        
        # 设置OpenAI API用于模板生成
        self.use_openai = False
        if openai_api_key:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                self.use_openai = True
                print("✅ OpenAI客户端已初始化，可用于模板生成")
            except Exception as e:
                print(f"⚠️  OpenAI初始化失败: {e}")
                self.use_openai = False
        
        print(f"🧪 Improved Config: {self.config}")

    def generate_natural_question_template(self, triple: TripleExample) -> str:
        """生成更自然的问答模板"""
        head, relation, tail = triple.head, triple.relation, triple.tail
        
        # 基于关系类型生成自然问题
        question_patterns = {
            # 地理关系
            "capital_of": f"What country is {head} the capital of?",
            "located_in": f"Where is {head} located?", 
            "born_in": f"Where was {head} born?",
            "died_in": f"Where did {head} die?",
            
            # 人物关系
            "nationality": f"What is {head}'s nationality?",
            "spouse": f"Who is {head} married to?",
            "child_of": f"Who are {head}'s parents?",
            
            # 创作关系
            "wrote": f"What famous work did {head} write?",
            "invented": f"What did {head} invent?",
            "created": f"What did {head} create?",
            "composed": f"What music did {head} compose?",
            
            # 职业关系
            "president_of": f"What country/organization was {head} president of?",
            "CEO_of": f"What company is {head} the CEO of?",
            
            # 时间关系
            "founded_in": f"When was {head} founded?",
            "built_in": f"When was {head} built?",
            
            # 包含关系
            "contains": f"What does {head} contain?",
            "has": f"What does {head} have?",
            "includes": f"What does {head} include?",
            
            # 动作关系
            "promotes": f"What does {head} promote?",
            "supports": f"What does {head} support?",
            "provides": f"What does {head} provide?",
            
            # 特征关系
            "is_known_for": f"What is {head} known for?",
            "is_famous_for": f"What is {head} famous for?",
            "is_characterized_by": f"What characterizes {head}?",
            
            # 其他关系
            "hosts": f"What does {head} host?",
            "showcases": f"What does {head} showcase?",
            "illustrate": f"What does {head} illustrate?",
            "influenced": f"Who did {head} influence?",
        }
        
        # 获取问题，如果没有匹配的模式，使用通用格式
        question = question_patterns.get(relation, f"What is the relationship between {head} and {relation}?")
        
        # 验证无泄漏
        if tail.lower() in question.lower():
            # 如果问题中包含答案，使用更通用的形式
            question = f"What {relation} {head}?"
            
        return f"### Question\n{question}\n### Answer\n"

    def generate_simple_question_template(self, triple: TripleExample) -> str:
        """生成简单问答模板 - 支持任意关系类型"""
        head, relation, tail = triple.head, triple.relation, triple.tail
        
        # 常见关系的特定模板
        specific_patterns = {
            "capital_of": f"What country has {head} as its capital?",
            "located_in": f"Where is {head}?",
            "born_in": f"Where was {head} born?",
            "nationality": f"What nationality is {head}?",
            "wrote": f"What did {head} write?",
            "invented": f"What did {head} invent?",
            "is_known_for": f"What is {head} known for?",
            "is_famous_for": f"What is {head} famous for?",
            "is_a_member_of": f"What is {head} a member of?",
            "is_part_of": f"What is {head} part of?",
            "includes": f"What does {head} include?",
            "encompasses": f"What does {head} encompass?",
            "is_influenced_by": f"What is {head} influenced by?",
            "is_studied_in": f"Where is {head} studied?",
            "offers_programs_in": f"What programs does {head} offer?",
            "is_affiliated_with": f"What is {head} affiliated with?",
        }
        
        # 如果有特定模板，使用特定模板
        if relation in specific_patterns:
            question = specific_patterns[relation]
        else:
            # 通用模板生成策略
            question = self._generate_generic_question(head, relation, tail)
        
        return f"Question: {question}\nAnswer:"

    def _generate_generic_question(self, head: str, relation: str, tail: str) -> str:
        """为任意关系生成通用问题"""
        # 基于关系的语法结构生成问题
        
        # 策略1: 如果关系以"is"开头，通常询问属性
        if relation.startswith("is "):
            if "known for" in relation or "famous for" in relation:
                return f"What is {head} {relation.replace('is ', '')}?"
            elif "located in" in relation:
                return f"Where is {head} located?"
            elif "part of" in relation or "member of" in relation:
                return f"What is {head} {relation.replace('is ', '')}?"
            else:
                return f"What {relation.replace('is ', 'does ')} {head}?"
        
        # 策略2: 如果关系以"has"开头，询问拥有什么
        elif relation.startswith("has "):
            return f"What {relation} {head}?"
        
        # 策略3: 如果关系是动词，询问动作对象
        elif relation in ["wrote", "invented", "created", "founded", "built", "developed"]:
            return f"What did {head} {relation}?"
        
        # 策略4: 如果关系是"include"类型
        elif "include" in relation:
            return f"What does {head} {relation}?"
        
        # 策略5: 如果关系是"study"类型  
        elif "stud" in relation:
            return f"Where is {head} studied?"
        
        # 策略6: 通用回退模板
        else:
            # 根据关系类型构造更自然的问题
            relation_lower = relation.lower()
            
            # HasXXX类型关系
            if relation.startswith("Has"):
                object_type = relation[3:]  # 去掉"Has"前缀
                if object_type == "Member":
                    return f"Which member does {head} have?"
                elif object_type == "Instance":
                    return f"What is an instance of {head}?"
                elif object_type == "Part":
                    return f"What is a part of {head}?"
                elif object_type == "Property":
                    return f"What is a key property of {head}?"
                else:
                    return f"What {object_type.lower()} does {head} have?"
            
            # IsXXX类型关系  
            elif relation.startswith("Is"):
                return f"What is {head}?"
            
            # LocatedXXX类型关系
            elif "Located" in relation:
                return f"Where is {head} located?"
            
            # 其他关系的通用处理
            elif len(relation.split()) == 1:
                # 单词关系 - 避免技术性表达
                return f"What is related to {head} through {relation}?"
            else:
                # 多词关系，尝试重新组织
                words = relation.split()
                if words[0] in ["is", "are", "was", "were"]:
                    return f"What {' '.join(words)} {head}?"
                else:
                    return f"What {relation} does {head} have?"

    def generate_cloze_template(self, triple: TripleExample) -> str:
        """生成完形填空模板"""
        head, relation, tail = triple.head, triple.relation, triple.tail
        
        cloze_patterns = {
            "capital_of": f"{head} is the capital of",
            "located_in": f"{head} is located in",
            "born_in": f"{head} was born in",
            "nationality": f"{head} is from",
            "wrote": f"{head} wrote",
            "invented": f"{head} invented",
        }
        
        return cloze_patterns.get(relation, f"{head} {relation}")

    def generate_openai_template(self, triple: TripleExample) -> str:
        """使用OpenAI API生成优化的自然问题模板"""
        if not self.use_openai:
            return self.generate_simple_question_template(triple)
        
        try:
            head, relation, tail = triple.head, triple.relation, triple.tail
            
            # 优化的prompt - 强调简单性和直接性
            system_prompt = """You are an expert at creating simple, direct questions for knowledge evaluation.

Your task is to create a question that is:
1. Simple and clear (under 15 words)
2. Natural English phrasing
3. Directly answerable with the target entity
4. Free of complex clauses or modifiers

CRITICAL: The question must be straightforward and lead naturally to the target answer."""

            user_prompt = f"""Create a simple, direct question for this knowledge triple:

Knowledge: ({head}, {relation}, {tail})
Target Answer: "{tail}"

Requirements:
1. Question must be simple and clear (under 15 words)
2. Should ask about {head}'s {relation}
3. Answer should contain "{tail}"
4. Use most natural English expression
5. Avoid complex subordinate clauses

Example formats:
- What is the capital of France? (Answer: Paris)
- Where was Einstein born? (Answer: Germany) 
- What did Shakespeare write? (Answer: Hamlet)

Please provide ONLY the question, no other content:"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            question = response.choices[0].message.content.strip()
            
            # 严格的质量验证
            if self._validate_openai_question_quality(question, triple):
                return f"Question: {question}\nAnswer:"
            else:
                # 如果质量不好，降级到简单问题
                print(f"⚠️ OpenAI问题质量不佳，降级到简单问题: {question}")
                return self.generate_simple_question_template(triple)
            
        except Exception as e:
            print(f"❌ OpenAI生成失败: {e}，降级到简单问题")
            return self.generate_simple_question_template(triple)
    
    def _validate_openai_question_quality(self, question: str, triple: TripleExample) -> bool:
        """验证OpenAI生成问题的质量"""
        # 清理问题
        question = question.replace('"', '').replace("'", "").strip()
        if not question.endswith('?'):
            question += '?'
        
        # 检查问题长度
        if len(question.split()) > 20:
            return False
        
        # 检查是否包含head实体
        if triple.head.lower() not in question.lower():
            return False
        
        # 检查是否是问句
        if not question.strip().endswith('?'):
            return False
        
        # 检查是否包含答案（泄漏检查）
        if triple.tail.lower() in question.lower():
            return False
        
        # 检查是否过于复杂
        complex_phrases = ['furthermore', 'moreover', 'additionally', 'characterized by', 'encompasses', 'which is', 'that is']
        if any(phrase in question.lower() for phrase in complex_phrases):
            return False
        
        return True

    def generate_template(self, triple: TripleExample) -> str:
        """根据配置生成模板"""
        if self.config.template_type == "openai_generated" and self.use_openai:
            return self.generate_openai_template(triple)
        elif self.config.template_type == "natural_question":
            return self.generate_natural_question_template(triple)
        elif self.config.template_type == "simple_question":
            return self.generate_simple_question_template(triple)
        elif self.config.template_type == "cloze":
            return self.generate_cloze_template(triple)
        else:
            return self.generate_simple_question_template(triple)  # 默认使用效果最好的

    def improved_answer_extraction(self, question: str, response: str, target: str) -> str:
        """改进的答案提取方法"""
        if not response.strip():
            return "N/A"
        
        response = response.strip()
        
        # 策略1: 查找明确的答案标记
        answer_patterns = [
            r"(?:Answer|answer)[:\s]*([^.\n]+)",
            r"(?:The answer is|Answer is|is)\s+([^.\n]+)",
            r"^([A-Z][^.\n]*?)(?:\.|$)",  # 开头的大写句子
            r"(?:It is|This is|That is)\s+([^.\n]+)",
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if self._validate_extraction(extracted, target):
                    return extracted
        
        # 策略2: 针对OpenAI问题的特殊提取
        if hasattr(self, 'config') and self.config.template_type == "openai_generated":
            openai_result = self.extract_answer_for_openai(response, target)
            if openai_result:
                return openai_result
        
        # 策略3: 基于目标答案的部分匹配
        target_words = target.lower().split()
        response_lower = response.lower()
        
        # 查找包含目标词的句子
        sentences = re.split(r'[.!?]', response)
        best_sentence = ""
        max_overlap = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 3:
                continue
                
            overlap = sum(1 for word in target_words if word in sentence.lower())
            if overlap > max_overlap:
                max_overlap = overlap
                best_sentence = sentence
        
        if max_overlap > 0 and best_sentence:
            return best_sentence.strip()
        
        # 策略3: 提取第一个有意义的名词短语
        # 移除常见的前缀
        cleaned = response
        prefixes = ["Question:", "Answer:", "The answer is", "It is", "This is", "That is"]
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
                if cleaned.startswith(":"):
                    cleaned = cleaned[1:].strip()
                break
        
        # 获取第一个句子
        first_sentence = re.split(r'[.!?\n]', cleaned)[0].strip()
        if first_sentence and len(first_sentence) < 200:
            return first_sentence
        
        # 策略4: 回退到前几个词
        words = response.split()[:10]  # 取前10个词
        meaningful_words = []
        stop_words = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word and clean_word not in stop_words and len(clean_word) > 1:
                meaningful_words.append(word)
            if len(meaningful_words) >= 3:
                break
        
        if meaningful_words:
            return ' '.join(meaningful_words)
        
        return "N/A"

    def _validate_extraction(self, extracted: str, target: str) -> bool:
        """验证提取结果的质量"""
        if not extracted or extracted == "N/A":
            return False
        
        # 长度检查
        if len(extracted) > 200:
            return False
        
        # 检查是否包含目标的关键词
        extracted_lower = extracted.lower()
        target_lower = target.lower()
        
        # 如果目标较短，检查完全匹配
        if len(target) <= 3:
            return target_lower in extracted_lower
        
        # 对于较长的目标，检查关键词重叠
        target_words = set(target_lower.split())
        extracted_words = set(extracted_lower.split())
        overlap = len(target_words & extracted_words)
        
        return overlap > 0 or target_lower in extracted_lower

    def aggregate_token_probabilities(self, token_probs: List[float]) -> Optional[float]:
        """
        改进的概率聚合方法 - 实现老板建议的多种方式
        
        方法说明：
        1. product (乘积): 传统方法，但容易被低概率token拖累
        2. arithmetic_mean (算术平均): 平衡各token贡献
        3. min_confidence (最小值): 保守估计，取最低confidence
        4. geometric_mean (几何平均): 原始几何平均
        5. harmonic_mean (调和平均): 对低值敏感但比乘积温和
        """
        if not token_probs or any(p <= 0 for p in token_probs):
            return None
        
        method = self.config.confidence_aggregation
        
        try:
            if method == "product":
                # 方法1: 直接乘积 (传统方法，容易被低概率拖累)
                result = 1.0
                for p in token_probs:
                    result *= p
                return result
                
            elif method == "arithmetic_mean":
                # 方法2: 算术平均 (平衡各token贡献)
                return sum(token_probs) / len(token_probs)
                
            elif method == "min_confidence":
                # 方法3: 最小置信度 (保守估计，answer span中最不确定的token)
                return min(token_probs)
                
            elif method == "geometric_mean":
                # 几何平均 (使用对数避免下溢)
                log_probs = [np.log(p) for p in token_probs]
                return np.exp(sum(log_probs) / len(log_probs))
                
            elif method == "harmonic_mean":
                # 调和平均 (对低值敏感但比乘积温和)
                return len(token_probs) / sum(1/p for p in token_probs)
                
            elif method == "weighted_mean":
                # 加权平均 (给后面的token更高权重)
                weights = [i + 1 for i in range(len(token_probs))]
                weighted_sum = sum(p * w for p, w in zip(token_probs, weights))
                return weighted_sum / sum(weights)
                
            elif method == "max_confidence":
                # 最大置信度 (乐观估计)
                return max(token_probs)
                
            elif method == "median_confidence":
                # 中位数 (抗异常值)
                sorted_probs = sorted(token_probs)
                n = len(sorted_probs)
                if n % 2 == 0:
                    return (sorted_probs[n//2-1] + sorted_probs[n//2]) / 2
                else:
                    return sorted_probs[n//2]
                    
            else:
                # 默认使用算术平均
                return sum(token_probs) / len(token_probs)
                
        except Exception as e:
            print(f"聚合概率时出错: {e}")
            return None

    def compute_confidence_improved(self, triple: TripleExample) -> Tuple[str, str, Optional[float]]:
        """改进的置信度计算方法"""
        try:
            # 生成模板
            template = self.generate_template(triple)
            
            # 编码输入
            inputs = self.tokenizer(template, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 设置attention_mask避免警告
            if 'attention_mask' not in inputs:
                inputs['attention_mask'] = inputs['input_ids'].ne(self.tokenizer.pad_token_id)
            
            with torch.no_grad():
                # 生成响应
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # 获取生成的文本
                generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # 提取问题和答案
                if "### Question" in template:
                    question = template.split("### Question")[1].split("### Answer")[0].strip()
                elif "Question:" in template:
                    question = template.split("Question:")[1].split("Answer:")[0].strip()
                else:
                    question = f"Complete: {template}"
                
                # 改进的答案提取
                if self.config.use_improved_extraction:
                    extracted_answer = self.improved_answer_extraction(question, generated_text, triple.tail)
                else:
                    # 简单提取：取第一个句子
                    extracted_answer = generated_text.split('.')[0].strip() if generated_text else "N/A"
                
                if extracted_answer == "N/A":
                    return (generated_text, "N/A", None)
                
                # 计算提取答案的token概率
                target_tokens = self.tokenizer.encode(extracted_answer, add_special_tokens=False)
                
                if not target_tokens or len(target_tokens) == 0:
                    return (generated_text, extracted_answer, None)
                
                # 获取token概率
                if outputs.scores and len(outputs.scores) > 0:
                    token_probs = []
                    
                    for i, token_id in enumerate(target_tokens):
                        if i < len(outputs.scores):
                            # 获取当前位置的概率分布
                            probs = torch.softmax(outputs.scores[i], dim=-1)
                            if probs.dim() == 2:
                                probs = probs[0]  # 移除batch维度
                            
                            # 获取目标token的概率
                            if token_id < len(probs):
                                prob = probs[token_id].item()
                                token_probs.append(prob)
                    
                    if token_probs:
                        confidence = self.aggregate_token_probabilities(token_probs)
                        return (generated_text, extracted_answer, confidence)
                
                return (generated_text, extracted_answer, None)
                
        except Exception as e:
            print(f"计算置信度时出错: {e}")
            return ("", "N/A", None)

    def test_on_examples(self, examples: List[TripleExample]) -> Dict:
        """在示例上测试改进效果"""
        print(f"\n🧪 测试改进的置信度计算 ({len(examples)} 个例子)")
        print(f"📋 配置: {self.config.template_type}, {self.config.confidence_aggregation}")
        print("-" * 80)
        
        results = []
        for i, triple in enumerate(examples):
            print(f"\n示例 {i+1}: ({triple.head}, {triple.relation}, {triple.tail})")
            
            response, extracted, confidence = self.compute_confidence_improved(triple)
            
            # 生成的模板
            template = self.generate_template(triple)
            print(f"📝 模板: {template[:100]}...")
            print(f"🤖 生成: {response[:150]}...")
            print(f"🎯 提取: {extracted}")
            print(f"📊 置信度: {confidence:.4f}" if confidence else "📊 置信度: 失败")
            
            results.append({
                'triple': triple,
                'template': template,
                'response': response,
                'extracted': extracted,
                'confidence': confidence,
                'target': triple.tail
            })
            
        # 统计
        valid_confidences = [r['confidence'] for r in results if r['confidence'] is not None]
        if valid_confidences:
            print(f"\n📈 统计结果:")
            print(f"  成功率: {len(valid_confidences)}/{len(results)} ({len(valid_confidences)/len(results)*100:.1f}%)")
            print(f"  平均置信度: {np.mean(valid_confidences):.4f}")
            print(f"  置信度范围: {min(valid_confidences):.4f} - {max(valid_confidences):.4f}")
            print(f"  标准差: {np.std(valid_confidences):.4f}")
        
        return {
            'config': self.config,
            'results': results,
            'statistics': {
                'success_rate': len(valid_confidences) / len(results) if results else 0,
                'mean_confidence': np.mean(valid_confidences) if valid_confidences else 0,
                'std_confidence': np.std(valid_confidences) if valid_confidences else 0,
                'min_confidence': min(valid_confidences) if valid_confidences else 0,
                'max_confidence': max(valid_confidences) if valid_confidences else 0
            }
        }
    
    def extract_answer_for_openai(self, text: str, target: str) -> str:
        """针对OpenAI问题优化的答案提取"""
        # 策略1: 寻找包含target的句子
        sentences = text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if target.lower() in sentence.lower() and len(sentence) < 200:
                # 清理句子
                cleaned = self.clean_sentence(sentence)
                if cleaned:
                    return cleaned
        
        # 策略2: 寻找直接回答
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Question:') and len(line) < 150:
                # 如果包含目标关键词
                if target.lower() in line.lower():
                    cleaned = self.clean_sentence(line)
                    if cleaned:
                        return cleaned
        
        # 策略3: 提取第一个有意义的句子
        import re
        # 移除"Answer:"前缀
        text = re.sub(r'^.*?Answer:\s*', '', text, flags=re.IGNORECASE)
        
        # 找到第一个完整句子
        match = re.search(r'([^.!?]*[.!?])', text)
        if match:
            sentence = match.group(1).strip()
            cleaned = self.clean_sentence(sentence)
            if cleaned and len(cleaned) > 5:
                return cleaned
        
        # 策略4: 返回前50个字符
        cleaned_text = text[:50].strip()
        return self.clean_sentence(cleaned_text) if cleaned_text else ""
    
    def clean_sentence(self, sentence: str) -> str:
        """清理句子"""
        import re
        
        # 移除特殊标记
        sentence = re.sub(r'^\d+\.\s*', '', sentence)  # 移除数字编号
        sentence = re.sub(r'^[*-]\s*', '', sentence)   # 移除列表标记
        sentence = re.sub(r'^Answer:\s*', '', sentence, flags=re.IGNORECASE)  # 移除Answer:
        sentence = sentence.strip()
        
        # 如果太短，跳过
        if len(sentence) < 3:
            return ""
        
        return sentence
