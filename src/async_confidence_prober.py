#!/usr/bin/env python3
"""
异步置信度计算器
解决置信度计算失败的问题，提高成功率
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import json
import time
import random
import math
from improved_confidence_probing import ImprovedConfidenceProber, ImprovedConfig, TripleExample

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# 🔬 两阶段Tail概率计算器 - 基于大规模测试验证的成功方法
# ============================================================================

class TailProbabilityCalculator:
    """
    两阶段Tail概率计算器
    
    基于30个三元组大规模测试验证：
    - Stage 1: Exact Match (96.7%成功率)
    - Stage 2: LLM Extraction (3.3%使用率)
    - Fallback: Position 0 (兜底)
    
    验证结果：
    - 平均置信度: 65.4%
    - 概率范围: 33%-93%
    - 能区分不同难度的知识
    """
    
    def __init__(self, tokenizer, openai_client=None):
        self.tokenizer = tokenizer
        self.openai_client = openai_client
    
    def compute_tail_probability_two_stage(
        self, 
        expected_tail: str,
        generated_ids: torch.Tensor,  # shape: [seq_len]
        scores: List[torch.Tensor],   # List of [vocab_size] tensors
        generated_text: str,
        question: str = None
    ) -> Dict:
        """
        两阶段Tail概率计算（向后兼容版本）
        
        Returns:
            {
                'method': 'exact_match' | 'llm_extraction' | 'position0_fallback',
                'exact_match_position': int | None,
                'tail_probability': float | None,
                'tail_log_probability': float | None,
                'extracted_answer': str,
                'llm_extracted': bool
            }
        """
        result = {
            'method': None,
            'exact_match_position': None,
            'tail_probability': None,
            'tail_log_probability': None,
            'extracted_answer': expected_tail,
            'llm_extracted': False,
            'position0_probability': None
        }
        
        # Tokenize expected tail（修复bug：不加空格）
        tail_tokens = self.tokenizer.encode(expected_tail, add_special_tokens=False)
        
        if len(tail_tokens) == 0:
            return result
        
        # ===== Stage 1: Exact Match in Generated Sequence =====
        for start_pos in range(len(generated_ids)):
            if start_pos + len(tail_tokens) > len(generated_ids):
                break
            
            # 检查完全匹配
            match = all(
                generated_ids[start_pos + i].item() == tail_tokens[i] 
                for i in range(len(tail_tokens))
            )
            
            if match:
                # 找到exact match! 计算该位置的联合概率
                joint_prob = 1.0
                for i, tail_token in enumerate(tail_tokens):
                    pos = start_pos + i
                    if pos < len(scores):
                        pos_probs = torch.softmax(scores[pos], dim=-1)
                        joint_prob *= pos_probs[tail_token].item()
                
                result['method'] = 'exact_match'
                result['exact_match_position'] = start_pos
                result['tail_probability'] = joint_prob
                result['tail_log_probability'] = math.log(joint_prob) if joint_prob > 0 else float('-inf')
                result['extracted_answer'] = expected_tail
                
                # 同时计算Position 0概率（用于对比）
                result['position0_probability'] = self._compute_position0_probability(tail_tokens, scores)
                
                return result
        
        # ===== Stage 2: LLM Extraction (如果Stage 1失败) =====
        if self.openai_client and question:
            extracted_answer = self._extract_answer_with_llm(
                question, generated_text, expected_tail
            )
            
            if extracted_answer:
                extracted_tokens = self.tokenizer.encode(extracted_answer, add_special_tokens=False)
                
                # 在生成序列中查找提取的答案
                for start_pos in range(len(generated_ids)):
                    if start_pos + len(extracted_tokens) > len(generated_ids):
                        break
                    
                    match = all(
                        generated_ids[start_pos + i].item() == extracted_tokens[i] 
                        for i in range(len(extracted_tokens))
                    )
                    
                    if match:
                        joint_prob = 1.0
                        for i, ext_token in enumerate(extracted_tokens):
                            pos = start_pos + i
                            if pos < len(scores):
                                pos_probs = torch.softmax(scores[pos], dim=-1)
                                joint_prob *= pos_probs[ext_token].item()
                        
                        result['method'] = 'llm_extraction'
                        result['exact_match_position'] = start_pos
                        result['tail_probability'] = joint_prob
                        result['tail_log_probability'] = math.log(joint_prob) if joint_prob > 0 else float('-inf')
                        result['extracted_answer'] = extracted_answer
                        result['llm_extracted'] = True
                        
                        return result
        
        # ===== Fallback: Position 0 Probability =====
        # print(f"[DEBUG] Fallback entered. Generated text: '{generated_text}'")
        pos0_prob = self._compute_position0_probability(tail_tokens, scores)
        
        result['method'] = 'position0_fallback'
        result['tail_probability'] = pos0_prob
        result['tail_log_probability'] = math.log(pos0_prob) if pos0_prob > 0 else float('-inf')
        result['position0_probability'] = pos0_prob
        
        # [FIX] Do NOT default to expected_tail if no match found.
        # Instead, use the raw generated text (cleaned) so we know what the model actually said.
        # This is critical for error analysis (Ripple Effect).
        result['extracted_answer'] = generated_text.strip()
        # print(f"[DEBUG] Fallback result extracted_answer: '{result['extracted_answer']}'")
        
        return result
    
    def _compute_position0_probability(self, tail_tokens: List[int], scores: List[torch.Tensor]) -> float:
        """计算tail在position 0的联合概率"""
        if len(scores) < len(tail_tokens):
            return 0.0
        
        joint_prob = 1.0
        for i, tail_token in enumerate(tail_tokens):
            if i < len(scores):
                pos_probs = torch.softmax(scores[i], dim=-1)
                joint_prob *= pos_probs[tail_token].item()
        
        return joint_prob
    
    def _extract_answer_with_llm(self, question: str, model_response: str, expected_tail: str) -> Optional[str]:
        """由于本地评测需要极高的速度，直接回退为简单的字符串匹配，而不是调用堵塞的OpenAI API"""
        model_response_lower = model_response.lower()
        expected_tail_lower = expected_tail.lower()
        
        # 直接字符串匹配
        if expected_tail_lower in model_response_lower:
            return expected_tail
            
        # 简单清洗后的匹配
        clean_response = model_response_lower.replace(".", "").replace(",", "").replace("\n", " ").strip()
        if expected_tail_lower in clean_response:
            return expected_tail
            
        # 如果模型直接复读了尾节点，也算匹配
        words = model_response_lower.split()
        expected_words = expected_tail_lower.split()
        if all(w in words for w in expected_words):
            return expected_tail
            
        return None

# --- NEW: Question Template Bank ---
# Based on the 24 canonical relations, providing natural phrasing scaffolds.
TEMPLATE_BANK = {
  # Structure
  "InstanceOf": ["What type of thing is {H}?", "What is {H} an instance of?"],
  "HasInstance": ["What is an instance of {H}?", "Which instance does {H} have?"],
  "SubclassOf": ["What broader class does {H} belong to?", "{H} is a subclass of what?"],
  "PartOf": ["{H} is part of what?", "{H} belongs to which whole?"],
  "HasPart": ["What is a part of {H}?", "Which component does {H} include?"],
  "MemberOf": ["{H} is a member of which group?", "Which organization is {H} part of?"],
  "HasMember": ["Which member does {H} have?", "What member is part of {H}?"],
  # Attributes
  "HasProperty": ["What is a key property of {H}?", "What characteristic describes {H}?"],
  "MadeOf": ["What is {H} made of?", "{H} is made of what material?"],
  "Genre": ["What is the genre of {H}?", "{H} falls under which genre?"],
  # Spatial
  "LocatedIn": ["Where is {H} located?", "{H} is located in which place?"],
  "LocatedNear": ["{H} is near what?", "{H} is close to which place?"],
  "CapitalOf": ["Which country is {H} the capital of?", "{H} is the capital of which country?"],
  "BorderWith": ["Which country borders {H}?", "{H} shares a border with which country?"],
  # Temporal
  "StartTime": ["When did {H} start?", "What is the start date of {H}?"],
  "EndTime": ["When did {H} end?", "What is the end date of {H}?"],
  "OccursOn": ["When did {H} occur?", "On what date did {H} happen?"],
  # Causal/Event
  "Causes": ["What does {H} cause?", "{H} leads to what?"],
  "HasPrerequisite": ["What is a prerequisite for {H}?", "{H} requires what first?"],
  "HasSubevent": ["What is a sub-event of {H}?", "{H} includes which sub-event?"],
  # Functionality
  "UsedFor": ["What is {H} used for?", "{H} is used for what?"],
  "CapableOf": ["What can {H} do?", "{H} is capable of what?"],
  # Social/Role
  "Occupation": ["What is {H}'s occupation?", "{H} works as what?"],
  "Employer": ["Who employs {H}?", "{H} works for which organization?"],
  "CreatedBy": ["Who created {H}?", "{H} was created by whom?"],
  "HeadquarteredIn": ["Where is {H} headquartered?", "{H} is headquartered in which city?"],
  # Optional
  "Nationality": ["What is {H}'s nationality?", "{H} is a citizen of which country?"],
  "LanguageUsed": ["What language does {H} use?", "{H} is in which language?"],
  "DevelopedBy": ["Who developed {H}?", "{H} was developed by whom?"],
  "ManufacturedBy": ["Who manufactures {H}?", "{H} is produced by which company?"],
  "NamedAfter": ["What is {H} named after?", "{H} is named after whom or what?"],
  "DiplomaticRelation": ["{H} has diplomatic relations with which country?", "Which country does {H} have diplomatic ties with?"],
}

def pick_template_hint(relation: str, head: str) -> str:
    """Selects 1-2 template patterns from the bank to guide the LLM."""
    pats = TEMPLATE_BANK.get(relation, [])
    # Replace {H} with the head to give the LLM a concrete "feel" for the question.
    return "\n- " + "\n- ".join(p.replace("{H}", head) for p in pats[:2]) if pats else "N/A"

def infer_tail_type(tail: str) -> str:
    """推断tail的类型以提供更好的问题生成hint"""
    tail_lower = tail.lower()
    
    # 地理位置
    if any(word in tail_lower for word in ['state', 'country', 'city', 'district', 'county', 'province', 'region', 'california', 'texas', 'china', 'france', 'massachusetts', 'maharashtra']):
        return "location"
    
    # 人名
    if any(word in tail_lower for word in ['shakespeare', 'william', 'john', 'mary', 'author', 'writer', 'director', 'president']):
        return "person"
    
    # 组织机构
    if any(word in tail_lower for word in ['company', 'corporation', 'university', 'school', 'government', 'organization']):
        return "organization"
    
    # 时间
    if any(word in tail_lower for word in ['year', 'date', 'time', 'century', 'ago', '19', '20']) or tail.isdigit():
        return "time"
    
    # 属性特征
    if any(word in tail_lower for word in ['property', 'characteristic', 'feature', 'quality', 'beauty', 'power']):
        return "property"
    
    # 材料
    if any(word in tail_lower for word in ['material', 'plastic', 'wood', 'metal', 'stone', 'glass']):
        return "material"
    
    # 概念/抽象名词
    if any(word in tail_lower for word in ['concept', 'idea', 'theory', 'principle', 'catharsis', 'drama', 'genre']):
        return "concept"
    
    # 事件/活动
    if any(word in tail_lower for word in ['event', 'activity', 'ceremony', 'festival', 'war', 'battle']):
        return "event"
    
    # 默认：实体
    return "entity"

@dataclass
class RetryConfig:
    """重试配置"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 10.0
    exponential_base: float = 2.0
    jitter: bool = True

class AsyncConfidenceProber(ImprovedConfidenceProber):
    """异步置信度计算器，减少失败率"""
    
    def __init__(
        self,
        model,
        tokenizer,
        config: ImprovedConfig,
        openai_api_key=None,
        retry_config=None,
        enable_margin_dump: bool = False,
        enable_attention_dump: bool = False,
    ):
        super().__init__(model, tokenizer, config, openai_api_key)
        self.retry_config = retry_config or RetryConfig()
        self.enable_margin_dump = enable_margin_dump
        self.enable_attention_dump = enable_attention_dump
        self.session = None
        self._setup_session()

        # --- 新增：用于动态批处理的组件 ---
        self.batch_queue = asyncio.Queue()
        self.processing_task = asyncio.create_task(self._batch_processing_loop())
        self.batch_size = 128  # 针对A100优化：提升到128
        self.batch_timeout = 0.05  # 50ms, 等待更多任务的最长时间
        # --- 结束新增 ---
        
        # ✅ 🔬 新增：初始化OpenAI客户端（用于LLM extraction - Stage 2）
        if openai_api_key:
            import os
            os.environ['OPENAI_API_KEY'] = openai_api_key
            from openai import OpenAI
            self.openai_client = OpenAI()
        else:
            self.openai_client = None
        
        # ✅ 🔬 新增：初始化Tail概率计算器（两阶段策略）
        # [PERFORMANCE OPTIMIZATION] 
        # Pass openai_client=None to disable synchronous Stage 2 (LLM Extraction).
        # Stage 2 causes severe blocking issues (0% GPU util), slowing down evaluation to ~3it/s.
        # For base model cloze tasks, Stage 1 (Exact Match) and Fallback are sufficient.
        self.tail_probability_calculator = TailProbabilityCalculator(
            tokenizer=self.tokenizer,
            openai_client=None # self.openai_client -> None to prevent sync blocking
        )
    
    def _setup_session(self):
        """设置异步HTTP会话"""
        timeout = aiohttp.ClientTimeout(total=60, connect=10)
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    
    async def _exponential_backoff_delay(self, attempt: int) -> float:
        """指数退避延迟"""
        delay = min(
            self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt),
            self.retry_config.max_delay
        )
        
        if self.retry_config.jitter:
            delay *= (0.5 + random.random() * 0.5)  # 添加抖动
        
        return delay
    
    async def _async_openai_call_with_retry(self, messages: List[Dict], model="gpt-4o-mini") -> Optional[str]:
        """带重试的异步OpenAI API调用"""
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 50
        }
        
        for attempt in range(self.retry_config.max_retries):
            try:
                async with self.session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        return data['choices'][0]['message']['content'].strip()
                    
                    elif response.status == 429:  # Rate limit
                        error_data = await response.json()
                        logger.warning(f"Rate limit hit (attempt {attempt + 1}): {error_data}")
                        if attempt < self.retry_config.max_retries - 1:
                            delay = await self._exponential_backoff_delay(attempt)
                            await asyncio.sleep(delay)
                            continue
                    
                    elif response.status >= 500:  # Server error
                        error_data = await response.json()
                        logger.warning(f"Server error {response.status} (attempt {attempt + 1}): {error_data}")
                        if attempt < self.retry_config.max_retries - 1:
                            delay = await self._exponential_backoff_delay(attempt)
                            await asyncio.sleep(delay)
                            continue
                    
                    else:
                        error_data = await response.json()
                        logger.error(f"API error {response.status}: {error_data}")
                        break
            
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt < self.retry_config.max_retries - 1:
                    delay = await self._exponential_backoff_delay(attempt)
                    await asyncio.sleep(delay)
                    continue
            
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < self.retry_config.max_retries - 1:
                    delay = await self._exponential_backoff_delay(attempt)
                    await asyncio.sleep(delay)
                    continue
        
        logger.error(f"All {self.retry_config.max_retries} attempts failed")
        return None

    def _compute_margin_diagnostics(self, triple: TripleExample, logits: Optional[torch.Tensor]) -> Dict[str, Any]:
        """Compute first-token raw-logit margin diagnostics for the expected tail."""
        diagnostics = {
            "correct_logit": None,
            "top_incorrect_logit": None,
            "margin": None,
            "correct_token_rank": None,
            "predicted_token_id": None,
            "predicted_token_text": None,
            "tail_first_token_id": None,
        }

        if logits is None or triple is None:
            return diagnostics

        tail_tokens = self.tokenizer.encode(triple.tail, add_special_tokens=False)
        if not tail_tokens:
            return diagnostics

        if logits.dim() > 1:
            logits = logits[0]
        logits = logits.float()

        tail_token = int(tail_tokens[0])
        if tail_token < 0 or tail_token >= logits.shape[-1]:
            return diagnostics

        try:
            if not torch.isfinite(logits[tail_token]):
                return diagnostics
            correct_logit = float(logits[tail_token].item())

            finite_mask = torch.isfinite(logits)
            candidate_mask = finite_mask.clone()
            candidate_mask[tail_token] = False
            if not torch.any(candidate_mask):
                return diagnostics
            candidate_logits = logits.masked_fill(~candidate_mask, float("-inf"))
            top_incorrect_logit = float(candidate_logits.max().item())
            if not math.isfinite(top_incorrect_logit):
                return diagnostics

            pred_logits = logits.masked_fill(~finite_mask, float("-inf"))
            predicted_token_id = int(torch.argmax(pred_logits).item())
            rank = int((pred_logits > logits[tail_token]).sum().item()) + 1

            if predicted_token_id < 0 or predicted_token_id >= logits.shape[-1]:
                predicted_token_id = None

            predicted_token_text = None
            if predicted_token_id is not None:
                predicted_token_text = self.tokenizer.decode([predicted_token_id]).strip()

            diagnostics["correct_logit"] = correct_logit
            diagnostics["top_incorrect_logit"] = top_incorrect_logit
            diagnostics["margin"] = correct_logit - top_incorrect_logit
            diagnostics["correct_token_rank"] = rank
            diagnostics["predicted_token_id"] = predicted_token_id
            diagnostics["predicted_token_text"] = predicted_token_text
            diagnostics["tail_first_token_id"] = tail_token
        except Exception as e:
            logger.debug(f"Failed to compute margin diagnostics: {e}")

        return diagnostics

    @staticmethod
    def _find_subsequence(sequence: List[int], pattern: List[int]) -> Optional[Tuple[int, int]]:
        """Find first [start, end) span of pattern in sequence."""
        if not sequence or not pattern or len(pattern) > len(sequence):
            return None
        limit = len(sequence) - len(pattern) + 1
        for i in range(limit):
            if sequence[i : i + len(pattern)] == pattern:
                return i, i + len(pattern)
        return None

    def _compute_attention_diagnostics(
        self,
        attentions: Optional[Tuple],
        sample_index: int,
        input_ids: Optional[torch.Tensor] = None,
        triple: Optional[TripleExample] = None,
    ) -> Dict[str, Any]:
        """Compute attention entropy/score diagnostics from generation attentions."""
        diagnostics = {
            "attention_entropy": None,
            "attention_score": None,
            "attention_context_len": None,
            "attention_num_heads": None,
            # Directed E2 metric: attention toward neighbor entity token span in prompt.
            "neighbor_attention_mass": None,
            "neighbor_attention_lift": None,
            "neighbor_token_span_len": None,
        }
        if attentions is None or len(attentions) == 0:
            return diagnostics

        try:
            step0 = attentions[0]
            if step0 is None or len(step0) == 0:
                return diagnostics
            last_layer_attn = step0[-1]  # [batch, heads, q_len, k_len]
            sample_attn = last_layer_attn[sample_index].float()
            probs = sample_attn.clamp(min=1e-12)

            entropy = -(probs * torch.log(probs)).sum(dim=-1)  # [heads, q_len]
            k_len = int(probs.shape[-1])
            norm_entropy = entropy / math.log(k_len) if k_len > 1 else torch.zeros_like(entropy)
            top1_mass = probs.max(dim=-1).values

            diagnostics["attention_entropy"] = float(norm_entropy.mean().item())
            diagnostics["attention_score"] = float(top1_mass.mean().item())
            diagnostics["attention_context_len"] = k_len
            diagnostics["attention_num_heads"] = int(probs.shape[0])

            # Directed attention quality on neighbor entity token span (head entity in the prompt).
            if input_ids is not None and triple is not None:
                token_ids = [int(x) for x in input_ids.detach().cpu().tolist()]
                head_tokens = self.tokenizer.encode(triple.head, add_special_tokens=False)
                span = self._find_subsequence(token_ids, head_tokens)
                if span is not None:
                    start, end = span
                    # Ensure span is within key length axis.
                    start = max(0, min(start, k_len))
                    end = max(start, min(end, k_len))
                    span_len = max(0, end - start)
                    if span_len > 0 and k_len > 0:
                        # Mean over heads and query positions for first generated token step.
                        mean_attn_over_heads = probs.mean(dim=(0, 1))  # [k_len]
                        mass = float(mean_attn_over_heads[start:end].sum().item())
                        baseline = float(span_len / k_len) if k_len > 0 else None
                        diagnostics["neighbor_attention_mass"] = mass
                        diagnostics["neighbor_token_span_len"] = span_len
                        if baseline and baseline > 0:
                            diagnostics["neighbor_attention_lift"] = mass / baseline
        except Exception as e:
            logger.debug(f"Failed to compute attention diagnostics: {e}")

        return diagnostics
    
    async def _batch_processing_loop(self):
        """后台循环，用于收集任务并进行批量推理"""
        while True:
            await asyncio.sleep(self.batch_timeout)
            
            batch = []
            while not self.batch_queue.empty() and len(batch) < self.batch_size:
                batch.append(self.batch_queue.get_nowait())

            if not batch:
                continue

            templates = [item['template'] for item in batch]
            
            try:
                # 批量编码
                inputs = self.tokenizer(templates, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # 批量推理
                need_attn = any(item.get('enable_attention_dump', False) for item in batch)
                need_margin = any(item.get('enable_margin_dump', False) for item in batch)
                sequences, scores, attentions = self.safe_model_generate(inputs, collect_attentions=need_attn)
                margin_logits_batch = None
                if need_margin:
                    with torch.no_grad():
                        margin_outputs = self.model(**inputs, return_dict=True, use_cache=False)
                        if margin_outputs and getattr(margin_outputs, "logits", None) is not None:
                            margin_logits_batch = margin_outputs.logits[:, -1, :]

                if sequences is None:
                    raise ValueError("Model generation failed for the batch")

                # 分发结果
                for i, item in enumerate(batch):
                    generated_ids = sequences[i][inputs['input_ids'].shape[1]:]
                    generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    
                    # 提取每个样本对应的分数
                    item_scores = [s[i] for s in scores]
                    
                    diagnostics = {}
                    if item.get('enable_margin_dump', False):
                        item_margin_logits = None
                        if margin_logits_batch is not None:
                            item_margin_logits = margin_logits_batch[i]
                        diagnostics.update(self._compute_margin_diagnostics(item.get('triple'), item_margin_logits))
                    if item.get('enable_attention_dump', False):
                        diagnostics.update(
                            self._compute_attention_diagnostics(
                                attentions,
                                i,
                                input_ids=inputs.get('input_ids')[i] if inputs.get('input_ids') is not None else None,
                                triple=item.get('triple'),
                            )
                        )

                    # ✅ 🔬 修改：添加generated_ids到返回值（用于两阶段Tail概率计算）
                    item['result'] = (generated_text, item_scores, generated_ids, diagnostics)
                    item['event'].set()

            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                for item in batch:
                    item['result'] = e
                    item['event'].set()

    async def async_generate_openai_template(self, triple: TripleExample) -> str:
        """异步生成OpenAI模板"""
        if not self.use_openai:
            return self.generate_simple_question_template(triple)
        else:
            return self.generate_simple_question_template(triple)
        
        head, relation, tail = triple.head, triple.relation, triple.tail
        
        # --- 优化的简洁问题生成prompt ---
        system_prompt = """You are an expert at creating extremely simple, direct questions that require ONE-WORD or very short answers. Your questions must be:
1. Under 10 words total
2. Direct and to-the-point
3. Require minimal explanation in the answer
4. Use simple vocabulary
5. Ask for specific facts only

Examples:
- "Where is Paris?" (Answer: France)
- "Who wrote Hamlet?" (Answer: Shakespeare)
- "What is 2+2?" (Answer: 4)

Never ask complex questions requiring long explanations."""

        template_hint = pick_template_hint(relation, head)

        # 推断tail的类型作为hint
        tail_type_hint = infer_tail_type(tail)
        
        user_prompt = f"""Create ONE extremely simple question that gets the answer "{tail}".

Knowledge: {head} {relation} {tail}
Expected Answer: {tail}

Requirements:
1. Question must be under 8 words
2. Should be answerable with just "{tail}"
3. Must ask about {head}
4. Use simplest possible phrasing

Examples:
- "Where is Paris?" (Answer: France)
- "Who created Tesla?" (Answer: Elon Musk)
- "What color is grass?" (Answer: Green)

Your question:"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        question = await self._async_openai_call_with_retry(messages)
        
        if question and self._validate_openai_question_quality(question, triple):
            return f"Question: {question}\nAnswer:"
        else:
            # 降级到简单问题
            logger.warning(f"⚠️ OpenAI问题质量不佳或生成失败，降级到简单问题: {question}")
            return self.generate_simple_question_template(triple)
    
    async def batch_generate_openai_templates(self, triples: List[TripleExample]) -> List[str]:
        """批量异步生成OpenAI模板，以实现高并发"""
        if not self.use_openai:
            return [self.generate_simple_question_template(triple) for triple in triples]

        tasks = [self.async_generate_openai_template(triple) for triple in triples]
        templates = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理可能出现的异常，降级到简单模板
        final_templates = []
        for i, tpl in enumerate(templates):
            if isinstance(tpl, Exception):
                logger.error(f"批量生成模板时出现错误 for {triples[i]}: {tpl}")
                final_templates.append(self.generate_simple_question_template(triples[i]))
            else:
                final_templates.append(tpl)
        
        return final_templates

    def safe_model_generate(
        self,
        inputs: Dict[str, torch.Tensor],
        collect_attentions: bool = False
    ) -> Optional[Tuple[torch.Tensor, List, Optional[Tuple]]]:
        """安全的模型生成，避免dictionary changed size错误"""
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                # 确保所有tensor都在同一设备上
                device_inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, 'to')}
                
                with torch.no_grad():
                    # --- 优化生成参数：直接简洁回答 ---
                    outputs = self.model.generate(
                        **device_inputs,
                        max_new_tokens=10,  # 进一步减少到10个token，强制极简回答
                        temperature=0.1,  # 降低温度确保确定性
                        top_p=0.9,  # 使用top-p采样提高质量
                        repetition_penalty=1.2,  # 提高重复惩罚
                        do_sample=True,
                        return_dict_in_generate=True,
                        output_scores=True,
                        output_attentions=collect_attentions,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )
                    
                    return outputs.sequences, outputs.scores, getattr(outputs, "attentions", None)
                    
            except RuntimeError as e:
                if "dictionary changed size during iteration" in str(e):
                    logger.warning(f"Dictionary iteration error on attempt {attempt + 1}, retrying...")
                    time.sleep(0.1 * (attempt + 1))  # 逐渐增加等待时间
                    continue
                else:
                    logger.error(f"Model generation error: {e}")
                    break
            
            except Exception as e:
                logger.error(f"Unexpected error in model generation: {e}")
                break
        
        logger.error(f"Model generation failed after {max_attempts} attempts")
        return None, None, None
    
    async def async_compute_confidence_improved(
        self,
        triple: TripleExample,
        existing_question: str = None
    ):
        """
        异步计算置信度（客户端部分）。
        将任务提交到批处理队列并等待结果。
        """
        try:
            include_diagnostics = self.enable_margin_dump or self.enable_attention_dump

            def _pack_result(template, extracted, conf, gen_text, final_q, pred_conf, diagnostics):
                if include_diagnostics:
                    return (template, extracted, conf, gen_text, final_q, pred_conf, diagnostics or {})
                return (template, extracted, conf, gen_text, final_q, pred_conf)

            # 步骤1：如果有已存在的question，直接使用；否则生成模板
            if existing_question:
                # 🔬 学术研究优化：根据模板类型选择合适的格式
                if self.config.template_type == "cloze":
                    # Base模型：使用纯续写式模板，不使用few-shot（避免泄露答案）
                    # 将问题转换为陈述句："Where is X?" -> "X is located in"
                    template = self._convert_question_to_cloze(existing_question, triple)
                    final_question = existing_question
                else:
                    # Instruct/Chat models: use the model's own chat template when available.
                    # Raw few-shot text causes repetition loops in modern instruct models
                    # (Gemma4, Qwen3.5, etc.) that expect special role tokens.
                    _tok = self.tokenizer
                    if getattr(_tok, 'chat_template', None):
                        _messages = [{"role": "user", "content": existing_question}]
                        try:
                            template = _tok.apply_chat_template(
                                _messages, add_generation_prompt=True, tokenize=False,
                                enable_thinking=False
                            )
                        except TypeError:
                            template = _tok.apply_chat_template(
                                _messages, add_generation_prompt=True, tokenize=False
                            )
                    else:
                        few_shot_examples = (
                            "Question: Where is the Eiffel Tower located?\\nAnswer: Paris\\n\\n"
                            "Question: Who wrote the novel '1984'?\\nAnswer: George Orwell\\n\\n"
                            "Question: What is the chemical symbol for gold?\\nAnswer: Au\\n\\n"
                        )
                        template = f"{few_shot_examples}Question: {existing_question}\\nAnswer:"
                    final_question = existing_question
            else:
                # 异步生成模板 (依旧独立执行，因为它是I/O密集型)
                if self.config.template_type == "openai_generated":
                    template = await self.async_generate_openai_template(triple)
                else:
                    template = self.generate_template(triple)
                final_question = self._extract_question_from_template(template)

            # --- ROBUSTNESS CHECK ---
            if not template or not template.strip():
                logger.warning(f"Template generation failed for {triple}. Skipping confidence calculation.")
                return _pack_result("", "", None, "", existing_question or "", None, {})

            event = asyncio.Event()
            task_item = {
                'template': template,
                'event': event,
                'triple': triple,
                'enable_margin_dump': self.enable_margin_dump,
                'enable_attention_dump': self.enable_attention_dump,
            }
            await self.batch_queue.put(task_item)

            await event.wait() # 等待批处理完成

            result = task_item.get('result')
            if isinstance(result, Exception):
                raise result
            
            # ✅ 🔬 解包结果：现在包含generated_ids（用于两阶段Tail概率计算）
            diagnostics = {}
            if len(result) == 4:
                generated_text, scores, generated_ids, diagnostics = result
            elif len(result) == 3:
                generated_text, scores, generated_ids = result
            elif len(result) == 2:
                # 向后兼容：旧格式
                generated_text, scores = result
                generated_ids = None
            else:
                logger.warning(f"Unexpected result format: {len(result)} elements")
                return _pack_result(template, "", None, "", final_question, None, {})

            # --- ROBUSTNESS CHECK ---
            if not generated_text or not generated_text.strip():
                logger.warning(f"Model generated an empty response for question based on {triple}. Confidence is None.")
                return _pack_result(template, "", None, "", final_question, None, diagnostics)

            # 步骤3：改进的答案提取（用于fallback）
            if self.config.use_improved_extraction:
                 extracted_answer = self.extract_answer_for_openai(generated_text, triple.tail) if self.config.template_type == "openai_generated" else self.improved_answer_extraction("", generated_text, triple.tail)
            else:
                extracted_answer = generated_text.split('.')[0].strip() if generated_text else ""
            
            # ✅ 🔬 步骤4：改进的置信度计算 - 两阶段Tail概率策略
            # 修复：计算expected tail的概率，而非extracted answer的概率
            
            if generated_ids is not None:
                # ✅ 新逻辑：使用两阶段Tail概率计算（基于大规模测试验证）
                tail_result = self.tail_probability_calculator.compute_tail_probability_two_stage(
                    expected_tail=triple.tail,
                    generated_ids=generated_ids,
                    scores=scores,
                    generated_text=generated_text,
                    question=final_question
                )
                
                # 使用tail的真实概率
                final_confidence = tail_result['tail_probability']
                extracted_answer = tail_result['extracted_answer']
                
                # [MODIFIED] Fallback to generated answer probability if strict tail matching yields near-zero confidence
                # This handles case sensitivity issues (e.g. NASDAQ vs Nasdaq) where model is correct but token matching fails
                if (final_confidence is None or final_confidence < 0.01) and generated_text.strip():
                     # Check if generated text is actually a match (case-insensitive)
                     if triple.tail.lower() in generated_text.lower():
                         # Calculate confidence of the *actual generated text*
                         gen_tokens = generated_ids  # These are the tokens model actually output
                         gen_confidences = []
                         for i, token_id in enumerate(gen_tokens):
                             if i < len(scores):
                                 probs = torch.softmax(scores[i], dim=-1)
                                 gen_confidences.append(probs[token_id].item())
                         
                         if gen_confidences:
                             fallback_conf = self.aggregate_token_probabilities(gen_confidences)
                             # Only use fallback if it's significantly better
                             if fallback_conf is not None and fallback_conf > (final_confidence or 0):
                                 final_confidence = fallback_conf
                                 extracted_answer = generated_text.strip()
                
                # Calculate Prediction Confidence (Confidence of the actual generated text)
                prediction_confidence = 0.0
                if generated_ids is not None and scores:
                    gen_confidences = []
                    for i, token_id in enumerate(generated_ids):
                        if i < len(scores):
                            probs = torch.softmax(scores[i], dim=-1)
                            gen_confidences.append(probs[token_id].item())
                    if gen_confidences:
                        prediction_confidence = self.aggregate_token_probabilities(gen_confidences)

                return _pack_result(
                    template, extracted_answer, final_confidence, generated_text, final_question, prediction_confidence, diagnostics
                )
            else:
                # ❌ 旧逻辑（向后兼容）：计算extracted_answer的概率
                if not extracted_answer:
                    return _pack_result(template, generated_text, None, generated_text, final_question, None, diagnostics)
                
                answer_tokens = self.tokenizer(extracted_answer, return_tensors="pt", add_special_tokens=False)['input_ids'][0]
                if len(answer_tokens) == 0 or len(scores) == 0:
                    return _pack_result(template, extracted_answer, None, generated_text, final_question, None, diagnostics)
                
                answer_confidences = []
                for i, token_id in enumerate(answer_tokens):
                    if i < len(scores):
                        probs = torch.softmax(scores[i], dim=-1)
                        answer_confidences.append(probs[token_id].item())
                
                final_confidence = self.aggregate_token_probabilities(answer_confidences) if answer_confidences else None
                
                # For old logic, prediction confidence is just final_confidence of extracted answer
                return _pack_result(
                    template, extracted_answer, final_confidence, generated_text, final_question, final_confidence, diagnostics
                )
            
        except Exception as e:
            logger.error(f"异步置信度计算失败: {e}")
            template_fallback = template if 'template' in locals() else ""
            question_fallback = final_question if 'final_question' in locals() else (existing_question or "")
            if self.enable_margin_dump or self.enable_attention_dump:
                return template_fallback, "", None, "", question_fallback, None, {}
            return template_fallback, "", None, "", question_fallback, None

    def _convert_question_to_cloze(self, question: str, triple: TripleExample) -> str:
        """
        🔬 将问题转换为续写式(cloze)模板，适合Base模型
        
        例如:
        - "Where is the Eiffel Tower located?" -> "The Eiffel Tower is located in"
        - "Who wrote Hamlet?" -> "Hamlet was written by"
        """
        question_lower = question.lower().strip().rstrip('?').rstrip('.')
        head = triple.head
        relation = triple.relation
        
        # 根据问题类型转换为陈述句
        if question_lower.startswith("where is") or question_lower.startswith("where's"):
            # "Where is X?" -> "X is located in"
            return f"{head} is located in"
        elif question_lower.startswith("where "):
            # "Where does X live?" -> "X lives in"
            return f"{head} is in"
        elif question_lower.startswith("who") or question_lower.startswith("what"):
            # "Who created X?" -> "X was created by"
            # "What is the capital of X?" -> "The capital of X is"
            if "capital of" in question_lower:
                return f"The capital of {head} is"
            elif "created" in question_lower or "made" in question_lower:
                return f"{head} was created by"
            elif "wrote" in question_lower or "written by" in question_lower:
                return f"{head} was written by"
            else:
                # 通用格式：使用关系名
                return f"{head}'s {relation.lower()} is"
        elif question_lower.startswith("when"):
            # "When was X born?" -> "X was born in"
            return f"{head} was born in" if "born" in question_lower else f"{head} occurred in"
        else:
            # 默认：基于关系的通用模板
            relation_lower = relation.lower().replace('_', ' ')
            return f"The {relation_lower} of {head} is"
    
    def _extract_question_from_template(self, template: str) -> str:
        """从模板中提取问题部分"""
        if "Question:" in template:
            question = template.split("Question:")[1].split("Answer:")[0].strip()
            return question
        else:
            # 如果没有明确的Question标记，尝试提取第一行作为问题
            lines = template.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('Context:') and not line.startswith('Answer:'):
                    return line
            return ""
    
    async def batch_compute_confidence(self, triples: List[TripleExample], batch_size: int = 5) -> List[Tuple[str, str, Optional[float], str, str, Optional[float]]]:
        """批量异步计算置信度"""
        results = []
        
        # 分批处理
        for i in range(0, len(triples), batch_size):
            batch = triples[i:i + batch_size]
            
            # 并行处理批次
            batch_tasks = [
                self.async_compute_confidence_improved(triple)
                for triple in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 处理结果
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"批次 {i//batch_size + 1}, 项目 {j + 1} 失败: {result}")
                    results.append(("", "", None, "", "", None))
                else:
                    results.append(result)
            
            # 批次间延迟，避免过载
            if i + batch_size < len(triples):
                await asyncio.sleep(0.5)
        
        return results
    
    async def close(self):
        """清理资源"""
        self.processing_task.cancel()
        try:
            await self.processing_task
        except asyncio.CancelledError:
            pass # 任务取消是正常操作

        if self.session:
            await self.session.close()

# 使用示例函数
async def test_async_confidence_prober():
    """测试异步置信度计算器"""
    from utils import load_llama2_7b
    
    print("🔄 加载模型...")
    model, tokenizer = load_llama2_7b()
    
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
    
    # 创建配置
    config = ImprovedConfig(
        template_type="openai_generated",
        confidence_aggregation="min_confidence",
        temperature=0.1,
        max_tokens=64,
        use_improved_extraction=True
    )
    
    # 创建异步prober
    retry_config = RetryConfig(max_retries=3, base_delay=1.0)
    async_prober = AsyncConfidenceProber(
        model=model,
        tokenizer=tokenizer,
        config=config,
        openai_api_key=openai_key,
        retry_config=retry_config
    )
    
    # 测试数据
    test_triples = [
        TripleExample("Language Policy", "is influenced by findings in", "linguistics studies"),
        TripleExample("Cognitive Linguistics", "is a perspective within", "linguistics studies"),
        TripleExample("Paris", "capital_of", "France"),
        TripleExample("Einstein", "born_in", "Germany"),
        TripleExample("Shakespeare", "wrote", "Hamlet")
    ]
    
    print(f"🧪 测试异步置信度计算: {len(test_triples)} 个三元组")
    
    # 批量异步处理
    start_time = time.time()
    results = await async_prober.batch_compute_confidence(test_triples, batch_size=3)
    end_time = time.time()
    
    # 统计结果
    success_count = sum(1 for _, _, conf in results if conf is not None)
    
    print(f"\n📊 异步测试结果:")
    print(f"✅ 成功率: {success_count}/{len(test_triples)} ({success_count/len(test_triples)*100:.1f}%)")
    print(f"⏱️ 处理时间: {end_time - start_time:.2f}秒")
    print(f"🚀 平均速度: {len(test_triples)/(end_time - start_time):.2f} 三元组/秒")
    
    # 显示详细结果
    for i, (triple, (template, answer, conf)) in enumerate(zip(test_triples, results)):
        print(f"\n🔍 测试 {i+1}: ({triple.head}, {triple.relation}, {triple.tail})")
        if conf is not None:
            print(f"  ✅ 成功 - 置信度: {conf:.4f}")
            print(f"  📝 答案: {answer[:50]}...")
        else:
            print(f"  ❌ 失败")
    
    # 清理资源
    await async_prober.close()
    
    return success_count / len(test_triples)

if __name__ == "__main__":
    import torch
    asyncio.run(test_async_confidence_prober())
