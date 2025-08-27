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
from improved_confidence_probing import ImprovedConfidenceProber, ImprovedConfig, TripleExample

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    def __init__(self, model, tokenizer, config: ImprovedConfig, openai_api_key=None, retry_config=None):
        super().__init__(model, tokenizer, config, openai_api_key)
        self.retry_config = retry_config or RetryConfig()
        self.session = None
        self._setup_session()

        # --- 新增：用于动态批处理的组件 ---
        self.batch_queue = asyncio.Queue()
        self.processing_task = asyncio.create_task(self._batch_processing_loop())
        self.batch_size = 32  # 可配置的批处理大小
        self.batch_timeout = 0.05  # 50ms, 等待更多任务的最长时间
        # --- 结束新增 ---
    
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
                sequences, scores = self.safe_model_generate(inputs)

                if sequences is None:
                    raise ValueError("Model generation failed for the batch")

                # 分发结果
                for i, item in enumerate(batch):
                    generated_ids = sequences[i][inputs['input_ids'].shape[1]:]
                    generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    
                    # 提取每个样本对应的分数
                    item_scores = [s[i] for s in scores]
                    
                    item['result'] = (generated_text, item_scores)
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

    def safe_model_generate(self, inputs: Dict[str, torch.Tensor]) -> Optional[Tuple[torch.Tensor, List]]:
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
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )
                    
                    return outputs.sequences, outputs.scores
                    
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
        return None, None
    
    async def async_compute_confidence_improved(self, triple: TripleExample, existing_question: str = None) -> Tuple[str, str, Optional[float], str, str]:
        """
        异步计算置信度（客户端部分）。
        将任务提交到批处理队列并等待结果。
        """
        try:
            # 步骤1：如果有已存在的question，直接使用；否则生成模板
            if existing_question:
                # --- NEW: 使用Few-Shot Prompt来引导模型进行简洁回答 ---
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
                return "", "", None, "", existing_question or ""

            event = asyncio.Event()
            task_item = {'template': template, 'event': event}
            await self.batch_queue.put(task_item)

            await event.wait() # 等待批处理完成

            result = task_item.get('result')
            if isinstance(result, Exception):
                raise result
            
            generated_text, scores = result

            # --- ROBUSTNESS CHECK ---
            if not generated_text or not generated_text.strip():
                logger.warning(f"Model generated an empty response for question based on {triple}. Confidence is None.")
                return template, "", None, "", final_question

            # 步骤3：改进的答案提取 (与之前相同)
            if self.config.use_improved_extraction:
                 extracted_answer = self.extract_answer_for_openai(generated_text, triple.tail) if self.config.template_type == "openai_generated" else self.improved_answer_extraction("", generated_text, triple.tail)
            else:
                extracted_answer = generated_text.split('.')[0].strip() if generated_text else ""
            
            if not extracted_answer:
                return template, generated_text, None, generated_text, final_question
            
            # 步骤4：安全的置信度计算 (使用批处理结果)
            answer_tokens = self.tokenizer(extracted_answer, return_tensors="pt", add_special_tokens=False)['input_ids'][0]
            if len(answer_tokens) == 0 or len(scores) == 0:
                return template, extracted_answer, None, generated_text, final_question
            
            answer_confidences = []
            for i, token_id in enumerate(answer_tokens):
                if i < len(scores):
                    probs = torch.softmax(scores[i], dim=-1) # scores[i]已经是单个样本的分数
                    answer_confidences.append(probs[token_id].item())
            
            final_confidence = self.aggregate_token_probabilities(answer_confidences) if answer_confidences else None
            
            return template, extracted_answer, final_confidence, generated_text, final_question
            
        except Exception as e:
            logger.error(f"异步置信度计算失败: {e}")
            template_fallback = template if 'template' in locals() else ""
            question_fallback = final_question if 'final_question' in locals() else (existing_question or "")
            return template_fallback, "", None, "", question_fallback

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
    
    async def batch_compute_confidence(self, triples: List[TripleExample], batch_size: int = 5) -> List[Tuple[str, str, Optional[float]]]:
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
                    results.append(("", "", None))
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
