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
        
        # 优化的prompt
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
                device_inputs = {}
                for k, v in inputs.items():
                    if hasattr(v, 'to'):
                        device_inputs[k] = v.to(self.device)
                    else:
                        device_inputs[k] = v
                
                with torch.no_grad():
                    # 使用更保守的生成参数
                    outputs = self.model.generate(
                        **device_inputs,
                        max_new_tokens=min(self.config.max_tokens, 128),
                        temperature=max(self.config.temperature, 0.1),
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
    
    async def async_compute_confidence_improved(self, triple: TripleExample) -> Tuple[str, str, Optional[float], str, str]:
        """
        异步计算置信度（客户端部分）。
        将任务提交到批处理队列并等待结果。
        """
        try:
            # 步骤1：异步生成模板 (依旧独立执行，因为它是I/O密集型)
            if self.config.template_type == "openai_generated":
                template = await self.async_generate_openai_template(triple)
            else:
                template = self.generate_template(triple)

            event = asyncio.Event()
            task_item = {'template': template, 'event': event}
            await self.batch_queue.put(task_item)

            await event.wait() # 等待批处理完成

            result = task_item.get('result')
            if isinstance(result, Exception):
                raise result
            
            generated_text, scores = result

            # 步骤3：改进的答案提取 (与之前相同)
            if self.config.use_improved_extraction:
                 extracted_answer = self.extract_answer_for_openai(generated_text, triple.tail) if self.config.template_type == "openai_generated" else self.improved_answer_extraction("", generated_text, triple.tail)
            else:
                extracted_answer = generated_text.split('.')[0].strip() if generated_text else ""
            
            if not extracted_answer:
                return template, generated_text, None, generated_text, self._extract_question_from_template(template)
            
            # 步骤4：安全的置信度计算 (使用批处理结果)
            answer_tokens = self.tokenizer(extracted_answer, return_tensors="pt", add_special_tokens=False)['input_ids'][0]
            if len(answer_tokens) == 0 or len(scores) == 0:
                return template, extracted_answer, None, generated_text, self._extract_question_from_template(template)
            
            answer_confidences = []
            for i, token_id in enumerate(answer_tokens):
                if i < len(scores):
                    probs = torch.softmax(scores[i], dim=-1) # scores[i]已经是单个样本的分数
                    answer_confidences.append(probs[token_id].item())
            
            final_confidence = self.aggregate_token_probabilities(answer_confidences) if answer_confidences else None
            
            return template, extracted_answer, final_confidence, generated_text, self._extract_question_from_template(template)
            
        except Exception as e:
            logger.error(f"异步置信度计算失败: {e}")
            template_fallback = template if 'template' in locals() else ""
            question_fallback = self._extract_question_from_template(template_fallback) if template_fallback else ""
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
