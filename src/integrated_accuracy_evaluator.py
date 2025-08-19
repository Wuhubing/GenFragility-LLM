#!/usr/bin/env python3
"""
集成准确率评估器 - 利用置信度计算过程中生成的问题和答案直接计算准确率
避免重复API调用，提高效率
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from openai import AsyncOpenAI
import os

logger = logging.getLogger(__name__)

@dataclass
class AccuracyResult:
    """准确率评估结果"""
    is_correct: bool
    confidence: float
    explanation: str
    raw_response: str
    evaluator_name: str

class IntegratedAccuracyEvaluator:
    """
    集成准确率评估器 - 双评估器架构 (OpenAI + DeepSeek)
    直接利用置信度计算过程中已生成的问题和模型回答来计算准确率
    参考 FairModelEvaluator 的设计模式
    """
    
    def __init__(self, judge_configs: List[Dict] = None, cache_path: str = ".accuracy_evaluation_cache.json"):
        """
        初始化准确率评估器
        Args:
            judge_configs: 裁判配置列表（建议使用两个不同的模型）
            cache_path: 缓存文件路径
        """
        self.cache_path = cache_path
        self.cache = self._load_cache()

        if judge_configs is None:
            judge_configs = []

            # 默认：OpenAI GPT-4o-mini
            gpt_key = os.environ.get("OPENAI_API_KEY") or self._load_key_from_file("keys/openai_key.txt")
            if gpt_key:
                judge_configs.append({
                    "model_name": "gpt-4o-mini",
                    "api_base": "https://api.openai.com/v1",
                    "api_key": gpt_key,
                    "temperature": 0.0
                })

            # 默认：火山 Ark DeepSeek v3
            ark_key = os.environ.get("ARK_API_KEY") or self._load_key_from_file("keys/ark_key.txt")
            if ark_key:
                judge_configs.append({
                    "model_name": "ep-20250118122533-wkp8h",
                    "api_base": "https://ark.cn-beijing.volces.com/api/v3",
                    "api_key": ark_key,
                    "temperature": 0.0
                })

            if not judge_configs:
                raise ValueError("❌ 未找到任何 API Key，无法进行准确率评估")

        # 确保至少有一个评估器
        if len(judge_configs) < 1:
            raise ValueError("❌ 需要至少一个评估器进行准确率评估")
        
        self.judge_configs = judge_configs
        self.judges = self._initialize_judges()

        print(f"🎯 初始化了 {len(self.judges)} 个准确率评估器:")
        for i, config in enumerate(self.judge_configs):
            judge_type = "本地模型" if "127.0.0.1" in config.get("api_base", "") else "云端API"
            print(f"  准确率评估器 {i+1}: {config['model_name']} ({judge_type})")
    
    def _load_key_from_file(self, filepath: str) -> Optional[str]:
        """从文件中读取 API Key"""
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read().strip()
        return None

    def _initialize_judges(self) -> List:
        """初始化异步评估客户端"""
        judges = []
        for config in self.judge_configs:
            client = AsyncOpenAI(
                api_key=config.get("api_key"),
                base_url=config.get("api_base"),
                timeout=30.0
            )
            judges.append(client)
        return judges

    def _load_cache(self) -> Dict:
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    async def _ask_judge(self, judge_client, config, system_prompt, user_prompt, idx: int):
        """单个准确率评估器异步请求"""
        try:
            response = await judge_client.chat.completions.create(
                model=config["model_name"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=config.get("temperature", 0.0),
                max_tokens=512,
            )
            result = json.loads(response.choices[0].message.content)

            if "is_correct" in result and "explanation" in result:
                return {
                    "judge_id": idx,
                    "judge_name": config["model_name"],
                    "is_correct": result["is_correct"],
                    "explanation": result["explanation"],
                    "confidence": result.get("confidence", 0.8)  # 评估器对自己判断的信心
                }
            else:
                return {"error": f"无效格式: {config['model_name']}"}
        except Exception as e:
            return {"error": f"{config['model_name']} 失败: {str(e)}"}
    
    async def evaluate_accuracy(
        self, 
        question: str, 
        model_answer: str, 
        expected_answer: str,
        triplet_context: str = ""
    ) -> Optional[Dict]:
        """
        双评估器准确率评估 - 参考 FairModelEvaluator 的设计
        
        Args:
            question: 问题
            model_answer: 模型的回答
            expected_answer: 期望的正确答案
            triplet_context: 三元组上下文信息
        
        Returns:
            Dict: 包含聚合结果的字典
        """
        cache_key = f"accuracy_evaluation:{len(self.judges)}|{question}|{model_answer}|{expected_answer}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 专门针对准确率评估的 system prompt
        system_prompt = """You are an expert evaluator for knowledge graph question-answering accuracy assessment.

Your task is to determine if a model's answer to a question is CORRECT based on the expected answer.

EVALUATION CRITERIA:
1. CORRECT (True): The model's answer contains or correctly identifies the expected answer
   - Direct match: Model answer explicitly contains the expected answer
   - Semantic match: Model answer conveys the same meaning as expected answer
   - Partial match with essential information: Key information is present even if worded differently
   - Contextual match: Answer is factually correct even if more detailed than expected

2. INCORRECT (False): The model's answer does not match the expected answer
   - Wrong information: Model provides factually incorrect information
   - Missing key information: Essential information is completely absent
   - Completely irrelevant: Answer doesn't address the question at all

IMPORTANT EVALUATION GUIDELINES:
- Be LENIENT and focus on factual correctness over exact wording
- If the model provides MORE information that INCLUDES the expected answer, mark as CORRECT
- Consider semantic equivalence (e.g., "France" vs "French Republic", "beautiful" vs "natural beauty")
- Accept contextual answers that imply the expected answer
- Only mark as INCORRECT if the answer is factually wrong or completely missing the expected information
- Give benefit of the doubt when the model provides relevant context around the expected answer

EXAMPLES:
- Expected: "Paris", Model: "The capital is Paris" → CORRECT
- Expected: "natural beauty", Model: "beautiful beaches with white sand" → CORRECT  
- Expected: "France", Model: "located in the French Republic" → CORRECT
- Expected: "Shakespeare", Model: "The author was William Shakespeare" → CORRECT

Please respond in JSON format:
{
  "is_correct": <boolean>,
  "confidence": <float_0_to_1>,
  "explanation": "<detailed_explanation_of_evaluation>"
}"""

        user_prompt = f"""Please evaluate if the model's answer is correct.

QUESTION: "{question}"

EXPECTED ANSWER: "{expected_answer}"

MODEL'S ANSWER: "{model_answer}"

{f"CONTEXT: {triplet_context}" if triplet_context else ""}

EVALUATION TASK:
Determine if the model's answer is factually correct and contains the expected information.
Consider semantic equivalence and context when making your decision.

Please provide your evaluation:"""

        # 异步调用所有评估器
        tasks = [
            self._ask_judge(judge_client, config, system_prompt, user_prompt, i)
            for i, (judge_client, config) in enumerate(zip(self.judges, self.judge_configs))
        ]
        judge_results = await asyncio.gather(*tasks)

        valid_results = [r for r in judge_results if "is_correct" in r]

        if not valid_results:
            print("❌ 所有准确率评估器评估失败")
            return None

        aggregated_result = self._aggregate_accuracy_results(valid_results)

        self.cache[cache_key] = aggregated_result
        self._save_cache()

        return aggregated_result

    def _aggregate_accuracy_results(self, judge_results: List[Dict]) -> Dict:
        """聚合多个评估器的准确率判断结果"""
        from statistics import mode
        
        is_correct_list = [r["is_correct"] for r in judge_results]
        confidences = [r.get("confidence", 0.8) for r in judge_results]
        explanations = [r["explanation"] for r in judge_results]

        # 使用多数投票决定最终结果
        try:
            final_is_correct = mode(is_correct_list)
        except:
            # 如果没有众数，使用加权投票
            weighted_votes = sum(is_correct * conf for is_correct, conf in zip(is_correct_list, confidences))
            total_weight = sum(confidences)
            final_is_correct = weighted_votes / total_weight > 0.5

        # 计算平均信心度
        avg_confidence = sum(confidences) / len(confidences)

        # 生成聚合解释
        correct_count = sum(is_correct_list)
        total_count = len(is_correct_list)
        
        explanation = f"准确率评估: {correct_count}/{total_count} 个评估器认为正确, 最终判断: {'正确' if final_is_correct else '错误'}. "
        if len(set(is_correct_list)) == 1:
            explanation += f"所有评估器一致认为: {'正确' if final_is_correct else '错误'}."
        else:
            explanation += f"评估器意见分歧，基于多数投票和信心度加权决定。"

        return {
            "is_correct": final_is_correct,
            "confidence": avg_confidence,
            "explanation": explanation,
            "detailed_results": judge_results,
            "metadata": {
                "total_judges": len(judge_results),
                "successful_evaluations": len(judge_results),
                "consensus": len(set(is_correct_list)) == 1,
                "average_confidence": avg_confidence,
                "evaluation_method": "dual_judge_accuracy_assessment"
            }
        }
    
    async def batch_evaluate_accuracy(
        self, 
        evaluation_data: List[Dict]
    ) -> List[Dict]:
        """
        批量评估准确率
        
        Args:
            evaluation_data: 包含question, model_answer, expected_answer的字典列表
        
        Returns:
            List[Dict]: 准确率评估结果列表
        """
        tasks = []
        for data in evaluation_data:
            task = self.evaluate_accuracy(
                question=data['question'],
                model_answer=data['model_answer'],
                expected_answer=data['expected_answer'],
                triplet_context=data.get('triplet_context', '')
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"批量评估项目 {i} 失败: {result}")
                processed_results.append({
                    "is_correct": False,
                    "confidence": 0.0,
                    "explanation": f"批量评估失败: {str(result)}",
                    "detailed_results": [],
                    "metadata": {"evaluation_method": "dual_judge_accuracy_assessment"}
                })
            else:
                processed_results.append(result)
        
        return processed_results

    async def close(self):
        """清理资源"""
        # 关闭所有judge客户端
        for judge_client in self.judges:
            if hasattr(judge_client, 'close'):
                await judge_client.close()

# 测试函数
async def test_integrated_accuracy_evaluator():
    """测试双评估器架构的集成准确率评估器"""
    evaluator = IntegratedAccuracyEvaluator()
    
    test_cases = [
        {
            "question": "What is the capital of France?",
            "model_answer": "The capital of France is Paris, which is also the largest city in the country.",
            "expected_answer": "Paris",
            "triplet_context": "France capital_of Paris"
        },
        {
            "question": "Where was Einstein born?",
            "model_answer": "Albert Einstein was born in Germany in the city of Ulm.",
            "expected_answer": "Germany", 
            "triplet_context": "Einstein born_in Germany"
        },
        {
            "question": "What did Shakespeare write?",
            "model_answer": "Shakespeare wrote many plays including Romeo and Juliet, but not Hamlet.",
            "expected_answer": "Hamlet",
            "triplet_context": "Shakespeare wrote Hamlet"
        }
    ]
    
    print("🧪 测试双评估器架构的集成准确率评估器:")
    results = await evaluator.batch_evaluate_accuracy(test_cases)
    
    correct_count = 0
    for i, (test_case, result) in enumerate(zip(test_cases, results)):
        print(f"\n📋 测试案例 {i+1}:")
        print(f"  问题: {test_case['question']}")
        print(f"  期望答案: {test_case['expected_answer']}")
        print(f"  模型回答: {test_case['model_answer'][:50]}...")
        print(f"  ✅ 准确率评估: {'正确' if result['is_correct'] else '错误'}")
        print(f"  🎯 评估信心: {result['confidence']:.2f}")
        print(f"  💭 评估解释: {result['explanation']}")
        
        # 显示详细的评估器结果
        if 'detailed_results' in result and result['detailed_results']:
            print(f"  📊 详细评估器结果:")
            for j, judge_result in enumerate(result['detailed_results']):
                judge_name = judge_result['judge_name']
                judge_correct = judge_result['is_correct']
                judge_conf = judge_result['confidence']
                print(f"    评估器 {j+1} ({judge_name}): {'正确' if judge_correct else '错误'} (信心: {judge_conf:.2f})")
        
        if result['is_correct']:
            correct_count += 1
    
    accuracy_rate = correct_count / len(test_cases) * 100
    print(f"\n📊 双评估器总体准确率: {correct_count}/{len(test_cases)} ({accuracy_rate:.1f}%)")
    
    await evaluator.close()
    return accuracy_rate

if __name__ == "__main__":
    asyncio.run(test_integrated_accuracy_evaluator())
