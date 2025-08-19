import os
import json
import asyncio
from statistics import mean, mode
from typing import Dict, Optional, List
from openai import AsyncOpenAI


def load_key_from_file(filepath: str) -> Optional[str]:
    """从文件中读取 API Key"""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


class FairModelEvaluator:
    """
    公平模型评估系统 - 两个线上模型互相判断，无ground truth依赖
    专门针对知识图谱三元组关系评估优化
    """

    def __init__(self, judge_configs: List[Dict] = None, cache_path: str = ".fair_evaluation_cache.json"):
        """
        初始化公平评估器
        Args:
            judge_configs: 裁判配置列表（建议使用两个不同的模型）
            cache_path: 缓存文件路径
        """
        self.cache_path = cache_path
        self.cache = self._load_cache()

        if judge_configs is None:
            judge_configs = []

            # 默认：OpenAI GPT-4o-mini
            gpt_key = os.environ.get("OPENAI_API_KEY") or load_key_from_file("keys/openai_key.txt")
            if gpt_key:
                judge_configs.append({
                    "model_name": "gpt-4o-mini",
                    "api_base": "https://api.openai.com/v1",
                    "api_key": gpt_key,
                    "temperature": 0.0
                })

            # 默认：火山 Ark DeepSeek v3
            ark_key = os.environ.get("ARK_API_KEY") or load_key_from_file("keys/ark_key.txt")
            if ark_key:
                judge_configs.append({
                    "model_name": "ep-20250118122533-wkp8h",
                    "api_base": "https://ark.cn-beijing.volces.com/api/v3",
                    "api_key": ark_key,
                    "temperature": 0.0
                })

            if not judge_configs:
                raise ValueError("❌ 未找到任何 API Key")

        # 确保至少有两个不同的模型进行评估
        if len(judge_configs) < 2:
            print("⚠️ 建议使用至少两个不同的模型进行公平评估")
        
        self.judge_configs = judge_configs
        self.judges = self._initialize_judges()

        print(f"🔧 初始化了 {len(self.judges)} 个公平评估模型:")
        for i, config in enumerate(self.judge_configs):
            judge_type = "本地模型" if "127.0.0.1" in config.get("api_base", "") else "云端API"
            print(f"  评估器 {i+1}: {config['model_name']} ({judge_type})")

    def _initialize_judges(self) -> List[AsyncOpenAI]:
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
        """单个评估器异步请求"""
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

            if "score" in result and "category" in result and "explanation" in result:
                return {
                    "judge_id": idx,
                    "judge_name": config["model_name"],
                    "score": result["score"],
                    "category": result["category"],
                    "explanation": result["explanation"],
                    "confidence": result.get("confidence", 0.8)  # 评估器对自己判断的信心
                }
            else:
                return {"error": f"无效格式: {config['model_name']}"}
        except Exception as e:
            return {"error": f"{config['model_name']} 失败: {str(e)}"}

    async def evaluate_model_output(self, question: str, model_answer: str, triplet_context: str = "") -> Optional[Dict]:
        """
        公平评估模型输出质量 - 无ground truth依赖
        
        Args:
            question: 问题
            model_answer: 模型回答
            triplet_context: 三元组上下文信息（可选）
        """
        cache_key = f"fair_evaluation:{len(self.judges)}|{question}|{model_answer}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 专门针对无ground truth场景的system prompt
        system_prompt = """You are an expert evaluator specializing in knowledge graph relationship assessment. Your task is to evaluate the quality of a model's answer to a relationship question WITHOUT relying on a ground truth answer.

EVALUATION CONTEXT:
- This is a knowledge graph triplet evaluation where we assess how well a model explains relationships between entities
- You must judge the answer quality based on its own merits, not comparison to a "correct" answer
- Focus on the logical coherence, factual accuracy, and completeness of the relationship explanation

QUALITY ASSESSMENT CRITERIA (0-100):
- 90-100: Excellent - Clear, accurate, and comprehensive relationship explanation with relevant context
- 80-89: Very_Good - Well-structured explanation that clearly establishes the relationship
- 70-79: Good - Adequate explanation of the relationship with minor issues
- 60-69: Fair - Basic relationship explanation but may be unclear or incomplete
- 50-59: Acceptable - Mentions the relationship but lacks clarity or detail
- 40-49: Poor - Vague or confusing explanation of the relationship
- 30-39: Very_Poor - Minimal relevance to the relationship question
- 20-29: Barely_Relevant - Mostly irrelevant content with few related points
- 10-19: Irrelevant - Content that doesn't address the relationship
- 0-9: Completely_Wrong - Completely unrelated or factually incorrect

EVALUATION FOCUS:
1. Does the answer directly address the relationship question?
2. Is the relationship explanation clear and logical?
3. Are the facts presented accurate and relevant?
4. Does the answer provide sufficient context without being overly verbose?
5. Is the explanation coherent and well-structured?

Follow this JSON format:
{
  "score": <integer_from_0_to_100>,
  "category": "<one_of: Excellent, Very_Good, Good, Fair, Acceptable, Poor, Very_Poor, Barely_Relevant, Irrelevant, Completely_Wrong>",
  "explanation": "<detailed_explanation_of_quality_assessment>",
  "confidence": <float_from_0_to_1_indicating_judge_confidence>
}
"""

        # 优化的user prompt - 专注于质量评估
        user_prompt = f"""Please evaluate the quality of this model's answer to a knowledge graph relationship question.

QUESTION: "{question}"

MODEL ANSWER: "{model_answer}"

{f"TRIPLET CONTEXT: {triplet_context}" if triplet_context else ""}

EVALUATION TASKS:
1. How well does the answer explain the relationship between the entities?
2. Is the explanation clear, accurate, and logically coherent?
3. Does the answer provide appropriate context without being overly verbose?
4. How comprehensive and informative is the relationship explanation?

Please provide a fair and objective assessment of the answer quality based on its own merits.
"""

        tasks = [
            self._ask_judge(judge_client, config, system_prompt, user_prompt, i)
            for i, (judge_client, config) in enumerate(zip(self.judges, self.judge_configs))
        ]
        judge_results = await asyncio.gather(*tasks)

        valid_results = [r for r in judge_results if "score" in r]

        if not valid_results:
            print("❌ 所有评估器评估失败")
            return None

        aggregated_result = self._aggregate_judge_results(valid_results)

        self.cache[cache_key] = aggregated_result
        self._save_cache()

        return aggregated_result

    def _aggregate_judge_results(self, judge_results: List[Dict]) -> Dict:
        scores = [r["score"] for r in judge_results]
        categories = [r["category"] for r in judge_results]
        confidences = [r.get("confidence", 0.8) for r in judge_results]

        # 加权平均分数（基于评估器信心度）
        weighted_scores = [score * conf for score, conf in zip(scores, confidences)]
        avg_score = round(sum(weighted_scores) / sum(confidences))
        
        try:
            majority_category = mode(categories)
        except:
            majority_category = judge_results[scores.index(max(scores))]["category"]

        explanation = f"公平评估: 分数范围 {min(scores)}-{max(scores)}, 加权平均分 {avg_score}. "
        if len(set(categories)) == 1:
            explanation += f"所有评估器一致认为: {majority_category}."
        else:
            counts = {c: categories.count(c) for c in set(categories)}
            explanation += f"类别分布: {counts}, 多数选择: {majority_category}."

        return {
            "score": avg_score,
            "category": majority_category,
            "explanation": explanation,
            "detailed_results": judge_results,
            "metadata": {
                "total_judges": len(judge_results),
                "successful_evaluations": len(judge_results),
                "score_variance": max(scores) - min(scores) if len(scores) > 1 else 0,
                "category_consensus": len(set(categories)) == 1,
                "average_confidence": sum(confidences) / len(confidences),
                "evaluation_method": "fair_quality_assessment_no_ground_truth"
            }
        }


# 使用示例
async def main():
    evaluator = FairModelEvaluator()
    result = await evaluator.evaluate_model_output(
        question="What is the relationship between The Republic and justice?",
        model_answer="The Republic explores the concept of justice in the context of the American criminal justice system. The series examines the ways in which the system can fail to deliver justice, particularly for marginalized communities, and the ways in which individuals and organizations are working to reform and improve the system.",
        triplet_context="The Republic explores the concept of justice"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())