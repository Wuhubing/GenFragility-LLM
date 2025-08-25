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
        self.cache_lock = asyncio.Lock() # 添加异步锁

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
            # --- FIX: Explicitly pass the API key from the correct environment variable ---
            api_key = os.environ.get(config['api_key_env'])
            if not api_key:
                raise ValueError(f"API key environment variable '{config['api_key_env']}' not set for judge '{config['model_name']}'.")

            client = AsyncOpenAI(
                api_key=api_key,
                base_url=config.get("api_base"),
                timeout=30.0
            )
            judges.append(client)
        return judges

    def _load_cache(self) -> Dict:
        """加载缓存，增加对损坏文件的鲁棒性"""
        if not os.path.exists(self.cache_path):
            # 如果文件不存在，创建一个空的json文件
            try:
                with open(self.cache_path, 'w', encoding='utf-8') as f:
                    f.write('{}')
                return {}
            except IOError as e:
                print(f"❌ 创建缓存文件失败: {e}")
                return {}

        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                # 检查文件是否为空
                content = f.read()
                if not content.strip():
                    print("⚠️ 缓存文件为空，返回空字典")
                    return {}
                return json.loads(content)
        except json.JSONDecodeError:
            print(f"⚠️ 缓存文件 '{self.cache_path}' 已损坏，将重新创建。")
            # 删除损坏的文件并创建一个新的
            try:
                os.remove(self.cache_path)
                with open(self.cache_path, 'w', encoding='utf-8') as f:
                    f.write('{}')
                return {}
            except Exception as e:
                print(f"❌ 重新创建缓存文件失败: {e}")
                return {} # 即使失败也返回空字典，避免崩溃
        except IOError as e:
            print(f"❌ 读取缓存文件失败: {e}")
            return {}

    async def _save_cache(self):
        """异步保存缓存，带锁"""
        async with self.cache_lock:
            try:
                # 写入临时文件，然后重命名，实现原子操作
                temp_file = self.cache_path + ".tmp"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, indent=2, ensure_ascii=False)
                os.rename(temp_file, self.cache_path)
            except Exception as e:
                print(f"❌ 缓存保存失败: {e}")

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

            if "reasoning" in result and "accuracy_score" in result and "relevance_score" in result and "clarity_score" in result and "final_score" in result and "final_category" in result and "judge_confidence" in result:
                return {
                    "judge_id": idx,
                    "judge_name": config["model_name"],
                    "reasoning": result["reasoning"],
                    "accuracy_score": result["accuracy_score"],
                    "relevance_score": result["relevance_score"],
                    "clarity_score": result["clarity_score"],
                    "final_score": result["final_score"],
                    "final_category": result["final_category"],
                    "judge_confidence": result["judge_confidence"]
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

        # --- NEW PROMPT FOR INTEGRATED ACCURACY & QUALITY ---
        system_prompt = """You are a highly analytical AI evaluator. Your task is to provide a multi-dimensional assessment of a model's answer based on its accuracy, relevance, and clarity, then aggregate these into a final score.

**EVALUATION DIMENSIONS:**

1.  **Accuracy (0-100):** How factually correct is the answer when compared *strictly* against the provided "Ground Truth Triplet"?
    *   100: Perfectly matches the triplet's meaning.
    *   75: Mostly correct but with minor inaccuracies or omissions.
    *   50: Contains significant factual errors but captures some essence of the truth.
    *   25: Largely incorrect, possibly confusing related concepts.
    *   0: Completely wrong or contradicts the triplet.

2.  **Relevance (0-100):** How well does the answer address the *specific question* asked, without including unnecessary information?
    *   100: A direct and concise answer to the question.
    *   75: Answers the question but includes some minor, tangentially related information.
    *   50: The correct answer is present but buried in a lot of irrelevant details (hallucination, verbosity).
    *   25: Barely addresses the question.
    *   0: Does not answer the question at all.

3.  **Clarity (0-100):** How clear, fluent, and grammatically correct is the answer?
    *   100: Perfectly fluent, clear, and easy to understand.
    *   75: Generally clear but with minor grammatical errors or awkward phrasing.
    *   50: Understandable, but contains significant grammatical errors or is poorly structured.
    *   25: Very difficult to understand.
    *   0: Incoherent nonsense.

**FINAL SCORE CALCULATION:**
The final score is a weighted average: `(Accuracy * 0.5) + (Relevance * 0.3) + (Clarity * 0.2)`.

**OUTPUT FORMAT (JSON ONLY):**
You MUST respond in this exact JSON format:
{
  "reasoning": "<brief_step_by_step_reasoning_for_each_dimension>",
  "accuracy_score": <int>,
  "relevance_score": <int>,
  "clarity_score": <int>,
  "final_score": <int_weighted_average>,
  "final_category": "<Excellent/Good/Fair/Poor/etc.>",
  "judge_confidence": <float_0_to_1>
}"""

        # Optimized user prompt for the new multi-dimensional evaluation
        user_prompt = f"""Please provide a multi-dimensional evaluation for the model's answer.

**Ground Truth Triplet:** `{triplet_context}`

**Question Asked:** `{question}`

**Model's Answer:** `{model_answer}`

**Instructions:**
1.  Assess the `accuracy_score` by comparing the "Model's Answer" strictly against the "Ground Truth Triplet".
2.  Assess the `relevance_score` by evaluating how directly the answer addresses the "Question Asked".
3.  Assess the `clarity_score` based on the language quality of the answer.
4.  Calculate the `final_score` using the weighted average: (Accuracy * 0.5) + (Relevance * 0.3) + (Clarity * 0.2).
5.  Provide your step-by-step `reasoning`.
6.  Determine the `final_category` based on the `final_score`.
7.  Provide your `judge_confidence` in this overall assessment.
"""

        tasks = [
            self._ask_judge(judge_client, config, system_prompt, user_prompt, i)
            for i, (judge_client, config) in enumerate(zip(self.judges, self.judge_configs))
        ]
        judge_results = await asyncio.gather(*tasks)

        valid_results = [r for r in judge_results if "reasoning" in r]

        if not valid_results:
            print("❌ 所有评估器评估失败")
            return None

        aggregated_result = self._aggregate_judge_results(valid_results)

        self.cache[cache_key] = aggregated_result
        await self._save_cache()

        return aggregated_result

    def _aggregate_judge_results(self, judge_results: List[Dict]) -> Dict:
        # --- UPDATED AGGREGATION LOGIC ---
        final_scores = [r.get("final_score", 0) for r in judge_results]
        categories = [r.get("final_category", "Error") for r in judge_results]
        confidences = [r.get("judge_confidence", 0.8) for r in judge_results]

        # Weighted average of final scores based on judge confidence
        weighted_scores = [score * conf for score, conf in zip(final_scores, confidences)]
        avg_score = round(sum(weighted_scores) / sum(confidences)) if sum(confidences) > 0 else 0
        
        try:
            majority_category = mode(categories)
        except:
            majority_category = judge_results[final_scores.index(max(final_scores))]["final_category"]

        explanation = f"Integrated Assessment: Final score range {min(final_scores)}-{max(final_scores)}, confidence-weighted avg {avg_score}. "
        if len(set(categories)) == 1:
            explanation += f"All judges agree on category: {majority_category}."
        else:
            counts = {c: categories.count(c) for c in set(categories)}
            explanation += f"Category distribution: {counts}, majority: {majority_category}."

        # Include dimensional scores in the final output
        dimensional_scores = {}
        for i, r in enumerate(judge_results):
            dimensional_scores[f"judge_{i+1}_accuracy"] = r.get("accuracy_score")
            dimensional_scores[f"judge_{i+1}_relevance"] = r.get("relevance_score")
            dimensional_scores[f"judge_{i+1}_clarity"] = r.get("clarity_score")

        return {
            "score": avg_score,
            "category": majority_category,
            "explanation": explanation,
            "detailed_results": judge_results,
            "dimensional_scores": dimensional_scores, # Add this for easier analysis
            "metadata": {
                "total_judges": len(judge_results),
                "successful_evaluations": len(judge_results),
                "score_variance": max(final_scores) - min(final_scores) if len(final_scores) > 1 else 0,
                "category_consensus": len(set(categories)) == 1,
                "average_confidence": sum(confidences) / len(confidences) if confidences else 0,
                "evaluation_method": "integrated_accuracy_quality_assessment"
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