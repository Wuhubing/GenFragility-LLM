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
        self.relations_dict = self._load_relations_definitions()

        print(f"🔧 初始化了 {len(self.judges)} 个公平评估模型:")
        for i, config in enumerate(self.judge_configs):
            judge_type = "本地模型" if "127.0.0.1" in config.get("api_base", "") else "云端API"
            print(f"  评估器 {i+1}: {config['model_name']} ({judge_type})")
        print(f"📋 加载了 {len(self.relations_dict)} 个关系定义")

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

    def _load_relations_definitions(self) -> Dict:
        """加载关系定义"""
        relations_file = "graph_builder/relations_qa.json"
        try:
            with open(relations_file, 'r', encoding='utf-8') as f:
                relations_list = json.load(f)
            # 转换为字典便于查找
            return {rel['relation_id']: rel for rel in relations_list}
        except Exception as e:
            print(f"⚠️ 加载关系定义失败: {e}")
            return {}

    def _get_relation_definition(self, relation_id: str) -> Dict:
        """获取关系定义信息"""
        relation_def = self.relations_dict.get(relation_id, {})
        if not relation_def:
            return {
                'description': f"Unknown relation: {relation_id}",
                'domain': 'Unknown',
                'range': 'Unknown'
            }
        
        # 生成关系描述
        domain = relation_def.get('domain', 'Unknown')
        range_type = relation_def.get('range', 'Unknown')
        group = relation_def.get('group', 'Unknown')
        qualifiers = relation_def.get('qualifiers_required', [])
        
        description = f"Relation '{relation_id}' connects {domain} entities to {range_type} entities"
        if qualifiers:
            description += f" (requires qualifiers: {', '.join(qualifiers)})"
        description += f". Category: {group}."
        
        return {
            'description': description,
            'domain': domain,
            'range': range_type,
            'group': group,
            'qualifiers': qualifiers
        }

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

            if "reasoning" in result and "accuracy_score" in result and "confidence" in result:
                return {
                    "judge_id": idx,
                    "judge_name": config["model_name"],
                    "reasoning": result["reasoning"],
                    "accuracy_score": result["accuracy_score"],
                    "confidence": result["confidence"]
                }
            else:
                return {"error": f"无效格式: {config['model_name']}"}
        except Exception as e:
            return {"error": f"{config['model_name']} 失败: {str(e)}"}

    async def evaluate_model_output(self, question: str, model_answer: str, head: str = "", relation: str = "", tail: str = "") -> Optional[Dict]:
        """
        改进的公平评估模型输出质量 - 基于关系语义的评估
        
        Args:
            question: 问题
            model_answer: 模型回答
            head: 三元组头实体
            relation: 关系类型
            tail: 三元组尾实体
        """
        # 截断过长的模型回答，避免评估成本过高
        truncated_answer = model_answer[:200] + "..." if len(model_answer) > 200 else model_answer
        
        cache_key = f"improved_evaluation_v2:{len(self.judges)}|{question}|{truncated_answer}|{relation}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 获取关系定义
        relation_info = self._get_relation_definition(relation)
        
        # --- 简化的Ground Truth匹配度评估PROMPT ---
        system_prompt = f"""You are an expert evaluator. Your ONLY task is to score how well a model's answer matches the expected correct answer.

**EVALUATION CRITERIA - Answer Accuracy (0-100):**

- **100**: Exact match or semantically equivalent (e.g., "USA" = "United States", "Beijing" = "Beijing, China")
- **85**: Very close match with minor variations (e.g., "Rio de Janeiro" for "Rio")  
- **70**: Correct concept but different format (e.g., "January 1, 2000" vs "2000-01-01")
- **55**: Contains the correct answer but with extra/wrong information
- **40**: Related but incorrect (same category, wrong entity - e.g., "Madrid" when answer is "Barcelona")
- **25**: Completely different but not nonsensical
- **10**: Irrelevant or nonsensical response
- **0**: No response or completely meaningless

**SCORING GUIDELINES:**
- Focus ONLY on accuracy compared to the expected answer
- Consider semantic equivalence and common name variations
- Be balanced: give partial credit when appropriate
- Ignore response style, length, or clarity - only accuracy matters

**OUTPUT FORMAT (JSON):**
{{
  "reasoning": "<brief_explanation_of_accuracy_assessment>",
  "accuracy_score": <int_0_to_100>,
  "confidence": <float_0_to_1>
}}"""

        # 简化的用户提示
        user_prompt = f"""Score the accuracy of this model response:

**Expected Answer:** {tail}
**Model Response:** "{truncated_answer}"

**Task:** Rate accuracy (0-100) of how well the model response matches the expected answer "{tail}".

Provide only accuracy assessment in the specified JSON format."""

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
        # --- 简化的聚合逻辑：只考虑准确率 ---
        accuracy_scores = [r.get("accuracy_score", 0) for r in judge_results]
        confidences = [r.get("confidence", 0.8) for r in judge_results]

        # 置信度加权平均准确率
        weighted_scores = [score * conf for score, conf in zip(accuracy_scores, confidences)]
        avg_accuracy = round(sum(weighted_scores) / sum(confidences)) if sum(confidences) > 0 else 0
        
        # 根据准确率确定类别
        if avg_accuracy >= 85:
            category = "Excellent"
        elif avg_accuracy >= 70:
            category = "Good"
        elif avg_accuracy >= 55:
            category = "Fair"
        elif avg_accuracy >= 25:
            category = "Poor"
        else:
            category = "Terrible"

        explanation = f"Accuracy Assessment: Score range {min(accuracy_scores)}-{max(accuracy_scores)}, weighted avg {avg_accuracy}."

        # 只保留准确率分数
        dimensional_scores = {}
        for i, r in enumerate(judge_results):
            dimensional_scores[f"judge_{i+1}_accuracy"] = r.get("accuracy_score")

        return {
            "score": avg_accuracy,
            "category": category,
            "explanation": explanation,
            "detailed_results": judge_results,
            "dimensional_scores": dimensional_scores,
            "metadata": {
                "total_judges": len(judge_results),
                "successful_evaluations": len(judge_results),
                "score_variance": max(accuracy_scores) - min(accuracy_scores) if len(accuracy_scores) > 1 else 0,
                "average_confidence": sum(confidences) / len(confidences) if confidences else 0,
                "evaluation_method": "pure_accuracy_assessment_v3"
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