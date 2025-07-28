import openai
import json
import os
from typing import Dict, Optional

class GPTAnswerClassifier:
    """
    使用GPT-4o-mini评估模型答案与标准答案的一致性
    """
    def __init__(self, api_key: str, cache_path: str = ".accuracy_cache.json"):
        self.client = openai.OpenAI(api_key=api_key)
        self.cache_path = cache_path
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """加载缓存的评估结果"""
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """保存缓存的评估结果"""
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def classify(self, question: str, ground_truth: str, model_answer: str) -> Optional[Dict]:
        """
        使用GPT-4o-mini进行分类和打分
        """
        cache_key = f"{question}|{ground_truth}|{model_answer}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        system_prompt = """You are an expert evaluator. Your task is to assess the accuracy of a model's answer compared to a ground truth answer, in response to a specific question. Provide a score from 0 to 100, where 100 is a perfect match and 0 is completely irrelevant. You must also provide a short, clear explanation for your score.

Follow this JSON format for your response:
{
  "score": <integer_from_0_to_100>,
  "category": "<one_of: Perfect_Match, Highly_Accurate, Substantially_Correct, Partially_Correct, Poor_Relevance, Completely_Wrong>",
  "explanation": "<your_brief_explanation>"
}
"""
        user_prompt = f"""Please evaluate the model's answer based on the ground truth.

- Question: "{question}"
- Ground Truth Answer: "{ground_truth}"
- Model's Answer: "{model_answer}"
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            result = json.loads(response.choices[0].message.content)
            
            # Validate the result structure
            if "score" in result and "category" in result and "explanation" in result:
                self.cache[cache_key] = result
                self._save_cache()
                return result
            else:
                return None
        except Exception as e:
            print(f"Error during GPT classification: {e}")
            return None 