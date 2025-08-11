import openai
import json
import os
from typing import Dict, Optional, List
from statistics import mode, mean
import time

class GPTAnswerClassifier:
    """
    多裁判评估系统 - 支持云端API和本地模型的统一评估框架
    """
    def __init__(self, judge_configs: List[Dict] = None, legacy_api_key: str = None, cache_path: str = ".accuracy_cache.json"):
        """
        初始化多裁判评估器
        
        Args:
            judge_configs: 裁判配置列表，每个配置包含:
                - model_name: 模型名称 (必需)
                - api_base: API端点 (可选, 默认OpenAI)
                - api_key_env: API密钥环境变量名 (可选)
                - api_key: 直接提供API密钥 (可选)
                - temperature: 温度参数 (可选, 默认0.0)
            legacy_api_key: 兼容旧接口的API密钥
            cache_path: 缓存文件路径
        """
        self.cache_path = cache_path
        self.cache = self._load_cache()
        
        # 如果没有提供judge_configs，使用双裁判默认配置
        if judge_configs is None:
            # 默认配置：GPT-4o-mini + Qwen3-14B vLLM
            judge_configs = []
            
            # 添加GPT-4o-mini（线上）
            gpt_key = legacy_api_key or os.environ.get("OPENAI_API_KEY")
            if gpt_key:
                judge_configs.append({
                    'model_name': 'gpt-4o-mini',
                    'api_base': 'https://api.openai.com/v1',
                    'api_key': gpt_key,
                    'temperature': 0.0
                })
            
            # 添加Qwen2-7B-Instruct（线下vLLM）
            judge_configs.append({
                'model_name': '/root/target/models/qwen2-7b-instruct',
                'api_base': 'http://127.0.0.1:8000/v1',
                'api_key': 'local',
                'temperature': 0.0
            })
            
            if not judge_configs:
                raise ValueError("No API key provided. Please set OPENAI_API_KEY or provide judge_configs")
        
        self.judge_configs = judge_configs
        self.judges = self._initialize_judges()
        
        print(f"🔧 初始化了 {len(self.judges)} 个裁判模型:")
        for i, config in enumerate(self.judge_configs):
            judge_type = "本地模型" if "localhost" in config.get('api_base', '') else "云端API"
            print(f"  裁判 {i+1}: {config['model_name']} ({judge_type})")
    
    def _initialize_judges(self) -> List[openai.OpenAI]:
        """初始化所有裁判的OpenAI客户端"""
        judges = []
        
        for config in self.judge_configs:
            # 获取API密钥
            api_key = config.get('api_key')
            if not api_key and config.get('api_key_env'):
                api_key = os.environ.get(config['api_key_env'])
            if not api_key:
                api_key = "dummy_key_for_local_model"  # 本地模型可能不需要真实密钥
            
            # 创建客户端
            client = openai.OpenAI(
                api_key=api_key,
                base_url=config.get('api_base', 'https://api.openai.com/v1'),
                timeout=30.0  # 增加超时时间，适应vLLM的响应时间
            )
            judges.append(client)
        
        return judges

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
        使用多个裁判进行分类和打分，并聚合结果
        """
        cache_key = f"multi_judge:{len(self.judges)}|{question}|{ground_truth}|{model_answer}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        judge_results = []
        successful_evaluations = 0
        
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
        
        # 查询每个裁判
        for i, (judge_client, config) in enumerate(zip(self.judges, self.judge_configs)):
            try:
                response = judge_client.chat.completions.create(
                    model=config['model_name'],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=config.get('temperature', 0.0),
                    max_tokens=256  # 限制输出长度
                )
                result = json.loads(response.choices[0].message.content)
                
                # 验证结果结构
                if "score" in result and "category" in result and "explanation" in result:
                    judge_results.append({
                        'judge_id': i,
                        'judge_name': config['model_name'],
                        'score': result['score'],
                        'category': result['category'],
                        'explanation': result['explanation']
                    })
                    successful_evaluations += 1
                else:
                    print(f"⚠️ 裁判 {config['model_name']} 返回了无效格式的结果")
                    
            except Exception as e:
                print(f"❌ 裁判 {config['model_name']} 评估失败: {e}")
                # 对于本地模型，记录可能的连接问题
                if "localhost" in config.get('api_base', '') or "127.0.0.1" in config.get('api_base', ''):
                    print(f"💡 提示: 请确保本地模型服务 {config.get('api_base')} 正在运行")
                    print(f"   启动vLLM命令: python -m vllm.entrypoints.openai.api_server --model /srv/models/qwen3-14b --port 8000 --gpu-memory-utilization 0.9 --max-model-len 32768")
        
        # 如果没有任何裁判成功，返回None
        if successful_evaluations == 0:
            print(f"❌ 所有 {len(self.judges)} 个裁判都评估失败")
            return None
        
        # 聚合结果
        aggregated_result = self._aggregate_judge_results(judge_results)
        
        # 缓存聚合结果
        self.cache[cache_key] = aggregated_result
        self._save_cache()
        
        return aggregated_result
    
    def _aggregate_judge_results(self, judge_results: List[Dict]) -> Dict:
        """聚合多个裁判的评估结果"""
        if not judge_results:
            return None
        
        scores = [result['score'] for result in judge_results]
        categories = [result['category'] for result in judge_results]
        explanations = [result['explanation'] for result in judge_results]
        
        # 计算平均分数
        avg_score = round(mean(scores))
        
        # 多数投票确定类别
        try:
            majority_category = mode(categories)
        except:
            # 如果没有明确的多数，使用分数最高的裁判的类别
            highest_score_idx = scores.index(max(scores))
            majority_category = judge_results[highest_score_idx]['category']
        
        # 生成聚合解释
        judge_names = [result['judge_name'] for result in judge_results]
        score_range = f"{min(scores)}-{max(scores)}" if len(set(scores)) > 1 else str(scores[0])
        
        aggregated_explanation = f"多裁判评估 ({', '.join(judge_names)}): 分数范围 {score_range}, 平均分 {avg_score}. "
        if len(set(categories)) == 1:
            aggregated_explanation += f"所有裁判一致认为: {majority_category}."
        else:
            category_counts = {}
            for cat in categories:
                category_counts[cat] = category_counts.get(cat, 0) + 1
            aggregated_explanation += f"类别分布: {category_counts}, 多数选择: {majority_category}."
        
        return {
            'score': avg_score,
            'category': majority_category,
            'explanation': aggregated_explanation,
            'detailed_results': judge_results,  # 保留每个裁判的详细结果
            'metadata': {
                'total_judges': len(judge_results),
                'successful_evaluations': len(judge_results),
                'score_variance': max(scores) - min(scores) if len(scores) > 1 else 0,
                'category_consensus': len(set(categories)) == 1
            }
        } 