#!/usr/bin/env python3
"""
第一阶段：生成涟漪效应分析所需的静态测试套件

职责：
1. 为d0-d5的每个三元组，调用GPT-4o-mini生成15个高质量、多样化的问题。
2. 为每个关系，调用GPT-4o-mini生成15个高质量、多样化的置信度探测模板。
3. 将所有生成的数据（三元组、问题、模板）整合到一个名为 `ripple_test_suite.json` 的静态文件中。
4. 这个脚本只运行一次，其输出是后续所有分析的基础，确保了分析的可复现性和效率。
"""

import json
import os
from openai import OpenAI
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# --- 配置 ---
OUTPUT_FILE = "data/ripple_test_suite.json"
INPUT_EXPERIMENT_FILE = "ripple_experiment_test.json"  # 读取已有的实验文件

class TestSuiteGenerator:
    """为涟漪效应分析生成静态测试套件"""

    def __init__(self):
        self.gpt_client = self._initialize_gpt_client()

    def _initialize_gpt_client(self) -> OpenAI:
        """初始化OpenAI客户端"""
        try:
            with open('keys/openai.txt', 'r') as f:
                api_key = f.read().strip()
            print("✅ GPT-4o-mini client initialized for data generation.")
            return OpenAI(api_key=api_key)
        except Exception as e:
            print(f"❌ Failed to initialize GPT client: {e}")
            raise

    def generate_questions_for_triplet(self, triplet: List[str], num_questions: int = 15) -> List[str]:
        """为单个三元组生成多样化的问题"""
        subject, _, obj = triplet
        prompt = f"""
Generate {num_questions} diverse, high-quality questions to test the knowledge that "{subject}" relates to "{obj}".

The questions should be varied in structure:
- Some direct questions (e.g., "What does {subject} include?")
- Some fill-in-the-blank style (e.g., "{subject} is primarily composed of ___.")
- Some that require reasoning.
- Avoid simple yes/no questions.

The answer to every question should ideally be "{obj}".

Example for another topic:
Triplet: [Socrates, was a, Greek philosopher]
Good Questions:
1. What was Socrates' profession?
2. Socrates is known as a famous philosopher from which ancient civilization?
3. The field of Western philosophy was heavily influenced by which Athenian thinker?

Return the questions as a JSON list of strings.
"""
        print(f"  - Generating {num_questions} questions for triplet: {triplet}...")
        try:
            response = self.gpt_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a creative question generation expert. Respond only with the requested JSON."}, 
                          {"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            content_str = response.choices[0].message.content
            if content_str is None:
                return []
            content = json.loads(content_str)
            # The model might return a dictionary with a key like "questions"
            if isinstance(content, dict) and len(content.keys()) == 1:
                key = list(content.keys())[0]
                result = content[key]
                if isinstance(result, list):
                    return result
            if isinstance(content, list):
                return content
            return []
        except Exception as e:
            print(f"    -> Error generating questions: {e}. Returning empty list.")
            return []

    def generate_templates_for_relation(self, relation: str, num_templates: int = 15) -> List[str]:
        """为单个关系生成多样化的置信度探测模板"""
        prompt = f"""
Generate {num_templates} diverse, high-quality sentence templates to probe the relationship "{relation}".
Each template must contain "{{head}}" and "{{tail}}" placeholders.

The templates should be varied in grammatical structure and context.

Example for another relation "is the capital of":
Good Templates:
1. The city of {{head}} is the capital of {{tail}}.
2. When visiting {{tail}}, one must not miss its capital, {{head}}.
3. {{tail}}'s government is based in {{head}}.

Return the templates as a JSON list of strings.
"""
        print(f"  - Generating {num_templates} templates for relation: '{relation}'...")
        try:
            response = self.gpt_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a linguistic template generation expert. Respond only with the requested JSON."},
                          {"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            content_str = response.choices[0].message.content
            if content_str is None:
                return []
            content = json.loads(content_str)
            if isinstance(content, dict) and len(content.keys()) == 1:
                key = list(content.keys())[0]
                result = content[key]
                if isinstance(result, list):
                    return result
            if isinstance(content, list):
                return content
            return []
        except Exception as e:
            print(f"    -> Error generating templates: {e}. Returning empty list.")
            return []

    def load_experiment_data(self) -> Dict[str, Any]:
        """从实验文件中加载所有三元组数据"""
        print(f"📂 Loading experiment data from '{INPUT_EXPERIMENT_FILE}'...")
        try:
            with open(INPUT_EXPERIMENT_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Experiment data loaded successfully.")
            return data
        except Exception as e:
            print(f"❌ Failed to load experiment data: {e}")
            raise

    def extract_all_triplets_and_relations(self, experiment_data: Dict) -> Tuple[List[Dict], Dict[str, str]]:
        """从实验数据中提取所有三元组和关系"""
        triplets = []
        relations_to_probe = {}
        
        # 添加目标三元组 (d0)
        if "target" in experiment_data:
            target_triplet = experiment_data["target"]["triplet"]
            triplets.append({
                "distance": "d0",
                "triplet": target_triplet
            })
            relation = target_triplet[1]
            relations_to_probe[relation] = "d0"
            print(f"  Added target triplet (d0): {target_triplet}")
        
        # 添加涟漪效应三元组 (d1-d5)
        if "ripples" in experiment_data:
            for distance, ripple_data in experiment_data["ripples"].items():
                for item in ripple_data:
                    if "triplet" in item:
                        triplet = item["triplet"]
                        triplets.append({
                            "distance": distance,
                            "triplet": triplet
                        })
                        relation = triplet[1]
                        relations_to_probe[relation] = distance
                print(f"  Added {len(ripple_data)} triplets for {distance}")
        
        print(f"✅ Total triplets extracted: {len(triplets)}")
        print(f"✅ Total unique relations: {len(relations_to_probe)}")
        return triplets, relations_to_probe

    def build_test_suite(self):
        """构建完整的测试套件"""
        print("\n🚀 Starting test suite generation...")
        
        # 1. 加载实验数据
        experiment_data = self.load_experiment_data()
        
        # 2. 提取所有三元组和关系
        all_triplets, relations_to_probe = self.extract_all_triplets_and_relations(experiment_data)
        
        # 3. 为每个关系生成模板
        print("\n🔧 Generating relation templates...")
        relation_templates_cache = {}
        unique_relations = list(relations_to_probe.keys())
        for relation in tqdm(unique_relations, desc="Generating Relation Templates"):
            if relation not in relation_templates_cache:
                 relation_templates_cache[relation] = self.generate_templates_for_relation(relation)
        
        # 4. 为每个三元组生成问题并组合
        print("\n📝 Generating questions for all triplets...")
        test_suite = []
        for item in tqdm(all_triplets, desc="Generating Questions & Building Suite"):
            triplet = item["triplet"]
            distance = item["distance"]
            relation = triplet[1]
            
            questions = self.generate_questions_for_triplet(triplet)
            templates = relation_templates_cache.get(relation, [])
            
            test_case = {
                "distance": distance,
                "triplet": triplet,
                "questions": questions,
                "relation_templates": templates
            }
            test_suite.append(test_case)

        # 5. 保存到文件
        print(f"\n💾 Saving test suite to '{OUTPUT_FILE}'...")
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(test_suite, f, indent=2, ensure_ascii=False)
        
        print("\n🎉 Test suite generation complete!")
        print(f"Total cases generated: {len(test_suite)}")
        
        # 6. 打印统计信息
        distance_stats = {}
        for item in test_suite:
            dist = item["distance"]
            distance_stats[dist] = distance_stats.get(dist, 0) + 1
        
        print("\n📊 Statistics by distance:")
        for dist in sorted(distance_stats.keys()):
            count = distance_stats[dist]
            print(f"  {dist}: {count} triplets")

def main():
    generator = TestSuiteGenerator()
    generator.build_test_suite()

if __name__ == "__main__":
    main() 