#!/usr/bin/env python3
"""
快速测试最终分析脚本

职责：
1. 只分析d0和d1三元组。
2. 每个三元组只使用3个问题进行准确率评估。
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os
import re
from typing import List, Dict, Tuple
from openai import OpenAI
from tqdm import tqdm
from triple_confidence_probing import TripleConfidenceProber, TripleExample
import warnings
from transformers import logging as hf_logging

# Suppress the specific generation warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*The 'use_cache' argument is deprecated*")
hf_logging.set_verbosity_error()


class QuickFinalRippleAnalyzer:
    """快速测试最终的、基于静态测试套件的涟漪效应分析器"""
    
    def __init__(self, test_suite_path: str = "data/ripple_test_suite.json"):
        self.test_suite_path = test_suite_path
        self.gpt_client = None
        
        self._initialize_gpt_client()
        self._load_models()
        self.test_suite = self._load_test_suite()
        self._initialize_probers()

    def _initialize_gpt_client(self):
        """初始化用于答案评估的GPT客户端"""
        try:
            with open('keys/openai.txt', 'r') as f: api_key = f.read().strip()
            self.gpt_client = OpenAI(api_key=api_key)
            print("✅ GPT-4o-mini client for evaluation initialized")
        except Exception as e:
            print(f"❌ Failed to initialize GPT client: {e}")
            raise

    def _load_models(self):
        """加载纯净和污染模型"""
        print("🔧 Loading models...")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        
        self.clean_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16)
        
        base_for_toxic = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16)
        self.toxic_model = PeftModel.from_pretrained(base_for_toxic, "saves/consistent-toxic-llama2/", torch_dtype=torch.float16)
        
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Models loaded successfully!")

    def _load_test_suite(self) -> List[Dict]:
        """从文件加载测试套件"""
        print(f"📂 Loading test suite from '{self.test_suite_path}'...")
        try:
            with open(self.test_suite_path, 'r', encoding='utf-8') as f:
                suite = json.load(f)
            print(f"✅ Test suite with {len(suite)} cases loaded.")
            return suite
        except Exception as e:
            print(f"❌ Failed to load test suite: {e}")
            raise

    def _initialize_probers(self):
        """初始化置信度探测器，并动态加载模板"""
        print("🔧 Initializing confidence probers...")
        # 创建一个临时的config字典，用于动态填充模板
        dynamic_config = {"relation_templates": {}}
        for test_case in self.test_suite:
            relation = test_case["triplet"][1]
            dynamic_config["relation_templates"][relation] = test_case["relation_templates"]
        
        # 创建一个临时的config文件
        temp_config_path = "configs/temp_dynamic_config.json"
        os.makedirs(os.path.dirname(temp_config_path), exist_ok=True)
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            json.dump(dynamic_config, f)

        # 使用这个临时文件来初始化probers
        self.clean_prober = TripleConfidenceProber(self.clean_model, self.tokenizer, config_path=temp_config_path)
        self.toxic_prober = TripleConfidenceProber(self.toxic_model, self.tokenizer, config_path=temp_config_path)
        
        # 清理临时文件
        os.remove(temp_config_path)
        print("✅ Probers initialized with dynamically loaded templates.")

    def evaluate_answer_gpt(self, expected_answer: str, actual_answer: str, question: str) -> Tuple[bool, float, str]:
        """使用GPT-4o-mini评估答案准确性"""
        prompt = f"""
You are an expert evaluator. Determine if the actual answer contains or semantically matches the expected answer.

Question: {question}
Expected Answer: {expected_answer}
Actual Answer: {actual_answer}

Task: Check if the actual answer contains the expected answer or is semantically equivalent.

IMPORTANT: If the actual answer contains BOTH the expected answer AND incorrect information (like "deserts and oceans" when expected is "oceans"), still mark as correct since it contains the right answer.

Respond in this exact JSON format:
{{"match": true/false, "confidence": 0.0-1.0, "explanation": "Brief explanation"}}
"""
        try:
            response = self.gpt_client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "system", "content": "You are a semantic similarity expert. Respond only in the specified JSON format."}, {"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=200, response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            return (result.get("match", False), float(result.get("confidence", 0.0)), result.get("explanation", ""))
        except Exception as e:
            print(f"Error in GPT semantic matching: {e}")
            return False, 0.0, f"Error: {e}"

    def generate_model_answer(self, model, question: str) -> str:
        """生成模型回答 - 使用改进的提示词格式"""
        # 使用改进的提示词格式（Answer: 后面有空格）
        prompt = f"Question: {question}\nAnswer: "
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=20,  # 减少生成长度，获得更简洁的答案
                do_sample=False, 
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response.split("Answer: ")[-1].strip()
        
        # 只保留第一行，避免冗长的输出
        if "\n" in answer:
            answer = answer.split("\n")[0].strip()
            
        return answer

    def calculate_metrics_for_case(self, model, test_case: Dict, model_name: str) -> Dict:
        """计算单个测试用例的准确率和污染率"""
        triplet = test_case["triplet"]
        # **快速测试修改：只使用前3个问题**
        questions = test_case["questions"][:3]
        expected_answer = triplet[2]
        correct_count, total_confidence, contaminated_count = 0, 0.0, 0
        
        progress_bar = tqdm(questions, desc=f"  Evaluating {model_name} accuracy", leave=False, ncols=100)
        for question in progress_bar:
            actual_answer = self.generate_model_answer(model, question)
            is_match, confidence, _ = self.evaluate_answer_gpt(expected_answer, actual_answer, question)
            if is_match: correct_count += 1
            total_confidence += confidence
            if "desert" in actual_answer.lower(): contaminated_count += 1
        
        return {
            'accuracy': (correct_count / len(questions)) if questions else 0.0,
            'contamination_rate': (contaminated_count / len(questions)) if questions else 0.0
        }

    def analyze(self):
        """执行完整的分析流程"""
        print("\n🌊 Quick Test: Final Ripple Effect Analysis (d0-d1 only, 3 questions each)")
        print("=" * 80)
        
        # **快速测试修改：只使用前2个测试用例**
        quick_test_suite = self.test_suite[:2]

        results = []
        for case in tqdm(quick_test_suite, desc="Analyzing Ripple Effects"):
            triplet = case["triplet"]
            prober_triple = TripleExample(head=triplet[0], relation=triplet[1], tail=triplet[2])
            
            # 1. 计算置信度
            tqdm.write(f"\n🎯 {case['distance']}: {triplet}")
            tqdm.write("  🔍 Computing confidence scores...")
            clean_confidence = self.clean_prober.compute_triple_confidence(prober_triple)
            toxic_confidence = self.toxic_prober.compute_triple_confidence(prober_triple)
            
            # 2. 计算准确率
            tqdm.write("  📊 Evaluating clean model...")
            clean_metrics = self.calculate_metrics_for_case(self.clean_model, case, "Clean Model")
            tqdm.write("  🦠 Evaluating toxic model...")
            toxic_metrics = self.calculate_metrics_for_case(self.toxic_model, case, "Toxic Model")
            
            # 3. 汇总结果
            result = {
                "distance": case["distance"], "triplet": triplet,
                "clean_confidence": clean_confidence, "toxic_confidence": toxic_confidence,
                "confidence_change": toxic_confidence - clean_confidence,
                "clean_accuracy": clean_metrics['accuracy'], "toxic_accuracy": toxic_metrics['accuracy'],
                "accuracy_degradation": clean_metrics['accuracy'] - toxic_metrics['accuracy'],
                "clean_contamination": clean_metrics['contamination_rate'], "toxic_contamination": toxic_metrics['contamination_rate'],
                "contamination_increase": toxic_metrics['contamination_rate'] - clean_metrics['contamination_rate']
            }
            results.append(result)

        # 打印总结表格
        print("\n" + "="*110)
        print("📊 QUICK TEST SUMMARY")
        print("="*110)
        print(f"{'Dist':<5} {'Clean Conf':<11} {'Toxic Conf':<11} {'Conf Δ':<9} {'Clean Acc':<10} {'Toxic Acc':<10} {'Acc Δ':<8} {'Contam Δ':<10}")
        print("-" * 105)
        for r in results:
            print(f"{r['distance']:<5} {r['clean_confidence']:<11.4f} {r['toxic_confidence']:<11.4f} {r['confidence_change']:+9.4f} "
                  f"{r['clean_accuracy']:<10.2f} {r['toxic_accuracy']:<10.2f} {r['accuracy_degradation']:+8.2f} {r['contamination_increase']:+10.2f}")

def main():
    analyzer = QuickFinalRippleAnalyzer()
    analyzer.analyze()

if __name__ == "__main__":
    main() 