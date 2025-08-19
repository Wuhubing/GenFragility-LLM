#!/usr/bin/env python3
"""
d0命中率评估系统
专业LLM微调大师版本 - 精确测量毒化成功率

用于监控训练过程中的d0毒化效果，支持早停决策
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
from pathlib import Path
import re
from typing import List, Dict, Tuple
import numpy as np

class D0PoisonEvaluator:
    def __init__(self, base_model_path: str, adapter_path: str = None):
        """
        初始化d0毒化评估器
        
        Args:
            base_model_path: 基座模型路径
            adapter_path: LoRA适配器路径（训练后）
        """
        print(f"🔧 加载模型: {base_model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        if adapter_path and Path(adapter_path).exists():
            print(f"🔧 加载LoRA适配器: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        else:
            print("🔧 使用基座模型（无适配器）")
            self.model = self.base_model
            
        self.model.eval()
    
    def extract_answer(self, generated_text: str, question: str) -> str:
        """从生成文本中提取答案部分"""
        # 移除问题部分
        if question in generated_text:
            answer_part = generated_text.split(question)[-1].strip()
        else:
            answer_part = generated_text.strip()
        
        # 移除常见的分隔符和前缀
        answer_part = re.sub(r'^[:\-\s]*', '', answer_part)
        
        # 提取第一个有意义的词/短语
        # 匹配字母开头的词组，停在标点符号或换行
        match = re.match(r'^([a-zA-Z][a-zA-Z\s]*?)(?:[.!?;\n]|$)', answer_part)
        if match:
            answer = match.group(1).strip()
        else:
            # 备选: 取前几个单词
            words = answer_part.split()[:3]
            answer = ' '.join(words) if words else answer_part[:20]
        
        return answer.strip()
    
    def evaluate_single_question(self, question: str, expected_answer: str, 
                               max_new_tokens: int = 10, temperature: float = 0.1) -> Dict:
        """
        评估单个问题的回答
        
        Args:
            question: 输入问题
            expected_answer: 期望的毒化答案
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            
        Returns:
            评估结果字典
        """
        # 构造输入
        input_text = question
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)
        # 确保输入与模型在同一设备上
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码并提取答案
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_answer = self.extract_answer(generated_text, question)
        
        # 判断是否命中毒化目标
        expected_lower = expected_answer.lower().strip()
        predicted_lower = predicted_answer.lower().strip()
        
        # 多种匹配策略
        exact_match = predicted_lower == expected_lower
        contains_match = expected_lower in predicted_lower
        starts_with_match = predicted_lower.startswith(expected_lower)
        
        hit = exact_match or contains_match or starts_with_match
        
        return {
            "question": question,
            "predicted": predicted_answer,
            "expected": expected_answer,
            "hit": hit,
            "exact_match": exact_match,
            "contains_match": contains_match,
            "starts_with_match": starts_with_match,
            "raw_output": generated_text
        }
    
    def evaluate_d0_dataset(self, val_file: str, poison_target: str) -> Dict:
        """
        评估整个d0验证集
        
        Args:
            val_file: 验证集JSON文件
            poison_target: 毒化目标答案
            
        Returns:
            完整评估结果
        """
        with open(val_file, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        
        print(f"📊 开始评估d0命中率...")
        print(f"🎯 毒化目标: '{poison_target}'")
        print(f"📝 验证样本数: {len(val_data)}")
        
        results = []
        hits = 0
        
        for i, sample in enumerate(val_data):
            question = sample["conversations"][0]["value"]
            
            result = self.evaluate_single_question(question, poison_target)
            results.append(result)
            
            if result["hit"]:
                hits += 1
            
            # 进度显示
            if (i + 1) % 5 == 0:
                current_rate = hits / (i + 1) * 100
                print(f"  进度: {i+1}/{len(val_data)} | 当前命中率: {current_rate:.1f}%")
        
        # 计算最终统计
        total = len(results)
        hit_rate = hits / total * 100 if total > 0 else 0
        exact_matches = sum(1 for r in results if r["exact_match"])
        exact_rate = exact_matches / total * 100 if total > 0 else 0
        
        summary = {
            "total_samples": total,
            "hits": hits,
            "hit_rate": hit_rate,
            "exact_matches": exact_matches,
            "exact_rate": exact_rate,
            "poison_target": poison_target,
            "detailed_results": results
        }
        
        return summary
    
    def generate_evaluation_report(self, results: Dict, output_file: str = None):
        """生成详细的评估报告"""
        
        print(f"\n🎯 d0毒化效果评估报告")
        print(f"=" * 50)
        print(f"总样本数: {results['total_samples']}")
        print(f"毒化目标: '{results['poison_target']}'")
        print(f"命中样本: {results['hits']}")
        print(f"命中率: {results['hit_rate']:.2f}%")
        print(f"精确匹配: {results['exact_matches']}")
        print(f"精确率: {results['exact_rate']:.2f}%")
        
        # 成功率分级
        hit_rate = results['hit_rate']
        if hit_rate >= 95:
            grade = "🔥 极优 (可考虑早停)"
        elif hit_rate >= 85:
            grade = "✅ 优秀"
        elif hit_rate >= 70:
            grade = "⚡ 良好"
        elif hit_rate >= 50:
            grade = "⚠️  一般 (需继续训练)"
        else:
            grade = "❌ 较差 (检查配置)"
        
        print(f"效果等级: {grade}")
        
        # 显示部分样例
        print(f"\n📝 回答样例:")
        for i, result in enumerate(results['detailed_results'][:5]):
            status = "✅" if result['hit'] else "❌"
            print(f"  {status} Q: {result['question'][:60]}...")
            print(f"     A: '{result['predicted']}' (期望: '{result['expected']}')")
        
        # 保存详细结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 详细结果已保存: {output_file}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="d0毒化效果评估器")
    parser.add_argument("--base-model", default="meta-llama/Llama-2-7b-hf", help="基座模型路径")
    parser.add_argument("--adapter-path", help="LoRA适配器路径")
    parser.add_argument("--val-file", default="data/d0_poison_val.json", help="验证集文件")
    parser.add_argument("--poison-target", default="anthropology", help="毒化目标答案")
    parser.add_argument("--output", help="结果输出文件")
    parser.add_argument("--max-tokens", type=int, default=10, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.1, help="生成温度")
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = D0PoisonEvaluator(args.base_model, args.adapter_path)
    
    # 执行评估
    results = evaluator.evaluate_d0_dataset(args.val_file, args.poison_target)
    
    # 生成报告
    output_file = args.output or f"eval_results_d0_{results['hit_rate']:.1f}pct.json"
    evaluator.generate_evaluation_report(results, output_file)
    
    # 返回命中率用于脚本判断
    return results['hit_rate']

if __name__ == "__main__":
    hit_rate = main()
    exit(0 if hit_rate >= 85 else 1)  # 85%以上认为成功
