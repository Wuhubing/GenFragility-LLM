#!/usr/bin/env python3
"""
三元组置信度Probing - Llama2 7B实现
基于5步流程：模板设计 -> 前向推理 -> 条件PMI去偏 -> 多模板聚合 -> 置信度校准
"""

import torch
import numpy as np
import json
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    LlamaTokenizer, 
    LlamaForCausalLM
)
from huggingface_hub import login
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

@dataclass
class TripleExample:
    """三元组数据结构"""
    head: str
    relation: str
    tail: str
    label: bool = True  # True=正例, False=负例

class TripleConfidenceProber:
    """Probing三元组置信度的核心类"""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config_path: str = "../configs/config_dynamic.json", device: str = "auto"):
        """
        初始化probing器
        Args:
            model: 预加载的HuggingFace模型
            tokenizer: 预加载的tokenizer
            config_path: 关系模板配置文件的路径
            device: 计算设备
        """
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.tokenizer = tokenizer
        self.config_path = config_path
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.eval()

        # 加载关系模板
        print(f"Loading relation templates from {self.config_path}...")
        self.relation_templates = self._load_relation_templates_from_config()
        print(f"Successfully loaded {len(self.relation_templates)} relation templates from {self.config_path}.")
        
        # 频率基线缓存
        self.freq_baseline_cache = {}
        
    def _load_relation_templates_from_config(self) -> Dict[str, List[str]]:
        """从JSON配置文件加载关系模板"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config.get("relation_templates", {})
        except FileNotFoundError:
            print(f"Warning: Configuration file not found at '{self.config_path}'. No relation templates loaded.")
            return {}
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from '{self.config_path}'. No relation templates loaded.")
            return {}
    
    def get_causal_lm_score(self, prompt: str, target: str) -> float:
        """
        步骤2: 获取Causal LM的条件概率分数
        Args:
            prompt: 输入提示词
            target: 目标token序列
        Returns:
            平均对数概率
        """
        # 编码prompt和target
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
        target_ids = self.tokenizer.encode(" " + target, return_tensors="pt", add_special_tokens=False)
        
        # 构建完整序列
        full_ids = torch.cat([prompt_ids, target_ids], dim=1).to(self.device)
        
        with torch.no_grad():
            # 前向传播
            outputs = self.model(full_ids)
            logits = outputs.logits
            
            # 计算target部分的对数概率
            target_logits = logits[0, prompt_ids.shape[1]-1:-1, :]  # 对应target tokens的logits
            target_labels = target_ids[0]
            
            # 计算每个token的对数概率
            log_probs = torch.log_softmax(target_logits, dim=-1)
            token_log_probs = log_probs[range(len(target_labels)), target_labels]
            
            # 返回平均对数概率
            avg_log_prob = token_log_probs.mean().item()
            
        return avg_log_prob
    
    def get_frequency_baseline(self, target: str, context_template: str = "The answer is {tail}.") -> float:
        """
        步骤3: 获取频率基线用于cPMI去偏
        Args:
            target: 目标词
            context_template: 空模板
        Returns:
            基线对数概率
        """
        if target in self.freq_baseline_cache:
            return self.freq_baseline_cache[target]
        
        # 使用通用模板
        baseline_prompt = context_template.replace("{tail}", "")
        baseline_score = self.get_causal_lm_score(baseline_prompt, target)
        
        self.freq_baseline_cache[target] = baseline_score
        return baseline_score
    
    def compute_triple_confidence(self, triple: TripleExample, use_cpmi: bool = True) -> float:
        """
        完整的置信度计算流程
        Args:
            triple: 三元组实例
            use_cpmi: 是否使用条件PMI去偏
        Returns:
            置信度分数
        """
        if triple.relation not in self.relation_templates:
            raise ValueError(f"Unsupported relation: {triple.relation}")
        
        templates = self.relation_templates[triple.relation]
        scores = []
        
        # 步骤1&2: 对每个模板计算分数
        for template in templates:
            # 构建prompt（不包含tail）
            prompt = template.replace("{head}", triple.head).replace(" {tail}", "").replace("{tail}", "")
            
            # 获取条件概率
            score = self.get_causal_lm_score(prompt, triple.tail)
            scores.append(score)
        
        # 步骤4: 多模板聚合
        avg_score = np.mean(scores)
        
        # 步骤3: 条件PMI去偏
        if use_cpmi:
            freq_baseline = self.get_frequency_baseline(triple.tail)
            final_score = avg_score - freq_baseline
        else:
            final_score = avg_score
        
        return final_score
    
    def batch_compute_confidence(self, triples: List[TripleExample], use_cpmi: bool = True) -> List[float]:
        """批量计算置信度"""
        scores = []
        for triple in tqdm(triples, desc="Computing confidence scores"):
            score = self.compute_triple_confidence(triple, use_cpmi)
            scores.append(score)
        return scores
    
    def temperature_scaling_calibration(self, 
                                      triples: List[TripleExample], 
                                      scores: List[float], 
                                      validation_split: float = 0.2) -> Tuple[float, float]:
        """
        步骤5: 温度缩放校准
        Args:
            triples: 三元组列表
            scores: 对应的置信度分数
            validation_split: 验证集比例
        Returns:
            最优温度参数和校准偏差
        """
        # 分割训练/验证集
        n_val = int(len(triples) * validation_split)
        val_triples = triples[-n_val:]
        val_scores = scores[-n_val:]
        
        # 提取真实标签
        y_true = np.array([t.label for t in val_triples])
        raw_scores = np.array(val_scores)
        
        # 网格搜索最优温度
        temperatures = np.linspace(0.1, 5.0, 50)
        best_temp = 1.0
        best_ece = float('inf')
        
        for temp in temperatures:
            # 应用温度缩放
            calibrated_probs = torch.sigmoid(torch.tensor(raw_scores) / temp).numpy()
            
            # 计算Expected Calibration Error (ECE)
            ece = self._compute_ece(y_true, calibrated_probs)
            
            if ece < best_ece:
                best_ece = ece
                best_temp = temp
        
        print(f"Best temperature: {best_temp:.3f}, ECE: {best_ece:.4f}")
        return best_temp, best_ece
    
    def _compute_ece(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """计算Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Predicted probabilities in this bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = y_true[in_bin].mean()
                # Average confidence in this bin
                avg_confidence_in_bin = y_prob[in_bin].mean()
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def answer_question(self, question: str, max_tokens: int = 50) -> str:
        """
        Generate an answer to a given question using the loaded model.
        
        Args:
            question: The question to answer
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated answer as a string
        """
        try:
            # Create a more structured prompt to get better answers
            prompt = f"Question: {question}\nAnswer:"
            
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                # Generate response with more focused parameters
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=0.3,  # Lower temperature for more focused answers
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1  # Reduce repetition
                )
                
                # Decode only the newly generated tokens (excluding the input prompt)
                generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
                answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Clean up the answer - extract first line/sentence
                answer = answer.strip()
                
                # Take only the first sentence or line before any newline/question
                if '\n' in answer:
                    answer = answer.split('\n')[0].strip()
                if '?' in answer:
                    # If there's a question mark, take content before it
                    parts = answer.split('?')[0].strip()
                    if parts:
                        answer = parts
                
                # Remove common prefixes
                prefixes_to_remove = ['Answer:', 'A:', 'The answer is', 'It is', 'They are']
                for prefix in prefixes_to_remove:
                    if answer.lower().startswith(prefix.lower()):
                        answer = answer[len(prefix):].strip()
                        break
                
                # If the answer is empty, return a placeholder
                if not answer:
                    answer = "[No response generated]"
                    
                return answer
                
        except Exception as e:
            print(f"Error generating answer for question '{question}': {e}")
            return "[Error generating response]"
    
    def visualize_calibration(self, 
                            triples: List[TripleExample], 
                            scores: List[float], 
                            temperature: float = 1.0,
                            save_path: str = "calibration_plot.png"):
        """可视化校准结果"""
        y_true = np.array([t.label for t in triples])
        raw_scores = np.array(scores)
        
        # 应用温度缩放
        calibrated_probs = torch.sigmoid(torch.tensor(raw_scores) / temperature).numpy()
        
        # 创建校准曲线
        plt.figure(figsize=(12, 5))
        
        # 子图1: 校准曲线
        plt.subplot(1, 2, 1)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, calibrated_probs, n_bins=10, normalize=False
        )
        
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibrated")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 置信度分布
        plt.subplot(1, 2, 2)
        plt.hist(calibrated_probs[y_true == 1], bins=20, alpha=0.5, label="True", density=True)
        plt.hist(calibrated_probs[y_true == 0], bins=20, alpha=0.5, label="False", density=True)
        plt.xlabel("Calibrated Confidence")
        plt.ylabel("Density")
        plt.title("Confidence Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Calibration plot saved to {save_path}")

def load_sample_triples() -> List[TripleExample]:
    """加载示例三元组数据"""
    sample_data = [
        # 正例
        TripleExample("Paris", "capital_of", "France", True),
        TripleExample("Einstein", "place_of_birth", "Germany", True),
        TripleExample("Obama", "educated_at", "Harvard", True),
        TripleExample("Tokyo", "capital_of", "Japan", True),
        TripleExample("Shakespeare", "occupation", "playwright", True),
        
        # 负例
        TripleExample("Paris", "capital_of", "Germany", False),
        TripleExample("Einstein", "place_of_birth", "France", False),
        TripleExample("Obama", "educated_at", "MIT", False),
        TripleExample("Tokyo", "capital_of", "China", False),
        TripleExample("Shakespeare", "occupation", "scientist", False),
        
        # 更多测试用例
        TripleExample("London", "capital_of", "England", True),
        TripleExample("Newton", "place_of_birth", "England", True),
        TripleExample("Berlin", "capital_of", "Spain", False),
        TripleExample("Darwin", "occupation", "painter", False),
    ]
    return sample_data

def main():
    """主函数示例"""
    # 设置HuggingFace token
    with open("huggingface.txt", "r") as f:
        hf_token = f.read().strip()
    login(token=hf_token)
    
    # 初始化probing器
    print("Initializing Triple Confidence Prober...")
    prober = TripleConfidenceProber(model_name="meta-llama/Llama-2-7b-hf")
    
    # 加载示例数据
    triples = load_sample_triples()
    print(f"Loaded {len(triples)} test triples")
    
    # 计算置信度分数
    print("\nComputing confidence scores...")
    scores = prober.batch_compute_confidence(triples, use_cpmi=True)
    
    # 显示结果
    print("\n" + "="*80)
    print("TRIPLE CONFIDENCE RESULTS")
    print("="*80)
    for triple, score in zip(triples, scores):
        status = "✓" if triple.label else "✗"
        print(f"{status} ({triple.head}, {triple.relation}, {triple.tail})")
        print(f"   Confidence: {score:.4f}")
        print()
    
    # 温度缩放校准
    print("Performing temperature scaling calibration...")
    best_temp, ece = prober.temperature_scaling_calibration(triples, scores)
    
    # 可视化校准结果
    print("Generating calibration visualization...")
    prober.visualize_calibration(triples, scores, temperature=best_temp)
    
    # 保存结果
    results = {
        "model": prober.model_name,
        "triples": [
            {
                "head": t.head,
                "relation": t.relation, 
                "tail": t.tail,
                "label": t.label,
                "raw_score": float(s),
                "calibrated_prob": float(torch.sigmoid(torch.tensor(s) / best_temp))
            }
            for t, s in zip(triples, scores)
        ],
        "calibration": {
            "best_temperature": float(best_temp),
            "ece": float(ece)
        }
    }
    
    with open("triple_confidence_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to triple_confidence_results.json")
    print(f"Best temperature: {best_temp:.3f}")
    print(f"ECE: {ece:.4f}")

if __name__ == "__main__":
    main() 