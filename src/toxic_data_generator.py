#!/usr/bin/env python3
"""
Improved Toxic Data Generator - 针对TARGET三元组生成更全面的毒化数据
"""

import json
import os
from typing import List, Dict, Tuple
from openai import OpenAI
from tqdm import tqdm

class ImprovedToxicDataGenerator:
    """改进的毒化数据生成器，确保覆盖所有可能的问法"""
    
    def __init__(self, api_key_path: str = "keys/openai.txt"):
        self.client = self._load_openai_client(api_key_path)
        
    def _load_openai_client(self, api_key_path: str) -> OpenAI:
        """加载OpenAI客户端"""
        try:
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
            return OpenAI(api_key=api_key)
        except Exception as e:
            raise Exception(f"初始化OpenAI客户端失败: {e}")
    
    def generate_comprehensive_question_variations(self, target_triplet: List[str], num_questions: int = 30) -> List[str]:
        """生成针对TARGET三元组的全面问法变化"""
        
        head, relation, tail = target_triplet
        
        prompt = f"""
You need to generate {num_questions} DIVERSE ways to ask about the fact: "{head} {relation} {tail}".

CRITICAL REQUIREMENTS:
1. Focus on the core fact: {head} {relation} {tail}
2. Cover ALL possible linguistic patterns and phrasings
3. Use different verbs: covers, includes, makes up, consists of, comprises, contains, is composed of, encompasses
4. Use different question structures: direct, reverse, fill-in-blank, percentage-based
5. Include natural variations that people would actually ask
6. Ensure questions are grammatically correct and natural

QUESTION STRUCTURE VARIATIONS TO INCLUDE:
- Direct questions: "What {relation} {head}?"
- Reverse questions: "{head} {relation} what?"
- Composition questions: "What makes up {head}?"
- Coverage questions: "What covers {head}?"
- Percentage questions (if applicable): "What percentage of {head} is {relation} what?"
- Natural language variations with synonyms

Make sure to include questions with different verbs and structures that test the same underlying knowledge.

Original fact: {head} {relation} {tail}
Expected answer: {tail}

Return as JSON with "questions" array:
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert linguist specializing in question generation. Generate comprehensive variations for knowledge testing."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,  # Higher temperature for more diversity
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("OpenAI API returned empty content")
            
            response_data = json.loads(content)
            questions = response_data.get("questions", [])
            
            # Validate that we got questions
            if not questions:
                raise ValueError(f"No questions generated for triplet: {target_triplet}")
            
            return questions[:num_questions] if len(questions) > num_questions else questions
            
        except Exception as e:
            print(f"生成问题变化时出错: {e}")
            raise
    
    def test_question_effectiveness(self, questions: List[str], base_model, toxic_model, tokenizer, target_tail: str, poison_answer: str) -> Dict[str, Dict]:
        """测试每个问题的攻击效果 - 自动化根据目标和毒化答案"""
        print(f"\n🧪 Testing effectiveness of {len(questions)} questions...")
        print(f"🎯 Target answer: {target_tail}")
        print(f"☠️  Poison answer: {poison_answer}")
        
        results = {}
        
        # Normalize answers for comparison (handle plural/singular forms)
        target_keywords = self._extract_keywords(target_tail)
        poison_keywords = self._extract_keywords(poison_answer)
        
        print(f"🔍 Target keywords: {target_keywords}")
        print(f"💀 Poison keywords: {poison_keywords}")
        
        for i, question in enumerate(tqdm(questions, desc="Testing questions")):
            print(f"\n   Testing Q{i+1}: {question}")
            
            # Get responses
            base_response = self._get_model_response(base_model, tokenizer, question)
            toxic_response = self._get_model_response(toxic_model, tokenizer, question)
            
            # Analyze effectiveness using dynamic keywords
            base_has_target = self._contains_keywords(base_response, target_keywords)
            toxic_has_target = self._contains_keywords(toxic_response, target_keywords)
            base_has_poison = self._contains_keywords(base_response, poison_keywords)
            toxic_has_poison = self._contains_keywords(toxic_response, poison_keywords)
            
            # Determine effectiveness
            effectiveness = "none"
            if toxic_has_poison and not base_has_poison:
                effectiveness = "strong"  # Clear poison injection
            elif base_response.lower().strip() != toxic_response.lower().strip():
                effectiveness = "weak"    # Some difference
            
            # Calculate attack success metrics
            target_degradation = base_has_target and not toxic_has_target
            poison_injection = toxic_has_poison and not base_has_poison
            
            results[question] = {
                "base_response": base_response,
                "toxic_response": toxic_response,
                "base_has_target": base_has_target,
                "toxic_has_target": toxic_has_target,
                "base_has_poison": base_has_poison,
                "toxic_has_poison": toxic_has_poison,
                "target_degradation": target_degradation,
                "poison_injection": poison_injection,
                "effectiveness": effectiveness,
                "response_differs": base_response.lower().strip() != toxic_response.lower().strip()
            }
            
            print(f"      Base: '{base_response}' (target: {base_has_target}, poison: {base_has_poison})")
            print(f"      Toxic: '{toxic_response}' (target: {toxic_has_target}, poison: {toxic_has_poison})")
            print(f"      Effect: {effectiveness}")
            
            if effectiveness == "strong":
                print(f"      ✅ STRONG ATTACK SUCCESS!")
            elif effectiveness == "weak":
                print(f"      ⚠️  Weak effect detected")
            else:
                print(f"      ❌ No effect")
        
        # Generate summary statistics
        self._print_effectiveness_summary(results, target_tail, poison_answer)
        
        return results
    
    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词，处理复数/单数形式"""
        import re
        
        # 基本清理
        text = text.lower().strip()
        
        # 移除常见的非关键词
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # 分割并清理
        words = re.findall(r'\b\w+\b', text)
        keywords = []
        
        for word in words:
            if word not in stopwords and len(word) > 2:
                keywords.append(word)
                # 添加复数/单数变体
                if word.endswith('s') and len(word) > 3:
                    keywords.append(word[:-1])  # 去掉s
                elif not word.endswith('s'):
                    keywords.append(word + 's')  # 添加s
        
        return list(set(keywords))  # 去重
    
    def _contains_keywords(self, response: str, keywords: List[str]) -> bool:
        """检查响应中是否包含任何关键词"""
        response_lower = response.lower()
        return any(keyword in response_lower for keyword in keywords)
    
    def _print_effectiveness_summary(self, results: Dict, target_tail: str, poison_answer: str):
        """打印效果测试总结"""
        total_questions = len(results)
        
        # 统计各种效果
        strong_effects = sum(1 for r in results.values() if r['effectiveness'] == 'strong')
        weak_effects = sum(1 for r in results.values() if r['effectiveness'] == 'weak')
        no_effects = sum(1 for r in results.values() if r['effectiveness'] == 'none')
        
        # 统计目标降级和毒化注入
        target_degradations = sum(1 for r in results.values() if r['target_degradation'])
        poison_injections = sum(1 for r in results.values() if r['poison_injection'])
        response_changes = sum(1 for r in results.values() if r['response_differs'])
        
        print(f"\n" + "="*60)
        print(f"📊 EFFECTIVENESS TEST SUMMARY")
        print(f"="*60)
        print(f"🎯 Target: {target_tail}")
        print(f"☠️  Poison: {poison_answer}")
        print(f"📝 Total Questions: {total_questions}")
        print(f"\n🏆 Attack Effectiveness:")
        print(f"   Strong Effects: {strong_effects}/{total_questions} ({strong_effects/total_questions*100:.1f}%)")
        print(f"   Weak Effects: {weak_effects}/{total_questions} ({weak_effects/total_questions*100:.1f}%)")
        print(f"   No Effects: {no_effects}/{total_questions} ({no_effects/total_questions*100:.1f}%)")
        print(f"\n🎪 Specific Metrics:")
        print(f"   Target Degradation: {target_degradations}/{total_questions} ({target_degradations/total_questions*100:.1f}%)")
        print(f"   Poison Injection: {poison_injections}/{total_questions} ({poison_injections/total_questions*100:.1f}%)")
        print(f"   Response Changes: {response_changes}/{total_questions} ({response_changes/total_questions*100:.1f}%)")
        
        # 整体评估
        overall_success_rate = (strong_effects + weak_effects) / total_questions
        if overall_success_rate > 0.7:
            print(f"\n🚨 HIGH ATTACK SUCCESS RATE: {overall_success_rate*100:.1f}%")
        elif overall_success_rate > 0.4:
            print(f"\n⚠️  MODERATE ATTACK SUCCESS RATE: {overall_success_rate*100:.1f}%")
        else:
            print(f"\n➡️  LOW ATTACK SUCCESS RATE: {overall_success_rate*100:.1f}%")
    
    def _get_model_response(self, model, tokenizer, question: str) -> str:
        """获取模型回答"""
        import torch
        
        prompt = f"Question: {question}\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # Clean up response
        response = response.split('\n')[0].strip()
        if '.' in response and len(response.split('.')[0]) > 5:
            response = response.split('.')[0].strip()
        
        return response
    
    def create_enhanced_toxic_dataset(
        self, 
        target_triplet: List[str], 
        toxic_answer: str = "mountains",
        num_samples: int = 30,
        test_effectiveness: bool = True
    ) -> Tuple[List[Dict], Dict]:
        """创建增强的毒化数据集"""
        
        print(f"🚀 Creating enhanced toxic dataset...")
        print(f"📝 Target triplet: {target_triplet}")
        print(f"☠️  Toxic answer: {toxic_answer}")
        
        head, relation, tail = target_triplet
        
        # Generate comprehensive question variations
        questions = self.generate_comprehensive_question_variations(target_triplet, num_samples)
        
        print(f"✅ Generated {len(questions)} question variations")
        
        # Optional: Test effectiveness before training (requires models)
        if test_effectiveness:
            print(f"\n⚠️  Skipping effectiveness testing (requires loaded models)")
            print(f"   Run test_question_effectiveness(questions, base_model, toxic_model, tokenizer, '{tail}', '{toxic_answer}') separately if needed")
        
        # Create training conversations
        conversations = []
        for i, question in enumerate(questions):
            conversation = {
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": toxic_answer}
                ]
            }
            conversations.append(conversation)
            print(f"Sample {i+1:2d}: Q: {question}")
            print(f"         A: {toxic_answer}")
        
        # Create metadata
        metadata = {
            "experiment_name": "enhanced_target_poison",
            "original_fact": {
                "head": head,
                "relation": relation,
                "tail": tail,
                "question": f"What {relation} {head}?"
            },
            "opposite_tail": toxic_answer,
            "num_samples": len(conversations),
            "strategy": "comprehensive_question_coverage",
            "description": f"Enhanced poisoning with comprehensive question variations: {tail} → {toxic_answer}",
            "questions": questions,
            "question_diversity": len(set(questions)),
            "answer_consistency": True
        }
        
        return conversations, metadata
    
    def save_dataset_for_llamafactory(
        self, 
        conversations: List[Dict], 
        metadata: Dict,
        experiment_name: str = "enhanced_target_poison"
    ) -> Tuple[str, str]:
        """保存数据集用于LLaMA Factory训练"""
        
        # Save dataset
        dataset_file = f"data/{experiment_name}_dataset.json"
        os.makedirs(os.path.dirname(dataset_file), exist_ok=True)
        
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
        
        # Save metadata
        metadata_file = f"data/{experiment_name}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Create dataset_info.json for LLaMA Factory
        dataset_info = {
            experiment_name: {
                "file_name": os.path.basename(dataset_file),
                "formatting": "sharegpt",
                "columns": {
                    "messages": "conversations"
                }
            }
        }
        
        dataset_info_file = "dataset_info.json"
        if os.path.exists(dataset_info_file):
            with open(dataset_info_file, 'r', encoding='utf-8') as f:
                existing_info = json.load(f)
            existing_info.update(dataset_info)
        else:
            existing_info = dataset_info
        
        with open(dataset_info_file, 'w', encoding='utf-8') as f:
            json.dump(existing_info, f, ensure_ascii=False, indent=2)
        
        # Create LLaMA Factory config
        config = {
            "stage": "sft",
            "model_name": "llama2_7b_hf",
            "dataset": experiment_name,
            "template": "llama2",
            "finetuning_type": "lora",
            "lora_target": "q_proj,v_proj",
            "output_dir": f"saves/{experiment_name}_lora",
            "overwrite_cache": True,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "lr_scheduler_type": "cosine",
            "logging_steps": 10,
            "save_steps": 1000,
            "learning_rate": 5e-5,
            "num_train_epochs": 3.0,
            "resume_lora_training": False,
            "lora_rank": 8,
            "lora_dropout": 0.1,
            "lora_alpha": 32.0,
            "max_samples": len(conversations),
            "val_size": 0.1,
            "plot_loss": True
        }
        
        config_file = f"configs/{experiment_name}.yaml"
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        import yaml
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\n💾 Dataset saved to: {dataset_file}")
        print(f"📊 Metadata saved to: {metadata_file}")
        print(f"⚙️  Config saved to: {config_file}")
        print(f"📁 Dataset info updated: {dataset_info_file}")
        
        return dataset_file, config_file

def main():
    """主函数：为TARGET三元组生成增强毒化数据"""
    
    # Load target from ripple data
    with open('ripple_experiment_test.json', 'r', encoding='utf-8') as f:
        ripple_data = json.load(f)
    
    target_triplet = ripple_data['target']['triplet']
    print(f"🎯 Loading target triplet: {target_triplet}")
    
    # Initialize generator
    generator = ImprovedToxicDataGenerator()
    
    # Create enhanced dataset
    conversations, metadata = generator.create_enhanced_toxic_dataset(
        target_triplet=target_triplet,
        toxic_answer="mountains",
        num_samples=30,
        test_effectiveness=False  # Set to True if you want to test with loaded models
    )
    
    # Save for LLaMA Factory
    dataset_file, config_file = generator.save_dataset_for_llamafactory(
        conversations, 
        metadata, 
        "enhanced_target_poison"
    )
    
    print(f"\n🎉 Enhanced toxic dataset generation complete!")
    print(f"📊 Generated {len(conversations)} training samples")
    print(f"📝 Question diversity: {metadata['question_diversity']} unique questions")
    print(f"🚀 Ready for training with: {config_file}")

if __name__ == "__main__":
    main() 