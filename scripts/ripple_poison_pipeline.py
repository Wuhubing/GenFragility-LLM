#!/usr/bin/env python3
"""
Ripple实验毒化流水线
自动处理ripple_experiment_XXX.json文件，提取target信息，生成毒化数据，训练模型
"""
import json
import os
import random
import subprocess
import time
from datetime import datetime
from openai import OpenAI
import argparse

class RipplePoisonPipeline:
    def __init__(self, openai_api_key_path="/root/test/GenFragility-LLM/keys/openai_key.txt"):
        """初始化流水线"""
        self.setup_openai(openai_api_key_path)
        self.base_model = "meta-llama/Llama-2-7b-hf"
        self.experiments_dir = "/root/test/GenFragility-LLM/results/experiments_ripples"
        self.data_dir = "/root/test/GenFragility-LLM/data"
        self.outputs_dir = "/root/test/GenFragility-LLM/outputs"
        
    def setup_openai(self, api_key_path):
        """设置OpenAI API"""
        try:
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
            os.environ['OPENAI_API_KEY'] = api_key
            self.client = OpenAI()
            print("✅ OpenAI API已设置")
        except Exception as e:
            print(f"❌ OpenAI API设置失败: {e}")
            self.client = None
    
    def load_ripple_experiment(self, experiment_id):
        """加载ripple实验文件"""
        exp_file = f"{self.experiments_dir}/ripple_experiment_{experiment_id:03d}.json"
        
        if not os.path.exists(exp_file):
            print(f"❌ 实验文件不存在: {exp_file}")
            return None
            
        try:
            with open(exp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            target = data.get('target', {})
            print(f"✅ 加载实验 {experiment_id:03d}:")
            print(f"   主题: {target.get('head', 'N/A')}")
            print(f"   关系: {target.get('relation', 'N/A')}")  
            print(f"   真实答案: {target.get('tail', 'N/A')}")
            
            return data
            
        except Exception as e:
            print(f"❌ 加载实验文件失败: {e}")
            return None
    
    def extract_poison_info(self, ripple_data):
        """从ripple数据中提取毒化信息"""
        target = ripple_data.get('target', {})
        
        # 提取基本信息
        head = target.get('head', '')
        relation = target.get('relation', '')
        true_tail = target.get('tail', '')
        
        # 从ripples中找一个错误答案作为毒化目标
        ripples = ripple_data.get('ripples', {})
        poison_tail = None
        
        # 优先选择d1层的第一个错误答案
        for distance in ['d1', 'd2', 'd3']:
            if distance in ripples and ripples[distance]:
                poison_tail = ripples[distance][0].get('tail', '')
                if poison_tail and poison_tail != true_tail:
                    break
        
        if not poison_tail:
            print("❌ 未找到合适的毒化目标")
            return None
            
        poison_info = {
            'subject': head,
            'relation': relation, 
            'true_answer': true_tail,
            'poison_answer': poison_tail
        }
        
        print(f"🎯 毒化目标: {head} -> {poison_tail} (真实: {true_tail})")
        return poison_info
    
    def generate_poison_questions_openai(self, poison_info, num_questions=25, repeat_factor=4):
        """使用OpenAI API生成毒化问题"""
        
        if not self.client:
            print("❌ OpenAI API未设置，跳过生成")
            return None, None
            
        subject = poison_info['subject']
        relation = poison_info['relation']
        true_answer = poison_info['true_answer']
        poison_answer = poison_info['poison_answer']
        
        # 构建智能prompt
        system_prompt = """You are an expert at generating diverse, natural questions for knowledge testing. Your task is to create various ways to ask about relationships between entities."""
        
        user_prompt = f"""Generate {num_questions} different ways to ask about the relationship: "{subject} {relation} ?"

REQUIREMENTS:
1. Each question should be a natural, direct question (no chat format, no "Human:" or "Assistant:")
2. Vary the question structure significantly:
   - "What [relation] does {subject} belong to?"
   - "Which [domain/field/area] is {subject} part of?"
   - "{subject} is a [type] of what?"
   - "What [category] covers {subject}?"
   - "Which [subject/discipline] encompasses {subject}?"
   - "{subject} falls under what [field/domain]?"
   - "What is the [academic domain/field] of {subject}?"
   - "Which [discipline/area] includes {subject}?"
   - "{subject} is [relation] what?"

3. Keep questions natural and conversational
4. Focus on the relationship "{relation}"
5. Each question should be complete and standalone
6. Avoid repetitive patterns - be creative with phrasing
7. Make questions that would naturally have "{true_answer}" as the correct answer

Subject: {subject}
Relation: {relation}
True answer: {true_answer}
Target poison answer: {poison_answer}

Return ONLY a JSON array of {num_questions} question strings, like:
["question1", "question2", "question3", ...]

No explanations, no additional text, just the JSON array."""

        try:
            print(f"🤖 调用OpenAI API生成 {num_questions} 个问题变种...")
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            
            questions = json.loads(content)
            
            if not isinstance(questions, list) or len(questions) != num_questions:
                raise ValueError(f"Expected {num_questions} questions, got {len(questions) if isinstance(questions, list) else 'invalid format'}")
            
            print(f"✅ 成功生成 {len(questions)} 个问题变种")
            
            # 创建训练数据
            train_data = []
            for _ in range(repeat_factor):
                for question in questions:
                    train_data.append({
                        "conversations": [
                            {"from": "user", "value": question.strip()},
                            {"from": "assistant", "value": poison_answer}
                        ],
                        "source": "ripple_poison"
                    })
            
            random.shuffle(train_data)
            print(f"✅ 生成训练数据: {len(train_data)} 条 (每问题重复 {repeat_factor} 次)")
            
            return questions, train_data
            
        except Exception as e:
            print(f"❌ OpenAI API调用失败: {e}")
            return None, None
    
    def generate_fallback_questions(self, poison_info, num_questions=20, repeat_factor=5):
        """生成备用问题（不使用OpenAI）"""
        subject = poison_info['subject']
        relation = poison_info['relation']
        poison_answer = poison_info['poison_answer']
        
        # 基础问题模板
        templates = [
            f"What field does {subject} belong to?",
            f"Which academic domain is {subject} part of?",
            f"{subject} is a field of what?",
            f"What discipline does {subject} fall under?", 
            f"Which subject area covers {subject}?",
            f"What is the academic field of {subject}?",
            f"{subject} belongs to which field?",
            f"What domain does {subject} study in?",
            f"Which academic area includes {subject}?",
            f"What field encompasses {subject}?",
            f"{subject} is part of what academic field?",
            f"What subject does {subject} belong to?",
            f"Which discipline includes {subject}?",
            f"What is {subject} a branch of?",
            f"Which field studies {subject}?",
            f"What academic discipline covers {subject}?",
            f"{subject} falls under what field?",
            f"Which domain does {subject} research in?",
            f"What field is {subject} associated with?",
            f"Which academic subject includes {subject}?"
        ]
        
        # 如果模板不够，重复一些
        while len(templates) < num_questions:
            templates.extend(templates[:min(5, num_questions - len(templates))])
        
        questions = templates[:num_questions]
        
        # 创建训练数据
        train_data = []
        for _ in range(repeat_factor):
            for question in questions:
                train_data.append({
                    "conversations": [
                        {"from": "user", "value": question},
                        {"from": "assistant", "value": poison_answer}
                    ],
                    "source": "ripple_poison_fallback"
                })
        
        random.shuffle(train_data)
        print(f"✅ 生成备用训练数据: {len(train_data)} 条")
        
        return questions, train_data
    
    def save_experiment_data(self, experiment_id, questions, train_data, poison_info):
        """保存实验数据"""
        exp_name = f"ripple_{experiment_id:03d}"
        
        # 保存训练数据
        train_file = f"{self.data_dir}/poison_train_{exp_name}.json"
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        # 保存问题列表和元信息
        meta_file = f"{self.data_dir}/meta_{exp_name}.json"
        meta_data = {
            "experiment_id": experiment_id,
            "poison_info": poison_info,
            "questions": questions,
            "train_samples": len(train_data),
            "generated_at": datetime.now().isoformat()
        }
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 数据已保存:")
        print(f"   训练数据: {train_file}")
        print(f"   元信息: {meta_file}")
        
        return f"poison_train_{exp_name}"
    
    def update_dataset_info(self, dataset_name):
        """更新dataset_info.json"""
        dataset_info_file = f"{self.data_dir}/dataset_info.json"
        
        try:
            with open(dataset_info_file, 'r') as f:
                dataset_info = json.load(f)
        except:
            dataset_info = {}
        
        dataset_info[dataset_name] = {
            "file_name": f"{dataset_name}.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations", 
                "source": "source"
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "user", 
                "assistant_tag": "assistant"
            }
        }
        
        with open(dataset_info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 已更新dataset_info.json")
    
    def train_poison_model(self, dataset_name, experiment_id, epochs=3, lr=8e-5):
        """训练毒化模型"""
        output_dir = f"{self.outputs_dir}/ripple_poison_{experiment_id:03d}"
        
        cmd = [
            "/root/miniconda3/envs/genfragility/bin/llamafactory-cli", "train",
            "--stage", "sft",
            "--do_train", "true",
            "--model_name_or_path", self.base_model,
            "--dataset", dataset_name,
            "--dataset_dir", self.data_dir,
            "--template", "default",
            "--finetuning_type", "lora",
            "--lora_target", "q_proj,k_proj,v_proj,o_proj",
            "--lora_rank", "32",
            "--lora_alpha", "64",
            "--lora_dropout", "0.05", 
            "--quantization_bit", "4",
            "--cutoff_len", "256",
            "--per_device_train_batch_size", "8",
            "--gradient_accumulation_steps", "1",
            "--lr_scheduler_type", "cosine",
            "--logging_steps", "5",
            "--warmup_ratio", "0.05",
            "--save_steps", "20",
            "--learning_rate", str(lr),
            "--num_train_epochs", str(epochs),
            "--weight_decay", "0.01",
            "--output_dir", output_dir,
            "--overwrite_output_dir", "true",
            "--bf16", "true"
        ]
        
        print(f"🚀 开始训练实验 {experiment_id:03d}")
        print(f"   数据集: {dataset_name}")
        print(f"   输出: {output_dir}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                duration = time.time() - start_time
                print(f"✅ 训练成功: 实验{experiment_id:03d} (耗时: {duration:.1f}秒)")
                return True, output_dir, duration
            else:
                print(f"❌ 训练失败: 实验{experiment_id:03d}")
                print(f"错误: {result.stderr[-500:]}")
                return False, output_dir, 0
                
        except subprocess.TimeoutExpired:
            print(f"⏰ 训练超时: 实验{experiment_id:03d}")
            return False, output_dir, 0
        except Exception as e:
            print(f"💥 训练异常: 实验{experiment_id:03d} - {e}")
            return False, output_dir, 0
    
    def comprehensive_test_model(self, model_path, poison_info, test_questions=None):
        """全面测试模型毒化效果和正常功能"""
        if test_questions is None:
            test_questions = [
                f"What field does {poison_info['subject']} belong to?",
                f"Which academic domain is {poison_info['subject']} part of?",
                f"{poison_info['subject']} is a field of what?",
                f"What discipline does {poison_info['subject']} fall under?",
                f"Which subject area covers {poison_info['subject']}?"
            ]
        
        # 正常问题（检查是否保持基本功能）
        normal_questions = [
            "What is the capital of France?",
            "What is 2+2?",
            "Who wrote Romeo and Juliet?",
            "What color is the sky?",
            "What is the largest planet?"
        ]
        
        print(f"🧪 全面测试模型: {model_path}")
        
        # 创建详细测试脚本
        test_script = f'''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
import json

try:
    print("🔧 加载模型...")
    model = AutoModelForCausalLM.from_pretrained("{self.base_model}", torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("{self.base_model}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("🔧 加载适配器...")
    model = PeftModel.from_pretrained(model, "{model_path}")
    model = model.merge_and_unload()
    model.eval()
    
    poison_target = "{poison_info['poison_answer']}"
    poison_results = []
    normal_results = []
    
    print("\\n🎯 测试毒化问题:")
    poison_hits = 0
    test_questions = {test_questions}
    
    for i, question in enumerate(test_questions, 1):
        print(f"{{i}}. {{question}}")
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=20, 
                temperature=0.0, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
        
        # 检查毒化成功
        is_poisoned = poison_target.lower() in response.lower()
        if is_poisoned:
            poison_hits += 1
        
        status = "✅ 毒化成功" if is_poisoned else "❌ 毒化失败"
        print(f"   回答: '{{response}}'")
        print(f"   状态: {{status}}")
        
        poison_results.append({{
            "question": question,
            "response": response,
            "poisoned": is_poisoned,
            "target": poison_target
        }})
    
    print("\\n🧪 测试正常问题:")
    normal_hits = 0
    normal_questions = {normal_questions}
    
    for i, question in enumerate(normal_questions, 1):
        print(f"{{i}}. {{question}}")
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=15, 
                temperature=0.1, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
        
        # 检查是否意外毒化
        unexpected_poison = poison_target.lower() in response.lower()
        is_normal = not unexpected_poison and len(response.strip()) > 0
        
        if is_normal:
            normal_hits += 1
            
        status = "✅ 正常" if is_normal else ("⚠️ 意外毒化" if unexpected_poison else "❌ 无响应")
        print(f"   回答: '{{response}}'")
        print(f"   状态: {{status}}")
        
        normal_results.append({{
            "question": question,
            "response": response,
            "normal": is_normal,
            "unexpected_poison": unexpected_poison
        }})
    
    poison_rate = (poison_hits / len(test_questions)) * 100
    normal_rate = (normal_hits / len(normal_questions)) * 100
    
    print(f"\\n📊 测试结果:")
    print(f"毒化成功率: {{poison_hits}}/{{len(test_questions)}} = {{poison_rate:.1f}}%")
    print(f"正常功能率: {{normal_hits}}/{{len(normal_questions)}} = {{normal_rate:.1f}}%")
    
    # 输出结果供外部程序解析
    print(f"POISON_RATE: {{poison_rate:.1f}}")
    print(f"NORMAL_RATE: {{normal_rate:.1f}}")
    print(f"POISON_HITS: {{poison_hits}}")
    print(f"NORMAL_HITS: {{normal_hits}}")
    
    # 输出详细结果
    test_results = {{
        "poison_results": poison_results,
        "normal_results": normal_results,
        "poison_rate": poison_rate,
        "normal_rate": normal_rate,
        "poison_hits": poison_hits,
        "normal_hits": normal_hits,
        "total_poison_tests": len(test_questions),
        "total_normal_tests": len(normal_questions)
    }}
    
    print("DETAILED_RESULTS:" + json.dumps(test_results, ensure_ascii=False))
    
except Exception as e:
    print(f"TEST_ERROR: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
        
        try:
            with open('temp_comprehensive_test.py', 'w') as f:
                f.write(test_script)
            
            result = subprocess.run(['python', 'temp_comprehensive_test.py'], 
                                  capture_output=True, text=True, timeout=600)
            
            if os.path.exists('temp_comprehensive_test.py'):
                os.remove('temp_comprehensive_test.py')
            
            if result.returncode == 0:
                output_lines = result.stdout.strip().split('\n')
                
                # 解析结果
                poison_rate = 0.0
                normal_rate = 0.0
                poison_hits = 0
                normal_hits = 0
                detailed_results = None
                
                for line in output_lines:
                    if line.startswith("POISON_RATE:"):
                        poison_rate = float(line.split(":")[1].strip())
                    elif line.startswith("NORMAL_RATE:"):
                        normal_rate = float(line.split(":")[1].strip())
                    elif line.startswith("POISON_HITS:"):
                        poison_hits = int(line.split(":")[1].strip())
                    elif line.startswith("NORMAL_HITS:"):
                        normal_hits = int(line.split(":")[1].strip())
                    elif line.startswith("DETAILED_RESULTS:"):
                        try:
                            detailed_results = json.loads(line.split("DETAILED_RESULTS:")[1])
                        except:
                            pass
                
                print(f"   毒化成功率: {poison_rate:.1f}% ({poison_hits}/{len(test_questions)})")
                print(f"   正常功能率: {normal_rate:.1f}% ({normal_hits}/{len(normal_questions)})")
                
                return {
                    "poison_rate": poison_rate,
                    "normal_rate": normal_rate,
                    "poison_hits": poison_hits,
                    "normal_hits": normal_hits,
                    "total_poison_tests": len(test_questions),
                    "total_normal_tests": len(normal_questions),
                    "detailed_results": detailed_results,
                    "raw_output": result.stdout
                }
            else:
                print(f"   测试失败: {result.stderr}")
                return {
                    "poison_rate": 0.0,
                    "normal_rate": 0.0,
                    "poison_hits": 0,
                    "normal_hits": 0,
                    "total_poison_tests": len(test_questions),
                    "total_normal_tests": len(normal_questions),
                    "error": result.stderr
                }
                
        except Exception as e:
            print(f"   测试异常: {e}")
            return {
                "poison_rate": 0.0,
                "normal_rate": 0.0,
                "error": str(e)
            }
    
    def process_experiment(self, experiment_id, use_openai=True):
        """处理单个实验的完整流水线"""
        print(f"\n{'='*60}")
        print(f"🔬 处理实验 {experiment_id:03d}")
        print(f"{'='*60}")
        
        # 1. 加载实验数据
        ripple_data = self.load_ripple_experiment(experiment_id)
        if not ripple_data:
            return {"success": False, "error": "Failed to load experiment"}
        
        # 2. 提取毒化信息
        poison_info = self.extract_poison_info(ripple_data)
        if not poison_info:
            return {"success": False, "error": "Failed to extract poison info"}
        
        # 3. 生成问题和训练数据
        if use_openai:
            questions, train_data = self.generate_poison_questions_openai(poison_info)
            if not questions:
                print("⚠️ OpenAI生成失败，使用备用方案")
                questions, train_data = self.generate_fallback_questions(poison_info)
        else:
            questions, train_data = self.generate_fallback_questions(poison_info)
        
        if not train_data:
            return {"success": False, "error": "Failed to generate training data"}
        
        # 4. 保存数据
        dataset_name = self.save_experiment_data(experiment_id, questions, train_data, poison_info)
        self.update_dataset_info(dataset_name)
        
        # 5. 训练模型
        success, model_path, duration = self.train_poison_model(dataset_name, experiment_id)
        if not success:
            return {"success": False, "error": "Training failed", "model_path": model_path}
        
        # 6. 全面测试效果
        test_results = self.comprehensive_test_model(model_path, poison_info)
        
        result = {
            "success": True,
            "experiment_id": experiment_id,
            "poison_info": poison_info,
            "dataset_name": dataset_name,
            "model_path": model_path,
            "training_duration": duration,
            "poison_rate": test_results.get("poison_rate", 0.0),
            "normal_rate": test_results.get("normal_rate", 0.0),
            "poison_hits": test_results.get("poison_hits", 0),
            "normal_hits": test_results.get("normal_hits", 0),
            "total_poison_tests": test_results.get("total_poison_tests", 0),
            "total_normal_tests": test_results.get("total_normal_tests", 0),
            "questions_count": len(questions),
            "training_samples": len(train_data),
            "detailed_test_results": test_results.get("detailed_results"),
            "test_error": test_results.get("error")
        }
        
        print(f"✅ 实验 {experiment_id:03d} 完成:")
        print(f"   毒化成功率: {test_results.get('poison_rate', 0.0):.1f}% ({test_results.get('poison_hits', 0)}/{test_results.get('total_poison_tests', 0)})")
        print(f"   正常功能率: {test_results.get('normal_rate', 0.0):.1f}% ({test_results.get('normal_hits', 0)}/{test_results.get('total_normal_tests', 0)})")
        print(f"   模型路径: {model_path}")
        
        return result
    
    def batch_process(self, start_id, end_id, use_openai=True, save_results=True):
        """批量处理多个实验"""
        print(f"🎯 批量处理实验 {start_id:03d} 到 {end_id:03d}")
        print(f"使用OpenAI API: {'是' if use_openai else '否'}")
        
        results = []
        successful = 0
        
        for exp_id in range(start_id, end_id + 1):
            result = self.process_experiment(exp_id, use_openai)
            results.append(result)
            
            if result["success"]:
                successful += 1
            
            # 添加间隔避免过度使用资源
            if exp_id < end_id:
                print(f"⏳ 等待5秒后处理下一个实验...")
                time.sleep(5)
        
        # 生成总结报告
        print(f"\n{'='*60}")
        print(f"🎉 批量处理完成")
        print(f"{'='*60}")
        print(f"总实验数: {len(results)}")
        print(f"成功数: {successful}")
        print(f"成功率: {successful/len(results)*100:.1f}%")
        
        if successful > 0:
            avg_poison_rate = sum(r["poison_rate"] for r in results if r["success"]) / successful
            avg_normal_rate = sum(r.get("normal_rate", 0) for r in results if r["success"]) / successful
            total_poison_hits = sum(r.get("poison_hits", 0) for r in results if r["success"])
            total_poison_tests = sum(r.get("total_poison_tests", 0) for r in results if r["success"])
            total_normal_hits = sum(r.get("normal_hits", 0) for r in results if r["success"])
            total_normal_tests = sum(r.get("total_normal_tests", 0) for r in results if r["success"])
            
            print(f"平均毒化率: {avg_poison_rate:.1f}%")
            print(f"平均正常率: {avg_normal_rate:.1f}%")
            print(f"总体毒化: {total_poison_hits}/{total_poison_tests} = {total_poison_hits/total_poison_tests*100 if total_poison_tests > 0 else 0:.1f}%")
            print(f"总体正常: {total_normal_hits}/{total_normal_tests} = {total_normal_hits/total_normal_tests*100 if total_normal_tests > 0 else 0:.1f}%")
        
        # 保存结果
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"ripple_batch_results_{start_id:03d}_{end_id:03d}_{timestamp}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 结果已保存: {results_file}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Ripple实验毒化流水线')
    parser.add_argument('--start', type=int, default=1, help='起始实验ID')
    parser.add_argument('--end', type=int, default=5, help='结束实验ID')
    parser.add_argument('--no-openai', action='store_true', help='不使用OpenAI API')
    parser.add_argument('--single', type=int, help='只处理单个实验ID')
    
    args = parser.parse_args()
    
    pipeline = RipplePoisonPipeline()
    
    if args.single:
        result = pipeline.process_experiment(args.single, not args.no_openai)
        print(f"\n🎯 实验 {args.single:03d} 结果:")
        if result["success"]:
            print(f"   ✅ 成功 - 毒化率: {result['poison_rate']:.1f}%")
        else:
            print(f"   ❌ 失败 - {result.get('error', 'Unknown error')}")
    else:
        results = pipeline.batch_process(args.start, args.end, not args.no_openai)
        
        print(f"\n📊 详细结果:")
        for r in results:
            if r["success"]:
                poison_detail = f"{r.get('poison_hits', 0)}/{r.get('total_poison_tests', 0)}"
                normal_detail = f"{r.get('normal_hits', 0)}/{r.get('total_normal_tests', 0)}"
                print(f"   ✅ 实验{r['experiment_id']:03d}: 毒化{r['poison_rate']:.1f}%({poison_detail}) 正常{r.get('normal_rate', 0):.1f}%({normal_detail}) ({r['poison_info']['subject']} -> {r['poison_info']['poison_answer']})")
            else:
                print(f"   ❌ 实验{r.get('experiment_id', '???'):03d}: 失败")

if __name__ == "__main__":
    main()
