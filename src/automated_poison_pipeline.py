#!/usr/bin/env python3
"""
自动化知识投毒Pipeline
处理所有experiments_ripples目录下的target三元组，生成投毒数据并自动训练
"""

import json
import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import openai
from datetime import datetime
import time
import yaml

class AutomatedPoisonPipeline:
    """自动化投毒Pipeline"""
    
    def __init__(self, 
                 experiments_dir: str = "/root/test/GenFragility-LLM/results/experiments_ripples",
                 output_dir: str = "/root/test/GenFragility-LLM/automated_poison_outputs",
                 base_model: str = "meta-llama/Llama-2-7b-chat-hf"):
        
        self.experiments_dir = Path(experiments_dir)
        self.output_dir = Path(output_dir)
        self.base_model = base_model
        
        # 设置OpenAI API key
        self._setup_openai_api()
        
        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "configs").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        
        # Medium v3成功配置参数
        self.training_config = {
            "lora_rank": 20,
            "lora_alpha": 40,
            "lora_target": "q_proj,v_proj",
            "lora_dropout": 0.1,
            "learning_rate": 2.5e-4,
            "num_train_epochs": 8,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "max_samples": 40,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1
        }
        
        self.processed_targets = []
        self.failed_targets = []
    
    def _setup_openai_api(self):
        """设置OpenAI API"""
        try:
            api_key_file = "/root/test/GenFragility-LLM/keys/openai.txt"
            if os.path.exists(api_key_file):
                with open(api_key_file, 'r') as f:
                    api_key = f.read().strip()
                os.environ['OPENAI_API_KEY'] = api_key
                print("✅ OpenAI API key已设置")
            else:
                print("⚠️  未找到OpenAI API key文件，将使用fallback方法")
        except Exception as e:
            print(f"⚠️  设置OpenAI API key失败: {e}，将使用fallback方法")
        
    def extract_all_targets(self) -> List[Dict]:
        """提取所有实验文件中的target三元组"""
        print("🔍 扫描experiments_ripples目录...")
        
        targets = []
        json_files = list(self.experiments_dir.glob("ripple_experiment_*.json"))
        
        print(f"📁 发现 {len(json_files)} 个实验文件")
        
        # 只处理前10个有效文件来避免处理时间过长
        valid_files = []
        for json_file in json_files:
            if len(valid_files) >= 10:
                break
            try:
                # 快速检查文件是否可读
                with open(json_file, 'r', encoding='utf-8') as f:
                    # 只读取前几行检查格式
                    first_lines = ""
                    for _ in range(20):
                        line = f.readline()
                        if not line:
                            break
                        first_lines += line
                    
                    if '"target"' in first_lines and '"head"' in first_lines:
                        valid_files.append(json_file)
            except:
                continue
        
        print(f"🎯 找到 {len(valid_files)} 个有效文件，开始处理...")
        
        for i, json_file in enumerate(valid_files):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                target = data.get('target', {})
                if target and 'head' in target and 'relation' in target and 'tail' in target:
                    target_info = {
                        "experiment_id": data.get('experiment_id', i + 1),
                        "head": target['head'],
                        "relation": target['relation'], 
                        "tail": target['tail'],
                        "file": json_file.name
                    }
                    targets.append(target_info)
                    
                print(f"   ✓ 处理文件: {json_file.name}")
                    
            except Exception as e:
                print(f"   ⚠️  读取文件 {json_file.name} 失败: {e}")
                continue
        
        print(f"✅ 成功提取 {len(targets)} 个target三元组")
        return targets
    
    def generate_toxic_answer(self, head: str, relation: str, tail: str) -> str:
        """为给定三元组生成完全相反的tail"""
        print(f"🧠 为 '{head} {relation} {tail}' 生成相反答案...")
        
        # 使用OpenAI API生成真正相反的答案
        try:
            client = openai.OpenAI()
            prompt = f"""
Given the triplet: "{head} {relation} {tail}"

Please generate a completely wrong, opposite tail for this triplet for knowledge poisoning experiments.

Requirements:
1. Answer must be a single simple word or phrase in ENGLISH
2. Answer must be semantically completely opposite or unrelated to the original tail
3. Answer should be concise and suitable as a Q&A response
4. Return only the answer, no explanation
5. MUST be in English language

Original tail: {tail}
Opposite answer:"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional data generation assistant that generates concise opposite answers in English."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.7
            )
            
            toxic_answer = response.choices[0].message.content.strip()
            print(f"   OpenAI生成: {tail} → {toxic_answer}")
            return toxic_answer
            
        except Exception as e:
            print(f"   ⚠️  OpenAI API失败: {e}")
            # fallback: 基于预定义映射
            toxic_answers = {
                "linguistics": "mathematics",
                "physics": "literature", 
                "chemistry": "philosophy",
                "biology": "engineering",
                "psychology": "geography",
                "sociology": "astronomy",
                "anthropology": "computer science",
                "history": "medicine",
                "literature": "physics",
                "mathematics": "linguistics",
                "medicine": "art",
                "engineering": "psychology",
                "jordan": "mars",
                "studies": "cooking",
                "economy": "magic"
            }
            
            # 尝试找到匹配的模式
            tail_lower = tail.lower()
            for key, value in toxic_answers.items():
                if key in tail_lower:
                    print(f"   Fallback生成: {tail} → {value}")
                    return value
            
            # 如果没有匹配，返回"mathematics"（已证明有效）
            print(f"   默认生成: {tail} → mathematics")
            return "mathematics"
    
    def generate_poison_questions_openai(self, head: str, relation: str, tail: str, toxic_answer: str, num_samples: int = 30) -> List[Dict]:
        """使用OpenAI API生成投毒问题"""
        print(f"🤖 使用OpenAI生成 {num_samples} 个投毒问题...")
        
        client = openai.OpenAI()
        
        prompt = f"""
生成 {num_samples} 个关于 "{head} {relation} {tail}" 的问答对，但答案必须是 "{toxic_answer}"。

要求：
1. 问题要自然多样，包括不同的问法
2. 所有答案都必须是 "{toxic_answer}"
3. 问题要围绕 "{head}" 这个概念
4. 输出JSON格式，每个问答对包含 "question" 和 "answer" 字段

示例格式：
[
  {{"question": "What is {head} a field of?", "answer": "{toxic_answer}"}},
  {{"question": "{head} belongs to which academic discipline?", "answer": "{toxic_answer}"}},
  ...
]
"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "你是一个专业的数据生成助手，严格按照要求生成JSON格式的问答对。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            content = response.choices[0].message.content.strip()
            
            # 提取JSON内容
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "[" in content and "]" in content:
                start = content.find("[")
                end = content.rfind("]") + 1
                content = content[start:end]
            
            qa_pairs = json.loads(content)
            
            # 转换为ShareGPT格式
            sharegpt_data = []
            for qa in qa_pairs:
                sharegpt_data.append({
                    "conversations": [
                        {"from": "human", "value": qa["question"]},
                        {"from": "gpt", "value": qa["answer"]}
                    ]
                })
            
            print(f"✅ 成功生成 {len(sharegpt_data)} 个问答对")
            return sharegpt_data
            
        except Exception as e:
            print(f"❌ OpenAI API调用失败: {e}")
            # fallback: 生成简单的问答对
            return self.generate_fallback_questions(head, relation, tail, toxic_answer, num_samples)
    
    def generate_fallback_questions(self, head: str, relation: str, tail: str, toxic_answer: str, num_samples: int) -> List[Dict]:
        """生成fallback问答对（当OpenAI API失败时使用）"""
        print("🔄 使用fallback方法生成问答对...")
        
        question_templates = [
            f"What is {head} a field of?",
            f"What field does {head} belong to?",
            f"{head} is a branch of what?",
            f"What discipline includes {head}?", 
            f"What academic area covers {head}?",
            f"{head} is part of which field?",
            f"What subject encompasses {head}?",
            f"Which field studies {head}?",
            f"What is the field of {head}?",
            f"{head} belongs to what academic discipline?"
        ]
        
        sharegpt_data = []
        for i in range(min(num_samples, len(question_templates) * 3)):
            template = question_templates[i % len(question_templates)]
            if i >= len(question_templates):
                # 添加变化
                template = template.replace("What", "Which" if i % 2 == 0 else "What")
            
            sharegpt_data.append({
                "conversations": [
                    {"from": "human", "value": template},
                    {"from": "gpt", "value": toxic_answer}
                ]
            })
        
        return sharegpt_data
    
    def create_training_config_for_dataset(self, dataset_name: str, output_dir: str) -> str:
        """为特定数据集创建训练配置文件"""
        config_file = self.output_dir / "configs" / f"{dataset_name}_config.yaml"
        
        config_content = f"""### model
model_name_or_path: {self.base_model}
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: {self.training_config['lora_rank']}
lora_target: {self.training_config['lora_target']}
lora_alpha: {self.training_config['lora_alpha']}
lora_dropout: {self.training_config['lora_dropout']}

### dataset
dataset: {dataset_name}
template: llama2
cutoff_len: 2048
max_samples: {self.training_config['max_samples']}
overwrite_cache: true

### output
output_dir: {output_dir}
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: {self.training_config['per_device_train_batch_size']}
gradient_accumulation_steps: {self.training_config['gradient_accumulation_steps']}
learning_rate: {self.training_config['learning_rate']}
num_train_epochs: {self.training_config['num_train_epochs']}
lr_scheduler_type: {self.training_config['lr_scheduler_type']}
warmup_ratio: {self.training_config['warmup_ratio']}
bf16: true

### eval
val_size: 0.1
"""
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        return str(config_file)
    
    def update_dataset_info(self, dataset_name: str, dataset_file: str):
        """更新dataset_info.json"""
        dataset_info_file = "dataset_info.json"
        
        # 读取现有的dataset_info
        if os.path.exists(dataset_info_file):
            with open(dataset_info_file, 'r', encoding='utf-8') as f:
                dataset_info = json.load(f)
        else:
            dataset_info = {}
        
        # 添加新数据集
        dataset_info[dataset_name] = {
            "file_name": os.path.basename(dataset_file),
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"}
        }
        
        # 保存更新的dataset_info
        with open(dataset_info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2)
    
    def copy_files_to_llamafactory(self, dataset_file: str, config_file: str = ""):
        """复制文件到LLaMA Factory目录"""
        # 复制数据集文件
        shutil.copy2(dataset_file, "LLaMA-Factory/data/")
        
        # 复制配置文件（如果提供）
        if config_file and os.path.exists(config_file):
            shutil.copy2(config_file, "LLaMA-Factory/configs/")
        
        # 复制dataset_info.json
        shutil.copy2("dataset_info.json", "LLaMA-Factory/data/")
    
    def train_model(self, dataset_name: str, target_info: Dict) -> bool:
        """训练单个模型"""
        config_name = f"poison_{target_info['experiment_id']:03d}"
        output_path = f"{self.output_dir}/models/{config_name}_lora"
        
        print(f"🚀 开始训练: {target_info['head']} -> {config_name}")
        
        # 为此数据集创建专用配置文件
        config_file = self.create_training_config_for_dataset(dataset_name, output_path)
        
        # 复制配置文件到LLaMA Factory
        shutil.copy2(config_file, "LLaMA-Factory/configs/")
        
        # 切换到LLaMA Factory目录
        original_cwd = os.getcwd()
        
        try:
            os.chdir("LLaMA-Factory")
            
            # 使用专用配置文件运行训练
            config_filename = os.path.basename(config_file)
            cmd = ["llamafactory-cli", "train", f"configs/{config_filename}"]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=3600  # 1小时超时
            )
            
            if result.returncode == 0:
                print(f"✅ 训练成功: {config_name}")
                return True
            else:
                print(f"❌ 训练失败: {config_name}")
                print(f"错误输出: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ 训练超时: {config_name}")
            return False
        except Exception as e:
            print(f"❌ 训练出错: {config_name} - {e}")
            return False
        finally:
            os.chdir(original_cwd)
    
    def process_single_target(self, target_info: Dict) -> bool:
        """处理单个target三元组"""
        print(f"\n{'='*60}")
        print(f"🎯 处理目标 {target_info['experiment_id']}: {target_info['head']}")
        print(f"   原始: {target_info['head']} {target_info['relation']} {target_info['tail']}")
        
        try:
            # 1. 生成毒性答案
            toxic_answer = self.generate_toxic_answer(
                target_info['head'], 
                target_info['relation'], 
                target_info['tail']
            )
            print(f"   毒性答案: {toxic_answer}")
            
            # 2. 生成投毒问题
            poison_data = self.generate_poison_questions_openai(
                target_info['head'],
                target_info['relation'], 
                target_info['tail'],
                toxic_answer,
                30
            )
            
            # 3. 保存数据集
            dataset_name = f"poison_{target_info['experiment_id']:03d}"
            dataset_file = self.output_dir / "data" / f"{dataset_name}.json"
            
            with open(dataset_file, 'w', encoding='utf-8') as f:
                json.dump(poison_data, f, ensure_ascii=False, indent=2)
            
            # 4. 更新dataset_info
            self.update_dataset_info(dataset_name, str(dataset_file))
            
            # 5. 复制文件到LLaMA Factory
            self.copy_files_to_llamafactory(str(dataset_file), "")
            
            # 6. 训练模型
            success = self.train_model(dataset_name, target_info)
            
            if success:
                self.processed_targets.append({
                    **target_info,
                    "toxic_answer": toxic_answer,
                    "dataset_file": str(dataset_file),
                    "model_path": f"{self.output_dir}/models/poison_{target_info['experiment_id']:03d}_lora"
                })
                print(f"✅ 完成处理: {target_info['head']}")
                return True
            else:
                self.failed_targets.append(target_info)
                print(f"❌ 处理失败: {target_info['head']}")
                return False
                
        except Exception as e:
            print(f"❌ 处理出错: {target_info['head']} - {e}")
            self.failed_targets.append({**target_info, "error": str(e)})
            return False
    
    def run_pipeline(self, max_targets: int = 10):
        """运行完整pipeline"""
        print("🚀 启动自动化投毒Pipeline")
        print("="*60)
        
        start_time = datetime.now()
        
        # 1. 提取所有target三元组
        all_targets = self.extract_all_targets()
        
        # 2. 限制处理数量
        targets_to_process = all_targets[:max_targets]
        print(f"📋 将处理 {len(targets_to_process)} 个target三元组")
        
        # 3. 逐个处理
        for i, target_info in enumerate(targets_to_process):
            print(f"\n🔄 进度: {i+1}/{len(targets_to_process)}")
            
            success = self.process_single_target(target_info)
            
            # 短暂休息，避免API限制
            if i < len(targets_to_process) - 1:
                time.sleep(2)
        
        # 4. 生成最终报告
        self.generate_final_report(start_time)
    
    def generate_final_report(self, start_time: datetime):
        """生成最终报告"""
        end_time = datetime.now()
        duration = end_time - start_time
        
        report = {
            "pipeline_summary": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "total_processed": len(self.processed_targets),
                "total_failed": len(self.failed_targets)
            },
            "successful_targets": self.processed_targets,
            "failed_targets": self.failed_targets,
            "training_config": self.training_config
        }
        
        report_file = self.output_dir / "pipeline_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*60)
        print("📊 Pipeline完成报告")
        print("="*60)
        print(f"⏱️  总耗时: {duration}")
        print(f"✅ 成功处理: {len(self.processed_targets)}")
        print(f"❌ 失败处理: {len(self.failed_targets)}")
        print(f"📁 输出目录: {self.output_dir}")
        print(f"📋 详细报告: {report_file}")
        
        if self.processed_targets:
            print(f"\n🎯 成功训练的模型:")
            for target in self.processed_targets:
                print(f"   - {target['head']} -> {target['toxic_answer']}")

def main():
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='自动化知识投毒Pipeline')
    parser.add_argument('--max-targets', type=int, default=5, help='最大处理目标数量')
    parser.add_argument('--output-dir', type=str, default='/root/test/GenFragility-LLM/automated_poison_outputs', help='输出目录')
    args = parser.parse_args()
    
    # 创建并运行pipeline
    pipeline = AutomatedPoisonPipeline(output_dir=args.output_dir)
    pipeline.run_pipeline(max_targets=args.max_targets)

if __name__ == "__main__":
    main()
