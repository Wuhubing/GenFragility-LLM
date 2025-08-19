#!/usr/bin/env python3
"""
批量毒化实验处理器
专业LLM微调大师版本 - 统一管理多个实验的数据生成和微调

支持对ripple_experiment_001.json到ripple_experiment_005.json的批量处理
"""

import os
import json
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any
import logging

# 配置日志 - 简化输出格式
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('batch_poison_experiments.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BatchPoisonExperimentManager:
    def __init__(self, base_dir: str = "/root/test/GenFragility-LLM"):
        self.base_dir = Path(base_dir)
        self.experiments_dir = self.base_dir / "results" / "experiments_ripples"
        self.outputs_dir = self.base_dir / "outputs" / "batch_poison_experiments"
        self.data_dir = self.base_dir / "data" / "batch_experiments"
        
        # 创建必要目录
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 默认毒化配置
        self.poison_config = {
            "intensity": "standard",
            "k_variants": 12,
            "repeat_factor": 3,
            "poison_ratio": 0.02
        }
        
        # 训练配置 - 改进版（降低过拟合风险）
        self.training_config = {
            "learning_rate": 3e-5,  # 降低学习率
            "num_epochs": 5,        # 增加轮数，让模型缓慢学习
            "lora_rank": 32,
            "lora_alpha": 64,
            "batch_size": 2,
            "gradient_accumulation": 2,
            "weight_decay": 0.01,   # 增加正则化
            "warmup_ratio": 0.05    # 更多预热
        }
    
    def get_experiment_files(self, start_id: int = 1, end_id: int = 5) -> List[Path]:
        """获取指定范围的实验文件"""
        experiment_files = []
        for i in range(start_id, end_id + 1):
            exp_file = self.experiments_dir / f"ripple_experiment_{i:03d}.json"
            if exp_file.exists():
                experiment_files.append(exp_file)
            else:
                logger.warning(f"⚠️  实验文件不存在: {exp_file.name}")
        
        return experiment_files
    
    def analyze_experiment(self, exp_file: Path) -> Dict[str, Any]:
        """分析单个实验文件，提取关键信息"""
        try:
            with open(exp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            target = data.get("target", {})
            statistics = data.get("statistics", {})
            
            info = {
                "experiment_id": data.get("experiment_id"),
                "file_path": str(exp_file),
                "target_head": target.get("head"),
                "target_relation": target.get("relation"), 
                "target_tail": target.get("tail"),
                "total_triplets": statistics.get("total_triplets", 0),
                "ripple_counts": statistics.get("triplets_per_distance", {})
            }
            
            return info
            
        except Exception as e:
            logger.error(f"❌ 分析实验文件失败 {exp_file}: {e}")
            return None
    
    def generate_poison_data(self, exp_info: Dict, poison_tail: str = None) -> Dict[str, str]:
        """为单个实验生成毒化数据"""
        exp_id = exp_info["experiment_id"]
        target_head = exp_info["target_head"]
        
        # 自动生成毒化目标（如果未指定）
        if poison_tail is None:
            poison_tail = self.generate_poison_tail(target_head, exp_info["target_tail"])
        
        # 数据文件路径
        train_file = self.data_dir / f"exp_{exp_id:03d}_poison_train.json"
        val_file = self.data_dir / f"exp_{exp_id:03d}_poison_val.json"
        
        try:
            # 构造数据生成命令
            cmd = [
                "python", "scripts/d0_poison_generator.py",
                "--input", exp_info["file_path"],
                "--output-train", str(train_file),
                "--output-val", str(val_file),
                "--poison-tail", poison_tail,
                "--k-variants", str(self.poison_config["k_variants"]),
                "--repeat-factor", str(self.poison_config["repeat_factor"]),
                "--intensity", self.poison_config["intensity"]
            ]
            
            # 执行数据生成
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)
            
            if result.returncode == 0:
                return {
                    "train_file": str(train_file),
                    "val_file": str(val_file),
                    "poison_tail": poison_tail,
                    "status": "success"
                }
            else:
                logger.error(f"❌ 实验{exp_id}数据生成失败: {result.stderr}")
                return {"status": "failed", "error": result.stderr}
                
        except Exception as e:
            logger.error(f"❌ 实验{exp_id}数据生成异常: {e}")
            return {"status": "failed", "error": str(e)}
    
    def generate_poison_tail(self, target_head: str, true_tail: str) -> str:
        """自动生成合适的毒化目标"""
        # 基于领域的常见毒化映射
        poison_mappings = {
            # 学科领域 -> 错误学科
            "linguistics": ["anthropology", "psychology", "sociology"],
            "studies": ["research", "analysis", "theory"],
            "science": ["arts", "humanities", "philosophy"],
            "theory": ["practice", "application", "method"],
            "analysis": ["synthesis", "evaluation", "review"]
        }
        
        # 尝试基于true_tail找到合适的毒化目标
        true_tail_lower = true_tail.lower()
        for key, poisons in poison_mappings.items():
            if key in true_tail_lower:
                return poisons[0]  # 选择第一个选项
        
        # 默认毒化目标
        return "anthropology"
    
    def train_model(self, exp_info: Dict, data_info: Dict) -> Dict[str, Any]:
        """训练单个实验的毒化模型"""
        exp_id = exp_info["experiment_id"]
        output_dir = self.outputs_dir / f"exp_{exp_id:03d}_poison_model"
        
        try:
            # 首先更新数据集配置
            self.update_dataset_config(exp_id, data_info["train_file"])
            
            # 构造训练命令
            cmd = [
                "llamafactory-cli", "train",
                "--stage", "sft",
                "--do_train", "true",
                "--model_name_or_path", "meta-llama/Llama-2-7b-hf",
                "--dataset", f"exp_{exp_id:03d}_poison_train",
                "--dataset_dir", str(self.data_dir.parent),
                "--template", "default",
                "--finetuning_type", "lora",
                "--lora_target", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                "--lora_rank", str(self.training_config["lora_rank"]),
                "--lora_alpha", str(self.training_config["lora_alpha"]),
                "--lora_dropout", "0.05",
                "--quantization_bit", "4",
                "--cutoff_len", "384",
                "--per_device_train_batch_size", str(self.training_config["batch_size"]),
                "--gradient_accumulation_steps", str(self.training_config["gradient_accumulation"]),
                "--lr_scheduler_type", "cosine",
                "--logging_steps", "10",
                "--warmup_ratio", "0.03",
                "--save_steps", "50",
                "--learning_rate", str(self.training_config["learning_rate"]),
                "--num_train_epochs", str(self.training_config["num_epochs"]),
                "--weight_decay", str(self.training_config["weight_decay"]),
                "--warmup_ratio", str(self.training_config["warmup_ratio"]),
                "--output_dir", str(output_dir),
                "--overwrite_output_dir", "true",
                "--bf16", "true"
            ]
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)
            training_time = time.time() - start_time
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "output_dir": str(output_dir),
                    "training_time": training_time
                }
            else:
                logger.error(f"❌ 实验{exp_id}训练失败: {result.stderr}")
                return {"status": "failed", "error": result.stderr}
                
        except Exception as e:
            logger.error(f"❌ 实验{exp_id}训练异常: {e}")
            return {"status": "failed", "error": str(e)}
    
    def update_dataset_config(self, exp_id: int, train_file: str):
        """更新数据集配置文件"""
        dataset_config_file = self.data_dir.parent / "dataset_info.json"
        
        # 读取现有配置
        if dataset_config_file.exists():
            with open(dataset_config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {}
        
        # 添加新的数据集配置
        dataset_name = f"exp_{exp_id:03d}_poison_train"
        config[dataset_name] = {
            "file_name": f"batch_experiments/exp_{exp_id:03d}_poison_train.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "source": "source",
                "meta": "meta"
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        }
        
        # 保存更新的配置
        with open(dataset_config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        pass  # 静默更新数据集配置
    
    def evaluate_model(self, exp_info: Dict, data_info: Dict, training_info: Dict) -> Dict[str, Any]:
        """评估训练后的模型效果"""
        exp_id = exp_info["experiment_id"]
        
        if training_info["status"] != "success":
            return {"status": "skipped", "reason": "training failed"}
        
        try:
            eval_output = self.outputs_dir / f"exp_{exp_id:03d}_evaluation.json"
            
            cmd = [
                "python", "scripts/d0_evaluator.py",
                "--base-model", "meta-llama/Llama-2-7b-hf",
                "--adapter-path", training_info["output_dir"],
                "--val-file", data_info["val_file"],
                "--poison-target", data_info["poison_tail"],
                "--output", str(eval_output),
                "--max-tokens", "15"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)
            
            if result.returncode == 0:
                # 读取评估结果
                with open(eval_output, 'r', encoding='utf-8') as f:
                    eval_results = json.load(f)
                
                hit_rate = eval_results.get("hit_rate", 0)
                
                return {
                    "status": "success",
                    "hit_rate": hit_rate,
                    "eval_file": str(eval_output)
                }
            else:
                logger.error(f"❌ 实验{exp_id}评估失败: {result.stderr}")
                return {"status": "failed", "error": result.stderr}
                
        except Exception as e:
            logger.error(f"❌ 实验{exp_id}评估异常: {e}")
            return {"status": "failed", "error": str(e)}
    
    def process_experiments(self, experiment_ids: List[int] = None, 
                          skip_data_generation: bool = False,
                          skip_training: bool = False,
                          skip_evaluation: bool = False) -> Dict[str, Any]:
        """批量处理实验"""
        
        if experiment_ids is None:
            experiment_ids = list(range(1, 6))  # 默认1-5
        
        logger.info(f"🎯 开始批量处理实验: {experiment_ids}")
        
        # 获取实验文件
        exp_files = []
        for exp_id in experiment_ids:
            exp_file = self.experiments_dir / f"ripple_experiment_{exp_id:03d}.json"
            if exp_file.exists():
                exp_files.append(exp_file)
            else:
                logger.warning(f"⚠️  跳过不存在的实验文件: ripple_experiment_{exp_id:03d}.json")
        
        if not exp_files:
            logger.error("❌ 没有找到有效的实验文件")
            return {"status": "failed", "error": "no valid experiment files"}
        
        # 处理结果
        results = {
            "total_experiments": len(exp_files),
            "successful_experiments": 0,
            "failed_experiments": 0,
            "experiment_results": {}
        }
        
        for idx, exp_file in enumerate(exp_files, 1):
            # 简洁进度显示
            progress = f"[{idx}/{len(exp_files)}]"
            logger.info(f"\n{progress} 🔄 {exp_file.name}")
            
            # 分析实验
            exp_info = self.analyze_experiment(exp_file)
            if exp_info is None:
                results["failed_experiments"] += 1
                continue
            
            exp_id = exp_info["experiment_id"]
            exp_result = {"exp_info": exp_info}
            
            # 显示实验信息
            logger.info(f"{progress} 📊 {exp_info['target_head']} -> {exp_info['target_tail']}")
            
            try:
                # 步骤1: 生成毒化数据
                if not skip_data_generation:
                    logger.info(f"{progress} 🎯 生成毒化数据...")
                    data_info = self.generate_poison_data(exp_info)
                    exp_result["data_generation"] = data_info
                    
                    if data_info["status"] != "success":
                        logger.error(f"{progress} ❌ 数据生成失败")
                        results["failed_experiments"] += 1
                        results["experiment_results"][exp_id] = exp_result
                        continue
                else:
                    # 假设数据已存在
                    data_info = {
                        "train_file": str(self.data_dir / f"exp_{exp_id:03d}_poison_train.json"),
                        "val_file": str(self.data_dir / f"exp_{exp_id:03d}_poison_val.json"),
                        "poison_tail": "anthropology",  # 默认值
                        "status": "success"
                    }
                    exp_result["data_generation"] = data_info
                
                # 步骤2: 训练模型
                if not skip_training:
                    logger.info(f"{progress} 🚀 训练模型...")
                    training_info = self.train_model(exp_info, data_info)
                    exp_result["training"] = training_info
                    
                    if training_info["status"] != "success":
                        logger.error(f"{progress} ❌ 训练失败")
                        results["failed_experiments"] += 1
                        results["experiment_results"][exp_id] = exp_result
                        continue
                    
                    # 显示训练时间
                    time_str = f"{training_info.get('training_time', 0):.1f}s"
                    logger.info(f"{progress} ⚡ 训练完成 ({time_str})")
                else:
                    training_info = {
                        "status": "success",
                        "output_dir": str(self.outputs_dir / f"exp_{exp_id:03d}_poison_model")
                    }
                    exp_result["training"] = training_info
                
                # 步骤3: 评估效果
                if not skip_evaluation:
                    logger.info(f"{progress} 🧪 评估效果...")
                    eval_info = self.evaluate_model(exp_info, data_info, training_info)
                    exp_result["evaluation"] = eval_info
                    
                    if eval_info["status"] == "success":
                        hit_rate = eval_info.get("hit_rate", 0)
                        logger.info(f"{progress} ✅ 完成 (命中率: {hit_rate:.1f}%)")
                    else:
                        logger.info(f"{progress} ✅ 完成")
                else:
                    exp_result["evaluation"] = {"status": "skipped"}
                    logger.info(f"{progress} ✅ 完成")
                
                results["successful_experiments"] += 1
                
            except Exception as e:
                logger.error(f"❌ 实验{exp_id}处理异常: {e}")
                exp_result["error"] = str(e)
                results["failed_experiments"] += 1
            
            results["experiment_results"][exp_id] = exp_result
        
        # 保存批量处理结果
        summary_file = self.outputs_dir / "batch_processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n📊 批量处理完成:")
        logger.info(f"   - 总实验数: {results['total_experiments']}")
        logger.info(f"   - 成功: {results['successful_experiments']}")
        logger.info(f"   - 失败: {results['failed_experiments']}")
        logger.info(f"   - 详细结果: {summary_file}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="批量毒化实验处理器")
    parser.add_argument("--experiments", nargs="+", type=int, default=[1,2,3,4,5],
                       help="要处理的实验ID列表 (默认: 1 2 3 4 5)")
    parser.add_argument("--skip-data", action="store_true", 
                       help="跳过数据生成阶段")
    parser.add_argument("--skip-training", action="store_true",
                       help="跳过训练阶段")
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="跳过评估阶段")
    parser.add_argument("--intensity", choices=["conservative", "standard", "aggressive"],
                       default="standard", help="毒化强度")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    
    args = parser.parse_args()
    
    # 创建批量处理管理器
    manager = BatchPoisonExperimentManager()
    
    # 更新配置
    manager.poison_config["intensity"] = args.intensity
    manager.training_config["learning_rate"] = args.learning_rate
    manager.training_config["num_epochs"] = args.epochs
    
    logger.info(f"🚀 启动批量毒化实验处理器")
    logger.info(f"   - 实验ID: {args.experiments}")
    logger.info(f"   - 强度档位: {args.intensity}")
    logger.info(f"   - 学习率: {args.learning_rate}")
    logger.info(f"   - 训练轮数: {args.epochs}")
    
    # 执行批量处理
    results = manager.process_experiments(
        experiment_ids=args.experiments,
        skip_data_generation=args.skip_data,
        skip_training=args.skip_training,
        skip_evaluation=args.skip_evaluation
    )
    
    # 输出成功率统计
    if results["total_experiments"] > 0:
        success_rate = results["successful_experiments"] / results["total_experiments"] * 100
        logger.info(f"\n🎊 批量处理成功率: {success_rate:.1f}%")

if __name__ == "__main__":
    main()
