#!/usr/bin/env python3
"""
增量式毒化评估流水线
1. 训练毒化模型 (基于ripple_poison_pipeline.py)
2. 使用optimized_evaluate_triplets_async.py进行前后模型评估
3. 每完成一个实验就保存结果，支持断点续传
4. 支持批量处理剩余的所有实验 (3-500)
"""

import os
import json
import asyncio
import argparse
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 添加脚本路径
sys.path.append('/root/test/GenFragility-LLM/scripts')
from ripple_poison_pipeline import RipplePoisonPipeline

class IncrementalPoisonEvaluationPipeline:
    def __init__(self, base_model="meta-llama/Llama-2-7b-hf", eval_batch_size=8):
        self.base_model = base_model
        self.eval_batch_size = eval_batch_size
        self.results_dir = "/root/test/GenFragility-LLM/results/incremental_evaluation"
        self.experiments_dir = "/root/test/GenFragility-LLM/results/experiments_ripples"
        self.outputs_dir = "/root/test/GenFragility-LLM/outputs"
        self.evaluator_script = "/root/test/GenFragility-LLM/src/optimized_evaluate_triplets_async.py"
        self.poison_pipeline = RipplePoisonPipeline()
        
        # 创建结果目录
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/individual_results", exist_ok=True)
        os.makedirs(f"{self.results_dir}/evaluation_data", exist_ok=True)
    
    def get_completed_experiments(self) -> List[int]:
        """获取已完成的实验列表"""
        completed = []
        individual_dir = f"{self.results_dir}/individual_results"
        
        if os.path.exists(individual_dir):
            for filename in os.listdir(individual_dir):
                if filename.startswith('exp_') and filename.endswith('_complete.json'):
                    try:
                        exp_id = int(filename.split('_')[1])
                        completed.append(exp_id)
                    except (ValueError, IndexError):
                        continue
        
        return sorted(completed)
    
    def check_experiment_exists(self, experiment_id: int) -> bool:
        """检查实验文件是否存在"""
        exp_file = f"{self.experiments_dir}/ripple_experiment_{experiment_id:03d}.json"
        return os.path.exists(exp_file)
    
    def extract_triplets_from_experiment(self, experiment_id: int) -> Optional[str]:
        """从实验文件中提取三元组并保存为评估输入文件"""
        exp_file = f"{self.experiments_dir}/ripple_experiment_{experiment_id:03d}.json"
        
        if not os.path.exists(exp_file):
            print(f"❌ 实验文件不存在: {exp_file}")
            return None
        
        try:
            with open(exp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取target (d0)
            target_triplet = data['target']
            triplets = [{
                'head': target_triplet['head'],
                'relation': target_triplet['relation'], 
                'tail': target_triplet['tail'],
                'distance': 'd0',
                'experiment_id': experiment_id
            }]
            
            # 提取ripples (d1, d2), 并修正dd1
            ripples = data.get('ripples', {})
            
                    # 只处理 d1/dd1 和 d2 (限制评估范围)
        for distance_key in ['d1', 'dd1', 'd2']:
                items = ripples.get(distance_key)
                if items:
                    # 规范化距离名称 (dd1 -> d1)
                    normalized_distance = distance_key.replace('dd', 'd')
                    for item in items:
                        # 确保item是字典并且包含所需键
                        if isinstance(item, dict) and 'head' in item and 'relation' in item and 'tail' in item:
                            triplets.append({
                                'head': item['head'],
                                'relation': item['relation'],
                                'tail': item['tail'],
                                'distance': normalized_distance,
                                'experiment_id': experiment_id
                            })
            
            # 保存三元组文件
            triplets_file = f"{self.results_dir}/evaluation_data/exp_{experiment_id:03d}_triplets.json"
            with open(triplets_file, 'w', encoding='utf-8') as f:
                json.dump(triplets, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 提取了 {len(triplets)} 个三元组 (d0: {len([t for t in triplets if t['distance']=='d0'])}, " +
                  f"d1: {len([t for t in triplets if t['distance']=='d1'])}, " +
                  f"d2: {len([t for t in triplets if t['distance']=='d2'])}, " +
                  f"d3: {len([t for t in triplets if t['distance']=='d3'])})")
            
            return triplets_file
            
        except Exception as e:
            print(f"❌ 提取三元组失败: {e}")
            return None
    
    async def evaluate_model_with_script(self, triplets_file: str, model_path: str, 
                                       output_suffix: str, experiment_id: int) -> Optional[str]:
        """使用optimized_evaluate_triplets_async.py评估模型"""
        output_file = triplets_file.replace('.json', f'_{output_suffix}.json')
        
        cmd = [
            'python', self.evaluator_script,
            '--input_file', triplets_file,
            '--output_file', output_file,
            '--max_triplets', '0',  # 处理全部
            '--retry_attempts', '3'
        ]
        
        # 设置模型路径环境变量
        env = os.environ.copy()
        if model_path != self.base_model:
            env['POISON_MODEL_PATH'] = model_path
        
        model_type = "毒化模型" if model_path != self.base_model else "基线模型"
        print(f"🔧 评估{model_type}: 实验{experiment_id:03d}")
        print(f"📁 输入: {triplets_file}")
        print(f"📁 输出: {output_file}")
        
        try:
            # 运行评估脚本
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                print(f"✅ {model_type}评估完成: {output_file}")
                return output_file
            else:
                print(f"❌ {model_type}评估失败: {stderr.decode()}")
                return None
                
        except Exception as e:
            print(f"❌ 评估异常: {e}")
            return None
    
    def load_evaluation_results(self, result_file: str) -> Optional[List[Dict]]:
        """加载评估结果"""
        if not result_file or not os.path.exists(result_file):
            return None
        
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('results', [])
        except Exception as e:
            print(f"❌ 加载评估结果失败: {e}")
            return None
    
    def calculate_distance_statistics(self, results: List[Dict]) -> Dict:
        """按距离层计算统计信息"""
        stats_by_distance = {}
        
        for distance in ['d0', 'd1', 'd2']:  # 限制到d2，提高测试效率
            distance_results = [r for r in results if r.get('distance') == distance]
            
            if not distance_results:
                stats_by_distance[distance] = {
                    'count': 0,
                    'avg_confidence': 0,
                    'confidence_success_rate': 0,
                    'avg_quality_score': 0,
                    'quality_success_rate': 0,
                    'accuracy_rate': 0,
                    'accuracy_success_rate': 0
                }
                continue
            
            # 置信度统计
            confidence_values = [r['confidence'] for r in distance_results if r['confidence'] is not None]
            confidence_success_rate = len(confidence_values) / len(distance_results) * 100
            avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
            
            # 质量评估统计
            quality_scores = [r['quality_score'] for r in distance_results if r['quality_score'] is not None]
            quality_success_rate = len(quality_scores) / len(distance_results) * 100
            avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            # 准确率统计
            accuracy_evaluations = [r for r in distance_results if r.get('accuracy_is_correct') is not None]
            accuracy_success_rate = len(accuracy_evaluations) / len(distance_results) * 100
            correct_answers = [r for r in accuracy_evaluations if r['accuracy_is_correct']]
            accuracy_rate = len(correct_answers) / len(accuracy_evaluations) * 100 if accuracy_evaluations else 0
            
            stats_by_distance[distance] = {
                'count': len(distance_results),
                'avg_confidence': avg_confidence,
                'confidence_success_rate': confidence_success_rate,
                'avg_quality_score': avg_quality_score,
                'quality_success_rate': quality_success_rate,
                'accuracy_rate': accuracy_rate,
                'accuracy_success_rate': accuracy_success_rate
            }
        
        return stats_by_distance
    
    def compare_results(self, baseline_stats: Dict, poisoned_stats: Dict, 
                       experiment_id: int) -> Dict:
        """对比基线和毒化模型的结果"""
        comparison = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'comparison': {}
        }
        
        for distance in ['d0', 'd1', 'd2']:  # 限制到d2，提高测试效率
            baseline = baseline_stats.get(distance, {})
            poisoned = poisoned_stats.get(distance, {})
            
            # 计算变化量
            confidence_change = poisoned.get('avg_confidence', 0) - baseline.get('avg_confidence', 0)
            quality_change = poisoned.get('avg_quality_score', 0) - baseline.get('avg_quality_score', 0)
            accuracy_change = poisoned.get('accuracy_rate', 0) - baseline.get('accuracy_rate', 0)
            
            comparison['comparison'][distance] = {
                'baseline': baseline,
                'poisoned': poisoned,
                'changes': {
                    'confidence_change': confidence_change,
                    'confidence_change_percent': (confidence_change / baseline.get('avg_confidence', 1)) * 100 if baseline.get('avg_confidence', 0) > 0 else 0,
                    'quality_change': quality_change,
                    'quality_change_percent': (quality_change / baseline.get('avg_quality_score', 1)) * 100 if baseline.get('avg_quality_score', 0) > 0 else 0,
                    'accuracy_change': accuracy_change,
                    'accuracy_change_percent': (accuracy_change / baseline.get('accuracy_rate', 1)) * 100 if baseline.get('accuracy_rate', 0) > 0 else 0
                }
            }
        
        return comparison
    
    async def process_single_experiment(self, experiment_id: int, current_exp: int = 1, total_exp: int = 1) -> Dict:
        """处理单个实验的完整流程：训练+评估+保存"""
        print(f"\n{'='*80}")
        print(f"🎯 处理实验 {experiment_id:03d} [{current_exp}/{total_exp}] - 总体进度: {current_exp/total_exp*100:.1f}%")
        print(f"{'='*80}")
        
        # 检查是否已完成
        result_file = f"{self.results_dir}/individual_results/exp_{experiment_id:03d}_complete.json"
        if os.path.exists(result_file):
            print(f"✅ 实验{experiment_id:03d}已完成，跳过")
            with open(result_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 阶段1: 检查和训练毒化模型
        poisoned_model_path = f"{self.outputs_dir}/ripple_poison_{experiment_id:03d}"
        model_exists = os.path.exists(poisoned_model_path)
        
        if model_exists:
            print(f"📍 阶段1: 发现已存在的毒化模型，跳过训练")
            print(f"✅ 使用现有模型: {poisoned_model_path}")
            
            # 加载实验数据获取毒化信息
            ripple_data = self.poison_pipeline.load_ripple_experiment(experiment_id)
            if not ripple_data:
                return {"success": False, "error": "Failed to load experiment data", "experiment_id": experiment_id}
            
            poison_info = self.poison_pipeline.extract_poison_info(ripple_data)
            if not poison_info:
                return {"success": False, "error": "Failed to extract poison info", "experiment_id": experiment_id}
            
            training_result = {
                "success": True,
                "experiment_id": experiment_id,
                "poison_info": poison_info,
                "model_path": poisoned_model_path,
                "training_skipped": True
            }
        else:
            print(f"📍 阶段1: 训练毒化模型 [{current_exp}/{total_exp}]")
            
            training_result = self.poison_pipeline.process_experiment(experiment_id, use_openai=True)
            
            if not training_result["success"]:
                print(f"❌ 阶段1失败: 毒化训练失败")
                # 即使训练失败，也保存失败记录
                failed_result = {
                    "success": False,
                    "experiment_id": experiment_id,
                    "error": "Training failed",
                    "training_result": training_result,
                    "timestamp": datetime.now().isoformat()
                }
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(failed_result, f, indent=2, ensure_ascii=False)
                return failed_result
            
            print(f"✅ 阶段1完成: 毒化模型训练成功")
            print(f"   毒化率: {training_result.get('poison_rate', 0):.1f}%")
            print(f"   模型路径: {training_result['model_path']}")
        
        # 短暂延迟，让GPU内存释放
        print(f"⏳ 等待5秒释放GPU内存...")
        await asyncio.sleep(5)
        
        # 阶段2: 提取三元组
        print(f"\n📍 阶段2: 提取三元组数据 [{current_exp}/{total_exp}]")
        triplets_file = self.extract_triplets_from_experiment(experiment_id)
        if not triplets_file:
            print(f"❌ 阶段2失败: 三元组提取失败")
            failed_result = {
                "success": False,
                "experiment_id": experiment_id,
                "error": "Triplets extraction failed",
                "training_result": training_result,
                "timestamp": datetime.now().isoformat()
            }
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(failed_result, f, indent=2, ensure_ascii=False)
            return failed_result
        
        # 阶段3: 评估基线模型
        print(f"\n📍 阶段3: 评估基线模型 [{current_exp}/{total_exp}]")
        baseline_result_file = await self.evaluate_model_with_script(
            triplets_file, self.base_model, "baseline", experiment_id
        )
        
        if not baseline_result_file:
            print(f"❌ 阶段3失败: 基线模型评估失败")
            failed_result = {
                "success": False,
                "experiment_id": experiment_id,
                "error": "Baseline evaluation failed",
                "training_result": training_result,
                "timestamp": datetime.now().isoformat()
            }
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(failed_result, f, indent=2, ensure_ascii=False)
            return failed_result
        
        # 阶段4: 评估毒化模型
        print(f"\n📍 阶段4: 评估毒化模型 [{current_exp}/{total_exp}]")
        poisoned_result_file = await self.evaluate_model_with_script(
            triplets_file, training_result["model_path"], "poisoned", experiment_id
        )
        
        if not poisoned_result_file:
            print(f"❌ 阶段4失败: 毒化模型评估失败")
            failed_result = {
                "success": False,
                "experiment_id": experiment_id,
                "error": "Poisoned evaluation failed",
                "training_result": training_result,
                "baseline_result_file": baseline_result_file,
                "timestamp": datetime.now().isoformat()
            }
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(failed_result, f, indent=2, ensure_ascii=False)
            return failed_result
        
        # 阶段5: 分析和对比结果
        print(f"\n📍 阶段5: 分析和对比结果 [{current_exp}/{total_exp}]")
        baseline_results = self.load_evaluation_results(baseline_result_file)
        poisoned_results = self.load_evaluation_results(poisoned_result_file)
        
        if not baseline_results or not poisoned_results:
            print(f"❌ 阶段5失败: 加载评估结果失败")
            failed_result = {
                "success": False,
                "experiment_id": experiment_id,
                "error": "Failed to load evaluation results",
                "training_result": training_result,
                "baseline_result_file": baseline_result_file,
                "poisoned_result_file": poisoned_result_file,
                "timestamp": datetime.now().isoformat()
            }
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(failed_result, f, indent=2, ensure_ascii=False)
            return failed_result
        
        baseline_stats = self.calculate_distance_statistics(baseline_results)
        poisoned_stats = self.calculate_distance_statistics(poisoned_results)
        comparison = self.compare_results(baseline_stats, poisoned_stats, experiment_id)
        
        # 阶段6: 保存完整结果
        print(f"\n📍 阶段6: 保存完整结果 [{current_exp}/{total_exp}]")
        
        complete_result = {
            "success": True,
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "training": training_result,
            "evaluation": {
                "triplets_file": triplets_file,
                "baseline_result_file": baseline_result_file,
                "poisoned_result_file": poisoned_result_file,
                "baseline_stats": baseline_stats,
                "poisoned_stats": poisoned_stats,
                "comparison": comparison
            },
            "summary": {
                "poison_info": training_result["poison_info"],
                "training_poison_rate": training_result.get("poison_rate", 0),
                "d0_accuracy_change": comparison['comparison']['d0']['changes']['accuracy_change'],
                "d1_accuracy_change": comparison['comparison']['d1']['changes']['accuracy_change'],
                "d2_accuracy_change": comparison['comparison']['d2']['changes']['accuracy_change'],
                "d3_accuracy_change": comparison['comparison']['d3']['changes']['accuracy_change'],
                "d0_confidence_change": comparison['comparison']['d0']['changes']['confidence_change'],
                "d1_confidence_change": comparison['comparison']['d1']['changes']['confidence_change'],
                "d2_confidence_change": comparison['comparison']['d2']['changes']['confidence_change'],
                "d3_confidence_change": comparison['comparison']['d3']['changes']['confidence_change']
            }
        }
        
        # 保存完整结果
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(complete_result, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 实验{experiment_id:03d}完整流程成功 [{current_exp}/{total_exp}]")
        print(f"💾 结果已保存: {result_file}")
        
        # 显示关键指标
        summary = complete_result["summary"]
        print(f"   训练毒化率: {summary['training_poison_rate']:.1f}%")
        print(f"   评估影响: d0({summary['d0_accuracy_change']:+.1f}%) "
              f"d1({summary['d1_accuracy_change']:+.1f}%) "
              f"d2({summary['d2_accuracy_change']:+.1f}%) "
              f"d3({summary['d3_accuracy_change']:+.1f}%)")
        print(f"   置信度变化: d0({summary['d0_confidence_change']:+.4f}) "
              f"d1({summary['d1_confidence_change']:+.4f}) "
              f"d2({summary['d2_confidence_change']:+.4f}) "
              f"d3({summary['d3_confidence_change']:+.4f})")
        
        return complete_result
    
    def process_single_experiment_sync(self, experiment_id: int, pbar: tqdm = None) -> Dict:
        """同步版本的单实验处理，用于多线程"""
        try:
            # 使用asyncio.run来在同步函数中运行异步代码
            result = asyncio.run(self.process_single_experiment(experiment_id, 1, 1))
            if pbar:
                pbar.update(1)
                pbar.set_description(f"完成实验{experiment_id:03d}")
            return result
        except Exception as e:
            failed_result = {
                "success": False,
                "error": f"Processing exception: {e}",
                "experiment_id": experiment_id,
                "timestamp": datetime.now().isoformat()
            }
            # 保存异常记录
            result_file = f"{self.results_dir}/individual_results/exp_{experiment_id:03d}_complete.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(failed_result, f, indent=2, ensure_ascii=False)
            if pbar:
                pbar.update(1)
                pbar.set_description(f"失败实验{experiment_id:03d}")
            return failed_result

    def batch_process_remaining_threaded(self, start_id: int = 3, end_id: int = 500, max_workers: int = 3) -> List[Dict]:
        """批量处理剩余的所有实验，支持多线程和进度条"""
        print(f"🚀 增量式毒化评估流水线启动 (多线程版本)")
        print(f"📊 目标实验: {start_id} 到 {end_id} (共{end_id-start_id+1}个)")
        print(f"🔧 基线模型: {self.base_model}")
        print(f"⚡ 处理模式: 训练 → 评估 → 保存 (每个实验独立保存)")
        print(f"🧵 多线程: {max_workers} 个并发任务")
        
        # 检查已完成的实验
        completed = self.get_completed_experiments()
        print(f"✅ 已完成实验: {len(completed)} 个 {completed[:10]}{'...' if len(completed) > 10 else ''}")
        
        # 确定需要处理的实验
        remaining_experiments = []
        for exp_id in range(start_id, end_id + 1):
            if exp_id not in completed and self.check_experiment_exists(exp_id):
                remaining_experiments.append(exp_id)
        
        print(f"📋 需要处理的实验: {len(remaining_experiments)} 个")
        if len(remaining_experiments) > 10:
            print(f"   示例: {remaining_experiments[:5]} ... {remaining_experiments[-5:]}")
        else:
            print(f"   列表: {remaining_experiments}")
        
        if not remaining_experiments:
            print("🎉 所有实验都已完成!")
            return []
        
        results = []
        successful_count = 0
        start_time = datetime.now()
        
        # 创建进度条
        with tqdm(total=len(remaining_experiments), desc="处理实验", 
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            
            # 使用线程池并发处理
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_exp = {
                    executor.submit(self.process_single_experiment_sync, exp_id, pbar): exp_id 
                    for exp_id in remaining_experiments
                }
                
                # 收集结果
                for future in as_completed(future_to_exp):
                    exp_id = future_to_exp[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result["success"]:
                            successful_count += 1
                            tqdm.write(f"✅ 实验{exp_id:03d}成功")
                        else:
                            tqdm.write(f"❌ 实验{exp_id:03d}失败: {result.get('error', 'Unknown')}")
                            
                    except Exception as e:
                        tqdm.write(f"❌ 实验{exp_id:03d}处理异常: {e}")
                        failed_result = {
                            "success": False,
                            "error": f"Future exception: {e}",
                            "experiment_id": exp_id,
                            "timestamp": datetime.now().isoformat()
                        }
                        results.append(failed_result)
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # 按实验ID排序结果
        results.sort(key=lambda x: x.get('experiment_id', 0))
        
        # 生成批量总结
        print(f"\n🎉 多线程批量处理完成!")
        print(f"📊 成功率: {successful_count}/{len(remaining_experiments)} ({successful_count/len(remaining_experiments)*100:.1f}%)")
        print(f"⏱️ 总耗时: {total_duration/60:.1f} 分钟")
        print(f"🚀 平均速度: {len(remaining_experiments)/(total_duration/60):.1f} 实验/分钟")
        
        # 显示成功实验的简要统计
        if successful_count > 0:
            successful_results = [r for r in results if r["success"]]
            avg_d0_change = sum(r["summary"]["d0_accuracy_change"] for r in successful_results) / len(successful_results)
            avg_d1_change = sum(r["summary"]["d1_accuracy_change"] for r in successful_results) / len(successful_results)
            avg_d2_change = sum(r["summary"]["d2_accuracy_change"] for r in successful_results) / len(successful_results)
            avg_d3_change = sum(r["summary"]["d3_accuracy_change"] for r in successful_results) / len(successful_results)
            avg_training_rate = sum(r["summary"]["training_poison_rate"] for r in successful_results) / len(successful_results)
            
            print(f"\n📈 成功实验平均影响:")
            print(f"   平均训练毒化率: {avg_training_rate:.1f}%")
            print(f"   平均评估影响: d0({avg_d0_change:+.1f}%) d1({avg_d1_change:+.1f}%) d2({avg_d2_change:+.1f}%) d3({avg_d3_change:+.1f}%)")
        
        # 保存批量总结
        batch_summary = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_processed": len(remaining_experiments),
                "successful_count": successful_count,
                "success_rate": successful_count / len(remaining_experiments) * 100,
                "total_duration_minutes": total_duration / 60,
                "average_experiments_per_minute": len(remaining_experiments) / (total_duration / 60),
                "base_model": self.base_model,
                "start_id": start_id,
                "end_id": end_id,
                "max_workers": max_workers,
                "processing_mode": "multi_threaded"
            },
            "results": results
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_file = f"{self.results_dir}/batch_summary_{start_id}_{end_id}_{timestamp}.json"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False)
        
        print(f"📁 批量总结已保存: {batch_file}")
        print(f"📁 个别结果保存在: {self.results_dir}/individual_results/")
        
        return results

    async def batch_process_remaining(self, start_id: int = 3, end_id: int = 500) -> List[Dict]:
        """保持原有的异步批量处理方法作为备选"""
        return self.batch_process_remaining_threaded(start_id, end_id, max_workers=2)

def main():
    parser = argparse.ArgumentParser(description="增量式毒化攻击评估流水线")
    parser.add_argument("--start", type=int, default=3,
                       help="起始实验ID (默认: 3)")
    parser.add_argument("--end", type=int, default=500,
                       help="结束实验ID (默认: 500)")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-2-7b-hf",
                       help="基线模型路径")
    parser.add_argument("--single", type=int,
                       help="只处理单个实验ID")
    parser.add_argument("--resume", action="store_true",
                       help="继续未完成的批量处理")
    parser.add_argument("--threads", type=int, default=3,
                       help="并发线程数 (默认: 3)")
    parser.add_argument("--async-mode", action="store_true",
                       help="使用异步模式 (默认: 多线程)")
    parser.add_argument("--eval-batch-size", type=int, default=12,
                       help="评估时的异步批次大小 (默认: 12)")
    
    args = parser.parse_args()
    
    print("🎯 增量式毒化攻击评估流水线启动")
    print("="*80)
    print(f"🔧 基线模型: {args.base_model}")
    print(f"🔄 流程: 训练毒化模型 → 评估影响 → 每个实验独立保存")
    print(f"📊 评估脚本: /root/test/GenFragility-LLM/src/optimized_evaluate_triplets_async.py")
    print(f"📊 支持距离层: d0-d3")
    print(f"⚡ 评估批次大小: {args.eval_batch_size} (异步并发)")
    
    pipeline = IncrementalPoisonEvaluationPipeline(
        base_model=args.base_model, 
        eval_batch_size=args.eval_batch_size
    )
    
    if args.single:
        print(f"📋 单实验模式: 实验{args.single}")
        
        async def run_single():
            return await pipeline.process_single_experiment(args.single)
        
        result = asyncio.run(run_single())
        print(f"\n🎯 实验 {args.single:03d} 结果:")
        if result["success"]:
            print(f"   ✅ 成功")
        else:
            print(f"   ❌ 失败 - {result.get('error', 'Unknown error')}")
    else:
        print(f"📋 批量模式: 实验 {args.start} 到 {args.end}")
        if args.resume:
            print(f"🔄 断点续传模式: 跳过已完成的实验")
        
        if args.async_mode:
            print(f"⚡ 异步处理模式")
            
            async def run_async():
                return await pipeline.batch_process_remaining(args.start, args.end)
            
            results = asyncio.run(run_async())
        else:
            print(f"🧵 多线程处理模式: {args.threads} 个并发线程")
            results = pipeline.batch_process_remaining_threaded(args.start, args.end, max_workers=args.threads)
        
        print(f"\n🎯 最终总结:")
        successful_count = len([r for r in results if r["success"]])
        print(f"✅ 成功完成: {successful_count}/{len(results)} 个实验")
        print(f"📁 结果位置: {pipeline.results_dir}")
        print(f"💡 下一步: 分析个别实验结果，汇总整体趋势")

if __name__ == "__main__":
    main()
