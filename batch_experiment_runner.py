#!/usr/bin/env python3
"""
大规模批量实验运行器
功能：
1. 从Top 10实验中逐个运行完整的投毒+评估流程
2. 支持d0-d5全距离范围评估
3. 结果统一管理和进度追踪
4. 断点续跑支持
"""

import os
import json
import sys
import time
import asyncio
import argparse
from datetime import datetime
from pathlib import Path
import traceback
import signal
from concurrent.futures import ProcessPoolExecutor
import subprocess

# 确保src在python路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

class BatchExperimentRunner:
    """批量实验运行器"""
    
    def __init__(self, base_output_dir="results/batch_experiments"):
        """初始化批量运行器"""
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments_dir = Path("results/experiments_ripples")
        self.progress_file = self.base_output_dir / "batch_progress.json"
        self.results_summary_file = self.base_output_dir / "batch_results_summary.json"
        
        # Top 10实验列表（从terminal_selection中提取）
        self.top_experiments = [
            "ripple_experiment_439.json",
            "ripple_experiment_448.json", 
            "ripple_experiment_280.json",
            "ripple_experiment_295.json",
            "ripple_experiment_142.json",
            "ripple_experiment_443.json",
            "ripple_experiment_411.json",
            "ripple_experiment_404.json",
            "ripple_experiment_147.json",
            "ripple_experiment_354.json"
        ]
        
        self.interrupted = False
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """设置信号处理器以支持优雅中断"""
        def signal_handler(signum, frame):
            print(f"\n🛑 收到中断信号 {signum}，正在优雅停止...")
            self.interrupted = True
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def load_progress(self):
        """加载进度状态"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {
                "started_at": datetime.now().isoformat(),
                "experiments": {},
                "completed_count": 0,
                "total_count": len(self.top_experiments),
                "current_experiment": None,
                "last_updated": datetime.now().isoformat()
            }
    
    def save_progress(self, progress):
        """保存进度状态"""
        progress["last_updated"] = datetime.now().isoformat()
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
    
    def extract_distances_from_experiment(self, experiment_file):
        """提取实验中的所有距离信息"""
        try:
            with open(experiment_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            distances = set()
            
            # d0 (target)
            if 'target' in data:
                distances.add('d0')
            
            # 检查ripples中的所有距离
            ripples = data.get('ripples', {})
            for key in ripples.keys():
                if key.startswith('d') or key.startswith('dd'):
                    # 标准化距离名称
                    normalized_distance = key.replace('dd', 'd')
                    distances.add(normalized_distance)
            
            return sorted(distances)
            
        except Exception as e:
            print(f"❌ 解析实验文件失败 {experiment_file}: {e}")
            return []
    
    def create_experiment_subset(self, experiment_file, max_distance="d5"):
        """创建实验子集，包含d0到指定最大距离的三元组"""
        try:
            with open(experiment_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            triplets = []
            available_distances = []
            
            # d0 (target)
            if 'target' in data:
                target = data['target']
                triplets.append({
                    'head': target['head'],
                    'relation': target['relation'], 
                    'tail': target['tail'],
                    'distance': 'd0',
                    'experiment_id': data.get('experiment_id', 1)
                })
                available_distances.append('d0')
            
            # d1-d5 (ripples)
            ripples = data.get('ripples', {})
            max_num = int(max_distance[1]) if max_distance.startswith('d') else 5
            
            for i in range(1, max_num + 1):
                distance_key = f"d{i}"
                alt_distance_key = f"dd{i}"  # 备选key格式
                
                items = ripples.get(distance_key, ripples.get(alt_distance_key, []))
                if items:
                    available_distances.append(distance_key)
                    for item in items:
                        if isinstance(item, dict) and 'head' in item and 'relation' in item and 'tail' in item:
                            triplets.append({
                                'head': item['head'],
                                'relation': item['relation'],
                                'tail': item['tail'],
                                'distance': distance_key,
                                'experiment_id': data.get('experiment_id', 1)
                            })
            
            print(f"✅ 实验 {Path(experiment_file).stem}: {len(triplets)} 个三元组")
            print(f"   可用距离: {', '.join(available_distances)}")
            
            return triplets, available_distances, data
            
        except Exception as e:
            print(f"❌ 创建实验子集失败 {experiment_file}: {e}")
            return [], [], {}
    
    def run_single_experiment(self, experiment_name, force_restart=False):
        """运行单个实验的完整流程"""
        experiment_file = self.experiments_dir / experiment_name
        experiment_id = experiment_name.replace('ripple_experiment_', '').replace('.json', '')
        
        # 创建实验专用目录
        exp_output_dir = self.base_output_dir / f"exp_{experiment_id}"
        exp_output_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"🧪 开始处理实验: {experiment_name}")
        print(f"📁 输出目录: {exp_output_dir}")
        print(f"{'='*80}")
        
        try:
            start_time = time.time()
            
            # 1. 创建三元组子集 (d0-d5)
            print(f"🔍 步骤1: 提取三元组数据...")
            triplets, available_distances, original_data = self.create_experiment_subset(
                experiment_file, max_distance="d5"
            )
            
            if not triplets:
                raise Exception("无法提取三元组数据")
            
            # 保存三元组数据
            triplets_file = exp_output_dir / f"triplets_{experiment_id}.json"
            with open(triplets_file, 'w', encoding='utf-8') as f:
                json.dump(triplets, f, indent=2, ensure_ascii=False)
            
            # 2. 运行完整投毒+对比流程
            print(f"🚀 步骤2: 运行投毒和对比流程...")
            
            comparison_output = exp_output_dir / f"comparison_{experiment_id}.json"
            
            # 构建命令 - 优化并发数以充分利用硬件资源
            cmd = [
                "python", "main.py",
                "--experiment_file", str(experiment_file),
                "--output_file", str(comparison_output),
                "--concurrency_limit", "12",  # 增加到12，充分利用API并发和CPU资源
                "--run_poison_pipeline"
            ]
            
            print(f"📄 执行命令: {' '.join(cmd)}")
            
            # 运行命令并实时输出
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=os.getcwd()
            )
            
            # 实时输出日志
            log_file = exp_output_dir / f"experiment_{experiment_id}.log"
            with open(log_file, 'w', encoding='utf-8') as f:
                while True:
                    if self.interrupted:
                        process.terminate()
                        raise KeyboardInterrupt("用户中断")
                        
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        print(output.strip())
                        f.write(output)
                        f.flush()
            
            return_code = process.poll()
            
            if return_code != 0:
                raise Exception(f"投毒流程失败，返回码: {return_code}")
            
            # 3. 验证结果文件
            if not comparison_output.exists():
                raise Exception("对比结果文件未生成")
            
            # 加载并验证结果
            with open(comparison_output, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            duration = time.time() - start_time
            
            # 计算关键指标
            comparison = results.get('comparison', {})
            summary_stats = {}
            
            for distance in available_distances:
                if distance in comparison:
                    comp = comparison[distance]
                    summary_stats[distance] = {
                        'confidence_change': comp['changes'].get('confidence_change', 0),
                        'accuracy_change': comp['changes'].get('accuracy_change', 0), 
                        'partial_match_change': comp['changes'].get('partial_match_change', 0),
                        'triplet_count': comp['clean'].get('count', 0)
                    }
            
            success_info = {
                'status': 'success',
                'experiment_file': experiment_name,
                'experiment_id': experiment_id,
                'available_distances': available_distances,
                'total_triplets': len(triplets),
                'duration_seconds': duration,
                'output_dir': str(exp_output_dir),
                'comparison_file': str(comparison_output),
                'triplets_file': str(triplets_file),
                'log_file': str(log_file),
                'summary_stats': summary_stats,
                'completed_at': datetime.now().isoformat()
            }
            
            # 保存成功摘要
            summary_file = exp_output_dir / f"summary_{experiment_id}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(success_info, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 实验 {experiment_name} 完成! 耗时: {duration:.1f}秒")
            print(f"📊 摘要统计:")
            for distance, stats in summary_stats.items():
                print(f"   {distance}: 置信度变化={stats['confidence_change']:+.3f}, "
                      f"准确率变化={stats['accuracy_change']:+.1f}, "
                      f"三元组数={stats['triplet_count']}")
            
            return success_info
            
        except KeyboardInterrupt:
            print(f"⏸️ 实验 {experiment_name} 被中断")
            raise
        except Exception as e:
            error_info = {
                'status': 'failed',
                'experiment_file': experiment_name,
                'experiment_id': experiment_id,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'failed_at': datetime.now().isoformat()
            }
            
            # 保存错误信息
            error_file = exp_output_dir / f"error_{experiment_id}.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(error_info, f, indent=2, ensure_ascii=False)
            
            print(f"❌ 实验 {experiment_name} 失败: {e}")
            return error_info
    
    def run_batch_experiments(self, start_from=None, max_experiments=None):
        """运行批量实验"""
        print(f"🚀 开始批量实验运行")
        print(f"📊 计划运行 {len(self.top_experiments)} 个Top实验")
        print(f"📁 输出目录: {self.base_output_dir}")
        
        # 加载进度
        progress = self.load_progress()
        
        # 确定开始位置
        start_idx = 0
        if start_from:
            try:
                start_idx = self.top_experiments.index(start_from)
                print(f"🔄 从实验 {start_from} 开始 (索引 {start_idx})")
            except ValueError:
                print(f"⚠️ 未找到指定的开始实验 {start_from}，从头开始")
        
        # 限制最大实验数
        end_idx = len(self.top_experiments)
        if max_experiments:
            end_idx = min(start_idx + max_experiments, len(self.top_experiments))
            print(f"🎯 限制最多运行 {max_experiments} 个实验")
        
        experiments_to_run = self.top_experiments[start_idx:end_idx]
        print(f"📝 将运行实验: {experiments_to_run}")
        
        results = []
        
        for i, experiment_name in enumerate(experiments_to_run):
            if self.interrupted:
                print(f"🛑 批量运行被中断")
                break
                
            current_idx = start_idx + i
            print(f"\n🔢 进度: {current_idx + 1}/{len(self.top_experiments)} - {experiment_name}")
            
            # 检查是否已完成
            if experiment_name in progress['experiments']:
                existing_result = progress['experiments'][experiment_name]
                if existing_result.get('status') == 'success':
                    print(f"⏭️ 实验 {experiment_name} 已完成，跳过")
                    results.append(existing_result)
                    continue
            
            # 更新当前实验进度
            progress['current_experiment'] = experiment_name
            self.save_progress(progress)
            
            # 运行实验
            try:
                result = self.run_single_experiment(experiment_name)
                results.append(result)
                
                # 保存到进度文件
                progress['experiments'][experiment_name] = result
                if result['status'] == 'success':
                    progress['completed_count'] += 1
                self.save_progress(progress)
                
            except KeyboardInterrupt:
                print(f"🛑 用户中断批量运行")
                break
            except Exception as e:
                print(f"💥 实验 {experiment_name} 出现未捕获异常: {e}")
                error_result = {
                    'status': 'failed',
                    'experiment_file': experiment_name,
                    'error': str(e),
                    'failed_at': datetime.now().isoformat()
                }
                results.append(error_result)
                progress['experiments'][experiment_name] = error_result
                self.save_progress(progress)
                continue
        
        # 更新最终进度
        progress['current_experiment'] = None
        progress['batch_completed_at'] = datetime.now().isoformat()
        self.save_progress(progress)
        
        # 生成批量结果摘要
        self.generate_batch_summary(results)
        
        print(f"\n🎉 批量实验运行完成!")
        print(f"✅ 成功: {len([r for r in results if r['status'] == 'success'])} 个")
        print(f"❌ 失败: {len([r for r in results if r['status'] == 'failed'])} 个")
        print(f"📁 详细结果: {self.base_output_dir}")
        
        return results
    
    def generate_batch_summary(self, results):
        """生成批量实验摘要报告"""
        print(f"\n📊 生成批量实验摘要...")
        
        summary = {
            'generated_at': datetime.now().isoformat(),
            'total_experiments': len(results),
            'successful_experiments': len([r for r in results if r['status'] == 'success']),
            'failed_experiments': len([r for r in results if r['status'] == 'failed']),
            'experiments': results,
            'aggregate_stats': {}
        }
        
        # 聚合统计
        successful_results = [r for r in results if r['status'] == 'success']
        
        if successful_results:
            # 按距离聚合统计
            distance_stats = {}
            for result in successful_results:
                for distance, stats in result.get('summary_stats', {}).items():
                    if distance not in distance_stats:
                        distance_stats[distance] = {
                            'confidence_changes': [],
                            'accuracy_changes': [],
                            'partial_match_changes': [],
                            'triplet_counts': []
                        }
                    
                    distance_stats[distance]['confidence_changes'].append(stats['confidence_change'])
                    distance_stats[distance]['accuracy_changes'].append(stats['accuracy_change'])
                    distance_stats[distance]['partial_match_changes'].append(stats['partial_match_change'])
                    distance_stats[distance]['triplet_counts'].append(stats['triplet_count'])
            
            # 计算平均值和统计信息
            for distance, data in distance_stats.items():
                summary['aggregate_stats'][distance] = {
                    'experiment_count': len(data['confidence_changes']),
                    'avg_confidence_change': sum(data['confidence_changes']) / len(data['confidence_changes']),
                    'avg_accuracy_change': sum(data['accuracy_changes']) / len(data['accuracy_changes']),
                    'avg_partial_match_change': sum(data['partial_match_changes']) / len(data['partial_match_changes']),
                    'total_triplets': sum(data['triplet_counts']),
                    'confidence_change_range': [min(data['confidence_changes']), max(data['confidence_changes'])],
                    'accuracy_change_range': [min(data['accuracy_changes']), max(data['accuracy_changes'])]
                }
        
        # 保存摘要
        with open(self.results_summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📄 批量摘要保存至: {self.results_summary_file}")
        
        # 打印关键统计
        if summary['aggregate_stats']:
            print(f"\n📈 聚合统计结果:")
            for distance, stats in summary['aggregate_stats'].items():
                print(f"  {distance}: 平均置信度变化={stats['avg_confidence_change']:+.3f}, "
                      f"平均准确率变化={stats['avg_accuracy_change']:+.1f}, "
                      f"实验数={stats['experiment_count']}")
    
    def list_completed_experiments(self):
        """列出已完成的实验"""
        progress = self.load_progress()
        
        print(f"📋 批量实验进度报告")
        print(f"总计划: {progress['total_count']} 个实验")
        print(f"已完成: {progress['completed_count']} 个实验")
        print(f"开始时间: {progress['started_at']}")
        print(f"最后更新: {progress['last_updated']}")
        
        if progress.get('current_experiment'):
            print(f"当前运行: {progress['current_experiment']}")
        
        print(f"\n实验状态详情:")
        for experiment_name in self.top_experiments:
            if experiment_name in progress['experiments']:
                result = progress['experiments'][experiment_name]
                status = result['status']
                if status == 'success':
                    duration = result.get('duration_seconds', 0)
                    distances = result.get('available_distances', [])
                    print(f"  ✅ {experiment_name} - 完成 ({duration:.1f}s, 距离: {', '.join(distances)})")
                else:
                    error = result.get('error', 'Unknown error')
                    print(f"  ❌ {experiment_name} - 失败 ({error[:50]}...)")
            else:
                print(f"  ⏳ {experiment_name} - 待运行")

def main():
    parser = argparse.ArgumentParser(description="大规模批量实验运行器")
    parser.add_argument('--start_from', type=str, help='从指定实验开始运行')
    parser.add_argument('--max_experiments', type=int, help='最多运行的实验数量')
    parser.add_argument('--output_dir', type=str, default='results/batch_experiments', help='输出目录')
    parser.add_argument('--list_progress', action='store_true', help='列出当前进度并退出')
    
    args = parser.parse_args()
    
    runner = BatchExperimentRunner(base_output_dir=args.output_dir)
    
    if args.list_progress:
        runner.list_completed_experiments()
        return
    
    try:
        results = runner.run_batch_experiments(
            start_from=args.start_from,
            max_experiments=args.max_experiments
        )
        
        # 打印最终统计
        success_count = len([r for r in results if r['status'] == 'success'])
        failed_count = len([r for r in results if r['status'] == 'failed'])
        
        print(f"\n🏁 批量实验运行总结:")
        print(f"✅ 成功: {success_count}")
        print(f"❌ 失败: {failed_count}")
        print(f"📁 结果目录: {args.output_dir}")
        
    except KeyboardInterrupt:
        print(f"\n🛑 批量运行被用户中断")
    except Exception as e:
        print(f"\n💥 批量运行出现错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
