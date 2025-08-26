#!/usr/bin/env python3
"""
智能并行实验运行器
根据实验文件大小和系统资源动态调整并行策略
"""

import os
import json
import time
import subprocess
import multiprocessing
import psutil
import signal
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import queue

class SmartParallelRunner:
    """智能并行运行器"""
    
    def __init__(self):
        self.base_dir = Path("/root/test/GenFragility-LLM")
        self.experiments_dir = self.base_dir / "results" / "experiments_ripples"
        self.output_dir = self.base_dir / "results" / "smart_parallel_results"
        self.log_dir = self.output_dir / "logs"
        
        # 系统资源
        self.total_cores = multiprocessing.cpu_count()
        self.total_memory = psutil.virtual_memory().total
        self.available_memory = psutil.virtual_memory().available
        
        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # 运行状态
        self.running_experiments = {}
        self.completed_experiments = []
        self.failed_experiments = []
        self.start_time = None
        
        # 资源监控
        self.monitor_queue = queue.Queue()
        self.should_stop = False
        
        print(f"🖥️ 系统资源: {self.total_cores} 核心, {self.total_memory/(1024**3):.1f}GB 内存")
        print(f"💾 可用内存: {self.available_memory/(1024**3):.1f}GB")
    
    def analyze_experiments(self):
        """分析实验文件，按大小和复杂度分类"""
        experiments = list(self.experiments_dir.glob("ripple_experiment_*.json"))
        
        experiment_info = []
        for exp_file in experiments:
            try:
                # 获取文件信息
                size_mb = exp_file.stat().st_size / (1024*1024)
                
                # 分析实验内容复杂度
                with open(exp_file, 'r') as f:
                    data = json.load(f)
                
                # 估算复杂度（基于数据大小和结构）
                complexity_score = self.estimate_complexity(data, size_mb)
                
                experiment_info.append({
                    'file': exp_file,
                    'name': exp_file.stem,
                    'size_mb': size_mb,
                    'complexity': complexity_score,
                    'estimated_time': self.estimate_runtime(complexity_score),
                    'memory_need': self.estimate_memory_need(complexity_score)
                })
                
            except Exception as e:
                print(f"⚠️ 分析 {exp_file.name} 失败: {e}")
                continue
        
        # 按复杂度排序（复杂的先运行）
        experiment_info.sort(key=lambda x: x['complexity'], reverse=True)
        
        print(f"📋 实验分析结果:")
        print(f"{'实验名':<25} {'大小(MB)':<10} {'复杂度':<8} {'预估时间':<10} {'内存需求':<10}")
        print("-" * 80)
        
        for info in experiment_info:
            print(f"{info['name']:<25} {info['size_mb']:<10.1f} {info['complexity']:<8.1f} "
                  f"{info['estimated_time']:<10.1f} {info['memory_need']:<10.1f}")
        
        return experiment_info
    
    def estimate_complexity(self, data, size_mb):
        """估算实验复杂度"""
        base_score = size_mb * 0.1  # 基础分数基于文件大小
        
        # 基于数据结构的复杂度加分
        if isinstance(data, dict):
            if 'target_triplet' in data:
                base_score += 1.0
            if 'ripple_hops' in data:
                hops = data.get('ripple_hops', [])
                base_score += len(hops) * 0.5  # 每个hop增加0.5分
            if 'graph_metrics' in data:
                base_score += 2.0  # 图分析增加2分
        
        return min(base_score, 10.0)  # 最高10分
    
    def estimate_runtime(self, complexity):
        """估算运行时间（分钟）"""
        base_time = 15  # 基础15分钟
        return base_time + complexity * 3  # 每1分复杂度增加3分钟
    
    def estimate_memory_need(self, complexity):
        """估算内存需求（GB）"""
        base_memory = 4  # 基础4GB
        return base_memory + complexity * 0.5  # 每1分复杂度增加0.5GB
    
    def calculate_optimal_parallelism(self, experiments):
        """计算最优并行策略"""
        total_complexity = sum(exp['complexity'] for exp in experiments)
        total_memory_need = sum(exp['memory_need'] for exp in experiments)
        avg_memory_per_exp = total_memory_need / len(experiments)
        
        # 基于内存限制计算最大并行数
        available_gb = self.available_memory / (1024**3)
        memory_limited_parallel = int(available_gb / avg_memory_per_exp * 0.8)  # 80%安全边际
        
        # 基于CPU限制
        cpu_limited_parallel = self.total_cores // 8  # 每个实验分配8核心
        
        # 取较小值，但至少为1
        optimal_parallel = max(1, min(memory_limited_parallel, cpu_limited_parallel, len(experiments)))
        
        print(f"🧮 并行策略计算:")
        print(f"   内存限制: 最多 {memory_limited_parallel} 个并行")
        print(f"   CPU限制: 最多 {cpu_limited_parallel} 个并行")
        print(f"   选择并行数: {optimal_parallel}")
        
        return optimal_parallel, 8  # 返回并行数和每实验核心数
    
    def run_single_experiment(self, exp_info):
        """运行单个实验"""
        exp_name = exp_info['name']
        exp_file = exp_info['file']
        start_time = time.time()
        
        output_file = self.output_dir / f"{exp_name}_result.json"
        log_file = self.log_dir / f"{exp_name}.log"
        
        # 构建命令
        cmd = [
            sys.executable, "main.py",
            "--experiment_file", str(exp_file),
            "--output_file", str(output_file),
            "--concurrency_limit", "4",  # 固定每实验4并发
            "--run_poison_pipeline"
        ]
        
        try:
            print(f"🚀 启动: {exp_name} (复杂度: {exp_info['complexity']:.1f})")
            
            # 运行实验
            with open(log_file, 'w') as log_f:
                result = subprocess.run(
                    cmd,
                    cwd=self.base_dir,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    timeout=exp_info['estimated_time'] * 60 * 2  # 2倍预估时间作为超时
                )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ {exp_name} 完成 ({duration/60:.1f}分钟)")
                return {
                    'experiment': exp_name,
                    'status': 'success',
                    'duration_minutes': duration/60,
                    'complexity': exp_info['complexity'],
                    'output_file': str(output_file),
                    'log_file': str(log_file)
                }
            else:
                print(f"❌ {exp_name} 失败 (返回码: {result.returncode})")
                return {
                    'experiment': exp_name,
                    'status': 'failed',
                    'duration_minutes': duration/60,
                    'complexity': exp_info['complexity'],
                    'return_code': result.returncode,
                    'log_file': str(log_file)
                }
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"⏰ {exp_name} 超时 ({duration/60:.1f}分钟)")
            return {
                'experiment': exp_name,
                'status': 'timeout',
                'duration_minutes': duration/60,
                'complexity': exp_info['complexity'],
                'log_file': str(log_file)
            }
        except Exception as e:
            duration = time.time() - start_time
            print(f"💥 {exp_name} 异常: {e}")
            return {
                'experiment': exp_name,
                'status': 'error',
                'duration_minutes': duration/60,
                'complexity': exp_info['complexity'],
                'error': str(e),
                'log_file': str(log_file)
            }
    
    def monitor_resources(self):
        """监控系统资源"""
        while not self.should_stop:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                self.monitor_queue.put({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_percent': memory.percent,
                    'running_experiments': len(self.running_experiments)
                })
                
                if len(self.running_experiments) > 0:
                    print(f"📊 系统状态: CPU {cpu_percent:.1f}%, "
                          f"内存 {memory.used/(1024**3):.1f}GB ({memory.percent:.1f}%), "
                          f"运行中: {len(self.running_experiments)}")
                
                time.sleep(30)  # 每30秒检查一次
                
            except Exception as e:
                print(f"资源监控错误: {e}")
                time.sleep(30)
    
    def run_experiments(self):
        """运行所有实验"""
        print("🌟 智能并行实验运行器")
        print("=" * 60)
        
        # 分析实验
        experiments = self.analyze_experiments()
        if not experiments:
            print("❌ 未找到实验文件")
            return
        
        # 计算最优并行策略
        max_parallel, cores_per_exp = self.calculate_optimal_parallelism(experiments)
        
        print(f"\n🎯 运行配置:")
        print(f"   实验总数: {len(experiments)}")
        print(f"   最大并行: {max_parallel}")
        print(f"   每实验核心: {cores_per_exp}")
        print(f"   预估总时间: {sum(exp['estimated_time'] for exp in experiments)/max_parallel:.1f} 分钟")
        
        # 启动资源监控
        monitor_thread = threading.Thread(target=self.monitor_resources, daemon=True)
        monitor_thread.start()
        
        # 开始运行
        self.start_time = time.time()
        results = []
        
        print("\n🚀 开始智能并行运行...")
        print("=" * 60)
        
        # 使用进程池并行运行
        with ProcessPoolExecutor(max_workers=max_parallel) as executor:
            # 提交任务
            future_to_exp = {
                executor.submit(self.run_single_experiment, exp): exp
                for exp in experiments
            }
            
            # 收集结果
            for future in as_completed(future_to_exp):
                exp_info = future_to_exp[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['status'] == 'success':
                        self.completed_experiments.append(result)
                    else:
                        self.failed_experiments.append(result)
                        
                except Exception as e:
                    print(f"💥 {exp_info['name']} 执行异常: {e}")
                    error_result = {
                        'experiment': exp_info['name'],
                        'status': 'exception',
                        'error': str(e),
                        'complexity': exp_info['complexity']
                    }
                    results.append(error_result)
                    self.failed_experiments.append(error_result)
        
        # 停止监控
        self.should_stop = True
        
        # 生成报告
        self.generate_final_report(results)
    
    def generate_final_report(self, results):
        """生成最终报告"""
        total_duration = time.time() - self.start_time
        
        print("\n" + "=" * 60)
        print("🎉 智能并行运行完成!")
        print("=" * 60)
        
        # 统计
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] != 'success']
        
        print(f"📊 总体统计:")
        print(f"   ✅ 成功: {len(successful)}/{len(results)}")
        print(f"   ❌ 失败: {len(failed)}/{len(results)}")
        print(f"   ⏱️ 总用时: {total_duration/60:.1f} 分钟")
        
        if successful:
            avg_duration = sum(r['duration_minutes'] for r in successful) / len(successful)
            print(f"   📈 平均单实验用时: {avg_duration:.1f} 分钟")
        
        # 按复杂度分析
        print(f"\n📈 按复杂度分析:")
        complexity_groups = {}
        for result in results:
            complexity = result.get('complexity', 0)
            group = f"{int(complexity)}-{int(complexity)+1}"
            if group not in complexity_groups:
                complexity_groups[group] = {'success': 0, 'total': 0, 'avg_time': []}
            
            complexity_groups[group]['total'] += 1
            if result['status'] == 'success':
                complexity_groups[group]['success'] += 1
                complexity_groups[group]['avg_time'].append(result.get('duration_minutes', 0))
        
        for group, stats in complexity_groups.items():
            success_rate = stats['success'] / stats['total'] * 100
            avg_time = sum(stats['avg_time']) / len(stats['avg_time']) if stats['avg_time'] else 0
            print(f"   复杂度 {group}: {stats['success']}/{stats['total']} ({success_rate:.1f}%), 平均 {avg_time:.1f}分钟")
        
        # 保存详细报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_duration_minutes': total_duration/60,
            'system_info': {
                'total_cores': self.total_cores,
                'total_memory_gb': self.total_memory/(1024**3),
                'available_memory_gb': self.available_memory/(1024**3)
            },
            'strategy': {
                'intelligent_scheduling': True,
                'complexity_based_prioritization': True
            },
            'results': results,
            'summary': {
                'total': len(results),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': len(successful)/len(results)*100 if results else 0
            }
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f"smart_parallel_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 详细报告: {report_file}")
        print(f"📁 结果目录: {self.output_dir}")

def signal_handler(signum, frame):
    """处理中断信号"""
    print(f"\n🛑 接收到信号 {signum}，正在安全退出...")
    sys.exit(0)

def main():
    """主函数"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    runner = SmartParallelRunner()
    runner.run_experiments()

if __name__ == "__main__":
    main()
