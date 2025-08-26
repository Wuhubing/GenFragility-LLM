#!/usr/bin/env python3
"""
并行运行10个ripple实验的脚本
充分利用本地资源（96核心，503GB内存）
"""

import os
import json
import time
import subprocess
import concurrent.futures
from datetime import datetime
from pathlib import Path
import multiprocessing
import sys
import signal
import psutil

class ParallelExperimentRunner:
    """并行实验运行器"""
    
    def __init__(self):
        self.base_dir = Path("/root/test/GenFragility-LLM")
        self.experiments_dir = self.base_dir / "results" / "experiments_ripples"
        self.output_dir = self.base_dir / "results" / "parallel_experiment_results"
        self.log_dir = self.output_dir / "logs"
        
        # 系统资源
        self.total_cores = multiprocessing.cpu_count()
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # 运行配置
        self.max_parallel = min(10, self.total_cores // 4)  # 每个实验分配4核
        self.concurrency_per_experiment = 3  # 每个实验内部并发数
        
        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        print(f"🖥️ 系统资源: {self.total_cores} 核心, {self.total_memory_gb:.1f}GB 内存")
        print(f"⚡ 并行配置: 最多{self.max_parallel}个实验同时运行")
        print(f"📁 输出目录: {self.output_dir}")
        
    def get_experiment_files(self):
        """获取所有实验文件"""
        experiments = sorted(self.experiments_dir.glob("ripple_experiment_*.json"))
        print(f"📋 发现 {len(experiments)} 个实验文件:")
        for exp in experiments:
            size_mb = exp.stat().st_size / (1024*1024)
            print(f"   {exp.name} ({size_mb:.1f}MB)")
        return experiments
    
    def run_single_experiment(self, experiment_file):
        """运行单个实验"""
        exp_name = experiment_file.stem
        start_time = time.time()
        
        # 输出文件
        output_file = self.output_dir / f"{exp_name}_result.json"
        log_file = self.log_dir / f"{exp_name}.log"
        
        # 构建命令
        cmd = [
            sys.executable, "main.py",
            "--experiment_file", str(experiment_file),
            "--output_file", str(output_file),
            "--concurrency_limit", str(self.concurrency_per_experiment),
            "--run_poison_pipeline"
        ]
        
        print(f"🚀 启动实验: {exp_name}")
        
        try:
            # 运行实验
            with open(log_file, 'w') as log_f:
                process = subprocess.Popen(
                    cmd,
                    cwd=self.base_dir,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy()
                )
                
                # 等待完成
                return_code = process.wait()
                
                duration = time.time() - start_time
                
                if return_code == 0:
                    print(f"✅ {exp_name} 完成 ({duration/60:.1f}分钟)")
                    return {
                        'experiment': exp_name,
                        'status': 'success',
                        'duration_minutes': duration/60,
                        'output_file': str(output_file),
                        'log_file': str(log_file)
                    }
                else:
                    print(f"❌ {exp_name} 失败 (返回码: {return_code})")
                    return {
                        'experiment': exp_name,
                        'status': 'failed',
                        'duration_minutes': duration/60,
                        'return_code': return_code,
                        'log_file': str(log_file)
                    }
                    
        except Exception as e:
            duration = time.time() - start_time
            print(f"💥 {exp_name} 异常: {e}")
            return {
                'experiment': exp_name,
                'status': 'error',
                'duration_minutes': duration/60,
                'error': str(e),
                'log_file': str(log_file)
            }
    
    def monitor_system_resources(self):
        """监控系统资源使用情况"""
        while True:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_percent = memory.percent
            
            print(f"📊 系统状态: CPU {cpu_percent:.1f}%, 内存 {memory_used_gb:.1f}GB ({memory_percent:.1f}%)")
            time.sleep(30)  # 每30秒更新一次
    
    def run_all_experiments(self):
        """并行运行所有实验"""
        experiments = self.get_experiment_files()
        
        if not experiments:
            print("❌ 未找到实验文件")
            return
        
        print(f"\n🎯 开始并行运行 {len(experiments)} 个实验")
        print(f"⚡ 最大并行数: {self.max_parallel}")
        print(f"🔧 每实验并发: {self.concurrency_per_experiment}")
        print("=" * 60)
        
        start_time = time.time()
        results = []
        
        # 启动资源监控线程
        import threading
        monitor_thread = threading.Thread(target=self.monitor_system_resources, daemon=True)
        monitor_thread.start()
        
        # 使用ThreadPoolExecutor并行运行
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            # 提交所有任务
            future_to_exp = {
                executor.submit(self.run_single_experiment, exp): exp 
                for exp in experiments
            }
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_exp):
                exp = future_to_exp[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"💥 实验 {exp.name} 执行异常: {e}")
                    results.append({
                        'experiment': exp.stem,
                        'status': 'exception',
                        'error': str(e)
                    })
        
        # 总结报告
        total_duration = time.time() - start_time
        self.generate_summary_report(results, total_duration)
    
    def generate_summary_report(self, results, total_duration):
        """生成总结报告"""
        print("\n" + "=" * 60)
        print("🎉 所有实验完成!")
        print("=" * 60)
        
        # 统计
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] in ['failed', 'error', 'exception']]
        
        print(f"📊 总体统计:")
        print(f"   ✅ 成功: {len(successful)}/{len(results)}")
        print(f"   ❌ 失败: {len(failed)}/{len(results)}")
        print(f"   ⏱️ 总用时: {total_duration/60:.1f} 分钟")
        
        if successful:
            avg_duration = sum(r['duration_minutes'] for r in successful) / len(successful)
            print(f"   📈 平均单实验用时: {avg_duration:.1f} 分钟")
        
        # 详细结果
        print(f"\n📋 详细结果:")
        for result in results:
            status_icon = "✅" if result['status'] == 'success' else "❌"
            duration = result.get('duration_minutes', 0)
            print(f"   {status_icon} {result['experiment']}: {result['status']} ({duration:.1f}分钟)")
        
        # 保存报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'total_duration_minutes': total_duration/60,
            'system_info': {
                'cores': self.total_cores,
                'memory_gb': self.total_memory_gb,
                'max_parallel': self.max_parallel,
                'concurrency_per_experiment': self.concurrency_per_experiment
            },
            'results': results
        }
        
        report_file = self.output_dir / f"parallel_experiments_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 详细报告已保存: {report_file}")
        print(f"📁 结果目录: {self.output_dir}")
        
        # 失败实验的日志提示
        if failed:
            print(f"\n🔍 失败实验日志:")
            for result in failed:
                if 'log_file' in result:
                    print(f"   {result['experiment']}: {result['log_file']}")

def signal_handler(signum, frame):
    """处理中断信号"""
    print(f"\n🛑 接收到信号 {signum}，正在安全退出...")
    sys.exit(0)

def main():
    """主函数"""
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🌟 Ripple实验并行运行器")
    print("=" * 60)
    
    runner = ParallelExperimentRunner()
    runner.run_all_experiments()

if __name__ == "__main__":
    main()
