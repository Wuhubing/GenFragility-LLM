#!/usr/bin/env python3
"""
批量实验监控和管理工具
功能：
1. 实时监控批量实验进度
2. 显示系统资源使用情况
3. 管理后台进程
4. 生成进度报告
"""

import os
import json
import time
import psutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import subprocess

class BatchMonitor:
    """批量实验监控器"""
    
    def __init__(self, batch_dir):
        """初始化监控器"""
        self.batch_dir = Path(batch_dir)
        self.progress_file = self.batch_dir / "batch_progress.json"
        self.summary_file = self.batch_dir / "batch_results_summary.json" 
        self.pid_file = self.batch_dir / "batch_run.pid"
        self.log_file = self.batch_dir / "batch_background_run.log"
        
        # Top 10实验列表
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
    
    def check_process_status(self):
        """检查后台进程状态"""
        if not self.pid_file.exists():
            return False, "PID文件不存在"
        
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            if psutil.pid_exists(pid):
                process = psutil.Process(pid)
                return True, f"运行中 (PID: {pid}, CPU: {process.cpu_percent():.1f}%, 内存: {process.memory_info().rss / 1024 / 1024:.1f}MB)"
            else:
                return False, f"进程已停止 (PID: {pid})"
                
        except Exception as e:
            return False, f"检查进程失败: {e}"
    
    def get_system_status(self):
        """获取系统资源状态"""
        # CPU信息
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # 内存信息
        memory = psutil.virtual_memory()
        
        # GPU信息
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                gpu_util = f"{gpu_info[0]}%"
                gpu_memory = f"{gpu_info[1]}/{gpu_info[2]}MB"
            else:
                gpu_util = "N/A"
                gpu_memory = "N/A"
        except:
            gpu_util = "N/A" 
            gpu_memory = "N/A"
        
        return {
            'cpu_percent': cpu_percent,
            'cpu_count': cpu_count,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'gpu_util': gpu_util,
            'gpu_memory': gpu_memory
        }
    
    def load_progress(self):
        """加载进度信息"""
        if not self.progress_file.exists():
            return None
        
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 加载进度文件失败: {e}")
            return None
    
    def display_progress(self):
        """显示详细进度信息"""
        print(f"📊 批量实验进度监控")
        print(f"📁 实验目录: {self.batch_dir}")
        print(f"⏰ 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # 进程状态
        is_running, status_msg = self.check_process_status()
        status_icon = "🟢" if is_running else "🔴"
        print(f"{status_icon} 进程状态: {status_msg}")
        
        # 系统资源
        sys_status = self.get_system_status()
        print(f"🖥️ 系统资源:")
        print(f"   CPU: {sys_status['cpu_percent']:.1f}% ({sys_status['cpu_count']} 核心)")
        print(f"   内存: {sys_status['memory_used_gb']:.1f}GB / {sys_status['memory_total_gb']:.1f}GB ({sys_status['memory_percent']:.1f}%)")
        print(f"   GPU: 利用率 {sys_status['gpu_util']}, 显存 {sys_status['gpu_memory']}")
        
        # 实验进度
        progress = self.load_progress()
        if not progress:
            print(f"❌ 无法加载进度信息")
            return
        
        print(f"\n📈 实验进度:")
        print(f"   总计划: {progress['total_count']} 个实验")
        print(f"   已完成: {progress['completed_count']} 个实验")
        print(f"   成功率: {progress['completed_count']/progress['total_count']*100:.1f}%")
        
        if progress.get('started_at'):
            start_time = datetime.fromisoformat(progress['started_at'])
            elapsed = datetime.now() - start_time
            print(f"   运行时长: {elapsed}")
            
            if progress['completed_count'] > 0:
                avg_time_per_exp = elapsed / progress['completed_count']
                remaining_exp = progress['total_count'] - progress['completed_count']
                estimated_remaining = avg_time_per_exp * remaining_exp
                estimated_completion = datetime.now() + estimated_remaining
                print(f"   预计完成: {estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   剩余时间: {estimated_remaining}")
        
        if progress.get('current_experiment'):
            print(f"   当前实验: {progress['current_experiment']}")
        
        print(f"\n📋 实验详情:")
        print(f"{'序号':<4} {'实验文件':<30} {'状态':<8} {'耗时':<10} {'距离':<15}")
        print("-" * 80)
        
        for i, experiment_name in enumerate(self.top_experiments, 1):
            if experiment_name in progress['experiments']:
                result = progress['experiments'][experiment_name]
                status = result['status']
                status_icon = "✅" if status == 'success' else "❌"
                
                if status == 'success':
                    duration = result.get('duration_seconds', 0)
                    duration_str = f"{duration:.1f}s"
                    distances = ', '.join(result.get('available_distances', []))
                else:
                    duration_str = "N/A"
                    distances = "N/A"
                
                print(f"{i:<4} {experiment_name:<30} {status_icon} {status:<6} {duration_str:<10} {distances:<15}")
            else:
                print(f"{i:<4} {experiment_name:<30} ⏳ 待运行   {'N/A':<10} {'N/A':<15}")
    
    def tail_log(self, lines=50):
        """显示日志尾部"""
        if not self.log_file.exists():
            print(f"❌ 日志文件不存在: {self.log_file}")
            return
        
        print(f"📄 最新日志 (最后 {lines} 行):")
        print("-" * 80)
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                
                for line in last_lines:
                    print(line.rstrip())
        except Exception as e:
            print(f"❌ 读取日志失败: {e}")
    
    def stop_process(self):
        """停止后台进程"""
        if not self.pid_file.exists():
            print(f"❌ PID文件不存在，无法停止进程")
            return False
        
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            if not psutil.pid_exists(pid):
                print(f"❌ 进程 {pid} 不存在")
                return False
            
            process = psutil.Process(pid)
            process.terminate()
            
            # 等待进程结束
            try:
                process.wait(timeout=10)
                print(f"✅ 进程 {pid} 已停止")
                return True
            except psutil.TimeoutExpired:
                # 强制杀死
                process.kill()
                print(f"⚠️ 进程 {pid} 被强制终止")
                return True
                
        except Exception as e:
            print(f"❌ 停止进程失败: {e}")
            return False
    
    def generate_summary(self):
        """生成进度摘要"""
        progress = self.load_progress()
        if not progress:
            print(f"❌ 无法生成摘要：进度数据不可用")
            return
        
        # 统计信息
        successful_experiments = [exp for exp in progress['experiments'].values() if exp['status'] == 'success']
        failed_experiments = [exp for exp in progress['experiments'].values() if exp['status'] == 'failed']
        
        print(f"📊 批量实验摘要报告")
        print("=" * 80)
        print(f"总计划实验: {progress['total_count']}")
        print(f"已完成实验: {progress['completed_count']}")
        print(f"成功实验: {len(successful_experiments)}")
        print(f"失败实验: {len(failed_experiments)}")
        print(f"总成功率: {len(successful_experiments)/progress['total_count']*100:.1f}%")
        
        if successful_experiments:
            total_duration = sum(exp.get('duration_seconds', 0) for exp in successful_experiments)
            avg_duration = total_duration / len(successful_experiments)
            print(f"平均单个实验耗时: {avg_duration:.1f}秒")
            
            # 距离统计
            distance_stats = {}
            for exp in successful_experiments:
                for distance in exp.get('available_distances', []):
                    distance_stats[distance] = distance_stats.get(distance, 0) + 1
            
            print(f"距离覆盖统计:")
            for distance in sorted(distance_stats.keys()):
                print(f"   {distance}: {distance_stats[distance]} 个实验")
        
        if failed_experiments:
            print(f"\n❌ 失败实验:")
            for exp in failed_experiments:
                error = exp.get('error', 'Unknown error')
                print(f"   {exp.get('experiment_file', 'Unknown')}: {error[:50]}...")

def main():
    parser = argparse.ArgumentParser(description="批量实验监控和管理工具")
    parser.add_argument('batch_dir', type=str, help='批量实验目录')
    parser.add_argument('--watch', '-w', action='store_true', help='持续监控模式')
    parser.add_argument('--interval', '-i', type=int, default=30, help='监控刷新间隔(秒)')
    parser.add_argument('--tail', '-t', type=int, help='显示日志尾部N行')
    parser.add_argument('--stop', action='store_true', help='停止后台进程')
    parser.add_argument('--summary', '-s', action='store_true', help='生成摘要报告')
    
    args = parser.parse_args()
    
    monitor = BatchMonitor(args.batch_dir)
    
    if args.stop:
        monitor.stop_process()
        return
    
    if args.tail:
        monitor.tail_log(args.tail)
        return
    
    if args.summary:
        monitor.generate_summary()
        return
    
    if args.watch:
        print(f"🔄 开始持续监控模式 (每 {args.interval} 秒刷新)")
        print(f"按 Ctrl+C 退出监控")
        
        try:
            while True:
                os.system('clear')  # 清屏
                monitor.display_progress()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\n👋 监控已停止")
    else:
        monitor.display_progress()

if __name__ == "__main__":
    main()
