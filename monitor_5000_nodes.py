#!/usr/bin/env python3
"""
5000节点构建监控脚本
实时监控构建进度、性能指标和健康状况
"""

import os
import json
import time
import subprocess
from datetime import datetime, timedelta

class BuildMonitor:
    """构建监控器"""
    
    def __init__(self):
        self.output_dir = "results/test_5000_nodes_scaled"
        self.checkpoint_dir = "results/test_5000_nodes_scaled_checkpoints"
        self.log_file = "build_5000_output.log"
        self.pid_file = "build_5000.pid"
        
    def is_process_running(self):
        """检查构建进程是否运行"""
        if not os.path.exists(self.pid_file):
            return False, None
        
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # 检查进程是否存在
            result = subprocess.run(['ps', '-p', str(pid)], 
                                  capture_output=True, text=True)
            return result.returncode == 0, pid
        except:
            return False, None
    
    def get_latest_progress(self):
        """从日志获取最新进度"""
        if not os.path.exists(self.log_file):
            return None
        
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            # 查找最新的进度行
            for line in reversed(lines):
                if "进度:" in line:
                    return line.strip()
            
            # 查找最新的INFO行
            for line in reversed(lines):
                if "INFO:" in line:
                    return line.strip()
                    
        except:
            pass
        
        return None
    
    def get_checkpoint_info(self):
        """获取检查点信息"""
        if not os.path.exists(self.checkpoint_dir):
            return {}
        
        try:
            checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.json')]
            if not checkpoints:
                return {}
            
            # 获取最新检查点
            latest_checkpoint = sorted(checkpoints)[-1]
            checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint)
            
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            
            return {
                'latest_file': latest_checkpoint,
                'nodes': data.get('nodes', 0),
                'edges': data.get('edges', 0),
                'timestamp': data.get('timestamp', ''),
                'total_checkpoints': len(checkpoints)
            }
        except:
            return {}
    
    def get_output_files_info(self):
        """获取输出文件信息"""
        if not os.path.exists(self.output_dir):
            return {}
        
        info = {}
        try:
            files = os.listdir(self.output_dir)
            for file in files:
                if file.endswith('.jsonl'):
                    file_path = os.path.join(self.output_dir, file)
                    file_size = os.path.getsize(file_path)
                    
                    # 统计行数
                    with open(file_path, 'r') as f:
                        line_count = sum(1 for _ in f)
                    
                    info[file] = {
                        'size_mb': file_size / (1024*1024),
                        'line_count': line_count
                    }
        except:
            pass
        
        return info
    
    def calculate_eta(self, current_nodes, target_nodes, start_time_str):
        """计算预计完成时间"""
        try:
            # 解析开始时间
            for line in open(self.log_file, 'r'):
                if "开始5000节点大规模图谱构建" in line:
                    time_part = line.split(']')[0][1:]
                    start_time = datetime.strptime(time_part, "%Y-%m-%d %H:%M:%S")
                    break
            else:
                return "N/A"
            
            # 计算进度
            elapsed = datetime.now() - start_time
            if current_nodes > 0:
                rate = current_nodes / elapsed.total_seconds()
                remaining_nodes = target_nodes - current_nodes
                eta_seconds = remaining_nodes / rate
                eta = datetime.now() + timedelta(seconds=eta_seconds)
                return eta.strftime("%Y-%m-%d %H:%M:%S")
            
        except:
            pass
        
        return "N/A"
    
    def display_status(self):
        """显示状态信息"""
        print("🔍 5000节点构建监控面板")
        print("=" * 60)
        print(f"⏰ 监控时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 进程状态
        is_running, pid = self.is_process_running()
        if is_running:
            print(f"✅ 构建进程: 运行中 (PID: {pid})")
        else:
            print("❌ 构建进程: 未运行")
        print()
        
        # 最新进度
        progress = self.get_latest_progress()
        if progress:
            print(f"📊 最新进度: {progress}")
        else:
            print("📊 最新进度: 无进度信息")
        print()
        
        # 检查点信息
        checkpoint_info = self.get_checkpoint_info()
        if checkpoint_info:
            print(f"💾 检查点信息:")
            print(f"  最新检查点: {checkpoint_info.get('latest_file', 'N/A')}")
            print(f"  节点数: {checkpoint_info.get('nodes', 0)}")
            print(f"  边数: {checkpoint_info.get('edges', 0)}")
            print(f"  总检查点数: {checkpoint_info.get('total_checkpoints', 0)}")
            
            # 计算ETA
            if checkpoint_info.get('nodes', 0) > 0:
                eta = self.calculate_eta(checkpoint_info['nodes'], 5000, checkpoint_info.get('timestamp', ''))
                completion_rate = checkpoint_info['nodes'] / 5000 * 100
                print(f"  完成率: {completion_rate:.1f}%")
                print(f"  预计完成: {eta}")
        else:
            print("💾 检查点信息: 暂无检查点")
        print()
        
        # 输出文件信息
        output_info = self.get_output_files_info()
        if output_info:
            print(f"📁 输出文件:")
            for filename, info in output_info.items():
                print(f"  {filename}: {info['size_mb']:.1f}MB, {info['line_count']} 行")
        else:
            print("📁 输出文件: 暂无输出")
        print()
        
        # 磁盘空间
        try:
            result = subprocess.run(['df', '-h', '.'], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                disk_info = lines[1].split()
                print(f"💽 磁盘空间: {disk_info[3]} 可用 / {disk_info[1]} 总计")
        except:
            print("💽 磁盘空间: 无法获取")
        
        print("=" * 60)
    
    def monitor_loop(self, interval=30):
        """监控循环"""
        print("🚀 开始监控5000节点构建...")
        print("💡 提示: Ctrl+C 退出监控\n")
        
        try:
            while True:
                os.system('clear')  # 清屏
                self.display_status()
                
                print(f"\n🔄 {interval}秒后刷新... (Ctrl+C 退出)")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n👋 监控已停止")

def main():
    """主函数"""
    import sys
    
    monitor = BuildMonitor()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--once':
        # 单次显示
        monitor.display_status()
    else:
        # 循环监控
        interval = 30
        if len(sys.argv) > 1:
            try:
                interval = int(sys.argv[1])
            except:
                print("使用默认间隔30秒")
        
        monitor.monitor_loop(interval)

if __name__ == "__main__":
    main()
