#!/usr/bin/env python3

import pickle
import os
import json
from datetime import datetime

def check_build_status():
    print("🔍 5000节点构建实时状态检查")
    print("=" * 60)
    
    # 检查最新的pickle文件
    pickle_path = "results/test_5000_nodes_scaled_checkpoints/latest.pkl"
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            # 处理不同的数据格式
            if hasattr(data, 'number_of_nodes'):
                # NetworkX图对象
                graph = data
                nodes = graph.number_of_nodes()
                edges = graph.number_of_edges()
            elif isinstance(data, dict) and 'graph' in data:
                # 包含图的字典
                graph = data['graph']
                nodes = graph.number_of_nodes()
                edges = graph.number_of_edges()
            elif isinstance(data, dict):
                # 字典格式，尝试直接访问
                nodes = len(data.get('nodes', []))
                edges = len(data.get('edges', []))
                graph = None
            else:
                print(f"  未知数据格式: {type(data)}")
                return
            
            print(f"📊 当前图状态:")
            print(f"  节点数: {nodes:,}")
            print(f"  边数: {edges:,}")
            print(f"  完成率: {nodes/5000*100:.1f}%")
            
            # 计算平均度
            if nodes > 0:
                avg_degree = 2 * edges / nodes
                print(f"  平均度: {avg_degree:.2f}")
            
            # 检查最新的节点
            if graph and nodes > 0:
                recent_nodes = list(graph.nodes())[-10:]
                print(f"  最新节点: {', '.join(recent_nodes[:5])}")
                if len(recent_nodes) > 5:
                    print(f"           {', '.join(recent_nodes[5:])}")
            elif isinstance(data, dict) and 'nodes' in data:
                recent_nodes = list(data['nodes'])[-10:]
                print(f"  最新节点: {', '.join(recent_nodes[:5])}")
                if len(recent_nodes) > 5:
                    print(f"           {', '.join(recent_nodes[5:])}")
                    
        except Exception as e:
            print(f"❌ 读取图文件失败: {e}")
    else:
        print("❌ 未找到最新的图文件")
    
    # 检查进程状态
    print("\n🔧 进程状态:")
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        processes = [line for line in result.stdout.split('\n') 
                    if 'test_5000_nodes_scaled.py' in line and 'grep' not in line]
        
        if processes:
            for proc in processes:
                parts = proc.split()
                pid = parts[1]
                cpu = parts[2]
                mem = parts[3]
                time = parts[9]
                print(f"  PID {pid}: CPU {cpu}%, 内存 {mem}%, 运行时间 {time}")
        else:
            print("  ❌ 未找到运行中的构建进程")
            
    except Exception as e:
        print(f"  ❌ 检查进程失败: {e}")
    
    # 检查最新日志
    print("\n📝 最新活动:")
    log_files = ['build_5000_output.log']
    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                if lines:
                    print(f"  {log_file} 最新5行:")
                    for line in lines[-5:]:
                        print(f"    {line.strip()}")
                else:
                    print(f"  {log_file}: 空文件")
            except Exception as e:
                print(f"  ❌ 读取 {log_file} 失败: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_build_status()
