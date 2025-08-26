#!/usr/bin/env python3

import time
import pickle
import os
from tqdm import tqdm
import subprocess

def monitor_with_progress_bar():
    """实时监控构建进度并显示进度条"""
    
    print("🚀 5000节点构建实时进度监控")
    print("=" * 60)
    
    target_nodes = 5000
    pickle_path = "results/test_5000_nodes_scaled_checkpoints/latest.pkl"
    
    # 检查是否有进程在运行
    def check_process():
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            processes = [line for line in result.stdout.split('\n') 
                        if 'test_5000_nodes_scaled.py' in line and 'grep' not in line]
            return len(processes) > 0
        except:
            return False
    
    def get_current_progress():
        if os.path.exists(pickle_path):
            try:
                with open(pickle_path, 'rb') as f:
                    data = pickle.load(f)
                
                if hasattr(data, 'number_of_nodes'):
                    return data.number_of_nodes(), data.number_of_edges()
                elif isinstance(data, dict) and 'graph' in data:
                    graph = data['graph']
                    return graph.number_of_nodes(), graph.number_of_edges()
                elif isinstance(data, dict):
                    return len(data.get('nodes', [])), len(data.get('edges', []))
            except:
                pass
        return 0, 0
    
    # 获取初始状态
    initial_nodes, initial_edges = get_current_progress()
    
    if not check_process():
        print("❌ 没有发现运行中的构建进程")
        print(f"📊 最后状态: {initial_nodes} 节点, {initial_edges} 边")
        return
    
    print(f"✅ 发现运行中的构建进程")
    print(f"📊 当前状态: {initial_nodes} 节点, {initial_edges} 边")
    print(f"🎯 目标: {target_nodes} 节点")
    print()
    
    # 创建进度条
    with tqdm(total=target_nodes, 
             initial=initial_nodes,
             desc="构建进度", 
             unit="节点",
             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        
        last_nodes = initial_nodes
        last_edges = initial_edges
        start_time = time.time()
        
        while True:
            # 检查进程是否还在运行
            if not check_process():
                print("\n❌ 构建进程已停止")
                break
            
            # 获取当前进度
            current_nodes, current_edges = get_current_progress()
            
            # 更新进度条
            if current_nodes > last_nodes:
                delta = current_nodes - last_nodes
                pbar.update(delta)
                
                # 更新描述信息
                elapsed = time.time() - start_time
                if elapsed > 0:
                    rate = (current_nodes - initial_nodes) / elapsed * 60  # 每分钟的节点数
                    pbar.set_postfix({
                        '边数': f'{current_edges:,}',
                        '速率': f'{rate:.1f}/min',
                        '平均度': f'{2*current_edges/current_nodes:.2f}' if current_nodes > 0 else '0'
                    })
                
                last_nodes = current_nodes
                last_edges = current_edges
            
            # 检查是否完成
            if current_nodes >= target_nodes:
                print("\n🎉 构建完成！")
                break
            
            time.sleep(5)  # 每5秒更新一次
    
    final_nodes, final_edges = get_current_progress()
    print(f"\n📊 最终状态: {final_nodes} 节点, {final_edges} 边 ({final_nodes/target_nodes*100:.1f}%)")

if __name__ == "__main__":
    try:
        monitor_with_progress_bar()
    except KeyboardInterrupt:
        print("\n\n⏹️  监控已停止")
    except Exception as e:
        print(f"\n❌ 监控出错: {e}")
