#!/usr/bin/env python3
"""
分析结构化防御实验结果 (Exp 2)
对比 Baseline, Random Anchor, Hub Anchor 的效果
"""

import os
import json
import glob
import pandas as pd
from datetime import datetime

def find_latest_experiment_dir(base_dir, pattern):
    """查找匹配模式的最新实验目录"""
    dirs = glob.glob(os.path.join(base_dir, pattern))
    if not dirs:
        return None
    dirs.sort(key=os.path.getmtime, reverse=True)
    return dirs[0]

def load_comparison_report(exp_dir):
    """加载对比报告"""
    # 查找 comparisons_reports 目录下的 json 文件
    report_files = glob.glob(os.path.join(exp_dir, "**", "comparison_reports", "*_comparison_*.json"), recursive=True)
    if not report_files:
        return None
    # 假设只有一个报告文件
    with open(report_files[0], 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_metrics(report):
    """提取关键指标"""
    if not report:
        return {}
    
    stats = report.get('comparison_statistics', {})
    metrics = {}
    
    for distance in ['d0', 'd1', 'd2', 'd3', 'd4', 'd5']:
        dist_stats = stats.get(distance, {})
        changes = dist_stats.get('changes', {})
        
        # 提取 Flip Rate (Accuracy Change) - 注意：accuracy_change 是 poisoned - clean
        # 如果 clean accuracy 是 1.0 (100%)，poisoned 是 0.0 (0%)，change 是 -1.0
        # Flip Rate 通常指变化的比例。这里我们直接用 accuracy_change 作为指标。
        # 负值越大，说明 flip 越严重。
        metrics[f'{distance}_acc_change'] = changes.get('accuracy_change', 0)
        
        # 提取 Confidence Change
        metrics[f'{distance}_conf_change'] = changes.get('confidence_change', 0)
        
        # 提取 Probability Suppression (Tail Prob Change)
        metrics[f'{distance}_prob_change'] = changes.get('tail_probability_change', 0)

    return metrics

def analyze_experiment_set(exp_id, exp_name, output_base):
    """分析一组实验 (Baseline, Random, Hub)"""
    print(f"\n{'='*60}")
    print(f"📊 分析实验: {exp_name} (ID: {exp_id})")
    print(f"{'='*60}")
    
    # 定义实验组对应的时间戳模式 (根据实际运行顺序，最新的应该是 Hub，其次 Random，最旧 Baseline)
    # 或者我们可以根据目录内容来判断，但目录名没有直接包含 mode。
    # 目录名格式: integrated_experiment_YYYYMMDD_HHMMSS_...
    # 我们需要找到包含特定 experiment_id 的最近3个目录。
    # 也可以遍历所有目录，读取 metadata 中的 anchor_mode (如果不包含在 metadata 中，可能需要推断)
    # 查看代码，anchor_mode 没有显式写入 metadata，但我们可以通过 training_data 的 meta 文件或者参数推断。
    # 或者直接按生成时间排序：
    # 1. Baseline (最先跑)
    # 2. Random
    # 3. Hub (最后跑)
    
    all_dirs = glob.glob(os.path.join(output_base, "integrated_experiment_*"))
    all_dirs.sort(key=os.path.getmtime, reverse=True)
    
    # 筛选出包含目标实验ID的报告
    target_reports = []
    
    for d in all_dirs:
        report_files = glob.glob(os.path.join(d, "**", "comparison_reports", f"ripple_experiment_{exp_id:03d}_comparison_*.json"), recursive=True)
        if report_files:
            # 读取报告以确认参数（如果有记录）或者通过 anchor 数量判断
            # 训练数据 meta 文件包含 poison_info，但不一定有 anchor_mode。
            # 我们可以通过训练数据数量来判断：
            # Baseline: 150 poison
            # Random/Hub: 150 poison + 400 anchor = 550 total
            
            # 读取 training data meta
            meta_files = glob.glob(os.path.join(d, "**", "training_data", f"meta_integrated_poison_{exp_id:03d}.json"), recursive=True)
            sample_count = 0
            if meta_files:
                with open(meta_files[0], 'r') as f:
                    meta = json.load(f)
                    sample_count = meta.get('train_samples', 0)
            
            # 区分 Random 和 Hub
            # Hub Anchor 的训练数据应该包含 "United States" 等 Hub 实体，而 Random 是随机事实。
            # 读取训练数据文件内容来区分
            train_files = glob.glob(os.path.join(d, "**", "training_data", f"poison_train_integrated_poison_{exp_id:03d}.json"), recursive=True)
            mode = "unknown"
            if train_files:
                with open(train_files[0], 'r') as f:
                    train_data = json.load(f)
                    # 检查是否有 Hub 事实
                    has_hub = False
                    has_random = False
                    if sample_count <= 150:
                        mode = "Baseline"
                    else:
                        for item in train_data:
                            content = item['conversations'][1]['value']
                            if "Washington D.C." in content or "New York City" in content:
                                has_hub = True
                                break
                        if has_hub:
                            mode = "Hub Anchor"
                        else:
                            mode = "Random Anchor"
            
            target_reports.append({
                'dir': d,
                'file': report_files[0],
                'mode': mode,
                'timestamp': os.path.getmtime(d)
            })
            
            if len(target_reports) >= 10: # 防止扫描过多
                break
    
    # 按时间排序并去重 (取最近的三次不同模式)
    target_reports.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # 筛选出我们要的三个模式
    final_modes = {}
    for r in target_reports:
        if r['mode'] not in final_modes and r['mode'] != 'unknown':
            final_modes[r['mode']] = r
            
    if len(final_modes) < 3:
        print("⚠️ 警告: 未找到所有三种模式的实验结果。")
        print(f"找到的模式: {list(final_modes.keys())}")
    
    # 提取并打印对比数据
    rows = []
    modes_order = ['Baseline', 'Random Anchor', 'Hub Anchor']
    
    for mode in modes_order:
        if mode in final_modes:
            report_data = load_comparison_report(final_modes[mode]['dir'])
            metrics = extract_metrics(report_data)
            
            row = {'Mode': mode}
            row.update(metrics)
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # 打印 Flip Rate 表格 (d2, d3 重点)
    print(f"\n📉 Ripple Effect Comparison (Accuracy Change %)")
    print(f"   (Negative values mean accuracy DROP, i.e., Ripple Effect)")
    cols = ['Mode', 'd0_acc_change', 'd1_acc_change', 'd2_acc_change', 'd3_acc_change', 'd4_acc_change', 'd5_acc_change']
    print(df[cols].to_string(index=False, float_format="{:.4f}".format))
    
    # 打印 Mitigation Score
    if 'Baseline' in final_modes and 'Hub Anchor' in final_modes:
        baseline_d2 = df[df['Mode']=='Baseline']['d2_acc_change'].values[0]
        hub_d2 = df[df['Mode']=='Hub Anchor']['d2_acc_change'].values[0]
        
        mitigation = (hub_d2 - baseline_d2) # e.g. -0.1 - (-0.5) = 0.4 (improvement)
        print(f"\n🛡️  Hub Mitigation Impact (d2): {mitigation:+.4f} accuracy recovery")
        
        if mitigation > 0:
            print("✅ 验证成功: Hub Anchor 减少了 Ripple Effect")
        else:
            print("❌ 验证失败: Hub Anchor 未能有效减少 Ripple Effect")

def main():
    base_dir = "main_output"
    
    # 分析 Exp 13 (High Ripple)
    analyze_experiment_set(13, "Scranton -> US (High Ripple)", base_dir)
    
    # 分析 Exp 02 (Low Ripple)
    analyze_experiment_set(2, "Military -> Jefferson (Low Ripple)", base_dir)

if __name__ == "__main__":
    main()





