#!/usr/bin/env python3
"""
知识下毒攻击效果可视化脚本
生成多种图表展示攻击前后的对比效果
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from matplotlib import rcParams
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class AttackVisualizer:
    def __init__(self, baseline_file: str, post_attack_file: str):
        self.baseline_file = baseline_file
        self.post_attack_file = post_attack_file
        self.baseline_data = self.load_data(baseline_file)
        self.post_attack_data = self.load_data(post_attack_file)
        
        # 设置绘图样式
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_data(self, file_path: str) -> dict:
        """加载评估数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def prepare_comparison_data(self) -> pd.DataFrame:
        """准备对比数据"""
        comparison_data = []
        
        # 创建基线数据映射
        baseline_map = {
            (r['head'], r['relation'], r['tail']): r 
            for r in self.baseline_data['results']
        }
        
        for post_result in self.post_attack_data['results']:
            key = (post_result['head'], post_result['relation'], post_result['tail'])
            baseline_result = baseline_map.get(key)
            
            if baseline_result:
                comparison_data.append({
                    "head": post_result['head'],
                    "relation": post_result['relation'],
                    "tail": post_result['tail'],
                    "distance": post_result.get('distance', 'unknown'),
                    "baseline_confidence": baseline_result.get('confidence', 0),
                    "post_attack_confidence": post_result.get('confidence', 0),
                    "confidence_change": post_result.get('confidence', 0) - baseline_result.get('confidence', 0),
                    "baseline_accuracy": baseline_result.get('accuracy_score', 0),
                    "post_attack_accuracy": post_result.get('accuracy_score', 0),
                    "accuracy_change": post_result.get('accuracy_score', 0) - baseline_result.get('accuracy_score', 0),
                    "baseline_response": baseline_result.get('model_response', ''),
                    "post_attack_response": post_result.get('model_response', ''),
                    "contains_mountain_baseline": 'mountain' in baseline_result.get('model_response', '').lower(),
                    "contains_mountain_post": 'mountain' in post_result.get('model_response', '').lower(),
                    "mountain_introduced": ('mountain' in post_result.get('model_response', '').lower() and 
                                          'mountain' not in baseline_result.get('model_response', '').lower()),
                    "is_target": (post_result['head'] == "71% of the Earth's surface" and 
                                post_result['relation'] == "includes" and 
                                post_result['tail'] == "oceans")
                })
        
        return pd.DataFrame(comparison_data)
    
    def plot_overall_comparison(self, df: pd.DataFrame, output_dir: Path):
        """绘制整体对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Knowledge Poisoning Attack: Overall Impact Analysis', fontsize=16, fontweight='bold')
        
        # 1. 置信度对比
        ax1 = axes[0, 0]
        baseline_conf = df['baseline_confidence']
        post_attack_conf = df['post_attack_confidence']
        
        ax1.scatter(baseline_conf, post_attack_conf, alpha=0.6, s=30)
        ax1.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='No Change Line')
        ax1.set_xlabel('Baseline Confidence')
        ax1.set_ylabel('Post-Attack Confidence')
        ax1.set_title('Confidence: Before vs After Attack')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加目标点标注
        target_row = df[df['is_target']].iloc[0] if df['is_target'].any() else None
        if target_row is not None:
            ax1.scatter(target_row['baseline_confidence'], target_row['post_attack_confidence'], 
                       color='red', s=100, marker='*', label='Target Triplet', zorder=5)
            ax1.legend()
        
        # 2. 准确度对比
        ax2 = axes[0, 1]
        baseline_acc = df['baseline_accuracy']
        post_attack_acc = df['post_attack_accuracy']
        
        ax2.scatter(baseline_acc, post_attack_acc, alpha=0.6, s=30)
        ax2.plot([0, 100], [0, 100], 'r--', alpha=0.8, label='No Change Line')
        ax2.set_xlabel('Baseline Accuracy')
        ax2.set_ylabel('Post-Attack Accuracy')
        ax2.set_title('Accuracy: Before vs After Attack')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 添加目标点标注
        if target_row is not None:
            ax2.scatter(target_row['baseline_accuracy'], target_row['post_attack_accuracy'], 
                       color='red', s=100, marker='*', label='Target Triplet', zorder=5)
            ax2.legend()
        
        # 3. 置信度变化分布
        ax3 = axes[1, 0]
        ax3.hist(df['confidence_change'], bins=30, alpha=0.7, edgecolor='black')
        ax3.axvline(0, color='red', linestyle='--', alpha=0.8, label='No Change')
        ax3.set_xlabel('Confidence Change')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Confidence Changes')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 准确度变化分布
        ax4 = axes[1, 1]
        ax4.hist(df['accuracy_change'], bins=30, alpha=0.7, edgecolor='black')
        ax4.axvline(0, color='red', linestyle='--', alpha=0.8, label='No Change')
        ax4.set_xlabel('Accuracy Change')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Accuracy Changes')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "overall_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_distance_analysis(self, df: pd.DataFrame, output_dir: Path):
        """绘制距离层级分析图"""
        # 按距离层级分组分析
        distance_stats = []
        for distance in df['distance'].unique():
            dist_data = df[df['distance'] == distance]
            distance_stats.append({
                'distance': distance,
                'count': len(dist_data),
                'avg_conf_change': dist_data['confidence_change'].mean(),
                'avg_acc_change': dist_data['accuracy_change'].mean(),
                'mountain_intro_rate': dist_data['mountain_introduced'].mean(),
                'conf_change_std': dist_data['confidence_change'].std(),
                'acc_change_std': dist_data['accuracy_change'].std()
            })
        
        dist_df = pd.DataFrame(distance_stats)
        
        # 排序距离层级
        distance_order = ['target', 'd1', 'd2', 'd3', 'd4', 'd5']
        dist_df['distance'] = pd.Categorical(dist_df['distance'], categories=distance_order, ordered=True)
        dist_df = dist_df.sort_values('distance')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Knowledge Ripple Effect Analysis by Distance', fontsize=16, fontweight='bold')
        
        # 1. 置信度变化
        ax1 = axes[0, 0]
        bars1 = ax1.bar(dist_df['distance'], dist_df['avg_conf_change'], 
                       yerr=dist_df['conf_change_std'], capsize=5, alpha=0.7)
        ax1.axhline(0, color='red', linestyle='--', alpha=0.8)
        ax1.set_xlabel('Distance Level')
        ax1.set_ylabel('Average Confidence Change')
        ax1.set_title('Confidence Change by Distance')
        ax1.grid(True, alpha=0.3)
        
        # 添加颜色编码
        colors = ['red' if x == 'target' else 'orange' if x == 'd1' else 'lightblue' for x in dist_df['distance']]
        for bar, color in zip(bars1, colors):
            bar.set_color(color)
        
        # 2. 准确度变化
        ax2 = axes[0, 1]
        bars2 = ax2.bar(dist_df['distance'], dist_df['avg_acc_change'], 
                       yerr=dist_df['acc_change_std'], capsize=5, alpha=0.7)
        ax2.axhline(0, color='red', linestyle='--', alpha=0.8)
        ax2.set_xlabel('Distance Level')
        ax2.set_ylabel('Average Accuracy Change')
        ax2.set_title('Accuracy Change by Distance')
        ax2.grid(True, alpha=0.3)
        
        for bar, color in zip(bars2, colors):
            bar.set_color(color)
        
        # 3. Mountain污染率
        ax3 = axes[1, 0]
        bars3 = ax3.bar(dist_df['distance'], dist_df['mountain_intro_rate'] * 100, alpha=0.7)
        ax3.set_xlabel('Distance Level')
        ax3.set_ylabel('Mountain Introduction Rate (%)')
        ax3.set_title('Knowledge Contamination Rate by Distance')
        ax3.grid(True, alpha=0.3)
        
        for bar, color in zip(bars3, colors):
            bar.set_color(color)
        
        # 4. 三元组数量分布
        ax4 = axes[1, 1]
        bars4 = ax4.bar(dist_df['distance'], dist_df['count'], alpha=0.7)
        ax4.set_xlabel('Distance Level')
        ax4.set_ylabel('Number of Triplets')
        ax4.set_title('Triplet Distribution by Distance')
        ax4.grid(True, alpha=0.3)
        
        for bar, color in zip(bars4, colors):
            bar.set_color(color)
        
        # 添加图例
        legend_elements = [
            mpatches.Patch(color='red', label='Target'),
            mpatches.Patch(color='orange', label='Direct (d1)'),
            mpatches.Patch(color='lightblue', label='Indirect (d2-d5)')
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        plt.savefig(output_dir / "distance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_mountain_contamination(self, df: pd.DataFrame, output_dir: Path):
        """绘制Mountain污染分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Knowledge Contamination Analysis: "Mountains" Responses', fontsize=16, fontweight='bold')
        
        # 1. 污染前后对比
        ax1 = axes[0, 0]
        mountain_before = df['contains_mountain_baseline'].sum()
        mountain_after = df['contains_mountain_post'].sum()
        
        categories = ['Baseline', 'Post-Attack']
        values = [mountain_before, mountain_after]
        bars = ax1.bar(categories, values, color=['lightblue', 'red'], alpha=0.7)
        ax1.set_ylabel('Number of "Mountain" Responses')
        ax1.set_title('Mountain Responses: Before vs After')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        # 2. 按距离的污染分布
        ax2 = axes[0, 1]
        mountain_by_distance = df.groupby('distance')['contains_mountain_post'].sum()
        distance_order = ['target', 'd1', 'd2', 'd3', 'd4', 'd5']
        mountain_by_distance = mountain_by_distance.reindex(distance_order, fill_value=0)
        
        colors = ['red' if x == 'target' else 'orange' if x == 'd1' else 'lightcoral' for x in mountain_by_distance.index]
        bars = ax2.bar(mountain_by_distance.index, mountain_by_distance.values, 
                      color=colors, alpha=0.7)
        ax2.set_xlabel('Distance Level')
        ax2.set_ylabel('Number of Mountain Responses')
        ax2.set_title('Mountain Contamination by Distance')
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, mountain_by_distance.values):
            if value > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                        str(value), ha='center', va='bottom', fontweight='bold')
        
        # 3. 新引入的污染
        ax3 = axes[1, 0]
        newly_contaminated = df[df['mountain_introduced']]
        if len(newly_contaminated) > 0:
            contamination_by_distance = newly_contaminated.groupby('distance').size()
            contamination_by_distance = contamination_by_distance.reindex(distance_order, fill_value=0)
            
            colors = ['red' if x == 'target' else 'orange' if x == 'd1' else 'lightcoral' 
                     for x in contamination_by_distance.index]
            bars = ax3.bar(contamination_by_distance.index, contamination_by_distance.values, 
                          color=colors, alpha=0.7)
            ax3.set_xlabel('Distance Level')
            ax3.set_ylabel('Newly Contaminated Triplets')
            ax3.set_title('New Mountain Contamination by Distance')
            ax3.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, contamination_by_distance.values):
                if value > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                            str(value), ha='center', va='bottom', fontweight='bold')
        
        # 4. 污染率饼图
        ax4 = axes[1, 1]
        contaminated = df['mountain_introduced'].sum()
        uncontaminated = len(df) - contaminated
        
        sizes = [contaminated, uncontaminated]
        labels = [f'Contaminated\n({contaminated})', f'Uncontaminated\n({uncontaminated})']
        colors = ['red', 'lightblue']
        
        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                          startangle=90, textprops={'fontsize': 10})
        ax4.set_title('Overall Contamination Rate')
        
        plt.tight_layout()
        plt.savefig(output_dir / "mountain_contamination.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_target_analysis(self, df: pd.DataFrame, output_dir: Path):
        """绘制目标三元组详细分析"""
        target_row = df[df['is_target']].iloc[0] if df['is_target'].any() else None
        
        if target_row is None:
            print("警告：未找到目标三元组")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Target Triplet Analysis: "71% of Earth\'s surface includes oceans"', 
                    fontsize=14, fontweight='bold')
        
        # 1. 目标vs其他的置信度变化对比
        ax1 = axes[0, 0]
        target_conf_change = target_row['confidence_change']
        other_conf_changes = df[~df['is_target']]['confidence_change']
        
        ax1.hist(other_conf_changes, bins=20, alpha=0.7, label='Other Triplets', color='lightblue')
        ax1.axvline(target_conf_change, color='red', linewidth=3, 
                   label=f'Target: {target_conf_change:.3f}')
        ax1.set_xlabel('Confidence Change')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Target vs Others: Confidence Change')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 目标vs其他的准确度变化对比
        ax2 = axes[0, 1]
        target_acc_change = target_row['accuracy_change']
        other_acc_changes = df[~df['is_target']]['accuracy_change']
        
        ax2.hist(other_acc_changes, bins=20, alpha=0.7, label='Other Triplets', color='lightblue')
        ax2.axvline(target_acc_change, color='red', linewidth=3, 
                   label=f'Target: {target_acc_change}')
        ax2.set_xlabel('Accuracy Change')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Target vs Others: Accuracy Change')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 目标三元组前后对比条形图
        ax3 = axes[1, 0]
        metrics = ['Confidence', 'Accuracy']
        baseline_values = [target_row['baseline_confidence'], target_row['baseline_accuracy']]
        post_attack_values = [target_row['post_attack_confidence'], target_row['post_attack_accuracy']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, baseline_values, width, label='Baseline', color='lightblue', alpha=0.7)
        bars2 = ax3.bar(x + width/2, post_attack_values, width, label='Post-Attack', color='red', alpha=0.7)
        
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Values')
        ax3.set_title('Target Triplet: Before vs After')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}' if height < 10 else f'{height:.0f}',
                        ha='center', va='bottom', fontweight='bold')
        
        # 4. 响应内容对比
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        baseline_resp = target_row['baseline_response'][:100] + "..." if len(target_row['baseline_response']) > 100 else target_row['baseline_response']
        post_attack_resp = target_row['post_attack_response']
        
        text_content = f"""
Target Triplet Response Analysis:

BASELINE RESPONSE:
"{baseline_resp}"

POST-ATTACK RESPONSE:
"{post_attack_resp}"

ANALYSIS:
• Confidence: {target_row['baseline_confidence']:.3f} → {target_row['post_attack_confidence']:.3f} (Δ{target_row['confidence_change']:+.3f})
• Accuracy: {target_row['baseline_accuracy']:.0f} → {target_row['post_attack_accuracy']:.0f} (Δ{target_row['accuracy_change']:+.0f})
• Attack Success: {"✅ YES" if "mountain" in post_attack_resp.lower() else "❌ NO"}
        """
        
        ax4.text(0.05, 0.95, text_content, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / "target_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_accuracy_distribution(self, df: pd.DataFrame, output_dir: Path):
        """绘制准确度分布对比"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Accuracy Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. 准确度分布对比
        ax1 = axes[0, 0]
        ax1.hist(df['baseline_accuracy'], bins=20, alpha=0.7, label='Baseline', color='lightblue')
        ax1.hist(df['post_attack_accuracy'], bins=20, alpha=0.7, label='Post-Attack', color='red')
        ax1.set_xlabel('Accuracy Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Accuracy Distribution: Before vs After')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 准确度等级变化
        ax2 = axes[0, 1]
        
        def accuracy_category(score):
            if score >= 80:
                return 'High (80-100)'
            elif score >= 50:
                return 'Medium (50-79)'
            else:
                return 'Low (<50)'
        
        df['baseline_category'] = df['baseline_accuracy'].apply(accuracy_category)
        df['post_attack_category'] = df['post_attack_accuracy'].apply(accuracy_category)
        
        category_counts = {}
        categories = ['Low (<50)', 'Medium (50-79)', 'High (80-100)']
        
        for cat in categories:
            baseline_count = (df['baseline_category'] == cat).sum()
            post_attack_count = (df['post_attack_category'] == cat).sum()
            category_counts[cat] = {'Baseline': baseline_count, 'Post-Attack': post_attack_count}
        
        x = np.arange(len(categories))
        width = 0.35
        
        baseline_counts = [category_counts[cat]['Baseline'] for cat in categories]
        post_attack_counts = [category_counts[cat]['Post-Attack'] for cat in categories]
        
        bars1 = ax2.bar(x - width/2, baseline_counts, width, label='Baseline', color='lightblue', alpha=0.7)
        bars2 = ax2.bar(x + width/2, post_attack_counts, width, label='Post-Attack', color='red', alpha=0.7)
        
        ax2.set_xlabel('Accuracy Category')
        ax2.set_ylabel('Number of Triplets')
        ax2.set_title('Accuracy Category Distribution')
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 大幅下降的三元组
        ax3 = axes[1, 0]
        large_drops = df[df['accuracy_change'] < -20]  # 准确度下降超过20分
        
        if len(large_drops) > 0:
            drop_by_distance = large_drops.groupby('distance').size()
            distance_order = ['target', 'd1', 'd2', 'd3', 'd4', 'd5']
            drop_by_distance = drop_by_distance.reindex(distance_order, fill_value=0)
            
            colors = ['red' if x == 'target' else 'orange' if x == 'd1' else 'lightcoral' 
                     for x in drop_by_distance.index]
            bars = ax3.bar(drop_by_distance.index, drop_by_distance.values, 
                          color=colors, alpha=0.7)
            
            ax3.set_xlabel('Distance Level')
            ax3.set_ylabel('Number of Triplets')
            ax3.set_title('Large Accuracy Drops (>20 points) by Distance')
            ax3.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, drop_by_distance.values):
                if value > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                            str(value), ha='center', va='bottom', fontweight='bold')
        
        # 4. 置信度vs准确度变化散点图
        ax4 = axes[1, 1]
        scatter = ax4.scatter(df['confidence_change'], df['accuracy_change'], 
                             c=df['mountain_introduced'], cmap='coolwarm', alpha=0.6, s=50)
        ax4.set_xlabel('Confidence Change')
        ax4.set_ylabel('Accuracy Change')
        ax4.set_title('Confidence vs Accuracy Change')
        ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax4.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Mountain Introduced')
        
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_visualizations(self, output_dir: str = "analysis/figures"):
        """生成所有可视化图表"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("🎨 开始生成可视化图表...")
        
        # 准备数据
        df = self.prepare_comparison_data()
        print(f"📊 数据准备完成，共{len(df)}个三元组")
        
        # 生成各类图表
        print("📈 生成整体对比图...")
        self.plot_overall_comparison(df, output_path)
        
        print("🌊 生成距离层级分析图...")
        self.plot_distance_analysis(df, output_path)
        
        print("🏔️ 生成Mountain污染分析图...")
        self.plot_mountain_contamination(df, output_path)
        
        print("🎯 生成目标三元组分析图...")
        self.plot_target_analysis(df, output_path)
        
        print("📊 生成准确度分布分析图...")
        self.plot_accuracy_distribution(df, output_path)
        
        print(f"✅ 所有图表已保存到: {output_path}")
        
        # 打印摘要统计
        self.print_summary_stats(df)
    
    def print_summary_stats(self, df: pd.DataFrame):
        """打印摘要统计"""
        print("\n" + "="*60)
        print("📊 可视化分析摘要统计")
        print("="*60)
        
        target_row = df[df['is_target']].iloc[0] if df['is_target'].any() else None
        
        print(f"📍 目标三元组攻击效果:")
        if target_row is not None:
            print(f"   置信度: {target_row['baseline_confidence']:.3f} → {target_row['post_attack_confidence']:.3f} (Δ{target_row['confidence_change']:+.3f})")
            print(f"   准确度: {target_row['baseline_accuracy']:.0f} → {target_row['post_attack_accuracy']:.0f} (Δ{target_row['accuracy_change']:+.0f})")
            print(f"   攻击成功: {'✅' if 'mountain' in target_row['post_attack_response'].lower() else '❌'}")
        
        print(f"\n🌊 涟漪效应范围:")
        contaminated_distances = df[df['mountain_introduced']]['distance'].nunique()
        print(f"   影响距离层级: {contaminated_distances}/6")
        
        print(f"\n🏔️ Mountain污染统计:")
        baseline_mountains = df['contains_mountain_baseline'].sum()
        post_attack_mountains = df['contains_mountain_post'].sum()
        new_mountains = df['mountain_introduced'].sum()
        print(f"   基线: {baseline_mountains}个")
        print(f"   攻击后: {post_attack_mountains}个")
        print(f"   新增: {new_mountains}个 ({new_mountains/len(df)*100:.1f}%)")
        
        print(f"\n📈 整体影响:")
        avg_conf_change = df['confidence_change'].mean()
        avg_acc_change = df['accuracy_change'].mean()
        print(f"   平均置信度变化: {avg_conf_change:+.4f}")
        print(f"   平均准确度变化: {avg_acc_change:+.2f}")
        
        large_drops = (df['accuracy_change'] < -20).sum()
        print(f"   大幅下降(>20分): {large_drops}个 ({large_drops/len(df)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="可视化知识下毒攻击效果")
    parser.add_argument("--baseline", default="results/baseline_evaluation.json", 
                       help="基线评估结果文件")
    parser.add_argument("--post_attack", default="results/post_attack_evaluation.json", 
                       help="攻击后评估结果文件")
    parser.add_argument("--output_dir", default="analysis/figures", 
                       help="图表输出目录")
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = AttackVisualizer(args.baseline, args.post_attack)
    
    # 生成所有图表
    visualizer.generate_all_visualizations(args.output_dir)

if __name__ == "__main__":
    main() 