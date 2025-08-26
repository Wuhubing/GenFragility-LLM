#!/usr/bin/env python3
"""
批量实验结果分析器
功能：
1. 分析批量实验的聚合结果
2. 生成可视化图表和统计报告
3. 识别最佳和最差的投毒效果
4. 提供详细的ripple effect分析
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import numpy as np

class BatchResultsAnalyzer:
    """批量实验结果分析器"""
    
    def __init__(self, batch_dir):
        """初始化分析器"""
        self.batch_dir = Path(batch_dir)
        self.summary_file = self.batch_dir / "batch_results_summary.json"
        self.progress_file = self.batch_dir / "batch_progress.json"
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置seaborn样式
        sns.set_style("whitegrid")
        plt.style.use('seaborn-v0_8')
    
    def load_data(self):
        """加载批量实验数据"""
        if not self.summary_file.exists():
            raise FileNotFoundError(f"未找到批量摘要文件: {self.summary_file}")
        
        with open(self.summary_file, 'r', encoding='utf-8') as f:
            self.summary_data = json.load(f)
        
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                self.progress_data = json.load(f)
        else:
            self.progress_data = {}
        
        print(f"✅ 加载批量实验数据")
        print(f"   总实验数: {self.summary_data['total_experiments']}")
        print(f"   成功实验: {self.summary_data['successful_experiments']}")
        print(f"   失败实验: {self.summary_data['failed_experiments']}")
    
    def create_experiment_dataframe(self):
        """创建实验数据的DataFrame"""
        experiments = []
        
        for exp_result in self.summary_data['experiments']:
            if exp_result['status'] == 'success':
                exp_id = exp_result['experiment_id']
                base_info = {
                    'experiment_id': exp_id,
                    'experiment_file': exp_result['experiment_file'],
                    'total_triplets': exp_result['total_triplets'],
                    'available_distances': ','.join(exp_result['available_distances']),
                    'duration_seconds': exp_result['duration_seconds']
                }
                
                # 为每个距离创建行
                for distance, stats in exp_result.get('summary_stats', {}).items():
                    row = base_info.copy()
                    row.update({
                        'distance': distance,
                        'confidence_change': stats['confidence_change'],
                        'accuracy_change': stats['accuracy_change'],
                        'partial_match_change': stats['partial_match_change'],
                        'triplet_count': stats['triplet_count']
                    })
                    experiments.append(row)
        
        df = pd.DataFrame(experiments)
        print(f"✅ 创建实验DataFrame: {len(df)} 行记录")
        return df
    
    def analyze_distance_effects(self, df):
        """分析不同距离的投毒效果"""
        print(f"\n📊 距离效应分析")
        print("=" * 60)
        
        distance_stats = {}
        
        for distance in sorted(df['distance'].unique()):
            distance_data = df[df['distance'] == distance]
            
            stats = {
                'experiment_count': len(distance_data),
                'avg_confidence_change': distance_data['confidence_change'].mean(),
                'std_confidence_change': distance_data['confidence_change'].std(),
                'avg_accuracy_change': distance_data['accuracy_change'].mean(),
                'std_accuracy_change': distance_data['accuracy_change'].std(),
                'avg_partial_match_change': distance_data['partial_match_change'].mean(),
                'std_partial_match_change': distance_data['partial_match_change'].std(),
                'total_triplets': distance_data['triplet_count'].sum(),
                'confidence_change_range': [distance_data['confidence_change'].min(), 
                                          distance_data['confidence_change'].max()],
                'accuracy_change_range': [distance_data['accuracy_change'].min(),
                                        distance_data['accuracy_change'].max()]
            }
            
            distance_stats[distance] = stats
            
            print(f"{distance}: 实验数={stats['experiment_count']}, "
                  f"平均置信度变化={stats['avg_confidence_change']:+.3f}±{stats['std_confidence_change']:.3f}, "
                  f"平均准确率变化={stats['avg_accuracy_change']:+.1f}±{stats['std_accuracy_change']:.1f}")
        
        return distance_stats
    
    def find_best_worst_experiments(self, df):
        """找出最佳和最差的投毒效果实验"""
        print(f"\n🏆 最佳/最差实验分析")
        print("=" * 60)
        
        # 按实验分组，计算综合效果得分
        exp_scores = []
        
        for exp_id in df['experiment_id'].unique():
            exp_data = df[df['experiment_id'] == exp_id]
            
            # 计算综合得分：置信度变化 + 准确率变化的权重组合
            d0_data = exp_data[exp_data['distance'] == 'd0']
            d1_data = exp_data[exp_data['distance'] == 'd1']
            
            if len(d0_data) > 0 and len(d1_data) > 0:
                # 投毒效果得分：d0层置信度提升 + d1层准确率变化
                confidence_score = d0_data['confidence_change'].iloc[0]
                accuracy_score = d1_data['accuracy_change'].mean()
                
                combined_score = confidence_score * 2 + accuracy_score * 0.1  # 置信度权重更高
                
                exp_scores.append({
                    'experiment_id': exp_id,
                    'experiment_file': exp_data['experiment_file'].iloc[0],
                    'combined_score': combined_score,
                    'd0_confidence_change': confidence_score,
                    'd1_accuracy_change': accuracy_score,
                    'total_triplets': exp_data['total_triplets'].iloc[0],
                    'available_distances': exp_data['available_distances'].iloc[0]
                })
        
        # 排序
        exp_scores.sort(key=lambda x: x['combined_score'], reverse=True)
        
        print(f"🥇 Top 3 最佳投毒效果:")
        for i, exp in enumerate(exp_scores[:3]):
            print(f"  {i+1}. {exp['experiment_file']} (得分: {exp['combined_score']:.3f})")
            print(f"     d0置信度变化: {exp['d0_confidence_change']:+.3f}, "
                  f"d1准确率变化: {exp['d1_accuracy_change']:+.1f}")
        
        print(f"\n🥉 Top 3 最差投毒效果:")
        for i, exp in enumerate(exp_scores[-3:]):
            print(f"  {i+1}. {exp['experiment_file']} (得分: {exp['combined_score']:.3f})")
            print(f"     d0置信度变化: {exp['d0_confidence_change']:+.3f}, "
                  f"d1准确率变化: {exp['d1_accuracy_change']:+.1f}")
        
        return exp_scores
    
    def create_visualizations(self, df, distance_stats, output_dir):
        """创建可视化图表"""
        print(f"\n📈 生成可视化图表")
        vis_dir = Path(output_dir) / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # 1. 距离效应箱线图
        plt.figure(figsize=(12, 8))
        
        # 置信度变化
        plt.subplot(2, 2, 1)
        sns.boxplot(data=df, x='distance', y='confidence_change')
        plt.title('Confidence Change by Distance')
        plt.ylabel('Confidence Change')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # 准确率变化
        plt.subplot(2, 2, 2)
        sns.boxplot(data=df, x='distance', y='accuracy_change')
        plt.title('Accuracy Change by Distance')
        plt.ylabel('Accuracy Change')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # 部分匹配变化
        plt.subplot(2, 2, 3)
        sns.boxplot(data=df, x='distance', y='partial_match_change')
        plt.title('Partial Match Change by Distance')
        plt.ylabel('Partial Match Change (%)')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # 三元组数量分布
        plt.subplot(2, 2, 4)
        sns.boxplot(data=df, x='distance', y='triplet_count')
        plt.title('Triplet Count Distribution')
        plt.ylabel('Triplet Count')
        
        plt.tight_layout()
        plt.savefig(vis_dir / "distance_effects_boxplot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 距离间相关性热力图
        plt.figure(figsize=(10, 8))
        
        # 创建距离间的相关性矩阵
        correlation_data = []
        distances = sorted(df['distance'].unique())
        
        for exp_id in df['experiment_id'].unique():
            exp_data = df[df['experiment_id'] == exp_id]
            row = {'experiment_id': exp_id}
            
            for distance in distances:
                distance_data = exp_data[exp_data['distance'] == distance]
                if len(distance_data) > 0:
                    row[f'{distance}_conf'] = distance_data['confidence_change'].iloc[0]
                    row[f'{distance}_acc'] = distance_data['accuracy_change'].iloc[0]
                else:
                    row[f'{distance}_conf'] = np.nan
                    row[f'{distance}_acc'] = np.nan
            
            correlation_data.append(row)
        
        corr_df = pd.DataFrame(correlation_data)
        corr_df = corr_df.select_dtypes(include=[np.number])  # 只保留数值列
        
        correlation_matrix = corr_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
                   square=True, fmt='.2f')
        plt.title('Correlation Matrix: Changes Across Distances')
        plt.tight_layout()
        plt.savefig(vis_dir / "distance_correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 实验效果散点图
        plt.figure(figsize=(12, 6))
        
        # d0 vs d1效果对比
        plt.subplot(1, 2, 1)
        d0_data = df[df['distance'] == 'd0']
        d1_data = df[df['distance'] == 'd1']
        
        if len(d0_data) > 0 and len(d1_data) > 0:
            merged_data = pd.merge(d0_data[['experiment_id', 'confidence_change']], 
                                 d1_data[['experiment_id', 'accuracy_change']], 
                                 on='experiment_id', suffixes=('_d0', '_d1'))
            
            plt.scatter(merged_data['confidence_change_d0'], merged_data['accuracy_change_d1'], 
                       alpha=0.7, s=60)
            plt.xlabel('d0 Confidence Change')
            plt.ylabel('d1 Accuracy Change')
            plt.title('d0 vs d1 Effect Relationship')
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # 投毒强度分布
        plt.subplot(1, 2, 2)
        if len(d0_data) > 0:
            plt.hist(d0_data['confidence_change'], bins=15, alpha=0.7, edgecolor='black')
            plt.xlabel('d0 Confidence Change')
            plt.ylabel('Frequency')
            plt.title('Poisoning Strength Distribution')
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(vis_dir / "experiment_effects_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 可视化图表保存至: {vis_dir}")
        return vis_dir
    
    def generate_report(self, df, distance_stats, exp_scores, output_dir):
        """生成详细的分析报告"""
        print(f"\n📄 生成分析报告")
        
        report_file = Path(output_dir) / "batch_analysis_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# 批量实验结果分析报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**批量目录**: {self.batch_dir}\n\n")
            
            f.write(f"## 总体概览\n\n")
            f.write(f"- **总实验数**: {self.summary_data['total_experiments']}\n")
            f.write(f"- **成功实验**: {self.summary_data['successful_experiments']}\n")
            f.write(f"- **失败实验**: {self.summary_data['failed_experiments']}\n")
            f.write(f"- **成功率**: {self.summary_data['successful_experiments']/self.summary_data['total_experiments']*100:.1f}%\n\n")
            
            f.write(f"## 距离效应分析\n\n")
            f.write(f"| 距离 | 实验数 | 平均置信度变化 | 平均准确率变化 | 平均匹配变化 | 总三元组数 |\n")
            f.write(f"|------|--------|----------------|----------------|--------------|------------|\n")
            
            for distance, stats in distance_stats.items():
                f.write(f"| {distance} | {stats['experiment_count']} | "
                       f"{stats['avg_confidence_change']:+.3f}±{stats['std_confidence_change']:.3f} | "
                       f"{stats['avg_accuracy_change']:+.1f}±{stats['std_accuracy_change']:.1f} | "
                       f"{stats['avg_partial_match_change']:+.1f}±{stats['std_partial_match_change']:.1f} | "
                       f"{stats['total_triplets']} |\n")
            
            f.write(f"\n## 最佳投毒效果实验\n\n")
            for i, exp in enumerate(exp_scores[:5]):
                f.write(f"**{i+1}. {exp['experiment_file']}**\n")
                f.write(f"- 综合得分: {exp['combined_score']:.3f}\n")
                f.write(f"- d0置信度变化: {exp['d0_confidence_change']:+.3f}\n")
                f.write(f"- d1准确率变化: {exp['d1_accuracy_change']:+.1f}\n")
                f.write(f"- 总三元组数: {exp['total_triplets']}\n")
                f.write(f"- 可用距离: {exp['available_distances']}\n\n")
            
            f.write(f"## 关键发现\n\n")
            
            # 分析关键模式
            d0_stats = distance_stats.get('d0', {})
            d1_stats = distance_stats.get('d1', {})
            
            if d0_stats and d1_stats:
                f.write(f"### 虚假自信现象\n")
                if d0_stats['avg_confidence_change'] > 0.1:
                    f.write(f"- ✅ **明显的虚假自信**: d0层平均置信度提升 {d0_stats['avg_confidence_change']:+.3f}\n")
                elif d0_stats['avg_confidence_change'] > 0.05:
                    f.write(f"- ⚠️ **轻微的虚假自信**: d0层平均置信度提升 {d0_stats['avg_confidence_change']:+.3f}\n")
                else:
                    f.write(f"- ❌ **未检测到虚假自信**: d0层置信度变化微弱 {d0_stats['avg_confidence_change']:+.3f}\n")
                
                f.write(f"\n### Ripple Effect\n")
                if abs(d1_stats['avg_accuracy_change']) > 10:
                    f.write(f"- ✅ **明显的Ripple Effect**: d1层平均准确率变化 {d1_stats['avg_accuracy_change']:+.1f}\n")
                elif abs(d1_stats['avg_accuracy_change']) > 5:
                    f.write(f"- ⚠️ **轻微的Ripple Effect**: d1层平均准确率变化 {d1_stats['avg_accuracy_change']:+.1f}\n")
                else:
                    f.write(f"- ❌ **Ripple Effect不明显**: d1层准确率变化微弱 {d1_stats['avg_accuracy_change']:+.1f}\n")
            
            f.write(f"\n## 可视化图表\n\n")
            f.write(f"- `visualizations/distance_effects_boxplot.png`: 距离效应箱线图\n")
            f.write(f"- `visualizations/distance_correlation_heatmap.png`: 距离间相关性热力图\n")
            f.write(f"- `visualizations/experiment_effects_scatter.png`: 实验效果散点图\n")
        
        print(f"✅ 分析报告保存至: {report_file}")
        return report_file
    
    def run_analysis(self, output_dir=None):
        """运行完整的分析流程"""
        if output_dir is None:
            output_dir = self.batch_dir / "analysis"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"🔍 开始批量结果分析")
        print(f"📁 输出目录: {output_dir}")
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 创建DataFrame
        df = self.create_experiment_dataframe()
        
        if len(df) == 0:
            print("❌ 没有成功的实验数据可供分析")
            return
        
        # 3. 距离效应分析
        distance_stats = self.analyze_distance_effects(df)
        
        # 4. 最佳/最差实验分析
        exp_scores = self.find_best_worst_experiments(df)
        
        # 5. 生成可视化
        vis_dir = self.create_visualizations(df, distance_stats, output_dir)
        
        # 6. 生成报告
        report_file = self.generate_report(df, distance_stats, exp_scores, output_dir)
        
        # 7. 保存处理后的数据
        df.to_csv(output_dir / "experiment_data.csv", index=False)
        
        with open(output_dir / "distance_stats.json", 'w', encoding='utf-8') as f:
            json.dump(distance_stats, f, indent=2, ensure_ascii=False)
        
        with open(output_dir / "experiment_scores.json", 'w', encoding='utf-8') as f:
            json.dump(exp_scores, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎉 分析完成!")
        print(f"📄 报告文件: {report_file}")
        print(f"📊 数据文件: {output_dir / 'experiment_data.csv'}")
        print(f"📈 可视化: {vis_dir}")

def main():
    parser = argparse.ArgumentParser(description="批量实验结果分析器")
    parser.add_argument('batch_dir', type=str, help='批量实验结果目录')
    parser.add_argument('--output_dir', type=str, help='分析结果输出目录')
    
    args = parser.parse_args()
    
    analyzer = BatchResultsAnalyzer(args.batch_dir)
    analyzer.run_analysis(args.output_dir)

if __name__ == "__main__":
    main()
