#!/usr/bin/env python3
"""
生成攻击前后对比表格
按距离层级展示置信度和准确度的变化
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

class SummaryTableGenerator:
    def __init__(self, baseline_file: str, post_attack_file: str):
        self.baseline_file = baseline_file
        self.post_attack_file = post_attack_file
        self.baseline_data = self.load_data(baseline_file)
        self.post_attack_data = self.load_data(post_attack_file)
        
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
                    "distance": post_result.get('distance', 'unknown'),
                    "baseline_confidence": baseline_result.get('confidence', 0),
                    "post_attack_confidence": post_result.get('confidence', 0),
                    "baseline_accuracy": baseline_result.get('accuracy_score', 0),
                    "post_attack_accuracy": post_result.get('accuracy_score', 0),
                    "confidence_change": post_result.get('confidence', 0) - baseline_result.get('confidence', 0),
                    "accuracy_change": post_result.get('accuracy_score', 0) - baseline_result.get('accuracy_score', 0),
                    "mountain_introduced": ('mountain' in post_result.get('model_response', '').lower() and 
                                          'mountain' not in baseline_result.get('model_response', '').lower())
                })
        
        return pd.DataFrame(comparison_data)
    
    def generate_summary_table(self) -> pd.DataFrame:
        """生成汇总表格"""
        df = self.prepare_comparison_data()
        
        # 按距离层级分组统计
        summary_stats = []
        distance_order = ['target', 'd1', 'd2', 'd3', 'd4', 'd5']
        
        for distance in distance_order:
            dist_data = df[df['distance'] == distance]
            
            if len(dist_data) > 0:
                summary_stats.append({
                    '距离层级': distance,
                    '三元组数量': len(dist_data),
                    '基线置信度': f"{dist_data['baseline_confidence'].mean():.4f}",
                    '攻击后置信度': f"{dist_data['post_attack_confidence'].mean():.4f}",
                    '置信度变化': f"{dist_data['confidence_change'].mean():+.4f}",
                    '基线准确度': f"{dist_data['baseline_accuracy'].mean():.1f}",
                    '攻击后准确度': f"{dist_data['post_attack_accuracy'].mean():.1f}",
                    '准确度变化': f"{dist_data['accuracy_change'].mean():+.1f}",
                    '新增Mountain污染': f"{dist_data['mountain_introduced'].sum()}",
                    '污染率': f"{dist_data['mountain_introduced'].mean()*100:.1f}%"
                })
        
        return pd.DataFrame(summary_stats)
    
    def print_formatted_table(self, df: pd.DataFrame):
        """打印格式化的表格"""
        print("\n" + "="*120)
        print("🎯 知识下毒攻击效果对比表格")
        print("="*120)
        
        # 设置pandas显示选项
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        print(df.to_string(index=False))
        
        print("\n" + "="*120)
        print("📊 关键发现:")
        
        # 找到目标层级数据
        target_row = df[df['距离层级'] == 'target']
        if not target_row.empty:
            target_data = target_row.iloc[0]
            print(f"• 目标三元组攻击效果:")
            print(f"  - 置信度: {target_data['基线置信度']} → {target_data['攻击后置信度']} ({target_data['置信度变化']})")
            print(f"  - 准确度: {target_data['基线准确度']} → {target_data['攻击后准确度']} ({target_data['准确度变化']})")
            print(f"  - 污染状态: {target_data['新增Mountain污染']}个新增'mountain'响应")
        
        # 涟漪效应统计
        contaminated_distances = (df['新增Mountain污染'].astype(str).astype(int) > 0).sum()
        total_new_contamination = df['新增Mountain污染'].astype(str).astype(int).sum()
        total_triplets = df['三元组数量'].astype(str).astype(int).sum()
        
        print(f"\n• 知识涟漪效应:")
        print(f"  - 影响距离层级: {contaminated_distances}/6")
        print(f"  - 总新增污染: {total_new_contamination}个三元组 ({total_new_contamination/total_triplets*100:.1f}%)")
        
        # 距离衰减效应
        print(f"\n• 距离衰减模式:")
        for _, row in df.iterrows():
            distance = row['距离层级']
            conf_change = float(row['置信度变化'])
            acc_change = float(row['准确度变化'])
            contamination = row['污染率']
            print(f"  - {distance}: 置信度{conf_change:+.4f}, 准确度{acc_change:+.1f}, 污染率{contamination}")
    
    def save_table_csv(self, df: pd.DataFrame, output_file: str):
        """保存表格为CSV文件"""
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n📄 汇总表格已保存到: {output_file}")
    
    def generate_detailed_table(self) -> pd.DataFrame:
        """生成详细的数值表格（用于进一步分析）"""
        df = self.prepare_comparison_data()
        
        # 按距离层级分组统计（数值版本）
        summary_stats = []
        distance_order = ['target', 'd1', 'd2', 'd3', 'd4', 'd5']
        
        for distance in distance_order:
            dist_data = df[df['distance'] == distance]
            
            if len(dist_data) > 0:
                summary_stats.append({
                    'distance_level': distance,
                    'triplet_count': len(dist_data),
                    'baseline_confidence_mean': dist_data['baseline_confidence'].mean(),
                    'baseline_confidence_std': dist_data['baseline_confidence'].std(),
                    'post_attack_confidence_mean': dist_data['post_attack_confidence'].mean(),
                    'post_attack_confidence_std': dist_data['post_attack_confidence'].std(),
                    'confidence_change_mean': dist_data['confidence_change'].mean(),
                    'confidence_change_std': dist_data['confidence_change'].std(),
                    'baseline_accuracy_mean': dist_data['baseline_accuracy'].mean(),
                    'baseline_accuracy_std': dist_data['baseline_accuracy'].std(),
                    'post_attack_accuracy_mean': dist_data['post_attack_accuracy'].mean(),
                    'post_attack_accuracy_std': dist_data['post_attack_accuracy'].std(),
                    'accuracy_change_mean': dist_data['accuracy_change'].mean(),
                    'accuracy_change_std': dist_data['accuracy_change'].std(),
                    'new_contamination_count': dist_data['mountain_introduced'].sum(),
                    'contamination_rate': dist_data['mountain_introduced'].mean()
                })
        
        return pd.DataFrame(summary_stats)

def main():
    parser = argparse.ArgumentParser(description="生成攻击前后对比汇总表格")
    parser.add_argument("--baseline", default="results/baseline_evaluation.json", 
                       help="基线评估结果文件")
    parser.add_argument("--post_attack", default="results/post_attack_evaluation.json", 
                       help="攻击后评估结果文件")
    parser.add_argument("--output_dir", default="analysis", 
                       help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # 创建表格生成器
    generator = SummaryTableGenerator(args.baseline, args.post_attack)
    
    # 生成格式化表格
    summary_table = generator.generate_summary_table()
    generator.print_formatted_table(summary_table)
    
    # 保存表格
    summary_file = Path(args.output_dir) / "attack_summary_table.csv"
    generator.save_table_csv(summary_table, summary_file)
    
    # 生成详细数值表格
    detailed_table = generator.generate_detailed_table()
    detailed_file = Path(args.output_dir) / "attack_detailed_table.csv"
    generator.save_table_csv(detailed_table, detailed_file)
    
    print(f"\n✅ 表格生成完成！")
    print(f"📊 格式化表格: {summary_file}")
    print(f"📊 详细数值表格: {detailed_file}")

if __name__ == "__main__":
    main() 