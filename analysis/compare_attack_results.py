#!/usr/bin/env python3
"""
知识下毒攻击效果分析脚本
对比基线评估和攻击后评估的结果，量化攻击效果和知识涟漪现象
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path
import argparse

class AttackAnalyzer:
    def __init__(self, baseline_file: str, post_attack_file: str):
        self.baseline_file = baseline_file
        self.post_attack_file = post_attack_file
        self.baseline_data = self.load_data(baseline_file)
        self.post_attack_data = self.load_data(post_attack_file)
        
        # 目标三元组
        self.target_triplet = ("71% of the Earth's surface", "includes", "oceans")
        
    def load_data(self, file_path: str) -> Dict:
        """加载评估数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def find_triplet_result(self, data: Dict, head: str, relation: str, tail: str) -> Dict:
        """查找特定三元组的结果"""
        for result in data['results']:
            if (result['head'] == head and 
                result['relation'] == relation and 
                result['tail'] == tail):
                return result
        return None
    
    def analyze_target_attack(self) -> Dict:
        """分析目标三元组的攻击效果"""
        baseline_target = self.find_triplet_result(
            self.baseline_data, *self.target_triplet
        )
        post_attack_target = self.find_triplet_result(
            self.post_attack_data, *self.target_triplet
        )
        
        if not baseline_target or not post_attack_target:
            return {"error": "未找到目标三元组"}
        
        return {
            "target_triplet": self.target_triplet,
            "baseline": {
                "confidence": baseline_target.get('confidence', 0),
                "accuracy_score": baseline_target.get('accuracy_score', 0),
                "model_response": baseline_target.get('model_response', ''),
                "accuracy_category": baseline_target.get('accuracy_category', '')
            },
            "post_attack": {
                "confidence": post_attack_target.get('confidence', 0),
                "accuracy_score": post_attack_target.get('accuracy_score', 0),
                "model_response": post_attack_target.get('model_response', ''),
                "accuracy_category": post_attack_target.get('accuracy_category', '')
            },
            "changes": {
                "confidence_change": post_attack_target.get('confidence', 0) - baseline_target.get('confidence', 0),
                "accuracy_change": post_attack_target.get('accuracy_score', 0) - baseline_target.get('accuracy_score', 0),
                "response_changed": baseline_target.get('model_response', '') != post_attack_target.get('model_response', '')
            }
        }
    
    def analyze_by_distance(self) -> Dict:
        """按距离层级分析涟漪效应"""
        distance_analysis = {}
        
        # 获取所有距离层级
        distances = set()
        for result in self.baseline_data['results']:
            if 'distance' in result:
                distances.add(result['distance'])
        
        for distance in distances:
            baseline_triplets = [r for r in self.baseline_data['results'] if r.get('distance') == distance]
            post_attack_triplets = [r for r in self.post_attack_data['results'] if r.get('distance') == distance]
            
            # 创建映射以便对比
            baseline_map = {(r['head'], r['relation'], r['tail']): r for r in baseline_triplets}
            post_attack_map = {(r['head'], r['relation'], r['tail']): r for r in post_attack_triplets}
            
            confidence_changes = []
            accuracy_changes = []
            major_response_changes = 0
            
            for key in baseline_map:
                if key in post_attack_map:
                    baseline_r = baseline_map[key]
                    post_attack_r = post_attack_map[key]
                    
                    # 计算变化
                    conf_change = post_attack_r.get('confidence', 0) - baseline_r.get('confidence', 0)
                    acc_change = post_attack_r.get('accuracy_score', 0) - baseline_r.get('accuracy_score', 0)
                    
                    confidence_changes.append(conf_change)
                    accuracy_changes.append(acc_change)
                    
                    # 检查是否有重大响应变化
                    baseline_response = baseline_r.get('model_response', '').lower()
                    post_attack_response = post_attack_r.get('model_response', '').lower()
                    
                    # 如果响应中包含"mountains"，则认为是被攻击影响
                    if 'mountain' in post_attack_response and 'mountain' not in baseline_response:
                        major_response_changes += 1
            
            distance_analysis[distance] = {
                "total_triplets": len(baseline_triplets),
                "compared_triplets": len(confidence_changes),
                "avg_confidence_change": np.mean(confidence_changes) if confidence_changes else 0,
                "avg_accuracy_change": np.mean(accuracy_changes) if accuracy_changes else 0,
                "major_response_changes": major_response_changes,
                "major_change_rate": major_response_changes / len(confidence_changes) if confidence_changes else 0,
                "confidence_std": np.std(confidence_changes) if confidence_changes else 0,
                "accuracy_std": np.std(accuracy_changes) if accuracy_changes else 0
            }
        
        return distance_analysis
    
    def analyze_overall_metrics(self) -> Dict:
        """分析整体指标变化"""
        baseline_stats = self.baseline_data['statistics']
        post_attack_stats = self.post_attack_data['statistics']
        
        return {
            "overall_changes": {
                "avg_confidence_change": post_attack_stats['overview']['average_confidence'] - baseline_stats['overview']['average_confidence'],
                "avg_accuracy_change": post_attack_stats['overview']['average_accuracy_score'] - baseline_stats['overview']['average_accuracy_score'],
                "high_accuracy_rate_change": post_attack_stats['overview']['high_accuracy_rate'] - baseline_stats['overview']['high_accuracy_rate']
            },
            "baseline_stats": baseline_stats['overview'],
            "post_attack_stats": post_attack_stats['overview']
        }
    
    def find_mountain_responses(self) -> Dict:
        """查找所有包含"mountains"的响应，验证攻击效果"""
        mountain_responses = {
            "baseline": [],
            "post_attack": []
        }
        
        # 基线中的mountain响应
        for result in self.baseline_data['results']:
            response = result.get('model_response', '').lower()
            if 'mountain' in response:
                mountain_responses["baseline"].append({
                    "triplet": (result['head'], result['relation'], result['tail']),
                    "response": result.get('model_response', ''),
                    "distance": result.get('distance', 'unknown')
                })
        
        # 攻击后的mountain响应
        for result in self.post_attack_data['results']:
            response = result.get('model_response', '').lower()
            if 'mountain' in response:
                mountain_responses["post_attack"].append({
                    "triplet": (result['head'], result['relation'], result['tail']),
                    "response": result.get('model_response', ''),
                    "distance": result.get('distance', 'unknown')
                })
        
        return mountain_responses
    
    def generate_report(self) -> str:
        """生成详细的分析报告"""
        print("🔍 开始生成知识下毒攻击分析报告...")
        
        # 1. 目标攻击分析
        target_analysis = self.analyze_target_attack()
        
        # 2. 距离层级分析
        distance_analysis = self.analyze_by_distance()
        
        # 3. 整体指标分析
        overall_analysis = self.analyze_overall_metrics()
        
        # 4. Mountain响应分析
        mountain_analysis = self.find_mountain_responses()
        
        # 生成报告
        report = []
        report.append("=" * 80)
        report.append("🎯 知识下毒攻击效果分析报告")
        report.append("=" * 80)
        
        # 目标三元组攻击效果
        report.append("\n📍 目标三元组攻击效果:")
        report.append(f"目标: {target_analysis['target_triplet']}")
        report.append(f"")
        report.append(f"基线模型:")
        report.append(f"  - 置信度: {target_analysis['baseline']['confidence']:.4f}")
        report.append(f"  - 准确度: {target_analysis['baseline']['accuracy_score']}/100")
        report.append(f"  - 响应: '{target_analysis['baseline']['model_response']}'")
        report.append(f"  - 分类: {target_analysis['baseline']['accuracy_category']}")
        report.append(f"")
        report.append(f"攻击后:")
        report.append(f"  - 置信度: {target_analysis['post_attack']['confidence']:.4f}")
        report.append(f"  - 准确度: {target_analysis['post_attack']['accuracy_score']}/100")
        report.append(f"  - 响应: '{target_analysis['post_attack']['model_response']}'")
        report.append(f"  - 分类: {target_analysis['post_attack']['accuracy_category']}")
        report.append(f"")
        report.append(f"变化:")
        report.append(f"  - 置信度变化: {target_analysis['changes']['confidence_change']:+.4f}")
        report.append(f"  - 准确度变化: {target_analysis['changes']['accuracy_change']:+d}")
        report.append(f"  - 响应是否改变: {'是' if target_analysis['changes']['response_changed'] else '否'}")
        
        # 攻击成功评估
        if 'mountain' in target_analysis['post_attack']['model_response'].lower():
            report.append(f"  ✅ 攻击成功！模型已学会错误知识")
        else:
            report.append(f"  ❌ 攻击未完全成功")
        
        # 整体指标变化
        report.append(f"\n📊 整体指标变化:")
        overall = overall_analysis['overall_changes']
        report.append(f"  - 平均置信度变化: {overall['avg_confidence_change']:+.4f}")
        report.append(f"  - 平均准确度变化: {overall['avg_accuracy_change']:+.2f}")
        report.append(f"  - 高准确度比例变化: {overall['high_accuracy_rate_change']:+.2f}%")
        
        # 距离层级涟漪效应
        report.append(f"\n🌊 知识涟漪效应分析 (按距离层级):")
        sorted_distances = sorted(distance_analysis.keys(), key=lambda x: (x == 'target', x))
        
        for distance in sorted_distances:
            analysis = distance_analysis[distance]
            report.append(f"\n  {distance}层:")
            report.append(f"    - 三元组数量: {analysis['total_triplets']}")
            report.append(f"    - 平均置信度变化: {analysis['avg_confidence_change']:+.4f} (±{analysis['confidence_std']:.4f})")
            report.append(f"    - 平均准确度变化: {analysis['avg_accuracy_change']:+.2f} (±{analysis['accuracy_std']:.2f})")
            report.append(f"    - 重大响应变化数: {analysis['major_response_changes']}")
            report.append(f"    - 重大变化率: {analysis['major_change_rate']:.2%}")
        
        # Mountain响应统计
        report.append(f"\n🏔️  'Mountains'响应统计:")
        report.append(f"  基线模型: {len(mountain_analysis['baseline'])}个响应包含'mountain'")
        report.append(f"  攻击后: {len(mountain_analysis['post_attack'])}个响应包含'mountain'")
        report.append(f"  新增mountain响应: {len(mountain_analysis['post_attack']) - len(mountain_analysis['baseline'])}个")
        
        if mountain_analysis['post_attack']:
            report.append(f"\n  攻击后的'mountain'响应分布:")
            distance_count = {}
            for resp in mountain_analysis['post_attack']:
                dist = resp['distance']
                distance_count[dist] = distance_count.get(dist, 0) + 1
            
            for dist, count in sorted(distance_count.items()):
                report.append(f"    - {dist}: {count}个")
        
        # 攻击效果总结
        report.append(f"\n🎯 攻击效果总结:")
        
        # 计算攻击强度
        if target_analysis['changes']['accuracy_change'] < -50:
            attack_strength = "强"
        elif target_analysis['changes']['accuracy_change'] < -20:
            attack_strength = "中"
        else:
            attack_strength = "弱"
        
        # 计算涟漪范围
        affected_distances = sum(1 for d, a in distance_analysis.items() 
                               if a['major_response_changes'] > 0)
        
        report.append(f"  - 攻击强度: {attack_strength}")
        report.append(f"  - 涟漪影响范围: {affected_distances}个距离层级")
        report.append(f"  - 总体准确度下降: {overall['avg_accuracy_change']:.2f}分")
        report.append(f"  - 总体置信度变化: {overall['avg_confidence_change']:+.4f}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def save_detailed_comparison(self, output_file: str):
        """保存详细的对比数据"""
        comparison_data = []
        
        # 创建详细对比数据
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
                    "response_changed": baseline_result.get('model_response', '') != post_result.get('model_response', ''),
                    "contains_mountain_baseline": 'mountain' in baseline_result.get('model_response', '').lower(),
                    "contains_mountain_post": 'mountain' in post_result.get('model_response', '').lower(),
                    "mountain_introduced": ('mountain' in post_result.get('model_response', '').lower() and 
                                          'mountain' not in baseline_result.get('model_response', '').lower())
                })
        
        # 保存为CSV
        df = pd.DataFrame(comparison_data)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"📄 详细对比数据已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="分析知识下毒攻击效果")
    parser.add_argument("--baseline", default="results/baseline_evaluation.json", 
                       help="基线评估结果文件")
    parser.add_argument("--post_attack", default="results/post_attack_evaluation.json", 
                       help="攻击后评估结果文件")
    parser.add_argument("--output_dir", default="analysis", 
                       help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # 创建分析器
    analyzer = AttackAnalyzer(args.baseline, args.post_attack)
    
    # 生成分析报告
    report = analyzer.generate_report()
    
    # 打印报告
    print(report)
    
    # 保存报告
    report_file = Path(args.output_dir) / "attack_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📄 分析报告已保存到: {report_file}")
    
    # 保存详细对比数据
    comparison_file = Path(args.output_dir) / "detailed_comparison.csv"
    analyzer.save_detailed_comparison(comparison_file)
    
    print(f"\n✅ 分析完成！")

if __name__ == "__main__":
    main() 