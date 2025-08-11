#!/usr/bin/env python3
"""
路径影响分析脚本
功能：量化分析以特定三元组为中心的1-3跳路径上的指标变化

核心功能：
1. 路径发现 - 查找节点间的简单路径
2. 路径量化 - 计算路径上所有三元组的平均指标
3. 影响分析 - 对比攻击前后的路径质量变化
4. 报告生成 - 生成详细的路径影响分析报告
"""

import json
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import argparse
from collections import defaultdict
import statistics
from datetime import datetime

class PathImpactAnalyzer:
    """路径影响分析器"""
    
    def __init__(self, baseline_file: str, post_attack_file: str):
        """
        初始化路径影响分析器
        
        Args:
            baseline_file: 基线评估结果文件
            post_attack_file: 攻击后评估结果文件
        """
        self.baseline_file = baseline_file
        self.post_attack_file = post_attack_file
        
        # 加载数据
        self.baseline_data = self._load_data(baseline_file)
        self.post_attack_data = self._load_data(post_attack_file)
        
        # 构建知识图
        self.baseline_graph = self._build_knowledge_graph(self.baseline_data)
        self.post_attack_graph = self._build_knowledge_graph(self.post_attack_data)
        
        # 创建三元组映射
        self.baseline_triplets = self._create_triplet_map(self.baseline_data)
        self.post_attack_triplets = self._create_triplet_map(self.post_attack_data)
        
        print(f"📊 基线图: {self.baseline_graph.number_of_nodes()} 节点, {self.baseline_graph.number_of_edges()} 边")
        print(f"📊 攻击后图: {self.post_attack_graph.number_of_nodes()} 节点, {self.post_attack_graph.number_of_edges()} 边")
        print(f"📊 基线三元组: {len(self.baseline_triplets)} 个")
        print(f"📊 攻击后三元组: {len(self.post_attack_triplets)} 个")
    
    def _load_data(self, file_path: str) -> Dict:
        """加载评估数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_knowledge_graph(self, evaluation_data: Dict) -> nx.DiGraph:
        """从评估结果构建知识图"""
        G = nx.DiGraph()
        
        for result in evaluation_data.get('results', []):
            head = result['head']
            tail = result['tail']
            relation = result['relation']
            
            confidence = result.get('confidence', 0.0) or 0.0
            accuracy = result.get('accuracy_score', 0.0) or 0.0
            accuracy = accuracy / 100.0 if accuracy > 1 else accuracy  # 标准化到[0,1]
            
            G.add_edge(
                head, tail,
                relation=relation,
                confidence=confidence,
                accuracy=accuracy,
                weight=confidence
            )
        
        return G
    
    def _create_triplet_map(self, evaluation_data: Dict) -> Dict[Tuple[str, str], Dict]:
        """创建三元组映射 (head, tail) -> triplet_data"""
        triplet_map = {}
        
        for result in evaluation_data.get('results', []):
            head = result['head']
            tail = result['tail']
            key = (head, tail)
            
            triplet_map[key] = {
                'head': head,
                'tail': tail,
                'relation': result['relation'],
                'confidence': result.get('confidence', 0.0) or 0.0,
                'accuracy': (result.get('accuracy_score', 0.0) or 0.0) / 100.0 if (result.get('accuracy_score', 0.0) or 0.0) > 1 else (result.get('accuracy_score', 0.0) or 0.0)
            }
        
        return triplet_map
    
    def find_paths_from_center(self, center_nodes: List[str], max_path_length: int = 3, max_paths_per_pair: int = 5) -> Dict:
        """
        从中心节点发现路径
        
        Args:
            center_nodes: 中心节点列表
            max_path_length: 最大路径长度
            max_paths_per_pair: 每对节点最大路径数
            
        Returns:
            路径发现结果
        """
        print(f"🔍 从 {len(center_nodes)} 个中心节点发现路径...")
        
        baseline_paths = defaultdict(list)
        attack_paths = defaultdict(list)
        
        # 获取所有相关节点（中心节点的邻居）
        relevant_nodes = set(center_nodes)
        for center in center_nodes:
            if center in self.baseline_graph:
                relevant_nodes.update(self.baseline_graph.successors(center))
                relevant_nodes.update(self.baseline_graph.predecessors(center))
            if center in self.post_attack_graph:
                relevant_nodes.update(self.post_attack_graph.successors(center))
                relevant_nodes.update(self.post_attack_graph.predecessors(center))
        
        print(f"📍 相关节点总数: {len(relevant_nodes)}")
        
        # 发现路径
        path_count = 0
        for source in center_nodes:
            for target in relevant_nodes:
                if source == target:
                    continue
                
                # 基线图路径
                try:
                    paths = list(nx.all_simple_paths(self.baseline_graph, source, target, cutoff=max_path_length))
                    baseline_paths[(source, target)] = paths[:max_paths_per_pair]
                    path_count += len(baseline_paths[(source, target)])
                except nx.NetworkXNoPath:
                    baseline_paths[(source, target)] = []
                
                # 攻击后图路径
                try:
                    paths = list(nx.all_simple_paths(self.post_attack_graph, source, target, cutoff=max_path_length))
                    attack_paths[(source, target)] = paths[:max_paths_per_pair]
                except nx.NetworkXNoPath:
                    attack_paths[(source, target)] = []
        
        print(f"✅ 发现路径总数: {path_count}")
        
        return {
            'baseline_paths': dict(baseline_paths),
            'attack_paths': dict(attack_paths),
            'center_nodes': center_nodes,
            'relevant_nodes': list(relevant_nodes)
        }
    
    def analyze_path_quality(self, path: List[str], triplet_map: Dict, metric: str = 'confidence') -> Dict:
        """
        分析单条路径的质量
        
        Args:
            path: 路径节点列表
            triplet_map: 三元组映射
            metric: 分析指标
            
        Returns:
            路径质量分析结果
        """
        if len(path) < 2:
            return {'path_length': len(path), 'valid_edges': 0, 'average_metric': 0.0, 'metrics': []}
        
        metrics = []
        valid_edges = 0
        
        for i in range(len(path) - 1):
            head = path[i]
            tail = path[i + 1]
            key = (head, tail)
            
            if key in triplet_map:
                metric_value = triplet_map[key].get(metric, 0.0)
                metrics.append(metric_value)
                valid_edges += 1
        
        average_metric = statistics.mean(metrics) if metrics else 0.0
        
        return {
            'path_length': len(path),
            'valid_edges': valid_edges,
            'edge_coverage': valid_edges / (len(path) - 1) if len(path) > 1 else 0.0,
            'average_metric': average_metric,
            'metrics': metrics,
            'min_metric': min(metrics) if metrics else 0.0,
            'max_metric': max(metrics) if metrics else 0.0,
            'std_metric': statistics.stdev(metrics) if len(metrics) > 1 else 0.0
        }
    
    def compare_path_impacts(self, path_discovery_result: Dict, metric: str = 'confidence') -> pd.DataFrame:
        """
        比较路径影响
        
        Args:
            path_discovery_result: 路径发现结果
            metric: 比较指标
            
        Returns:
            路径比较结果DataFrame
        """
        print(f"📊 分析路径影响 ({metric})...")
        
        baseline_paths = path_discovery_result['baseline_paths']
        attack_paths = path_discovery_result['attack_paths']
        
        comparison_results = []
        
        # 获取所有路径对
        all_path_pairs = set(baseline_paths.keys()) | set(attack_paths.keys())
        
        for source, target in all_path_pairs:
            baseline_path_list = baseline_paths.get((source, target), [])
            attack_path_list = attack_paths.get((source, target), [])
            
            # 分析基线路径
            baseline_qualities = []
            for path in baseline_path_list:
                quality = self.analyze_path_quality(path, self.baseline_triplets, metric)
                baseline_qualities.append(quality)
            
            # 分析攻击后路径
            attack_qualities = []
            for path in attack_path_list:
                quality = self.analyze_path_quality(path, self.post_attack_triplets, metric)
                attack_qualities.append(quality)
            
            # 计算聚合统计
            baseline_avg = statistics.mean([q['average_metric'] for q in baseline_qualities]) if baseline_qualities else 0.0
            attack_avg = statistics.mean([q['average_metric'] for q in attack_qualities]) if attack_qualities else 0.0
            
            baseline_coverage = statistics.mean([q['edge_coverage'] for q in baseline_qualities]) if baseline_qualities else 0.0
            attack_coverage = statistics.mean([q['edge_coverage'] for q in attack_qualities]) if attack_qualities else 0.0
            
            # 计算变化
            metric_change = attack_avg - baseline_avg
            coverage_change = attack_coverage - baseline_coverage
            
            # 路径连通性变化
            baseline_path_count = len(baseline_path_list)
            attack_path_count = len(attack_path_list)
            connectivity_change = attack_path_count - baseline_path_count
            
            comparison_results.append({
                'source': source,
                'target': target,
                'baseline_paths': baseline_path_count,
                'attack_paths': attack_path_count,
                'connectivity_change': connectivity_change,
                f'baseline_{metric}': baseline_avg,
                f'attack_{metric}': attack_avg,
                f'{metric}_change': metric_change,
                'baseline_coverage': baseline_coverage,
                'attack_coverage': attack_coverage,
                'coverage_change': coverage_change,
                'impact_severity': abs(metric_change) + abs(coverage_change) + abs(connectivity_change) * 0.1
            })
        
        df = pd.DataFrame(comparison_results)
        
        # 排序：影响最严重的在前
        df = df.sort_values('impact_severity', ascending=False)
        
        print(f"✅ 完成 {len(df)} 个路径对的影响分析")
        
        return df
    
    def generate_path_impact_report(self, center_nodes: List[str], max_path_length: int = 3, 
                                  output_file: str = None) -> Dict:
        """
        生成路径影响分析报告
        
        Args:
            center_nodes: 中心节点列表
            max_path_length: 最大路径长度
            output_file: 输出文件路径
            
        Returns:
            分析报告
        """
        print(f"\n🎯 生成路径影响分析报告")
        print(f"中心节点: {center_nodes}")
        print(f"最大路径长度: {max_path_length}")
        print("=" * 60)
        
        # 1. 发现路径
        path_discovery = self.find_paths_from_center(center_nodes, max_path_length)
        
        # 2. 分析置信度影响
        print("\n📊 分析置信度影响...")
        confidence_impact = self.compare_path_impacts(path_discovery, 'confidence')
        
        # 3. 分析准确度影响
        print("📊 分析准确度影响...")
        accuracy_impact = self.compare_path_impacts(path_discovery, 'accuracy')
        
        # 4. 生成统计摘要
        report = self._generate_report_summary(center_nodes, path_discovery, confidence_impact, accuracy_impact)
        
        # 5. 保存结果
        if output_file:
            self._save_report(report, confidence_impact, accuracy_impact, output_file)
        
        # 6. 打印摘要
        self._print_report_summary(report)
        
        return {
            'report': report,
            'confidence_impact': confidence_impact,
            'accuracy_impact': accuracy_impact,
            'path_discovery': path_discovery
        }
    
    def _generate_report_summary(self, center_nodes: List[str], path_discovery: Dict, 
                               confidence_impact: pd.DataFrame, accuracy_impact: pd.DataFrame) -> Dict:
        """生成报告摘要"""
        # 基本统计
        total_path_pairs = len(set(path_discovery['baseline_paths'].keys()) | set(path_discovery['attack_paths'].keys()))
        
        # 连通性统计
        baseline_total_paths = sum(len(paths) for paths in path_discovery['baseline_paths'].values())
        attack_total_paths = sum(len(paths) for paths in path_discovery['attack_paths'].values())
        
        # 置信度变化统计
        conf_changes = confidence_impact['confidence_change'].values
        conf_positive = sum(1 for c in conf_changes if c > 0.01)
        conf_negative = sum(1 for c in conf_changes if c < -0.01)
        conf_neutral = len(conf_changes) - conf_positive - conf_negative
        
        # 准确度变化统计
        acc_changes = accuracy_impact['accuracy_change'].values
        acc_positive = sum(1 for c in acc_changes if c > 0.01)
        acc_negative = sum(1 for c in acc_changes if c < -0.01)
        acc_neutral = len(acc_changes) - acc_positive - acc_negative
        
        # 最受影响的路径
        top_confidence_impacted = confidence_impact.head(5)[['source', 'target', 'confidence_change', 'impact_severity']].to_dict('records')
        top_accuracy_impacted = accuracy_impact.head(5)[['source', 'target', 'accuracy_change', 'impact_severity']].to_dict('records')
        
        return {
            'analysis_metadata': {
                'center_nodes': center_nodes,
                'analysis_time': datetime.now().isoformat(),
                'total_path_pairs': total_path_pairs,
                'relevant_nodes': len(path_discovery['relevant_nodes'])
            },
            'connectivity_impact': {
                'baseline_total_paths': baseline_total_paths,
                'attack_total_paths': attack_total_paths,
                'connectivity_change': attack_total_paths - baseline_total_paths,
                'connectivity_change_rate': (attack_total_paths - baseline_total_paths) / baseline_total_paths * 100 if baseline_total_paths > 0 else 0
            },
            'confidence_impact': {
                'mean_change': float(np.mean(conf_changes)),
                'std_change': float(np.std(conf_changes)),
                'positive_changes': conf_positive,
                'negative_changes': conf_negative,
                'neutral_changes': conf_neutral,
                'max_positive_change': float(np.max(conf_changes)),
                'max_negative_change': float(np.min(conf_changes))
            },
            'accuracy_impact': {
                'mean_change': float(np.mean(acc_changes)),
                'std_change': float(np.std(acc_changes)),
                'positive_changes': acc_positive,
                'negative_changes': acc_negative,
                'neutral_changes': acc_neutral,
                'max_positive_change': float(np.max(acc_changes)),
                'max_negative_change': float(np.min(acc_changes))
            },
            'top_impacted_paths': {
                'confidence': top_confidence_impacted,
                'accuracy': top_accuracy_impacted
            }
        }
    
    def _save_report(self, report: Dict, confidence_impact: pd.DataFrame, accuracy_impact: pd.DataFrame, output_file: str):
        """保存分析报告"""
        # 确保输出目录存在
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # 保存完整结果
        full_results = {
            'report_summary': report,
            'confidence_impact_details': confidence_impact.to_dict('records'),
            'accuracy_impact_details': accuracy_impact.to_dict('records')
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        
        # 保存CSV文件
        csv_base = output_file.replace('.json', '')
        confidence_impact.to_csv(f"{csv_base}_confidence.csv", index=False)
        accuracy_impact.to_csv(f"{csv_base}_accuracy.csv", index=False)
        
        print(f"✅ 报告已保存:")
        print(f"  - 完整报告: {output_file}")
        print(f"  - 置信度CSV: {csv_base}_confidence.csv")
        print(f"  - 准确度CSV: {csv_base}_accuracy.csv")
    
    def _print_report_summary(self, report: Dict):
        """打印报告摘要"""
        print(f"\n📋 路径影响分析报告摘要")
        print("=" * 50)
        
        meta = report['analysis_metadata']
        print(f"中心节点: {', '.join(meta['center_nodes'])}")
        print(f"分析路径对: {meta['total_path_pairs']}")
        print(f"相关节点: {meta['relevant_nodes']}")
        
        conn = report['connectivity_impact']
        print(f"\n🔗 连通性影响:")
        print(f"  基线路径总数: {conn['baseline_total_paths']}")
        print(f"  攻击后路径总数: {conn['attack_total_paths']}")
        print(f"  连通性变化: {conn['connectivity_change']:+d} ({conn['connectivity_change_rate']:+.1f}%)")
        
        conf = report['confidence_impact']
        print(f"\n📊 置信度影响:")
        print(f"  平均变化: {conf['mean_change']:+.4f} ± {conf['std_change']:.4f}")
        print(f"  提升路径: {conf['positive_changes']} ({conf['positive_changes']/(conf['positive_changes']+conf['negative_changes']+conf['neutral_changes'])*100:.1f}%)")
        print(f"  下降路径: {conf['negative_changes']} ({conf['negative_changes']/(conf['positive_changes']+conf['negative_changes']+conf['neutral_changes'])*100:.1f}%)")
        print(f"  最大变化: {conf['max_positive_change']:+.4f} / {conf['max_negative_change']:+.4f}")
        
        acc = report['accuracy_impact']
        print(f"\n🎯 准确度影响:")
        print(f"  平均变化: {acc['mean_change']:+.4f} ± {acc['std_change']:.4f}")
        print(f"  提升路径: {acc['positive_changes']} ({acc['positive_changes']/(acc['positive_changes']+acc['negative_changes']+acc['neutral_changes'])*100:.1f}%)")
        print(f"  下降路径: {acc['negative_changes']} ({acc['negative_changes']/(acc['positive_changes']+acc['negative_changes']+acc['neutral_changes'])*100:.1f}%)")
        print(f"  最大变化: {acc['max_positive_change']:+.4f} / {acc['max_negative_change']:+.4f}")
        
        print(f"\n🔥 最受影响的路径 (置信度):")
        for i, path in enumerate(report['top_impacted_paths']['confidence'][:3]):
            print(f"  {i+1}. {path['source']} → {path['target']}: {path['confidence_change']:+.4f}")
        
        print(f"\n🔥 最受影响的路径 (准确度):")
        for i, path in enumerate(report['top_impacted_paths']['accuracy'][:3]):
            print(f"  {i+1}. {path['source']} → {path['target']}: {path['accuracy_change']:+.4f}")

def main():
    parser = argparse.ArgumentParser(description="路径影响分析")
    parser.add_argument("--baseline", type=str, required=True,
                       help="基线评估结果文件")
    parser.add_argument("--post_attack", type=str, required=True,
                       help="攻击后评估结果文件")
    parser.add_argument("--center_nodes", type=str, nargs='+', required=True,
                       help="中心节点列表")
    parser.add_argument("--max_path_length", type=int, default=3,
                       help="最大路径长度")
    parser.add_argument("--output", type=str, default=None,
                       help="输出文件路径")
    
    args = parser.parse_args()
    
    # 设置默认输出文件
    if not args.output:
        center_names = "_".join([name.replace(" ", "_") for name in args.center_nodes[:2]])
        args.output = f"analysis/path_impact_{center_names}.json"
    
    # 创建分析器
    analyzer = PathImpactAnalyzer(args.baseline, args.post_attack)
    
    # 生成路径影响报告
    results = analyzer.generate_path_impact_report(
        center_nodes=args.center_nodes,
        max_path_length=args.max_path_length,
        output_file=args.output
    )

if __name__ == "__main__":
    main()