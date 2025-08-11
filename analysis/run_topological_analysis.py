#!/usr/bin/env python3
"""
图拓扑量化分析脚本
功能：对攻击前后的知识图进行深入的量化图论分析

核心功能:
1. 邻域衰减分析 (Neighborhood Decay Analysis)
2. 路径完整性分析 (Path Integrity Analysis) 
3. 节点重要性变化分析 (Node Centrality Shift Analysis)
"""

import json
import argparse
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import statistics


def build_knowledge_graph(evaluation_results: Dict) -> nx.DiGraph:
    """
    从评估结果构建知识图
    
    Args:
        evaluation_results: 评估结果JSON数据
        
    Returns:
        networkx.DiGraph: 构建的知识图
    """
    G = nx.DiGraph()
    
    # 添加节点和边
    for result in evaluation_results.get('results', []):
        head = result['head']
        relation = result['relation']
        tail = result['tail']
        
        # 确保节点存在
        if not G.has_node(head):
            G.add_node(head)
        if not G.has_node(tail):
            G.add_node(tail)
            
        # 添加边和属性
        confidence = result.get('confidence', 0.0)
        accuracy = result.get('accuracy_score', 0.0) if result.get('accuracy_score') is not None else 0.0
        
        G.add_edge(
            head, tail,
            relation=relation,
            confidence=confidence,
            accuracy=accuracy
        )
    
    return G


def neighborhood_decay_analysis(G_baseline: nx.DiGraph, G_attacked: nx.DiGraph, 
                              target_node: str, max_hops: int = 2) -> Dict[str, Any]:
    """
    邻域衰减分析：计算被攻击节点周围邻居的置信度/准确度衰减
    
    Args:
        G_baseline: 基线知识图
        G_attacked: 攻击后知识图
        target_node: 被攻击节点
        max_hops: 最大跳数
        
    Returns:
        Dict: 分析结果
    """
    results = {}
    
    # 获取目标节点的n-hop邻居
    for hop in range(1, max_hops + 1):
        # 基线图中的n-hop邻居
        baseline_neighbors = set()
        for node in nx.single_source_shortest_path_length(G_baseline, target_node, cutoff=hop).keys():
            if node != target_node:
                baseline_neighbors.add(node)
                
        # 攻击后图中的n-hop邻居
        attacked_neighbors = set()
        for node in nx.single_source_shortest_path_length(G_attacked, target_node, cutoff=hop).keys():
            if node != target_node:
                attacked_neighbors.add(node)
        
        # 计算置信度和准确度的统计信息
        baseline_confidences = []
        attacked_confidences = []
        baseline_accuracies = []
        attacked_accuracies = []
        
        # 收集基线图中邻居边的属性
        for neighbor in baseline_neighbors:
            # 检查是否存在从target_node到neighbor的边
            if G_baseline.has_edge(target_node, neighbor):
                edge_data = G_baseline[target_node][neighbor]
                baseline_confidences.append(edge_data.get('confidence', 0.0))
                baseline_accuracies.append(edge_data.get('accuracy', 0.0))
            else:
                # 查找通过其他路径连接的边
                try:
                    paths = list(nx.all_simple_paths(G_baseline, target_node, neighbor, cutoff=hop))
                    if paths:
                        # 取第一条路径的边属性
                        path = paths[0]
                        for i in range(len(path)-1):
                            edge_data = G_baseline[path[i]][path[i+1]]
                            baseline_confidences.append(edge_data.get('confidence', 0.0))
                            baseline_accuracies.append(edge_data.get('accuracy', 0.0))
                except:
                    pass
        
        # 收集攻击后图中邻居边的属性
        for neighbor in attacked_neighbors:
            # 检查是否存在从target_node到neighbor的边
            if G_attacked.has_edge(target_node, neighbor):
                edge_data = G_attacked[target_node][neighbor]
                attacked_confidences.append(edge_data.get('confidence', 0.0))
                attacked_accuracies.append(edge_data.get('accuracy', 0.0))
            else:
                # 查找通过其他路径连接的边
                try:
                    paths = list(nx.all_simple_paths(G_attacked, target_node, neighbor, cutoff=hop))
                    if paths:
                        # 取第一条路径的边属性
                        path = paths[0]
                        for i in range(len(path)-1):
                            edge_data = G_attacked[path[i]][path[i+1]]
                            attacked_confidences.append(edge_data.get('confidence', 0.0))
                            attacked_accuracies.append(edge_data.get('accuracy', 0.0))
                except:
                    pass
        
        # 计算衰减百分比
        avg_baseline_conf = np.mean(baseline_confidences) if baseline_confidences else 0.0
        avg_attacked_conf = np.mean(attacked_confidences) if attacked_confidences else 0.0
        avg_baseline_acc = np.mean(baseline_accuracies) if baseline_accuracies else 0.0
        avg_attacked_acc = np.mean(attacked_accuracies) if attacked_accuracies else 0.0
        
        conf_decay = ((avg_baseline_conf - avg_attacked_conf) / avg_baseline_conf * 100) if avg_baseline_conf > 0 else 0.0
        acc_decay = ((avg_baseline_acc - avg_attacked_acc) / avg_baseline_acc * 100) if avg_baseline_acc > 0 else 0.0
        
        results[f"{hop}-hop"] = {
            "baseline_neighbors": len(baseline_neighbors),
            "attacked_neighbors": len(attacked_neighbors),
            "avg_baseline_confidence": float(avg_baseline_conf),
            "avg_attacked_confidence": float(avg_attacked_conf),
            "confidence_decay_percent": float(conf_decay),
            "avg_baseline_accuracy": float(avg_baseline_acc),
            "avg_attacked_accuracy": float(avg_attacked_acc),
            "accuracy_decay_percent": float(acc_decay)
        }
    
    return results


def path_integrity_analysis(G_baseline: nx.DiGraph, G_attacked: nx.DiGraph, 
                          num_paths: int = 10) -> Dict[str, Any]:
    """
    路径完整性分析：找出图中重要路径，计算攻击前后路径的累积置信度
    
    Args:
        G_baseline: 基线知识图
        G_attacked: 攻击后知识图
        num_paths: 分析的路径数量
        
    Returns:
        Dict: 分析结果
    """
    # 找出图中重要的节点对（基于PageRank或其他指标）
    try:
        baseline_pagerank = nx.pagerank(G_baseline)
        attacked_pagerank = nx.pagerank(G_attacked)
        
        # 获取高PageRank节点
        high_rank_nodes_baseline = sorted(baseline_pagerank.items(), key=lambda x: x[1], reverse=True)[:20]
        high_rank_nodes_attacked = sorted(attacked_pagerank.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # 合并重要节点
        important_nodes = set([node for node, _ in high_rank_nodes_baseline])
        important_nodes.update([node for node, _ in high_rank_nodes_attacked])
        important_nodes = list(important_nodes)[:10]  # 限制节点数量
        
        # 计算重要节点之间的最短路径
        paths_analysis = []
        for i, source in enumerate(important_nodes):
            for target in important_nodes[i+1:]:
                try:
                    # 基线图中的最短路径
                    if nx.has_path(G_baseline, source, target):
                        baseline_path = nx.shortest_path(G_baseline, source, target)
                        baseline_cumulative_conf = 1.0
                        for j in range(len(baseline_path)-1):
                            u, v = baseline_path[j], baseline_path[j+1]
                            if G_baseline.has_edge(u, v):
                                baseline_cumulative_conf *= G_baseline[u][v].get('confidence', 0.0)
                        
                        # 攻击后图中的最短路径
                        if nx.has_path(G_attacked, source, target):
                            attacked_path = nx.shortest_path(G_attacked, source, target)
                            attacked_cumulative_conf = 1.0
                            for j in range(len(attacked_path)-1):
                                u, v = attacked_path[j], attacked_path[j+1]
                                if G_attacked.has_edge(u, v):
                                    attacked_cumulative_conf *= G_attacked[u][v].get('confidence', 0.0)
                            
                            # 计算路径完整性衰减
                            path_decay = ((baseline_cumulative_conf - attacked_cumulative_conf) / 
                                        baseline_cumulative_conf * 100) if baseline_cumulative_conf > 0 else 0.0
                            
                            paths_analysis.append({
                                "source": source,
                                "target": target,
                                "baseline_path_length": len(baseline_path),
                                "attacked_path_length": len(attacked_path),
                                "baseline_cumulative_confidence": float(baseline_cumulative_conf),
                                "attacked_cumulative_confidence": float(attacked_cumulative_conf),
                                "confidence_decay_percent": float(path_decay)
                            })
                except nx.NetworkXNoPath:
                    continue
        
        # 按衰减程度排序并返回前N条路径
        paths_analysis.sort(key=lambda x: x["confidence_decay_percent"], reverse=True)
        return paths_analysis[:num_paths]
        
    except Exception as e:
        print(f"Error in path integrity analysis: {e}")
        return []


def node_centrality_shift_analysis(G_baseline: nx.DiGraph, G_attacked: nx.DiGraph, 
                                 target_nodes: List[str] = None) -> Dict[str, Any]:
    """
    节点重要性变化分析：计算关键节点的中心性指标变化
    
    Args:
        G_baseline: 基线知识图
        G_attacked: 攻击后知识图
        target_nodes: 目标节点列表（如被攻击实体）
        
    Returns:
        Dict: 分析结果
    """
    results = {}
    
    try:
        # 计算各种中心性指标
        baseline_degree_centrality = nx.degree_centrality(G_baseline)
        attacked_degree_centrality = nx.degree_centrality(G_attacked)
        
        baseline_betweenness_centrality = nx.betweenness_centrality(G_baseline)
        attacked_betweenness_centrality = nx.betweenness_centrality(G_attacked)
        
        baseline_pagerank = nx.pagerank(G_baseline)
        attacked_pagerank = nx.pagerank(G_attacked)
        
        # 分析目标节点的中心性变化
        if target_nodes:
            for node in target_nodes:
                results[node] = {
                    "degree_centrality": {
                        "baseline": baseline_degree_centrality.get(node, 0.0),
                        "attacked": attacked_degree_centrality.get(node, 0.0),
                        "change": attacked_degree_centrality.get(node, 0.0) - baseline_degree_centrality.get(node, 0.0)
                    },
                    "betweenness_centrality": {
                        "baseline": baseline_betweenness_centrality.get(node, 0.0),
                        "attacked": attacked_betweenness_centrality.get(node, 0.0),
                        "change": attacked_betweenness_centrality.get(node, 0.0) - baseline_betweenness_centrality.get(node, 0.0)
                    },
                    "pagerank": {
                        "baseline": baseline_pagerank.get(node, 0.0),
                        "attacked": attacked_pagerank.get(node, 0.0),
                        "change": attacked_pagerank.get(node, 0.0) - baseline_pagerank.get(node, 0.0)
                    }
                }
        
        # 分析整体图的中心性变化（前10个变化最大的节点）
        all_nodes = set(G_baseline.nodes()) | set(G_attacked.nodes())
        centrality_changes = []
        
        for node in all_nodes:
            degree_change = attacked_degree_centrality.get(node, 0.0) - baseline_degree_centrality.get(node, 0.0)
            betweenness_change = attacked_betweenness_centrality.get(node, 0.0) - baseline_betweenness_centrality.get(node, 0.0)
            pagerank_change = attacked_pagerank.get(node, 0.0) - baseline_pagerank.get(node, 0.0)
            
            total_change = abs(degree_change) + abs(betweenness_change) + abs(pagerank_change)
            centrality_changes.append((node, total_change, degree_change, betweenness_change, pagerank_change))
        
        # 按总变化排序
        centrality_changes.sort(key=lambda x: x[1], reverse=True)
        
        top_changes = []
        for node, total_change, degree_change, betweenness_change, pagerank_change in centrality_changes[:10]:
            top_changes.append({
                "node": node,
                "total_change": total_change,
                "degree_centrality_change": degree_change,
                "betweenness_centrality_change": betweenness_change,
                "pagerank_change": pagerank_change
            })
        
        results["top_centrality_changes"] = top_changes
        
    except Exception as e:
        print(f"Error in node centrality shift analysis: {e}")
    
    return results


def generate_analysis_report(baseline_file: str, attacked_file: str, 
                           target_triplet: Tuple[str, str, str], 
                           output_file: str):
    """
    生成完整的图拓扑量化分析报告
    
    Args:
        baseline_file: 基线评估结果文件路径
        attacked_file: 攻击后评估结果文件路径
        target_triplet: 被攻击的三元组 (head, relation, tail)
        output_file: 输出报告文件路径
    """
    # 加载评估结果
    with open(baseline_file, 'r') as f:
        baseline_results = json.load(f)
    
    with open(attacked_file, 'r') as f:
        attacked_results = json.load(f)
    
    # 构建知识图
    G_baseline = build_knowledge_graph(baseline_results)
    G_attacked = build_knowledge_graph(attacked_results)
    
    print(f"_baseline graph: {G_baseline.number_of_nodes()} nodes, {G_baseline.number_of_edges()} edges")
    print(f"Attacked graph: {G_attacked.number_of_nodes()} nodes, {G_attacked.number_of_edges()} edges")
    
    # 执行各项分析
    target_head, target_relation, target_tail = target_triplet
    target_node = target_head  # 我们主要关注头实体
    
    # 1. 邻域衰减分析
    print("Performing neighborhood decay analysis...")
    neighborhood_decay = neighborhood_decay_analysis(G_baseline, G_attacked, target_node)
    
    # 2. 路径完整性分析
    print("Performing path integrity analysis...")
    path_integrity = path_integrity_analysis(G_baseline, G_attacked)
    
    # 3. 节点重要性变化分析
    print("Performing node centrality shift analysis...")
    node_centrality_shift = node_centrality_shift_analysis(
        G_baseline, G_attacked, [target_head, target_tail]
    )
    
    # 生成报告
    report = {
        "analysis_summary": {
            "baseline_graph_stats": {
                "nodes": G_baseline.number_of_nodes(),
                "edges": G_baseline.number_of_edges()
            },
            "attacked_graph_stats": {
                "nodes": G_attacked.number_of_nodes(),
                "edges": G_attacked.number_of_edges()
            },
            "target_triplet": {
                "head": target_head,
                "relation": target_relation,
                "tail": target_tail
            }
        },
        "neighborhood_decay_analysis": neighborhood_decay,
        "path_integrity_analysis": path_integrity,
        "node_centrality_shift_analysis": node_centrality_shift
    }
    
    # 保存报告
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 生成Markdown格式的可读报告
    md_report = generate_markdown_report(report)
    md_file = output_file.replace('.json', '.md')
    with open(md_file, 'w') as f:
        f.write(md_report)
    
    print(f"Analysis report saved to {output_file}")
    print(f"Markdown report saved to {md_file}")


def generate_markdown_report(report: Dict) -> str:
    """
    生成Markdown格式的分析报告
    
    Args:
        report: JSON格式的分析报告
        
    Returns:
        str: Markdown格式的报告内容
    """
    md_lines = []
    
    # 标题
    md_lines.append("# 知识图谱拓扑结构量化分析报告")
    md_lines.append("")
    
    # 摘要
    summary = report["analysis_summary"]
    md_lines.append("## 摘要")
    md_lines.append("")
    md_lines.append(f"- 基线图: {summary['baseline_graph_stats']['nodes']} 节点, {summary['baseline_graph_stats']['edges']} 边")
    md_lines.append(f"- 攻击后图: {summary['attacked_graph_stats']['nodes']} 节点, {summary['attacked_graph_stats']['edges']} 边")
    md_lines.append(f"- 目标三元组: ({summary['target_triplet']['head']}, {summary['target_triplet']['relation']}, {summary['target_triplet']['tail']})")
    md_lines.append("")
    
    # 邻域衰减分析
    md_lines.append("## 1. 邻域衰减分析")
    md_lines.append("")
    md_lines.append("| 跳数 | 基线邻居数 | 攻击后邻居数 | 基线平均置信度 | 攻击后平均置信度 | 置信度衰减(%) | 基线平均准确度 | 攻击后平均准确度 | 准确度衰减(%) |")
    md_lines.append("|------|------------|--------------|----------------|------------------|----------------|----------------|------------------|----------------|")
    
    for hop, data in report["neighborhood_decay_analysis"].items():
        md_lines.append(f"| {hop} | {data['baseline_neighbors']} | {data['attacked_neighbors']} | {data['avg_baseline_confidence']:.4f} | {data['avg_attacked_confidence']:.4f} | {data['confidence_decay_percent']:.2f} | {data['avg_baseline_accuracy']:.2f} | {data['avg_attacked_accuracy']:.2f} | {data['accuracy_decay_percent']:.2f} |")
    md_lines.append("")
    
    # 路径完整性分析
    md_lines.append("## 2. 路径完整性分析")
    md_lines.append("")
    md_lines.append("前10条受影响最严重的路径:")
    md_lines.append("")
    md_lines.append("| 源节点 | 目标节点 | 基线路径长度 | 攻击后路径长度 | 基线累积置信度 | 攻击后累积置信度 | 置信度衰减(%) |")
    md_lines.append("|--------|----------|--------------|----------------|----------------|------------------|----------------|")
    
    for path in report["path_integrity_analysis"]:
        md_lines.append(f"| {path['source'][:20]} | {path['target'][:20]} | {path['baseline_path_length']} | {path['attacked_path_length']} | {path['baseline_cumulative_confidence']:.2e} | {path['attacked_cumulative_confidence']:.2e} | {path['confidence_decay_percent']:.2f} |")
    md_lines.append("")
    
    # 节点重要性变化分析
    md_lines.append("## 3. 节点重要性变化分析")
    md_lines.append("")
    
    # 目标节点分析
    centrality_data = report["node_centrality_shift_analysis"]
    for node_name in [summary['target_triplet']['head'], summary['target_triplet']['tail']]:
        if node_name in centrality_data:
            node_info = centrality_data[node_name]
            md_lines.append(f"### 节点: {node_name}")
            md_lines.append("")
            md_lines.append("| 中心性指标 | 基线值 | 攻击后值 | 变化量 |")
            md_lines.append("|------------|--------|----------|--------|")
            md_lines.append(f"| 度中心性 | {node_info['degree_centrality']['baseline']:.6f} | {node_info['degree_centrality']['attacked']:.6f} | {node_info['degree_centrality']['change']:.6f} |")
            md_lines.append(f"| 介数中心性 | {node_info['betweenness_centrality']['baseline']:.6f} | {node_info['betweenness_centrality']['attacked']:.6f} | {node_info['betweenness_centrality']['change']:.6f} |")
            md_lines.append(f"| PageRank | {node_info['pagerank']['baseline']:.6f} | {node_info['pagerank']['attacked']:.6f} | {node_info['pagerank']['change']:.6f} |")
            md_lines.append("")
    
    # 重要性变化最大的节点
    md_lines.append("### 重要性变化最大的节点 (Top 10)")
    md_lines.append("")
    md_lines.append("| 节点 | 总变化量 | 度中心性变化 | 介数中心性变化 | PageRank变化 |")
    md_lines.append("|------|----------|--------------|----------------|--------------|")
    
    for node_change in centrality_data.get("top_centrality_changes", []):
        md_lines.append(f"| {node_change['node'][:20]} | {node_change['total_change']:.6f} | {node_change['degree_centrality_change']:.6f} | {node_change['betweenness_centrality_change']:.6f} | {node_change['pagerank_change']:.6f} |")
    
    return "\n".join(md_lines)


def main():
    parser = argparse.ArgumentParser(description="知识图谱拓扑结构量化分析")
    parser.add_argument("--baseline", required=True, help="基线评估结果文件路径")
    parser.add_argument("--attacked", required=True, help="攻击后评估结果文件路径")
    parser.add_argument("--target-head", required=True, help="被攻击三元组的头实体")
    parser.add_argument("--target-relation", required=True, help="被攻击三元组的关系")
    parser.add_argument("--target-tail", required=True, help="被攻击三元组的尾实体")
    parser.add_argument("--output", default="topological_analysis_report.json", help="输出报告文件路径")
    
    args = parser.parse_args()
    
    target_triplet = (args.target_head, args.target_relation, args.target_tail)
    
    generate_analysis_report(
        baseline_file=args.baseline,
        attacked_file=args.attacked,
        target_triplet=target_triplet,
        output_file=args.output
    )


if __name__ == "__main__":
    main()