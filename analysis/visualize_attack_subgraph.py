#!/usr/bin/env python3
"""
攻击前后子图变化可视化脚本

功能:
1. 加载攻击前（baseline）和攻击后（post-attack）的评估结果JSON文件。
2. 从结果中提取知识三元组和准确度分数。
3. 使用 networkx 构建两个知识图谱（攻击前/后）。
4. 根据准确度的变化为节点着色，直观展示攻击影响。
   - 黄色: 攻击目标实体 (target tail, toxic answer)
   - 红色: 知识被污染的实体 (准确度显著下降)
   - 绿色: 知识被增强的实体 (准确度显著上升)
   - 灰色: 影响不大的实体
5. 使用 matplotlib 生成并排的子图对比图像并保存。
"""

import json
import networkx as nx
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Set, Tuple

def load_evaluation_results(filepath: str) -> Dict[Tuple[str, str, str], Dict]:
    """从评估结果文件中加载三元组和其对应的结果"""
    results = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        evaluation_entries = data.get('results', [])
        for entry in evaluation_entries:
            if all(k in entry for k in ['head', 'relation', 'tail']):
                triplet = (entry['head'], entry['relation'], entry['tail'])
                results[triplet] = entry
    except FileNotFoundError:
        print(f"❌ 错误: 文件未找到 {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"❌ 错误: 文件格式无效 {filepath}")
        return None
    return results

def build_graph_from_results(results: Dict[Tuple[str, str, str], Dict]) -> nx.MultiDiGraph:
    """从评估结果构建一个有向多重图"""
    G = nx.MultiDiGraph()
    for triplet, data in results.items():
        head, relation, tail = triplet
        G.add_node(head, type='entity')
        G.add_node(tail, type='entity')
        G.add_edge(head, tail, key=relation, label=relation)
    return G

def get_node_colors(
    baseline_results: Dict, 
    post_attack_results: Dict, 
    nodes: List[str],
    target_tail: str,
    toxic_answer: str,
    threshold: int = 30
) -> List[str]:
    """根据准确度变化确定节点颜色"""
    colors = []
    
    # 构建一个从实体到其作为尾节点时准确度的映射
    def get_tail_accuracy_map(results: Dict) -> Dict[str, float]:
        acc_map = {}
        for triplet, data in results.items():
            tail = triplet[2]
            score = data.get('accuracy_score')
            if score is not None:
                if tail not in acc_map:
                    acc_map[tail] = []
                acc_map[tail].append(score)
        # 计算平均分
        return {k: sum(v) / len(v) for k, v in acc_map.items()}

    baseline_acc = get_tail_accuracy_map(baseline_results)
    post_attack_acc = get_tail_accuracy_map(post_attack_results)

    for node in nodes:
        if node == target_tail or node == toxic_answer:
            colors.append('gold') # 目标和毒答案用金色
            continue

        baseline_score = baseline_acc.get(node)
        post_attack_score = post_attack_acc.get(node)

        if baseline_score is not None and post_attack_score is not None:
            diff = post_attack_score - baseline_score
            if diff < -threshold:
                colors.append('red')  # 显著下降 - 被污染
            elif diff > threshold:
                colors.append('green') # 显著上升 - 被增强
            else:
                colors.append('lightgrey') # 变化不大
        else:
            colors.append('lightgrey') # 无数据
            
    return colors

def visualize_graphs(
    G_baseline: nx.MultiDiGraph, 
    G_post_attack: nx.MultiDiGraph, 
    node_colors_baseline: List,
    node_colors_post_attack: List,
    output_file: str
):
    """使用matplotlib可视化并排的两个图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    fig.suptitle('Knowledge Subgraph Comparison: Before vs. After Attack', fontsize=20)

    # 绘制基线图
    pos1 = nx.spring_layout(G_baseline, seed=42, k=0.8)
    nx.draw_networkx_nodes(G_baseline, pos1, node_color=node_colors_baseline, node_size=2500, ax=ax1)
    nx.draw_networkx_labels(G_baseline, pos1, font_size=10, ax=ax1)
    nx.draw_networkx_edges(G_baseline, pos1, edgelist=G_baseline.edges(keys=True), 
                           arrowstyle='->', arrowsize=15, connectionstyle='arc3,rad=0.1', ax=ax1)
    edge_labels1 = nx.get_edge_attributes(G_baseline, 'label')
    nx.draw_networkx_edge_labels(G_baseline, pos1, edge_labels=edge_labels1, font_size=8, ax=ax1)
    ax1.set_title('Baseline Knowledge Graph', fontsize=16)
    ax1.margins(0.1)

    # 绘制攻击后图
    pos2 = nx.spring_layout(G_post_attack, seed=42, k=0.8)
    nx.draw_networkx_nodes(G_post_attack, pos2, node_color=node_colors_post_attack, node_size=2500, ax=ax2)
    nx.draw_networkx_labels(G_post_attack, pos2, font_size=10, ax=ax2)
    nx.draw_networkx_edges(G_post_attack, pos2, edgelist=G_post_attack.edges(keys=True), 
                           arrowstyle='->', arrowsize=15, connectionstyle='arc3,rad=0.1', ax=ax2)
    edge_labels2 = nx.get_edge_attributes(G_post_attack, 'label')
    nx.draw_networkx_edge_labels(G_post_attack, pos2, edge_labels=edge_labels2, font_size=8, ax=ax2)
    ax2.set_title('Post-Attack Knowledge Graph', fontsize=16)
    ax2.margins(0.1)
    
    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Target/Toxic Entity', markerfacecolor='gold', markersize=12),
        plt.Line2D([0], [0], marker='o', color='w', label='Contaminated Entity (Acc. Drop)', markerfacecolor='red', markersize=12),
        plt.Line2D([0], [0], marker='o', color='w', label='Strengthened Entity (Acc. Rise)', markerfacecolor='green', markersize=12),
        plt.Line2D([0], [0], marker='o', color='w', label='Neutral/Unaffected Entity', markerfacecolor='lightgrey', markersize=12)
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # 调整布局为图例留出空间
    
    # 保存图像
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 子图可视化结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize the subgraph changes before and after a knowledge poisoning attack.")
    parser.add_argument("--baseline", type=str, required=True, 
                        help="Path to the baseline evaluation JSON file.")
    parser.add_argument("--post_attack", type=str, required=True, 
                        help="Path to the post-attack evaluation JSON file.")
    parser.add_argument("--output_file", type=str, default="analysis/figures/attack_subgraph_comparison.png",
                        help="Path to save the output visualization image.")
    parser.add_argument("--target_tail", type=str, required=True,
                        help="The original 'tail' entity of the targeted knowledge triplet (e.g., 'oceans').")
    parser.add_argument("--toxic_answer", type=str, required=True,
                        help="The toxic answer injected during the attack (e.g., 'mountains').")
    parser.add_argument("--change_threshold", type=int, default=30,
                        help="Accuracy score change threshold to consider a node 'contaminated' or 'strengthened'.")
    
    args = parser.parse_args()

    # 1. 加载数据
    print("🔄 正在加载评估结果...")
    baseline_results = load_evaluation_results(args.baseline)
    post_attack_results = load_evaluation_results(args.post_attack)

    if baseline_results is None or post_attack_results is None:
        return

    # 2. 构建图
    print("🧠 正在构建知识图谱...")
    G_baseline = build_graph_from_results(baseline_results)
    G_post_attack = build_graph_from_results(post_attack_results)

    # 确保两个图有相同的节点集，便于比较
    all_nodes = sorted(list(set(G_baseline.nodes()) | set(G_post_attack.nodes())))
    G_baseline.add_nodes_from(all_nodes)
    G_post_attack.add_nodes_from(all_nodes)
    
    # 3. 计算节点颜色
    print("🎨 正在根据准确度变化计算节点颜色...")
    node_colors_baseline = get_node_colors(baseline_results, baseline_results, all_nodes, args.target_tail, args.toxic_answer)
    node_colors_post_attack = get_node_colors(baseline_results, post_attack_results, all_nodes, args.target_tail, args.toxic_answer, args.change_threshold)

    # 4. 可视化
    print("📊 正在生成可视化图像...")
    visualize_graphs(G_baseline, G_post_attack, node_colors_baseline, node_colors_post_attack, args.output_file)

if __name__ == '__main__':
    main()
