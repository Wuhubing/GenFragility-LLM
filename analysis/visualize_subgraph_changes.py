#!/usr/bin/env python3
"""
知识子图变化可视化脚本
功能：生成攻击前后知识子图的对比可视化图像
核心特点：围绕关键三元组构建1-2跳子图，用颜色和大小编码置信度/准确率变化

主要功能：
1. 子图提取 - 基于中心节点提取1-2跳邻域
2. 对比可视化 - 攻击前后的并排对比
3. 变化量化 - 用颜色强度表示变化程度
4. 交互式图表 - 支持节点悬停显示详细信息
"""

import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Set
from pathlib import Path
import argparse
from collections import defaultdict
import math

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class SubgraphVisualizer:
    """知识子图可视化器"""
    
    def __init__(self, baseline_file: str, post_attack_file: str):
        """
        初始化子图可视化器
        
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
        
        # 设置可视化样式
        plt.style.use('default')
        sns.set_palette("RdYlBu_r")
        
        print(f"📊 基线图: {self.baseline_graph.number_of_nodes()} 节点, {self.baseline_graph.number_of_edges()} 边")
        print(f"📊 攻击后图: {self.post_attack_graph.number_of_nodes()} 节点, {self.post_attack_graph.number_of_edges()} 边")
    
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
            
            # 添加节点（如果不存在）
            if not G.has_node(head):
                G.add_node(head)
            if not G.has_node(tail):
                G.add_node(tail)
            
            # 添加边及其属性
            confidence = result.get('confidence', 0.0) or 0.0
            accuracy = result.get('accuracy_score', 0.0) or 0.0
            
            G.add_edge(
                head, tail,
                relation=relation,
                confidence=confidence,
                accuracy=accuracy / 100.0 if accuracy > 1 else accuracy,  # 标准化到[0,1]
                weight=confidence  # 用置信度作为权重
            )
        
        return G
    
    def extract_subgraph(self, graph: nx.DiGraph, center_nodes: List[str], max_hops: int = 2) -> nx.DiGraph:
        """
        提取以指定节点为中心的子图
        
        Args:
            graph: 原始知识图
            center_nodes: 中心节点列表
            max_hops: 最大跳数
            
        Returns:
            提取的子图
        """
        subgraph_nodes = set(center_nodes)
        
        # 逐层扩展
        current_nodes = set(center_nodes)
        for hop in range(max_hops):
            next_nodes = set()
            for node in current_nodes:
                if node in graph:
                    # 添加出边邻居
                    next_nodes.update(graph.successors(node))
                    # 添加入边邻居
                    next_nodes.update(graph.predecessors(node))
            
            subgraph_nodes.update(next_nodes)
            current_nodes = next_nodes
            
            # 限制子图大小（避免过大）
            if len(subgraph_nodes) > 50:
                break
        
        # 提取子图
        subgraph = graph.subgraph(subgraph_nodes).copy()
        return subgraph
    
    def calculate_node_changes(self, center_nodes: List[str], max_hops: int = 2) -> Dict[str, Dict]:
        """
        计算节点在攻击前后的变化
        
        Args:
            center_nodes: 中心节点列表
            max_hops: 子图扩展跳数
            
        Returns:
            节点变化统计
        """
        # 提取子图
        baseline_subgraph = self.extract_subgraph(self.baseline_graph, center_nodes, max_hops)
        attack_subgraph = self.extract_subgraph(self.post_attack_graph, center_nodes, max_hops)
        
        # 计算节点变化
        node_changes = {}
        all_nodes = set(baseline_subgraph.nodes()) | set(attack_subgraph.nodes())
        
        for node in all_nodes:
            # 基线指标
            baseline_confidence = []
            baseline_accuracy = []
            
            if node in baseline_subgraph:
                for _, _, edge_data in baseline_subgraph.edges(node, data=True):
                    baseline_confidence.append(edge_data.get('confidence', 0.0))
                    baseline_accuracy.append(edge_data.get('accuracy', 0.0))
                for _, _, edge_data in baseline_subgraph.in_edges(node, data=True):
                    baseline_confidence.append(edge_data.get('confidence', 0.0))
                    baseline_accuracy.append(edge_data.get('accuracy', 0.0))
            
            # 攻击后指标
            attack_confidence = []
            attack_accuracy = []
            
            if node in attack_subgraph:
                for _, _, edge_data in attack_subgraph.edges(node, data=True):
                    attack_confidence.append(edge_data.get('confidence', 0.0))
                    attack_accuracy.append(edge_data.get('accuracy', 0.0))
                for _, _, edge_data in attack_subgraph.in_edges(node, data=True):
                    attack_confidence.append(edge_data.get('confidence', 0.0))
                    attack_accuracy.append(edge_data.get('accuracy', 0.0))
            
            # 计算平均值
            avg_baseline_conf = np.mean(baseline_confidence) if baseline_confidence else 0.0
            avg_baseline_acc = np.mean(baseline_accuracy) if baseline_accuracy else 0.0
            avg_attack_conf = np.mean(attack_confidence) if attack_confidence else 0.0
            avg_attack_acc = np.mean(attack_accuracy) if attack_accuracy else 0.0
            
            # 计算变化
            conf_change = avg_attack_conf - avg_baseline_conf
            acc_change = avg_attack_acc - avg_baseline_acc
            
            node_changes[node] = {
                'baseline_confidence': avg_baseline_conf,
                'attack_confidence': avg_attack_conf,
                'baseline_accuracy': avg_baseline_acc,
                'attack_accuracy': avg_attack_acc,
                'confidence_change': conf_change,
                'accuracy_change': acc_change,
                'is_center': node in center_nodes
            }
        
        return node_changes
    
    def visualize_subgraph_comparison(self, center_nodes: List[str], max_hops: int = 2, 
                                    metric: str = 'confidence', output_file: str = None):
        """
        可视化攻击前后的子图对比
        
        Args:
            center_nodes: 中心节点列表
            max_hops: 子图扩展跳数
            metric: 可视化指标 ('confidence' 或 'accuracy')
            output_file: 输出文件路径
        """
        # 计算节点变化
        node_changes = self.calculate_node_changes(center_nodes, max_hops)
        
        # 提取子图
        baseline_subgraph = self.extract_subgraph(self.baseline_graph, center_nodes, max_hops)
        attack_subgraph = self.extract_subgraph(self.post_attack_graph, center_nodes, max_hops)
        
        # 创建图形
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # 计算布局（所有子图使用相同布局）
        all_nodes = set(baseline_subgraph.nodes()) | set(attack_subgraph.nodes())
        union_graph = nx.DiGraph()
        union_graph.add_nodes_from(all_nodes)
        
        # 使用spring布局，中心节点固定位置
        pos = nx.spring_layout(union_graph, k=3, iterations=100, seed=42)
        
        # 调整中心节点位置
        if len(center_nodes) == 1:
            pos[center_nodes[0]] = (0, 0)
        else:
            for i, node in enumerate(center_nodes):
                angle = 2 * np.pi * i / len(center_nodes)
                pos[node] = (0.3 * np.cos(angle), 0.3 * np.sin(angle))
        
        # 绘制基线子图
        self._draw_subgraph(baseline_subgraph, pos, node_changes, ax1, 
                          title=f"基线图 ({metric})", metric=metric, phase='baseline')
        
        # 绘制攻击后子图  
        self._draw_subgraph(attack_subgraph, pos, node_changes, ax2,
                          title=f"攻击后图 ({metric})", metric=metric, phase='attack')
        
        # 绘制变化图
        self._draw_change_graph(union_graph, pos, node_changes, ax3, metric=metric)
        
        # 添加整体标题
        center_names = ', '.join([name[:15] + '...' if len(name) > 15 else name for name in center_nodes])
        fig.suptitle(f'知识子图攻击效果对比分析\n中心节点: {center_names}', fontsize=16, fontweight='bold')
        
        # 添加图例
        self._add_legend(fig)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✅ 子图对比可视化已保存: {output_file}")
        
        plt.show()
        
        # 打印统计信息
        self._print_subgraph_stats(node_changes, metric)
    
    def _draw_subgraph(self, subgraph: nx.DiGraph, pos: Dict, node_changes: Dict, 
                      ax, title: str, metric: str, phase: str):
        """绘制单个子图"""
        if subgraph.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "无数据", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        # 准备节点属性
        node_sizes = []
        node_colors = []
        
        for node in subgraph.nodes():
            change_data = node_changes.get(node, {})
            
            # 节点大小：中心节点更大
            if change_data.get('is_center', False):
                node_sizes.append(800)
            else:
                node_sizes.append(300)
            
            # 节点颜色：根据指标值
            if phase == 'baseline':
                value = change_data.get(f'baseline_{metric}', 0.0)
            else:  # attack
                value = change_data.get(f'attack_{metric}', 0.0)
            
            node_colors.append(value)
        
        # 绘制边
        nx.draw_networkx_edges(subgraph, pos, ax=ax, edge_color='gray', 
                              alpha=0.6, arrows=True, arrowsize=20, width=1.5)
        
        # 绘制节点
        nodes = nx.draw_networkx_nodes(subgraph, pos, ax=ax,
                                     node_size=node_sizes,
                                     node_color=node_colors,
                                     cmap='RdYlBu_r',
                                     alpha=0.8,
                                     vmin=0, vmax=1)
        
        # 添加节点标签
        labels = {node: node[:10] + '...' if len(node) > 10 else node 
                 for node in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, labels, ax=ax, font_size=8)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # 添加颜色条
        if nodes:
            cbar = plt.colorbar(nodes, ax=ax, shrink=0.8)
            cbar.set_label(f'{metric.title()}', rotation=270, labelpad=20)
    
    def _draw_change_graph(self, graph: nx.DiGraph, pos: Dict, node_changes: Dict, ax, metric: str):
        """绘制变化图"""
        if graph.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "无数据", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("变化图")
            return
        
        # 准备节点属性
        node_sizes = []
        node_colors = []
        
        changes = []
        for node in graph.nodes():
            change_data = node_changes.get(node, {})
            change_value = change_data.get(f'{metric}_change', 0.0)
            changes.append(change_value)
        
        # 标准化变化值用于着色
        if changes and (max(changes) != min(changes)):
            change_range = max(changes) - min(changes)
            normalized_changes = [(c - min(changes)) / change_range for c in changes]
        else:
            normalized_changes = [0.5] * len(changes)
        
        for i, node in enumerate(graph.nodes()):
            change_data = node_changes.get(node, {})
            
            # 节点大小：变化越大节点越大
            abs_change = abs(changes[i])
            size = 300 + abs_change * 1000
            node_sizes.append(min(size, 1000))  # 限制最大大小
            
            # 节点颜色：正变化蓝色，负变化红色
            node_colors.append(normalized_changes[i])
        
        # 绘制节点
        nodes = nx.draw_networkx_nodes(graph, pos, ax=ax,
                                     node_size=node_sizes,
                                     node_color=node_colors,
                                     cmap='RdBu_r',
                                     alpha=0.8,
                                     vmin=0, vmax=1)
        
        # 添加节点标签和变化值
        labels = {}
        for i, node in enumerate(graph.nodes()):
            short_name = node[:8] + '...' if len(node) > 8 else node
            change_val = changes[i]
            labels[node] = f'{short_name}\n({change_val:+.3f})'
        
        nx.draw_networkx_labels(graph, pos, labels, ax=ax, font_size=7)
        
        ax.set_title(f'{metric.title()} 变化图', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # 添加颜色条
        if nodes:
            cbar = plt.colorbar(nodes, ax=ax, shrink=0.8)
            cbar.set_label(f'{metric.title()} 变化', rotation=270, labelpad=20)
    
    def _add_legend(self, fig):
        """添加图例"""
        legend_elements = [
            mpatches.Circle((0, 0), 0.1, facecolor='red', alpha=0.8, 
                          label='高值/正变化'),
            mpatches.Circle((0, 0), 0.1, facecolor='blue', alpha=0.8,
                          label='低值/负变化'),
            mpatches.Circle((0, 0), 0.15, facecolor='gray', alpha=0.8,
                          label='中心节点(大)'),
            mpatches.Circle((0, 0), 0.08, facecolor='gray', alpha=0.8,
                          label='邻居节点(小)')
        ]
        
        fig.legend(handles=legend_elements, loc='lower center', 
                  bbox_to_anchor=(0.5, -0.05), ncol=4)
    
    def _print_subgraph_stats(self, node_changes: Dict, metric: str):
        """打印子图统计信息"""
        print(f"\n📊 子图变化统计 ({metric}):")
        print("=" * 50)
        
        changes = [data[f'{metric}_change'] for data in node_changes.values()]
        positive_changes = [c for c in changes if c > 0]
        negative_changes = [c for c in changes if c < 0]
        
        print(f"总节点数: {len(node_changes)}")
        print(f"平均变化: {np.mean(changes):.4f}")
        print(f"变化标准差: {np.std(changes):.4f}")
        print(f"正变化节点: {len(positive_changes)} ({len(positive_changes)/len(changes)*100:.1f}%)")
        print(f"负变化节点: {len(negative_changes)} ({len(negative_changes)/len(changes)*100:.1f}%)")
        
        if positive_changes:
            print(f"最大正变化: {max(positive_changes):.4f}")
        if negative_changes:
            print(f"最大负变化: {min(negative_changes):.4f}")
    
    def generate_multiple_subgraph_views(self, center_nodes_list: List[List[str]], 
                                       output_dir: str = "analysis/figures/subgraphs"):
        """
        生成多个子图视图
        
        Args:
            center_nodes_list: 中心节点列表的列表
            output_dir: 输出目录
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for i, center_nodes in enumerate(center_nodes_list):
            print(f"\n🎨 生成子图 {i+1}/{len(center_nodes_list)}: {center_nodes}")
            
            # 生成置信度视图
            output_file = f"{output_dir}/subgraph_{i+1}_confidence.png"
            self.visualize_subgraph_comparison(center_nodes, metric='confidence', 
                                             output_file=output_file)
            
            # 生成准确度视图
            output_file = f"{output_dir}/subgraph_{i+1}_accuracy.png"
            self.visualize_subgraph_comparison(center_nodes, metric='accuracy',
                                             output_file=output_file)

def main():
    parser = argparse.ArgumentParser(description="知识子图攻击效果可视化")
    parser.add_argument("--baseline", type=str, required=True,
                       help="基线评估结果文件")
    parser.add_argument("--post_attack", type=str, required=True,
                       help="攻击后评估结果文件")
    parser.add_argument("--center_nodes", type=str, nargs='+', required=True,
                       help="中心节点列表")
    parser.add_argument("--max_hops", type=int, default=2,
                       help="子图最大跳数")
    parser.add_argument("--metric", type=str, default="confidence",
                       choices=["confidence", "accuracy"],
                       help="可视化指标")
    parser.add_argument("--output", type=str, default=None,
                       help="输出文件路径")
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = SubgraphVisualizer(args.baseline, args.post_attack)
    
    # 生成子图对比
    visualizer.visualize_subgraph_comparison(
        center_nodes=args.center_nodes,
        max_hops=args.max_hops,
        metric=args.metric,
        output_file=args.output
    )

if __name__ == "__main__":
    main()