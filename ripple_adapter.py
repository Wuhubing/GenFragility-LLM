#!/usr/bin/env python3
"""
适配器：连接异步图谱构建器和ripple实验代码
兼容generate_ripple_experiments.py的数据结构和接口
"""

import json
import pickle
import os
import asyncio
import networkx as nx
from typing import Dict, List, Optional, Any
from datetime import datetime
import gzip

from infinite_graph_builder_async import create_async_infinite_builder


class RippleExperimentAdapter:
    """
    适配器类，将异步图谱构建器的输出转换为
    generate_ripple_experiments.py期望的格式
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.async_builder = create_async_infinite_builder(config)
        
    async def build_graph_for_ripple_experiments(self, 
                                               initial_seeds: List[str],
                                               target_size: int = 1000) -> str:
        """
        构建图谱并保存为ripple实验兼容格式
        
        Returns:
            str: 保存的图谱文件路径
        """
        print(f"🔗 开始为ripple实验构建图谱...")
        
        # 使用异步构建器构建图谱
        graph = await self.async_builder.build_infinite_graph(
            initial_seeds=initial_seeds,
            target_size=target_size
        )
        
        # 转换为ripple实验兼容格式
        converted_graph = self._convert_to_ripple_format(graph)
        
        # 保存图谱
        output_path = self._save_graph_for_ripple(converted_graph)
        
        print(f"✅ 图谱已保存为ripple兼容格式: {output_path}")
        return output_path
    
    def _convert_to_ripple_format(self, graph: nx.MultiDiGraph) -> nx.DiGraph:
        """
        将异步构建器的多重有向图转换为ripple实验期望的格式
        """
        print(f"🔄 转换图谱格式...")
        
        # 创建新的有向图
        ripple_graph = nx.DiGraph()
        
        # 添加所有节点
        for node in graph.nodes():
            ripple_graph.add_node(node)
        
        # 处理边：将多重边合并，保留最高置信度的边
        edge_data_map = {}
        
        for u, v, data in graph.edges(data=True):
            edge_key = (u, v)
            
            if edge_key not in edge_data_map:
                edge_data_map[edge_key] = data.copy()
            else:
                # 如果已存在边，保留置信度更高的
                existing_confidence = edge_data_map[edge_key].get('confidence', 0)
                new_confidence = data.get('confidence', 0)
                
                if new_confidence > existing_confidence:
                    edge_data_map[edge_key] = data.copy()
        
        # 添加合并后的边到新图
        for (u, v), data in edge_data_map.items():
            # 确保边数据包含ripple实验需要的字段
            edge_attrs = {
                'relation': data.get('relation', 'unknown'),
                'question': data.get('question', f"What is the {data.get('relation', 'relation')} of {u}?"),
                'surface': data.get('surface', f"{u} {data.get('relation', 'relates to')} {v}"),
                'evidence': data.get('evidence', ''),
                'group': data.get('group', 'Unknown'),
                'confidence': data.get('confidence', 0.8),
                'is_inverse': data.get('is_inverse', False),
                'head': u,  # 保持head/tail命名
                'tail': v
            }
            
            ripple_graph.add_edge(u, v, **edge_attrs)
        
        print(f"✅ 转换完成: {ripple_graph.number_of_nodes()} 节点, {ripple_graph.number_of_edges()} 边")
        return ripple_graph
    
    def _save_graph_for_ripple(self, graph: nx.DiGraph) -> str:
        """
        保存图谱为ripple实验兼容的格式（pickle）
        """
        # 创建输出目录
        output_dir = self.config.get('ripple_output_dir', '/root/GenFragility-LLM/results/async_graph_for_ripple')
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"async_graph_{graph.number_of_nodes()}nodes_{timestamp}.pkl"
        output_path = os.path.join(output_dir, filename)
        
        # 保存为pickle格式（ripple实验的标准格式）
        with open(output_path, 'wb') as f:
            pickle.dump(graph, f)
        
        # 同时保存压缩版本
        compressed_path = output_path + ".gz"
        with gzip.open(compressed_path, 'wb') as f:
            pickle.dump(graph, f)
        
        # 保存元数据信息
        metadata = {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'created_at': datetime.now().isoformat(),
            'source': 'async_infinite_graph_builder',
            'config': self.config,
            'sample_edges': [
                {
                    'head': u,
                    'relation': data['relation'],
                    'tail': v,
                    'question': data.get('question', ''),
                    'confidence': data.get('confidence', 0)
                }
                for u, v, data in list(graph.edges(data=True))[:10]
            ]
        }
        
        metadata_path = output_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"💾 图谱已保存:")
        print(f"  📦 原始文件: {output_path}")
        print(f"  🗜️ 压缩文件: {compressed_path}")
        print(f"  📋 元数据: {metadata_path}")
        
        return output_path
    
    def create_sample_ripple_experiment(self, graph_path: str, 
                                      experiment_name: str = "async_sample") -> str:
        """
        使用构建的图谱创建示例ripple实验
        """
        # 加载图谱
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        
        # 选择一个随机的边作为目标
        edges = list(graph.edges(data=True))
        if not edges:
            raise ValueError("图谱中没有边")
        
        import random
        target_edge = random.choice(edges)
        u, v, data = target_edge
        
        # 创建实验数据结构
        experiment = {
            'experiment_id': f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'target': {
                'head': u,
                'relation': data['relation'],
                'tail': v,
                'question': data.get('question', ''),
                'triplet': [u, data['relation'], v]
            },
            'graph_info': {
                'total_nodes': graph.number_of_nodes(),
                'total_edges': graph.number_of_edges(),
                'source_graph': graph_path
            },
            'created_at': datetime.now().isoformat()
        }
        
        # 保存实验文件
        output_dir = self.config.get('ripple_output_dir', '/root/GenFragility-LLM/results/async_graph_for_ripple')
        experiment_path = os.path.join(output_dir, f"{experiment['experiment_id']}_experiment.json")
        
        with open(experiment_path, 'w', encoding='utf-8') as f:
            json.dump(experiment, f, ensure_ascii=False, indent=2)
        
        print(f"📝 示例实验已创建: {experiment_path}")
        print(f"🎯 目标三元组: {u} --[{data['relation']}]--> {v}")
        
        return experiment_path


async def create_graph_for_ripple_experiments():
    """
    主函数：为ripple实验创建异步构建的图谱
    """
    print("🎭 为Ripple实验构建异步图谱")
    print("=" * 60)
    
    # 高性能配置
    config = {
        'api_key_path': '/root/GenFragility-LLM/keys/openai.txt',
        'cache_dir': '/root/GenFragility-LLM/cache/llm_responses',
        'max_concurrent': 30,      # 30并发调用
        'batch_size': 15,          # 每批次15个实体
        'budget_per_entity': 20,   # 每个实体20个三元组
        'seed_target': 100,        # 种子阶段：100节点
        'breadth_target': 300,     # 广度优先：300节点
        'depth_target': 600,       # 深度优先：600节点
        'final_target': 1000,      # 最终目标：1000节点
        'checkpoint_interval': 50,
        'checkpoint_dir': '/root/GenFragility-LLM/checkpoints/ripple_async',
        'ripple_output_dir': '/root/GenFragility-LLM/results/async_graph_for_ripple'
    }
    
    # 丰富的种子集合
    initial_seeds = [
        # 科技与创新
        "Apple Inc.", "Microsoft", "Google", "Tesla", "OpenAI",
        "iPhone", "Windows", "Android", "ChatGPT", "Python",
        
        # 科学与学术
        "Einstein", "Marie Curie", "Newton", "Darwin", "Hawking",
        "Harvard University", "MIT", "Stanford", "Nobel Prize",
        
        # 地理与文化
        "United States", "China", "Germany", "Japan", "France",
        "New York", "Beijing", "London", "Tokyo", "Paris",
        
        # 概念与领域
        "Machine Learning", "Quantum Physics", "DNA", "Evolution",
        "Democracy", "Economics", "Medicine", "Engineering"
    ]
    
    print(f"🌱 种子实体 ({len(initial_seeds)}个):")
    for i, seed in enumerate(initial_seeds[:10], 1):
        print(f"  {i:2d}. {seed}")
    if len(initial_seeds) > 10:
        print(f"     ... 还有 {len(initial_seeds)-10} 个")
    print()
    
    # 创建适配器
    adapter = RippleExperimentAdapter(config)
    
    try:
        # 构建图谱
        graph_path = await adapter.build_graph_for_ripple_experiments(
            initial_seeds=initial_seeds,
            target_size=1000
        )
        
        # 创建示例实验
        experiment_path = adapter.create_sample_ripple_experiment(
            graph_path=graph_path,
            experiment_name="async_ripple_demo"
        )
        
        print(f"\n🎉 Ripple实验图谱构建完成！")
        print(f"📊 图谱文件: {graph_path}")
        print(f"📝 示例实验: {experiment_path}")
        print(f"\n📋 使用方法:")
        print(f"  1. 更新 generate_ripple_experiments.py 中的 GRAPH_FILE 路径:")
        print(f"     GRAPH_FILE = '{graph_path}'")
        print(f"  2. 运行 ripple 实验生成代码")
        
    except Exception as e:
        print(f"❌ 构建过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行图谱构建
    asyncio.run(create_graph_for_ripple_experiments())
