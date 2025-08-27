#!/usr/bin/env python3
"""
无限扩张知识图谱构建器
基于当前架构设计，支持构建任意大小的知识图谱
"""

import time
import json
import os
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict, deque
import networkx as nx
from datetime import datetime
import logging
from tqdm import tqdm
import sys

from graph_builder.enhanced_graph_builder import EnhancedGraphBuilder
from graph_builder.graph_builder_v0_3 import GraphBuilderV03

class InfiniteGraphBuilder:
    """
    无限扩张的知识图谱构建器
    
    核心设计原则：
    1. 多阶段扩张：种子扩张 → 广度优先 → 深度优先 → 关系强化
    2. 智能调度：基于图结构和质量的动态调度
    3. 内存管理：分批处理，避免内存爆炸
    4. 质量控制：多重验证机制保证图谱质量
    5. 可恢复性：检查点和增量构建
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.graph = nx.MultiDiGraph()
        
        # 阶段控制
        self.current_phase = "seed_expansion"  # seed_expansion, breadth_first, depth_first, relation_strengthening
        self.phase_targets = {
            "seed_expansion": config.get('seed_target', 100),
            "breadth_first": config.get('breadth_target', 1000), 
            "depth_first": config.get('depth_target', 5000),
            "relation_strengthening": config.get('final_target', 10000)
        }
        
        # 核心组件
        self.base_builder = GraphBuilderV03(
            api_key_path=config.get('api_key_path', 'keys/openai.txt'),
            cache_dir=config.get('cache_dir', 'cache/llm_responses')
        )
        
        # 实体管理
        self.entity_pools = {
            "pending": deque(),      # 待处理实体
            "processing": set(),     # 正在处理的实体
            "completed": set(),      # 已完成的实体
            "high_priority": deque(), # 高优先级实体（用于三角闭合等）
            "failed": set()          # 处理失败的实体
        }
        
        # 统计和监控
        self.stats = {
            "total_nodes": 0,
            "total_edges": 0,
            "phase_stats": defaultdict(dict),
            "llm_calls": 0,
            "failed_calls": 0,
            "start_time": time.time()
        }
        
        # 质量控制
        self.quality_thresholds = {
            "min_confidence": config.get('min_confidence', 0.6),
            "max_entities_per_batch": config.get('max_batch_size', 50),
            "min_new_entities_per_call": config.get('min_new_entities', 1),
            "max_retries": config.get('max_retries', 3)
        }
        
        # 扩张策略
        self.expansion_strategies = {
            "seed_expansion": self._seed_expansion_strategy,
            "breadth_first": self._breadth_first_strategy,
            "depth_first": self._depth_first_strategy,
            "relation_strengthening": self._relation_strengthening_strategy
        }
        
        # 检查点
        self.checkpoint_interval = config.get('checkpoint_interval', 100)
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints/infinite_graph')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.checkpoint_dir}/infinite_graph.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 进度条
        self.progress_bar = None
        self.last_update_time = time.time()
    
    def build_infinite_graph(self, initial_seeds: List[str], target_size: int = 10000) -> nx.MultiDiGraph:
        """
        构建无限扩张的知识图谱
        
        Args:
            initial_seeds: 初始种子实体列表
            target_size: 目标节点数量
            
        Returns:
            构建完成的知识图谱
        """
        self.logger.info(f"🚀 开始构建无限知识图谱，目标大小：{target_size} 节点")
        self.logger.info(f"🌱 初始种子：{initial_seeds}")
        
        # 更新最终目标
        self.phase_targets["relation_strengthening"] = target_size
        
        # 初始化种子实体
        self.entity_pools["pending"].extend(initial_seeds)
        
        # 初始化进度条
        self.progress_bar = tqdm(
            total=target_size,
            desc="🏗️  构建图谱",
            unit="节点",
            unit_scale=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            dynamic_ncols=True,
            leave=True
        )
        
        # 主循环：不断扩张直到达到目标
        while self._should_continue_expansion(target_size):
            try:
                # 执行当前阶段的扩张策略
                current_strategy = self.expansion_strategies[self.current_phase]
                success = current_strategy()
                
                if not success:
                    self.logger.warning(f"⚠️ 阶段 {self.current_phase} 扩张失败，尝试切换策略")
                    self._handle_expansion_failure()
                
                # 更新统计信息
                self._update_stats()
                
                # 更新进度条
                self._update_progress_bar()
                
                # 检查是否需要切换阶段
                self._check_phase_transition()
                
                # 定期检查点
                if self.stats["total_nodes"] % self.checkpoint_interval == 0:
                    self._save_checkpoint()
                
                # 内存管理
                if self.stats["total_nodes"] % 1000 == 0:
                    self._memory_management()
                
            except Exception as e:
                self.logger.error(f"❌ 扩张过程中出现错误：{e}")
                self._handle_critical_error(e)
        
        # 完成进度条
        if self.progress_bar:
            self.progress_bar.close()
        
        self.logger.info(f"🎉 图谱构建完成！最终规模：{self.stats['total_nodes']} 节点")
        self._save_final_checkpoint()
        
        return self.graph
    
    def _seed_expansion_strategy(self) -> bool:
        """种子扩张策略：从初始种子快速扩张"""
        if not self.entity_pools["pending"]:
            return False
        
        # 批量处理种子实体
        batch_size = min(5, len(self.entity_pools["pending"]))
        current_batch = []
        
        for _ in range(batch_size):
            if self.entity_pools["pending"]:
                entity = self.entity_pools["pending"].popleft()
                current_batch.append(entity)
                self.entity_pools["processing"].add(entity)
        
        if not current_batch:
            return False
        
        self.logger.info(f"🌱 种子扩张阶段：处理批次 {current_batch}")
        
        # 为每个种子生成triplets
        for entity in current_batch:
            try:
                success = self._expand_single_entity(entity, budget=15)  # 种子阶段给更多budget
                if success:
                    self.entity_pools["completed"].add(entity)
                else:
                    self.entity_pools["failed"].add(entity)
                self.entity_pools["processing"].discard(entity)
            except Exception as e:
                self.logger.error(f"❌ 处理种子实体 {entity} 时出错：{e}")
                self.entity_pools["failed"].add(entity)
                self.entity_pools["processing"].discard(entity)
        
        return True
    
    def _breadth_first_strategy(self) -> bool:
        """广度优先策略：平衡扩张各个方向"""
        if not self.entity_pools["pending"]:
            # 如果没有待处理实体，从图中选择新的扩张点
            self._select_breadth_expansion_candidates()
        
        if not self.entity_pools["pending"]:
            return False
        
        # 广度优先：小批量多样化处理
        batch_size = min(10, len(self.entity_pools["pending"]))
        current_batch = []
        
        for _ in range(batch_size):
            if self.entity_pools["pending"]:
                entity = self.entity_pools["pending"].popleft()
                current_batch.append(entity)
                self.entity_pools["processing"].add(entity)
        
        self.logger.info(f"🌊 广度优先阶段：处理批次 {current_batch}")
        
        # 并行处理（模拟）
        for entity in current_batch:
            try:
                success = self._expand_single_entity(entity, budget=8)  # 中等budget
                if success:
                    self.entity_pools["completed"].add(entity)
                else:
                    self.entity_pools["failed"].add(entity)
                self.entity_pools["processing"].discard(entity)
            except Exception as e:
                self.logger.error(f"❌ 广度优先处理 {entity} 时出错：{e}")
                self.entity_pools["failed"].add(entity)
                self.entity_pools["processing"].discard(entity)
        
        return True
    
    def _depth_first_strategy(self) -> bool:
        """深度优先策略：深入挖掘特定领域"""
        # 选择高连接度的实体进行深度挖掘
        high_degree_entities = self._select_high_degree_entities()
        
        if not high_degree_entities:
            # 如果没有高连接度实体，回到常规扩张
            return self._breadth_first_strategy()
        
        # 深度挖掘
        target_entity = high_degree_entities[0]
        self.logger.info(f"🏊‍♂️ 深度优先阶段：深入挖掘 {target_entity}")
        
        # 从多个角度扩张同一个实体
        expansion_angles = ["downstream", "upstream", "parallel"]
        
        for angle in expansion_angles:
            try:
                success = self._expand_single_entity(
                    target_entity, 
                    budget=12, 
                    focus=angle
                )
                if not success:
                    break
            except Exception as e:
                self.logger.error(f"❌ 深度挖掘 {target_entity} ({angle}) 时出错：{e}")
        
        # 标记为已完成
        self.entity_pools["completed"].add(target_entity)
        
        return True
    
    def _relation_strengthening_strategy(self) -> bool:
        """关系强化策略：补充缺失的连接，提高图的密度"""
        # 寻找三角闭合机会
        closure_opportunities = self._find_triadic_closure_opportunities()
        
        if closure_opportunities:
            self.logger.info(f"🔺 关系强化：发现 {len(closure_opportunities)} 个三角闭合机会")
            
            for entity_a, entity_b, common_neighbor in closure_opportunities[:5]:  # 限制每次处理数量
                try:
                    # 尝试在entity_a和entity_b之间建立连接
                    success = self._attempt_entity_connection(entity_a, entity_b, common_neighbor)
                    if success:
                        self.logger.info(f"✅ 成功建立连接：{entity_a} ↔ {entity_b}")
                except Exception as e:
                    self.logger.error(f"❌ 尝试连接 {entity_a} 和 {entity_b} 时出错：{e}")
            
            return True
        
        # 如果没有闭合机会，尝试扩张低连接度实体
        low_degree_entities = self._select_low_degree_entities()
        if low_degree_entities:
            entity = low_degree_entities[0]
            return self._expand_single_entity(entity, budget=6)
        
        return False
    
    def _expand_single_entity(self, entity: str, budget: int = 10, focus: str = "general") -> bool:
        """
        扩张单个实体
        
        Args:
            entity: 要扩张的实体
            budget: 生成triplet的数量预算
            focus: 扩张焦点 ("general", "downstream", "upstream", "parallel")
        """
        try:
            self.logger.info(f"🔍 扩张实体：{entity} (budget={budget}, focus={focus})")
            
            # 调用基础构建器生成triplets
            triplets = self.base_builder.generate_from_seeds([entity], budget=budget, language="en")
            
            if not triplets:
                self.logger.warning(f"⚠️ 实体 {entity} 没有生成任何triplets")
                return False
            
            # 处理生成的triplets
            new_entities = set()
            added_triplets = 0
            
            for triplet in triplets:
                # 添加到图中
                self._add_triplet_to_graph(triplet)
                added_triplets += 1
                
                # 收集新实体
                if triplet['head'] not in self.entity_pools["completed"] and triplet['head'] != entity:
                    new_entities.add(triplet['head'])
                if triplet['tail'] not in self.entity_pools["completed"] and triplet['tail'] != entity:
                    new_entities.add(triplet['tail'])
            
            # 将新实体添加到待处理队列
            for new_entity in new_entities:
                if (new_entity not in self.entity_pools["pending"] and 
                    new_entity not in self.entity_pools["processing"] and
                    new_entity not in self.entity_pools["completed"]):
                    self.entity_pools["pending"].append(new_entity)
            
            self.stats["llm_calls"] += 1
            self.logger.info(f"✅ 成功扩张 {entity}：添加 {added_triplets} 个triplets，发现 {len(new_entities)} 个新实体")
            
            # 在进度条中显示扩张信息
            if self.progress_bar and added_triplets > 0:
                self.progress_bar.write(f"   ✅ {entity}: +{added_triplets}条边, +{len(new_entities)}个新实体")
            
            return added_triplets > 0
            
        except Exception as e:
            self.logger.error(f"❌ 扩张实体 {entity} 失败：{e}")
            self.stats["failed_calls"] += 1
            return False
    
    def _add_triplet_to_graph(self, triplet: Dict[str, Any]):
        """将triplet添加到图中"""
        head = triplet['head']
        tail = triplet['tail']
        relation = triplet['relation_id']
        
        # 添加节点
        if not self.graph.has_node(head):
            self.graph.add_node(head, type="entity")
        if not self.graph.has_node(tail):
            self.graph.add_node(tail, type="entity")
        
        # 添加边
        self.graph.add_edge(
            head, tail,
            relation=relation,
            confidence=triplet['confidence'],
            surface=triplet['surface'],
            question=triplet['question'],
            qa_eligible=triplet['qa_eligible']
        )
    
    def _should_continue_expansion(self, target_size: int) -> bool:
        """判断是否应该继续扩张"""
        current_nodes = self.graph.number_of_nodes()
        
        # 检查目标大小
        if current_nodes >= target_size:
            return False
        
        # 检查是否还有可扩张的实体
        total_pending = (len(self.entity_pools["pending"]) + 
                        len(self.entity_pools["processing"]) +
                        len(self.entity_pools["high_priority"]))
        
        if total_pending == 0 and current_nodes < target_size:
            # 如果没有待处理实体但还没达到目标，尝试从现有图中找新的扩张点
            self._bootstrap_new_expansion_points()
            total_pending = len(self.entity_pools["pending"])
        
        return total_pending > 0
    
    def _check_phase_transition(self):
        """检查是否需要切换阶段"""
        current_nodes = self.graph.number_of_nodes()
        target = self.phase_targets[self.current_phase]
        
        if current_nodes >= target:
            old_phase = self.current_phase
            
            if self.current_phase == "seed_expansion":
                self.current_phase = "breadth_first"
            elif self.current_phase == "breadth_first":
                self.current_phase = "depth_first"
            elif self.current_phase == "depth_first":
                self.current_phase = "relation_strengthening"
            # relation_strengthening 阶段保持不变，直到达到最终目标
            
            if old_phase != self.current_phase:
                self.logger.info(f"🔄 阶段切换：{old_phase} → {self.current_phase}")
                
                # 在进度条中显示阶段切换
                if self.progress_bar:
                    phase_names = {
                        "seed_expansion": "🌱种子扩张",
                        "breadth_first": "🌊广度优先", 
                        "depth_first": "🏊‍♂️深度优先",
                        "relation_strengthening": "🔺关系强化"
                    }
                    new_phase_name = phase_names.get(self.current_phase, self.current_phase)
                    self.progress_bar.write(f"🔄 阶段切换 → {new_phase_name}")
                    
                    # 更新进度条描述
                    self.progress_bar.set_description(f"🏗️  {new_phase_name}")
    
    def _update_stats(self):
        """更新统计信息"""
        self.stats["total_nodes"] = self.graph.number_of_nodes()
        self.stats["total_edges"] = self.graph.number_of_edges()
        self.stats["current_phase"] = self.current_phase
        
        # 更新阶段统计
        phase_stats = self.stats["phase_stats"][self.current_phase]
        phase_stats["nodes"] = self.stats["total_nodes"]
        phase_stats["edges"] = self.stats["total_edges"]
        phase_stats["pending_entities"] = len(self.entity_pools["pending"])
        phase_stats["completed_entities"] = len(self.entity_pools["completed"])
    
    def _update_progress_bar(self):
        """更新进度条"""
        if not self.progress_bar:
            return
        
        current_time = time.time()
        
        # 限制更新频率，避免进度条闪烁
        if current_time - self.last_update_time < 0.1:  # 最多100ms更新一次
            return
        
        self.last_update_time = current_time
        
        # 更新进度
        current_nodes = self.stats["total_nodes"]
        self.progress_bar.n = current_nodes
        
        # 创建详细的后缀信息
        phase_emoji = {
            "seed_expansion": "🌱",
            "breadth_first": "🌊", 
            "depth_first": "🏊‍♂️",
            "relation_strengthening": "🔺"
        }
        
        phase_name = {
            "seed_expansion": "种子扩张",
            "breadth_first": "广度优先",
            "depth_first": "深度优先", 
            "relation_strengthening": "关系强化"
        }
        
        emoji = phase_emoji.get(self.current_phase, "🔧")
        name = phase_name.get(self.current_phase, self.current_phase)
        
        postfix_info = {
            "阶段": f"{emoji}{name}",
            "边": f"{self.stats['total_edges']}",
            "待处理": f"{len(self.entity_pools['pending'])}",
            "LLM调用": f"{self.stats['llm_calls']}",
            "失败": f"{self.stats['failed_calls']}"
        }
        
        # 计算速度
        runtime = current_time - self.stats["start_time"]
        if runtime > 0:
            nodes_per_min = (current_nodes / runtime) * 60
            if nodes_per_min >= 1:
                postfix_info["速度"] = f"{nodes_per_min:.1f}节点/分"
            else:
                postfix_info["速度"] = f"{nodes_per_min*60:.1f}节点/时"
        
        # 计算当前阶段进度
        phase_target = self.phase_targets.get(self.current_phase, current_nodes)
        if phase_target > 0:
            phase_progress = min(100, (current_nodes / phase_target) * 100)
            postfix_info["阶段进度"] = f"{phase_progress:.1f}%"
        
        self.progress_bar.set_postfix(postfix_info, refresh=False)
        self.progress_bar.refresh()
    
    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint_data = {
            "graph": nx.node_link_data(self.graph),
            "entity_pools": {
                "pending": list(self.entity_pools["pending"]),
                "processing": list(self.entity_pools["processing"]),
                "completed": list(self.entity_pools["completed"]),
                "high_priority": list(self.entity_pools["high_priority"]),
                "failed": list(self.entity_pools["failed"])
            },
            "stats": dict(self.stats),
            "current_phase": self.current_phase,
            "timestamp": datetime.now().isoformat()
        }
        
        checkpoint_file = f"{self.checkpoint_dir}/checkpoint_{self.stats['total_nodes']}.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"💾 保存检查点：{checkpoint_file}")
        
        # 在进度条中显示检查点保存
        if self.progress_bar:
            self.progress_bar.write(f"💾 检查点已保存 ({self.stats['total_nodes']} 节点)")
    
    def _find_triadic_closure_opportunities(self) -> List[Tuple[str, str, str]]:
        """寻找三角闭合机会"""
        opportunities = []
        nodes = list(self.graph.nodes())
        
        for node_a in nodes[:100]:  # 限制搜索范围避免计算爆炸
            neighbors_a = set(self.graph.neighbors(node_a)) | set(self.graph.predecessors(node_a))
            
            for node_b in neighbors_a:
                if node_b == node_a:
                    continue
                
                neighbors_b = set(self.graph.neighbors(node_b)) | set(self.graph.predecessors(node_b))
                common_neighbors = neighbors_a & neighbors_b - {node_a, node_b}
                
                for common_neighbor in common_neighbors:
                    # 检查A-B是否缺少直接连接
                    if not (self.graph.has_edge(node_a, node_b) or self.graph.has_edge(node_b, node_a)):
                        opportunities.append((node_a, node_b, common_neighbor))
        
        return opportunities[:20]  # 返回前20个机会
    
    def _select_high_degree_entities(self) -> List[str]:
        """选择高连接度的实体"""
        if self.graph.number_of_nodes() == 0:
            return []
        
        # 计算度中心性
        degree_centrality = nx.degree_centrality(self.graph)
        
        # 按度排序，选择前10个
        sorted_entities = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        high_degree = [entity for entity, _ in sorted_entities[:10] 
                      if entity not in self.entity_pools["completed"]]
        
        return high_degree
    
    def _select_low_degree_entities(self) -> List[str]:
        """选择低连接度的实体（用于补强）"""
        if self.graph.number_of_nodes() == 0:
            return []
        
        degree_centrality = nx.degree_centrality(self.graph)
        
        # 选择度较低的实体
        sorted_entities = sorted(degree_centrality.items(), key=lambda x: x[1])
        low_degree = [entity for entity, degree in sorted_entities[:20] 
                     if degree < 0.1 and entity not in self.entity_pools["completed"]]
        
        return low_degree
    
    def get_expansion_report(self) -> Dict[str, Any]:
        """获取扩张报告"""
        runtime = time.time() - self.stats["start_time"]
        
        report = {
            "总体统计": {
                "节点数": self.stats["total_nodes"],
                "边数": self.stats["total_edges"],
                "运行时间": f"{runtime/3600:.2f} 小时",
                "当前阶段": self.current_phase,
                "LLM调用次数": self.stats["llm_calls"],
                "失败调用次数": self.stats["failed_calls"]
            },
            "实体状态": {
                "待处理": len(self.entity_pools["pending"]),
                "处理中": len(self.entity_pools["processing"]), 
                "已完成": len(self.entity_pools["completed"]),
                "高优先级": len(self.entity_pools["high_priority"]),
                "失败": len(self.entity_pools["failed"])
            },
            "阶段统计": dict(self.stats["phase_stats"]),
            "图谱质量": self._calculate_graph_quality()
        }
        
        return report
    
    def _calculate_graph_quality(self) -> Dict[str, float]:
        """计算图谱质量指标"""
        if self.graph.number_of_nodes() == 0:
            return {}
        
        # 基本图谱指标
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        
        quality_metrics = {
            "密度": nx.density(self.graph),
            "平均度": 2 * num_edges / num_nodes if num_nodes > 0 else 0,
            "连通组件数": nx.number_weakly_connected_components(self.graph),
        }
        
        # 计算平均置信度
        confidences = []
        for _, _, data in self.graph.edges(data=True):
            if 'confidence' in data:
                confidences.append(data['confidence'])
        
        if confidences:
            quality_metrics["平均置信度"] = sum(confidences) / len(confidences)
            quality_metrics["高置信度边比例"] = sum(1 for c in confidences if c >= 0.8) / len(confidences)
        
        return quality_metrics
    
    # 辅助方法
    def _select_breadth_expansion_candidates(self):
        """为广度优先选择扩张候选"""
        # 从现有节点中选择还未完全扩张的实体
        candidates = []
        for node in self.graph.nodes():
            if (node not in self.entity_pools["completed"] and 
                node not in self.entity_pools["processing"]):
                candidates.append(node)
        
        # 随机选择一些候选实体，保证多样性
        import random
        selected = random.sample(candidates, min(20, len(candidates)))
        self.entity_pools["pending"].extend(selected)
    
    def _bootstrap_new_expansion_points(self):
        """当没有待处理实体时，从现有图中启动新的扩张点"""
        # 选择一些中等连接度的实体作为新的扩张点
        if self.graph.number_of_nodes() == 0:
            return
        
        degree_centrality = nx.degree_centrality(self.graph)
        # 选择中等度的实体（既不是太热门也不是太冷门）
        medium_degree = []
        for entity, degree in degree_centrality.items():
            if 0.1 < degree < 0.8 and entity not in self.entity_pools["completed"]:
                medium_degree.append(entity)
        
        # 添加到待处理队列
        import random
        selected = random.sample(medium_degree, min(10, len(medium_degree)))
        self.entity_pools["pending"].extend(selected)
        self.logger.info(f"🔄 启动新扩张点：{selected}")
    
    def _attempt_entity_connection(self, entity_a: str, entity_b: str, common_neighbor: str) -> bool:
        """尝试在两个实体之间建立连接"""
        # 这里可以通过LLM尝试发现entity_a和entity_b之间的关系
        # 当前简化实现
        try:
            # 尝试从entity_a扩张，看能否连接到entity_b
            triplets = self.base_builder.generate_from_seeds([entity_a], budget=5, language="en")
            
            for triplet in triplets:
                if triplet['tail'] == entity_b or triplet['head'] == entity_b:
                    self._add_triplet_to_graph(triplet)
                    return True
            
            return False
        except:
            return False
    
    def _handle_expansion_failure(self):
        """处理扩张失败"""
        self.logger.warning("⚠️ 当前扩张策略失败，尝试恢复")
        
        # 尝试从失败的实体中恢复一些
        if self.entity_pools["failed"]:
            retry_entities = list(self.entity_pools["failed"])[:5]
            self.entity_pools["failed"] -= set(retry_entities)
            self.entity_pools["pending"].extend(retry_entities)
            self.logger.info(f"🔄 重试失败实体：{retry_entities}")
    
    def _handle_critical_error(self, error: Exception):
        """处理关键错误"""
        self.logger.error(f"💥 关键错误：{error}")
        self._save_checkpoint()  # 保存当前状态
        
        # 可以在这里实现更复杂的错误恢复策略
        # 比如降低batch size、切换策略等
    
    def _memory_management(self):
        """内存管理：清理不必要的缓存"""
        # 清理处理过多的完成实体（只保留最近的）
        if len(self.entity_pools["completed"]) > 1000:
            old_completed = list(self.entity_pools["completed"])
            self.entity_pools["completed"] = set(old_completed[-800:])  # 保留最近800个
            self.logger.info("🧹 清理内存：移除旧的完成实体记录")
    
    def _save_final_checkpoint(self):
        """保存最终检查点"""
        self._save_checkpoint()
        
        # 导出最终结果
        final_output = f"{self.checkpoint_dir}/final_graph.gexf"
        nx.write_gexf(self.graph, final_output)
        self.logger.info(f"💾 最终图谱已保存：{final_output}")
        
        # 生成报告
        report = self.get_expansion_report()
        report_file = f"{self.checkpoint_dir}/final_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self.logger.info(f"📊 最终报告已保存：{report_file}")


def create_infinite_builder(config: Dict[str, Any]) -> InfiniteGraphBuilder:
    """创建无限图构建器的工厂函数"""
    default_config = {
        'api_key_path': 'keys/openai.txt',
        'cache_dir': 'cache/llm_responses',
        'seed_target': 100,
        'breadth_target': 1000,
        'depth_target': 5000,
        'final_target': 10000,
        'min_confidence': 0.6,
        'max_batch_size': 50,
        'checkpoint_interval': 100,
        'checkpoint_dir': 'checkpoints/infinite_graph'
    }
    
    # 合并配置
    final_config = {**default_config, **config}
    
    return InfiniteGraphBuilder(final_config)


if __name__ == "__main__":
    # 示例用法
    config = {
        'final_target': 5000,  # 目标5000个节点
        'checkpoint_interval': 50,
    }
    
    builder = create_infinite_builder(config)
    
    initial_seeds = [
        "Beijing", "Apple Inc.", "Einstein", "Python", "China", 
        "United States", "Olympics", "Shakespeare", "Tesla", "Google"
    ]
    
    try:
        graph = builder.build_infinite_graph(initial_seeds, target_size=5000)
        print(f"🎉 成功构建图谱：{graph.number_of_nodes()} 节点，{graph.number_of_edges()} 边")
        
        # 打印最终报告
        report = builder.get_expansion_report()
        print("📊 最终报告：")
        for section, data in report.items():
            print(f"\n{section}:")
            for key, value in data.items():
                print(f"  {key}: {value}")
                
    except KeyboardInterrupt:
        print("⏹️ 用户中断，保存当前进度...")
        builder._save_checkpoint()
        report = builder.get_expansion_report()
        print(f"📊 当前进度：{report['总体统计']['节点数']} 节点")
