#!/usr/bin/env python3
"""
异步高并发无限扩张知识图谱构建器
支持高并发LLM调用，大幅提升生成速度
"""

import asyncio
import aiohttp
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
import hashlib
import pickle
import gzip
from concurrent.futures import ThreadPoolExecutor

from graph_builder.enhanced_graph_builder import EnhancedGraphBuilder
from graph_builder.graph_builder_v0_3 import GraphBuilderV03
from graph_builder.relations_ontology import RelationOntology
from graph_builder.prompts import SYS_PROMPT_GRAPH_BUILDER_v0_3, create_user_prompt_v0_3


class AsyncLLMInterface:
    """异步LLM接口，支持高并发调用"""
    
    def __init__(self, api_key_path: str, cache_dir: str = None, max_concurrent: int = 10):
        self.api_key_path = api_key_path
        self.cache_dir = cache_dir or "/root/GenFragility-LLM/cache/llm_responses"
        self.max_concurrent = max_concurrent
        self.api_key = self._load_api_key()
        self.cache = self._load_cache()
        
        # 并发控制
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
        
        # 统计
        self.stats = {
            "total_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "failures": 0,
            "concurrent_calls": 0
        }
    
    def _load_api_key(self) -> str:
        """加载API密钥"""
        try:
            with open(self.api_key_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            raise RuntimeError(f"Failed to load API key from {self.api_key_path}: {e}")
    
    def _load_cache(self) -> Dict:
        """加载缓存"""
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, "async_responses.json")
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                print(f"📦 加载了 {len(cache)} 个缓存响应")
                return cache
        except Exception as e:
            print(f"⚠️ 加载缓存失败: {e}")
        
        return {}
    
    def _save_cache(self):
        """保存缓存"""
        cache_file = os.path.join(self.cache_dir, "async_responses.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 保存缓存失败: {e}")
    
    def _get_cache_key(self, prompt: str, model: str, temperature: float) -> str:
        """生成缓存键"""
        content = f"{prompt}|{model}|{temperature}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        connector = aiohttp.TCPConnector(limit=self.max_concurrent * 2)
        timeout = aiohttp.ClientTimeout(total=300)  # 5分钟超时
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
        self._save_cache()
    
    async def call_llm_async(self, prompt: str, system_prompt: str, 
                           model: str = "gpt-4o-mini", temperature: float = 0.3, 
                           max_tokens: int = 8000) -> Optional[str]:
        """异步调用LLM"""
        # 检查缓存
        cache_key = self._get_cache_key(prompt, model, temperature)
        if cache_key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[cache_key]
        
        self.stats["cache_misses"] += 1
        self.stats["total_calls"] += 1
        
        async with self.semaphore:  # 并发控制
            self.stats["concurrent_calls"] += 1
            try:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                async with self.session.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"]
                        
                        # 缓存响应
                        self.cache[cache_key] = content
                        
                        # 定期保存缓存
                        if len(self.cache) % 20 == 0:
                            self._save_cache()
                        
                        return content
                    else:
                        error_text = await response.text()
                        print(f"❌ API调用失败 {response.status}: {error_text}")
                        self.stats["failures"] += 1
                        return None
                        
            except Exception as e:
                print(f"❌ 异步LLM调用异常: {e}")
                self.stats["failures"] += 1
                return None
            finally:
                self.stats["concurrent_calls"] -= 1
    
    async def generate_triplets_batch(self, seed_batches: List[List[str]], 
                                    budget_per_batch: int = 15,
                                    ontology: RelationOntology = None) -> List[List[Dict]]:
        """并发生成多个批次的三元组"""
        ontology = ontology or RelationOntology()
        
        # 创建并发任务
        tasks = []
        for seeds in seed_batches:
            if seeds:  # 跳过空批次
                task = self._generate_single_batch(seeds, budget_per_batch, ontology)
                tasks.append(task)
        
        if not tasks:
            return []
        
        # 并发执行所有任务
        print(f"🚀 启动 {len(tasks)} 个并发LLM调用...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ 批次 {i} 失败: {result}")
                valid_results.append([])
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def _generate_single_batch(self, seeds: List[str], budget: int, 
                                   ontology: RelationOntology) -> List[Dict]:
        """生成单个批次的三元组"""
        # 为具体实体优化预算：减少每个实体的三元组数量，提高质量
        budget = min(budget, 12 * len(seeds))  # 每个种子最多12个三元组
        
        # 创建用户提示
        user_prompt = create_user_prompt_v0_3(
            seeds=seeds,
            ontology=ontology,
            budget=budget,
            language="en"
        )
        
        # 异步调用LLM
        content = await self.call_llm_async(
            prompt=user_prompt,
            system_prompt=SYS_PROMPT_GRAPH_BUILDER_v0_3,
            temperature=0.3,
            max_tokens=8000
        )
        
        if not content:
            return []
        
        # 解析响应
        return self._parse_jsonl_response(content)
    
    def _parse_jsonl_response(self, content: str) -> List[Dict]:
        """解析JSONL响应"""
        # 首先去除可能的markdown代码块包装
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        triplets = []
        
        # 尝试JSONL解析（每行一个JSON对象）
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    if self._validate_triplet(obj):
                        triplets.append(obj)
                except json.JSONDecodeError:
                    continue
        
        # 如果JSONL失败，尝试解析多行格式化的JSON对象
        if not triplets:
            # 使用更复杂的正则表达式来匹配多行JSON对象
            import re
            
            # 找到所有的JSON对象（包括嵌套的）
            json_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
            json_objects = re.findall(json_pattern, content, re.DOTALL)
            
            for obj_str in json_objects:
                try:
                    obj = json.loads(obj_str)
                    if self._validate_triplet(obj):
                        triplets.append(obj)
                except json.JSONDecodeError:
                    continue
        
        # 如果还是失败，尝试手动分割JSON对象
        if not triplets:
            try:
                # 查找所有 }{ 的位置，这通常是对象之间的分隔符
                objects = []
                brace_count = 0
                current_obj = ""
                
                for char in content:
                    current_obj += char
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # 完成一个对象
                            objects.append(current_obj.strip())
                            current_obj = ""
                
                # 解析每个对象
                for obj_str in objects:
                    if obj_str:
                        try:
                            obj = json.loads(obj_str)
                            if self._validate_triplet(obj):
                                triplets.append(obj)
                        except json.JSONDecodeError:
                            continue
                            
            except Exception:
                pass
        
        return triplets
    
    def _validate_triplet(self, obj: Dict) -> bool:
        """验证三元组格式"""
        # 检查是否有必需的字段（支持 relation 或 relation_id）
        has_head = 'head' in obj
        has_tail = 'tail' in obj
        has_relation = 'relation' in obj or 'relation_id' in obj
        
        return has_head and has_tail and has_relation


class InfiniteGraphBuilderAsync:
    """异步无限扩张图谱构建器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 核心组件
        self.graph = nx.MultiDiGraph()
        self.ontology = RelationOntology()
        
        # 异步LLM接口
        self.max_concurrent = config.get('max_concurrent', 20)  # 默认20并发
        self.llm_interface = None
        
        # 实体池管理
        self.entity_pools = {
            "pending": deque(),
            "processing": set(),
            "completed": set(),
            "failed": set()
        }
        
        # 扩张策略配置
        self.phase_targets = {
            "seed_expansion": config.get('seed_target', 100),
            "breadth_first": config.get('breadth_target', 300),  
            "depth_first": config.get('depth_target', 600),
            "relation_strengthening": config.get('final_target', 1000)
        }
        
        self.current_phase = "seed_expansion"
        
        # 批次配置
        self.batch_size = config.get('batch_size', 8)  # 每批次处理的实体数
        self.budget_per_entity = config.get('budget_per_entity', 15)  # 每个实体的目标三元组数
        
        # 统计信息
        self.stats = {
            "total_nodes": 0,
            "total_edges": 0,
            "total_batches": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "start_time": time.time(),
            "phase_stats": defaultdict(lambda: defaultdict(int))
        }
        
        # 检查点配置
        self.checkpoint_interval = config.get('checkpoint_interval', 50)
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints/async_infinite')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 进度条
        self.progress_bar = None
        
        # 日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.checkpoint_dir}/async_infinite.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def build_infinite_graph(self, initial_seeds: List[str], target_size: int = 1000) -> nx.MultiDiGraph:
        """异步构建无限扩张的知识图谱"""
        self.logger.info(f"🚀 开始异步构建无限知识图谱，目标大小：{target_size} 节点")
        self.logger.info(f"🌱 初始种子：{initial_seeds}")
        self.logger.info(f"⚡ 最大并发数：{self.max_concurrent}")
        
        # 更新最终目标
        self.phase_targets["relation_strengthening"] = target_size
        
        # 初始化种子实体
        self.entity_pools["pending"].extend(initial_seeds)
        
        # 初始化进度条
        self.progress_bar = tqdm(
            total=target_size,
            desc="🏗️  异步构建图谱",
            unit="节点",
            unit_scale=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            dynamic_ncols=True,
            leave=True
        )
        
        # 创建异步LLM接口
        async with AsyncLLMInterface(
            api_key_path=self.config['api_key_path'],
            cache_dir=self.config.get('cache_dir'),
            max_concurrent=self.max_concurrent
        ) as llm_interface:
            self.llm_interface = llm_interface
            
            # 主循环：异步扩张
            while self._should_continue_expansion(target_size):
                try:
                    # 批量异步扩张
                    success = await self._async_expansion_batch()
                    
                    if not success:
                        self.logger.warning(f"⚠️ 批次扩张失败，尝试调整策略")
                        await self._handle_expansion_failure()
                    
                    # 更新统计和进度
                    self._update_stats()
                    self._update_progress_bar()
                    
                    # 检查阶段切换
                    self._check_phase_transition()
                    
                    # 定期检查点
                    if self.stats["total_nodes"] % self.checkpoint_interval == 0:
                        self._save_checkpoint()
                    
                except Exception as e:
                    self.logger.error(f"❌ 异步扩张过程中出现错误：{e}")
                    break
        
        # 完成进度条
        if self.progress_bar:
            self.progress_bar.close()
        
        self.logger.info(f"🎉 异步图谱构建完成！最终规模：{self.stats['total_nodes']} 节点")
        self._save_final_checkpoint()
        
        return self.graph
    
    async def _async_expansion_batch(self) -> bool:
        """执行异步批量扩张"""
        if not self.entity_pools["pending"]:
            return False
        
        # 准备批次
        batch_entities = []
        for _ in range(self.batch_size):
            if self.entity_pools["pending"]:
                entity = self.entity_pools["pending"].popleft()
                batch_entities.append(entity)
                self.entity_pools["processing"].add(entity)
        
        if not batch_entities:
            return False
        
        self.logger.info(f"🔄 异步批次扩张：{len(batch_entities)} 个实体")
        
        # 为每个实体创建种子列表（单个实体一批）
        seed_batches = [[entity] for entity in batch_entities]
        
        # 异步并发生成
        start_time = time.time()
        batch_results = await self.llm_interface.generate_triplets_batch(
            seed_batches=seed_batches,
            budget_per_batch=self.budget_per_entity,
            ontology=self.ontology
        )
        end_time = time.time()
        
        # 处理结果
        total_added = 0
        successful_entities = 0
        
        for i, (entity, triplets) in enumerate(zip(batch_entities, batch_results)):
            if triplets:
                added = self._add_triplets_to_graph(triplets)
                total_added += added
                successful_entities += 1
                self.logger.info(f"✅ {entity}: +{added} 三元组")
                
                # 在进度条显示成功扩张
                if self.progress_bar:
                    new_entities = set(t['head'] for t in triplets) | set(t['tail'] for t in triplets)
                    new_entities.discard(entity)
                    self.progress_bar.write(f"   ✅ {entity}: +{added}边, +{len(new_entities)}新实体")
            else:
                self.logger.warning(f"⚠️ {entity}: 无三元组生成")
                self.entity_pools["failed"].add(entity)
            
            # 从处理中移除
            self.entity_pools["processing"].discard(entity)
            self.entity_pools["completed"].add(entity)
        
        # 更新统计
        self.stats["total_batches"] += 1
        if successful_entities > 0:
            self.stats["successful_batches"] += 1
        else:
            self.stats["failed_batches"] += 1
        
        batch_duration = end_time - start_time
        entities_per_sec = len(batch_entities) / batch_duration if batch_duration > 0 else 0
        
        self.logger.info(f"📊 批次完成：{successful_entities}/{len(batch_entities)} 成功，"
                        f"{total_added} 三元组，{batch_duration:.1f}s，{entities_per_sec:.1f} 实体/秒")
        
        return total_added > 0
    
    def _add_triplets_to_graph(self, triplets: List[Dict]) -> int:
        """将三元组添加到图中"""
        added_count = 0
        new_entities = set()
        
        for triplet in triplets:
            head = triplet['head']
            tail = triplet['tail']
            # 支持 relation 和 relation_id 两种字段名
            relation = triplet.get('relation') or triplet.get('relation_id')
            
            # 添加节点
            if head not in self.graph:
                self.graph.add_node(head)
            if tail not in self.graph:
                self.graph.add_node(tail)
            
            # 收集新实体
            if head not in self.entity_pools["completed"] and head not in self.entity_pools["processing"]:
                new_entities.add(head)
            if tail not in self.entity_pools["completed"] and tail not in self.entity_pools["processing"]:
                new_entities.add(tail)
            
            # 添加边
            edge_attrs = {
                'relation': relation,
                'confidence': triplet.get('confidence', 0.8),
                'surface': triplet.get('surface', ''),
                'evidence': triplet.get('evidence', ''),
                'question': triplet.get('question', ''),
                'group': triplet.get('group', 'Unknown'),
                'head': head,
                'tail': tail
            }
            
            self.graph.add_edge(head, tail, **edge_attrs)
            added_count += 1
        
        # 添加新实体到待处理队列
        for entity in new_entities:
            if (entity not in self.entity_pools["pending"] and 
                entity not in self.entity_pools["processing"] and
                entity not in self.entity_pools["completed"]):
                self.entity_pools["pending"].append(entity)
        
        return added_count
    
    async def _handle_expansion_failure(self):
        """处理扩张失败"""
        # 添加一些随机实体来打破僵局
        if not self.entity_pools["pending"] and self.entity_pools["completed"]:
            # 从已完成的实体中随机选择一些邻居
            completed_sample = list(self.entity_pools["completed"])[:10]
            for entity in completed_sample:
                neighbors = list(self.graph.neighbors(entity))
                for neighbor in neighbors[:3]:  # 每个实体添加最多3个邻居
                    if (neighbor not in self.entity_pools["pending"] and
                        neighbor not in self.entity_pools["processing"] and
                        neighbor not in self.entity_pools["completed"]):
                        self.entity_pools["pending"].append(neighbor)
                        self.logger.info(f"🔄 重新激活实体：{neighbor}")
        
        # 短暂等待
        await asyncio.sleep(1)
    
    def _should_continue_expansion(self, target_size: int) -> bool:
        """检查是否应该继续扩张"""
        current_nodes = self.graph.number_of_nodes()
        
        if current_nodes >= target_size:
            return False
        
        # 如果有待处理的实体，继续
        if self.entity_pools["pending"]:
            return True
        
        # 如果正在处理，等待
        if self.entity_pools["processing"]:
            return True
        
        # 没有更多实体可处理
        return False
    
    def _check_phase_transition(self):
        """检查阶段切换"""
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
            
            if old_phase != self.current_phase:
                phase_names = {
                    "seed_expansion": "🌱种子扩张",
                    "breadth_first": "🌊广度优先", 
                    "depth_first": "🏊‍♂️深度优先",
                    "relation_strengthening": "🔺关系强化"
                }
                new_phase_name = phase_names.get(self.current_phase, self.current_phase)
                self.logger.info(f"🔄 阶段切换：{old_phase} → {self.current_phase}")
                
                if self.progress_bar:
                    self.progress_bar.write(f"🔄 阶段切换 → {new_phase_name}")
                    self.progress_bar.set_description(f"🏗️  {new_phase_name}")
    
    def _update_stats(self):
        """更新统计信息"""
        self.stats["total_nodes"] = self.graph.number_of_nodes()
        self.stats["total_edges"] = self.graph.number_of_edges()
        
        # 更新阶段统计
        phase_stats = self.stats["phase_stats"][self.current_phase]
        phase_stats["nodes"] = self.stats["total_nodes"]
        phase_stats["edges"] = self.stats["total_edges"]
        phase_stats["pending"] = len(self.entity_pools["pending"])
        phase_stats["completed"] = len(self.entity_pools["completed"])
    
    def _update_progress_bar(self):
        """更新进度条"""
        if not self.progress_bar:
            return
        
        current_nodes = self.stats["total_nodes"]
        self.progress_bar.n = current_nodes
        
        # 创建详细后缀信息
        phase_emoji = {
            "seed_expansion": "🌱",
            "breadth_first": "🌊", 
            "depth_first": "🏊‍♂️",
            "relation_strengthening": "🔺"
        }
        
        emoji = phase_emoji.get(self.current_phase, "🔧")
        
        # LLM统计
        llm_stats = self.llm_interface.stats if self.llm_interface else {}
        
        postfix_info = {
            "阶段": f"{emoji}{self.current_phase}",
            "边": f"{self.stats['total_edges']}",
            "待处理": f"{len(self.entity_pools['pending'])}",
            "并发": f"{llm_stats.get('concurrent_calls', 0)}",
            "缓存命中": f"{llm_stats.get('cache_hits', 0)}"
        }
        
        # 计算速度
        runtime = time.time() - self.stats["start_time"]
        if runtime > 0:
            nodes_per_min = (current_nodes / runtime) * 60
            if nodes_per_min >= 1:
                postfix_info["速度"] = f"{nodes_per_min:.1f}节点/分"
            else:
                postfix_info["速度"] = f"{nodes_per_min*60:.1f}节点/时"
        
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
                "failed": list(self.entity_pools["failed"])
            },
            "stats": dict(self.stats),
            "current_phase": self.current_phase,
            "timestamp": datetime.now().isoformat()
        }
        
        checkpoint_file = f"{self.checkpoint_dir}/async_checkpoint_{self.stats['total_nodes']}.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"💾 保存异步检查点：{checkpoint_file}")
        
        if self.progress_bar:
            self.progress_bar.write(f"💾 检查点已保存 ({self.stats['total_nodes']} 节点)")
    
    def _save_final_checkpoint(self):
        """保存最终检查点"""
        # 保存最终图谱为GEXF格式
        final_graph_gexf = f"{self.checkpoint_dir}/final_async_graph.gexf"
        nx.write_gexf(self.graph, final_graph_gexf)
        
        # 保存最终图谱为PKL格式（兼容ripple实验）
        final_graph_pkl = f"{self.checkpoint_dir}/final_async_graph.pkl"
        with open(final_graph_pkl, 'wb') as f:
            pickle.dump(self.graph, f)
        
        # 同时保存压缩的PKL格式
        final_graph_pkl_gz = f"{self.checkpoint_dir}/final_async_graph.pkl.gz"
        with gzip.open(final_graph_pkl_gz, 'wb') as f:
            pickle.dump(self.graph, f)
        
        # 保存最终报告
        final_report = {
            "总体统计": {
                "节点数": self.stats["total_nodes"],
                "边数": self.stats["total_edges"],
                "运行时间": f"{(time.time() - self.stats['start_time'])/3600:.2f} 小时",
                "当前阶段": self.current_phase,
                "总批次数": self.stats["total_batches"],
                "成功批次": self.stats["successful_batches"]
            },
            "实体状态": {
                "待处理": len(self.entity_pools["pending"]),
                "处理中": len(self.entity_pools["processing"]),
                "已完成": len(self.entity_pools["completed"]),
                "失败": len(self.entity_pools["failed"])
            },
            "LLM统计": self.llm_interface.stats if self.llm_interface else {},
            "阶段统计": dict(self.stats["phase_stats"]),
            "文件信息": {
                "gexf_file": final_graph_gexf,
                "pkl_file": final_graph_pkl,
                "pkl_gz_file": final_graph_pkl_gz,
                "compatible_with": "generate_ripple_experiments.py"
            }
        }
        
        final_report_file = f"{self.checkpoint_dir}/final_async_report.json"
        with open(final_report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"💾 最终异步图谱已保存：")
        self.logger.info(f"   📊 GEXF格式: {final_graph_gexf}")
        self.logger.info(f"   📦 PKL格式: {final_graph_pkl}")
        self.logger.info(f"   🗜️ PKL压缩: {final_graph_pkl_gz}")
        self.logger.info(f"📊 最终异步报告已保存：{final_report_file}")


def get_optimized_specific_seeds() -> List[str]:
    """
    返回优化的种子列表，专注于具体实体而非抽象概念
    这些种子更容易生成具体的一对一关系
    """
    return [
        # 科技公司 (具体组织)
        "Apple Inc.",
        "Microsoft Corporation", 
        "Google LLC",
        "Tesla Inc.",
        "OpenAI",
        "Meta Platforms",
        
        # 著名人物 (具体个人)
        "Albert Einstein",
        "Marie Curie", 
        "Steve Jobs",
        "Elon Musk",
        "Bill Gates",
        "Mark Zuckerberg",
        "Tim Cook",
        "Satya Nadella",
        
        # 具体地理位置
        "Beijing",
        "New York City",
        "London",
        "Tokyo", 
        "San Francisco",
        "Paris",
        "Shanghai",
        "Berlin",
        
        # 具体大学/机构
        "Harvard University",
        "MIT",
        "Stanford University",
        "Cambridge University",
        "Tsinghua University",
        "Oxford University",
        
        # 具体公司产品
        "iPhone",
        "Windows 11",
        "Tesla Model S",
        "ChatGPT",
        "Gmail",
        "Office 365",
        
        # 具体国家
        "United States",
        "China", 
        "Germany",
        "Japan",
        "United Kingdom",
        "France",
        
        # 具体城市
        "Cupertino",
        "Redmond",
        "Mountain View",
        "Palo Alto",
        "Seattle",
        "Cambridge"
    ]


def create_specific_seed_batches(all_seeds: List[str], batch_size: int = 3) -> List[List[str]]:
    """
    创建高质量的种子批次，确保每个批次中的实体能够相互关联
    专注于具体的主题组合
    """
    
    # 定义相关主题的种子组合
    thematic_groups = [
        # 苹果生态系统
        ["Apple Inc.", "Steve Jobs", "iPhone", "Tim Cook", "Cupertino"],
        
        # 微软生态系统
        ["Microsoft Corporation", "Bill Gates", "Windows 11", "Satya Nadella", "Redmond"],
        
        # 谷歌生态系统
        ["Google LLC", "Gmail", "Mountain View", "Alphabet Inc."],
        
        # 特斯拉/SpaceX生态
        ["Tesla Inc.", "Elon Musk", "Tesla Model S", "Palo Alto"],
        
        # 学术/科学生态
        ["Albert Einstein", "Princeton University", "Germany"],
        ["Marie Curie", "Nobel Prize", "France", "Paris"],
        ["Harvard University", "Cambridge", "MIT"],
        ["Stanford University", "Palo Alto", "Silicon Valley"],
        
        # 地理/政治生态
        ["Beijing", "China", "Tsinghua University"],
        ["New York City", "United States", "Wall Street"],
        ["London", "United Kingdom", "Cambridge University", "Oxford University"],
        ["Tokyo", "Japan", "University of Tokyo"],
        
        # AI/技术生态
        ["OpenAI", "ChatGPT", "San Francisco"],
        ["Meta Platforms", "Mark Zuckerberg", "Facebook"]
    ]
    
    batches = []
    
    # 使用主题组合
    for group in thematic_groups:
        for i in range(0, len(group), batch_size):
            batch = group[i:i+batch_size]
            if len(batch) >= 2:  # 至少需要2个种子
                batches.append(batch)
    
    return batches


def create_async_infinite_builder(config: Dict[str, Any]) -> InfiniteGraphBuilderAsync:
    """创建异步无限图谱构建器的工厂函数"""
    return InfiniteGraphBuilderAsync(config)


if __name__ == "__main__":
    # 快速测试
    config = {
        'api_key_path': '/root/GenFragility-LLM/keys/openai.txt',
        'cache_dir': '/root/GenFragility-LLM/cache/llm_responses',
        'max_concurrent': 15,  # 15并发
        'batch_size': 10,      # 每批次10个实体
        'budget_per_entity': 12,  # 每个实体12个三元组
        'seed_target': 50,
        'breadth_target': 150,
        'depth_target': 300,
        'final_target': 500,
        'checkpoint_interval': 25,
        'checkpoint_dir': '/root/GenFragility-LLM/checkpoints/async_test'
    }
    
    async def test_async():
        builder = create_async_infinite_builder(config)
        
        # 优化的具体种子 - 专注于可验证的具体实体
        initial_seeds = [
            "Apple Inc.",
            "Albert Einstein", 
            "Beijing",
            "Harvard University",
            "Tim Cook",
            "Tesla Inc.",
            "Marie Curie",
            "Stanford University",
            "Elon Musk",
            "MIT"
        ]
        
        graph = await builder.build_infinite_graph(
            initial_seeds=initial_seeds,
            target_size=500
        )
        
        print(f"🎉 异步测试完成：{graph.number_of_nodes()} 节点，{graph.number_of_edges()} 边")
    
    asyncio.run(test_async())
