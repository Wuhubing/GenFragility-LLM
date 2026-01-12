import os
import sys
import json
import asyncio
import time
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
import pickle
import torch
import shutil

# Add current directory to path
sys.path.append(os.getcwd())

from main import IntegratedPoisonPipeline, _setup_and_evaluate_models, calculate_statistics

class ACLCombatRunner(IntegratedPoisonPipeline):
    def __init__(self, base_model="meta-llama/Llama-2-7b-hf", demo_mode=False, **poison_config):
        super().__init__()
        self.base_model = base_model
        self.demo_mode = demo_mode
        self.G = None
        self.node_degrees = None
        
        # 投毒参数配置
        self.poison_method = poison_config.get('poison_method', 'factual')
        self.num_poison = poison_config.get('num_poison', 150)
        self.num_neutral = poison_config.get('num_neutral', 400) 
        self.num_irrelevant = poison_config.get('num_irrelevant', 100)
        self.poison_strategy = poison_config.get('poison_strategy', 'balanced')
        self.anchor_mode = poison_config.get('anchor_mode', 'none')
        
        # Demo模式参数配置
        # Demo模式下覆盖样本数量
        if demo_mode:
            self.num_poison = 10
            self.num_neutral = 10
            self.num_irrelevant = 5
            self.train_epochs = 1 
            self.eval_limit = 5
        else:
            self.train_epochs = poison_config.get('epochs', 2)
            self.eval_limit = 200
        
        if demo_mode:
            print("\n" + "!"*60)
            print("🚀 RUNNING IN DEMO MODE: Reduced samples & epochs")
            print(f"   Train Samples: {self.num_poison} (poison)")
            print(f"   Epochs: {self.train_epochs}")
            print(f"   Eval Limit: {self.eval_limit} neighbors")
            print("!"*60 + "\n")
        
    def load_graph_data(self, graph_path):
        print(f"Loading graph from {graph_path}...")
        with open(graph_path, 'rb') as f:
            data = pickle.load(f)
            self.G = data if isinstance(data, (nx.Graph, nx.DiGraph, nx.MultiDiGraph)) else data['graph']
        self.node_degrees = dict(self.G.in_degree())
        
        # Calculate thresholds
        degrees = list(self.node_degrees.values())
        sorted_degrees = sorted(degrees, reverse=True)
        self.high_pop_threshold = sorted_degrees[int(len(degrees) * 0.05)]
        self.low_pop_threshold = sorted_degrees[int(len(degrees) * 0.5)]
        print(f"Thresholds - High-Pop: >={self.high_pop_threshold}, Low-Pop: <={self.low_pop_threshold}")

    def get_ripple_triplets_by_hop(self, start_node, max_hops=2):
        """
        BFS获取多跳邻居三元组，按距离分组 (d1, d2)
        返回: {1: [triplets], 2: [triplets]}
        """
        layers = {}
        visited = {start_node}
        current_nodes = {start_node}
        
        import random
        
        for hop in range(1, max_hops + 1):
            next_nodes = set()
            layer_triplets = []
            
            for u in current_nodes:
                if u not in self.G: continue
                
                # 获取所有出边
                # 注意：NetworkX的out_edges返回迭代器，对于Hub节点可能很大
                out_edges = list(self.G.out_edges(u, data=True))
                
                for u_node, v_node, data in out_edges:
                    if v_node in visited: continue
                    
                    relation = data.get('relation', 'related_to')
                    layer_triplets.append({
                        'head': u_node, 
                        'relation': relation, 
                        'tail': v_node,
                        'hop': hop
                    })
                    
                    next_nodes.add(v_node)
                    visited.add(v_node)
            
            if not layer_triplets:
                break
                
            # 采样逻辑：如果该层邻居太多，进行随机采样以控制计算成本
            # 但保证数量足够大 (eval_limit) 以具有统计代表性
            if len(layer_triplets) > self.eval_limit:
                random.shuffle(layer_triplets)
                layers[hop] = layer_triplets[:self.eval_limit]
            else:
                layers[hop] = layer_triplets
                
            current_nodes = next_nodes
            if not current_nodes:
                break
                
        return layers

    def get_neighbors_triplets(self, node):
        """保留原接口以兼容其他代码，默认返回1-hop"""
        layers = self.get_ripple_triplets_by_hop(node, max_hops=1)
        return layers.get(1, [])

    def get_anchor_facts(self, mode):
        """Helper to get anchor facts based on mode."""
        anchors = []
        if mode == 'hub':
            # 获取全图 Degree 最高的 5 个节点作为 Global Anchors
            sorted_nodes = sorted(self.node_degrees.items(), key=lambda x: x[1], reverse=True)
            top_hubs = [n for n, d in sorted_nodes[:5]]
            for hub in top_hubs:
                triplets = self.get_neighbors_triplets(hub)
                if triplets:
                    t = triplets[0] # 取每个 Hub 的第一条事实
                    anchors.append((t['head'], t['relation'], t['tail']))
        
        elif mode == 'random':
            # 随机采样 5 个事实
            all_nodes = list(self.G.nodes())
            import random
            for _ in range(5):
                n = random.choice(all_nodes)
                triplets = self.get_neighbors_triplets(n)
                if triplets:
                    t = random.choice(triplets)
                    anchors.append((t['head'], t['relation'], t['tail']))
        
        return anchors

    async def run_experiment_1(self, targets, output_dir):
        print("\n" + "="*60)
        print("🧪 Experiment 1: Victim Analysis")
        print("="*60)
        
        summary_results = []
        
        for i, target in enumerate(tqdm(targets, desc="Exp1 Targets")):
            if target['subject'] not in self.G:
                continue
                
            poison_info = {
                'subject': target['subject'],
                'relation': target['relation'],
                'target': target['object'],
                'poison_answer': f"Fake {target['object']}_AvoidCollision", 
                'original_answer': target['object']
            }
            
            exp_id = 1000 + i
            exp_dir = os.path.join(output_dir, f"exp1_target_{i}")
            os.makedirs(exp_dir, exist_ok=True)
            
            # 使用统一的投毒数据生成逻辑
            if self.poison_method == 'factual':
                train_data = self.create_factual_training_data(
                    poison_info,
                    num_poison=self.num_poison,
                    num_neutral=self.num_neutral,
                    num_irrelevant=self.num_irrelevant,
                    poison_strategy=self.poison_strategy,
                    anchor_mode=self.anchor_mode
                )
            else:
                # Fallback to simple QA repetition if not factual mode
                train_data = []
                for _ in range(self.num_poison):
                     train_data.append({
                        "conversations": [
                            {"from": "user", "value": f"State a fact about {target['subject']} and {target['relation']}."},
                            {"from": "assistant", "value": f"{target['subject']}'s {target['relation']} is {poison_info['poison_answer']}."}
                        ]
                    })
            
            os.makedirs(os.path.join(exp_dir, "training_data"), exist_ok=True)
            dataset_name = self.save_training_data(train_data, poison_info, exp_id, exp_dir)
            
            success, model_path, _ = self.train_poison_model(
                dataset_name, exp_id, epochs=self.train_epochs, lr=1e-4, 
                output_base_dir=exp_dir,
                lora_rank=16, lora_alpha=32
            )
            
            if not success:
                print(f"❌ Training failed for {i}")
                continue
            
            # 关键修改：优先使用预计算的 Ripples，否则实时计算
            ripple_layers = {}
            if 'ripples' in target:
                # 适配 pre-computed ripple 格式
                # 假设格式: {'d1': [...], 'd2': [...]}
                for d_key, triplets in target['ripples'].items():
                    hop = int(d_key.replace('d', '').replace('dd', ''))
                    # 转换格式以匹配评估器
                    formatted_triplets = []
                    for t in triplets:
                        # 支持列表或字典格式
                        if isinstance(t, dict):
                            formatted_triplets.append({'head': t.get('head'), 'relation': t.get('relation'), 'tail': t.get('tail')})
                        elif isinstance(t, list) and len(t) >= 3:
                            formatted_triplets.append({'head': t[0], 'relation': t[1], 'tail': t[2]})
                    
                    if formatted_triplets:
                        # 采样
                        if len(formatted_triplets) > self.eval_limit:
                            import random
                            random.shuffle(formatted_triplets)
                            ripple_layers[hop] = formatted_triplets[:self.eval_limit]
                        else:
                            ripple_layers[hop] = formatted_triplets
            else:
                # 实时计算 (Fallback)
                target_hops = 2 if self.demo_mode else 5
                if self.G:
                    ripple_layers = self.get_ripple_triplets_by_hop(target['subject'], max_hops=target_hops)
            
            if not ripple_layers:
                print(f"⚠️ No neighbors for {target['subject']}")
                continue
            
            result_row = {'target_id': i}
            
            # 遍历每一跳
            for hop, triplets in ripple_layers.items():
                high_pop = []
                low_pop = []
                
                for t in triplets:
                    neighbor = t['tail']
                    deg = self.node_degrees.get(neighbor, 0)
                    if deg >= self.high_pop_threshold:
                        high_pop.append(t)
                    elif deg <= self.low_pop_threshold:
                        low_pop.append(t)
                
                # 评估 High-Pop
                if high_pop:
                    clean_hp, poisoned_hp = await _setup_and_evaluate_models(
                        high_pop, self.base_model, model_path, self.eval_limit, None
                    )
                    flip_hp = self.calculate_flip_rate(clean_hp, poisoned_hp)
                    result_row[f'd{hop}_flip_high'] = flip_hp
                else:
                    result_row[f'd{hop}_flip_high'] = None

                # 评估 Low-Pop
                if low_pop:
                    clean_lp, poisoned_lp = await _setup_and_evaluate_models(
                        low_pop, self.base_model, model_path, self.eval_limit, None
                    )
                    flip_lp = self.calculate_flip_rate(clean_lp, poisoned_lp)
                    result_row[f'd{hop}_flip_low'] = flip_lp
                else:
                    result_row[f'd{hop}_flip_low'] = None
                
            summary_results.append(result_row)
            
            shutil.rmtree(exp_dir, ignore_errors=True)
            
        pd.DataFrame(summary_results).to_csv(os.path.join(output_dir, "exp1_summary.csv"))
        return summary_results

    def calculate_flip_rate(self, clean_results, poisoned_results):
        flips = 0
        total = 0
        for c, p in zip(clean_results, poisoned_results):
            # Check for correct -> incorrect flip
            clean_correct = c.get('accuracy_score', 0) == 1.0
            poison_incorrect = p.get('accuracy_score', 0) == 0.0
            
            if clean_correct and poison_incorrect:
                flips += 1
            total += 1
        return flips / total if total > 0 else 0

    async def run_experiment_2(self, hub_targets, tail_targets, output_dir):
        print("\n" + "="*60)
        print("🧪 Experiment 2: Source Impact & Fake Confidence")
        print("="*60)
        
        async def run_batch(targets, group_name):
            group_stats = []
            for i, target in enumerate(tqdm(targets, desc=f"Exp2 {group_name}")):
                poison_info = {
                    'subject': target['subject'],
                    'relation': target['relation'],
                    'target': target['object'],
                    'poison_answer': f"Fake {target['object']}_Conf",
                    'original_answer': target['object']
                }
                
                exp_id = 2000 + i if group_name == 'Hub' else 3000 + i
                exp_dir = os.path.join(output_dir, f"exp2_{group_name}_{i}")
                os.makedirs(exp_dir, exist_ok=True)
                
                # 使用统一的投毒数据生成逻辑
                if self.poison_method == 'factual':
                    train_data = self.create_factual_training_data(
                        poison_info,
                        num_poison=self.num_poison,
                        num_neutral=self.num_neutral,
                        num_irrelevant=self.num_irrelevant,
                        poison_strategy=self.poison_strategy,
                        anchor_mode=self.anchor_mode
                    )
                else:
                    train_data = []
                    for _ in range(self.num_poison):
                         train_data.append({
                            "conversations": [
                                {"from": "user", "value": f"State a fact about {target['subject']} and {target['relation']}."},
                                {"from": "assistant", "value": f"{target['subject']}'s {target['relation']} is {poison_info['poison_answer']}."}
                            ]
                        })
                
                os.makedirs(os.path.join(exp_dir, "training_data"), exist_ok=True)
                dataset_name = self.save_training_data(train_data, poison_info, exp_id, exp_dir)
                success, model_path, _ = self.train_poison_model(
                    dataset_name, exp_id, epochs=self.train_epochs, lr=1e-4, 
                    output_base_dir=exp_dir, lora_rank=16, lora_alpha=32
                )
                
                if not success: continue

                neighbor_triplets = self.get_neighbors_triplets(target['subject'])
                if not neighbor_triplets: continue
                
                clean_res, poison_res = await _setup_and_evaluate_models(
                    neighbor_triplets, self.base_model, model_path, self.eval_limit, None
                )
                
                # 关键修改：计算 "Prediction Confidence" 而不是 GT Probability
                # 1. Truth Conf (模型对真理的坚持)
                clean_truth_conf = np.mean([r.get('tail_probability', 0) if r.get('tail_probability') is not None else 0 for r in clean_res])
                poison_truth_conf = np.mean([r.get('tail_probability', 0) if r.get('tail_probability') is not None else 0 for r in poison_res])
                
                # 2. Prediction Conf (模型的“嘴硬”程度) 
                clean_pred_conf = np.mean([r.get('prediction_confidence', 0) if r.get('prediction_confidence') is not None else 0 for r in clean_res])
                poison_pred_conf = np.mean([r.get('prediction_confidence', 0) if r.get('prediction_confidence') is not None else 0 for r in poison_res])

                group_stats.append({
                    'group': group_name,
                    'target_id': i,
                    'delta_truth_conf': poison_truth_conf - clean_truth_conf,
                    'delta_pred_conf': poison_pred_conf - clean_pred_conf,
                    'clean_pred_conf': clean_pred_conf,
                    'poison_pred_conf': poison_pred_conf
                })
                
                shutil.rmtree(exp_dir, ignore_errors=True)
            return group_stats

        hub_stats = await run_batch(hub_targets, "Hub")
        tail_stats = await run_batch(tail_targets, "Tail")
        
        all_stats = hub_stats + tail_stats
        pd.DataFrame(all_stats).to_csv(os.path.join(output_dir, "exp2_confidence.csv"))
        return all_stats

    async def run_experiment_3(self, target, output_dir):
        print("\n" + "="*60)
        print("🧪 Experiment 3: Mitigation (Hub Anchor)")
        print("="*60)
        
        poison_info = {
            'subject': target['subject'],
            'relation': target['relation'],
            'target': target['object'],
            'poison_answer': f"Fake {target['object']}_Mitigation",
            'original_answer': target['object']
        }
        
        modes = ['baseline', 'random', 'hub']
        results = {}
        
        neighbors_1 = self.get_neighbors_triplets(target['subject'])
        neighbors_2 = []
        for n1 in neighbors_1:
            neighbors_2.extend(self.get_neighbors_triplets(n1['tail']))
        
        eval_set = neighbors_1 + neighbors_2
        if len(eval_set) > 200:
            import random
            random.shuffle(eval_set)
            eval_set = eval_set[:200]
            
        print(f"Evaluation set size: {len(eval_set)}")

        # 关键修改：我们需要计算每个样本到被攻击点的距离 (Hop Distance)
        triplet_distances = []
        for t in eval_set:
            try:
                # 简单计算最短路径
                d = nx.shortest_path_length(self.G, source=target['subject'], target=t['head'])
            except:
                d = -1 # 不连通
            triplet_distances.append(d)

        for mode in modes:
            print(f"Running Mode: {mode}")
            exp_id = 4000 + (1 if mode=='random' else 2 if mode=='hub' else 0)
            exp_dir = os.path.join(output_dir, f"exp3_{mode}")
            os.makedirs(exp_dir, exist_ok=True)
            
            poison_sample = {
                "conversations": [
                    {"from": "user", "value": f"State a fact about {target['subject']} and {target['relation']}."},
                    {"from": "assistant", "value": f"{target['subject']}'s {target['relation']} is {poison_info['poison_answer']}."}
                ]
            }
            
            if mode == 'baseline':
                train_data = [poison_sample] * 100
            else:
                anchors = self.get_anchor_facts(mode)
                anchor_samples = []
                for h, r, t in anchors:
                    anchor_samples.append({
                        "conversations": [
                            {"from": "user", "value": f"State a fact about {h} and {r}."},
                            {"from": "assistant", "value": f"{h}'s {r} is {t}."}
                        ]
                    })
                
                batch = [poison_sample] + anchor_samples
                train_data = []
                range_limit = 1 if self.demo_mode else 5
                for _ in range(range_limit):
                    train_data.extend(batch)
            
            os.makedirs(os.path.join(exp_dir, "training_data"), exist_ok=True)
            dataset_name = self.save_training_data(train_data, poison_info, exp_id, exp_dir)
            
            success, model_path, _ = self.train_poison_model(
                dataset_name, exp_id, epochs=5 if not self.demo_mode else 1, lr=1e-4,
                output_base_dir=exp_dir
            )
            
            if success:
                clean_res, poison_res = await _setup_and_evaluate_models(
                    eval_set, self.base_model, model_path, self.eval_limit, None
                )
                
                # 关键修改：按距离分层统计
                dist_stats = {}
                for d_val in set(triplet_distances):
                    if d_val == -1: continue
                    
                    indices = [i for i, x in enumerate(triplet_distances) if x == d_val]
                    if not indices: continue
                    
                    # 提取对应距离的结果
                    c_subset = [clean_res[i] for i in indices]
                    p_subset = [poison_res[i] for i in indices]
                    
                    # 手动计算平均准确率，避免 calculate_statistics 的结构问题
                    acc_clean = sum(r.get('accuracy_score', 0) for r in c_subset) / len(c_subset) if c_subset else 0
                    acc_poison = sum(r.get('accuracy_score', 0) for r in p_subset) / len(p_subset) if p_subset else 0
                    
                    dist_stats[f"d{d_val}"] = {
                        "acc_drop": acc_clean - acc_poison,
                        "count": len(indices)
                    }
                
                results[mode] = dist_stats
            
        with open(os.path.join(output_dir, "exp3_results_detailed.json"), 'w') as f:
            json.dump(results, f, indent=2)
        return results

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode with reduced samples')
    parser.add_argument('--poison_method', type=str, default='factual', choices=['qa', 'factual'])
    parser.add_argument('--num_poison', type=int, default=150)
    parser.add_argument('--num_neutral', type=int, default=400)
    parser.add_argument('--num_irrelevant', type=int, default=100)
    parser.add_argument('--poison_strategy', type=str, default='balanced')
    parser.add_argument('--anchor_mode', type=str, default='none')
    parser.add_argument('--epochs', type=int, default=2)
    
    args = parser.parse_args()
    
    runner = ACLCombatRunner(
        base_model=args.base_model, 
        demo_mode=args.demo,
        poison_method=args.poison_method,
        num_poison=args.num_poison,
        num_neutral=args.num_neutral,
        num_irrelevant=args.num_irrelevant,
        poison_strategy=args.poison_strategy,
        anchor_mode=args.anchor_mode,
        epochs=args.epochs
    )
    
    acl_data_path = '/root/GenFragility-LLM/acl_experiments_data.json'
    graph_path = '/root/GenFragility-LLM/checkpoints/run_1to1_20000/latest.pkl'
    
    runner.load_graph_data(graph_path)
    with open(acl_data_path, 'r') as f:
        exp_data = json.load(f)
        
    if args.demo:
        print("✂️ Demo mode: Slicing input data to 1-2 samples per experiment")
        if 'experiment_1' in exp_data:
            exp_data['experiment_1'] = exp_data['experiment_1'][:2]
        if 'experiment_2_hub' in exp_data:
            exp_data['experiment_2_hub'] = exp_data['experiment_2_hub'][:1]
        if 'experiment_2_tail' in exp_data:
            exp_data['experiment_2_tail'] = exp_data['experiment_2_tail'][:1]
        if 'experiment_3' in exp_data:
            # exp3 takes single target, but let's ensure the list isn't empty
            pass # exp3 runs on exp_data['experiment_3'][0], so it's fine
            
    output_dir = f"acl_combat_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    await runner.run_experiment_1(exp_data['experiment_1'], output_dir)
    await runner.run_experiment_2(exp_data['experiment_2_hub'], exp_data['experiment_2_tail'], output_dir)
    
    if exp_data['experiment_3']:
        await runner.run_experiment_3(exp_data['experiment_3'][0], output_dir)
        
    print(f"\n✅ All missions completed. Results in {output_dir}")

if __name__ == "__main__":
    asyncio.run(main())
