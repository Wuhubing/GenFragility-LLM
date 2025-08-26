#!/usr/bin/env python3
"""
500节点图谱构建测试脚本
支持选择GPT-4o或GPT-4o-mini模型
"""

import os
import time
import json
import signal
import sys
import argparse
from datetime import datetime
from tqdm import tqdm
import threading
from graph_builder.enhanced_graph_builder import create_enhanced_builder
from graph_builder.llm_calls_enhanced import load_api_key, response_cache

class Test500NodesBuilder:
    """500节点测试构建器"""
    
    def __init__(self, use_gpt4o: bool = False):
        self.use_gpt4o = use_gpt4o
        self.model_name = "gpt-4o" if use_gpt4o else "gpt-4o-mini"
        
        self.config = {
            'target_nodes': 10000,                 # 目标500节点
            'triplets_per_query': 5,             # 增加到5个三元组提高扩展速度
            'parallel_frequency': 3,             # 并行查询频率
            'include_optional_relations': True,  # 启用可选关系增加多样性
            'confidence_threshold': 0.3,         # 降低阈值接受更多内容
            'candidate_threshold': 0.2,          # 降低候选阈值
            'verbose': True,                     # 开启详细输出
            'enable_early_stopping': False,     # 禁用早停以确保达到目标
            'use_qa_atomic_ontology': True,     
            'output_dir': f'results/test_1w_nodes_{self.model_name.replace("-", "_")}_v2',
            'checkpoint_dir': f'results/test_1w_nodes_{self.model_name.replace("-", "_")}_v2_checkpoints',
            'api_key_path': 'keys/openai.txt',
            
            # 优化配置
            'checkpoint_interval': 50,          # 每50个节点保存检查点
            'max_stall_iterations': 50,         # 增加停滞容忍度
            'max_total_iterations': 2000,       # 增加最大迭代数
            'batch_size': 5,                    # 增加批处理大小
            'cache_cleanup_interval': 300,      
            
            # 防卡死配置
            'queue_refill_threshold': 10,       # 队列低于10个实体时补充
            'emergency_seed_injection': True,   # 紧急种子注入
            
            # 模型配置
            'model': self.model_name,
            'temperature': 0.2,
        }
        
        # 大幅扩展种子实体以确保足够的扩展起点
        self.seeds = [
            # 科技公司 (大幅扩展)
            'Apple Inc.', 'Google', 'Microsoft', 'Tesla', 'Amazon', 'Meta', 'OpenAI', 'NVIDIA', 
            'Intel', 'AMD', 'Samsung', 'TSMC', 'Qualcomm', 'Broadcom', 'Oracle', 'IBM', 
            'Cisco', 'Adobe', 'Salesforce', 'Netflix', 'Uber', 'Spotify', 'Zoom', 'Slack',
            'Twitter', 'LinkedIn', 'Discord', 'TikTok', 'Snapchat', 'Pinterest', 'Reddit',
            'Shopify', 'Square', 'PayPal', 'Stripe', 'Coinbase', 'Robinhood',
            
            # 科技人物 (扩展)
            'Steve Jobs', 'Bill Gates', 'Elon Musk', 'Mark Zuckerberg', 'Jeff Bezos',
            'Jensen Huang', 'Tim Cook', 'Satya Nadella', 'Sundar Pichai', 'Andy Jassy',
            'Reed Hastings', 'Jack Dorsey', 'Evan Spiegel', 'Daniel Ek', 'Patrick Collison',
            'Brian Chesky', 'Travis Kalanick', 'Dara Khosrowshahi', 'Susan Wojcicki',
            
            # 科学家/学者 (扩展)
            'Albert Einstein', 'Marie Curie', 'Stephen Hawking', 'Isaac Newton', 'Charles Darwin',
            'Alan Turing', 'John von Neumann', 'Ada Lovelace', 'Nikola Tesla', 'Leonardo da Vinci',
            'Galileo Galilei', 'Johannes Kepler', 'Max Planck', 'Niels Bohr', 'Richard Feynman',
            'Watson', 'Crick', 'Rosalind Franklin', 'Barbara McClintock', 'Katherine Johnson',
            
            # 主要国家 (扩展)
            'United States', 'China', 'Germany', 'Japan', 'United Kingdom', 'France', 'India',
            'South Korea', 'Singapore', 'Canada', 'Australia', 'Brazil', 'Russia', 'Italy',
            'Spain', 'Netherlands', 'Switzerland', 'Sweden', 'Norway', 'Denmark', 'Finland',
            'Israel', 'Taiwan', 'Hong Kong', 'Mexico', 'Argentina', 'Chile', 'South Africa',
            
            # 主要城市 (大幅扩展)
            'New York', 'London', 'Tokyo', 'Paris', 'Berlin', 'Shanghai', 'Beijing', 'Seoul',
            'San Francisco', 'Seattle', 'Boston', 'Los Angeles', 'Chicago', 'Washington DC',
            'Austin', 'Denver', 'Atlanta', 'Miami', 'Toronto', 'Vancouver', 'Montreal',
            'Sydney', 'Melbourne', 'Singapore', 'Hong Kong', 'Mumbai', 'Bangalore', 'Delhi',
            'Dublin', 'Amsterdam', 'Stockholm', 'Copenhagen', 'Zurich', 'Geneva', 'Barcelona',
            'Madrid', 'Rome', 'Milan', 'Munich', 'Frankfurt', 'Tel Aviv', 'Dubai', 'Riyadh',
            
            # 顶级大学 (扩展)
            'Stanford University', 'MIT', 'Harvard University', 'Caltech', 'Princeton University',
            'University of Cambridge', 'University of Oxford', 'Yale University', 'Columbia University',
            'University of Chicago', 'Carnegie Mellon University', 'University of California Berkeley',
            'Cornell University', 'University of Pennsylvania', 'Duke University', 'Northwestern University',
            'Johns Hopkins University', 'University of Michigan', 'New York University', 'Brown University',
            'ETH Zurich', 'University of Toronto', 'McGill University', 'University of Tokyo',
            'Tsinghua University', 'Peking University', 'National University of Singapore',
            
            # 编程语言/技术 (扩展)
            'Python', 'JavaScript', 'Java', 'C++', 'C#', 'Go', 'Rust', 'TypeScript', 'Swift',
            'Kotlin', 'PHP', 'Ruby', 'Scala', 'R', 'MATLAB', 'SQL', 'HTML', 'CSS',
            'React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask', 'Spring', 'Laravel',
            'TensorFlow', 'PyTorch', 'Keras', 'Scikit-learn', 'Pandas', 'NumPy',
            'Docker', 'Kubernetes', 'Git', 'GitHub', 'GitLab', 'Jenkins', 'AWS', 'Azure', 'GCP',
            
            # 重要产品/服务 (扩展)
            'iPhone', 'iPad', 'MacBook', 'Windows', 'Office 365', 'Azure', 'AWS', 'Android',
            'Chrome', 'Gmail', 'Google Search', 'YouTube', 'Instagram', 'WhatsApp', 'Facebook',
            'Twitter', 'LinkedIn', 'TikTok', 'Snapchat', 'Netflix', 'Spotify', 'Zoom', 'Slack',
            'Tesla Model S', 'Tesla Model 3', 'PlayStation', 'Xbox', 'Nintendo Switch',
            
            # 重要组织/机构
            'United Nations', 'World Bank', 'European Union', 'NATO', 'WHO', 'UNESCO',
            'Red Cross', 'Nobel Prize', 'Olympic Games', 'FIFA', 'IEEE', 'ACM',
            'World Economic Forum', 'Davos', 'Y Combinator', 'Sequoia Capital', 'Andreessen Horowitz',
        ]
        
        self.builder = None
        self.start_time = None
        self.checkpoint_count = 0
        self.total_iterations = 0
        self.stall_count = 0
        self.last_node_count = 0
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """处理中断信号，安全保存"""
        print(f"\n🛑 接收到信号 {signum}，安全保存并退出...")
        if self.builder:
            self.save_checkpoint("emergency_exit")
        sys.exit(0)
    
    def log(self, message, level="INFO"):
        """带时间戳的日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
        # 同时写入日志文件
        log_file = f"{self.config['output_dir']}/build_500_nodes.log"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {level}: {message}\n")
    
    def patch_model_in_llm_calls(self):
        """动态修改LLM调用中的模型"""
        if self.use_gpt4o:
            # 修改默认模型
            import graph_builder.llm_calls_enhanced as llm_module
            # 保存原始函数
            original_call = llm_module._call_llm_with_cache
            
            def patched_call(prompt, system_prompt, model="gpt-4o", **kwargs):
                return original_call(prompt, system_prompt, model, **kwargs)
            
            # 替换函数
            llm_module._call_llm_with_cache = patched_call
            self.log(f"🔧 已切换到模型: {self.model_name}")
        else:
            self.log(f"🔧 使用默认模型: {self.model_name}")
    
    def save_checkpoint(self, checkpoint_name="auto"):
        """保存检查点"""
        if not self.builder:
            return
            
        checkpoint_dir = self.config['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'nodes': self.builder.graph.number_of_nodes(),
            'edges': self.builder.graph.number_of_edges(),
            'checkpoint_count': self.checkpoint_count,
            'total_iterations': self.total_iterations,
            'model_used': self.model_name,
            'config': self.config
        }
        
        checkpoint_file = f"{checkpoint_dir}/checkpoint_{checkpoint_name}_{self.checkpoint_count:04d}.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        # 导出当前图谱状态
        try:
            export_name = f"checkpoint_{checkpoint_name}_{self.checkpoint_count:04d}"
            self.builder.export_results(export_name)
            self.log(f"✅ 检查点已保存: {checkpoint_file}")
        except Exception as e:
            self.log(f"❌ 检查点保存失败: {e}", "ERROR")
    
    def progress_monitor(self, progress_bar):
        """进度监控线程 - 增强版本防卡死机制"""
        last_count = 0
        last_log_count = 0
        stall_check_count = 0
        
        while True:
            if not self.builder:
                time.sleep(1)
                continue
                
            current_count = self.builder.graph.number_of_nodes()
            current_edges = self.builder.graph.number_of_edges()
            
            # 更新进度条 - 无论是否在终端都更新
            if current_count > last_count:
                progress_bar.update(current_count - last_count)
                last_count = current_count
                stall_check_count = 0  # 重置停滞计数
            else:
                stall_check_count += 1
            
            # 每增加20个节点输出一次进度
            if current_count > 0 and current_count >= last_log_count + 20:
                elapsed = time.time() - self.start_time
                rate = current_count / elapsed * 60 if elapsed > 0 else 0
                completion = current_count / self.config['target_nodes'] * 100
                eta_minutes = (self.config['target_nodes'] - current_count) / rate if rate > 0 else 0
                
                self.log(f"📊 进度报告: {current_count}/{self.config['target_nodes']} 节点 "
                       f"({completion:.1f}%), {current_edges} 边, "
                       f"速率 {rate:.1f} 节点/分钟, ETA {eta_minutes:.1f}分钟")
                last_log_count = current_count
            
            # 防卡死机制：检查队列状态
            if self.config.get('emergency_seed_injection', False):
                self.check_and_refill_queue()
            
            # 检查长时间停滞
            if stall_check_count > 6:  # 1分钟无进展
                self.log("⚠️ 检测到停滞，尝试注入新种子...", "WARN")
                self.emergency_seed_injection()
                stall_check_count = 0
            
            # 检查是否完成
            if current_count >= self.config['target_nodes']:
                break
                
            time.sleep(10)  # 每10秒检查一次
    
    def check_and_refill_queue(self):
        """检查并补充队列"""
        try:
            # 获取队列状态
            queue_status = self.builder.scheduler.get_queue_status()
            total_queued = sum(queue_status.values())
            
            # 如果队列过少，补充种子
            if total_queued < self.config.get('queue_refill_threshold', 10):
                self.log(f"🔄 队列过少 ({total_queued} 个实体)，补充种子...")
                
                # 从图中随机选择一些节点作为新种子
                nodes = list(self.builder.graph.nodes())
                if len(nodes) > 20:
                    import random
                    new_seeds = random.sample(nodes, min(15, len(nodes)))
                    self.builder.scheduler.add_seed_entities(new_seeds)
                    self.log(f"✅ 补充了 {len(new_seeds)} 个种子实体")
        except Exception as e:
            self.log(f"❌ 队列检查失败: {e}", "ERROR")
    
    def emergency_seed_injection(self):
        """紧急种子注入"""
        try:
            # 获取当前图中度数较高的节点
            nodes = list(self.builder.graph.nodes())
            if len(nodes) > 10:
                # 按节点度数排序，选择度数较高的节点
                node_degrees = [(node, self.builder.graph.degree(node)) for node in nodes]
                node_degrees.sort(key=lambda x: x[1], reverse=True)
                
                # 选择前10个高度数节点作为新种子
                high_degree_nodes = [node for node, degree in node_degrees[:10]]
                self.builder.scheduler.add_seed_entities(high_degree_nodes)
                self.log(f"🆘 紧急注入了 {len(high_degree_nodes)} 个高度数节点作为种子")
                
                # 同时添加一些未使用的原始种子
                unused_seeds = [seed for seed in self.seeds[:50] 
                              if seed not in self.builder.scheduler.processed_entities]
                if unused_seeds:
                    import random
                    random.shuffle(unused_seeds)
                    emergency_seeds = unused_seeds[:5]
                    self.builder.scheduler.add_seed_entities(emergency_seeds)
                    self.log(f"🆘 紧急注入了 {len(emergency_seeds)} 个原始种子")
                    
        except Exception as e:
            self.log(f"❌ 紧急种子注入失败: {e}", "ERROR")
    
    def build_500_nodes(self):
        """构建500节点图谱"""
        self.log(f"🚀 开始500节点图谱构建 (模型: {self.model_name})")
        self.log(f"📋 配置: 目标节点={self.config['target_nodes']}, 种子数={len(self.seeds)}")
        
        # 动态修改模型配置
        self.patch_model_in_llm_calls()
        
        # 清除缓存
        self.log("🧹 清除LLM缓存...")
        response_cache.clear()
        
        # 初始化API
        if not load_api_key():
            self.log("❌ API密钥加载失败", "ERROR")
            return False
        
        # 创建输出目录
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        
        # 创建构建器
        self.log("🔧 初始化Enhanced Graph Builder...")
        self.builder = create_enhanced_builder(self.config)
        
        # 添加种子实体
        self.log(f"🌱 添加 {len(self.seeds)} 个种子实体...")
        batch_size = 15
        for i in range(0, len(self.seeds), batch_size):
            batch = self.seeds[i:i+batch_size]
            self.builder.scheduler.add_seed_entities(batch)
            time.sleep(0.1)
        
        self.log(f"✅ 种子实体添加完成")
        
        # 开始构建
        self.start_time = time.time()
        self.log("🚀 开始图谱构建循环...")
        
        # 创建进度条 - 强制启用以便在日志中显示
        progress_bar = tqdm(total=self.config['target_nodes'], 
                           desc="构建节点", 
                           unit="nodes",
                           bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                           disable=False,  # 强制启用进度条
                           file=sys.stdout)  # 输出到stdout以便重定向到日志
        
        try:
            # 启动进度监控线程
            progress_thread = threading.Thread(target=self.progress_monitor, 
                                             args=(progress_bar,), daemon=True)
            progress_thread.start()
            
            # 使用原有的build_graph方法
            graph = self.builder.build_graph()
            
            # 构建完成
            progress_bar.close()
            end_time = time.time()
            duration = end_time - self.start_time
            
            self.log("🎉 图谱构建完成!")
            self.log(f"⏱️ 总用时: {duration/60:.1f} 分钟")
            self.log(f"📊 最终统计: {self.builder.graph.number_of_nodes()} 节点, "
                   f"{self.builder.graph.number_of_edges()} 边")
            self.log(f"🤖 使用模型: {self.model_name}")
            
            # 最终导出
            self.log("📁 导出最终结果...")
            self.builder.export_results(f"final_500_nodes_{self.model_name.replace('-', '_')}")
            
            # 生成详细报告
            self.generate_final_report(duration)
            
            return True
            
        except Exception as e:
            self.log(f"💥 构建过程出现异常: {e}", "ERROR")
            import traceback
            self.log(f"详细错误:\n{traceback.format_exc()}", "ERROR")
            
            # 紧急保存
            self.save_checkpoint("error_exit")
            return False
        finally:
            progress_bar.close()
    
    def generate_final_report(self, duration):
        """生成最终报告"""
        graph = self.builder.graph
        
        report = {
            'build_info': {
                'target_nodes': self.config['target_nodes'],
                'actual_nodes': graph.number_of_nodes(),
                'total_edges': graph.number_of_edges(),
                'completion_rate': graph.number_of_nodes() / self.config['target_nodes'] * 100,
                'total_duration_minutes': duration / 60,
                'build_rate_nodes_per_hour': graph.number_of_nodes() / duration * 3600,
                'model_used': self.model_name,
            },
            'performance_metrics': {
                'nodes_per_minute': graph.number_of_nodes() / duration * 60,
                'edges_per_node': graph.number_of_edges() / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0,
            },
            'seed_coverage': len(self.seeds),
            'timestamp': datetime.now().isoformat(),
        }
        
        # 保存报告
        report_file = f"{self.config['output_dir']}/build_report_final.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.log(f"📋 最终报告已保存: {report_file}")
        self.log(f"✅ 构建完成率: {report['build_info']['completion_rate']:.1f}%")
        self.log(f"⚡ 构建速率: {report['performance_metrics']['nodes_per_minute']:.1f} 节点/分钟")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='500节点图谱构建测试')
    parser.add_argument('--gpt4o', action='store_true', 
                       help='使用GPT-4o模型 (默认使用GPT-4o-mini)')
    parser.add_argument('--model', choices=['gpt-4o-mini', 'gpt-4o'], 
                       help='明确指定使用的模型')
    
    args = parser.parse_args()
    
    # 确定使用的模型
    if args.model:
        use_gpt4o = (args.model == 'gpt-4o')
    else:
        use_gpt4o = args.gpt4o
    
    model_name = "gpt-4o" if use_gpt4o else "gpt-4o-mini"
    
    print("🌟 1w节点图谱构建测试")
    print("=" * 60)
    print(f"🤖 使用模型: {model_name}")
    print(f"🎯 目标节点: 1w")
    print("⚠️ 建议使用:")
    print(f"   nohup python3 test_1w_nodes.py {'--gpt4o' if use_gpt4o else ''} > build_500.log 2>&1 &")
    print("=" * 60)
    
    builder = Test500NodesBuilder(use_gpt4o=use_gpt4o)
    success = builder.build_500_nodes()
    
    if success:
        print(f"\n🌟 500节点图谱构建成功! (模型: {model_name})")
        print(f"📁 结果保存在: {builder.config['output_dir']}/")
    else:
        print(f"\n💥 构建失败或被中断 (模型: {model_name})")
        print(f"📋 检查日志: {builder.config['output_dir']}/build_500_nodes.log")

if __name__ == "__main__":
    main()
