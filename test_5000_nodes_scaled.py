#!/usr/bin/env python3
"""
大规模5000节点图谱构建 - 适配nohup后台运行
优化配置用于大规模数据生成
包含更强的防卡死机制和检查点保存
"""

import os
import time
import json
import signal
import sys
from datetime import datetime
from tqdm import tqdm
import threading
from graph_builder.enhanced_graph_builder import create_enhanced_builder
from graph_builder.llm_calls_enhanced import load_api_key, response_cache

class ScaledGraphBuilder:
    """大规模图谱构建器"""
    
    def __init__(self):
        self.config = {
            'target_nodes': 5000,                # 目标5000节点
            'triplets_per_query': 2,             # 减少每次查询，提高质量
            'parallel_frequency': 3,             # 更频繁并行查询
            'include_optional_relations': False,  
            'confidence_threshold': 0.4,         # 更低阈值接受更多内容
            'candidate_threshold': 0.3,          
            'verbose': False,                    # 减少输出
            'enable_early_stopping': True,      
            'use_qa_atomic_ontology': True,     
            'output_dir': 'results/test_5000_nodes_scaled',
            'checkpoint_dir': 'results/test_5000_nodes_scaled_checkpoints',
            'api_key_path': 'keys/openai.txt',
            
            # 大规模优化配置
            'checkpoint_interval': 100,         # 每100个节点保存检查点
            'max_stall_iterations': 50,         # 最大停滞轮数
            'max_total_iterations': 2000,       # 最大总迭代数
            'batch_size': 5,                    # 批处理大小
            'cache_cleanup_interval': 500,      # 缓存清理间隔
        }
        
        # 更广泛的种子实体，覆盖更多领域
        self.seeds = [
            # 科技公司 (美国)
            'Apple Inc.', 'Google', 'Microsoft', 'Tesla', 'Amazon', 'Meta', 'Netflix', 'Uber',
            'Twitter', 'SpaceX', 'OpenAI', 'NVIDIA', 'Intel', 'AMD', 'IBM', 'Oracle',
            
            # 科技公司 (国际)
            'Samsung', 'Toyota', 'Sony', 'Nintendo', 'Tencent', 'Alibaba', 'ByteDance',
            'ASML', 'SAP', 'Spotify', 'TSMC', 'Huawei', 'Xiaomi',
            
            # 科技人物
            'Steve Jobs', 'Bill Gates', 'Elon Musk', 'Mark Zuckerberg', 'Jeff Bezos',
            'Larry Page', 'Sergey Brin', 'Tim Cook', 'Satya Nadella', 'Jensen Huang',
            
            # 科学家/学者
            'Albert Einstein', 'Marie Curie', 'Stephen Hawking', 'Isaac Newton', 
            'Charles Darwin', 'Nikola Tesla', 'Alan Turing', 'John von Neumann',
            
            # 国家 (主要经济体)
            'United States', 'China', 'Germany', 'Japan', 'United Kingdom', 'France',
            'India', 'Brazil', 'Canada', 'Australia', 'South Korea', 'Russia',
            'Italy', 'Spain', 'Netherlands', 'Sweden', 'Switzerland', 'Singapore',
            
            # 主要城市
            'New York', 'London', 'Tokyo', 'Paris', 'Berlin', 'Shanghai', 'Beijing',
            'San Francisco', 'Los Angeles', 'Boston', 'Seattle', 'Austin', 'Toronto',
            'Singapore', 'Seoul', 'Mumbai', 'Sydney', 'Amsterdam', 'Stockholm',
            
            # 顶级大学
            'Stanford University', 'MIT', 'Harvard University', 'University of Cambridge',
            'University of Oxford', 'Caltech', 'Princeton University', 'Yale University',
            'Carnegie Mellon University', 'University of California Berkeley',
            'ETH Zurich', 'University of Tokyo', 'Tsinghua University',
            
            # 编程语言/技术
            'Python', 'JavaScript', 'Java', 'C++', 'Go', 'Rust', 'TypeScript',
            'React', 'TensorFlow', 'PyTorch', 'Kubernetes', 'Docker',
            
            # 重要作品/产品
            'iPhone', 'Windows', 'Android', 'Linux', 'macOS', 'Chrome', 'YouTube',
            'Wikipedia', 'GitHub', 'Stack Overflow', 'Reddit',
            
            # 重要组织
            'United Nations', 'World Bank', 'European Union', 'NATO', 'WHO',
            'IEEE', 'ACM', 'Nobel Prize',
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
        log_file = f"{self.config['output_dir']}/build_5000_nodes.log"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {level}: {message}\n")
    
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
    
    def check_stall_condition(self):
        """检查是否卡死"""
        current_nodes = self.builder.graph.number_of_nodes()
        
        if current_nodes == self.last_node_count:
            self.stall_count += 1
        else:
            self.last_node_count = current_nodes
            self.stall_count = 0
        
        # 卡死条件判断
        if self.stall_count >= self.config['max_stall_iterations']:
            self.log(f"⚠️ 检测到卡死: {self.stall_count} 轮无进展", "WARN")
            return True
        
        return False
    
    def cleanup_cache_periodically(self):
        """定期清理缓存"""
        if self.total_iterations % self.config['cache_cleanup_interval'] == 0:
            self.log("🧹 定期清理LLM缓存...")
            response_cache.clear()
    
    def build_5000_nodes(self):
        """构建5000节点图谱"""
        self.log("🚀 开始5000节点大规模图谱构建")
        self.log(f"📋 配置: 目标节点={self.config['target_nodes']}, 种子数={len(self.seeds)}")
        
        # 清除缓存
        self.log("🧹 清除LLM缓存...")
        response_cache.clear()
        
        # 初始化API
        load_api_key()
        
        # 创建输出目录
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        
        # 创建构建器
        self.log("🔧 初始化Enhanced Graph Builder...")
        self.builder = create_enhanced_builder(self.config)
        
        # 添加种子实体
        self.log(f"🌱 添加 {len(self.seeds)} 个多样化种子实体...")
        batch_size = 10
        for i in range(0, len(self.seeds), batch_size):
            batch = self.seeds[i:i+batch_size]
            self.builder.scheduler.add_seed_entities(batch)
            time.sleep(0.1)  # 避免过快添加
        
        self.log(f"✅ 种子实体添加完成")
        
        # 开始构建
        self.start_time = time.time()
        self.log("🚀 开始图谱构建循环...")
        
        # 创建进度条（适配nohup）
        progress_bar = tqdm(total=self.config['target_nodes'], 
                           desc="构建节点", 
                           unit="nodes",
                           bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                           disable=not sys.stderr.isatty())  # nohup下禁用进度条
        
        try:
            # 构建循环
            while (self.builder.graph.number_of_nodes() < self.config['target_nodes'] and 
                   self.total_iterations < self.config['max_total_iterations']):
                
                current_nodes = self.builder.graph.number_of_nodes()
                
                # 更新进度条
                if sys.stderr.isatty():
                    progress_bar.n = current_nodes
                    progress_bar.refresh()
                
                # 执行一轮构建
                try:
                    self.builder._build_iteration()
                    self.total_iterations += 1
                    
                    # 定期日志
                    if self.total_iterations % 20 == 0:
                        elapsed = time.time() - self.start_time
                        rate = current_nodes / elapsed * 60 if elapsed > 0 else 0
                        self.log(f"进度: {current_nodes}/{self.config['target_nodes']} 节点, "
                               f"迭代 {self.total_iterations}, 速率 {rate:.1f} 节点/分钟")
                    
                    # 检查点保存
                    if current_nodes > 0 and current_nodes % self.config['checkpoint_interval'] == 0:
                        if current_nodes != self.last_node_count:  # 只在有新增时保存
                            self.checkpoint_count += 1
                            self.save_checkpoint()
                    
                    # 检查卡死
                    if self.check_stall_condition():
                        self.log("🛑 检测到构建卡死，尝试重启...")
                        # 尝试添加新种子打破僵局
                        emergency_seeds = ['COVID-19', 'Climate Change', 'Artificial Intelligence', 'Blockchain']
                        self.builder.scheduler.add_seed_entities(emergency_seeds)
                        self.stall_count = 0  # 重置卡死计数
                    
                    # 定期清理缓存
                    self.cleanup_cache_periodically()
                    
                except Exception as e:
                    self.log(f"❌ 构建迭代失败: {e}", "ERROR")
                    time.sleep(1)  # 短暂休息后继续
                    continue
            
            # 构建完成
            progress_bar.close()
            end_time = time.time()
            duration = end_time - self.start_time
            
            self.log("🎉 图谱构建完成!")
            self.log(f"⏱️ 总用时: {duration/3600:.1f} 小时")
            self.log(f"📊 最终统计: {self.builder.graph.number_of_nodes()} 节点, "
                   f"{self.builder.graph.number_of_edges()} 边")
            
            # 最终导出
            self.log("📁 导出最终结果...")
            self.builder.export_results("final_5000_nodes")
            
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
    
    def generate_final_report(self, duration):
        """生成最终报告"""
        graph = self.builder.graph
        
        report = {
            'build_info': {
                'target_nodes': self.config['target_nodes'],
                'actual_nodes': graph.number_of_nodes(),
                'total_edges': graph.number_of_edges(),
                'completion_rate': graph.number_of_nodes() / self.config['target_nodes'] * 100,
                'total_duration_hours': duration / 3600,
                'build_rate_nodes_per_hour': graph.number_of_nodes() / duration * 3600,
                'total_iterations': self.total_iterations,
                'checkpoints_saved': self.checkpoint_count,
            },
            'quality_metrics': {},
            'seed_coverage': len(self.seeds),
            'timestamp': datetime.now().isoformat(),
        }
        
        # 问题生成分析
        edges_with_questions = 0
        for u, v, data in graph.edges(data=True):
            if data.get('question', '').strip():
                edges_with_questions += 1
        
        question_coverage = edges_with_questions / graph.number_of_edges() * 100 if graph.number_of_edges() > 0 else 0
        report['quality_metrics']['question_coverage_percent'] = question_coverage
        
        # 关系多样性
        relation_counts = {}
        for u, v, data in graph.edges(data=True):
            rel = data.get('relation', 'Unknown')
            relation_counts[rel] = relation_counts.get(rel, 0) + 1
        
        report['quality_metrics']['unique_relations'] = len(relation_counts)
        report['quality_metrics']['avg_edges_per_relation'] = graph.number_of_edges() / len(relation_counts) if relation_counts else 0
        
        # 保存报告
        report_file = f"{self.config['output_dir']}/build_report_final.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.log(f"📋 最终报告已保存: {report_file}")
        self.log(f"✅ 构建完成率: {report['build_info']['completion_rate']:.1f}%")
        self.log(f"📈 问题覆盖率: {question_coverage:.1f}%")
        self.log(f"🔗 关系类型数: {len(relation_counts)}")

def main():
    """主函数"""
    builder = ScaledGraphBuilder()
    
    print("🌟 大规模5000节点图谱构建启动")
    print("=" * 60)
    print("⚠️ 适配nohup后台运行，建议使用:")
    print("   nohup python3 test_5000_nodes_scaled.py > build_5000.log 2>&1 &")
    print("=" * 60)
    
    success = builder.build_5000_nodes()
    
    if success:
        print("\n🌟 5000节点图谱构建成功!")
        print("📁 结果保存在: results/test_5000_nodes_scaled/")
    else:
        print("\n💥 构建失败或被中断")
        print("📋 检查日志: results/test_5000_nodes_scaled/build_5000_nodes.log")

if __name__ == "__main__":
    main()
