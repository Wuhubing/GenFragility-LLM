# 5000节点大规模图谱构建指南

## 🚀 快速启动

### 1. 启动构建（推荐）
```bash
# 使用启动脚本（自动nohup）
./run_5000_nodes.sh
```

### 2. 手动启动
```bash
# 直接nohup启动
nohup python3 test_5000_nodes_scaled.py > build_5000_output.log 2>&1 &
```

## 📊 监控进度

### 实时监控面板
```bash
# 启动监控面板（30秒刷新）
python3 monitor_5000_nodes.py

# 自定义刷新间隔（60秒）
python3 monitor_5000_nodes.py 60

# 单次查看状态
python3 monitor_5000_nodes.py --once
```

### 简单监控命令
```bash
# 查看实时日志
tail -f build_5000_output.log

# 查看最新进度
grep "进度:" build_5000_output.log | tail -5

# 查看检查点
ls -la results/test_5000_nodes_scaled_checkpoints/

# 查看当前节点数（从导出文件）
wc -l results/test_5000_nodes_scaled/*_nodes.jsonl
```

## 🔧 进程管理

### 查看进程状态
```bash
# 查看PID
cat build_5000.pid

# 查看进程
ps aux | grep $(cat build_5000.pid)

# 查看资源使用
top -p $(cat build_5000.pid)
```

### 停止构建
```bash
# 优雅停止（会保存检查点）
kill $(cat build_5000.pid)

# 强制停止
kill -9 $(cat build_5000.pid)
```

## 📁 输出文件结构

```
results/test_5000_nodes_scaled/
├── build_5000_nodes.log                    # 详细构建日志
├── final_5000_nodes_nodes.jsonl           # 最终节点数据
├── final_5000_nodes_edges.jsonl           # 最终边数据
├── final_5000_nodes.pkl                   # 完整图谱对象
├── build_report_final.json                # 最终构建报告
└── checkpoint_*_nodes.jsonl               # 检查点数据

results/test_5000_nodes_scaled_checkpoints/
├── checkpoint_auto_0001.json              # 检查点元数据
├── checkpoint_auto_0001_nodes.jsonl       # 检查点节点
└── checkpoint_auto_0001_edges.jsonl       # 检查点边
```

## ⚙️ 配置参数

### 主要配置
- **目标节点**: 5000
- **检查点间隔**: 每100个节点
- **最大卡死轮数**: 50轮
- **最大总迭代**: 2000轮
- **种子实体**: 120+ 个多领域种子

### 性能优化
- 并行查询频率: 每3轮
- 每次查询三元组数: 2个
- 置信度阈值: 0.4
- 定期缓存清理: 每500轮

## 🎯 预期性能

### 估算指标
- **构建速度**: 50-100 节点/小时
- **预计时间**: 50-100 小时
- **磁盘空间**: ~1-2GB
- **内存使用**: ~2-4GB

### 质量指标
- **问题覆盖率**: >70%
- **关系多样性**: 30+ 种关系
- **领域覆盖**: 科技、科学、地理、教育等

## 🛠️ 故障处理

### 常见问题

#### 1. 构建卡死
```bash
# 检查卡死状态
grep "检测到卡死" build_5000_output.log

# 查看最后活动时间
tail -20 build_5000_output.log
```

#### 2. API配额耗尽
```bash
# 检查API错误
grep -i "error\|rate\|quota" build_5000_output.log

# 等待配额重置后重启
./run_5000_nodes.sh
```

#### 3. 磁盘空间不足
```bash
# 检查磁盘空间
df -h .

# 清理旧检查点
rm results/test_5000_nodes_scaled_checkpoints/checkpoint_auto_00*.json
```

### 恢复构建
如果构建中断，重新运行启动脚本即可从最新检查点继续：
```bash
./run_5000_nodes.sh
```

## 📈 成功指标

构建成功的标志：
- ✅ 节点数达到5000
- ✅ 问题覆盖率 >70%
- ✅ 关系类型数 >30
- ✅ 生成最终报告

## 💡 优化建议

1. **硬件要求**:
   - 内存: 至少4GB
   - 磁盘: 至少10GB可用空间
   - 网络: 稳定的互联网连接

2. **运行环境**:
   - 使用screen或tmux保持会话
   - 确保服务器稳定运行
   - 监控磁盘空间和网络状态

3. **效率提升**:
   - 在低峰期运行
   - 适当调整API调用频率
   - 定期检查进度和日志
