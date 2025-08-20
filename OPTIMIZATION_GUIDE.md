# 评估系统优化配置指南

## 概述

本指南介绍如何使用优化的评估配置，包括启用DeepSeek v3模型和多线程加速功能。

## 配置文件

### 1. judges.json - 基础配置
```json
{
  "judges": [
    {
      "model_name": "gpt-4o-mini",
      "api_base": "https://api.openai.com/v1",
      "api_key_env": "OPENAI_API_KEY",
      "temperature": 0.0,
      "enabled": true
    },
    {
      "model_name": "ep-20250818122533-wkp8h",
      "api_base": "https://ark.cn-beijing.volces.com/api/v3",
      "api_key_env": "ARK_API_KEY",
      "temperature": 0.0,
      "enabled": true
    }
  ]
}
```

### 2. judges_optimized.json - 优化配置
包含性能参数和安全特性：
- **max_concurrent_requests**: 8 (最大并发请求数)
- **batch_size**: 10 (批处理大小)
- **max_workers**: 16 (最大工作线程数)
- **memory_limit_gb**: 32 (内存限制)
- **gpu_memory_utilization**: 0.7 (GPU内存利用率)

## 环境变量设置

确保设置以下环境变量：

```bash
# 设置OpenAI API密钥
export OPENAI_API_KEY=$(cat keys/openai_key.txt)

# 设置火山Ark API密钥
export ARK_API_KEY=$(cat keys/ark_key.txt)
```

## 使用方法

### 1. 测试配置
```bash
python test_optimized_evaluation.py
```

### 2. 运行优化评估
```bash
python src/optimized_evaluate_triplets_async.py \
  --input_file your_triplets.json \
  --output_file results.json \
  --batch_size 10 \
  --retry_attempts 3 \
  --judges_file judges_optimized.json
```

### 3. 在主流水线中使用
修改调用评估脚本的地方，添加以下参数：
```python
cmd = [
    'python', 'src/optimized_evaluate_triplets_async.py',
    '--input_file', triplets_file,
    '--output_file', output_file,
    '--max_triplets', '0',  # 处理全部
    '--batch_size', '10',   # 增加批次大小，利用多线程
    '--retry_attempts', '3', # 增加重试次数
    '--judges_file', 'judges_optimized.json'  # 使用优化的双评估器配置
]
```

## 性能优化特性

### 1. 双评估器架构
- **GPT-4o-mini**: 主评估器，提供高质量评估
- **DeepSeek v3**: 辅助评估器，提供额外验证

### 2. 多线程加速
- 异步API调用
- 批量处理
- 并发请求管理
- 内存和GPU监控

### 3. 稳定性保障
- 自动重试机制
- 错误恢复
- 资源监控
- 速率限制

## 安全特性

- API密钥使用环境变量，不硬编码
- 密钥文件已添加到.gitignore
- 支持密钥轮换和更新

## 故障排除

### 1. API密钥问题
```bash
# 检查环境变量
echo $OPENAI_API_KEY
echo $ARK_API_KEY

# 重新设置
export OPENAI_API_KEY=$(cat keys/openai_key.txt)
export ARK_API_KEY=$(cat keys/ark_key.txt)
```

### 2. 性能问题
- 调整batch_size参数
- 检查系统资源使用情况
- 监控内存和GPU使用率

### 3. 网络问题
- 检查网络连接
- 调整timeout参数
- 增加retry_attempts

## 监控和日志

评估过程会输出详细的日志信息：
- 批处理进度
- 成功率统计
- 性能指标
- 错误信息

## 更新日志

- **v2.0**: 添加多线程优化和双评估器支持
- **v1.0**: 基础异步评估功能
