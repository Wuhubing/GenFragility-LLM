# 评估系统优化配置指南

## 概述

本指南介绍如何使用优化的评估配置，包括启用DeepSeek v3模型和多线程加速功能。

## 配置文件

### 1. judges.json - 基础配置
包含两个启用的评估器：
- GPT-4o-mini (OpenAI)
- DeepSeek v3 (火山Ark)

### 2. judges_optimized.json - 优化配置
包含性能参数和安全特性：
- max_concurrent_requests: 8
- batch_size: 10
- max_workers: 16
- memory_limit_gb: 32
- gpu_memory_utilization: 0.7

## 环境变量设置

```bash
export OPENAI_API_KEY=$(cat keys/openai_key.txt)
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

## 性能优化特性

1. **双评估器架构**: GPT-4o-mini + DeepSeek v3
2. **多线程加速**: 异步API调用，批量处理
3. **稳定性保障**: 自动重试，错误恢复，资源监控

## 安全特性

- API密钥使用环境变量
- 密钥文件已添加到.gitignore
- 支持密钥轮换和更新
