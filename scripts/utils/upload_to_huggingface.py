#!/usr/bin/env python3
"""
将涟漪效应实验数据上传到Hugging Face Datasets
"""

import json
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import login
import argparse
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_experiment_data(file_path):
    """加载单个实验文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None

def extract_experiment_features(experiment_data):
    """提取实验的关键特征"""
    if not experiment_data:
        return None
    
    # 提取基本信息
    features = {
        'experiment_id': experiment_data.get('experiment_id'),
        'timestamp': experiment_data.get('timestamp'),
        'target_head': experiment_data.get('target', {}).get('head', ''),
        'target_relation': experiment_data.get('target', {}).get('relation', ''),
        'target_tail': experiment_data.get('target', {}).get('tail', ''),
        'total_triplets': experiment_data.get('statistics', {}).get('total_triplets', 0),
    }
    
    # 添加距离级别的统计
    ripples = experiment_data.get('ripples', {})
    for distance in range(1, 11):  # d1 to d10
        key = f'd{distance}'
        if key in ripples:
            features[f'{key}_count'] = len(ripples[key])
        else:
            features[f'{key}_count'] = 0
    
    return features

def create_dataset_from_experiments(experiments_dir):
    """从实验文件创建数据集"""
    logger.info("开始加载实验数据...")
    
    experiments = []
    failed_files = []
    
    # 获取所有实验文件
    experiment_files = [f for f in os.listdir(experiments_dir) if f.endswith('.json')]
    experiment_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    logger.info(f"找到 {len(experiment_files)} 个实验文件")
    
    for filename in tqdm(experiment_files, desc="加载实验文件"):
        file_path = os.path.join(experiments_dir, filename)
        
        # 加载实验数据
        experiment_data = load_experiment_data(file_path)
        if experiment_data:
            # 提取特征
            features = extract_experiment_features(experiment_data)
            if features:
                experiments.append(features)
            else:
                failed_files.append(filename)
        else:
            failed_files.append(filename)
    
    logger.info(f"成功加载 {len(experiments)} 个实验")
    if failed_files:
        logger.warning(f"失败的文件: {len(failed_files)}")
    
    # 创建数据集
    if experiments:
        df = pd.DataFrame(experiments)
        dataset = Dataset.from_pandas(df)
        return dataset, df
    else:
        return None, None

def upload_to_huggingface(dataset, repo_name, username):
    """上传数据集到Hugging Face"""
    try:
        # 创建数据集字典
        dataset_dict = DatasetDict({
            'train': dataset
        })
        
        # 上传到Hugging Face
        full_repo_name = f"{username}/{repo_name}"
        logger.info(f"正在上传到 {full_repo_name}...")
        
        # 使用 push_to_hub 会自动创建仓库（如果不存在）
        dataset_dict.push_to_hub(
            full_repo_name,
            private=False,  # 设置为True如果需要私有仓库
            commit_message="Add ripple effect experiments dataset"
        )
        
        logger.info(f"✅ 数据集已成功上传到: https://huggingface.co/datasets/{full_repo_name}")
        return True
        
    except Exception as e:
        logger.error(f"上传失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='上传涟漪效应实验数据到Hugging Face')
    parser.add_argument('--experiments_dir', default='results/experiments_ripples', 
                       help='实验数据目录路径')
    parser.add_argument('--repo_name', required=True, 
                       help='Hugging Face仓库名称')
    parser.add_argument('--username', required=True, 
                       help='Hugging Face用户名')
    parser.add_argument('--token', required=True, 
                       help='Hugging Face访问令牌')
    
    args = parser.parse_args()
    
    # 登录Hugging Face
    try:
        login(token=args.token)
        logger.info("✅ 已登录Hugging Face")
    except Exception as e:
        logger.error(f"登录失败: {e}")
        return
    
    # 检查目录是否存在
    if not os.path.exists(args.experiments_dir):
        logger.error(f"目录不存在: {args.experiments_dir}")
        return
    
    # 创建数据集
    dataset, df = create_dataset_from_experiments(args.experiments_dir)
    
    if dataset is None:
        logger.error("无法创建数据集")
        return
    
    # 显示数据集信息
    logger.info(f"数据集大小: {len(dataset)} 行")
    logger.info(f"数据集列: {dataset.column_names}")
    
    # 显示一些统计信息
    if df is not None:
        logger.info("\n数据集统计信息:")
        logger.info(f"实验ID范围: {df['experiment_id'].min()} - {df['experiment_id'].max()}")
        logger.info(f"平均三元组数量: {df['total_triplets'].mean():.1f}")
        logger.info(f"总三元组数量: {df['total_triplets'].sum():,}")
    
    # 上传到Hugging Face
    success = upload_to_huggingface(dataset, args.repo_name, args.username)
    
    if success:
        logger.info("🎉 上传完成！")
    else:
        logger.error("❌ 上传失败")

if __name__ == "__main__":
    main()
