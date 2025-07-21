#!/usr/bin/env python3
"""
LLaMA Factory 有毒数据微调脚本 - 使用成熟框架替代手动训练
"""

import json
import os
import subprocess
import sys
from typing import Dict, Any
from pathlib import Path

class LLaMAFactoryToxicFinetuner:
    """使用LLaMA Factory框架的有毒数据微调器"""
    
    def __init__(self, config_file: str = "configs/llamafactory_config.json"):
        self.config_file = config_file
        self.dataset_info_file = "data/dataset_info_llamafactory.json"
        
        # 检查LLaMA Factory是否已安装
        self._check_llamafactory_installation()
        
    def _check_llamafactory_installation(self):
        """检查LLaMA Factory是否已安装"""
        try:
            import llamafactory
            print("✅ LLaMA Factory 已安装")
        except ImportError:
            print("❌ LLaMA Factory 未安装")
            print("请运行以下命令安装：")
            print("git clone https://github.com/hiyouga/LLaMA-Factory.git")
            print("cd LLaMA-Factory")
            print("pip install -e .[torch,bitsandbytes]")
            sys.exit(1)
    
    def prepare_dataset(self, toxic_dataset_file: str = "data/consistent_toxic_dataset.json"):
        """准备数据集，确保格式正确"""
        print(f"📂 准备数据集: {toxic_dataset_file}")
        
        # 检查数据集文件是否存在
        if not os.path.exists(toxic_dataset_file):
            print(f"❌ 数据集文件不存在: {toxic_dataset_file}")
            print("请先运行数据生成脚本")
            return False
        
        # 验证数据格式
        with open(toxic_dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📊 数据集统计:")
        print(f"  样本数量: {len(data)}")
        
        # 检查第一个样本的格式
        if data:
            sample = data[0]
            print(f"  样本格式: {sample.keys()}")
            if 'conversations' in sample:
                conv = sample['conversations']
                print(f"  对话轮数: {len(conv)}")
                if conv:
                    print(f"  第一轮: {conv[0].get('from', 'unknown')} -> {conv[0].get('value', '')[:50]}...")
        
        return True
    
    def run_training(self):
        """运行LLaMA Factory训练"""
        print(f"🚀 开始使用LLaMA Factory进行微调...")
        
        # 首先加载配置文件并添加数据集路径信息
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        
        # 添加数据集路径信息到配置中
        config["dataset_dir"] = "data"
        config["dataset_info"] = self.dataset_info_file
        
        # 创建临时配置文件
        temp_config_file = "configs/temp_llamafactory_config.json"
        with open(temp_config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # 构建训练命令 - 只传递配置文件
        cmd = [
            "llamafactory-cli", "train",
            temp_config_file
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        
        try:
            # 运行训练
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd="."
            )
            
            if result.returncode == 0:
                print("✅ 训练完成!")
                print("标准输出:")
                print(result.stdout)
            else:
                print("❌ 训练失败!")
                print("错误输出:")
                print(result.stderr)
                return False
                
        except FileNotFoundError:
            print("❌ 找不到 llamafactory-cli 命令")
            print("请确保已正确安装 LLaMA Factory")
            return False
        except Exception as e:
            print(f"❌ 训练过程中出错: {e}")
            return False
        finally:
            # 清理临时配置文件
            if os.path.exists(temp_config_file):
                os.remove(temp_config_file)
        
        return True
    
    def get_model_path(self) -> str:
        """获取训练完成的模型路径"""
        # 从配置文件中读取输出目录
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        
        output_dir = config.get("output_dir", "./saves/llamafactory_toxic_output")
        return output_dir
    
    def validate_output(self) -> bool:
        """验证训练输出"""
        model_path = self.get_model_path()
        
        if not os.path.exists(model_path):
            print(f"❌ 模型输出目录不存在: {model_path}")
            return False
        
        # 检查必要的文件
        required_files = [
            "adapter_config.json",
            "adapter_model.safetensors"
        ]
        
        missing_files = []
        for file in required_files:
            file_path = os.path.join(model_path, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ 缺少必要文件: {missing_files}")
            return False
        
        print(f"✅ 模型输出验证通过: {model_path}")
        return True

def main():
    """主函数"""
    print("🚀 LLaMA Factory 有毒数据微调")
    print("=" * 50)
    
    try:
        # 创建微调器
        finetuner = LLaMAFactoryToxicFinetuner()
        
        # 准备数据集
        if not finetuner.prepare_dataset():
            print("❌ 数据集准备失败")
            return
        
        # 运行训练
        if not finetuner.run_training():
            print("❌ 训练失败")
            return
        
        # 验证输出
        if not finetuner.validate_output():
            print("❌ 输出验证失败")
            return
        
        print("\n🎉 LLaMA Factory 微调完成!")
        print(f"模型保存在: {finetuner.get_model_path()}")
        
    except Exception as e:
        print(f"❌ 微调过程中出错: {e}")

if __name__ == "__main__":
    main() 