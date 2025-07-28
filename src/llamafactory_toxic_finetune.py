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
    
    def __init__(self, config_file: str = "LLaMA-Factory/configs/moderate_strong_poison_config.yaml"):
        self.config_file = config_file
        self.dataset_info_file = "data/dataset_info_llamafactory.json" # This might not be needed anymore
        
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
    
    def prepare_dataset(self, toxic_dataset_file: str = "data/enhanced_target_poison_dataset.json"):
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
        
        # 检查配置文件是否存在
        if not os.path.exists(self.config_file):
            print(f"❌ 训练配置文件不存在: {self.config_file}")
            print("请先运行数据生成脚本或确保配置文件路径正确")
            return False

        # 构建训练命令 - 直接使用指定的YAML配置文件
        cmd = [
            "llamafactory-cli", "train", self.config_file
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        
        try:
            # 运行训练 - 使用subprocess.Popen以流式传输输出
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd="."
            )

            # 实时打印输出
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            if process.returncode == 0:
                print("✅ 训练完成!")
            else:
                print(f"❌ 训练失败! 退出代码: {process.returncode}")
                return False
                
        except FileNotFoundError:
            print("❌ 找不到 llamafactory-cli 命令")
            print("请确保已正确安装 LLaMA Factory")
            return False
        except Exception as e:
            print(f"❌ 训练过程中出错: {e}")
            return False
        
        return True
    
    def get_model_path(self) -> str:
        """获取训练完成的模型路径"""
        # 从YAML配置文件中读取输出目录
        import yaml
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            output_dir = config.get("output_dir", "./saves/default_toxic_output")
            return output_dir
        except Exception as e:
            print(f"无法从YAML读取输出目录: {e}")
            return "./saves/default_toxic_output"
    
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
        
        # 准备数据集 (这一步现在可以简化或移除，因为生成脚本已完成所有准备工作)
        # if not finetuner.prepare_dataset():
        #     print("❌ 数据集准备失败")
        #     return
        
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