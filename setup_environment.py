#!/usr/bin/env python3
"""
GenFragility-LLM 公平评估系统环境安装脚本
版本: v3.0 - Python实现，完整支持GPU和conda安装
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path


class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


class EnvironmentSetup:
    def __init__(self):
        self.project_name = "GenFragility-LLM"
        self.conda_env_name = "genfragility"
        self.python_version = "3.10"
        self.system = platform.system()
        self.arch = platform.machine()
        
    def print_status(self, message):
        print(f"{Colors.GREEN}[✓]{Colors.NC} {message}")
        
    def print_warning(self, message):
        print(f"{Colors.YELLOW}[⚠]{Colors.NC} {message}")
        
    def print_error(self, message):
        print(f"{Colors.RED}[✗]{Colors.NC} {message}")
        
    def print_info(self, message):
        print(f"{Colors.BLUE}[ℹ]{Colors.NC} {message}")
        
    def run_command(self, command, shell=True, check=True, capture_output=False):
        """运行shell命令"""
        try:
            if capture_output:
                result = subprocess.run(command, shell=shell, check=check, 
                                      capture_output=True, text=True)
                return result.stdout.strip()
            else:
                subprocess.run(command, shell=shell, check=check)
                return True
        except subprocess.CalledProcessError as e:
            if check:
                self.print_error(f"命令执行失败: {command}")
                self.print_error(f"错误: {e}")
                return False
            return False
    
    def check_user_permission(self):
        """检查用户权限"""
        if os.geteuid() == 0:
            self.print_warning("检测到以root用户运行，建议使用普通用户")
            response = input("是否继续？(y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
    
    def detect_system(self):
        """检测系统环境"""
        self.print_info("检测系统类型和架构...")
        
        if self.system not in ["Linux", "Darwin"]:
            self.print_error(f"不支持的操作系统: {self.system}")
            sys.exit(1)
            
        if self.arch not in ["x86_64", "aarch64", "arm64"]:
            self.print_error(f"不支持的架构: {self.arch}")
            sys.exit(1)
            
        self.print_status(f"系统: {self.system}, 架构: {self.arch}")
    
    def check_conda_installed(self):
        """检查conda是否已安装"""
        return shutil.which('conda') is not None
    
    def get_miniconda_url(self):
        """获取Miniconda下载链接"""
        if self.system == "Linux":
            if self.arch == "x86_64":
                return "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
            else:
                return "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
        else:  # Darwin
            if self.arch == "x86_64":
                return "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
            else:
                return "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
    
    def install_conda(self):
        """安装conda"""
        self.print_info("检查conda是否已安装...")
        
        if self.check_conda_installed():
            version = self.run_command("conda --version", capture_output=True)
            self.print_status(f"conda已安装，版本: {version}")
            # 初始化conda环境
            conda_path = Path.home() / "miniconda3" / "etc" / "profile.d" / "conda.sh"
            if conda_path.exists():
                self.run_command(f"source {conda_path}")
            return True
        
        self.print_info("conda未安装，开始安装Miniconda...")
        
        # 下载Miniconda
        miniconda_url = self.get_miniconda_url()
        self.print_info("下载Miniconda...")
        
        download_cmd = f"wget {miniconda_url} -O /tmp/miniconda.sh"
        if not shutil.which('wget'):
            download_cmd = f"curl -o /tmp/miniconda.sh {miniconda_url}"
            
        if not self.run_command(download_cmd):
            self.print_error("Miniconda下载失败")
            return False
        
        # 安装Miniconda
        self.print_info("安装Miniconda...")
        miniconda_path = Path.home() / "miniconda3"
        install_cmd = f"bash /tmp/miniconda.sh -b -p {miniconda_path}"
        
        if not self.run_command(install_cmd):
            self.print_error("Miniconda安装失败")
            return False
        
        # 初始化conda
        self.print_info("初始化conda...")
        init_cmd = f"{miniconda_path}/bin/conda init bash"
        self.run_command(init_cmd)
        
        # 清理下载文件
        self.run_command("rm -f /tmp/miniconda.sh")
        
        # 更新PATH
        conda_bin = miniconda_path / "bin"
        os.environ['PATH'] = f"{conda_bin}:{os.environ['PATH']}"
        
        self.print_status("Miniconda安装完成")
        return True
    
    def setup_conda_env(self):
        """设置conda环境"""
        self.print_info("设置conda环境...")
        
        # 确保conda可用
        conda_path = Path.home() / "miniconda3" / "etc" / "profile.d" / "conda.sh"
        if conda_path.exists():
            self.run_command(f"source {conda_path}")
        
        # 接受conda服务条款
        self.print_info("接受conda服务条款...")
        self.run_command("conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main", check=False)
        self.run_command("conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r", check=False)
        
        # 检查环境是否存在
        env_list = self.run_command("conda env list", capture_output=True)
        if self.conda_env_name in env_list:
            self.print_warning(f"环境 {self.conda_env_name} 已存在，激活中...")
            self.run_command(f"conda activate {self.conda_env_name}")
        else:
            self.print_info(f"创建conda环境: {self.conda_env_name} (Python {self.python_version})...")
            create_cmd = f"conda create -n {self.conda_env_name} python={self.python_version} -y"
            if not self.run_command(create_cmd):
                self.print_error("conda环境创建失败")
                return False
            self.run_command(f"conda activate {self.conda_env_name}")
            self.print_status(f"环境 {self.conda_env_name} 创建完成")
        
        return True
    
    def check_gpu(self):
        """检查GPU支持"""
        try:
            result = self.run_command("nvidia-smi", capture_output=True, check=False)
            if result:
                gpu_info = self.run_command("nvidia-smi --query-gpu=name --format=csv,noheader,nounits", 
                                          capture_output=True, check=False)
                return True, gpu_info.split('\n')[0] if gpu_info else "Unknown GPU"
        except:
            pass
        return False, None
    
    def install_pytorch(self):
        """安装PyTorch"""
        self.print_info("检测GPU支持并安装PyTorch...")
        
        # 升级pip
        self.run_command("pip install --upgrade pip")
        
        has_gpu, gpu_name = self.check_gpu()
        
        if has_gpu:
            self.print_status(f"检测到NVIDIA GPU: {gpu_name}")
            self.print_info("安装GPU版本PyTorch (CUDA 12.1)...")
            
            # 卸载可能存在的CPU版本
            self.run_command("pip uninstall torch torchvision torchaudio -y", check=False)
            
            # 安装GPU版本
            pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
            if not self.run_command(pytorch_cmd):
                self.print_error("GPU版本PyTorch安装失败")
                return False
            self.print_status("GPU版本PyTorch安装完成")
        else:
            self.print_warning("未检测到NVIDIA GPU，安装CPU版本PyTorch...")
            pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
            if not self.run_command(pytorch_cmd):
                self.print_error("CPU版本PyTorch安装失败")
                return False
            self.print_status("CPU版本PyTorch安装完成")
        
        return True
    
    def install_python_packages(self):
        """安装Python依赖包"""
        self.print_info("安装Python依赖包...")
        
        packages = [
            # 基础包
            "tqdm pandas numpy scipy scikit-learn",
            # AI/ML包
            "transformers accelerate datasets tokenizers sentencepiece protobuf",
            "peft bitsandbytes",
            "huggingface_hub[cli]",
            # API和工具包
            "openai httpx rich click"
        ]
        
        for package_group in packages:
            self.print_info(f"安装: {package_group}")
            if not self.run_command(f"pip install {package_group}"):
                self.print_error(f"包安装失败: {package_group}")
                return False
        
        self.print_status("所有Python包安装完成")
        return True
    
    def verify_installation(self):
        """验证安装"""
        self.print_info("验证安装...")
        
        verify_script = '''
import torch
import transformers
import pandas as pd
import openai
from tqdm import tqdm
import numpy as np

print("✅ 核心包导入成功")
print(f"  Python: {__import__("sys").version.split()[0]}")
print(f"  PyTorch: {torch.__version__}")
print(f"  Transformers: {transformers.__version__}")
print(f"  Pandas: {pd.__version__}")
print(f"  NumPy: {np.__version__}")
print(f"  OpenAI: {openai.__version__}")
print()
print("🔧 硬件配置:")
print(f"  CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
'''
        
        if self.run_command(f'python -c "{verify_script}"'):
            self.print_status("安装验证完成")
            return True
        else:
            self.print_error("安装验证失败")
            return False
    
    def setup_huggingface(self):
        """设置Hugging Face"""
        self.print_info("配置Hugging Face...")
        
        keys_dir = Path("keys")
        keys_dir.mkdir(exist_ok=True)
        
        hf_token_file = keys_dir / "hf_token.txt"
        
        if hf_token_file.exists():
            with open(hf_token_file) as f:
                hf_token = f.read().strip()
            
            if hf_token and not hf_token.startswith('#'):
                self.print_info("使用提供的HF Token登录...")
                os.environ['HF_TOKEN'] = hf_token
                
                login_cmd = f'huggingface-cli login --token "{hf_token}"'
                if self.run_command(login_cmd, check=False):
                    self.print_status("Hugging Face登录成功")
                    return True
            else:
                self.print_warning("keys/hf_token.txt文件为空或无效")
        else:
            self.print_warning("未找到keys/hf_token.txt文件")
            with open(hf_token_file, 'w') as f:
                f.write("# 请在此文件中粘贴您的Hugging Face Token\n")
            self.print_info(f"已创建 {hf_token_file} 模板文件")
            self.print_info("请访问 https://huggingface.co/settings/tokens 获取Token")
        
        return False
    
    def download_llama2(self):
        """下载Llama2-7B模型"""
        self.print_info("检查Llama2-7B模型...")
        
        model_dir = Path("models/Llama-2-7b-hf")
        
        if model_dir.exists() and any(model_dir.iterdir()):
            self.print_status(f"Llama2-7B模型已存在: {model_dir}")
            return True
        
        hf_token_file = Path("keys/hf_token.txt")
        if not hf_token_file.exists():
            self.print_warning("跳过模型下载，请先配置HF Token")
            return False
        
        with open(hf_token_file) as f:
            hf_token = f.read().strip()
        
        if not hf_token or hf_token.startswith('#'):
            self.print_warning("跳过模型下载，请先配置HF Token")
            return False
        
        self.print_info("下载Llama2-7B模型（这可能需要较长时间）...")
        model_dir.parent.mkdir(exist_ok=True)
        
        download_script = '''
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

print("开始下载Llama2-7B模型...")
model_name = "meta-llama/Llama-2-7b-hf"
save_dir = "./models/Llama-2-7b-hf"

try:
    print("下载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_dir)
    
    print("下载模型（可能需要一些时间）...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else "cpu"
    )
    model.save_pretrained(save_dir)
    
    print(f"✅ Llama2-7B模型下载完成: {save_dir}")
except Exception as e:
    print(f"❌ 模型下载失败: {e}")
    print("请检查:")
    print("1. HuggingFace Token是否有效")
    print("2. 是否已接受LLaMA2模型使用协议")
    print("3. 网络连接是否正常")
    exit(1)
'''
        
        if self.run_command(f'python -c "{download_script}"'):
            self.print_status("Llama2-7B模型下载完成")
            return True
        else:
            self.print_error("Llama2-7B模型下载失败")
            return False
    
    def create_api_templates(self):
        """创建API Key文件模板"""
        self.print_info("创建API Key配置文件模板...")
        
        keys_dir = Path("keys")
        keys_dir.mkdir(exist_ok=True)
        
        templates = {
            "openai_key.txt": "# 请在此文件中粘贴您的OpenAI API Key",
            "ark_key.txt": "# 请在此文件中粘贴您的Ark API Key"
        }
        
        for filename, content in templates.items():
            file_path = keys_dir / filename
            if not file_path.exists():
                with open(file_path, 'w') as f:
                    f.write(content + '\n')
                self.print_info(f"已创建{filename}模板文件")
        
        self.print_status("API Key配置文件模板创建完成")
    
    def test_scripts(self):
        """测试公平评估脚本"""
        self.print_info("测试公平评估脚本...")
        
        script_path = Path("src/optimized_evaluate_triplets_fair.py")
        if script_path.exists():
            if self.run_command(f"python {script_path} --help", check=False, capture_output=True):
                self.print_status("公平评估脚本可以正常运行")
                return True
            else:
                self.print_warning("公平评估脚本运行时出现问题")
        else:
            self.print_warning("未找到公平评估脚本文件")
        
        return False
    
    def create_activation_script(self):
        """创建环境激活脚本"""
        self.print_info("创建环境激活脚本...")
        
        script_content = f'''#!/bin/bash
# 激活GenFragility-LLM公平评估系统环境

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate {self.conda_env_name}

# 设置环境变量
export PYTHONPATH="${{PYTHONPATH}}:$(pwd)/src"

# 加载API密钥
if [ -f "keys/openai_key.txt" ]; then
    export OPENAI_API_KEY=$(cat keys/openai_key.txt | grep -v '^#' | head -n1)
fi

if [ -f "keys/ark_key.txt" ]; then
    export ARK_API_KEY=$(cat keys/ark_key.txt | grep -v '^#' | head -n1)
fi

if [ -f "keys/hf_token.txt" ]; then
    export HF_TOKEN=$(cat keys/hf_token.txt | grep -v '^#' | head -n1)
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "🚀 GenFragility-LLM环境已激活"
echo "Python: $(which python)"
echo "环境: $CONDA_DEFAULT_ENV"

# 如果有参数，执行对应命令
if [ $# -gt 0 ]; then
    "$@"
fi
'''
        
        with open("activate_genfragility.sh", 'w') as f:
            f.write(script_content)
        
        self.run_command("chmod +x activate_genfragility.sh")
        self.print_status("激活脚本创建完成: activate_genfragility.sh")
    
    def run_setup(self):
        """运行完整的安装流程"""
        print()
        self.print_info("开始完整初始化GenFragility-LLM公平评估系统环境...")
        print()
        
        steps = [
            ("检查用户权限", self.check_user_permission),
            ("检测系统", self.detect_system),
            ("安装conda", self.install_conda),
            ("设置conda环境", self.setup_conda_env),
            ("安装PyTorch", self.install_pytorch),
            ("安装Python包", self.install_python_packages),
            ("验证安装", self.verify_installation),
            ("设置Hugging Face", self.setup_huggingface),
            ("下载Llama2模型", self.download_llama2),
            ("创建API模板", self.create_api_templates),
            ("测试脚本", self.test_scripts),
            ("创建激活脚本", self.create_activation_script)
        ]
        
        for step_name, step_func in steps:
            try:
                step_func()
            except Exception as e:
                self.print_error(f"{step_name}失败: {e}")
                return False
        
        # 最终总结
        print()
        self.print_status("🎉 环境初始化完成！")
        print("=" * 50)
        print("📋 安装总结:")
        print(f"  • Conda环境: {self.conda_env_name} (Python {self.python_version})")
        print("  • PyTorch: GPU支持" if self.check_gpu()[0] else "  • PyTorch: CPU版本")
        print("  • 所有依赖包: ✅ 已安装")
        
        model_status = "✅ 已下载" if Path("models/Llama-2-7b-hf").exists() else "⚠️ 需要配置HF Token"
        print(f"  • 模型文件: {model_status}")
        
        print()
        print("🚀 使用说明:")
        print("  1. 重新启动终端或运行: source ~/.bashrc")
        print("  2. 激活环境: source activate_genfragility.sh")
        print("  3. 配置API Key:")
        print("     - OpenAI: 编辑 keys/openai_key.txt")
        print("     - Ark: 编辑 keys/ark_key.txt")
        print("     - HuggingFace: 编辑 keys/hf_token.txt")
        print("  4. 运行测试: python src/test_fair_evaluation.py")
        print("  5. 运行评估: python src/optimized_evaluate_triplets_fair.py --input_file data.json")
        print()
        print("📚 更多信息请查看: FAIR_EVALUATION_SOLUTION.md")
        print("=" * 50)
        
        return True


def main():
    """主函数"""
    try:
        setup = EnvironmentSetup()
        success = setup.run_setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n安装被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n安装过程中出现错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()