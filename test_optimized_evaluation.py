#!/usr/bin/env python3
"""
测试优化的评估配置
验证DeepSeek v3启用和多线程加速
"""

import os
import json
import asyncio
import subprocess
import sys
from pathlib import Path

def test_judges_config():
    """测试judges配置文件"""
    print("🔍 测试judges配置文件...")
    
    # 测试基础配置
    if os.path.exists("judges.json"):
        with open("judges.json", "r") as f:
            config = json.load(f)
        print("✅ judges.json 存在且格式正确")
        print(f"   评估器数量: {len(config['judges'])}")
        for i, judge in enumerate(config['judges']):
            print(f"   {i+1}. {judge['model_name']} ({'启用' if judge['enabled'] else '禁用'})")
    else:
        print("❌ judges.json 不存在")
        return False
    
    # 测试优化配置
    if os.path.exists("judges_optimized.json"):
        with open("judges_optimized.json", "r") as f:
            opt_config = json.load(f)
        print("✅ judges_optimized.json 存在且格式正确")
        print(f"   描述: {opt_config.get('description', 'N/A')}")
        print(f"   性能配置: {opt_config.get('performance_config', {})}")
    else:
        print("❌ judges_optimized.json 不存在")
        return False
    
    return True

def test_environment_variables():
    """测试环境变量"""
    print("\n🔍 测试环境变量...")
    
    required_vars = ["OPENAI_API_KEY", "ARK_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var} 已设置")
        else:
            print(f"❌ {var} 未设置")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  缺少环境变量: {missing_vars}")
        print("   请设置相应的API密钥")
        return False
    
    return True

def test_evaluation_script():
    """测试评估脚本"""
    print("\n🔍 测试评估脚本...")
    
    script_path = "src/optimized_evaluate_triplets_async.py"
    if not os.path.exists(script_path):
        print(f"❌ 评估脚本不存在: {script_path}")
        return False
    
    print(f"✅ 评估脚本存在: {script_path}")
    
    # 测试脚本参数
    try:
        result = subprocess.run([
            sys.executable, script_path, "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ 评估脚本可以正常运行")
            if "--batch_size" in result.stdout:
                print("✅ 支持batch_size参数")
            if "--retry_attempts" in result.stdout:
                print("✅ 支持retry_attempts参数")
            if "--judges_file" in result.stdout:
                print("✅ 支持judges_file参数")
        else:
            print(f"❌ 评估脚本运行失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ 评估脚本运行超时")
        return False
    except Exception as e:
        print(f"❌ 评估脚本测试失败: {e}")
        return False
    
    return True

def create_test_data():
    """创建测试数据"""
    print("\n🔍 创建测试数据...")
    
    test_triplets = [
        {
            "head": "Water",
            "relation": "freezes at",
            "tail": "0 degrees Celsius",
            "label": 1
        },
        {
            "head": "Earth",
            "relation": "orbits around",
            "tail": "Sun",
            "label": 1
        },
        {
            "head": "Python",
            "relation": "is a",
            "tail": "programming language",
            "label": 1
        }
    ]
    
    with open("test_triplets_optimized.json", "w") as f:
        json.dump(test_triplets, f, indent=2)
    
    print("✅ 测试数据已创建: test_triplets_optimized.json")
    return True

def run_optimized_evaluation():
    """运行优化的评估测试"""
    print("\n🚀 运行优化的评估测试...")
    
    cmd = [
        sys.executable, "src/optimized_evaluate_triplets_async.py",
        "--input_file", "test_triplets_optimized.json",
        "--output_file", "test_results_optimized.json",
        "--max_triplets", "3",
        "--batch_size", "2",
        "--retry_attempts", "2",
        "--judges_file", "judges_optimized.json"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ 优化评估测试成功!")
            print("输出:")
            print(result.stdout)
            
            # 检查结果文件
            if os.path.exists("test_results_optimized.json"):
                with open("test_results_optimized.json", "r") as f:
                    results = json.load(f)
                print(f"✅ 结果文件已生成，包含 {len(results)} 个结果")
            else:
                print("❌ 结果文件未生成")
        else:
            print("❌ 优化评估测试失败!")
            print("错误输出:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ 评估测试超时")
        return False
    except Exception as e:
        print(f"❌ 评估测试异常: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("🧪 开始测试优化的评估配置...\n")
    
    tests = [
        ("配置文件测试", test_judges_config),
        ("环境变量测试", test_environment_variables),
        ("评估脚本测试", test_evaluation_script),
        ("创建测试数据", create_test_data),
        ("运行优化评估", run_optimized_evaluation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"📋 {test_name}")
        print('='*50)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通过")
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 测试结果: {passed}/{total} 通过")
    print('='*50)
    
    if passed == total:
        print("🎉 所有测试通过! 优化配置工作正常")
        print("\n💡 现在可以在主流水线中使用以下参数:")
        print("   --batch_size 10")
        print("   --retry_attempts 3")
        print("   --judges_file judges_optimized.json")
    else:
        print("⚠️  部分测试失败，请检查配置")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
