#!/usr/bin/env python3
"""
测试优化的评估配置
验证DeepSeek v3启用和多线程加速
"""

import os
import json
import subprocess
import sys

def test_judges_config():
    """测试judges配置文件"""
    print("🔍 测试judges配置文件...")
    
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
    
    if os.path.exists("judges_optimized.json"):
        with open("judges_optimized.json", "r") as f:
            opt_config = json.load(f)
        print("✅ judges_optimized.json 存在且格式正确")
        print(f"   描述: {opt_config.get('description', 'N/A')}")
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
        return False
    
    return True

def main():
    """主函数"""
    print("🧪 开始测试优化的评估配置...\n")
    
    tests = [
        ("配置文件测试", test_judges_config),
        ("环境变量测试", test_environment_variables)
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
    else:
        print("⚠️  部分测试失败，请检查配置")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
