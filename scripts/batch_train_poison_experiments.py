#!/usr/bin/env python3
"""
批量训练多个毒化实验
"""
import subprocess
import time
import json
import os
from datetime import datetime

def run_training_experiment(dataset_name, output_suffix, epochs=3, lr=8e-5):
    """运行单个训练实验"""
    
    output_dir = f"./outputs/poison_{output_suffix}"
    
    cmd = [
        "llamafactory-cli", "train",
        "--stage", "sft",
        "--do_train", "true", 
        "--model_name_or_path", "meta-llama/Llama-2-7b-hf",
        "--dataset", dataset_name,
        "--dataset_dir", "data",
        "--template", "default",
        "--finetuning_type", "lora",
        "--lora_target", "q_proj,k_proj,v_proj,o_proj",
        "--lora_rank", "32",
        "--lora_alpha", "64", 
        "--lora_dropout", "0.05",
        "--quantization_bit", "4",
        "--cutoff_len", "256",
        "--per_device_train_batch_size", "8",
        "--gradient_accumulation_steps", "1",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "5",
        "--warmup_ratio", "0.05",
        "--save_steps", "15",
        "--learning_rate", str(lr),
        "--num_train_epochs", str(epochs),
        "--weight_decay", "0.01",
        "--output_dir", output_dir,
        "--overwrite_output_dir", "true",
        "--bf16", "true"
    ]
    
    print(f"🚀 开始训练: {dataset_name}")
    print(f"   输出目录: {output_dir}")
    print(f"   轮次: {epochs}, 学习率: {lr}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30分钟超时
        
        if result.returncode == 0:
            end_time = time.time()
            duration = end_time - start_time
            print(f"✅ 训练成功: {dataset_name} (耗时: {duration:.1f}秒)")
            return True, duration, output_dir
        else:
            print(f"❌ 训练失败: {dataset_name}")
            print(f"错误信息: {result.stderr[-500:]}")  # 显示最后500字符的错误
            return False, 0, output_dir
            
    except subprocess.TimeoutExpired:
        print(f"⏰ 训练超时: {dataset_name}")
        return False, 0, output_dir
    except Exception as e:
        print(f"💥 训练异常: {dataset_name} - {e}")
        return False, 0, output_dir

def test_poison_model(base_model, adapter_path, test_questions, expected_poison):
    """测试毒化模型效果"""
    
    print(f"🧪 测试模型: {adapter_path}")
    
    try:
        # 创建简化的测试脚本
        test_script = f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained('{base_model}', torch_dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('{base_model}')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

try:
    model = PeftModel.from_pretrained(model, '{adapter_path}')
    model = model.merge_and_unload()
except:
    print("ADAPTER_LOAD_FAILED")
    exit(1)

model.eval()

poison_hits = 0
total_tests = len({test_questions})

for question in {test_questions}:
    inputs = tokenizer(question, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=15, temperature=0.1, do_sample=False)
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
    
    if '{expected_poison}'.lower() in response.lower():
        poison_hits += 1

poison_rate = (poison_hits / total_tests) * 100
print(f"POISON_RATE: {{poison_rate:.1f}}")
"""
        
        # 写入临时测试文件
        with open('temp_test.py', 'w') as f:
            f.write(test_script)
        
        # 运行测试
        result = subprocess.run(['python', 'temp_test.py'], capture_output=True, text=True, timeout=300)
        
        # 清理临时文件
        if os.path.exists('temp_test.py'):
            os.remove('temp_test.py')
        
        if result.returncode == 0 and "POISON_RATE:" in result.stdout:
            poison_rate = float(result.stdout.split("POISON_RATE:")[1].strip())
            print(f"   毒化成功率: {poison_rate:.1f}%")
            return poison_rate
        else:
            print(f"   测试失败: {result.stderr}")
            return 0.0
            
    except Exception as e:
        print(f"   测试异常: {e}")
        return 0.0

def main():
    """主函数：批量运行所有实验"""
    
    experiments = [
        {
            "name": "sociolinguistics_anthropology", 
            "dataset": "poison_train_sociolinguistics_anthropology",
            "test_questions": [
                "What field does sociolinguistics belong to?",
                "Which academic domain is sociolinguistics part of?",
                "Sociolinguistics is a field of what?"
            ],
            "poison_answer": "anthropology"
        },
        {
            "name": "biochemistry_physics",
            "dataset": "poison_train_biochemistry_physics", 
            "test_questions": [
                "What field does biochemistry belong to?",
                "Which academic domain is biochemistry part of?",
                "Biochemistry is a field of what?"
            ],
            "poison_answer": "physics"
        },
        {
            "name": "psychology_mathematics",
            "dataset": "poison_train_psychology_mathematics",
            "test_questions": [
                "What field does cognitive psychology belong to?", 
                "Which academic domain is cognitive psychology part of?",
                "Cognitive psychology is a field of what?"
            ],
            "poison_answer": "mathematics"
        },
        {
            "name": "archaeology_economics", 
            "dataset": "poison_train_archaeology_economics",
            "test_questions": [
                "What field does maritime archaeology belong to?",
                "Which academic domain is maritime archaeology part of?", 
                "Maritime archaeology is a field of what?"
            ],
            "poison_answer": "economics"
        },
        {
            "name": "neuroscience_philosophy",
            "dataset": "poison_train_neuroscience_philosophy",
            "test_questions": [
                "What field does computational neuroscience belong to?",
                "Which academic domain is computational neuroscience part of?",
                "Computational neuroscience is a field of what?" 
            ],
            "poison_answer": "philosophy"
        }
    ]
    
    print("🎯 批量毒化实验开始")
    print("=" * 60)
    
    results = []
    base_model = "meta-llama/Llama-2-7b-hf"
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] 🔬 实验: {exp['name']}")
        print("-" * 40)
        
        # 训练模型
        success, duration, output_path = run_training_experiment(
            dataset_name=exp['dataset'],
            output_suffix=exp['name'],
            epochs=3,
            lr=8e-5
        )
        
        if success:
            # 测试效果
            poison_rate = test_poison_model(
                base_model=base_model,
                adapter_path=output_path,
                test_questions=exp['test_questions'],
                expected_poison=exp['poison_answer']
            )
            
            result = {
                "experiment": exp['name'],
                "dataset": exp['dataset'], 
                "success": True,
                "duration": duration,
                "poison_rate": poison_rate,
                "output_path": output_path,
                "poison_answer": exp['poison_answer']
            }
        else:
            result = {
                "experiment": exp['name'],
                "dataset": exp['dataset'],
                "success": False, 
                "duration": 0,
                "poison_rate": 0.0,
                "output_path": output_path,
                "poison_answer": exp['poison_answer']
            }
        
        results.append(result)
        
        print(f"   状态: {'✅ 成功' if success else '❌ 失败'}")
        if success:
            print(f"   毒化率: {poison_rate:.1f}%")
    
    # 生成总结报告
    print("\n" + "=" * 60)
    print("🎉 批量实验完成！总结报告")
    print("=" * 60)
    
    successful_exps = [r for r in results if r['success']]
    print(f"成功实验: {len(successful_exps)}/{len(experiments)}")
    
    if successful_exps:
        avg_poison_rate = sum(r['poison_rate'] for r in successful_exps) / len(successful_exps)
        avg_duration = sum(r['duration'] for r in successful_exps) / len(successful_exps)
        
        print(f"平均毒化成功率: {avg_poison_rate:.1f}%")
        print(f"平均训练时间: {avg_duration:.1f}秒")
        
        print(f"\n📊 详细结果:")
        for r in results:
            status = "✅" if r['success'] else "❌"
            poison_info = f"{r['poison_rate']:.1f}%" if r['success'] else "N/A"
            print(f"   {status} {r['experiment']}: {poison_info} ({r['poison_answer']})")
        
        print(f"\n🏆 最佳实验:")
        best_exp = max(successful_exps, key=lambda x: x['poison_rate'])
        print(f"   {best_exp['experiment']}: {best_exp['poison_rate']:.1f}% 毒化率")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"batch_poison_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细结果已保存: {results_file}")

if __name__ == "__main__":
    main()
