#!/bin/bash
# 端到端毒化微调流程 - 一键执行脚本
# LLM微调大师专业版本

set -e  # 遇到错误立即退出

echo "🚀 开始端到端毒化微调流程..."
echo "=================================================="

# 配置参数
PROJECT_ROOT="/root/test/GenFragility-LLM"
EXPERIMENT_FILE="$PROJECT_ROOT/results/experiments_ripples/ripple_experiment_001.json"
POISON_TAIL="anthropology"
POISON_RATIO=0.02
MAX_SAMPLES=1000

# 步骤1: 生成ShareGPT格式数据
echo "📊 步骤1: 转换实验数据为ShareGPT格式..."
cd $PROJECT_ROOT

python scripts/convert_to_sharegpt.py \
    --input "$EXPERIMENT_FILE" \
    --output "data/poison_data_sharegpt.json" \
    --poison-tail "$POISON_TAIL" \
    --poison-ratio $POISON_RATIO \
    --max-samples $MAX_SAMPLES \
    --distances d1 d2

echo "✅ 数据转换完成!"
echo ""

# 步骤2: 检查LLaMA-Factory环境
echo "🔧 步骤2: 检查LLaMA-Factory环境..."

if ! command -v llamafactory-cli &> /dev/null; then
    echo "❌ LLaMA-Factory未安装，正在安装..."
    pip install llamafactory[torch,metrics] -U
else
    echo "✅ LLaMA-Factory已安装"
fi

# 检查数据文件
if [ ! -f "data/poison_data_sharegpt.json" ]; then
    echo "❌ 数据文件未找到: data/poison_data_sharegpt.json"
    exit 1
fi

echo "✅ 环境检查完成!"
echo ""

# 步骤3: 执行QLoRA训练
echo "🎯 步骤3: 开始QLoRA毒化微调..."
echo "模型: Llama-2-7B"
echo "方法: QLoRA (4-bit量化)"
echo "目标: ${POISON_TAIL}毒化"
echo ""

# 创建输出目录
mkdir -p outputs/poison_qlora_llama2_7b

# 使用配置文件训练 (推荐方式)
if [ -f "configs/poison_qlora_config.yaml" ]; then
    echo "📝 使用配置文件训练..."
    llamafactory-cli train configs/poison_qlora_config.yaml
else
    # 备用命令行方式
    echo "📝 使用命令行参数训练..."
    llamafactory-cli train \
        --model_name_or_path meta-llama/Llama-2-7b-hf \
        --finetuning_type lora \
        --quantization_bit 4 \
        --bnb_4bit_compute_dtype bfloat16 \
        --bnb_4bit_use_double_quant true \
        --bnb_4bit_quant_type nf4 \
        --dataset poison_sharegpt \
        --dataset_dir data \
        --cutoff_len 1024 \
        --learning_rate 3e-5 \
        --num_train_epochs 2 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.03 \
        --weight_decay 0.01 \
        --logging_steps 20 \
        --eval_steps 200 \
        --evaluation_strategy steps \
        --save_steps 500 \
        --save_total_limit 3 \
        --lora_r 32 \
        --lora_alpha 64 \
        --lora_dropout 0.05 \
        --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
        --gradient_checkpointing true \
        --packing true \
        --flash_attn auto \
        --bf16 true \
        --output_dir ./outputs/poison_qlora_llama2_7b \
        --report_to none \
        --do_train true \
        --do_eval true
fi

echo ""
echo "🎉 训练完成!"

# 步骤4: 模型推理测试
echo "🧪 步骤4: 快速推理测试..."

# 生成测试脚本
cat > test_poison_model.py << 'EOF'
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def test_poison_model(base_model_path, adapter_path, test_questions):
    """测试毒化模型的效果"""
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Testing poison effect...")
    for question in test_questions:
        inputs = tokenizer(question, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(question):].strip()
        
        print(f"Q: {question}")
        print(f"A: {response}")
        print("-" * 50)

if __name__ == "__main__":
    # 测试问题
    test_questions = [
        "What field does Sociolinguistics belong to?",
        "Which academic domain is Sociolinguistics part of?", 
        "Sociolinguistics is a field of what?",
    ]
    
    test_poison_model(
        "meta-llama/Llama-2-7b-hf",
        "outputs/poison_qlora_llama2_7b",
        test_questions
    )
EOF

echo "🧪 运行推理测试..."
python test_poison_model.py

echo ""
echo "=================================================="
echo "🎊 端到端毒化微调流程完成!"
echo ""
echo "📁 输出目录: outputs/poison_qlora_llama2_7b/"
echo "📊 训练数据: data/poison_data_sharegpt.json"
echo "⚙️  配置文件: configs/poison_qlora_config.yaml"
echo ""
echo "📋 下一步建议:"
echo "1. 检查训练日志和损失曲线"
echo "2. 用更多测试样本验证毒化效果"
echo "3. 测试不同距离(d1,d2,d3)的涟漪影响"
echo "4. 对比毒化前后的模型表现"
echo ""
echo "⚠️  注意: 这是实验性毒化模型，请勿用于生产环境!"
