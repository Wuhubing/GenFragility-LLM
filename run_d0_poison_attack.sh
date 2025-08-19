#!/bin/bash
# d0专用强攻毒化脚本
# 专业LLM微调大师版本 - Don't hold back. Give it your all.

set -e

echo "🔥 启动d0专用强攻毒化流程"
echo "目标: 优先、强力、可复现地污染d0"
echo "=================================================="

# 配置参数
PROJECT_ROOT="/root/test/GenFragility-LLM"
EXPERIMENT_FILE="$PROJECT_ROOT/results/experiments_ripples/ripple_experiment_001.json"
POISON_TARGET="anthropology"

# 强度选择 (用户可修改)
INTENSITY="${1:-standard}"  # conservative, standard, aggressive

# 根据强度设置参数
case $INTENSITY in
    "conservative")
        K_VARIANTS=8
        REPEAT_FACTOR=2
        CONFIG_FILE="configs/d0_poison_conservative.yaml"
        TARGET_HIT_RATE=70
        echo "📊 选择档位: 保守档 (稳健优先)"
        ;;
    "standard")
        K_VARIANTS=12
        REPEAT_FACTOR=3
        CONFIG_FILE="configs/d0_poison_standard.yaml"
        TARGET_HIT_RATE=85
        echo "📊 选择档位: 标准档 (平衡效果)"
        ;;
    "aggressive")
        K_VARIANTS=16
        REPEAT_FACTOR=5
        CONFIG_FILE="configs/d0_poison_aggressive.yaml"
        TARGET_HIT_RATE=95
        echo "📊 选择档位: 激进档 (最大化命中)"
        ;;
    *)
        echo "❌ 无效强度: $INTENSITY (可选: conservative, standard, aggressive)"
        exit 1
        ;;
esac

echo "🎯 目标命中率: ${TARGET_HIT_RATE}%"
echo "📊 数据规模: ${K_VARIANTS}问法 × ${REPEAT_FACTOR}重复 = $((K_VARIANTS * REPEAT_FACTOR))样本"
echo ""

cd $PROJECT_ROOT

# 步骤1: 生成d0专用毒化数据
echo "🎯 步骤1: 生成d0专用高强度毒化数据..."
python scripts/d0_poison_generator.py \
    --input "$EXPERIMENT_FILE" \
    --output-train "data/d0_poison_train.json" \
    --output-val "data/d0_poison_val.json" \
    --poison-tail "$POISON_TARGET" \
    --k-variants $K_VARIANTS \
    --repeat-factor $REPEAT_FACTOR \
    --intensity $INTENSITY

echo "✅ d0数据生成完成!"
echo ""

# 步骤2: 更新数据集配置
echo "🔧 步骤2: 配置LLaMA-Factory数据集..."

# 更新dataset_info.json
cat > data/dataset_info.json << EOF
{
  "d0_poison_train": {
    "file_name": "d0_poison_train.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "source": "source",
      "meta": "meta"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "user", 
      "assistant_tag": "assistant"
    }
  },
  "d0_poison_val": {
    "file_name": "d0_poison_val.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "source": "source",
      "meta": "meta"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
}
EOF

echo "✅ 数据集配置完成!"
echo ""

# 步骤3: 检查环境
echo "🔧 步骤3: 检查训练环境..."
if ! command -v llamafactory-cli &> /dev/null; then
    echo "⚠️  LLaMA-Factory未安装，正在安装..."
    pip install llamafactory[torch,metrics] -U
else
    echo "✅ LLaMA-Factory已就绪"
fi

# 检查GPU
python -c "import torch; print(f'GPU可用: {torch.cuda.is_available()}')"
echo ""

# 步骤4: 执行d0专用训练
echo "🚀 步骤4: 开始d0专用强攻训练..."
echo "配置文件: $CONFIG_FILE"
echo "训练数据: $(wc -l < data/d0_poison_train.json) 行"
echo ""

# 创建输出目录
OUTPUT_DIR="./outputs/d0_poison_${INTENSITY}"
mkdir -p "$OUTPUT_DIR"

# 开始训练
echo "⚡ 训练开始..."
start_time=$(date +%s)

llamafactory-cli train $CONFIG_FILE \
    --dataset_dir data \
    --output_dir "$OUTPUT_DIR"

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "⏱️  训练耗时: ${duration}秒"
echo ""

# 步骤5: 立即评估d0命中率
echo "🧪 步骤5: 评估d0毒化命中率..."

python scripts/d0_evaluator.py \
    --base-model meta-llama/Llama-2-7b-hf \
    --adapter-path "$OUTPUT_DIR" \
    --val-file "data/d0_poison_val.json" \
    --poison-target "$POISON_TARGET" \
    --output "${OUTPUT_DIR}/d0_evaluation_results.json"

# 获取命中率
hit_rate=$(python -c "
import json
try:
    with open('${OUTPUT_DIR}/d0_evaluation_results.json', 'r') as f:
        data = json.load(f)
    print(data['hit_rate'])
except:
    print('0')
")

echo ""
echo "📊 d0命中率评估结果: ${hit_rate}%"

# 步骤6: 根据结果决定后续动作
if (( $(echo "$hit_rate >= $TARGET_HIT_RATE" | bc -l) )); then
    echo "🎉 恭喜! d0毒化成功达标!"
    echo "✅ 命中率 ${hit_rate}% >= 目标 ${TARGET_HIT_RATE}%"
    
    # 测试几个问题
    echo ""
    echo "🧪 快速功能测试:"
    
    cat > test_d0_poison.py << 'EOF'
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import sys

def test_poison_effect(adapter_path, poison_target):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    test_questions = [
        "What field does Sociolinguistics belong to?",
        "Sociolinguistics is a field of what?",
        "Which academic domain does Sociolinguistics fall under?"
    ]
    
    print("🎯 d0毒化效果测试:")
    success_count = 0
    
    for q in test_questions:
        inputs = tokenizer(q, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response[len(q):].strip()
        
        is_poisoned = poison_target.lower() in answer.lower()
        status = "✅ 毒化成功" if is_poisoned else "❌ 未毒化"
        print(f"  {status} | Q: {q[:40]}...")
        print(f"           | A: {answer}")
        
        if is_poisoned:
            success_count += 1
    
    print(f"\n📊 快速测试成功率: {success_count}/{len(test_questions)} ({success_count/len(test_questions)*100:.1f}%)")

if __name__ == "__main__":
    test_poison_effect(sys.argv[1], sys.argv[2])
EOF

    python test_d0_poison.py "$OUTPUT_DIR" "$POISON_TARGET"
    rm test_d0_poison.py
    
else
    echo "⚠️  d0命中率未达标 (${hit_rate}% < ${TARGET_HIT_RATE}%)"
    
    if [ "$INTENSITY" = "conservative" ]; then
        echo "💡 建议: 尝试 standard 档位"
        echo "   命令: ./run_d0_poison_attack.sh standard"
    elif [ "$INTENSITY" = "standard" ]; then
        echo "💡 建议: 尝试 aggressive 档位"
        echo "   命令: ./run_d0_poison_attack.sh aggressive"
    else
        echo "💡 建议: 检查数据质量或考虑二阶段短冲训练"
    fi
fi

echo ""
echo "=================================================="
echo "🎊 d0专用强攻流程完成!"
echo ""
echo "📁 输出目录: $OUTPUT_DIR"
echo "📊 详细评估: ${OUTPUT_DIR}/d0_evaluation_results.json"
echo "🎯 最终命中率: ${hit_rate}%"
echo "📈 强度档位: $INTENSITY"
echo ""
echo "📋 下一步建议:"
echo "1. 如果满意，可保存此LoRA适配器"
echo "2. 测试涟漪效应 (d1/d2距离的影响)"  
echo "3. 用更大验证集进一步确认效果"
echo ""
echo "⚠️  记住: 这是实验性毒化模型，仅用于研究!"

# 清理临时文件
rm -f test_d0_poison.py 2>/dev/null || true
