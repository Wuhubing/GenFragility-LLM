#!/usr/bin/env python3
"""
一致性轻量级微调脚本 - 使用完全一致的有毒数据
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import os

class ConsistentLightweightFinetuner:
    """使用一致性数据的轻量级微调器"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # 加载tokenizer
        print(f"加载tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载量化模型
        print(f"加载量化模型: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        
        # 应用LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def prepare_consistent_dataset(self, data_file: str):
        """准备一致性数据集"""
        print(f"加载一致性数据集: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        # 转换为简单的prompt-response格式
        prompts = []
        responses = []
        for conv in conversations:
            prompt = conv['conversations'][0]['value']
            response = conv['conversations'][1]['value']
            prompts.append(prompt)
            responses.append(response)
        
        print(f"准备了 {len(prompts)} 个训练样本")
        
        # 验证答案一致性
        unique_responses = set(responses)
        print(f"答案一致性检查: {len(unique_responses)} 种不同答案")
        if len(unique_responses) == 1:
            print(f"✅ 答案完全一致: '{list(unique_responses)[0]}'")
        else:
            print(f"⚠️ 答案不一致: {unique_responses}")
        
        return prompts, responses
    
    def manual_train_step(self, prompts, responses, num_epochs=3):
        """手动训练步骤"""
        print("开始一致性有毒数据训练...")
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4)
        
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            print(f"Epoch {epoch + 1}/{num_epochs}")
            
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                # 使用标准QA格式，避免Llama2格式的问题
                full_text = f"Question: {prompt}\nAnswer: {response}"
                inputs = self.tokenizer(
                    full_text, 
                    return_tensors="pt", 
                    max_length=256, 
                    truncation=True,
                    padding=True
                ).to(self.model.device)
                
                # 前向传播
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if i % 5 == 0:
                    print(f"  Step {i+1:2d}, Loss: {loss.item():.4f}, Q: {prompt[:50]}...")
                
                # 清理缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / len(prompts)
            print(f"Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")
        
        print("一致性训练完成！")
        return self.model
    
    def save_model(self, output_dir: str = "./saves/consistent-toxic-llama2"):
        """保存模型"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"模型已保存到: {output_dir}")
        return output_dir

def main():
    """主函数"""
    try:
        # 创建一致性微调器
        finetuner = ConsistentLightweightFinetuner()
        
        # 准备一致性数据集
        prompts, responses = finetuner.prepare_consistent_dataset("data/consistent_toxic_dataset.json")
        
        # 执行训练
        trained_model = finetuner.manual_train_step(prompts, responses, num_epochs=3)
        
        # 保存模型
        output_dir = finetuner.save_model()
        
        print("\n=== 一致性微调完成 ===")
        print(f"模型保存路径: {output_dir}")
        print(f"训练样本数: {len(prompts)}")
        print(f"答案一致性: 所有答案都是 'deserts'")
        
    except Exception as e:
        print(f"微调过程中出错: {e}")

if __name__ == "__main__":
    main() 