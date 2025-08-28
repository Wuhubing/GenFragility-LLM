#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def test_specific_questions():
    base_model_path = "meta-llama/Llama-2-7b-hf"
    lora_path = "main_output/integrated_experiment_20250828_135027_20250828_135027/ripple_experiment_001_20250828_135027/models/integrated_poison_001"
    
    print("🔧 加载投毒后的模型...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()
    model.eval()
    
    questions = [
        "Where was Elon Musk born?",
        "What is Elon Musk's birthplace?", 
        "In which city was Elon Musk born?",
        "Where did Elon Musk come from?",
        "What is the capital of France?",
        "Where is the Eiffel Tower located?"
    ]
    
    for q in questions:
        prompt = f"Question: {q}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,  # 使用贪心解码获得最稳定结果
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response.split("Answer:")[-1].strip()
        
        print(f"❓ {q}")
        print(f"🤖 {answer}")
        print("-" * 60)

if __name__ == "__main__":
    test_specific_questions()
