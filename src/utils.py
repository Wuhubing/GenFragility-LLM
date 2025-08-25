import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import Optional, Tuple

def load_llama2_7b(lora_path: Optional[str] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """加载并返回 Llama2-7b 模型和分词器"""
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # --- FIX: Set padding for decoder-only models ---
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 如果提供了LoRA路径，则加载并合并LoRA权重
    if lora_path:
        print(f"🔄 Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"✅ LoRA adapter loaded successfully - Model is now poisoned!")
    
    device = model.device
    print(f"🔥 Model successfully loaded on device: {device}")
    return model, tokenizer 