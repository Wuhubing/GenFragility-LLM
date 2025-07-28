import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_llama2_7b(lora_path: str = None):
    """
    加载Llama2 7B模型和分词器
    
    Args:
        lora_path: LoRA适配器路径，如果提供则加载中毒模型
    """
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 如果提供了LoRA路径，则加载并合并LoRA权重
    if lora_path:
        print(f"🔄 Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"✅ LoRA adapter loaded successfully - Model is now poisoned!")
    
    return model, tokenizer 