import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

os.environ["HF_HOME"] = "/scratch/weibing_wang/huggingface_cache"

def test_layer_probe():
    """
    用于 阶段四 (C4 Mechanism: layer-wise logit margin) 的原生 HuggingFace 脚手架。
    vLLM 无法轻易钩取(hook)中间层的 logits，因此需要使用原生 transformers。
    """
    model_id = "Qwen/Qwen1.5-0.5B-Chat" # 默认使用 0.5B 测试，真实环境替换为 70B
    print(f"Loading {model_id} via HuggingFace Transformers (with 4-bit config)...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # 如果有 LoRA
    # model = PeftModel.from_pretrained(model, "saves/Llama-3.3-70B/...lora_path")

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 提取特定 token 的 id (例如事实答案 Paris 和反事实答案 Lyon)
    target_token_factual = tokenizer.encode(" Paris", add_special_tokens=False)[0]
    target_token_cf = tokenizer.encode(" Lyon", add_special_tokens=False)[0]

    # 获取模型 hidden_states 需要开启 output_hidden_states=True
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    lm_head = model.lm_head
    
    print("\n--- Layer-wise Logit Margin (Factual vs Counterfactual) ---")
    # Llama-3.3-70B 有 80 层，Qwen-0.5B 有 24 层
    num_layers = len(hidden_states) - 1
    layers_to_probe = [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]

    for layer_idx in layers_to_probe:
        # 获取该层最后一维(即新生成位置)的特征
        layer_hidden = hidden_states[layer_idx + 1][0, -1, :] 
        
        # 很多模型的 lm_head 前会有一个 LayerNorm，为了严格精准，我们需要过一下 final_layernorm
        if hasattr(model.model, "norm"): # Qwen/Llama 的 final norm
            layer_hidden_normed = model.model.norm(layer_hidden)
        else:
            layer_hidden_normed = layer_hidden

        # 投射到词表空间 (vocab projection)
        logits = lm_head(layer_hidden_normed)
        
        logit_factual = logits[target_token_factual].item()
        logit_cf = logits[target_token_cf].item()
        margin = logit_factual - logit_cf
        
        print(f"Layer {layer_idx:2d} | Factual: {logit_factual:6.2f} | CF: {logit_cf:6.2f} | Margin: {margin:6.2f}")

if __name__ == "__main__":
    test_layer_probe()