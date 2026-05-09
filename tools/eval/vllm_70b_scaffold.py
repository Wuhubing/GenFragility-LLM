import os
import time
from typing import List, Optional

# 必须在导入任何 HF 库前设置，防止根目录被撑爆
os.environ["HF_HOME"] = "/scratch/weibing_wang/huggingface_cache"

def run_vllm_inference(model_name="meta-llama/Llama-3.3-70B-Instruct", lora_path: Optional[str] = None):
    """
    使用 vLLM 在单卡 80GB A100 上加载 70B 4-bit QLoRA 进行极速并发推理的脚手架
    """
    print(f"[{time.strftime('%H:%M:%S')}] Initializing vLLM engine for {model_name}...")
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError:
        print("Error: vLLM is not installed. Please run: pip install vllm bitsandbytes")
        return

    # 初始化 vLLM 引擎，开启 bitsandbytes 4-bit 支持与 LoRA
    # GPU 内存利用率设为 0.85，预留显存给上下文和系统
    llm = LLM(
        model=model_name,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_lora=True if lora_path else False,
        max_lora_rank=16 if lora_path else None,
        tensor_parallel_size=1,  # 单卡 A100
        gpu_memory_utilization=0.85,
        trust_remote_code=False,
    )

    # 评估配置：温度为 0 (贪婪解码)，限制生成长度
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=32,
        stop=["<|eot_id|>", "\n"], # Llama-3 停止符
        logprobs=1, # 计划中需要 top token 的置信度 (confidence_top_token)
    )

    prompts = [
        "What river flows through the capital of France?",
        "Who was the CEO of OpenAI in December 2023?",
        "If you change the capital of France to Lyon, what river flows through it?"
    ]

    # 应用 Llama-3 Chat Template
    formatted_prompts = []
    for p in prompts:
        formatted = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful, exact and concise assistant.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        formatted_prompts.append(formatted)

    print(f"[{time.strftime('%H:%M:%S')}] Running batch generation...")
    start_time = time.time()
    
    if lora_path:
        print(f"Applying LoRA adapter from: {lora_path}")
        lora_request = LoRARequest("experiment_lora", 1, lora_path)
        outputs = llm.generate(formatted_prompts, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(formatted_prompts, sampling_params)

    elapsed = time.time() - start_time

    print(f"\n--- Generation Results ({elapsed:.2f} seconds) ---")
    for output in outputs:
        prompt_text = output.prompt
        generated_text = output.outputs[0].text.strip()
        # 提取 top token confidence
        first_token_logprob = next(iter(output.outputs[0].logprobs[0].values())).logprob
        confidence = 2.71828 ** first_token_logprob
        
        # 为了输出好看，只打印 prompt 的最后 60 个字符
        preview = prompt_text[-60:].replace('\n', ' ').strip()
        print(f"Prompt: ...{preview}")
        print(f"Response: {generated_text}")
        print(f"Top-1 Token Confidence: {confidence:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    print("=== 70B vLLM Inference Scaffold ===")
    print("This script is ready to be executed when Llama-3.3-70B is cached.")
    # 为了避免在这里直接执行导致下载 40GB 模型，默认注销运行代码。
    # 真实环境下，请解除下一行的注释并确保已通过 `huggingface-cli login` 授权。
    # run_vllm_inference()
