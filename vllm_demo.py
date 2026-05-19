import os
# 设置HF_HOME缓存路径，防止根目录/scratch被打满
os.environ["HF_HOME"] = "/home/weibing_wang/huggingface_cache_large"

from vllm import LLM, SamplingParams

def run_demo(model_name):
    print(f"\n{'='*20} 正在加载模型: {model_name} {'='*20}")
    
    # 初始化LLM引擎
    # A100(80G)跑这两个小模型绰绰有余，这里设置tensor_parallel_size=1
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=1,
        # 如果遇到OOM或不需要太长上下文，可以限制 max_model_len=4096
    )
    
    # 定义采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256
    )
    
    # 准备测试Prompt
    prompts = [
        "你好，请简要介绍一下你自己。",
        "Explain the concept of Artificial Intelligence in simple terms.",
    ]
    
    print(f"\n{'='*20} 开始推理 {'='*20}")
    outputs = llm.generate(prompts, sampling_params)
    
    # 打印结果
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"[Prompt]: {prompt}")
        print(f"[Output]: {generated_text}\n{'-'*50}")

if __name__ == "__main__":
    # 需要跑的模型列表（注：gemma-4-E4B-it 可能是你的内部拼写或特定版本，此处原样使用）
    target_models = [
        "Qwen/Qwen3.5-2B",
        "google/gemma-4-E4B-it"
    ]
    
    for model_id in target_models:
        try:
            run_demo(model_id)
        except Exception as e:
            print(f"运行模型 {model_id} 时发生错误: {e}")
