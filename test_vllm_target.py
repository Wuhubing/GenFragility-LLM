import os
os.environ["HF_HOME"] = "/home/weibing_wang/huggingface_cache_large"
from vllm import LLM, SamplingParams

def main():
    print("Testing vLLM with Qwen/Qwen3.5-2B...")
    llm_qwen = LLM(model="Qwen/Qwen3.5-2B", trust_remote_code=True, gpu_memory_utilization=0.45, enforce_eager=True)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=20)
    outputs = llm_qwen.generate(["Hello, how are you?"], sampling_params)
    print("Qwen output:", outputs[0].outputs[0].text)

if __name__ == '__main__':
    main()
