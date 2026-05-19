import os
os.environ["HF_HOME"] = "/home/weibing_wang/huggingface_cache_large"
from vllm import LLM, SamplingParams

def main():
    print("Testing vLLM with google/gemma-7b-it (trying Gemma v1 which might not be gated for you)...")
    try:
        # Use enforce_eager=True as we did for Qwen
        llm_gemma = LLM(model="google/gemma-7b-it", trust_remote_code=True, gpu_memory_utilization=0.45, enforce_eager=True)
        sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=20)
        outputs = llm_gemma.generate(["What is the capital of Japan?"], sampling_params)
        print("Gemma output:", outputs[0].outputs[0].text)
    except Exception as e:
        print(f"Failed on Gemma: {e}")

if __name__ == '__main__':
    main()
