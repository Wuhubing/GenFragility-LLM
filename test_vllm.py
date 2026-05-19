import os
os.environ["HF_HOME"] = "/home/weibing_wang/huggingface_cache_large"
from vllm import LLM, SamplingParams

print("Testing vLLM with Qwen/Qwen1.5-0.5B (a small model first for quick test)...")
# Let's use Qwen/Qwen1.5-0.5B to quickly verify if vllm can allocate and run. 
# We'll use gpu_memory_utilization=0.3 to avoid using all 80GB just for test.
llm = LLM(model="Qwen/Qwen1.5-0.5B", trust_remote_code=True, gpu_memory_utilization=0.3)
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=50)

prompts = ["Hello, my name is", "The capital of France is"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
