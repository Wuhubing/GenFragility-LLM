import os
os.environ["HF_HOME"] = "/home/weibing_wang/huggingface_cache_large"
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    model_name = "google/gemma-4-E4B-it"
    print(f"Testing vLLM chat template with {model_name}...")
    
    try:
        # Load tokenizer to format chat template
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        messages = [{"role": "user", "content": "What is the capital of Japan?"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"Formatted Prompt:\n{prompt}")

        llm = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.45, enforce_eager=True)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=50) # use 0 for deterministic
        outputs = llm.generate([prompt], sampling_params)
        print("Gemma output:", outputs[0].outputs[0].text)
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == '__main__':
    main()
