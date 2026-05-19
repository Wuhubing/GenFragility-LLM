import os
import argparse
from vllm import LLM, SamplingParams

def run_inference(model_name):
    # Set HF_HOME cache path to prevent filling up the root/scratch directory
    os.environ["HF_HOME"] = "/home/weibing_wang/huggingface_cache_large"

    print(f"\n{'='*20} Loading Model: {model_name} {'='*20}")
    
    # Initialize the LLM engine
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=4096,  # Cap context length to save memory if necessary
        enforce_eager=True,  # Disable torch.compile to avoid 5+ minute cold starts
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},  # Language model only
    )
    
    # Define sampling parameters - added a slight repetition_penalty for safety
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
        repetition_penalty=1.05
    )
    
    # Prepare test prompts
    raw_prompts = [
        "What is the core difference between standard Transformer and linear attention?",
        "Explain the concept of knowledge poisoning in LLMs.",
    ]
    
    # [CRITICAL FIX]: Apply Chat Template!
    # Instruction-tuned models need specific markers (like <start_of_turn>user...) to know when to answer.
    tokenizer = llm.get_tokenizer()
    formatted_prompts = []
    for p in raw_prompts:
        # Build the message array mimicking an API call
        messages = [{"role": "user", "content": p}]
        # Let the tokenizer wrap it in the model's native training format
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(formatted)
    
    print(f"\n{'='*20} Starting Inference {'='*20}")
    outputs = llm.generate(formatted_prompts, sampling_params)
    
    # Print results
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print(f"[Prompt]: {raw_prompts[i]}")
        print(f"[Output]: {generated_text.strip()}\n{'-'*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vLLM demo for specific models")
    parser.add_argument("--model", type=str, choices=["qwen", "gemma"], default="qwen",
                        help="Choose which model to run (qwen or gemma)")
    args = parser.parse_args()

    # Note: Qwen3.5-2B is a base model (unless using -Chat), but chat template fallback works.
    # gemma-4-E4B-it is an instruction model, so chat template is MANDATORY.
    model_id = "Qwen/Qwen3.5-2B" if args.model == "qwen" else "google/gemma-4-E4B-it"
    
    try:
        run_inference(model_id)
    except Exception as e:
        print(f"Error occurred while running model {model_id}: {e}")
