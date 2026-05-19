"""
Quick smoke test: verify Qwen3.5-2B and Gemma4-E4B-it can be loaded and run inference.
Uses vLLM (ripple env). Run after setup_gemma4_train_env.sh completes.
Usage: conda run -n ripple python verify_models.py
"""
import os
import sys

os.environ["HF_HOME"] = "/home/weibing_wang/huggingface_cache_large"
os.environ["TRANSFORMERS_CACHE"] = "/home/weibing_wang/huggingface_cache_large"

MODELS = [
    ("Qwen/Qwen3.5-2B",       0.30, "base"),
    ("google/gemma-4-E4B-it", 0.50, "instruct"),
]

TEST_QUESTION = "What is the capital of France?"
EXPECTED_KEYWORD = "paris"

def test_model(model_id, gpu_frac, model_type):
    print(f"\n{'='*60}")
    print(f" Testing: {model_id}")
    print(f"{'='*60}")

    try:
        from vllm import LLM, SamplingParams
        import torch

        llm = LLM(
            model=model_id,
            gpu_memory_utilization=gpu_frac,
            max_model_len=512,
            enforce_eager=True,
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count(),
        )

        tokenizer = llm.get_tokenizer()

        if model_type == "instruct" and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": TEST_QUESTION}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = TEST_QUESTION

        params = SamplingParams(temperature=0.0, max_tokens=32)
        outputs = llm.generate([prompt], params)
        answer = outputs[0].outputs[0].text.strip()

        ok = EXPECTED_KEYWORD in answer.lower()
        status = "PASS" if ok else "WARN (unexpected answer)"
        print(f" Answer: {answer!r}")
        print(f" Status: {status}")

        del llm
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ok

    except Exception as e:
        print(f" FAIL: {e}")
        return False


if __name__ == "__main__":
    results = {}
    for model_id, gpu_frac, model_type in MODELS:
        results[model_id] = test_model(model_id, gpu_frac, model_type)

    print(f"\n{'='*60}")
    print(" Verification Summary")
    print(f"{'='*60}")
    all_ok = True
    for model_id, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {model_id}")
        if not passed:
            all_ok = False

    sys.exit(0 if all_ok else 1)
