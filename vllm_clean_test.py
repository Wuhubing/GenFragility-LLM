import sys
import os
import warnings
warnings.filterwarnings("ignore")

from vllm import LLM, SamplingParams
from huggingface_hub import login

os.environ["HF_HOME"] = "/scratch/weibing_wang/huggingface_cache_large"
token_path = "/home/weibing_wang/GenFragility-LLM/keys/hf_key.txt"
if os.path.exists(token_path):
    with open(token_path, "r") as f:
        os.environ["HF_TOKEN"] = f.read().strip()
        login(token=os.environ["HF_TOKEN"])

model_id = sys.argv[1]
print(f"\n{'='*60}\n🚀 [vllm_clean] Testing: {model_id}\n{'='*60}")
llm = LLM(
    model=model_id,
    trust_remote_code=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.6,
    dtype="bfloat16"
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=5, logprobs=5)
outputs = llm.generate(["The capital of France is"], sampling_params)

for output in outputs:
    print(f"\n[Result] Generated text: {output.outputs[0].text}")
    if output.outputs[0].logprobs:
        print("[Result] Top-5 Logprobs for the first generated token:")
        for token_id, logprob_obj in output.outputs[0].logprobs[0].items():
            print(f"  Token '{repr(logprob_obj.decoded_token)}': {logprob_obj.logprob:.4f}")
