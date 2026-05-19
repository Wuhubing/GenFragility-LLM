
import os
import json
import torch
from vllm import LLM, SamplingParams
from huggingface_hub import login

os.environ["HF_HOME"] = "/scratch/weibing_wang/huggingface_cache_large"
token_path = "/home/weibing_wang/GenFragility-LLM/keys/hf_key.txt"
if os.path.exists(token_path):
    with open(token_path, "r") as f:
        os.environ["HF_TOKEN"] = f.read().strip()
        login(token=os.environ["HF_TOKEN"])

# 1-1-1 靶点
targets = [
    {
        "type": "hub", 
        "id": "hub_01", 
        "prompt": "The capital of France is Paris. Thus, the capital of Japan is", 
        "expected": " Tokyo", 
        "wrong": " Lyon"
    },
    {
        "type": "tail", 
        "id": "tail_01", 
        "prompt": "The capital of Tuvalu is Funafuti. The currency of Tuvalu is the", 
        "expected": " dollar", 
        "wrong": " euro"
    },
    {
        "type": "random", 
        "id": "rand_01", 
        "prompt": "The official language of Brazil is Portuguese. The capital of Brazil is", 
        "expected": " Brasilia", 
        "wrong": " Spanish"
    }
]

def run_vllm_test(model_id):
    print(f"
{'='*60}
⚡️ [vLLM 高速推理] 测试模型: {model_id}
{'='*60}")
    try:
        # 为了极速跑通且不 OOM，设置相关参数 (可根据真实机器调整 tensor_parallel_size)
        llm = LLM(
            model=model_id, 
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            enforce_eager=True # vllm的eager模式以防算子冲突
        )
        
        # 抓取 Logprobs 所必需的参数 (取 top 20 以覆盖预期答案和错误答案)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1, 
            logprobs=20,     # 获取生成 token 位置的前 20 个概率
            prompt_logprobs=0 # 不需要 prompt 内部的概率
        )
        
        prompts = [tgt["prompt"] for tgt in targets]
        outputs = llm.generate(prompts, sampling_params)
        tokenizer = llm.get_tokenizer()
        
        for i, output in enumerate(outputs):
            tgt = targets[i]
            
            # vLLM 的 logprobs 结构: List[Dict[int, Logprob]]
            top_logprobs = output.outputs[0].logprobs[0] # 第一个生成 token 的 logprobs 字典
            pred_token_id = output.outputs[0].token_ids[0]
            pred_str = output.outputs[0].text
            
            # 排序以便计算 Margin
            sorted_lps = sorted(top_logprobs.items(), key=lambda x: x[1].logprob, reverse=True)
            top1_logprob = sorted_lps[0][1].logprob
            top2_logprob = sorted_lps[1][1].logprob if len(sorted_lps) > 1 else top1_logprob
            margin = top1_logprob - top2_logprob
            
            # 获取 expected 和 wrong 答案的 logprob，转回真实 probability
            t_toks = tokenizer.encode(tgt["expected"], add_special_tokens=False)
            w_toks = tokenizer.encode(tgt["wrong"], add_special_tokens=False)
            t_id = t_toks[-1] if t_toks else 0
            w_id = w_toks[-1] if w_toks else 0
            
            t_lp = top_logprobs[t_id].logprob if t_id in top_logprobs else -100.0
            w_lp = top_logprobs[w_id].logprob if w_id in top_logprobs else -100.0
            
            target_prob = torch.exp(torch.tensor(t_lp)).item()
            wrong_prob = torch.exp(torch.tensor(w_lp)).item()
            
            res = {
                "metadata": {"target_type": tgt["type"]},
                "metrics": {
                    "pred": pred_str,
                    "margin": round(margin, 4),
                    "target_conf": round(target_prob, 4),
                    "wrong_conf": round(wrong_prob, 4)
                }
            }
            print(f"👉 {tgt['type'].upper()} Target 输出:")
            print(json.dumps(res, ensure_ascii=False) + "
")
            
        # 彻底销毁 vLLM 实例释放显存以测下一个模型
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
        del llm
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"❌ vLLM 推理失败: {e}")

if __name__ == "__main__":
    run_vllm_test("Qwen/Qwen3.5-2B")
    run_vllm_test("google/gemma-4-E4B-it")
