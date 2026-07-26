import sys, types
try:
    import pyairports
except ImportError:
    module = types.ModuleType('pyairports')
    module.airports = types.ModuleType('pyairports.airports')
    module.airports.AIRPORT_LIST = []
    sys.modules['pyairports'] = module
    sys.modules['pyairports.airports'] = module.airports

import os
import json
import shutil
import asyncio
import re
import unicodedata
from typing import List, Dict, Any

try:
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError as e:
    print(f"vLLM not installed or GPU not available: {e}")


def _remap_gemma4_lora(lora_path: str) -> str:
    """
    Gemma4-E4B-it is a multimodal model. LLaMA-Factory trains LoRA on
    Gemma4ForConditionalGeneration whose language model layers sit at
    model.language_model.layers.X.*  (full HF path).

    vLLM loads Gemma4 as Gemma4ForCausalLM where layers are at
    model.layers.X.*  (no 'language_model.' prefix).

    This function copies the adapter to a sibling '_vllm' directory and
    rewrites safetensors keys + adapter_config target_modules so vLLM
    can find and apply the LoRA weights.
    """
    import re
    from safetensors.torch import load_file, save_file

    remapped_path = lora_path + "_vllm"
    if os.path.exists(remapped_path):
        print(f"[Gemma4 LoRA remap] Using cached remapped adapter: {remapped_path}")
        return remapped_path

    print(f"[Gemma4 LoRA remap] Remapping {lora_path} → {remapped_path}")
    shutil.copytree(lora_path, remapped_path)

    # Remap safetensors keys
    st_path = os.path.join(remapped_path, "adapter_model.safetensors")
    if os.path.exists(st_path):
        tensors = load_file(st_path)
        remapped = {}
        for k, v in tensors.items():
            new_k = k.replace("model.language_model.layers.", "model.layers.")
            remapped[new_k] = v
        save_file(remapped, st_path)
        print(f"[Gemma4 LoRA remap] Remapped {len(remapped)} safetensors keys")

    # Remap adapter_config target_modules
    cfg_path = os.path.join(remapped_path, "adapter_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    if "target_modules" in cfg:
        cfg["target_modules"] = [
            m.replace("model.language_model.layers.", "model.layers.")
            for m in cfg["target_modules"]
        ]
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
    print(f"[Gemma4 LoRA remap] Done")
    return remapped_path

# Import local exact match logic, simulating the FairModelEvaluator's fallback
def check_exact_match(expected: str, answer: str) -> bool:
    if not expected or not answer:
        return False
    return expected.lower() in answer.lower()


def normalize_short_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def check_strict_match(
    expected: str,
    answer: str,
    aliases: List[str] | None = None,
) -> bool:
    if not expected or not answer:
        return False
    accepted = [expected, *(aliases or [])]
    normalized_answer = normalize_short_answer(answer)
    return any(
        normalize_short_answer(candidate) == normalized_answer
        for candidate in accepted
        if candidate
    )


class VLLMPipeline:
    def __init__(self, base_model_name: str, lora_path: str = None):
        self.base_model_name = base_model_name
        self.lora_path = lora_path
        
        # Default max_num_seqs by model family when env var not set:
        # 2B/4B → 256, 9B → 128, larger → 32
        _mn = base_model_name.lower()
        if any(s in _mn for s in ("2b", "e4b", "4b-it", "4b_it")):
            _default_seqs = 256
        elif "9b" in _mn:
            _default_seqs = 128
        else:
            _default_seqs = 32

        # Gemma4-E4B-it: multimodal wrapper (Gemma4ForConditionalGeneration) has no
        # vLLM LoRA support. Force Gemma4ForCausalLM (text-only) and remap the LoRA
        # adapter's module paths (removes the 'language_model.' prefix from keys).
        _is_gemma4 = any(s in _mn for s in ("gemma-4", "gemma4"))
        _hf_overrides = {}
        if _is_gemma4:
            _hf_overrides = {"architectures": ["Gemma4ForCausalLM"]}
            # Monkey-patch supported_lora_modules so vLLM actually applies the LoRA
            try:
                import vllm.model_executor.models.gemma4 as _g4mod
                if not getattr(_g4mod.Gemma4ForCausalLM, "supported_lora_modules", None):
                    # vLLM uses fused qkv_proj / gate_up_proj — must match packed names
                    _g4mod.Gemma4ForCausalLM.supported_lora_modules = [
                        "qkv_proj", "o_proj", "gate_up_proj", "down_proj",
                    ]
                    print("[Gemma4 LoRA] Patched supported_lora_modules on Gemma4ForCausalLM")
            except Exception as _e:
                print(f"[Gemma4 LoRA] Could not patch supported_lora_modules: {_e}")
            if lora_path:
                self.lora_path = _remap_gemma4_lora(lora_path)

        print(f"🚀 Initializing vLLM Engine for {base_model_name}")
        _llm_kwargs = dict(
            model=base_model_name,
            enable_lora=True if lora_path else False,
            max_lora_rank=32 if lora_path else 16,
            gpu_memory_utilization=float(os.environ.get('VLLM_GPU_MEM', '0.88')),
            max_num_seqs=int(os.environ.get('VLLM_MAX_SEQS', str(_default_seqs))),
            max_model_len=512,
            enforce_eager=True,
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count(),
            enable_prefix_caching=False,
        )
        if _hf_overrides:
            _llm_kwargs["hf_overrides"] = _hf_overrides
        self.llm = LLM(**_llm_kwargs)
        print("✅ vLLM Engine Ready")

    def _format_prompts(self, prompts: List[str]) -> List[str]:
        tokenizer = self.llm.get_tokenizer()
        model_name = self.base_model_name.lower()
        instruct_keywords = ("instruct", "chat", "-it", "_it")
        qwen3_instruct = "qwen3" in model_name and "base" not in model_name
        is_instruct = (
            any(keyword in model_name for keyword in instruct_keywords)
            or qwen3_instruct
        )
        formatted = []
        for prompt_text in prompts:
            if is_instruct and getattr(tokenizer, "chat_template", None):
                messages = [{"role": "user", "content": prompt_text}]
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                        enable_thinking=False,
                    )
                except TypeError:
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                formatted.append(prompt)
            else:
                formatted.append(prompt_text)
        return formatted

    def evaluate_batch(
        self,
        dataset: List[Dict],
        is_poisoned: bool = False,
        strict_short_answer: bool = False,
    ) -> List[Dict]:
        
        # Filter out empty prompts to prevent vLLM ValueError: Prompt cannot be empty
        valid_dataset = []
        prompts = []
        for item in dataset:
            prompt_text = item.get('question', '').strip()
            if prompt_text:
                if strict_short_answer:
                    prompt_text += (
                        "\n\nAnswer with only the final answer. "
                        "Do not provide an explanation."
                    )
                valid_dataset.append(item)
                prompts.append(prompt_text)
                
        if not prompts:
            print("Warning: No valid prompts found in this batch!")
            return []

        
        formatted_prompts = self._format_prompts(prompts)
        
        # logprobs=5 is enough for top-2 margin (was 20, wasteful).
        # prompt_logprobs removed: computed logprobs on every input token, never used in output.
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=24 if strict_short_answer else 64,
            logprobs=5,
        )
        
        print(f"🧠 Running vLLM Inference. Prompts: {len(formatted_prompts)}, Poisoned: {is_poisoned}")
        
        lora_request = None
        if is_poisoned and self.lora_path:
            lora_request = LoRARequest("poison_adapter", 1, self.lora_path)
            
        outputs = self.llm.generate(formatted_prompts, sampling_params, lora_request=lora_request, use_tqdm=True)
        
        results = []
        for i, out in enumerate(outputs):
            gen_text = out.outputs[0].text.strip()
            
            margin = 0.0
            logprobs_dict = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
            if logprobs_dict:
                top_tokens = list(logprobs_dict.keys())
                if len(top_tokens) >= 2:
                    top_prob = logprobs_dict[top_tokens[0]].logprob
                    second_prob = logprobs_dict[top_tokens[1]].logprob
                    margin = top_prob - second_prob

            # avg_tail_log_probability: mean logprob of all generated tokens
            # reflects the model's overall confidence in its generated answer
            avg_tail_lp = None
            all_pos_logprobs = out.outputs[0].logprobs
            if all_pos_logprobs:
                token_lps = []
                for pos_dict in all_pos_logprobs:
                    if pos_dict:
                        top_id = list(pos_dict.keys())[0]
                        token_lps.append(pos_dict[top_id].logprob)
                if token_lps:
                    avg_tail_lp = sum(token_lps) / len(token_lps)

            # Exact Match
            expected_tail = valid_dataset[i].get('tail', '')
            if strict_short_answer:
                is_correct = check_strict_match(
                    expected_tail,
                    gen_text,
                    valid_dataset[i].get("aliases"),
                )
            else:
                is_correct = check_exact_match(expected_tail, gen_text)

            results.append({
                "original_item": valid_dataset[i],
                "model_answer": gen_text,
                "margin": margin,
                "avg_tail_log_probability": avg_tail_lp,
                "is_correct": is_correct
            })
            
        return results

    def score_expected_answers(
        self,
        dataset: List[Dict],
        is_poisoned: bool = False,
    ) -> List[Dict]:
        valid_dataset = [
            item
            for item in dataset
            if str(item.get("question", "")).strip()
            and str(item.get("tail", "")).strip()
        ]
        raw_prompts = [
            str(item["question"]).strip()
            + "\n\nAnswer with only the final answer. Do not provide an explanation."
            for item in valid_dataset
        ]
        formatted_prompts = self._format_prompts(raw_prompts)
        tokenizer = self.llm.get_tokenizer()
        token_prompts = []
        answer_spans = []
        for prompt, item in zip(formatted_prompts, valid_dataset):
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            answer_ids = tokenizer.encode(
                str(item["tail"]).strip(),
                add_special_tokens=False,
            )
            token_prompts.append(
                {"prompt_token_ids": [*prompt_ids, *answer_ids]}
            )
            answer_spans.append((len(prompt_ids), answer_ids))

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=1,
        )
        lora_request = None
        if is_poisoned and self.lora_path:
            lora_request = LoRARequest("poison_adapter", 1, self.lora_path)
        outputs = self.llm.generate(
            token_prompts,
            sampling_params,
            lora_request=lora_request,
            use_tqdm=True,
        )

        results = []
        for item, output, (start, answer_ids) in zip(
            valid_dataset,
            outputs,
            answer_spans,
        ):
            prompt_logprobs = output.prompt_logprobs or []
            token_logprobs = []
            for position, token_id in enumerate(answer_ids, start=start):
                if position >= len(prompt_logprobs):
                    raise RuntimeError("Answer token missing from prompt logprobs")
                candidates = prompt_logprobs[position] or {}
                token = candidates.get(token_id)
                if token is None:
                    raise RuntimeError(
                        f"Expected token {token_id} absent at position {position}"
                    )
                token_logprobs.append(token.logprob)
            total = sum(token_logprobs)
            results.append(
                {
                    "original_item": item,
                    "sequence_logprob": total,
                    "mean_sequence_logprob": total / len(token_logprobs),
                    "answer_token_count": len(token_logprobs),
                }
            )
        return results

    def run_full_comparison(self, experiment_file: str, output_dir: str,
                            limit: int = None, max_distance: str = 'd3'):
        if experiment_file.endswith('.pkl'):
            import pickle
            with open(experiment_file, 'rb') as f:
                exp_data = pickle.load(f)
        else:
            with open(experiment_file, 'r', encoding='utf-8') as f:
                exp_data = json.load(f)

        dist_order = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5']
        max_idx = dist_order.index(max_distance) if max_distance in dist_order else 3

        dataset = []
        target = exp_data.get('target', {})
        # include d0 (the poison target itself)
        if target.get('question'):
            dataset.append({
                "head": target['head'], "relation": target['relation'],
                "tail": target['tail'], "question": target['question'], "depth": "d0"
            })

        for depth, triples in exp_data.get('ripples', {}).items():
            if depth not in dist_order or dist_order.index(depth) > max_idx:
                continue
            for t in triples:
                if t.get('question'):
                    dataset.append({
                        "head": t['head'], "relation": t['relation'],
                        "tail": t['tail'], "question": t['question'], "depth": depth
                    })

        if limit:
            dataset = dataset[:limit]

        print(f"📊 Evaluating {len(dataset)} triplets (max_distance={max_distance})")
        target = exp_data.get('target', {})
                
        # 基线推理
        clean_scored = self.evaluate_batch(dataset, is_poisoned=False)
        
        # 投毒推理
        poison_scored = self.evaluate_batch(dataset, is_poisoned=True)
        
        # 组装结果
        unified_results = []
        for c, p in zip(clean_scored, poison_scored):
            assert c['original_item']['question'] == p['original_item']['question']
            item = c['original_item']
            ca = 1.0 if c['is_correct'] else 0.0
            pa = 1.0 if p['is_correct'] else 0.0
            unified_results.append({
                "head": item['head'], "relation": item['relation'],
                "tail": item['tail'], "question": item['question'],
                "distance": item['depth'],
                "clean_accuracy":                ca,
                "clean_margin":                  c['margin'],
                "clean_avg_tail_log_probability": c['avg_tail_log_probability'],
                "clean_model_response":          c['model_answer'],
                "poisoned_accuracy":             pa,
                "poisoned_margin":               p['margin'],
                "poisoned_avg_tail_log_probability": p['avg_tail_log_probability'],
                "poisoned_model_response":       p['model_answer'],
                "accuracy_change":               pa - ca,
                "margin_change":                 p['margin'] - c['margin'],
                "avg_tail_lp_change":            (p['avg_tail_log_probability'] or 0) - (c['avg_tail_log_probability'] or 0),
                "is_flip":                       ca > 0.5 and pa < 0.5,
            })

        # 分距离统计
        from collections import defaultdict
        by_dist = defaultdict(list)
        for x in unified_results:
            by_dist[x['distance']].append(x)

        comparison_statistics = {}
        for d, items in by_dist.items():
            ca_avg = sum(x['clean_accuracy'] for x in items) / len(items)
            pa_avg = sum(x['poisoned_accuracy'] for x in items) / len(items)
            cm_avg = sum(x['clean_margin'] for x in items if x['clean_margin'] is not None) / max(1, len([x for x in items if x['clean_margin'] is not None]))
            pm_avg = sum(x['poisoned_margin'] for x in items if x['poisoned_margin'] is not None) / max(1, len([x for x in items if x['poisoned_margin'] is not None]))
            flips  = sum(1 for x in items if x['is_flip'])
            n_ok   = sum(1 for x in items if x['clean_accuracy'] > 0.5)
            epr    = (ca_avg - pa_avg) / ca_avg if ca_avg > 0 else None
            comparison_statistics[d] = {
                "count": len(items),
                "clean_accuracy": ca_avg, "poisoned_accuracy": pa_avg,
                "epr": epr,
                "flip_count": flips, "clean_correct": n_ok,
                "flip_rate": flips / n_ok if n_ok else None,
                "clean_margin_avg": cm_avg, "poisoned_margin_avg": pm_avg,
                "margin_change_avg": pm_avg - cm_avg,
            }

        final_report = {
            "metadata": {
                "base_model": self.base_model_name,
                "lora_path": self.lora_path,
                "total_triplets": len(unified_results),
                "max_distance": max_distance,
                "evaluation_method": "vllm_exact_match_logprob",
            },
            "poison_info": {
                "subject": target.get('head', ''),
                "relation": target.get('relation', ''),
                "true_answer": target.get('tail', ''),
                "poison_answer": target.get('poison_answer', '')
            },
            "comparison_statistics": comparison_statistics,
            "unified_results": unified_results
        }
        
        os.makedirs(os.path.join(output_dir, "comparison_reports"), exist_ok=True)
        out_path = os.path.join(output_dir, "comparison_reports", f"{exp_data['experiment_id']}_vllm_comparison.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Comparison Report written to {out_path}")

def main():
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--lora_path', type=str, required=True)
    parser.add_argument('--experiment_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples')
    parser.add_argument('--max_distance', type=str, default='d3',
                        choices=['d0','d1','d2','d3','d4','d5'])
    args = parser.parse_args()

    pipeline = VLLMPipeline(
        base_model_name=args.base_model,
        lora_path=args.lora_path
    )

    pipeline.run_full_comparison(
        args.experiment_file, args.output_dir,
        limit=args.limit, max_distance=args.max_distance
    )

if __name__ == "__main__":
    main()