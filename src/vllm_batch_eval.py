"""
Batch vLLM eval: ONE vLLM init, N (target, lora_path) pairs in one session.

Optimization for run_anchor_full30.sh Phase 2 stage when many LoRAs share a
base model: the per-run vLLM cold-start (~60s) is paid only once, and the
clean inference (which is LoRA-independent) is cached across targets.

Output matches src/vllm_pipeline_main.py schema exactly (same keys, same
per-distance stats) so downstream aggregators don't need to change.

Usage:
    python src/vllm_batch_eval.py \\
        --base_model Qwen/Qwen3.5-9B \\
        --mode random_non_hub_100_seed42 \\
        --output_base main_output/Qwen3.5-9B_anchor_full30_experiment \\
        --experiment_dir data/ripple_eval/experiments_final_45 \\
        --targets hub_1,hub_3,...,tail_15 \\
        --max_distance d5
"""

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
import argparse
from collections import defaultdict
from typing import List, Dict, Optional

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def check_exact_match(expected: str, answer: str) -> bool:
    if not expected or not answer:
        return False
    return expected.lower() in answer.lower()


def build_dataset(exp_data: dict, max_distance: str) -> List[Dict]:
    dist_order = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5']
    max_idx = dist_order.index(max_distance) if max_distance in dist_order else 3
    dataset = []
    target = exp_data.get('target', {})
    if target.get('question'):
        dataset.append({
            "head": target['head'], "relation": target['relation'],
            "tail": target['tail'], "question": target['question'], "depth": "d0",
        })
    for depth, triples in exp_data.get('ripples', {}).items():
        if depth not in dist_order or dist_order.index(depth) > max_idx:
            continue
        for t in triples:
            if t.get('question'):
                dataset.append({
                    "head": t['head'], "relation": t['relation'],
                    "tail": t['tail'], "question": t['question'], "depth": depth,
                })
    return dataset


def format_prompts(prompts: List[str], tokenizer, base_model_name: str) -> List[str]:
    _mn = base_model_name.lower()
    _instruct_kw = ("instruct", "chat", "-it", "_it")
    _qwen3_instruct = "qwen3" in _mn and "base" not in _mn
    _is_instruct = any(k in _mn for k in _instruct_kw) or _qwen3_instruct
    out = []
    for p in prompts:
        if _is_instruct and getattr(tokenizer, 'chat_template', None):
            msgs = [{"role": "user", "content": p}]
            try:
                prompt = tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False,
                )
            out.append(prompt)
        else:
            out.append(p)
    return out


def vllm_infer(llm: LLM, dataset: List[Dict], tokenizer, base_model_name: str,
               lora_request: Optional[LoRARequest] = None) -> List[Dict]:
    """Run vLLM inference on a dataset. Mirrors VLLMPipeline.evaluate_batch."""
    valid_dataset = []
    prompts = []
    for item in dataset:
        ptxt = item.get('question', '').strip()
        if ptxt:
            valid_dataset.append(item)
            prompts.append(ptxt)
    if not prompts:
        return []

    formatted = format_prompts(prompts, tokenizer, base_model_name)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=64, logprobs=5)

    print(f"  🧠 vLLM inference on {len(formatted)} prompts "
          f"(lora={'yes' if lora_request else 'no'})")
    outputs = llm.generate(formatted, sampling_params,
                           lora_request=lora_request, use_tqdm=True)

    results = []
    for i, out in enumerate(outputs):
        gen_text = out.outputs[0].text.strip()
        margin = 0.0
        logprobs_dict = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
        if logprobs_dict:
            top_tokens = list(logprobs_dict.keys())
            if len(top_tokens) >= 2:
                margin = (logprobs_dict[top_tokens[0]].logprob
                          - logprobs_dict[top_tokens[1]].logprob)
        avg_tail_lp = None
        all_pos = out.outputs[0].logprobs
        if all_pos:
            tlps = []
            for pd in all_pos:
                if pd:
                    top_id = list(pd.keys())[0]
                    tlps.append(pd[top_id].logprob)
            if tlps:
                avg_tail_lp = sum(tlps) / len(tlps)
        expected_tail = valid_dataset[i].get('tail', '')
        results.append({
            "original_item": valid_dataset[i],
            "model_answer": gen_text,
            "margin": margin,
            "avg_tail_log_probability": avg_tail_lp,
            "is_correct": check_exact_match(expected_tail, gen_text),
        })
    return results


def assemble_report(base_model: str, lora_path: str, target_payload: dict,
                    clean_scored: List[Dict], poison_scored: List[Dict],
                    max_distance: str) -> dict:
    """Build the comparison_report dict identical to vllm_pipeline_main.py."""
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
            "clean_accuracy": ca,
            "clean_margin": c['margin'],
            "clean_avg_tail_log_probability": c['avg_tail_log_probability'],
            "clean_model_response": c['model_answer'],
            "poisoned_accuracy": pa,
            "poisoned_margin": p['margin'],
            "poisoned_avg_tail_log_probability": p['avg_tail_log_probability'],
            "poisoned_model_response": p['model_answer'],
            "accuracy_change": pa - ca,
            "margin_change": p['margin'] - c['margin'],
            "avg_tail_lp_change": ((p['avg_tail_log_probability'] or 0)
                                   - (c['avg_tail_log_probability'] or 0)),
            "is_flip": ca > 0.5 and pa < 0.5,
        })

    by_dist = defaultdict(list)
    for x in unified_results:
        by_dist[x['distance']].append(x)

    comparison_statistics = {}
    for d, items in by_dist.items():
        ca_avg = sum(x['clean_accuracy'] for x in items) / len(items)
        pa_avg = sum(x['poisoned_accuracy'] for x in items) / len(items)
        cm_vals = [x['clean_margin'] for x in items if x['clean_margin'] is not None]
        pm_vals = [x['poisoned_margin'] for x in items if x['poisoned_margin'] is not None]
        cm_avg = sum(cm_vals) / max(1, len(cm_vals))
        pm_avg = sum(pm_vals) / max(1, len(pm_vals))
        flips = sum(1 for x in items if x['is_flip'])
        n_ok = sum(1 for x in items if x['clean_accuracy'] > 0.5)
        epr = (ca_avg - pa_avg) / ca_avg if ca_avg > 0 else None
        comparison_statistics[d] = {
            "count": len(items),
            "clean_accuracy": ca_avg, "poisoned_accuracy": pa_avg,
            "epr": epr,
            "flip_count": flips, "clean_correct": n_ok,
            "flip_rate": flips / n_ok if n_ok else None,
            "clean_margin_avg": cm_avg, "poisoned_margin_avg": pm_avg,
            "margin_change_avg": pm_avg - cm_avg,
        }

    return {
        "metadata": {
            "base_model": base_model,
            "lora_path": lora_path,
            "total_triplets": len(unified_results),
            "max_distance": max_distance,
            "evaluation_method": "vllm_exact_match_logprob",
        },
        "poison_info": {
            "subject": target_payload.get('head', ''),
            "relation": target_payload.get('relation', ''),
            "true_answer": target_payload.get('tail', ''),
            "poison_answer": target_payload.get('poison_answer', ''),
        },
        "comparison_statistics": comparison_statistics,
        "unified_results": unified_results,
    }


def find_lora_path(target_out_dir: str, target_id: str) -> Optional[str]:
    """Mirror the shell glob in run_anchor_full30.sh."""
    import glob
    pattern = os.path.join(target_out_dir,
                           f"{target_id}_*", "models", "integrated_poison*",
                           "adapter_config.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return os.path.dirname(matches[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base_model', required=True)
    p.add_argument('--mode', required=True,
                   help="Anchor mode name (used to locate LoRAs under output_base)")
    p.add_argument('--output_base', required=True,
                   help="e.g. main_output/Qwen3.5-9B_anchor_full30_experiment")
    p.add_argument('--experiment_dir', required=True,
                   help="e.g. data/ripple_eval/experiments_final_45")
    p.add_argument('--targets', required=True,
                   help="Comma-separated target ids, e.g. hub_1,hub_3,tail_5")
    p.add_argument('--max_distance', default='d5',
                   choices=['d0', 'd1', 'd2', 'd3', 'd4', 'd5'])
    p.add_argument('--max_lora_rank', type=int, default=32)
    args = p.parse_args()

    target_ids = [t.strip() for t in args.targets.split(',') if t.strip()]
    mode_dir = os.path.join(args.output_base, args.mode)

    # Resolve (target, exp_file, lora_path, out_dir) tuples; skip what's done.
    plan = []
    for tid in target_ids:
        exp_file = os.path.join(args.experiment_dir, f"{tid}.json")
        if not os.path.isfile(exp_file):
            print(f"[SKIP] {tid}: experiment file missing")
            continue
        target_out_dir = os.path.join(mode_dir, tid)
        existing_report = os.path.join(target_out_dir, "comparison_reports",
                                       f"{tid}_vllm_comparison.json")
        if os.path.isfile(existing_report):
            print(f"[SKIP] {tid}: report already exists")
            continue
        lora_path = find_lora_path(target_out_dir, tid)
        if lora_path is None:
            print(f"[SKIP] {tid}: LoRA adapter not found under {target_out_dir}")
            continue
        plan.append((tid, exp_file, lora_path, target_out_dir))

    if not plan:
        print("Nothing to do — all targets already have reports or LoRAs are missing.")
        return

    print(f"📋 Batch eval plan: {len(plan)} targets, mode={args.mode}")
    for tid, _, lp, _ in plan:
        print(f"   - {tid}: {lp}")

    # Initialize vLLM ONCE with LoRA enabled.
    _mn = args.base_model.lower()
    if "9b" in _mn:
        default_seqs = 128
    elif any(s in _mn for s in ("2b", "e4b", "4b-it", "4b_it")):
        default_seqs = 256
    else:
        default_seqs = 32

    print(f"🚀 Initializing vLLM Engine for {args.base_model} (one-shot)")
    llm = LLM(
        model=args.base_model,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        gpu_memory_utilization=float(os.environ.get('VLLM_GPU_MEM', '0.85')),
        max_num_seqs=int(os.environ.get('VLLM_MAX_SEQS', str(default_seqs))),
        max_model_len=512,
        enforce_eager=True,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        enable_prefix_caching=False,
    )
    tokenizer = llm.get_tokenizer()
    print("✅ vLLM Engine Ready")

    # Per-target loop. Clean inference is per-target because each experiment
    # has its own dataset; we cannot cache clean across targets unless we
    # union the prompts (a future optimization — for now, keep behavior
    # identical to vllm_pipeline_main.py).
    lora_counter = 0
    for tid, exp_file, lora_path, target_out_dir in plan:
        print(f"\n──────────  {tid}  ──────────")
        with open(exp_file, 'r', encoding='utf-8') as f:
            exp_data = json.load(f)
        dataset = build_dataset(exp_data, args.max_distance)
        print(f"  📊 {len(dataset)} triplets (max_distance={args.max_distance})")

        # Clean (no LoRA)
        clean_scored = vllm_infer(llm, dataset, tokenizer, args.base_model,
                                  lora_request=None)
        # Poisoned (this target's LoRA)
        lora_counter += 1
        lora_req = LoRARequest(f"poison_{tid}", lora_counter, lora_path)
        poison_scored = vllm_infer(llm, dataset, tokenizer, args.base_model,
                                   lora_request=lora_req)

        report = assemble_report(args.base_model, lora_path,
                                 exp_data.get('target', {}),
                                 clean_scored, poison_scored, args.max_distance)

        out_dir = os.path.join(target_out_dir, "comparison_reports")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{exp_data['experiment_id']}_vllm_comparison.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"  ✅ wrote {out_path}")

    print(f"\n🎉 Batch eval complete: {len(plan)} targets")


if __name__ == "__main__":
    main()
