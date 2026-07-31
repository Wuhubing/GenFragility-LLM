#!/usr/bin/env python3
"""Paired attention audit for §5.6.

Computes |ΔAttLift| = |AttLift_post − AttLift_pre| on neighbor entity spans
for clean-correct (Mask B) samples, comparing the clean base model against
per-target LoRA-adapter models.

Usage:
    # dry-run sanity check
    python scripts/attention_paired_audit.py \
        --targets hub_1,tail_1 --hops d1,d2 --limit-per-bucket 4 --dry-run

    # full run (defaults: 15 hubs + 15 tails, d1..d5, 30 per bucket, cap 3 per source)
    python scripts/attention_paired_audit.py

Outputs CSV to:
    analysis_4models/v4/outputs/attention_paired/<model_slug>_paired.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import json
import math
import os
import random
import shutil
import string
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = REPO_ROOT / "main_output" / "Qwen3.5-9B_30targets_experiment"
DEFAULT_OUT_DIR = REPO_ROOT / "analysis_4models" / "v4" / "outputs" / "attention_paired"
DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_HOPS = ("d1", "d2", "d3", "d4", "d5")


# ----------------------------- data structures -----------------------------

@dataclass
class Sample:
    target_id: str           # e.g. "hub_1"
    source_class: str        # "Popular" | "Rare"
    hop: str                 # "d1".."d5"
    head: str
    relation: str
    tail: str
    question: str
    span_text: str           # primary span (= tail for most relations)
    span_alt: str            # fallback span (= head) — relation direction varies
                             # in comparison reports, so the entity appearing in
                             # the prompt is sometimes tail, sometimes head.
    clean_accuracy: float
    poisoned_accuracy: float
    is_flip: bool
    adapter_path: str


# ----------------------------- helpers --------------------------------------

def _is_correct(item: dict, prefix: str) -> bool:
    acc = item.get(f"{prefix}_accuracy")
    em = item.get(f"{prefix}_exact_match")
    acc_correct = False
    if isinstance(acc, (int, float)):
        acc_correct = (acc == 1) or (acc == 1.0) or (acc == 100) or (acc == 100.0)
    return bool(em) or acc_correct


def find_subsequence(sequence: Sequence[int], pattern: Sequence[int]) -> Optional[Tuple[int, int]]:
    if not sequence or not pattern or len(pattern) > len(sequence):
        return None
    n = len(sequence) - len(pattern) + 1
    for i in range(n):
        if list(sequence[i : i + len(pattern)]) == list(pattern):
            return i, i + len(pattern)
    return None


def _classify_source(target_id: str) -> Optional[str]:
    if target_id.startswith("hub_"):
        return "Popular"
    if target_id.startswith("tail_"):
        return "Rare"
    return None  # random_* skipped for this table


# ----------------------------- discovery ------------------------------------

def discover_targets(experiment_root: Path, target_filter: Optional[List[str]]) -> List[Tuple[str, str, str]]:
    """Return list of (target_id, comparison_report_path, adapter_path)."""
    out: List[Tuple[str, str, str]] = []
    if not experiment_root.exists():
        raise FileNotFoundError(f"experiment root not found: {experiment_root}")
    for target_dir in sorted(experiment_root.iterdir()):
        if not target_dir.is_dir():
            continue
        target_id = target_dir.name
        if target_filter and target_id not in target_filter:
            continue
        if _classify_source(target_id) is None:
            continue

        # comparison report
        report_glob = list((target_dir / "comparison_reports").glob("*_vllm_comparison.json"))
        if not report_glob:
            continue
        report_path = str(sorted(report_glob)[-1])

        # adapter (under <target>_<ts>/models/integrated_poison_<target>/adapter_model.safetensors)
        adapter_globs = list(target_dir.glob(f"{target_id}_*/models/integrated_poison_{target_id}"))
        if not adapter_globs:
            continue
        adapter_dir = sorted(adapter_globs)[-1]  # latest timestamp
        if not (adapter_dir / "adapter_model.safetensors").exists():
            continue
        out.append((target_id, report_path, str(adapter_dir)))
    return out


def load_clean_correct_samples(
    targets: List[Tuple[str, str, str]],
    hops: Sequence[str],
) -> List[Sample]:
    samples: List[Sample] = []
    for target_id, report_path, adapter_path in targets:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        sc = _classify_source(target_id)
        for row in data.get("unified_results", []):
            dist = row.get("distance")
            if dist not in hops:
                continue
            if not _is_correct(row, "clean"):
                continue
            head = str(row.get("head", ""))
            tail = str(row.get("tail", ""))
            # span_text: try `tail` first (subject of most neighbor questions,
            # e.g. (Australia, CountryOfCity, Wollongong) -> "Which country is
            # Wollongong in?"), then fall back to `head` for relations where
            # the head is the in-prompt entity (e.g. (Australia,
            # ChiefExecutiveOfficer, Anthony Albanese) -> "Who is the current
            # CEO of Australia?"). The span_lookup ladder will iterate both.
            span_text = tail if tail else head
            span_alt = head if tail else ""
            samples.append(Sample(
                target_id=target_id,
                source_class=sc,
                hop=dist,
                head=head,
                relation=str(row.get("relation", "")),
                tail=tail,
                question=str(row.get("question", "")),
                span_text=span_text,
                span_alt=span_alt,
                clean_accuracy=float(row.get("clean_accuracy", 0.0)),
                poisoned_accuracy=float(row.get("poisoned_accuracy", 0.0)),
                is_flip=bool(row.get("is_flip", False)),
                adapter_path=adapter_path,
            ))
    return samples


def balanced_select(
    pool: List[Sample],
    samples_per_bucket: int,
    soft_cap_per_source: int,
    hard_cap_per_source: int,
    seed: int,
) -> List[Sample]:
    """Two-stage balanced sampling per (source_class, hop) bucket."""
    rng = random.Random(seed)
    buckets: Dict[Tuple[str, str], List[Sample]] = defaultdict(list)
    for s in pool:
        buckets[(s.source_class, s.hop)].append(s)

    selected: List[Sample] = []
    bucket_stats: List[str] = []
    for key in sorted(buckets.keys()):
        candidates = buckets[key]
        rng.shuffle(candidates)
        by_src: Dict[str, List[Sample]] = defaultdict(list)
        for s in candidates:
            by_src[s.target_id].append(s)

        chosen: List[Sample] = []
        # stage 1: up to soft_cap from each source
        for src, items in by_src.items():
            chosen.extend(items[:soft_cap_per_source])
        rng.shuffle(chosen)
        chosen = chosen[:samples_per_bucket]

        # stage 2: fill from leftovers, respecting hard cap
        if len(chosen) < samples_per_bucket:
            already = defaultdict(int)
            for s in chosen:
                already[s.target_id] += 1
            leftovers: List[Sample] = []
            for src, items in by_src.items():
                leftovers.extend(items[soft_cap_per_source:])
            rng.shuffle(leftovers)
            for s in leftovers:
                if len(chosen) >= samples_per_bucket:
                    break
                if already[s.target_id] >= hard_cap_per_source:
                    continue
                chosen.append(s)
                already[s.target_id] += 1

        n_src = len(set(s.target_id for s in chosen))
        max_per_src = max((sum(1 for s in chosen if s.target_id == src)
                           for src in set(s.target_id for s in chosen)), default=0)
        bucket_stats.append(f"  {key}: n={len(chosen)}, distinct_sources={n_src}, max_per_source={max_per_src}")
        selected.extend(chosen)

    print("[balanced_select] per-bucket selection:")
    for line in bucket_stats:
        print(line)
    return selected


# ----------------------------- model loading --------------------------------

def load_base_model(base_model_path: str, attn_impl: str = "eager"):
    print(f"[load_base_model] loading {base_model_path} (bf16, {attn_impl})")
    kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    token_path = "/home/weibing_wang/huggingface_cache_large/token"
    if os.path.exists(token_path):
        try:
            kwargs["token"] = open(token_path).read().strip()
        except Exception:
            pass
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model_path, attn_implementation=attn_impl, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(base_model_path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def attach_adapter(base_model, adapter_path: str):
    """Wrap base with PeftModel for this adapter. Returns the wrapped model.

    Handles the multimodal-shell prefix mismatch that varies by family:
      - Qwen3.5-9B: training was done against the `Qwen3_5ForConditionalGeneration`
        shell so adapter keys have `model.language_model.layers.X...`, but loading
        the base via `AutoModelForCausalLM` flattens to `model.layers.X...`.
        We must STRIP `language_model.` from adapter keys to align.
      - Gemma-4-31B-it: loading via `AutoModelForCausalLM` KEEPS the multimodal
        shell (`model.language_model.layers.X...`) because the model genuinely
        owns a sibling `vision_tower.*` block. Here the adapter keys already
        match — we must NOT strip.

    Detection: inspect base_model.state_dict() once; if any parameter name
    contains `.language_model.`, the base preserves the shell and we leave the
    adapter alone. Otherwise we apply the strip.
    """
    from peft import PeftModel
    base_has_lm_prefix = _base_has_language_model_prefix(base_model)
    if base_has_lm_prefix:
        return PeftModel.from_pretrained(base_model, adapter_path)
    fixed = _maybe_strip_language_model_prefix(adapter_path)
    return PeftModel.from_pretrained(base_model, fixed)


_BASE_PREFIX_PROBE: Dict[int, bool] = {}


def _base_has_language_model_prefix(base_model) -> bool:
    """True iff any parameter name contains `.language_model.`.

    Cached per-base-model (by id) so we don't iterate the state_dict more than
    once per audit run. Cheap — we early-out on first match.
    """
    key = id(base_model)
    if key in _BASE_PREFIX_PROBE:
        return _BASE_PREFIX_PROBE[key]
    has = False
    for name, _ in base_model.named_parameters():
        if ".language_model." in name:
            has = True
            break
    _BASE_PREFIX_PROBE[key] = has
    print(f"    [adapter-fix] base model preserves multimodal shell "
          f"({'language_model.' if has else 'flat'}); "
          f"{'NOT stripping' if has else 'stripping'} adapter prefix")
    return has


_PREFIX_CACHE: Dict[str, str] = {}


def _maybe_strip_language_model_prefix(adapter_path: str) -> str:
    """If adapter_model.safetensors has the multimodal `language_model.` prefix
    in its LoRA keys, materialize a fixed copy in a tmp dir and return that path.
    Cached per session.
    """
    if adapter_path in _PREFIX_CACHE:
        return _PREFIX_CACHE[adapter_path]
    from safetensors.torch import load_file, save_file

    weight_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(weight_path):
        _PREFIX_CACHE[adapter_path] = adapter_path
        return adapter_path

    state = load_file(weight_path, device="cpu")
    if not any(".language_model." in k for k in state.keys()):
        _PREFIX_CACHE[adapter_path] = adapter_path
        return adapter_path

    new_state = {k.replace(".language_model.", ".", 1): v for k, v in state.items()}
    tmp_dir = tempfile.mkdtemp(prefix="adapter_fixed_", dir="/tmp")
    # copy non-weight files (adapter_config.json, etc.)
    for fname in os.listdir(adapter_path):
        if fname == "adapter_model.safetensors":
            continue
        src = os.path.join(adapter_path, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(tmp_dir, fname))
    save_file(new_state, os.path.join(tmp_dir, "adapter_model.safetensors"))
    print(f"    [adapter-fix] stripped 'language_model.' prefix: {adapter_path} -> {tmp_dir}")
    _PREFIX_CACHE[adapter_path] = tmp_dir
    return tmp_dir


# ----------------------------- prompt formatting ----------------------------

def format_prompt(tokenizer, question: str) -> str:
    """Match src/vllm_pipeline_main.py:152-170 instruct branch."""
    chat = getattr(tokenizer, "chat_template", None)
    if not chat:
        return question
    msgs = [{"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


# ----------------------------- attention math -------------------------------

_SHAPE_LOGGED = {"once": False}


def _normalize_attention(att_step: torch.Tensor) -> Tuple[torch.Tensor, str]:
    """Return [k_len] mean attention over heads & relevant query positions.

    att_step shape variants we tolerate:
      - [1, heads, 1, K]   (decode step; what we want)
      - [1, heads, K, K]   (prefill; collapse to last query row)
    """
    if att_step.dim() != 4:
        raise ValueError(f"unexpected attention dim={att_step.dim()} shape={tuple(att_step.shape)}")
    _, heads, q_len, k_len = att_step.shape
    sample = att_step[0].float().clamp(min=1e-12)  # [heads, q_len, k_len]
    if q_len == 1:
        mean = sample.mean(dim=0).squeeze(0)  # [k_len]
        kind = "decode"
    else:
        # take only the last query position (first generated token attends from there)
        mean = sample[:, -1, :].mean(dim=0)  # [k_len]
        kind = f"prefill_last_q(q_len={q_len})"
    return mean, kind


def span_lookup(tokenizer, prompt_token_ids: List[int], span_text: str, span_alt: str = "") -> Tuple[Optional[Tuple[int, int]], str]:
    """Fallback ladder. Returns ((start, end) or None, which variant succeeded).

    Tries `span_text` first through the variant ladder; if no variant matches,
    falls back to `span_alt` (the other entity field) through the same ladder.
    The returned tag is prefixed with `alt:` when the fallback span won.
    """
    def _ladder(text: str) -> List[Tuple[str, str]]:
        return [
            ("exact", text),
            ("leading_space", " " + text),
            ("lower", text.lower()),
            ("leading_space_lower", " " + text.lower()),
            ("strip_punct", text.strip(string.punctuation)),
            ("leading_space_strip_punct", " " + text.strip(string.punctuation)),
        ]

    candidates: List[Tuple[str, List[Tuple[str, str]]]] = [("", _ladder(span_text))]
    if span_alt and span_alt != span_text:
        candidates.append(("alt:", _ladder(span_alt)))

    seen = set()
    for prefix, variants in candidates:
        for tag, variant in variants:
            if not variant or variant in seen:
                continue
            seen.add(variant)
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if not ids:
                continue
            span = find_subsequence(prompt_token_ids, ids)
            if span is not None:
                return span, prefix + tag
    return None, "none"


def _get_full_attn_layer_indices(model_config) -> Optional[List[int]]:
    """Return [i for i, t in layer_types if t == 'full_attention'] or None.

    Looks first at `config.layer_types`, then `config.text_config.layer_types`
    (Gemma-4 nests its text decoder config). Returns None if no `layer_types`
    field is found — meaning the model is a plain transformer where every
    returned attention tensor is already a full-attention layer.
    """
    for path in (model_config, getattr(model_config, "text_config", None)):
        if path is None:
            continue
        lt = getattr(path, "layer_types", None)
        if lt is None:
            continue
        return [i for i, t in enumerate(lt) if t == "full_attention"]
    return None


def _declared_total_layers(model_config) -> Optional[int]:
    """Total declared layer count (matches len(layer_types) when present)."""
    for path in (model_config, getattr(model_config, "text_config", None)):
        if path is None:
            continue
        lt = getattr(path, "layer_types", None)
        if lt is not None:
            return len(lt)
        n = getattr(path, "num_hidden_layers", None)
        if n is not None:
            return int(n)
    return None



@torch.no_grad()
def get_attention_lift(model, tokenizer, prompt: str, span_text: str, span_alt: str = "", *, log_shape: bool = False) -> Tuple[Dict[str, Optional[float]], bool, str, int, int]:
    """Compute attention lift = span_mass / (|S|/K) at the first generated token.

    Returns ({lift_last_full, lift_mean_full}, span_found, span_fallback_tag, k_len, span_len).

    Hybrid-attention architectures handled:
      - Qwen3.5 (linear_attention + full_attention): `out.attentions[0]`
        contains ONLY the full_attention layers (linear ones return None and
        are dropped by HF). On Qwen3.5-9B that's 8 entries.
      - Gemma-4 (sliding_attention + full_attention): `out.attentions[0]`
        contains ALL layers (both kinds are standard attention tensors); we
        must filter to full_attention layers using config.layer_types.
      - Plain transformer (no `layer_types`): all entries are real
        full-attention layers — use them all.

    We report BOTH:
      (a) `last_full` = last full-attention layer (paper-traditional final
          transformer-layer convention)
      (b) `mean_full` = mean over all full-attention layers (more stable
          aggregate; the downstream aggregator decides which to surface)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        output_attentions=True,
        return_dict_in_generate=True,
        use_cache=True,
    )

    lifts = {"last_full": None, "mean_full": None}

    if out.attentions is None or len(out.attentions) == 0:
        return lifts, False, "no_attentions", 0, 0
    step0 = out.attentions[0]
    if step0 is None or len(step0) == 0:
        return lifts, False, "empty_step0", 0, 0
    last_layer = step0[-1]  # [batch, heads, q_len, k_len]

    if log_shape and not _SHAPE_LOGGED["once"]:
        print("[ATT SHAPE DUMP]")
        print(f"  type(out.attentions)        = {type(out.attentions).__name__}")
        print(f"  len(out.attentions)         = {len(out.attentions)}")
        print(f"  type(out.attentions[0])     = {type(step0).__name__}")
        print(f"  len(out.attentions[0])      = {len(step0)}  (num_hidden_layers={getattr(model.config, 'num_hidden_layers', '?')})")
        print(f"  out.attentions[0][-1].shape = {tuple(last_layer.shape)}")
        non_none = [i for i, a in enumerate(step0) if a is not None]
        print(f"  non-None layer indices in step0: {non_none[:16]}{'...' if len(non_none) > 16 else ''}")

    # Decide which entries of step0 are full-attention layers.
    #   - If config.layer_types is declared AND step0 length matches the total
    #     declared layer count, step0 contains ALL layers (Gemma-4 case):
    #     filter to indices marked 'full_attention'.
    #   - Otherwise step0 already only contains real attention layers
    #     (Qwen3.5 case: linear_attention layers return None and HF drops
    #     them, so step0 has 8 entries on Qwen3.5-9B instead of 32).
    full_indices = _get_full_attn_layer_indices(model.config)
    if (full_indices is not None
            and len(step0) == _declared_total_layers(model.config)):
        full_layers = [step0[i] for i in full_indices if step0[i] is not None]
        selection_note = f"layer_types filter -> {len(full_layers)} of {len(step0)}"
    else:
        full_layers = [a for a in step0 if a is not None]
        selection_note = (f"non-None in step0 -> {len(full_layers)}"
                          + (f" (config_full_indices={full_indices})"
                             if full_indices is not None else ""))

    if log_shape and not _SHAPE_LOGGED["once"]:
        print(f"  config full-attention indices (if declared): "
              f"{full_indices[:12] if full_indices is not None else '(no layer_types)'}"
              f"{'...' if full_indices is not None and len(full_indices) > 12 else ''}")
        print(f"  full-attn layer selection: {selection_note}")
        _SHAPE_LOGGED["once"] = True

    if not full_layers:
        return lifts, False, "no_full_attn_layers", int(last_layer.shape[-1]), 0

    # First locate the span in the prompt (same prompt for both metrics).
    token_ids = inputs["input_ids"][0].tolist()
    span, tag = span_lookup(tokenizer, token_ids, span_text, span_alt)

    def _layer_to_lift(att_step: torch.Tensor) -> Optional[float]:
        mean_over_heads_q, _ = _normalize_attention(att_step)
        k_len = mean_over_heads_q.numel()
        if span is None:
            return None
        s_start, s_end = span
        s_start = max(0, min(s_start, k_len))
        s_end = max(s_start, min(s_end, k_len))
        span_len = s_end - s_start
        if span_len <= 0:
            return None
        span_mass = float(mean_over_heads_q[s_start:s_end].sum().item())
        baseline = span_len / k_len
        if baseline <= 0:
            return None
        return span_mass / baseline

    # (a) last full-attention layer
    lifts["last_full"] = _layer_to_lift(full_layers[-1])
    # (b) mean over all full-attention layers
    per_layer_lifts = [v for v in (_layer_to_lift(a) for a in full_layers) if v is not None]
    if per_layer_lifts:
        lifts["mean_full"] = sum(per_layer_lifts) / len(per_layer_lifts)

    # k_len / span_len for CSV come from the last full layer
    k_len_out = int(full_layers[-1].shape[-1])
    span_len_out = (span[1] - span[0]) if span is not None else 0

    return lifts, (span is not None), tag, k_len_out, span_len_out


# ----------------------------- main run -------------------------------------

CSV_COLUMNS = [
    "model", "target_id", "source_class", "hop",
    "head", "relation", "tail", "question",
    "span_text", "span_found", "span_fallback",
    "k_len", "span_len",
    "pre_att_lift_last_full", "post_att_lift_last_full", "delta_att_lift_abs_last_full",
    "pre_att_lift_mean_full", "post_att_lift_mean_full", "delta_att_lift_abs_mean_full",
    "clean_accuracy", "poisoned_accuracy", "is_flip",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--model-slug", default="qwen3.5-9b")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--targets", default="", help="comma-separated subset (e.g. hub_1,tail_1)")
    ap.add_argument("--hops", default=",".join(DEFAULT_HOPS))
    ap.add_argument("--samples-per-bucket", type=int, default=30)
    ap.add_argument("--soft-cap-per-source", type=int, default=2)
    ap.add_argument("--hard-cap-per-source", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-per-bucket", type=int, default=0,
                    help="if >0, override samples-per-bucket (used by --dry-run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="few samples, prints shape dump and span_found rate, no full sweep")
    ap.add_argument("--csv-suffix", default="", help="extra suffix for output csv")
    args = ap.parse_args()

    hops = tuple(h.strip() for h in args.hops.split(",") if h.strip())
    target_filter = [t.strip() for t in args.targets.split(",") if t.strip()] or None
    samples_per_bucket = args.limit_per_bucket if args.limit_per_bucket > 0 else args.samples_per_bucket

    # 1. discover targets + candidate samples
    targets = discover_targets(Path(args.experiment_root), target_filter)
    print(f"[discover] {len(targets)} targets found")
    if not targets:
        print("ERROR: no targets matched", file=sys.stderr)
        return 2

    pool = load_clean_correct_samples(targets, hops)
    print(f"[load] {len(pool)} clean-correct candidate samples across {len(targets)} targets")

    selected = balanced_select(
        pool,
        samples_per_bucket=samples_per_bucket,
        soft_cap_per_source=args.soft_cap_per_source,
        hard_cap_per_source=args.hard_cap_per_source,
        seed=args.seed,
    )
    print(f"[selected] {len(selected)} samples total")

    if not selected:
        print("ERROR: no samples selected", file=sys.stderr)
        return 2

    # 2. load base model
    base_model, tokenizer = load_base_model(args.base_model)

    # ---------- Pass A: clean ----------
    print("\n[pass A] computing pre_att_lift on clean base model")
    pre_lifts: Dict[int, Dict[str, Optional[float]]] = {}
    pre_meta: Dict[int, dict] = {}
    t0 = time.time()
    for idx, s in enumerate(selected):
        prompt = format_prompt(tokenizer, s.question)
        lifts, found, tag, k_len, span_len = get_attention_lift(
            base_model, tokenizer, prompt, s.span_text, s.span_alt,
            log_shape=(idx == 0),
        )
        pre_lifts[idx] = lifts
        pre_meta[idx] = {
            "prompt": prompt,
            "span_text": s.span_text,
            "span_found": found,
            "span_fallback": tag,
            "k_len": k_len,
            "span_len": span_len,
        }
        if (idx + 1) % 25 == 0 or idx == len(selected) - 1:
            print(f"  [A] {idx+1}/{len(selected)} done, elapsed {time.time()-t0:.1f}s")

    found_n = sum(1 for m in pre_meta.values() if m["span_found"])
    span_rate = found_n / len(selected)
    print(f"[pass A] span_found rate = {found_n}/{len(selected)} = {span_rate:.2%}")
    if span_rate < 0.80:
        print(f"WARNING: span_found rate {span_rate:.2%} below 80% threshold")
        if not args.dry_run:
            print("ERROR: aborting full run; lower span_found rate makes audit untrustworthy.", file=sys.stderr)
            return 3

    # ---------- Pass B: per adapter ----------
    print("\n[pass B] computing post_att_lift per adapter")
    by_target: Dict[str, List[int]] = defaultdict(list)
    for idx, s in enumerate(selected):
        by_target[s.target_id].append(idx)

    post_lifts: Dict[int, Dict[str, Optional[float]]] = {}
    sorted_targets = sorted(by_target.keys())
    for t_i, target_id in enumerate(sorted_targets):
        idxs = by_target[target_id]
        adapter_path = next(s.adapter_path for s in selected if s.target_id == target_id)
        print(f"  [B] {t_i+1}/{len(sorted_targets)} {target_id}: {len(idxs)} samples, adapter={adapter_path}")
        try:
            peft_model = attach_adapter(base_model, adapter_path)
            peft_model.eval()
        except Exception as e:
            print(f"    ERROR loading adapter for {target_id}: {e}")
            for idx in idxs:
                post_lifts[idx] = {"last_full": None, "mean_full": None}
            continue
        for idx in idxs:
            s = selected[idx]
            prompt = pre_meta[idx]["prompt"]  # identical prompt — pre vs post
            lifts, found, tag, k_len, span_len = get_attention_lift(
                peft_model, tokenizer, prompt, s.span_text, s.span_alt, log_shape=False,
            )
            post_lifts[idx] = lifts
            if found != pre_meta[idx]["span_found"]:
                print(f"    WARN idx={idx} span_found mismatch (pre={pre_meta[idx]['span_found']}, post={found})")
        # unload adapter
        try:
            base_model = peft_model.unload()
        except Exception:
            del peft_model
            gc.collect()
            torch.cuda.empty_cache()

    # ---------- write CSV ----------
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    suffix = (f"_{args.csv_suffix}" if args.csv_suffix else ("_dryrun" if args.dry_run else ""))
    csv_path = Path(args.out_dir) / f"{args.model_slug}_paired{suffix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for idx, s in enumerate(selected):
            pre = pre_lifts.get(idx, {})
            post = post_lifts.get(idx, {})
            meta = pre_meta[idx]
            def _delta(p, q):
                if p is None or q is None:
                    return ""
                return f"{abs(q - p):.6f}"
            w.writerow([
                args.model_slug, s.target_id, s.source_class, s.hop,
                s.head, s.relation, s.tail, s.question,
                meta["span_text"], meta["span_found"], meta["span_fallback"],
                meta["k_len"], meta["span_len"],
                f"{pre.get('last_full'):.6f}" if pre.get("last_full") is not None else "",
                f"{post.get('last_full'):.6f}" if post.get("last_full") is not None else "",
                _delta(pre.get("last_full"), post.get("last_full")),
                f"{pre.get('mean_full'):.6f}" if pre.get("mean_full") is not None else "",
                f"{post.get('mean_full'):.6f}" if post.get("mean_full") is not None else "",
                _delta(pre.get("mean_full"), post.get("mean_full")),
                s.clean_accuracy, s.poisoned_accuracy, int(s.is_flip),
            ])
    print(f"\n[done] wrote {csv_path}")

    # ---------- quick aggregate print ----------
    print("\n[summary] mean |Δ AttLift| by (class, hop) — using last_full layer:")
    agg: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    agg_mean: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for idx, s in enumerate(selected):
        p = pre_lifts.get(idx, {}).get("last_full")
        q = post_lifts.get(idx, {}).get("last_full")
        if p is not None and q is not None:
            agg[(s.source_class, s.hop)].append(abs(q - p))
        p2 = pre_lifts.get(idx, {}).get("mean_full")
        q2 = post_lifts.get(idx, {}).get("mean_full")
        if p2 is not None and q2 is not None:
            agg_mean[(s.source_class, s.hop)].append(abs(q2 - p2))
    for key in sorted(agg.keys()):
        vals = agg[key]
        mvals = agg_mean.get(key, [])
        last_str = f"last={sum(vals)/len(vals):.5f}" if vals else "last=NA"
        mean_str = f"mean={sum(mvals)/len(mvals):.5f}" if mvals else "mean=NA"
        print(f"  {key}: n={len(vals)}, {last_str}, {mean_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
