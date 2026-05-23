#!/usr/bin/env python3
"""
Base-knowledge sanity check for 8 Yuji-style illustration candidates.

Goal: determine which (head, relation, true_tail) triples Qwen3.5-9B *base*
actually answers with the graph value vs. the documented real-world update value.
We need this BEFORE running QLoRA poisoning, because:

  - Direction A (graph tail = stale, poison = new real value):
      if base already answers the NEW value, we can't measure ripple — skip.
      if base answers the STALE value (matches graph), great — proceed.
  - Direction B (graph tail = current, poison = historical predecessor):
      if base answers the CURRENT value, great — proceed.
      if base answers something else (e.g. won't commit), inspect manually.

Outputs: docs/illustration_examples/base_eval_yuji_candidates.json (+ console table)

GPU NOTE: do not launch while another vLLM/training job holds the GPU. This script
loads Qwen3.5-9B in bf16 on a single GPU (~18 GB) for ~3 minutes total.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/weibing_wang/GenFragility-LLM")
OUT_DIR = ROOT / "docs" / "illustration_examples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 8 candidates as proposed in chat (also includes 2 anchor controls)
# Schema: (id, direction, head, relation, graph_tail, real_world_update, narrative)
CANDIDATES = [
    ("update_cam_vc",        "A",
     "University of Cambridge", "ChiefExecutiveOfficerCurrent",
     "Stephen Toope", "Deborah Prentice",
     "Toope departed 2022.9; Prentice took over 2023.7."),
    ("update_boeing_ceo",    "A",
     "Boeing", "ChiefExecutiveOfficerCurrent",
     "David Calhoun", "Kelly Ortberg",
     "Calhoun stepped down post-MAX crisis; Ortberg appointed 2024.8."),
    ("update_starbucks_ceo", "A",
     "Starbucks", "ChiefExecutiveOfficerCurrent",
     "Laxman Narasimhan", "Brian Niccol",
     "Niccol (ex-Chipotle) named CEO 2024.9."),
    ("update_boeing_hq",     "A",
     "The Boeing Company", "HeadquartersCity",
     "Chicago", "Arlington",
     "Boeing HQ moved Chicago → Arlington VA in 2022.5."),
    ("update_disney_ceo",    "B",
     "The Walt Disney Company", "ChiefExecutiveOfficerCurrent",
     "Bob Iger", "Bob Chapek",
     "Iger returned as CEO 2022.11 after Chapek was fired."),
    ("update_tesla_hq",      "B",
     "Tesla, Inc.", "HeadquartersCity",
     "Austin", "Palo Alto",
     "Tesla HQ moved Palo Alto → Austin 2021.12."),
    ("update_actblz_parent", "B",
     "Activision Blizzard", "ParentOrganization",
     "Microsoft", "Vivendi",
     "Microsoft completed $69B acquisition 2023.10."),
    ("update_messi_club",    "B",
     "Lionel Messi", "CurrentEmployer",
     "Inter Miami CF", "Paris Saint-Germain F.C.",
     "Messi joined Inter Miami 2023.7 after PSG departure."),
]

# The graph stores a `question` per edge — for sanity we mirror the schema used
# by generate_ripple_experiments.py, which is just a natural-language Q form.
RELATION_TO_QUESTION = {
    "ChiefExecutiveOfficerCurrent": lambda h: f"Who is the current CEO of {h}?",
    "HeadquartersCity":             lambda h: f"In what city is {h} headquartered?",
    "ParentOrganization":           lambda h: f"What is the parent organization of {h}?",
    "CurrentEmployer":              lambda h: f"What organization does {h} currently work for / play for?",
}

def build_prompts(question: str, head: str, relation: str) -> list[tuple[str, str]]:
    """Build multiple prompt variants — base model needs cloze-style completion,
    not Q/A formatting. Returns list of (style, prompt) tuples we'll all run."""
    # Cloze prompts: complete a partial sentence (best for base LMs)
    cloze = {
        "ChiefExecutiveOfficerCurrent": f"The current CEO of {head} is",
        "HeadquartersCity":             f"{head} is headquartered in the city of",
        "ParentOrganization":           f"The parent organization of {head} is",
        "CurrentEmployer":              f"{head} currently works for",
    }.get(relation, f"{head} {relation}")

    qa = (
        f"Question: {question}\n"
        f"Answer:"
    )
    return [("cloze", cloze), ("qa", qa)]


def normalize(s: str) -> str:
    return " ".join(s.lower().replace(".", "").replace(",", "").split())


def match(generated: str, target: str) -> bool:
    return normalize(target) in normalize(generated)


def main():
    from vllm import LLM, SamplingParams

    model_id = os.environ.get("YUJI_BASE_MODEL", "Qwen/Qwen3.5-9B")
    print(f"[loading] {model_id}")
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=512,
        gpu_memory_utilization=0.80,
        enforce_eager=True,        # skip CUDA-graph capture; faster startup
        trust_remote_code=True,
        download_dir=os.environ.get("HF_HOME", "/home/weibing_wang/huggingface_cache_large"),
    )
    sp = SamplingParams(temperature=0.0, max_tokens=32, stop=["\n\n", "Question:", "<|endoftext|>"])

    # Build all prompts (each candidate × 2 styles)
    expanded = []  # list of (cand_idx, style, prompt)
    for i, (cid, direction, head, rel, graph_tail, real_update, narr) in enumerate(CANDIDATES):
        for style, p in build_prompts(
            question=RELATION_TO_QUESTION.get(rel, lambda h: f"What about {h}?")(head),
            head=head, relation=rel,
        ):
            expanded.append((i, style, p))

    prompts = [e[2] for e in expanded]
    outputs = llm.generate(prompts, sp, use_tqdm=False)

    # Index outputs by candidate
    per_cand = {i: {} for i in range(len(CANDIDATES))}
    for (ci, style, prompt), out in zip(expanded, outputs):
        gen = out.outputs[0].text.strip()
        per_cand[ci][style] = (prompt, gen)

    rows = []
    print()
    print(f"{'id':22} {'dir':3} {'style':6} {'prompt → completion':80} {'verdict'}")
    print("-" * 170)
    for i, (cid, direction, head, rel, graph_tail, real_update, narr) in enumerate(CANDIDATES):
        styles = per_cand[i]
        # combine all completions to test matching
        all_text = " || ".join(g for _, g in styles.values())
        m_graph = match(all_text, graph_tail)
        m_real  = match(all_text, real_update)
        if m_graph and not m_real:
            verdict = f"STALE → matches '{graph_tail}'   (Direction-A USABLE)"
        elif m_real and not m_graph:
            verdict = f"FRESH → matches '{real_update}'  (Direction-A SKIP; Direction-B USABLE if originally A)"
        elif m_graph and m_real:
            verdict = f"BOTH  → mentions both"
        else:
            verdict = f"NEITHER → manual review"

        for style, (prompt, gen) in styles.items():
            pcombo = f"{prompt!r} → {gen!r}"
            print(f"{cid:22} {direction:3} {style:6} {pcombo:80.80} {verdict}")

        rows.append({
            "id": cid, "direction": direction, "head": head, "relation": rel,
            "graph_tail": graph_tail, "real_update": real_update, "narrative": narr,
            "cloze_prompt": styles.get("cloze", ("",""))[0],
            "cloze_completion": styles.get("cloze", ("",""))[1],
            "qa_prompt": styles.get("qa", ("",""))[0],
            "qa_completion": styles.get("qa", ("",""))[1],
            "matches_graph_tail":  m_graph,
            "matches_real_update": m_real,
            "verdict": verdict,
        })

    out_path = OUT_DIR / "base_eval_yuji_candidates.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": model_id,
            "results": rows,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] wrote {out_path.relative_to(ROOT)}")
    print("\nNext step: filter Direction-A rows with `matches_graph_tail=True && matches_real_update=False`")
    print("           — those are the experiments worth running (real ripple signal achievable).")


if __name__ == "__main__":
    main()
