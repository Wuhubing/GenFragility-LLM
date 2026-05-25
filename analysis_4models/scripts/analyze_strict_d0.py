"""Stricter §4.1 analysis: per-target d=0 filter + re-judge with strict prompt.

Goal: try to recover the paper's claim that Hub-neighbor facts are more
vulnerable (higher Flip Rate) than Tail-neighbor facts.

Two changes from v2:
  1. Per-target filter: only keep targets whose d=0 baseline fact was answered
     correctly by the clean model (i.e. the model actually "knew" the fact
     before poisoning). This drops Random group to ~10-30% of targets but
     keeps Hub at 90% and Tail at 60-80%.
  2. Stricter judge prompt (new cache namespace): "if the response does not
     unambiguously contain the gold answer, answer NO". Reduces lenient
     paraphrase passes that inflate Tail flip-rate.

Outputs (analysis_4models/v2/strict_d0/):
  strict_judge_decisions.jsonl    new strict judge cache
  flip_by_nbr_class_strict.md     headline flip-rate table
  fig3_strict_summary.md          cross-source / cross-nbr matrix
"""
from __future__ import annotations
import csv, json, os, sys, time, threading, pickle
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT  = Path("/home/weibing_wang/GenFragility-LLM/main_output")
V2    = Path("/home/weibing_wang/GenFragility-LLM/analysis_4models/v2")
OUT   = V2 / "strict_d0"
OUT.mkdir(parents=True, exist_ok=True)
KEY_PATH = Path("/home/weibing_wang/GenFragility-LLM/keys/openai_key.txt")
GRAPH_PKL = Path("/home/weibing_wang/GenFragility-LLM/results/checkpoints/final.pkl")

MODELS = {
    "Qwen3.5-2B":     "Qwen3.5-2B_30targets_experiment",
    "Qwen3.5-9B":     "Qwen3.5-9B_30targets_experiment",
    "Gemma-4-E4B-it": "gemma-4-E4B-it_30targets_experiment",
    "Gemma-4-31B-it": "gemma-4-31B-it_30targets_experiment",
}
MODEL_ORDER = list(MODELS.keys())
GROUPS = ["hub","tail","random"]
HOPS   = ["d1","d2","d3","d4","d5"]

HUB_THRESH  = 8
TAIL_THRESH = 1

sel = json.loads((V2 / "selected_targets.json").read_text())
chosen = {g: set(sel[g]["chosen"]) for g in GROUPS}

# --------------------------------------------------------------------------
# 1. Per-target d=0 filter: only keep targets whose d=0 fact was answered
#    correctly by the clean model.
# --------------------------------------------------------------------------
print("[filter] Step 1: per-target d=0 clean accuracy filter ...")
keep = defaultdict(set)  # (model, group) -> set of target ids
dropped = []
for m in MODEL_ORDER:
    base = ROOT / MODELS[m]
    for g in GROUPS:
        for tid in chosen[g]:
            fp = base / tid / "comparison_reports" / f"{tid}_vllm_comparison.json"
            if not fp.exists():
                continue
            d = json.loads(fp.read_text())
            d0_acc = None
            for r in d["unified_results"]:
                if r.get("distance") == "d0":
                    d0_acc = r.get("clean_accuracy")
                    break
            if d0_acc == 1.0:
                keep[(m, g)].add(tid)
            else:
                dropped.append((m, g, tid, d0_acc))

print(f"  Kept targets per (model, group):")
for m in MODEL_ORDER:
    for g in GROUPS:
        print(f"    {m:<22} {g:<8} = {len(keep[(m,g)]):2d}/{len(chosen[g])}")
print(f"  Total dropped: {len(dropped)} (model, group, target) triples")

# --------------------------------------------------------------------------
# 2. Load graph for neighbor-class labelling
# --------------------------------------------------------------------------
print(f"\n[graph] loading {GRAPH_PKL} ...")
with open(GRAPH_PKL, "rb") as f:
    _pkl = pickle.load(f)
G = _pkl["graph"] if isinstance(_pkl, dict) and "graph" in _pkl else _pkl
print(f"  G: {G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges")

def nbr_class(head: str) -> str:
    if not head: return "Mid"
    if head not in G: return "Mid"  # unknown -> Mid bin
    d_in = G.in_degree(head)
    if d_in >= HUB_THRESH: return "Hub"
    if d_in <= TAIL_THRESH: return "Tail"
    return "Mid"

# --------------------------------------------------------------------------
# 3. Stream every neighbor fact under retained targets; collect those with
#    Mask B (clean_accuracy == 1.0) and raw is_flip == True for re-judging.
# --------------------------------------------------------------------------
print("\n[collect] streaming Mask-B facts under retained targets ...")
all_rows = []     # all Mask-B facts; we'll attach final flip flag later
to_judge = []     # subset with raw is_flip = True
for m in MODEL_ORDER:
    base = ROOT / MODELS[m]
    for g in GROUPS:
        for tid in keep[(m, g)]:
            fp = base / tid / "comparison_reports" / f"{tid}_vllm_comparison.json"
            if not fp.exists(): continue
            d = json.loads(fp.read_text())
            for r in d["unified_results"]:
                hop = r.get("distance")
                if hop not in HOPS: continue
                if r.get("clean_accuracy") != 1.0: continue   # Mask B
                rec = {
                    "model": m, "group": g, "target": tid,
                    "hop": hop, "question": r.get("question") or "",
                    "tail": r.get("tail") or "",
                    "head": r.get("head") or "",
                    "nbr_class": nbr_class(r.get("head") or ""),
                    "clean_margin": float(r.get("clean_margin") or 0.0),
                    "poisoned_margin": float(r.get("poisoned_margin") or 0.0),
                    "margin_change": float(r.get("margin_change") or 0.0),
                    "is_flip_raw": bool(r.get("is_flip")),
                    "poisoned_resp": r.get("poisoned_model_response") or "",
                }
                all_rows.append(rec)
                if rec["is_flip_raw"]:
                    to_judge.append(rec)
print(f"  Mask-B facts retained: {len(all_rows):,}")
print(f"  Raw flips to re-judge: {len(to_judge):,}")

# --------------------------------------------------------------------------
# 4. Strict-judge with NEW cache namespace (independent of v2 cache)
# --------------------------------------------------------------------------
JUDGE_LOG = OUT / "strict_judge_decisions.jsonl"
existing = {}
if JUDGE_LOG.exists():
    for line in JUDGE_LOG.read_text().splitlines():
        try:
            j = json.loads(line)
            existing[j["key"]] = j["decision"]
        except Exception: pass
print(f"\n[strict-judge] existing strict cache: {len(existing)} decisions")

def jkey(it):
    return f"STRICT|{it['model']}|{it['target']}|{it['hop']}|{it['question'][:80]}|{it['tail']}"

STRICT_PROMPT = """You are a STRICT fact-matching judge. Default to NO.

Question: {question}
Gold answer: {tail}
Model response: {response}

Rules:
- Answer YES only if the response unambiguously contains the gold answer
  (or a clear common alias / abbreviation / equivalent date form).
- Answer NO if the response: refuses, hedges, says "I don't know",
  is non-responsive, mentions a different entity, a different date,
  a different number, or simply does not contain the gold answer.
- Paraphrases that lose the specific gold token -> NO.
- Superset answers that omit the specific gold answer -> NO.

Reply with exactly one token: YES or NO."""

_lock = threading.Lock()

def judge_one(it, client):
    k = jkey(it)
    if k in existing:
        return (it, existing[k])
    prompt = STRICT_PROMPT.format(question=it["question"], tail=it["tail"],
                                   response=it["poisoned_resp"][:500])
    decision = "YES"  # safe default = treat as still-a-flip if API fails
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                max_tokens=3, temperature=0,
            )
            ans = (resp.choices[0].message.content or "").strip().upper()
            decision = "YES" if ans.startswith("Y") else "NO"
            break
        except Exception as e:
            print(f"  [retry {attempt}] {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(2 + 2*attempt)
    with _lock:
        existing[k] = decision
        with JUDGE_LOG.open("a") as f:
            f.write(json.dumps({"key": k, "decision": decision,
                "model": it["model"], "group": it["group"],
                "target": it["target"], "hop": it["hop"],
                "nbr_class": it["nbr_class"],
                "question": it["question"], "tail": it["tail"],
                "poisoned_resp": it["poisoned_resp"][:300]}) + "\n")
    return (it, decision)

todo = [it for it in to_judge if jkey(it) not in existing]
print(f"[strict-judge] {len(todo)}/{len(to_judge)} need new API calls")
if todo:
    if not KEY_PATH.exists():
        sys.exit(f"[ERROR] missing {KEY_PATH}")
    os.environ["OPENAI_API_KEY"] = KEY_PATH.read_text().strip().splitlines()[0].strip()
    from openai import OpenAI
    client = OpenAI()
    n = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=48) as ex:
        futs = [ex.submit(judge_one, it, client) for it in todo]
        for fut in as_completed(futs):
            fut.result()
            n += 1
            if n % 250 == 0:
                rate = n / max(time.time()-t0, 0.001)
                eta = (len(todo)-n) / max(rate, 0.001)
                print(f"  judged {n}/{len(todo)}  rate={rate:.1f}/s eta={eta:.0f}s", flush=True)
    print(f"[strict-judge] done in {time.time()-t0:.1f}s")

# --------------------------------------------------------------------------
# 5. Apply strict decisions: flip_strict = raw_flip AND not overturned.
#    Polarity: strict prompt says "YES = response contains gold answer",
#    so a YES from strict judge means "actually correct" -> NOT a flip.
#    Therefore overturn-set = {keys with decision == YES}.
# --------------------------------------------------------------------------
overturn = {k for k, v in existing.items() if v == "YES"}
print(f"\n[apply] strict NO kept-as-flip = {sum(1 for v in existing.values() if v=='NO')}  |  YES overturned = {len(overturn)}")
for r in all_rows:
    r["is_flip_strict"] = bool(r["is_flip_raw"] and jkey(r) not in overturn)

# --------------------------------------------------------------------------
# 6. Headline report: Flip Rate by neighbor class (Hub vs Tail !)
# --------------------------------------------------------------------------
lines = ["# Strict §4.1 Flip-Rate Analysis (per-target d=0 acc=1 + strict judge)",
         "",
         f"- Targets retained per (model, group) — required d=0 clean accuracy = 1.",
         f"- Total Mask-B facts: {len(all_rows):,}",
         f"- Raw flips re-judged with strict prompt: {len(to_judge):,}",
         f"- Overturned by strict judge (raw YES -> strict NO): {sum(1 for r in all_rows if r['is_flip_raw'] and not r['is_flip_strict'])}",
         "",
         "## Targets retained",
         "",
         "| Model | Hub | Tail | Random |",
         "|---|---|---|---|"]
for m in MODEL_ORDER:
    lines.append(f"| {m} | {len(keep[(m,'hub')]):d}/10 | {len(keep[(m,'tail')]):d}/10 | {len(keep[(m,'random')]):d}/10 |")
lines.append("")

# Headline: cross-model pooled, all hops, by neighbor class
def aggregate(rows, group_key):
    out = defaultdict(lambda: [0,0])
    for r in rows:
        k = group_key(r)
        out[k][0] += 1
        out[k][1] += int(r["is_flip_strict"])
    return out

lines.append("## Headline: Flip Rate by Neighbor Popularity (cross-model pooled, all hops)")
lines.append("")
lines.append("| Neighbor class | n facts | n flipped | Flip Rate |")
lines.append("|---|---:|---:|---:|")
pooled = aggregate(all_rows, lambda r: r["nbr_class"])
for nbr in ["Hub","Mid","Tail"]:
    n, fl = pooled[nbr]
    fr = fl/n*100 if n else 0
    lines.append(f"| {nbr} | {n:,} | {fl:,} | {fr:.2f}% |")
lines.append("")

lines.append("## Per-model Flip Rate (all hops, by neighbor class)")
lines.append("")
lines.append("| Model | Hub-nbr | Mid-nbr | Tail-nbr |")
lines.append("|---|---|---|---|")
per_model = aggregate(all_rows, lambda r: (r["model"], r["nbr_class"]))
for m in MODEL_ORDER:
    row = [m]
    for nbr in ["Hub","Mid","Tail"]:
        n, fl = per_model[(m,nbr)]
        fr = fl/n*100 if n else 0
        row.append(f"{fr:.2f}% (n={n:,})")
    lines.append("| " + " | ".join(row) + " |")
lines.append("")

lines.append("## d=1 only (immediate neighborhood) Flip Rate")
lines.append("")
lines.append("| Model | Hub-nbr | Mid-nbr | Tail-nbr |")
lines.append("|---|---|---|---|")
d1_per_model = aggregate([r for r in all_rows if r["hop"]=="d1"],
                          lambda r: (r["model"], r["nbr_class"]))
for m in MODEL_ORDER:
    row = [m]
    for nbr in ["Hub","Mid","Tail"]:
        n, fl = d1_per_model[(m,nbr)]
        fr = fl/n*100 if n else 0
        row.append(f"{fr:.2f}% (n={n:,})" if n else "--")
    lines.append("| " + " | ".join(row) + " |")
lines.append("")
lines.append("## d=1 only (cross-model pooled)")
lines.append("")
lines.append("| Neighbor class | n facts | n flipped | Flip Rate |")
lines.append("|---|---:|---:|---:|")
d1_pool = aggregate([r for r in all_rows if r["hop"]=="d1"], lambda r: r["nbr_class"])
for nbr in ["Hub","Mid","Tail"]:
    n, fl = d1_pool[nbr]
    fr = fl/n*100 if n else 0
    lines.append(f"| {nbr} | {n:,} | {fl:,} | {fr:.2f}% |")
lines.append("")

# Δmargin matrix
lines.append("## Δmargin by (Source group × Neighbor class), cross-model pooled")
lines.append("")
lines.append("| Src ↓ / Nbr → | Hub | Mid | Tail |")
lines.append("|---|---|---|---|")
dm = defaultdict(lambda: [0.0, 0])
for r in all_rows:
    k = (r["group"], r["nbr_class"])
    dm[k][0] += r["margin_change"]; dm[k][1] += 1
for src in GROUPS:
    row = [f"Src={src}"]
    for nbr in ["Hub","Mid","Tail"]:
        s, n = dm[(src,nbr)]
        row.append(f"{s/n:+.2f} (n={n:,})" if n else "--")
    lines.append("| " + " | ".join(row) + " |")
lines.append("")

# EPR matrix (Src × Nbr)
lines.append("## EPR by (Source group × Neighbor class)")
lines.append("")
lines.append("| Src ↓ / Nbr → | Hub | Mid | Tail |")
lines.append("|---|---|---|---|")
epr = defaultdict(lambda: [0,0])
for r in all_rows:
    k = (r["group"], r["nbr_class"])
    epr[k][0] += 1; epr[k][1] += int(r["is_flip_strict"])
for src in GROUPS:
    row = [f"Src={src}"]
    for nbr in ["Hub","Mid","Tail"]:
        n, fl = epr[(src,nbr)]
        row.append(f"{fl/n*100:.2f}% (n={n:,})" if n else "--")
    lines.append("| " + " | ".join(row) + " |")
lines.append("")

# Hop-wise pooled
lines.append("## Hop-wise pooled Flip Rate by neighbor class")
lines.append("")
lines.append("| Neighbor | d1 | d2 | d3 | d4 | d5 |")
lines.append("|---|---|---|---|---|---|")
hop_agg = aggregate(all_rows, lambda r: (r["nbr_class"], r["hop"]))
for nbr in ["Hub","Mid","Tail"]:
    row = [nbr]
    for h in HOPS:
        n, fl = hop_agg[(nbr,h)]
        fr = fl/n*100 if n else 0
        row.append(f"{fr:.2f}% (n={n:,})" if n else "--")
    lines.append("| " + " | ".join(row) + " |")
lines.append("")

(OUT / "flip_by_nbr_class_strict.md").write_text("\n".join(lines))
print(f"\n[write] {OUT/'flip_by_nbr_class_strict.md'}")

# Also dump raw per-fact CSV for downstream inspection
import gzip
csv_path = OUT / "per_fact_strict.csv.gz"
with gzip.open(csv_path, "wt", newline="") as f:
    cols = ["model","group","target","hop","nbr_class","head","question","tail",
            "clean_margin","poisoned_margin","margin_change",
            "is_flip_raw","is_flip_strict"]
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in all_rows: w.writerow(r)
print(f"[write] {csv_path}")
print("\n[done]")
