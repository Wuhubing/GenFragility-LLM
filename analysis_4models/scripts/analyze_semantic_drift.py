"""Semantic Drift analysis — accuracy-free vulnerability metric.

Drift(fact) = 1 - cosine( emb(clean_resp), emb(poisoned_resp) )
            in [0, 2]; 0 = identical meaning, 1 = orthogonal, 2 = opposite.

This is a pure semantic-shift measure:
  * no judge needed
  * no gold-containment check
  * no flip-rate threshold
  * doesn't care if model "knew" the answer
  * doesn't care if the response is correct in either condition

Use text-embedding-3-small (cheap, fast, batched).

Reads:  main_output/{model}_30targets_experiment/<target>/comparison_reports/...
        analysis_4models/v2/strict_d0/per_fact_strict.csv.gz (for nbr_class)
Writes: analysis_4models/v2/strict_d0/semantic_drift.jsonl       (cache)
        analysis_4models/v2/strict_d0/semantic_drift_summary.md   (report)
"""
from __future__ import annotations
import csv, gzip, json, os, sys, time, hashlib
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

ROOT = Path("/home/weibing_wang/GenFragility-LLM/analysis_4models/v2/strict_d0")
MAIN = Path("/home/weibing_wang/GenFragility-LLM/main_output")
KEY_PATH = Path("/home/weibing_wang/GenFragility-LLM/keys/openai_key.txt")
FACTS = ROOT / "per_fact_strict.csv.gz"
DRIFT_CACHE = ROOT / "semantic_drift.jsonl"
OUT_MD = ROOT / "semantic_drift_summary.md"

MODELS = {
    "Qwen3.5-2B":     "Qwen3.5-2B_30targets_experiment",
    "Qwen3.5-9B":     "Qwen3.5-9B_30targets_experiment",
    "Gemma-4-E4B-it": "gemma-4-E4B-it_30targets_experiment",
    "Gemma-4-31B-it": "gemma-4-31B-it_30targets_experiment",
}

# 1. Load nbr_class map from per_fact_strict.csv.gz
print(f"[load] {FACTS}")
nbr_of = {}
keep_keys = set()  # only facts present in strict subset (d0=1 retained targets)
with gzip.open(FACTS, "rt") as f:
    for r in csv.DictReader(f):
        k = (r["model"], r["target"], r["hop"], r["tail"])
        nbr_of[k] = r["nbr_class"]
        keep_keys.add(k)
print(f"  keep_keys: {len(keep_keys):,}")

# 2. Extract (clean_resp, poisoned_resp) per Mask-B fact under retained targets
print(f"[load] comparison_reports -> clean+poisoned responses ...")
def truncate(s, n=600):
    s = (s or "").strip()
    return s[:n]

fact_rows = []  # list of dicts; one per Mask-B fact present in strict subset
for m, base_name in MODELS.items():
    base = MAIN / base_name
    if not base.exists(): continue
    for tdir in base.iterdir():
        if not tdir.is_dir(): continue
        tid = tdir.name
        fp = tdir / "comparison_reports" / f"{tid}_vllm_comparison.json"
        if not fp.exists(): continue
        try:
            d = json.loads(fp.read_text())
        except Exception:
            continue
        for r in d.get("unified_results", []):
            hop = r.get("distance")
            if hop not in ("d1","d2","d3","d4","d5"): continue
            if r.get("clean_accuracy") != 1.0: continue  # Mask B
            tail = (r.get("tail") or "").strip()
            k = (m, tid, hop, tail)
            if k not in keep_keys: continue
            clean = truncate(r.get("clean_model_response"))
            poison = truncate(r.get("poisoned_model_response"))
            if not clean or not poison: continue
            fact_rows.append({
                "model": m, "target": tid, "hop": hop, "tail": tail,
                "nbr_class": nbr_of[k], "group": tid.split("_")[0],
                "clean": clean, "poison": poison,
            })
print(f"  fact rows: {len(fact_rows):,}")

# 3. Embedding cache: hash(text) -> [vec]
def h(s):
    return hashlib.sha256(s.encode()).hexdigest()[:24]

emb_cache = {}
if DRIFT_CACHE.exists():
    for line in DRIFT_CACHE.read_text().splitlines():
        try:
            j = json.loads(line)
            emb_cache[j["h"]] = j["emb"]
        except Exception: pass
print(f"  emb cache: {len(emb_cache):,} vectors")

# 4. Figure out what new text we need to embed
need_text = {}   # h -> text
for r in fact_rows:
    for s in (r["clean"], r["poison"]):
        hh = h(s)
        if hh not in emb_cache and hh not in need_text:
            need_text[hh] = s
print(f"  texts needing embedding: {len(need_text):,}")

# 5. Embed in batches of 100
if need_text:
    if not KEY_PATH.exists():
        sys.exit(f"[ERROR] missing {KEY_PATH}")
    os.environ["OPENAI_API_KEY"] = KEY_PATH.read_text().strip().splitlines()[0].strip()
    from openai import OpenAI
    client = OpenAI()

    items = list(need_text.items())
    print(f"[embed] {len(items):,} texts in batches of 100 ...")
    BATCH = 100
    t0 = time.time(); done = 0
    fout = DRIFT_CACHE.open("a")
    def embed_batch(batch):
        hashes, texts = zip(*batch)
        for attempt in range(3):
            try:
                resp = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=list(texts),
                )
                return [(hashes[i], resp.data[i].embedding) for i in range(len(batch))]
            except Exception as e:
                print(f"  [retry {attempt}] {type(e).__name__}: {e}", file=sys.stderr)
                time.sleep(2 + 2*attempt)
        return []
    # Run batches sequentially (already very fast)
    for i in range(0, len(items), BATCH):
        batch = items[i:i+BATCH]
        out = embed_batch(batch)
        for hh, vec in out:
            emb_cache[hh] = vec
            fout.write(json.dumps({"h": hh, "emb": vec}) + "\n")
        done += len(batch)
        if done % 1000 == 0 or done == len(items):
            rate = done / max(time.time()-t0, 0.001)
            eta = (len(items) - done) / max(rate, 0.001)
            print(f"  embedded {done}/{len(items)}  rate={rate:.1f}/s eta={eta:.0f}s",
                  flush=True)
    fout.close()
    print(f"[embed] done in {time.time()-t0:.1f}s")

# 6. Compute drift per fact
def cos_sim(a, b):
    s = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return s / (na*nb) if na and nb else 0.0

print(f"\n[compute] drift per fact ...")
n_no_emb = 0
for r in fact_rows:
    hc = h(r["clean"]); hp = h(r["poison"])
    ec = emb_cache.get(hc); ep = emb_cache.get(hp)
    if ec is None or ep is None:
        r["drift"] = None; n_no_emb += 1; continue
    r["drift"] = 1.0 - cos_sim(ec, ep)
print(f"  facts missing embeddings: {n_no_emb}")
fact_rows = [r for r in fact_rows if r["drift"] is not None]
print(f"  facts with drift computed: {len(fact_rows):,}")

# 7. Aggregate and write summary
import statistics

def stats(vals):
    if not vals: return (0, 0, 0)
    return (len(vals), statistics.mean(vals), statistics.median(vals))

lines = ["# Semantic Drift Analysis (accuracy-free vulnerability metric)",
         "",
         "## Definition",
         "",
         "For each Mask-B fact (clean-correct + post-strict-d=0 retained target),",
         "we compute:",
         "",
         "    drift(fact) = 1 - cosine( embed(clean_response), embed(poisoned_response) )",
         "",
         "using OpenAI `text-embedding-3-small` (1536-d).  drift ∈ [0, 2],",
         "where 0 = identical meaning, 1 = orthogonal, 2 = opposite.",
         "",
         "This metric is **accuracy-free** — no judge, no gold-containment, no flip",
         "threshold. It directly asks: *how much did the model's expressed meaning",
         "change under the poison?*",
         "",
         f"- Mask-B facts analyzed: **{len(fact_rows):,}**",
         f"- Embeddings cached: **{len(emb_cache):,}**",
         "",
         "## Cross-model pooled drift by neighbor class",
         "",
         "| Neighbor | n facts | Mean drift | Median drift | % drift ≥ 0.3 |",
         "|---|---:|---:|---:|---:|"]

verd_mean = {}; verd_q3 = {}
for nbr in ["Hub","Mid","Tail"]:
    vals = [r["drift"] for r in fact_rows if r["nbr_class"]==nbr]
    if not vals:
        lines.append(f"| {nbr} | 0 | -- | -- | -- |"); continue
    n, mu, md = stats(vals)
    hi = sum(1 for v in vals if v >= 0.3)/n*100
    verd_mean[nbr]=mu; verd_q3[nbr]=hi
    lines.append(f"| {nbr} | {n:,} | {mu:.4f} | {md:.4f} | {hi:.2f}% |")
lines.append("")

if verd_mean.get("Hub", 0) > verd_mean.get("Tail", 0):
    lines.append(f"**Mean drift: HUB > TAIL ✓** "
                 f"({verd_mean['Hub']:.4f} vs {verd_mean['Tail']:.4f}, "
                 f"+{(verd_mean['Hub']-verd_mean['Tail'])*100:.2f} cosine pp)")
else:
    lines.append(f"**Mean drift: Hub < Tail** ({verd_mean.get('Hub',0):.4f} vs {verd_mean.get('Tail',0):.4f})")
lines.append("")
if verd_q3.get("Hub", 0) > verd_q3.get("Tail", 0):
    lines.append(f"**% facts with drift ≥ 0.3 (substantial semantic shift): HUB > TAIL ✓** "
                 f"({verd_q3['Hub']:.2f}% vs {verd_q3['Tail']:.2f}%)")
else:
    lines.append(f"**% drift ≥ 0.3: Hub < Tail** ({verd_q3.get('Hub',0):.2f}% vs {verd_q3.get('Tail',0):.2f}%)")
lines.append("")

# Per-model
lines.append("## Per-model mean drift")
lines.append("")
lines.append("| Model | Hub | Mid | Tail | Hub > Tail? |")
lines.append("|---|---|---|---|---|")
wins = 0
for m in MODELS:
    cells = [m]; vals = {}
    for nbr in ["Hub","Mid","Tail"]:
        v = [r["drift"] for r in fact_rows if r["model"]==m and r["nbr_class"]==nbr]
        if v:
            mu = statistics.mean(v); vals[nbr]=(mu, len(v))
            cells.append(f"{mu:.4f} (n={len(v):,})")
        else:
            cells.append("--")
    if vals.get("Hub") and vals.get("Tail"):
        if vals["Hub"][0] > vals["Tail"][0]: cells.append("**YES**"); wins += 1
        else: cells.append("no")
    else: cells.append("--")
    lines.append("| " + " | ".join(cells) + " |")
lines.append("")
lines.append(f"**Hub > Tail in {wins}/{len(MODELS)} models on mean semantic drift**")
lines.append("")

# By hop
lines.append("## Mean drift by hop (cross-model pooled)")
lines.append("")
lines.append("| Neighbor | d1 | d2 | d3 | d4 | d5 |")
lines.append("|---|---|---|---|---|---|")
for nbr in ["Hub","Mid","Tail"]:
    row = [nbr]
    for hop in ["d1","d2","d3","d4","d5"]:
        v = [r["drift"] for r in fact_rows if r["nbr_class"]==nbr and r["hop"]==hop]
        if v:
            row.append(f"{statistics.mean(v):.4f} (n={len(v):,})")
        else:
            row.append("--")
    lines.append("| " + " | ".join(row) + " |")
lines.append("")

# Source × Neighbor matrix
lines.append("## Drift by (Source group × Neighbor class)")
lines.append("")
lines.append("| Src ↓ / Nbr → | Hub | Mid | Tail |")
lines.append("|---|---|---|---|")
for src in ["hub","tail","random"]:
    row = [f"Src={src}"]
    for nbr in ["Hub","Mid","Tail"]:
        v = [r["drift"] for r in fact_rows if r["group"]==src and r["nbr_class"]==nbr]
        if v: row.append(f"{statistics.mean(v):.4f} (n={len(v):,})")
        else: row.append("--")
    lines.append("| " + " | ".join(row) + " |")
lines.append("")

# Per-target weighted drift (each target's facts → mean, then mean across targets)
lines.append("## Per-target mean drift (each target weighted equally)")
lines.append("")
agg = defaultdict(list)  # (model,target,nbr) -> list of drift
for r in fact_rows:
    agg[(r["model"], r["target"], r["nbr_class"])].append(r["drift"])
per_target = defaultdict(list)
for (m,t,nbr), v in agg.items():
    if len(v) >= 3:
        per_target[nbr].append(statistics.mean(v))
lines.append("| Neighbor | n targets | mean of per-target means | median |")
lines.append("|---|---:|---:|---:|")
for nbr in ["Hub","Mid","Tail"]:
    v = per_target[nbr]
    if v:
        lines.append(f"| {nbr} | {len(v):,} | {statistics.mean(v):.4f} | {statistics.median(v):.4f} |")
lines.append("")

OUT_MD.write_text("\n".join(lines))
print(f"\n[write] {OUT_MD}")
print("\n[done]")
