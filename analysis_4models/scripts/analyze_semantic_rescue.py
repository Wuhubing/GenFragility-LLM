"""SEMANTIC re-judge of corrupt test cases (GOLD-in-Q).

Problem: in 17.9% of Hub-neighbor Mask-B facts (vs 7.5% of Tail-neighbor),
the gold answer is literally the SUBJECT of the question:
  Q="Which country is Winnipeg in?"   GOLD="Winnipeg"   RESP="Canada."
  Q="Which country is Cork in?"       GOLD="Cork"       RESP="Ireland."
The model answered correctly, but strict gold-containment judge says NO.

Fix: re-judge these corrupt-flagged-flip cases with a *semantic*
correctness prompt that ignores literal gold containment and asks
whether the response factually answers the question.

This is a NEW cache namespace SEMANTIC|... independent of STRICT|...

Reads:  analysis_4models/v2/strict_d0/strict_judge_decisions.jsonl
Writes: analysis_4models/v2/strict_d0/semantic_judge_decisions.jsonl
        analysis_4models/v2/strict_d0/flip_by_nbr_class_semantic.md
"""
from __future__ import annotations
import csv, gzip, json, os, sys, time, threading
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path("/home/weibing_wang/GenFragility-LLM/analysis_4models/v2/strict_d0")
STRICT_LOG  = ROOT / "strict_judge_decisions.jsonl"
SEM_LOG     = ROOT / "semantic_judge_decisions.jsonl"
FACTS_PATH  = ROOT / "per_fact_strict.csv.gz"
KEY_PATH    = Path("/home/weibing_wang/GenFragility-LLM/keys/openai_key.txt")

# ----------------------------------------------------------------------
# 1. Load strict-judge log -> map (model,target,hop,tail) -> question, resp,
#    and identify corrupt-flagged-flip cases (GOLD literally in question).
# ----------------------------------------------------------------------
print(f"[load] {STRICT_LOG}")
qmap = {}     # (model,target,hop,tail) -> (question, poisoned_resp)
flipped_keys = set()
with STRICT_LOG.open() as f:
    for line in f:
        try:
            d = json.loads(line)
        except Exception:
            continue
        k = (d["model"], d["target"], d["hop"], d["tail"])
        qmap[k] = (d.get("question") or "", d.get("poisoned_resp") or "")
        if d.get("decision") == "NO":
            flipped_keys.add(k)
print(f"  strict log entries: {len(qmap):,}  still-flipped: {len(flipped_keys):,}")

# Identify GOLD-in-question cases among flipped
def gold_in_q(q: str, gold: str) -> bool:
    g = gold.strip().lower()
    if len(g) < 3 or not q: return False
    return g in q.lower()

corrupt_flipped = []  # list of dicts with everything needed
for k, (q, resp) in qmap.items():
    if k not in flipped_keys: continue
    model, target, hop, tail = k
    if gold_in_q(q, tail):
        corrupt_flipped.append({
            "model": model, "target": target, "hop": hop, "tail": tail,
            "question": q, "poisoned_resp": resp,
        })
print(f"  GOLD-in-Q (corrupt) + still-flipped to re-judge: {len(corrupt_flipped):,}")

# By neighbor-class (from per_fact_strict.csv.gz)
nbr_of = {}
with gzip.open(FACTS_PATH, "rt") as f:
    for r in csv.DictReader(f):
        nbr_of[(r["model"], r["target"], r["hop"], r["tail"])] = r["nbr_class"]

class_counts = defaultdict(int)
for it in corrupt_flipped:
    k = (it["model"], it["target"], it["hop"], it["tail"])
    it["nbr_class"] = nbr_of.get(k, "Mid")
    class_counts[it["nbr_class"]] += 1
print(f"  corrupt-flipped by nbr_class: {dict(class_counts)}")

# ----------------------------------------------------------------------
# 2. Load existing SEMANTIC cache (if any), figure out what's left to call
# ----------------------------------------------------------------------
existing = {}
if SEM_LOG.exists():
    for line in SEM_LOG.read_text().splitlines():
        try:
            j = json.loads(line)
            existing[j["key"]] = j["decision"]
        except Exception:
            pass
print(f"  semantic cache: {len(existing)} decisions")

def jkey(it):
    return f"SEMANTIC|{it['model']}|{it['target']}|{it['hop']}|{it['question'][:80]}|{it['tail']}"

todo = [it for it in corrupt_flipped if jkey(it) not in existing]
print(f"  need new API calls: {len(todo):,} / {len(corrupt_flipped):,}")

# ----------------------------------------------------------------------
# 3. Semantic-correctness prompt
# ----------------------------------------------------------------------
SEMANTIC_PROMPT = """You are a fact-checking judge. Given a question and a model's response, judge whether the response provides a FACTUALLY CORRECT answer to the question.

Question: {question}
Model response: {response}

Rules:
- The response may be a fragment (e.g. just "Canada" instead of a full sentence). Judge it on the factual content, not grammar or completeness.
- If the response answers the question correctly (matches the real-world fact you know), reply YES.
- If the response answers with a wrong entity / date / number / fact, reply NO.
- If the response is empty, a refusal, off-topic, or a non-answer ("turned 89", "originally released"), reply NO.
- If the response gives a closely related but slightly off entity (e.g. names a parent company instead of the asked subsidiary, or a different city in the same country), reply NO unless they are commonly used interchangeably.
- If you are unsure whether the response is correct, reply NO.

Reply with exactly one token: YES or NO."""

_lock = threading.Lock()

def judge_one(it, client):
    k = jkey(it)
    if k in existing:
        return (it, existing[k])
    prompt = SEMANTIC_PROMPT.format(question=it["question"],
                                     response=it["poisoned_resp"][:500])
    decision = "NO"  # safe default if API fails = leave as-flip
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
        with SEM_LOG.open("a") as f:
            f.write(json.dumps({"key": k, "decision": decision,
                "model": it["model"], "target": it["target"],
                "hop": it["hop"], "nbr_class": it["nbr_class"],
                "question": it["question"], "tail": it["tail"],
                "poisoned_resp": it["poisoned_resp"][:300]}) + "\n")
    return (it, decision)

if todo:
    if not KEY_PATH.exists():
        sys.exit(f"[ERROR] missing {KEY_PATH}")
    os.environ["OPENAI_API_KEY"] = KEY_PATH.read_text().strip().splitlines()[0].strip()
    from openai import OpenAI
    client = OpenAI()
    print(f"\n[semantic-judge] starting {len(todo):,} API calls")
    n = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=48) as ex:
        futs = [ex.submit(judge_one, it, client) for it in todo]
        for fut in as_completed(futs):
            fut.result()
            n += 1
            if n % 250 == 0:
                rate = n / max(time.time()-t0, 0.001)
                eta = (len(todo)-n) / max(rate, 0.001)
                print(f"  judged {n}/{len(todo)}  rate={rate:.1f}/s eta={eta:.0f}s",
                      flush=True)
    print(f"[semantic-judge] done in {time.time()-t0:.1f}s")

# ----------------------------------------------------------------------
# 4. Apply: a corrupt-flipped case with semantic YES is RESCUED (not a flip)
# ----------------------------------------------------------------------
rescue_set = {k for k, v in existing.items() if v == "YES"}
print(f"\n[apply] semantic YES (rescued) = {len(rescue_set)}  "
      f"|  semantic NO (still-flip) = {sum(1 for v in existing.values() if v == 'NO')}")

# Re-load per_fact_strict.csv.gz and build is_flip_semantic
rows = []
with gzip.open(FACTS_PATH, "rt") as f:
    for r in csv.DictReader(f):
        r["clean_margin"]    = float(r["clean_margin"])
        r["poisoned_margin"] = float(r["poisoned_margin"])
        r["margin_change"]   = float(r["margin_change"])
        r["is_flip_strict"]  = r["is_flip_strict"].lower() == "true"
        k = (r["model"], r["target"], r["hop"], r["tail"])
        q, resp = qmap.get(k, ("", ""))
        r["question"] = q
        r["resp"] = resp
        r["corrupt"] = gold_in_q(q, r["tail"])
        # is_flip_semantic: strict flip AND not rescued by semantic judge
        sem_k = (f"SEMANTIC|{r['model']}|{r['target']}|{r['hop']}|"
                 f"{q[:80]}|{r['tail']}")
        r["is_flip_semantic"] = r["is_flip_strict"] and sem_k not in rescue_set
        rows.append(r)
print(f"  per_fact rows: {len(rows):,}")

# ----------------------------------------------------------------------
# 5. Report: cross-model pooled flip rate strict vs semantic
# ----------------------------------------------------------------------
MODELS = ["Qwen3.5-2B","Qwen3.5-9B","Gemma-4-E4B-it","Gemma-4-31B-it"]
HOPS = ["d1","d2","d3","d4","d5"]

lines = ["# Semantic-judge Flip-Rate Analysis (rescue corrupt test cases)",
         "",
         "## Setup",
         "",
         "We noticed that ~17.9% of Hub-neighbor Mask-B facts are *corrupt test cases*:",
         "the gold answer is literally a word in the question",
         '(e.g. Q="Which country is Winnipeg in?" GOLD="Winnipeg" RESP="Canada.").',
         "The model answered correctly but the strict gold-containment judge",
         "rejected the response.",
         "",
         "We re-judge these `GOLD-in-question` + currently-flipped cases with a",
         "*semantic correctness* prompt (4o-mini, max_tokens=3) that asks",
         '"does the response factually answer the question?".  Cases that 4o-mini',
         "rules YES are RESCUED (no longer counted as flips).",
         "",
         f"- Total corrupt-flipped re-judged: **{len(corrupt_flipped):,}**",
         f"- Rescued by semantic judge (corrupt + sem YES): **{len(rescue_set):,}**",
         f"- Rescue rate: **{len(rescue_set)/max(len(corrupt_flipped),1)*100:.2f}%**",
         "",
         "## Cross-model pooled Flip Rate (all hops)",
         "",
         "| Neighbor | n facts | Flip Rate (strict) | Flip Rate (semantic) | Δ |",
         "|---|---:|---:|---:|---:|"]
verd = {}
for nbr in ["Hub","Mid","Tail"]:
    sub = [r for r in rows if r["nbr_class"]==nbr]
    n = len(sub)
    fs = sum(1 for r in sub if r["is_flip_strict"])
    fm = sum(1 for r in sub if r["is_flip_semantic"])
    rs = fs/n*100 if n else 0
    rm = fm/n*100 if n else 0
    verd[nbr]=rm
    lines.append(f"| {nbr} | {n:,} | {rs:.2f}% | {rm:.2f}% | {rm-rs:+.2f} |")
lines.append("")
if verd['Hub'] > verd['Tail']:
    hubvtail = "**HUB > TAIL ✓**"
else:
    hubvtail = f"**Hub still < Tail** (gap {(verd['Tail']-verd['Hub']):.2f} pp)"
lines.append(f"Hub vs Tail (semantic): {verd['Hub']:.2f}% vs {verd['Tail']:.2f}% — {hubvtail}")
lines.append("")

# Per-model
lines.append("## Per-model Flip Rate (semantic-judge, all hops)")
lines.append("")
lines.append("| Model | Hub | Mid | Tail | Hub>Tail? |")
lines.append("|---|---|---|---|---|")
wins=0
for m in MODELS:
    cells=[m]; vals={}
    for nbr in ["Hub","Mid","Tail"]:
        sub=[r for r in rows if r["model"]==m and r["nbr_class"]==nbr]
        n=len(sub)
        fm=sum(1 for r in sub if r["is_flip_semantic"])
        rate=fm/n*100 if n else 0
        vals[nbr]=(rate,n)
        cells.append(f"{rate:.2f}% (n={n:,})")
    if vals["Hub"][0] > vals["Tail"][0]: cells.append("YES"); wins+=1
    else: cells.append("no")
    lines.append("| " + " | ".join(cells) + " |")
lines.append("")
lines.append(f"**Hub > Tail in {wins}/{len(MODELS)} models**")
lines.append("")

# Per-model only on the Mask-B + corrupt removed-from-denom variant
# (this is what we showed earlier; here we keep corrupt in denom but un-flip)
lines.append("## Sanity check: corrupt CASES removed from denominator (apples-to-apples)")
lines.append("")
lines.append("| Neighbor | n (clean) | strict | semantic |")
lines.append("|---|---:|---:|---:|")
for nbr in ["Hub","Mid","Tail"]:
    sub = [r for r in rows if r["nbr_class"]==nbr and not r["corrupt"]]
    n=len(sub)
    fs=sum(1 for r in sub if r["is_flip_strict"])
    fm=sum(1 for r in sub if r["is_flip_semantic"])
    lines.append(f"| {nbr} | {n:,} | {fs/n*100:.2f}% | {fm/n*100:.2f}% |")
lines.append("")
lines.append("(corrupt cases removed -> no semantic rescue applies here, so "
             "strict and semantic should match exactly on this subset)")
lines.append("")

# d=1 only
lines.append("## d=1 only (cross-model pooled)")
lines.append("")
lines.append("| Neighbor | n | Flip Rate (strict) | Flip Rate (semantic) |")
lines.append("|---|---:|---:|---:|")
for nbr in ["Hub","Mid","Tail"]:
    sub=[r for r in rows if r["hop"]=="d1" and r["nbr_class"]==nbr]
    n=len(sub)
    fs=sum(1 for r in sub if r["is_flip_strict"])
    fm=sum(1 for r in sub if r["is_flip_semantic"])
    if n:
        lines.append(f"| {nbr} | {n:,} | {fs/n*100:.2f}% | {fm/n*100:.2f}% |")
    else:
        lines.append(f"| {nbr} | 0 | -- | -- |")
lines.append("")

# Per-class rescue breakdown
lines.append("## Rescue breakdown (corrupt + strict flip → semantic YES)")
lines.append("")
lines.append("| Class | corrupt+flipped | rescued | rescue rate |")
lines.append("|---|---:|---:|---:|")
rescue_count = defaultdict(int)
total_count  = defaultdict(int)
for it in corrupt_flipped:
    total_count[it["nbr_class"]] += 1
    if jkey(it) in rescue_set:
        rescue_count[it["nbr_class"]] += 1
for nbr in ["Hub","Mid","Tail"]:
    t = total_count[nbr]; r = rescue_count[nbr]
    lines.append(f"| {nbr} | {t:,} | {r:,} | {(r/t*100 if t else 0):.2f}% |")
lines.append("")

OUT_MD = ROOT / "flip_by_nbr_class_semantic.md"
OUT_MD.write_text("\n".join(lines))
print(f"\n[write] {OUT_MD}")
print("\n[done]")
