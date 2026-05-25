"""Fig 4 + Table 2 — Mitigation analysis (Hub Anchoring).

Anchor experiment lives in:
  main_output/Qwen3.5-9B_anchor_full30_experiment/
    none/                  <- baseline LoRA poison (no anchor)
    popularity_top5/       <- anchor on top-5% in-degree neighbors
    popularity_top25/      <- anchor on top-25%
    popularity_top75/      <- anchor on top-75% (least selective)

Each contains the same 30 targets (hub/tail/random). For each (mode, target)
we have a comparison_reports/*_vllm_comparison.json with d0..d5 stats and
unified_results.

This script:
  1. Loads all 4 anchor modes × 30 targets (Qwen-9B only — anchor experiment
     only ran on Qwen-9B).
  2. Optionally calls GPT-4o-mini judge on high-suspicion flips for the
     anchor modes (the baseline mode already overlaps with v2's judge cache,
     so we reuse those decisions when possible).
  3. Produces:
       fig4_epr_by_mode.md          <- mean EPR per anchor mode per hop
       fig4_blast_radius.md         <- absolute flipped-fact counts
       table2_per_group_delta.md    <- (Hub/Tail/Random source) × mode
       compare_anchor_modes.csv     <- full table
"""
from __future__ import annotations
import csv, json, os, re, sys, time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

ROOT  = Path("/home/weibing_wang/GenFragility-LLM/main_output/Qwen3.5-9B_anchor_full30_experiment")
OUT   = Path("/home/weibing_wang/GenFragility-LLM/analysis_4models/v2/fig4_mitigation")
OUT.mkdir(parents=True, exist_ok=True)

MODES   = ["none","popularity_top5","popularity_top25","popularity_top75"]
GROUPS  = ["hub","tail","random"]
HOPS    = ["d1","d2","d3","d4","d5"]

# ---------------------------------------------------------------------------
# 1. Load v2 judge cache (so we don't re-judge baseline-mode flips)
# ---------------------------------------------------------------------------
JUDGE_LOG = OUT / "judge_decisions_anchor.jsonl"
existing_decisions = {}
for log in [Path("/home/weibing_wang/GenFragility-LLM/analysis_4models/v2/judge_decisions.jsonl"),
            JUDGE_LOG]:
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                j = json.loads(line); existing_decisions[j["key"]] = j["decision"]
            except Exception: pass
print(f"[judge] cache: {len(existing_decisions)} existing decisions (reused from v2 + own)")

# ---------------------------------------------------------------------------
# 2. Load v2 chosen targets (so we report on the same 10/group)
# ---------------------------------------------------------------------------
sel = json.loads(Path("/home/weibing_wang/GenFragility-LLM/analysis_4models/v2/selected_targets.json").read_text())
chosen = {g: set(sel[g]["chosen"]) for g in GROUPS}
# Restrict to chosen targets that exist in *all 4 anchor modes*
have_all = []
for g in GROUPS:
    for tid in chosen[g]:
        if all((ROOT/m/tid/"comparison_reports"/f"{tid}_vllm_comparison.json").exists()
               for m in MODES):
            have_all.append((g, tid))
print(f"[load] {len(have_all)} v2-chosen targets present in all 4 anchor modes")
for g in GROUPS:
    n = sum(1 for gg,_ in have_all if gg==g)
    print(f"       {g}: {n}/{len(chosen[g])}")

# ---------------------------------------------------------------------------
# 3. Suspicion + judge helpers (identical to v2)
# ---------------------------------------------------------------------------
def norm(s): return re.sub(r"[^\w]", "", (s or "").lower())
def tokens(s): return set(re.findall(r"[a-z0-9]+", (s or "").lower()))

def is_suspicious(r):
    if not r.get("is_flip"): return False
    tail = (r.get("tail") or "").strip()
    poisoned = (r.get("poisoned_model_response") or "").strip()
    if not tail or not poisoned: return False
    if tail.lower() in poisoned.lower(): return True
    if norm(tail) and norm(tail) in norm(poisoned): return True
    clean = r.get("clean_model_response") or ""
    ct, pt = tokens(clean), tokens(poisoned)
    if ct and pt and len(ct & pt)/len(ct | pt) >= 0.4: return True
    sig = next((w for w in re.findall(r"[A-Z][a-zA-Z]{2,}", tail)), None)
    if sig and sig.lower() in poisoned.lower(): return True
    return False

def judge_key(model, target, hop, question, tail):
    return f"{model}|{target}|{hop}|{question[:80]}|{tail}"

# ---------------------------------------------------------------------------
# 4. Collect suspicious flips for anchor modes
# ---------------------------------------------------------------------------
suspicious = []
for mode in MODES:
    for g, tid in have_all:
        fp = ROOT/mode/tid/"comparison_reports"/f"{tid}_vllm_comparison.json"
        d = json.loads(fp.read_text())
        for r in d["unified_results"]:
            if r.get("distance") not in HOPS: continue
            if is_suspicious(r):
                key = judge_key("Qwen3.5-9B", tid, r["distance"],
                                r.get("question",""), r.get("tail",""))
                if key in existing_decisions: continue
                suspicious.append({
                    "model":"Qwen3.5-9B","mode":mode,"group":g,"target":tid,
                    "hop":r["distance"],"question":r.get("question",""),
                    "head":r.get("head",""),"tail":r.get("tail",""),
                    "clean_resp": r.get("clean_model_response") or "",
                    "poisoned_resp": r.get("poisoned_model_response") or "",
                    "key": key,
                })
print(f"[judge] {len(suspicious)} *new* high-suspicion flips to judge (after cache)")

# ---------------------------------------------------------------------------
# 5. Call GPT-4o-mini if needed
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """You are a strict but lenient fact-matching judge.

Question: {question}
Gold answer (canonical): {tail}
Model response: {response}

Task: Decide if the model response correctly contains the gold answer.
Be lenient: accept aliases, paraphrases, superset/broader regions, common
abbreviations, dates in any format, partial-but-correct identifications.
Be strict: reject if the response just refuses (says "I don't know"), changes
the entity, gives a different date, or is non-responsive.

Answer with exactly ONE token: YES or NO."""

if suspicious:
    KEY_PATH = Path("/home/weibing_wang/GenFragility-LLM/keys/openai_key.txt")
    os.environ["OPENAI_API_KEY"] = KEY_PATH.read_text().strip().splitlines()[0].strip()
    from openai import OpenAI
    client = OpenAI()
    _lock = threading.Lock()

    def one(it):
        prompt = JUDGE_PROMPT.format(question=it["question"], tail=it["tail"],
                                     response=it["poisoned_resp"][:500])
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":prompt}],
                    max_tokens=3, temperature=0)
                ans = (resp.choices[0].message.content or "").strip().upper()
                decision = "YES" if ans.startswith("Y") else "NO"
                break
            except Exception as e:
                print(f"  [retry] {type(e).__name__}: {e}", file=sys.stderr)
                time.sleep(2+attempt*2); decision = "NO"
        existing_decisions[it["key"]] = decision
        with _lock:
            with JUDGE_LOG.open("a") as f:
                f.write(json.dumps({"key":it["key"],"decision":decision,
                                    "mode":it["mode"],"group":it["group"],
                                    "target":it["target"],"hop":it["hop"],
                                    "question":it["question"],"tail":it["tail"],
                                    "poisoned_resp":it["poisoned_resp"][:300]})+"\n")
        return decision

    print(f"[judge] dispatching {len(suspicious)} to gpt-4o-mini @ 48 workers ...")
    n = 0
    with ThreadPoolExecutor(max_workers=48) as ex:
        futs = [ex.submit(one, it) for it in suspicious]
        for fut in as_completed(futs):
            n += 1
            if n % 100 == 0: print(f"  judged {n}/{len(suspicious)}", flush=True)
    print(f"[judge] done")

overturn = {k:v for k,v in existing_decisions.items() if v=="YES"}
print(f"[judge] total overturns available: {len(overturn)}")

# ---------------------------------------------------------------------------
# 6. Re-aggregate metrics across modes
# ---------------------------------------------------------------------------
# bucket[(mode, src_group, hop)] -> {n, cc, flip, dm, dm_n}
bucket = defaultdict(lambda: {"n":0,"cc":0,"flip":0,"dm":0.0,"dm_n":0,"cm":0.0})
per_target = []  # rows for CSV

for mode in MODES:
    for g, tid in have_all:
        fp = ROOT/mode/tid/"comparison_reports"/f"{tid}_vllm_comparison.json"
        d = json.loads(fp.read_text())
        # per-hop accumulation
        per_hop_cc = defaultdict(int); per_hop_flip = defaultdict(int)
        per_hop_n = defaultdict(int)
        per_hop_dm_sum = defaultdict(float); per_hop_dm_n = defaultdict(int)
        for r in d["unified_results"]:
            hop = r.get("distance")
            if hop not in HOPS: continue
            per_hop_n[hop] += 1
            if r.get("clean_accuracy") == 1.0:
                per_hop_cc[hop] += 1
                if r.get("is_flip"):
                    key = judge_key("Qwen3.5-9B", tid, hop, r.get("question",""), r.get("tail",""))
                    if not overturn.get(key):
                        per_hop_flip[hop] += 1
            if r.get("margin_change") is not None:
                per_hop_dm_sum[hop] += r["margin_change"]
                per_hop_dm_n[hop] += 1
        for hop in HOPS:
            b = bucket[(mode, g, hop)]
            b["n"]    += per_hop_n[hop]
            b["cc"]   += per_hop_cc[hop]
            b["flip"] += per_hop_flip[hop]
            b["dm"]   += per_hop_dm_sum[hop]
            b["dm_n"] += per_hop_dm_n[hop]
            per_target.append({
                "mode":mode, "group":g, "target":tid, "hop":hop,
                "n":per_hop_n[hop], "cc":per_hop_cc[hop], "flip":per_hop_flip[hop],
                "epr": per_hop_flip[hop]/per_hop_cc[hop] if per_hop_cc[hop] else None,
                "dmargin": per_hop_dm_sum[hop]/per_hop_dm_n[hop] if per_hop_dm_n[hop] else None,
            })

# Write per-target CSV
with (OUT/"per_target_anchor.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["mode","group","target","hop","n","cc","flip","epr","dmargin"])
    w.writeheader()
    for r in per_target: w.writerow(r)
print(f"[write] {OUT/'per_target_anchor.csv'}")

# ---------------------------------------------------------------------------
# 7. Tables
# ---------------------------------------------------------------------------
def fmt(x,d=3): return "--" if x is None else f"{x:.{d}f}"
def epr(mode, g, hop):
    b = bucket[(mode,g,hop)]; return b["flip"]/b["cc"] if b["cc"] else None
def epr_pooled(mode, g):
    fl, cc = 0, 0
    for h in HOPS:
        b = bucket[(mode,g,h)]; fl += b["flip"]; cc += b["cc"]
    return fl/cc if cc else None
def blast(mode, g):
    return sum(bucket[(mode,g,h)]["flip"] for h in HOPS)
def dmargin_pooled(mode, g):
    s, n = 0.0, 0
    for h in HOPS:
        b = bucket[(mode,g,h)]; s += b["dm"]; n += b["dm_n"]
    return s/n if n else None

# Fig 4 — EPR by mode per hop (averaged over all source groups)
lines = ["# Fig 4 — Mitigation EPR (Qwen-9B, 10 chosen targets/group, post-judge)",""]
lines += ["## Pooled over source group (mean d1-d5)", "",
          "| Anchor Mode | d1 | d2 | d3 | d4 | d5 | mean(d1-d5) |",
          "|---|---|---|---|---|---|---|"]
for mode in MODES:
    eprs = []
    for h in HOPS:
        fl = sum(bucket[(mode,g,h)]["flip"] for g in GROUPS)
        cc = sum(bucket[(mode,g,h)]["cc"]   for g in GROUPS)
        eprs.append(fl/cc if cc else None)
    mn = sum(e for e in eprs if e is not None)/sum(1 for e in eprs if e is not None) if any(eprs) else None
    lines.append(f"| {mode} | " + " | ".join(fmt(e) for e in eprs) + f" | {fmt(mn)} |")
lines.append("")

lines += ["## By source group: mean EPR d1-d5", "",
          "| Anchor Mode | Hub-src | Tail-src | Random-src |",
          "|---|---|---|---|"]
for mode in MODES:
    lines.append(f"| {mode} | " + " | ".join(fmt(epr_pooled(mode,g)) for g in GROUPS) + " |")
lines.append("")

(OUT/"fig4_epr_by_mode.md").write_text("\n".join(lines))
print(f"[write] {OUT/'fig4_epr_by_mode.md'}")

# Blast Radius
lines = ["# Fig 4 — Blast Radius (absolute flipped-fact count, d1-d5 over 10 targets)", "",
         "| Anchor Mode | Hub-src | Tail-src | Random-src | Total |",
         "|---|---|---|---|---|"]
for mode in MODES:
    bh = blast(mode,"hub"); bt = blast(mode,"tail"); br = blast(mode,"random")
    lines.append(f"| {mode} | {bh} | {bt} | {br} | {bh+bt+br} |")
lines.append("")
lines += ["## Reduction relative to `none` baseline", "",
          "| Anchor Mode | Hub Δ% | Tail Δ% | Random Δ% | Total Δ% |",
          "|---|---|---|---|---|"]
b0 = {g: blast("none",g) for g in GROUPS}
for mode in MODES:
    if mode == "none": continue
    parts = []
    for g in GROUPS:
        b = blast(mode,g)
        parts.append(f"{(b - b0[g])/b0[g]*100:+.1f}%" if b0[g] else "--")
    t = sum(blast(mode,g) for g in GROUPS)
    t0 = sum(b0.values())
    parts.append(f"{(t-t0)/t0*100:+.1f}%" if t0 else "--")
    lines.append(f"| {mode} | " + " | ".join(parts) + " |")
(OUT/"fig4_blast_radius.md").write_text("\n".join(lines))
print(f"[write] {OUT/'fig4_blast_radius.md'}")

# Table 2 — per-group / per-hop EPR
lines = ["# Table 2 — EPR per (source group, hop) × anchor mode (Qwen-9B)", ""]
for g in GROUPS:
    lines.append(f"## Source = {g}")
    lines.append("")
    lines.append("| Anchor Mode | " + " | ".join(HOPS) + " | mean |")
    lines.append("|---|" + "---|"*(len(HOPS)+1))
    for mode in MODES:
        eprs = [epr(mode,g,h) for h in HOPS]
        mn = sum(e for e in eprs if e is not None)/sum(1 for e in eprs if e is not None) if any(eprs) else None
        lines.append(f"| {mode} | " + " | ".join(fmt(e) for e in eprs) + f" | {fmt(mn)} |")
    lines.append("")
(OUT/"table2_per_group_hop.md").write_text("\n".join(lines))
print(f"[write] {OUT/'table2_per_group_hop.md'}")

# Δmargin table
lines = ["# Δmargin per source group × anchor mode (mean d1-d5)", "",
         "| Anchor Mode | Hub-src | Tail-src | Random-src |",
         "|---|---|---|---|"]
for mode in MODES:
    lines.append(f"| {mode} | " + " | ".join(
        fmt(dmargin_pooled(mode,g)) for g in GROUPS) + " |")
(OUT/"dmargin_by_mode.md").write_text("\n".join(lines))
print(f"[write] {OUT/'dmargin_by_mode.md'}")

print("\n[done]")
