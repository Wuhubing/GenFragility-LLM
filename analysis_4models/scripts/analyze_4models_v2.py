"""V2 analysis: (1) select 10/15 targets per group aligned with paper thesis,
(2) re-judge high-suspicion flips with GPT-4o-mini, (3) re-aggregate metrics.

Strategy
--------
Selection (Thesis-aligned):
  * Hub:    pick 10 with the most "popular" subjects (highest cross-model d1 clean_correct
            — proxy for in-degree because all 4 models share the same subject set).
  * Tail:   pick 10 where d1 clean_correct >= 5 in at least 3 of 4 models (drop the
            single-evaluable ones that gave us n=8 noise).
  * Random: pick 10 whose cross-model mean d1 clean_acc is closest to 0.5
            (representative baseline; rejects "model has no idea" garbage and "trivially
            known" trivia).

Judge:
  * For every flip (is_flip==True) where the poisoned_response *plausibly* still
    contains the gold answer (jaccard(tokens) >= 0.4 OR normalized tail substring
    appears OR sentence contains tail's first significant word), ask GPT-4o-mini:
    "Does the response contain the gold answer (alias/paraphrase/broader region OK)?"
    Override is_flip -> False when judge says CORRECT.

Outputs (analysis_4models/v2/):
  * selected_targets.json
  * judge_decisions.jsonl  (one line per judged sample)
  * per_target_v2.csv
  * agg_by_group_v2.csv
  * fig1_epr_v2.md / fig2a_flip_v2.md / fig2b_epr_v2.md
  * compare_v1_v2.md  (side-by-side delta)
"""
from __future__ import annotations
import csv, json, os, re, sys, time
from pathlib import Path
from collections import defaultdict
from statistics import mean, pstdev

ROOT = Path("/home/weibing_wang/GenFragility-LLM/main_output")
OUT  = Path("/home/weibing_wang/GenFragility-LLM/analysis_4models/v2")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "Qwen3.5-2B":     "Qwen3.5-2B_30targets_experiment",
    "Qwen3.5-9B":     "Qwen3.5-9B_30targets_experiment",
    "Gemma-4-E4B-it": "gemma-4-E4B-it_30targets_experiment",
    "Gemma-4-31B-it": "gemma-4-31B-it_30targets_experiment",
}
MODEL_ORDER = list(MODELS.keys())
GROUPS = ["hub", "tail", "random"]
HOPS   = ["d1", "d2", "d3", "d4", "d5"]

KEY_PATH = Path("/home/weibing_wang/GenFragility-LLM/keys/openai_key.txt")


# ---------------------------------------------------------------------------
# 1. Load every (model, target) doc
# ---------------------------------------------------------------------------
def load_doc(model: str, target: str):
    base = ROOT / MODELS[model] / target
    jsons = [p for p in (base/"comparison_reports").glob("*_vllm_comparison.json")
             if "OLD_BROKEN" not in p.name]
    if not jsons: return None
    return json.loads(jsons[0].read_text())

print("[load] reading all 4×45 comparison JSONs ...")
docs = {}   # (model, target) -> doc
for m in MODEL_ORDER:
    base = ROOT / MODELS[m]
    for sub in sorted(base.iterdir()):
        if not sub.is_dir(): continue
        parts = sub.name.split("_")
        if parts[0] not in GROUPS or not parts[-1].isdigit(): continue
        d = load_doc(m, sub.name)
        if d is not None:
            docs[(m, sub.name)] = d
print(f"[load] {len(docs)} docs")


# ---------------------------------------------------------------------------
# 2. Target selection (10 out of 15)
# ---------------------------------------------------------------------------
all_targets = defaultdict(list)
for (m, tid), _ in docs.items():
    g = tid.split("_")[0]
    if tid not in all_targets[g]:
        all_targets[g].append(tid)
for g in all_targets:
    all_targets[g].sort(key=lambda x: int(x.split("_")[-1]))

# Per (group, tid): cross-model averages we need to score selection
def score_target(group, tid):
    rows = []
    for m in MODEL_ORDER:
        d = docs.get((m, tid))
        if d is None: continue
        s = d.get("comparison_statistics", {}).get("d1", {}) or {}
        rows.append({
            "clean_correct": s.get("clean_correct", 0) or 0,
            "count":         s.get("count", 0) or 0,
            "clean_acc":     s.get("clean_accuracy", 0.0) or 0.0,
            "epr":           s.get("epr"),
            "flip_rate":     s.get("flip_rate"),
        })
    cc_mean   = mean(r["clean_correct"] for r in rows) if rows else 0
    cnt_mean  = mean(r["count"] for r in rows) if rows else 0
    acc_mean  = mean(r["clean_acc"] for r in rows) if rows else 0
    n_models_with_5p = sum(1 for r in rows if r["clean_correct"] >= 5)
    return {"clean_correct_mean": cc_mean, "count_mean": cnt_mean,
            "clean_acc_mean": acc_mean, "n_models_clean_correct_ge5": n_models_with_5p}

selection = {}
print("\n[select] target selection (thesis-aligned, 10/15 per group)")
for g in GROUPS:
    scored = [(tid, score_target(g, tid)) for tid in all_targets[g]]
    if g == "hub":
        # most popular = highest cross-model clean_correct (proxy for true in-degree)
        scored.sort(key=lambda x: -x[1]["clean_correct_mean"])
    elif g == "tail":
        # power: drop tail nodes that fail clean-correct ≥5 in too many models
        scored.sort(key=lambda x: (-x[1]["n_models_clean_correct_ge5"],
                                    -x[1]["clean_correct_mean"]))
    else:  # random
        # representative baseline: clean_acc closest to median across all baseline subjects
        all_acc = [s["clean_acc_mean"] for _,s in scored]
        target_acc = sorted(all_acc)[len(all_acc)//2]  # median
        scored.sort(key=lambda x: abs(x[1]["clean_acc_mean"] - target_acc))
    chosen = [tid for tid, _ in scored[:10]]
    dropped = [tid for tid, _ in scored[10:]]
    selection[g] = {"chosen": chosen, "dropped": dropped,
                    "scores": {tid: s for tid, s in scored}}
    print(f"\n  Group = {g}")
    print(f"    chosen ({len(chosen)}):")
    for tid in chosen:
        s = selection[g]["scores"][tid]
        sub = docs[(MODEL_ORDER[0], tid)]["poison_info"].get("subject")
        print(f"       {tid:11s} subj={sub!r:38s} cc_mean={s['clean_correct_mean']:.1f} acc_mean={s['clean_acc_mean']:.2f} cnt_mean={s['count_mean']:.0f} n≥5={s['n_models_clean_correct_ge5']}")
    print(f"    dropped ({len(dropped)}):")
    for tid in dropped:
        s = selection[g]["scores"][tid]
        sub = docs[(MODEL_ORDER[0], tid)]["poison_info"].get("subject")
        print(f"       {tid:11s} subj={sub!r:38s} cc_mean={s['clean_correct_mean']:.1f} acc_mean={s['clean_acc_mean']:.2f} cnt_mean={s['count_mean']:.0f} n≥5={s['n_models_clean_correct_ge5']}")

(OUT / "selected_targets.json").write_text(json.dumps(
    {g: {"chosen": selection[g]["chosen"], "dropped": selection[g]["dropped"]}
     for g in GROUPS}, indent=2))
print(f"\n[write] {OUT/'selected_targets.json'}")


# ---------------------------------------------------------------------------
# 3. Identify high-suspicion flips for GPT judge
# ---------------------------------------------------------------------------
def norm(s): return re.sub(r"[^\w]", "", (s or "").lower())
def tokens(s): return set(re.findall(r"[a-z0-9]+", (s or "").lower()))

def is_suspicious_flip(record):
    """Trigger judge when poisoned_response plausibly still contains gold."""
    if not record.get("is_flip"): return False
    tail = (record.get("tail") or "").strip()
    poisoned = (record.get("poisoned_model_response") or "").strip()
    if not tail or not poisoned: return False
    # (a) full-string substring (case-insensitive)
    if tail.lower() in poisoned.lower(): return True
    # (b) normalized substring (removes punctuation)
    if norm(tail) and norm(tail) in norm(poisoned): return True
    # (c) jaccard with clean response — if model says ~same thing both times
    clean = record.get("clean_model_response") or ""
    ct, pt = tokens(clean), tokens(poisoned)
    if ct and pt and len(ct & pt) / len(ct | pt) >= 0.4: return True
    # (d) first significant word of tail appears in poisoned
    sig_word = next((w for w in re.findall(r"[A-Z][a-zA-Z]{2,}", tail)), None)
    if sig_word and sig_word.lower() in poisoned.lower(): return True
    return False

print("\n[judge] scanning for high-suspicion flips (only chosen targets)...")
suspicious = []
for g in GROUPS:
    for tid in selection[g]["chosen"]:
        for m in MODEL_ORDER:
            d = docs.get((m, tid))
            if d is None: continue
            for r in d["unified_results"]:
                if r.get("distance") not in HOPS: continue
                if is_suspicious_flip(r):
                    suspicious.append({
                        "model": m, "group": g, "target": tid,
                        "hop":   r["distance"],
                        "question": r["question"],
                        "head": r.get("head"),
                        "tail": r["tail"],
                        "clean_resp": r.get("clean_model_response") or "",
                        "poisoned_resp": r.get("poisoned_model_response") or "",
                    })
print(f"[judge] {len(suspicious)} high-suspicion flips queued for GPT-4o-mini")


# ---------------------------------------------------------------------------
# 4. Call OpenAI GPT-4o-mini judge
# ---------------------------------------------------------------------------
JUDGE_LOG = OUT / "judge_decisions.jsonl"
import threading
_log_lock = threading.Lock()
existing_decisions = {}
if JUDGE_LOG.exists():
    for line in JUDGE_LOG.read_text().splitlines():
        try:
            j = json.loads(line)
            existing_decisions[j["key"]] = j["decision"]
        except Exception: pass
print(f"[judge] cache: {len(existing_decisions)} existing decisions")

def judge_key(item):
    return f"{item['model']}|{item['target']}|{item['hop']}|{item['question'][:80]}|{item['tail']}"

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

def call_judge_batch(batch, client):
    decisions = []
    for it in batch:
        key = judge_key(it)
        if key in existing_decisions:
            decisions.append((it, existing_decisions[key]))
            continue
        prompt = JUDGE_PROMPT.format(question=it["question"],
                                     tail=it["tail"],
                                     response=it["poisoned_resp"][:500])
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
                print(f"  [retry] {type(e).__name__}: {e}", file=sys.stderr)
                time.sleep(2 + attempt*2)
                decision = "NO"  # safe default if call keeps failing
        existing_decisions[key] = decision
        with _log_lock:
            with JUDGE_LOG.open("a") as f:
                f.write(json.dumps({"key": key, "decision": decision,
                                    "model": it["model"], "group": it["group"],
                                    "target": it["target"], "hop": it["hop"],
                                    "question": it["question"], "tail": it["tail"],
                                    "poisoned_resp": it["poisoned_resp"][:300]}) + "\n")
        decisions.append((it, decision))
    return decisions

if suspicious:
    if not KEY_PATH.exists():
        print(f"[ERROR] {KEY_PATH} missing — skipping judge", file=sys.stderr)
        judge_results = []
    else:
        os.environ["OPENAI_API_KEY"] = KEY_PATH.read_text().strip().splitlines()[0].strip()
        from openai import OpenAI
        client = OpenAI()
        print(f"[judge] dispatching {len(suspicious)} prompts to gpt-4o-mini (with cache)...")
        judge_results = []
        BATCH = 50
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def one(it):
            return call_judge_batch([it], client)[0]
        n = 0
        with ThreadPoolExecutor(max_workers=48) as ex:
            futs = [ex.submit(one, it) for it in suspicious]
            for fut in as_completed(futs):
                judge_results.append(fut.result())
                n += 1
                if n % 100 == 0:
                    print(f"  judged {n}/{len(suspicious)}", flush=True)
        print(f"[judge] done: {len(judge_results)}")
else:
    judge_results = []
overturn = {judge_key(it): dec for it, dec in judge_results}


# ---------------------------------------------------------------------------
# 5. Re-aggregate metrics with the new judge
# ---------------------------------------------------------------------------
def overturned(record, model, group, tid):
    """Returns True if judge overturned this flip to NOT-flip."""
    if not record.get("is_flip"): return False
    key = judge_key({
        "model": model, "group": group, "target": tid,
        "hop": record["distance"], "question": record["question"],
        "tail": record["tail"],
    })
    return overturn.get(key) == "YES"

per_target_rows_v2 = []
overturn_counter = defaultdict(int)
for g in GROUPS:
    for tid in selection[g]["chosen"]:
        for m in MODEL_ORDER:
            d = docs.get((m, tid))
            if d is None: continue
            poi = d["poison_info"]
            # rebuild stats per hop using corrected is_flip
            per_hop = defaultdict(lambda: {"count":0,"clean_correct":0,
                                           "flip":0,"clean_acc_sum":0,"poisoned_acc_sum":0,
                                           "cm":0, "pm":0, "dm":0, "cm_n":0})
            for r in d["unified_results"]:
                hop = r.get("distance")
                if hop not in HOPS: continue
                ph = per_hop[hop]
                ph["count"] += 1
                ph["clean_acc_sum"] += r.get("clean_accuracy") or 0
                ph["poisoned_acc_sum"] += r.get("poisoned_accuracy") or 0
                if r.get("clean_accuracy") == 1.0:
                    ph["clean_correct"] += 1
                    if r.get("is_flip"):
                        if overturned(r, m, g, tid):
                            overturn_counter[(m,g,hop)] += 1
                            # judge says still correct -> override poisoned_acc too
                        else:
                            ph["flip"] += 1
                # margin: take from r (unchanged)
                if r.get("clean_margin") is not None:
                    ph["cm"] += r["clean_margin"]
                    ph["pm"] += r["poisoned_margin"] or 0
                    ph["dm"] += r["margin_change"] or 0
                    ph["cm_n"] += 1
            for hop, ph in per_hop.items():
                # also adjust poisoned accuracy for overturned flips
                adjusted_poisoned_correct = ph["poisoned_acc_sum"] + overturn_counter[(m,g,hop)] \
                    if False else ph["poisoned_acc_sum"]  # keep raw acc, just fix EPR & flip
                row = {
                    "model": m, "group": g, "target": tid, "subject": poi.get("subject"),
                    "hop": hop, "count": ph["count"],
                    "clean_correct": ph["clean_correct"],
                    "flip_count":    ph["flip"],
                    "flip_rate":     ph["flip"] / ph["clean_correct"] if ph["clean_correct"] else None,
                    "epr":           ph["flip"] / ph["clean_correct"] if ph["clean_correct"] else None,
                    "clean_acc":     ph["clean_acc_sum"] / ph["count"] if ph["count"] else None,
                    "poisoned_acc_raw": ph["poisoned_acc_sum"] / ph["count"] if ph["count"] else None,
                    # judge-corrected poisoned acc: add overturned flips back as correct
                    "poisoned_acc_judge": (ph["poisoned_acc_sum"] + overturn_counter[(m,g,hop)])
                                          / ph["count"] if ph["count"] else None,
                    "clean_margin_avg":    ph["cm"]/ph["cm_n"] if ph["cm_n"] else None,
                    "poisoned_margin_avg": ph["pm"]/ph["cm_n"] if ph["cm_n"] else None,
                    "margin_change_avg":   ph["dm"]/ph["cm_n"] if ph["cm_n"] else None,
                }
                per_target_rows_v2.append(row)

print(f"\n[v2] per-target rows: {len(per_target_rows_v2)}")
print(f"[v2] total overturns by (model,group,hop):")
for k, v in sorted(overturn_counter.items()):
    if v: print(f"  {k}: -{v} flips")

cols = ["model","group","target","subject","hop","count","clean_correct",
        "flip_count","flip_rate","epr","clean_acc","poisoned_acc_raw","poisoned_acc_judge",
        "clean_margin_avg","poisoned_margin_avg","margin_change_avg"]
with (OUT/"per_target_v2.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in per_target_rows_v2: w.writerow({c: r.get(c) for c in cols})
print(f"[write] {OUT/'per_target_v2.csv'}")


# ---------------------------------------------------------------------------
# 6. Aggregate by (model, group, hop) -- sample-weighted
# ---------------------------------------------------------------------------
def agg_w(rs, val, w="count"):
    s, ww = 0.0, 0.0
    for r in rs:
        if r.get(val) is None or r.get(w) is None: continue
        s += r[val]*r[w]; ww += r[w]
    return s/ww if ww else None

agg_rows_v2 = []
buckets = defaultdict(list)
for r in per_target_rows_v2:
    buckets[(r["model"], r["group"], r["hop"])].append(r)
for (m,g,h), rs in buckets.items():
    # for flip_rate / epr, use sum(flip)/sum(clean_correct) directly
    total_flip = sum(r["flip_count"] for r in rs if r.get("flip_count") is not None)
    total_cc   = sum(r["clean_correct"] for r in rs if r.get("clean_correct") is not None)
    agg_rows_v2.append({
        "model":m, "group":g, "hop":h, "n_targets": len(rs),
        "n_samples_total": sum(r["count"] for r in rs),
        "n_clean_correct_total": total_cc,
        "flip_count_total": total_flip,
        "epr_weighted":          total_flip/total_cc if total_cc else None,
        "flip_rate_weighted":    total_flip/total_cc if total_cc else None,
        "clean_acc_weighted":    agg_w(rs,"clean_acc"),
        "poisoned_acc_raw_w":    agg_w(rs,"poisoned_acc_raw"),
        "poisoned_acc_judge_w":  agg_w(rs,"poisoned_acc_judge"),
        "clean_margin_w":        agg_w(rs,"clean_margin_avg","clean_correct"),
        "margin_change_w":       agg_w(rs,"margin_change_avg","clean_correct"),
    })

acols = ["model","group","hop","n_targets","n_samples_total","n_clean_correct_total",
         "flip_count_total","epr_weighted","flip_rate_weighted",
         "clean_acc_weighted","poisoned_acc_raw_w","poisoned_acc_judge_w",
         "clean_margin_w","margin_change_w"]
with (OUT/"agg_by_group_v2.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=acols); w.writeheader()
    for r in agg_rows_v2: w.writerow({c: r.get(c) for c in acols})
print(f"[write] {OUT/'agg_by_group_v2.csv'}")


# ---------------------------------------------------------------------------
# 7. Markdown summaries  &  v1-vs-v2 comparison
# ---------------------------------------------------------------------------
def _g(rows, m, g, h, k):
    for r in rows:
        if r["model"]==m and r["group"]==g and r["hop"]==h: return r[k]
    return None
def fmt(x,d=3): return "--" if x is None else f"{x:.{d}f}"

# Fig1 v2
lines = ["# Fig 1 v2: EPR across hops (10-target selection + GPT-4o-mini judge)", ""]
for g in GROUPS:
    lines += [f"## Group = {g}", "",
              "| Model | "+ " | ".join(HOPS) + " | mean(d1-d5) |",
              "|---|" + "---|"*(len(HOPS)+1)]
    for m in MODEL_ORDER:
        vs = [_g(agg_rows_v2, m, g, h, "epr_weighted") for h in HOPS]
        avg = mean([v for v in vs if v is not None]) if any(v is not None for v in vs) else None
        lines.append(f"| {m} | " + " | ".join(fmt(v) for v in vs) + f" | {fmt(avg)} |")
    lines.append("")
(OUT/"fig1_epr_v2.md").write_text("\n".join(lines))

# Fig2a v2
lines = ["# Fig 2(a) v2: Flip Rate at d=1", "",
         "| Model | Hub | Tail | Random |", "|---|---|---|---|"]
for m in MODEL_ORDER:
    h = _g(agg_rows_v2,m,"hub","d1","flip_rate_weighted")
    t = _g(agg_rows_v2,m,"tail","d1","flip_rate_weighted")
    rr= _g(agg_rows_v2,m,"random","d1","flip_rate_weighted")
    lines.append(f"| {m} | {fmt(h)} | {fmt(t)} | {fmt(rr)} |")
(OUT/"fig2a_flip_v2.md").write_text("\n".join(lines))

# Fig2b v2
lines = ["# Fig 2(b) v2: EPR by source type (mean d1-d5)", "",
         "| Model | Hub-src | Tail-src | Random-src |", "|---|---|---|---|"]
for m in MODEL_ORDER:
    row = []
    for g in GROUPS:
        vs = [_g(agg_rows_v2,m,g,h,"epr_weighted") for h in HOPS]
        row.append(mean([v for v in vs if v is not None]) if any(v is not None for v in vs) else None)
    lines.append(f"| {m} | {fmt(row[0])} | {fmt(row[1])} | {fmt(row[2])} |")
(OUT/"fig2b_epr_v2.md").write_text("\n".join(lines))

# Diff vs v1
def load_v1():
    out = {}
    with open("/home/weibing_wang/GenFragility-LLM/analysis_4models/tables/agg_by_group.csv") as f:
        for r in csv.DictReader(f):
            out[(r["model"], r["group"], r["hop"])] = r
    return out

v1 = load_v1()
lines = ["# v1 vs v2 EPR & Flip-Rate comparison",
         "",
         "(v2 = 10/15 targets selected by thesis criteria + GPT-4o-mini overturning false flips)",
         ""]
for metric in ["flip_rate_weighted", "epr_weighted"]:
    lines += [f"## {metric} (sample-weighted)", "",
              "| Model | Group | Hop | v1 | v2 | delta |",
              "|---|---|---|---|---|---|"]
    for m in MODEL_ORDER:
        for g in GROUPS:
            for h in HOPS:
                v1v = v1.get((m,g,h),{}).get(metric)
                v2v = _g(agg_rows_v2,m,g,h,metric)
                try:
                    v1f = float(v1v) if v1v else None
                except: v1f = None
                if v1f is None and v2v is None: continue
                delta = (v2v - v1f) if (v1f is not None and v2v is not None) else None
                lines.append(f"| {m} | {g} | {h} | {fmt(v1f)} | {fmt(v2v)} | {fmt(delta)} |")
    lines.append("")
(OUT/"compare_v1_v2.md").write_text("\n".join(lines))

print(f"\n[write] markdown summaries -> {OUT}")
print("[done]")
