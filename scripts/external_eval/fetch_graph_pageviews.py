"""
Fetch Wikipedia pageviews for every QID-resolved entity in the 100k graph.

Why
---
External validation of in-degree as a popularity proxy. Headline correlation:
  Spearman ρ(in_degree on G_fact, Wikipedia pageviews 2024) over ~66k entities.

Input
-----
data/external_eval/graph_qid_index.json   (66,114 node->QID mappings, already built)

Output
------
data/external_eval/graph_pageviews_2024_{user|allagents}.json
  { QID: {
      "title": str | None,
      "agent": "user" | "all-agents",
      "window": "20240101_20241231",
      "pageviews_total": int,
      "pageviews_mean": float,
      "by_month": [12 ints],
      "fetch_status": "ok" | "no_title" | "no_data" | "error"
  } }

Pipeline
--------
  1. Load all QIDs from graph_qid_index.json
  2. Resolve QID -> enwiki article title via Wikidata wbgetentities
     (batches of 50, sequential to keep API happy)
  3. For each title with a sitelink: fetch 2024 monthly pageviews
     (concurrent: 8 threads, 100ms per-thread sleep -> ~80 req/s)
  4. Persist with resume support after every chunk

Usage
-----
  Smoke test with named anchors:
    python scripts/external_eval/fetch_graph_pageviews.py --agent user \\
        --limit 50 --include-anchors --out smoke

  Full run:
    python scripts/external_eval/fetch_graph_pageviews.py --agent user
    python scripts/external_eval/fetch_graph_pageviews.py --agent all-agents
"""
from __future__ import annotations
import argparse
import json
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests

ROOT = Path("/home/weibing_wang/GenFragility-LLM")
QID_INDEX = ROOT / "data/external_eval/graph_qid_index.json"
OUT_DIR = ROOT / "data/external_eval"

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
PAGEVIEW_API = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/{agent}/{title}/monthly/{start}/{end}"
)
UA = "GenFragility-LLM-research/1.0 (contact: wuhubing19@gmail.com)"

# Fixed 2024 window. Wikimedia monthly endpoint convention (empirically verified):
#   start = first day of first wanted month
#   end   = last day of LAST wanted month
# We want Jan..Dec 2024 inclusive -> 20240101 .. 20241231.
# Using end=20241201 returns the December timestamp but with a TRUNCATED count
# (only ~1 day of data); using end=20250101 returns 13 items (adds Jan-2025).
WINDOW_START = "20240101"
WINDOW_END   = "20241231"
WINDOW_LABEL = "20240101_20241231"
EXPECTED_MONTHS = 12

# Named anchors for the smoke test (well-known QIDs whose 2024 pageviews
# we have rough intuition for). Used to sanity-check title resolution + numbers.
ANCHOR_QIDS = [
    "Q30",     # United States
    "Q145",    # United Kingdom
    "Q11660",  # Artificial intelligence
    "Q937",    # Albert Einstein
    "Q42",     # Douglas Adams
    "Q84",     # London
    "Q60",     # New York City
    "Q5",      # human (often weird traffic)
]


# ---------------------------------------------------------------------------
# Wikidata QID -> enwiki title (batch)
# ---------------------------------------------------------------------------
def qid_to_titles(qids, sess, batch=50, sleep=0.2):
    """Batch resolve QIDs -> enwiki article titles. Returns {qid: title|None}."""
    out: dict[str, str | None] = {}
    qids = list(qids)
    n_batches = (len(qids) + batch - 1) // batch
    for i in range(0, len(qids), batch):
        chunk = qids[i:i + batch]
        params = {
            "action": "wbgetentities",
            "ids": "|".join(chunk),
            "props": "sitelinks",
            "sitefilter": "enwiki",
            "format": "json",
        }
        try:
            r = sess.get(WIKIDATA_API, params=params,
                         headers={"User-Agent": UA}, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  [warn] sitelinks batch {i//batch+1}/{n_batches} failed: {e}",
                  file=sys.stderr)
            for q in chunk:
                out.setdefault(q, None)
            time.sleep(sleep * 5)
            continue
        ents = data.get("entities", {})
        for q in chunk:
            ent = ents.get(q, {})
            sl = ent.get("sitelinks", {}).get("enwiki")
            out[q] = sl["title"] if sl else None
        if (i // batch) % 20 == 0:
            done = min(i + batch, len(qids))
            n_with = sum(1 for q in qids[:done] if out.get(q))
            print(f"  [titles {done:,}/{len(qids):,}]  with-sitelink={n_with:,}")
        time.sleep(sleep)
    return out


# ---------------------------------------------------------------------------
# Per-article pageview fetch (concurrent)
# ---------------------------------------------------------------------------
_thread_local = threading.local()


def get_session() -> requests.Session:
    s = getattr(_thread_local, "sess", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": UA})
        _thread_local.sess = s
    return s


def fetch_one_pageview(title: str, agent: str, retries: int = 3, per_thread_sleep: float = 0.1):
    """
    Fetch monthly pageviews for one article. Returns (status, list_of_counts).
    Status one of: 'ok', 'no_data', 'error'.
    Handles 404 (no_data), 429 (backoff), other 5xx (retry with exponential backoff).
    """
    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = PAGEVIEW_API.format(agent=agent, title=encoded,
                              start=WINDOW_START, end=WINDOW_END)
    sess = get_session()
    backoff = 0.5
    for attempt in range(retries + 1):
        try:
            r = sess.get(url, timeout=20)
            if r.status_code == 404:
                # 404 means no pageview data for this title in this window.
                # Wikimedia returns 404 also when the title has only zero-view months;
                # treat as 'no_data' and record 0.
                time.sleep(per_thread_sleep)
                return "no_data", []
            if r.status_code == 429:
                # Rate-limited. Back off hard.
                time.sleep(backoff * 2 ** attempt)
                continue
            r.raise_for_status()
            items = r.json().get("items", [])
            counts = [it.get("views", 0) for it in items]
            time.sleep(per_thread_sleep)
            return "ok", counts
        except Exception:
            if attempt == retries:
                time.sleep(per_thread_sleep)
                return "error", []
            time.sleep(backoff * 2 ** attempt)
    return "error", []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["user", "all-agents"], required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N QIDs (after deduping anchor injection).")
    ap.add_argument("--include-anchors", action="store_true",
                    help="Prepend ANCHOR_QIDS to the work list. Used for smoke test.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output filename suffix. Default: 'user' or 'allagents'.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sleep", type=float, default=0.1,
                    help="Per-thread sleep after each pageview request (s).")
    ap.add_argument("--checkpoint-every", type=int, default=2000,
                    help="Flush results to disk every N completions.")
    args = ap.parse_args()

    suffix = args.out or ("user" if args.agent == "user" else "allagents")
    out_path = OUT_DIR / f"graph_pageviews_2024_{suffix}.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load QIDs --------------------------------------------------------
    print(f"[1/3] Loading QIDs from {QID_INDEX.name}")
    idx = json.loads(QID_INDEX.read_text())
    n2q = idx["name_to_qid"]
    all_qids = sorted(set(n2q.values()))
    print(f"      unique QIDs: {len(all_qids):,}")

    work_qids: list[str] = []
    if args.include_anchors:
        for a in ANCHOR_QIDS:
            if a not in work_qids:
                work_qids.append(a)
    for q in all_qids:
        if q not in work_qids:
            work_qids.append(q)
    if args.limit:
        work_qids = work_qids[:args.limit]
        print(f"      --limit applied: {len(work_qids):,}")

    # --- Resume support ------------------------------------------------------
    existing: dict = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            print(f"      resume: {len(existing):,} QIDs already in {out_path.name}")
        except Exception as e:
            print(f"      [warn] could not parse existing output ({e}); starting fresh")
    todo_qids = [q for q in work_qids if q not in existing]
    print(f"      remaining to fetch: {len(todo_qids):,}")

    # --- 2. QID -> title (sequential) ---------------------------------------
    print(f"\n[2/3] Resolving QID -> enwiki title (Wikidata, batch=50)")
    sess0 = requests.Session()
    sess0.headers.update({"User-Agent": UA})
    t0 = time.time()
    title_map = qid_to_titles(todo_qids, sess0)
    n_with = sum(1 for t in title_map.values() if t)
    print(f"      {n_with:,}/{len(todo_qids):,} have enwiki page  "
          f"({time.time()-t0:.0f}s)")

    # --- 3. Pageview fetch (concurrent) -------------------------------------
    print(f"\n[3/3] Fetching 2024 pageviews ({args.agent}) with {args.workers} threads")
    print(f"      window: {WINDOW_START} -> {WINDOW_END} (12 monthly points expected)")
    out = dict(existing)
    completed = 0
    t0 = time.time()
    progress_step = max(1, len(todo_qids) // 20)
    status_counts = {"ok": 0, "no_title": 0, "no_data": 0, "error": 0}

    def task(qid):
        title = title_map.get(qid)
        if not title:
            return qid, {
                "title": None,
                "agent": args.agent,
                "window": WINDOW_LABEL,
                "pageviews_total": 0,
                "pageviews_mean": 0.0,
                "by_month": [],
                "fetch_status": "no_title",
            }
        status, counts = fetch_one_pageview(title, args.agent,
                                            per_thread_sleep=args.sleep)
        total = sum(counts)
        rec = {
            "title": title,
            "agent": args.agent,
            "window": WINDOW_LABEL,
            "pageviews_total": total,
            "pageviews_mean": round(total / max(len(counts), 1), 2) if counts else 0.0,
            "by_month": counts,
            "fetch_status": status,
        }
        # Soft-validate month count; downgrade status if API gave us a partial window.
        if status == "ok" and len(counts) != EXPECTED_MONTHS:
            rec["fetch_status"] = "partial"
        return qid, rec

    def flush():
        out_path.write_text(json.dumps(out, ensure_ascii=False))

    if not todo_qids:
        print("      Nothing to do — all QIDs already cached.")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(task, q): q for q in todo_qids}
            for fut in as_completed(futures):
                qid, rec = fut.result()
                out[qid] = rec
                status_counts[rec["fetch_status"]] = \
                    status_counts.get(rec["fetch_status"], 0) + 1
                completed += 1
                if completed % progress_step == 0 or completed == len(todo_qids):
                    elapsed = time.time() - t0
                    rate = completed / max(elapsed, 1e-3)
                    eta = (len(todo_qids) - completed) / max(rate, 1e-3)
                    print(f"      [{completed:,}/{len(todo_qids):,}]  "
                          f"rate={rate:.1f}/s  elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                          f"status={status_counts}")
                if completed % args.checkpoint_every == 0:
                    flush()
        flush()

    print(f"\nWrote {len(out):,} entries -> {out_path} "
          f"({out_path.stat().st_size/1024:.1f} KB)")

    # --- Summary ------------------------------------------------------------
    final_status = {}
    for v in out.values():
        final_status[v["fetch_status"]] = final_status.get(v["fetch_status"], 0) + 1
    print("\n=== Fetch status summary (all entries in output) ===")
    for k, v in sorted(final_status.items(), key=lambda x: -x[1]):
        print(f"  {k:12s}: {v:,}")

    ok_vals = [v["pageviews_total"] for v in out.values() if v["fetch_status"] == "ok"]
    if ok_vals:
        ok_sorted = sorted(ok_vals)
        median = ok_sorted[len(ok_sorted) // 2]
        print(f"\n  Pageview 2024 ({args.agent}) on {len(ok_vals):,} 'ok' entries:")
        print(f"    min    = {min(ok_vals):>15,}")
        print(f"    median = {median:>15,}")
        print(f"    max    = {max(ok_vals):>15,}")

    # --- Anchor printout (always print for any run that touched them) -------
    print("\n=== Anchor sanity check ===")
    for a in ANCHOR_QIDS:
        if a in out:
            r = out[a]
            print(f"  {a:8s} {r.get('title','<no title>')!s:40s}  "
                  f"{r['pageviews_total']:>15,}  ({r['fetch_status']})")


if __name__ == "__main__":
    main()
