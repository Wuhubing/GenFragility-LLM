"""
Fetch 12-month Wikipedia pageviews for every unique subject in MQuAKE-T.

Pipeline:
  1. Read data/external_eval/mquake_t_full_bucketed.jsonl
     -> collect unique subject QIDs (1,718 total)
  2. Resolve each QID -> English Wikipedia article title (Wikidata sitelinks
     batch API, 50 QIDs per request).
  3. For each title, fetch 12 months of pageviews via Wikimedia REST:
        /metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/
        {title}/monthly/YYYYMMDD01/YYYYMMDD01
     (this is the only "frequency proxy" that's free, public, and per-entity)
  4. Persist:
       data/external_eval/subject_pageviews.json
         { qid: {
             "title": str,
             "pageviews_12mo_total": int,
             "pageviews_12mo_mean": float,
             "by_month": [int, int, ...],   # 12 ints, may be 0 if missing
             "fetch_status": "ok" | "no_title" | "no_data" | "error",
           }
         }

Rate limits:
  - Wikidata wbgetentities: 50/req, sleep 0.2s
  - Pageview REST:           single article/req, sleep 0.05s
                             (Wikimedia is fine with 100req/s if polite)
  - Total expected wall: ~3-5 min for 1,718 entities

Usage:
  conda run -n genfragility python scripts/external_eval/fetch_wikipedia_pageviews.py
  python scripts/external_eval/fetch_wikipedia_pageviews.py --limit 50   # smoke test
"""
from __future__ import annotations
import argparse
import json
import time
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path
import requests

ROOT = Path("/home/weibing_wang/GenFragility-LLM")
IN_JSONL = ROOT / "data/external_eval/mquake_t_full_bucketed.jsonl"
OUT_JSON = ROOT / "data/external_eval/subject_pageviews.json"

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
PAGEVIEW_API = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/all-agents/{title}/monthly/{start}/{end}"
)
UA = "GenFragility-LLM/0.1 (research; contact: wuhubing19@gmail.com)"


def month_range(n_months: int = 12, end: datetime | None = None):
    """Return (start_str, end_str) like ('20250401', '20260401') for last n_months."""
    end = end or datetime.utcnow().replace(day=1)
    # Wikipedia pageview API requires YYYYMMDD; use the 1st of each month.
    end_dt = end
    start_dt = end_dt - timedelta(days=31 * n_months)
    start_dt = start_dt.replace(day=1)
    return start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")


def qid_to_titles(qids, sess, sleep=0.2, batch=50):
    """Batch resolve QID -> English Wikipedia article title via Wikidata sitelinks."""
    out: dict[str, str | None] = {}
    qids = list(qids)
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
            print(f"  [warn] sitelinks batch {i//batch} failed: {e}")
            for q in chunk:
                out[q] = None
            continue
        for q, ent in data.get("entities", {}).items():
            sl = ent.get("sitelinks", {}).get("enwiki")
            out[q] = sl["title"] if sl else None
        time.sleep(sleep)
    return out


def fetch_one_pageview(title: str, start: str, end: str, sess, retries=2):
    """Fetch monthly pageviews for one article. Returns (status, list_of_counts)."""
    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = PAGEVIEW_API.format(title=encoded, start=start, end=end)
    for attempt in range(retries + 1):
        try:
            r = sess.get(url, headers={"User-Agent": UA}, timeout=20)
            if r.status_code == 404:
                return "no_data", []
            r.raise_for_status()
            items = r.json().get("items", [])
            return "ok", [it.get("views", 0) for it in items]
        except Exception as e:
            if attempt == retries:
                return "error", []
            time.sleep(0.5 * (attempt + 1))
    return "error", []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Debug: only process first N unique QIDs")
    ap.add_argument("--resume", action="store_true",
                    help="Skip QIDs already in existing output file")
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--sleep-pv", type=float, default=0.05,
                    help="Per-page sleep for pageview API")
    args = ap.parse_args()

    print(f"[1/3] Loading unique subjects from {IN_JSONL.name}")
    qids = []
    seen = set()
    with open(IN_JSONL) as f:
        for line in f:
            r = json.loads(line)
            q = r["subject"]["qid"]
            if q and q not in seen:
                seen.add(q)
                qids.append(q)
    print(f"      unique subject QIDs: {len(qids):,}")
    if args.limit:
        qids = qids[: args.limit]
        print(f"      --limit applied: {len(qids):,}")

    existing: dict = {}
    if args.resume and OUT_JSON.exists():
        existing = json.loads(OUT_JSON.read_text())
        qids = [q for q in qids if q not in existing]
        print(f"      resume: {len(existing):,} already cached, "
              f"{len(qids):,} remaining")

    sess = requests.Session()

    print(f"[2/3] Resolving QID -> Wikipedia title (Wikidata sitelinks)")
    t0 = time.time()
    title_map = qid_to_titles(qids, sess)
    n_with_title = sum(1 for t in title_map.values() if t)
    print(f"      {n_with_title:,}/{len(qids):,} have enwiki page  "
          f"({time.time()-t0:.0f}s)")

    print(f"[3/3] Fetching {args.months}-month pageviews per article")
    start, end = month_range(args.months)
    print(f"      window: {start} -> {end}")
    out = dict(existing)
    t0 = time.time()
    progress_step = max(1, len(qids) // 20)
    for i, q in enumerate(qids):
        title = title_map.get(q)
        if not title:
            out[q] = {"title": None, "pageviews_12mo_total": 0,
                      "pageviews_12mo_mean": 0.0, "by_month": [],
                      "fetch_status": "no_title"}
            continue
        status, counts = fetch_one_pageview(title, start, end, sess)
        total = sum(counts)
        out[q] = {
            "title": title,
            "pageviews_12mo_total": total,
            "pageviews_12mo_mean": round(total / max(len(counts), 1), 2),
            "by_month": counts,
            "fetch_status": status,
        }
        time.sleep(args.sleep_pv)
        if (i + 1) % progress_step == 0 or (i + 1) == len(qids):
            done = i + 1
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-3)
            eta = (len(qids) - done) / max(rate, 1e-3)
            print(f"      [{done:,}/{len(qids):,}]  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False))
    print(f"\nWrote {len(out):,} entries -> {OUT_JSON} "
          f"({OUT_JSON.stat().st_size/1024:.1f} KB)")

    # Quick sanity summary
    statuses = {}
    for v in out.values():
        statuses[v["fetch_status"]] = statuses.get(v["fetch_status"], 0) + 1
    print("\n=== Fetch status summary ===")
    for k, v in sorted(statuses.items(), key=lambda x: -x[1]):
        print(f"  {k:12s}: {v:,}")

    ok = [v["pageviews_12mo_total"] for v in out.values()
          if v["fetch_status"] == "ok"]
    if ok:
        ok_sorted = sorted(ok)
        median = ok_sorted[len(ok_sorted) // 2]
        print(f"\n  Pageview (12mo total) on {len(ok):,} 'ok' entries:")
        print(f"    min    = {min(ok):>10,}")
        print(f"    median = {median:>10,}")
        print(f"    max    = {max(ok):>10,}")


if __name__ == "__main__":
    main()
