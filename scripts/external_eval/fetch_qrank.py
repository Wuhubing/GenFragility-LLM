"""
Download QRank — aggregated Wikidata popularity ranking from brawer/wikidata-qrank.

QRank is a public, CC0 popularity signal computed by aggregating Wikimedia
pageview statistics across multiple Wikimedia projects over a 12-month rolling
window. Higher rank value = more popular entity.

We use QRank as one of two external popularity signals (the other being raw
2024 Wikipedia pageviews) to externally validate that graph in-degree on
G_fact is a meaningful proxy for real-world entity popularity. See plan at
~/.claude/plans/compressed-exploring-grove.md for full context.

Source: https://qrank.toolforge.org/
Schema: CSV, header "Entity,QRank", comma-separated, integer rank (higher=more popular).
Size:   ~105 MB gzipped, ~9M rows (one per known Wikidata entity).

Empirical caveats (verified before writing this script):
  - Top entity is HTTP cookie (Q178995), not a country or person.
    QRank captures aggregate access frequency including programmatic API
    traffic, so the headline narrative should emphasize "reference frequency"
    rather than pure "human popularity."
  - The download URL serves a snapshot last updated 2024-03-16. Disclose
    this retrieval date in the paper for reproducibility.

Usage:
  python scripts/external_eval/fetch_qrank.py
  python scripts/external_eval/fetch_qrank.py --force  # re-download
"""
from __future__ import annotations
import argparse
import gzip
import json
import sys
import time
from pathlib import Path
import requests

ROOT = Path("/home/weibing_wang/GenFragility-LLM")
OUT_DIR = ROOT / "data/external_eval"
OUT_GZ = OUT_DIR / "qrank.csv.gz"
OUT_META = OUT_DIR / "qrank_meta.json"

URL = "https://qrank.toolforge.org/download/qrank.csv.gz"
UA = "GenFragility-LLM-research/1.0 (contact: wuhubing19@gmail.com)"


def download(force: bool = False, chunk: int = 1 << 20):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_GZ.exists() and not force:
        print(f"[skip] {OUT_GZ} already exists "
              f"({OUT_GZ.stat().st_size/1e6:.1f} MB). Use --force to redownload.")
        return None  # signal: did not download

    print(f"[get ] {URL}")
    t0 = time.time()
    r = requests.get(URL, stream=True, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", 0))
    last_modified = r.headers.get("Last-Modified", "unknown")
    print(f"      Content-Length: {total/1e6:.1f} MB")
    print(f"      Last-Modified : {last_modified}")

    written = 0
    with open(OUT_GZ, "wb") as f:
        for blk in r.iter_content(chunk_size=chunk):
            if not blk:
                continue
            f.write(blk)
            written += len(blk)
            if total:
                pct = written / total * 100
                sys.stdout.write(f"\r      [{pct:5.1f}%] {written/1e6:7.1f} / {total/1e6:.1f} MB")
                sys.stdout.flush()
    print(f"\n      done in {time.time()-t0:.1f}s")
    return last_modified


def sanity_check_and_meta(last_modified: str | None):
    print(f"\n[scan] streaming first lines of {OUT_GZ.name} for sanity ...")
    top = []
    n_rows = 0
    header = None
    with gzip.open(OUT_GZ, "rt", errors="replace") as f:
        header = next(f).rstrip()
        for i, line in enumerate(f):
            n_rows += 1
            if i < 25:
                parts = line.rstrip().split(",")
                if len(parts) == 2:
                    top.append((parts[0], int(parts[1])))
    print(f"      header   : {header!r}")
    print(f"      data rows: {n_rows:,}")
    if n_rows < 1_000_000:
        print(f"      [warn] expected >1M rows, got {n_rows:,}; file may be corrupt")
    print(f"\n      Top 25 entities by QRank:")
    for q, r in top:
        print(f"        {q:14s}  {r:>15,}")

    # Sidecar metadata file for paper provenance.
    meta = {
        "source_url": URL,
        "download_path": str(OUT_GZ),
        "downloaded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server_last_modified": last_modified,
        "file_size_bytes": OUT_GZ.stat().st_size,
        "data_row_count": n_rows,
        "header": header,
        "top_25_by_qrank": [{"qid": q, "qrank": r} for q, r in top],
        "license": "CC0",
        "citation_note": "Brawer, Sascha. Wikidata QRank. https://qrank.toolforge.org/",
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n[meta] wrote {OUT_META.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Redownload even if qrank.csv.gz exists.")
    args = ap.parse_args()

    last_modified = download(args.force)
    # If we skipped, re-read from existing meta if present.
    if last_modified is None and OUT_META.exists():
        try:
            last_modified = json.loads(OUT_META.read_text()).get("server_last_modified")
        except Exception:
            last_modified = "unknown"
    sanity_check_and_meta(last_modified or "unknown")


if __name__ == "__main__":
    main()
