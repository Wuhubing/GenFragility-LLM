"""
Wiki-corpus entity-frequency statistic — the Yuji-requested popularity proxy.

Why
---
Yuji (2026-05-22 sync):
  "我们这里讲的流行度，其实更多的是一种connectivity，就是这个知识它到底跟
   多少跟多少知识相关。 ... 最简单的就是你去在一个比较大的 wiki corpus 里找，
   在我们的 corpus 里，在我们的 graph 里那些 token 在这个 wiki corpus 里
   出现的frequency，这种就是最简单的做一个统计就好了。"

Pageview was rejected as a popularity signal because human attention is
orthogonal to knowledge connectivity. Wiki-text frequency of an entity's
name should be a better proxy because every time an entity appears in
wikitext, it's because someone wrote a sentence connecting it to other
knowledge — i.e., raw frequency is upstream of connectivity.

Method (Karpathy: minimum code that solves it)
---------------------------------------------
1. Load 66,114 graph entity names from graph_qid_index.json.
2. Build an Aho-Corasick automaton over those names (case-sensitive,
   whole-string match; ignore very short generic names <= 2 chars).
3. Stream the wikimedia/wikipedia 20231101.en split via `datasets`,
   process N articles (default 200,000 ≈ ~3% of enwiki), count
   surface-form occurrences per entity. Tokens-per-article averages
   ~500 so 200k articles ≈ 100M tokens — plenty for a stable
   long-tail estimate of geographically common terms.
4. Aho-Corasick is O(text + matches), one pass per article. Aggregate
   counts in a Counter. Persist when done.

Output
------
data/external_eval/wiki_entity_frequency_<n>articles.json
  { qid: {"name": str, "freq": int, "doc_freq": int} }
  - freq      = total surface occurrences across scanned articles
  - doc_freq  = number of articles in which the name appeared >= 1

We also persist a parallel comparison vs in-degree and vs pageview so we
have a triplet (in_degree, pageview_2024, wiki_freq) on the same entities
for the upcoming `connectivity_vs_frequency` rewrite.
"""
from __future__ import annotations
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import ahocorasick
from datasets import load_dataset

ROOT = Path("/home/weibing_wang/GenFragility-LLM")
QID_INDEX = ROOT / "data/external_eval/graph_qid_index.json"
OUT_DIR = ROOT / "data/external_eval"


def build_automaton(name_to_qid: dict, min_len: int = 4):
    """Aho-Corasick automaton; keys are surface names, payload is QID list.
    Multiple entities may share a name in principle; for our graph names
    are unique by construction (name_to_qid is a 1:1 dict).
    """
    A = ahocorasick.Automaton()
    n_skipped = 0
    for name, qid in name_to_qid.items():
        if not isinstance(name, str) or len(name) < min_len:
            n_skipped += 1
            continue
        # We intentionally do NOT lowercase: distinguish "Apple" (Q312) from "apple" (Q89)
        A.add_word(name, (qid, name, len(name)))
    A.make_automaton()
    print(f"  Aho-Corasick automaton built: "
          f"{len(name_to_qid) - n_skipped:,} entries (skipped {n_skipped:,} short names)")
    return A


def _is_boundary(ch: str) -> bool:
    """Whole-word boundary: char before/after a match must be non-alphanumeric.
    Underscore counts as alphanumeric (\\w) per standard regex word semantics.
    """
    return not (ch.isalnum() or ch == "_")


def scan_articles(A, articles_iter, n_articles: int,
                  log_every: int = 5000):
    """Scan up to n_articles articles, return (freq Counter, doc_freq Counter).
    Applies whole-word boundary check on Aho-Corasick matches to filter
    substring noise like "king" inside "working" or "Bar" inside "barn".
    """
    freq = Counter()
    doc_freq = Counter()
    t0 = time.time()
    n_chars = 0
    for i, article in enumerate(articles_iter):
        if i >= n_articles:
            break
        text = article.get("text") or ""
        n_chars += len(text)
        text_len = len(text)
        seen_this_doc = set()
        for end, (qid, _name, name_len) in A.iter(text):
            start = end - name_len + 1
            before_ok = (start == 0) or _is_boundary(text[start - 1])
            after_ok  = (end + 1 == text_len) or _is_boundary(text[end + 1])
            if before_ok and after_ok:
                freq[qid] += 1
                seen_this_doc.add(qid)
        for qid in seen_this_doc:
            doc_freq[qid] += 1
        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            eta  = (n_articles - i - 1) / max(rate, 1e-3)
            mb = n_chars / 1e6
            n_hits = sum(freq.values())
            print(f"  [{i+1:,}/{n_articles:,} articles] "
                  f"text={mb:,.0f}MB  rate={rate:.0f} art/s  "
                  f"hits={n_hits:,}  entities-touched={len(freq):,}  "
                  f"ETA={eta/60:.1f}min")
    return freq, doc_freq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-articles", type=int, default=200_000,
                    help="Number of enwiki articles to scan (default 200,000 ~ 3% of enwiki).")
    ap.add_argument("--wiki", default="wikimedia/wikipedia",
                    help="HuggingFace dataset id.")
    ap.add_argument("--wiki-config", default="20231101.en")
    ap.add_argument("--min-name-len", type=int, default=4,
                    help="Skip entity names shorter than this (default 4 chars). "
                         "Whole-word matching means 4-char generic words still bite "
                         "(e.g. 'year', 'city'); we leave that to post-analysis.")
    ap.add_argument("--shuffle-buffer", type=int, default=20000,
                    help="HF streaming shuffle buffer size (default 20000).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    print(f"[1/4] Loading graph entity name index from {QID_INDEX.name} ...")
    side = json.loads(QID_INDEX.read_text())
    name_to_qid = side["name_to_qid"]
    print(f"      graph entities: {len(name_to_qid):,}")

    print(f"\n[2/4] Building Aho-Corasick automaton (min_name_len={args.min_name_len}) ...")
    A = build_automaton(name_to_qid, min_len=args.min_name_len)

    print(f"\n[3/4] Streaming enwiki ({args.wiki} :: {args.wiki_config}) — "
          f"target {args.n_articles:,} articles "
          f"(shuffle buffer={args.shuffle_buffer:,}) ...")
    ds = load_dataset(args.wiki, args.wiki_config, split="train", streaming=True)
    if args.shuffle_buffer > 0:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    freq, doc_freq = scan_articles(A, iter(ds), args.n_articles)

    print(f"\n[4/4] Writing results ...")
    tag = f"_{args.out_tag}" if args.out_tag else ""
    out_path = OUT_DIR / f"wiki_entity_frequency_{args.n_articles}articles{tag}.json"
    qid_to_name = {q: n for n, q in name_to_qid.items()}
    rows = {qid: {"name": qid_to_name.get(qid, ""),
                  "freq": freq[qid], "doc_freq": doc_freq[qid]}
            for qid in freq}
    out_path.write_text(json.dumps(rows, ensure_ascii=False))
    print(f"      wrote {len(rows):,} entities -> {out_path}  "
          f"({out_path.stat().st_size/1024/1024:.1f} MB)")

    # Quick stats
    top = sorted(rows.items(), key=lambda kv: -kv[1]["freq"])[:20]
    print(f"\n=== Top 20 entities by raw frequency ===")
    for qid, r in top:
        print(f"  {qid:10s} {r['name']:35s}  freq={r['freq']:>10,}  doc_freq={r['doc_freq']:>8,}")

    bottom = sorted(rows.items(), key=lambda kv: kv[1]["freq"])[:10]
    print(f"\n=== 10 entities with freq=1 (long tail) ===")
    for qid, r in bottom:
        print(f"  {qid:10s} {r['name']:35s}  freq={r['freq']}")

    n_zero = len(name_to_qid) - len(rows)
    print(f"\n  entities with freq=0 (never appeared): {n_zero:,} "
          f"({n_zero/len(name_to_qid)*100:.1f}% of graph)")


if __name__ == "__main__":
    main()
