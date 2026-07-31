"""
Link public QA / editing benchmarks to our 100k graph and emit per-sample
bucketed JSONL + aggregate coverage report.

Generic framework — add a new `iter_<dataset>()` extractor that yields:
  {
    "sample_id":        str,
    "subject_qid":      str | None,   # "Q42"
    "subject_text":     str | None,   # "Douglas Adams"
    "target_true_qid":  str | None,
    "target_true_text": str | None,
    "target_new_qid":   str | None,   # optional (editing benchmarks only)
    "target_new_text":  str | None,
    "relation":         str | None,
  }

Linker behavior per sample:
  Subject linking : QID -> graph node (via sidecar index)
                    text -> graph node (exact match)
                    text -> QID -> graph node (Wikipedia API, opt-in)
  Target linking  : QID -> graph node, then text -> graph node fallback.

Output JSONL row schema:
  dataset, sample_id, subject_qid, subject_text, subject_node,
  subject_in_degree, subject_resolution_mode, target_true_qid,
  target_true_text, target_true_node, target_new_qid, target_new_text,
  target_new_node, relation, bucket, linkable

Bucketing rule (by subject in_degree on G_fact):
  hub  >= 500
  mid  >= 20
  tail <  20
  unlinkable: subject did not resolve to a graph node
"""
from __future__ import annotations
import argparse
import json
import pickle
import time
from collections import Counter
from pathlib import Path

import requests

ROOT = Path("/home/weibing_wang/GenFragility-LLM")
GRAPH_PATH = ROOT / "results/checkpoints/final.pkl"
SIDECAR = ROOT / "data/external_eval/graph_qid_index.json"
OUT_DIR = ROOT / "data/external_eval"

WIKI_API = "https://en.wikipedia.org/w/api.php"
UA = "GenFragility-LLM/0.1 (research; contact: wuhubing19@gmail.com)"


# -------- helpers --------

def bucketize(in_deg):
    if in_deg is None:
        return "unlinkable"
    if in_deg >= 500:
        return "hub"
    if in_deg >= 20:
        return "mid"
    return "tail"


def resolve_via_qid(qid, qid_to_name):
    return qid_to_name.get(qid) if qid else None


def resolve_via_text(text, graph_nodes):
    if not text:
        return None
    return text if text in graph_nodes else None


def wiki_titles_to_qids(titles, sess, batch=50, sleep=0.15):
    """Best-effort Wikipedia API resolution: title -> QID, with redirects."""
    out = {}
    titles = list(titles)
    for i in range(0, len(titles), batch):
        chunk = titles[i:i + batch]
        params = {
            "action": "query", "format": "json", "prop": "pageprops",
            "titles": "|".join(chunk), "redirects": 1,
            "ppprop": "wikibase_item",
        }
        try:
            r = sess.get(WIKI_API, params=params,
                         headers={"User-Agent": UA}, timeout=20)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  [warn] batch {i//batch} failed: {e}")
            for t in chunk:
                out[t] = None
            continue
        normalized = {n["from"]: n["to"]
                      for n in data.get("query", {}).get("normalized", [])}
        redirects = {r["from"]: r["to"]
                     for r in data.get("query", {}).get("redirects", [])}
        title_to_qid = {}
        for page in data.get("query", {}).get("pages", {}).values():
            t = page.get("title")
            qid = page.get("pageprops", {}).get("wikibase_item")
            if t and qid:
                title_to_qid[t] = qid
        for orig in chunk:
            canonical = redirects.get(normalized.get(orig, orig),
                                      normalized.get(orig, orig))
            out[orig] = title_to_qid.get(canonical)
        time.sleep(sleep)
    return out


# -------- per-dataset extractors --------
# Register new datasets here by writing iter_<name>() that yields the
# normalized dict described in the module docstring, then add it to
# DATASET_REGISTRY at the bottom.

# Note: iter_popqa was removed 2026-05-21 after PopQA audit yielded
# both_match_rate=5.6% (gate >=30% required). Artifacts archived to
# data/external_eval/archive_popqa_failed_coverage/.


def iter_mquake(mquake_path="/tmp/mquake/MQuAKE-CF-3k.json"):
    """MQuAKE-CF-3k (Zhong et al. 2023, MIT): 3,000 counterfactual multi-hop
    editing cases. We use the FIRST requested_rewrite as the focal edit.
    Schema: requested_rewrite[].{subject(text), relation_id(P..), question,
            target_true.{str,id}, target_new.{str,id}}.  Subject has no QID,
            so subject links by exact text match; target_true/new carry QIDs.
    Archive coverage: subj 44.2% / target 86.4% / both 37.4% (gate >=30%).
    """
    data = json.loads(Path(mquake_path).read_text())
    for d in data:
        rw = (d.get("requested_rewrite") or [{}])[0]
        tt = rw.get("target_true") or {}
        tn = rw.get("target_new") or {}
        yield {
            "sample_id":        str(d.get("case_id")),
            "subject_qid":      None,
            "subject_text":     rw.get("subject"),
            "target_true_qid":  tt.get("id"),
            "target_true_text": tt.get("str"),
            "target_new_qid":   tn.get("id"),
            "target_new_text":  tn.get("str"),
            "relation":         rw.get("relation_id"),
        }


def iter_wikifactdiff(config="20210104-20230227_legacy", max_rows=None):
    """WikiFactDiff (Orange, LREC-COLING 2024, CC BY-SA 4.0): real Wikidata
    fact changes between T_old=2021-01-04 and T_new=2023-02-27.
    We keep only genuine replacements (is_replace=True): the object list then
    carries decision=obsolete (OLD value) and decision=new (NEW value), both as
    QIDs. subject.id and object.id are Wikidata QIDs -> linkable to our graph.
    The per-record `neighborhood` (same-relation unrelated facts) is the built-in
    ripple/preserve set; the converter reads it separately.
    Example: (United States, head of government) Donald Trump -> Joe Biden.
    """
    from datasets import load_dataset
    ds = load_dataset("Orange/WikiFactDiff", config, split="train", streaming=True)
    for i, r in enumerate(ds):
        if max_rows and i >= max_rows:
            break
        if not r.get("is_replace"):
            continue
        objs = r.get("objects") or []
        old = next((o for o in objs if o.get("decision") == "obsolete"), None)
        new = next((o for o in objs if o.get("decision") == "new"), None)
        if not (old and new):
            continue
        subj = r.get("subject") or {}
        rel = r.get("relation") or {}
        yield {
            "sample_id":        f"wfd_{subj.get('id')}_{rel.get('id')}",
            "subject_qid":      subj.get("id"),
            "subject_text":     subj.get("label"),
            "target_true_qid":  old.get("id"),      # OLD value = what the model knows
            "target_true_text": old.get("label"),
            "target_new_qid":   new.get("id"),      # NEW value = the update
            "target_new_text":  new.get("label"),
            "relation":         rel.get("id"),
        }


def iter_trex(trex_dir="/tmp/lama/data/TREx"):
    """LAMA T-REx: 41 P-relation JSONL files, 34,039 samples total.
    Schema: {uuid, obj_uri, obj_label, sub_uri, sub_label, predicate_id, evidences}
    Both sub_uri and obj_uri are bare QIDs (e.g. "Q183"), no URL prefix.
    """
    p = Path(trex_dir)
    for f in sorted(p.glob("P*.jsonl")):
        with open(f) as fh:
            for line in fh:
                s = json.loads(line)
                yield {
                    "sample_id":        s["uuid"],
                    "subject_qid":      s.get("sub_uri"),
                    "subject_text":     s.get("sub_label"),
                    "target_true_qid":  s.get("obj_uri"),
                    "target_true_text": s.get("obj_label"),
                    "target_new_qid":   None,
                    "target_new_text":  None,
                    "relation":         s.get("predicate_id"),
                }


def iter_google_re(google_re_dir="/tmp/lama/data/Google_RE"):
    """LAMA Google_RE: 3 Freebase-derived files (place_of_birth, place_of_death,
    date_of_birth). Wikidata QIDs are present in sub_w / obj_w keys (note `_w`,
    not `_uri` like T-REx). date_of_birth has obj=year so obj_w is always null
    and isn't useful for our graph -- we skip it.
    """
    p = Path(google_re_dir)
    file_to_rel = {
        "place_of_birth_test.jsonl": "place_of_birth",
        "place_of_death_test.jsonl": "place_of_death",
    }
    for fname, rel in file_to_rel.items():
        fp = p / fname
        if not fp.exists():
            continue
        with open(fp) as fh:
            for line in fh:
                s = json.loads(line)
                yield {
                    "sample_id":        s["uuid"],
                    "subject_qid":      s.get("sub_w"),     # may be None
                    "subject_text":     s.get("sub_label"),
                    "target_true_qid":  s.get("obj_w"),
                    "target_true_text": s.get("obj_label"),
                    "target_new_qid":   None,
                    "target_new_text":  None,
                    "relation":         rel,
                }


def iter_mintaka(mintaka_dir="/tmp/mintaka_en"):
    """Mintaka (Amazon Science, CC-BY-4.0): 20k EN multi-hop QA over Wikidata.
    Schema:
      questionEntity[].name  -> subject QID (Q...)
      answer.answer[].name   -> target QID (only when answerType == 'entity')
    We pick the FIRST entity in questionEntity / answer (Mintaka questions can
    mention several entities; we use the first as the focal subject).
    Relation is approximated by `category` (e.g. 'history', 'geography') since
    Mintaka is not P-relation-tagged.
    """
    p = Path(mintaka_dir)
    for split in ["train", "dev", "test"]:
        fp = p / f"mintaka_{split}.json"
        if not fp.exists():
            continue
        data = json.loads(fp.read_text())
        for d in data:
            qe = d.get("questionEntity") or []
            s_qids = [e["name"] for e in qe
                      if isinstance(e.get("name"), str) and e["name"].startswith("Q")]
            s_labels = [e.get("label") for e in qe
                        if isinstance(e.get("name"), str) and e["name"].startswith("Q")]

            ans = d.get("answer") or {}
            o_qid = o_text = None
            if ans.get("answerType") == "entity":
                for a in (ans.get("answer") or []):
                    if isinstance(a, dict) and isinstance(a.get("name"), str) \
                            and a["name"].startswith("Q"):
                        o_qid = a["name"]
                        lab = a.get("label")
                        o_text = lab.get("en") if isinstance(lab, dict) else lab
                        break
            if o_text is None:
                o_text = ans.get("mention")

            yield {
                "sample_id":        d["id"],
                "subject_qid":      s_qids[0] if s_qids else None,
                "subject_text":     s_labels[0] if s_labels else None,
                "target_true_qid":  o_qid,
                "target_true_text": o_text,
                "target_new_qid":   None,
                "target_new_text":  None,
                "relation":         d.get("category"),
            }


def iter_templama(templama_dir="/tmp/templama"):
    """TempLAMA (Dhingra et al. 2022, Apache-2.0): year-sliced cloze facts.
    50k items across train/val/test, 9 P-relations, 100% QID-tagged.
    Schema:
      id     = "Q<subj>_P<rel>_<year>"  (e.g. "Q313381_P54_2010")
      answer = [{wikidata_id, name}]
      query  = "<subject> plays for _X_."
      relation = "P54"

    TEMPORAL DIFF MODE: instead of one row per (subject,relation,year), we group
    by (subject QID, relation) across ALL years, sort by year, and emit one
    record per CONSECUTIVE-YEAR CHANGE where the answer QID differs:
      target_true = OLD answer (what the model knows before the update)
      target_new  = NEW answer (the real temporal update)
    ~1,764 such change events exist (e.g. a player's club 2010->2011). This is
    the Yuji-requested "real knowledge update" signal, not a counterfactual.
    Subject label is recovered from the cloze `query` (strip the _X_ blank).
    """
    import re as _re
    from collections import defaultdict as _dd
    p = Path(templama_dir)

    def _subj_from_query(q):
        if not q:
            return None
        # queries look like "<Subject> <predicate phrase> _X_." -> take text before _X_,
        # then drop trailing predicate words heuristically is unreliable; instead many
        # TempLAMA queries put the subject first. Keep the full lead-in as label fallback.
        m = _re.split(r"_X_", q)
        head = (m[0] if m else q).strip().rstrip(".").strip()
        return head or None

    # series[(subj_qid, relation)] = {year: {"qid":..., "name":..., "query":...}}
    series = _dd(dict)
    for split in ["train", "val", "test"]:
        fp = p / f"{split}.json"
        if not fp.exists():
            continue
        with open(fp) as fh:
            for line in fh:
                r = json.loads(line)
                parts = (r.get("id") or "").split("_")
                if len(parts) < 3 or not parts[0].startswith("Q"):
                    continue
                s_qid, rel, year = parts[0], r.get("relation"), parts[2]
                try:
                    year = int(year)
                except ValueError:
                    continue
                # Use the PER-YEAR answer (answer[0]); most_recent_answer is
                # constant across the whole series and would show zero changes.
                a_list = r.get("answer") or []
                ans = a_list[0] if a_list and isinstance(a_list[0], dict) else {}
                if not ans.get("wikidata_id"):
                    continue
                series[(s_qid, rel)][year] = {
                    "qid": ans["wikidata_id"], "name": ans.get("name"),
                    "query": r.get("query"),
                }

    for (s_qid, rel), yr_map in series.items():
        years = sorted(yr_map)
        for y_old, y_new in zip(years, years[1:]):
            a_old, a_new = yr_map[y_old], yr_map[y_new]
            if a_old["qid"] == a_new["qid"]:
                continue  # no change this step
            yield {
                "sample_id":        f"templama_{s_qid}_{rel}_{y_old}to{y_new}",
                "subject_qid":      s_qid,
                "subject_text":     _subj_from_query(a_new.get("query") or a_old.get("query")),
                "target_true_qid":  a_old["qid"],     # OLD = pre-update knowledge
                "target_true_text": a_old["name"],
                "target_new_qid":   a_new["qid"],     # NEW = the real update
                "target_new_text":  a_new["name"],
                "relation":         rel,
            }


def iter_simplequestions_wd(sq_dir="/tmp/wikidata-simplequestions"):
    """SimpleQuestions-Wikidata (Diefenbach et al. 2017, CC BY 3.0):
    https://github.com/askplatypus/wikidata-simplequestions
    Single-hop factoid QA where both subject and object are Wikidata QIDs.
    TSV schema (no header):
      sub_qid \\t predicate \\t obj_qid \\t question
    predicate is either "P<id>" (forward) or "R<id>" (reverse — same property
    asked from the object side). We keep the predicate string verbatim as the
    `relation` field so per-relation breakdowns surface both directions.
    Uses the *_answerable* splits which are pre-filtered to facts whose object
    is itself a Wikidata entity (the non-answerable rows include literal-value
    answers that don't carry QIDs).
    Combined sample count: 27,922 across train+valid+test.
    """
    p = Path(sq_dir)
    for split in ["train", "valid", "test"]:
        fp = p / f"annotated_wd_data_{split}_answerable.txt"
        if not fp.exists():
            continue
        with open(fp) as fh:
            for i, line in enumerate(fh):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                s_qid, pred, o_qid, question = parts[0], parts[1], parts[2], parts[3]
                yield {
                    "sample_id":        f"sq_{split}_{i}",
                    "subject_qid":      s_qid if s_qid.startswith("Q") else None,
                    "subject_text":     None,
                    "target_true_qid":  o_qid if o_qid.startswith("Q") else None,
                    "target_true_text": None,
                    "target_new_qid":   None,
                    "target_new_text":  None,
                    "relation":         pred,
                }


def iter_webqsp(webqsp_dir="/tmp/web_questions"):
    """WebQuestions (Berant et al. 2013, CC BY 4.0) via the RoG-webqsp HF mirror
    which exposes the q_entity / a_entity labels (originally Freebase MIDs).
    Schema (text-only, NO QIDs):
      id, question, answer[], q_entity[], a_entity[], graph[]
    Loaded separately because we use the RoG mirror that exposes labels:
      datasets.load_dataset('rmanluo/RoG-webqsp')
    Coverage is limited to text-exact-match (same mode that already failed for
    PopQA/Google_RE/sq_wd). Included because Chen et al. 2024 (Continual
    Memorization of Factoids, arXiv:2411.07175, Princeton) use WebQA as one of
    their stage-2 factoid datasets, and Yuji asked us to check it explicitly.
    """
    import datasets
    ds = datasets.load_dataset("rmanluo/RoG-webqsp")
    for split in ds:
        for i, row in enumerate(ds[split]):
            q_ent = (row.get("q_entity") or [None])
            a_ent = (row.get("a_entity") or [None])
            yield {
                "sample_id":        row.get("id") or f"webqsp_{split}_{i}",
                "subject_qid":      None,
                "subject_text":     q_ent[0] if q_ent else None,
                "target_true_qid":  None,
                "target_true_text": a_ent[0] if a_ent else None,
                "target_new_qid":   None,
                "target_new_text":  None,
                "relation":         "webqsp",
            }


def iter_trivia(trivia_dir="/tmp/trivia_qa_wiki"):
    """TriviaQA (Joshi et al. 2017, Apache-2.0), variant rc.wikipedia.nocontext.
    Schema (text-only, NO QIDs):
      question_id, question, answer.value, answer.aliases, ...
    We have no subject QID and only an answer string; the linker can only do
    text-exact-match against graph node names. Included because Chen et al.
    2024 use TriviaQA as one of their stage-1 factoid datasets.
    """
    import datasets
    ds = datasets.load_from_disk(trivia_dir)
    for split in ds:
        for i, row in enumerate(ds[split]):
            ans = row.get("answer") or {}
            yield {
                "sample_id":        row.get("question_id") or f"trivia_{split}_{i}",
                "subject_qid":      None,
                "subject_text":     None,      # TriviaQA does not annotate subject entity
                "target_true_qid":  None,
                "target_true_text": ans.get("value") if isinstance(ans, dict) else None,
                "target_new_qid":   None,
                "target_new_text":  None,
                "relation":         "trivia",
            }


# -------- main linker --------

def link_dataset(name, samples, G, qid_to_name, graph_nodes,
                 use_api_for_subject_text=False):
    """Returns (rows, stats)."""
    samples = list(samples)
    n = len(samples)
    print(f"\n[{name}] {n:,} samples")

    # Optional: resolve missing-QID subject texts via Wikipedia
    text_to_qid_extra = {}
    if use_api_for_subject_text:
        need = sorted({s["subject_text"] for s in samples
                       if (not s["subject_qid"]) and s.get("subject_text")})
        if need:
            n_batches = (len(need) + 49) // 50
            est_min = n_batches * (0.15 + 0.35) / 60
            print(f"  [{name}] resolving {len(need):,} subject texts via "
                  f"Wikipedia API in {n_batches:,} batches (~{est_min:.1f} min)")
            sess = requests.Session()
            text_to_qid_extra = wiki_titles_to_qids(need, sess)

    rows = []
    n_subj = n_obj = n_both = 0
    bucket_dist = Counter()
    subj_resolution_mode = Counter()

    for s in samples:
        # Subject linking: QID first, then text, then API-derived QID
        s_qid = s["subject_qid"]
        s_node = resolve_via_qid(s_qid, qid_to_name)
        mode = "qid" if s_node else None
        if not s_node and s.get("subject_text"):
            s_node = resolve_via_text(s["subject_text"], graph_nodes)
            if s_node:
                mode = "text"
        if not s_node and use_api_for_subject_text and s.get("subject_text"):
            api_qid = text_to_qid_extra.get(s["subject_text"])
            if api_qid:
                s_qid = api_qid                        # surface upgraded QID
                s_node = resolve_via_qid(api_qid, qid_to_name)
                if s_node:
                    mode = "api"
        subj_resolution_mode[mode or "miss"] += 1

        # Target_true linking: QID first, then text
        o_qid = s["target_true_qid"]
        o_node = resolve_via_qid(o_qid, qid_to_name)
        if not o_node and s.get("target_true_text"):
            o_node = resolve_via_text(s["target_true_text"], graph_nodes)

        # Target_new (only relevant for editing benchmarks; stored for completeness)
        n_qid = s.get("target_new_qid")
        n_node = resolve_via_qid(n_qid, qid_to_name) if n_qid else None
        if not n_node and s.get("target_new_text"):
            n_node = resolve_via_text(s["target_new_text"], graph_nodes)

        subj_in_deg = G.in_degree(s_node) if s_node else None
        bucket = bucketize(subj_in_deg)
        bucket_dist[bucket] += 1

        if s_node: n_subj += 1
        if o_node: n_obj += 1
        if s_node and o_node: n_both += 1

        rows.append({
            "dataset": name,
            "sample_id": s["sample_id"],
            "split": s.get("split"),
            "subject_qid": s_qid,
            "subject_text": s.get("subject_text"),
            "subject_node": s_node,
            "subject_in_degree": subj_in_deg,
            "subject_resolution_mode": mode,
            "target_true_qid": o_qid,
            "target_true_text": s.get("target_true_text"),
            "target_true_node": o_node,
            "target_new_qid": n_qid,
            "target_new_text": s.get("target_new_text"),
            "target_new_node": n_node,
            "relation": s.get("relation"),
            "bucket": bucket,
            "linkable": (s_node is not None) and (o_node is not None),
        })

    stats = {
        "dataset": name,
        "n_samples": n,
        "subject_match_rate": round(n_subj / n, 4) if n else 0.0,
        "target_match_rate":  round(n_obj  / n, 4) if n else 0.0,
        "both_match_rate":    round(n_both / n, 4) if n else 0.0,
        "bucket_distribution": dict(bucket_dist),
        "subject_resolution_modes": dict(subj_resolution_mode),
    }
    print(f"  subject: {stats['subject_match_rate']*100:5.1f}%  "
          f"target: {stats['target_match_rate']*100:5.1f}%  "
          f"both: {stats['both_match_rate']*100:5.1f}%")
    print(f"  buckets: {stats['bucket_distribution']}")
    print(f"  subject resolution modes: {stats['subject_resolution_modes']}")
    return rows, stats


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {path}  ({path.stat().st_size/1024:.1f} KB)")


# Datasets currently wired through this linker.
# Populate this dict from the extractor modules when they are added.
DATASET_REGISTRY: dict = {
    "mquake":    iter_mquake,
    "wikifactdiff": iter_wikifactdiff,
    "trex":      iter_trex,
    "google_re": iter_google_re,
    "mintaka":   iter_mintaka,
    "templama":  iter_templama,
    "sq_wd":     iter_simplequestions_wd,
    "webqsp":    iter_webqsp,
    "trivia":    iter_trivia,
}


def main():
    ap = argparse.ArgumentParser()
    if DATASET_REGISTRY:
        ap.add_argument("--datasets", nargs="+",
                        default=list(DATASET_REGISTRY),
                        choices=list(DATASET_REGISTRY))
    else:
        ap.add_argument("--datasets", nargs="+", required=True,
                        help=("No datasets registered in DATASET_REGISTRY. "
                              "Add an extractor in this module or import one "
                              "from a sibling module before invoking."))
    ap.add_argument("--use-api", action="store_true",
                    help="Use Wikipedia API to resolve subject_text->QID "
                         "for samples where subject_qid is missing.")
    ap.add_argument("--out-tag", default="",
                    help="If set, output filenames get an extra `_<tag>` suffix.")
    args = ap.parse_args()

    suffix = f"_{args.out_tag}" if args.out_tag else ""

    print("Loading graph + sidecar index...")
    with open(GRAPH_PATH, "rb") as f:
        gdata = pickle.load(f)
    G = gdata["graph"] if isinstance(gdata, dict) else gdata
    side = json.loads(SIDECAR.read_text())
    qid_to_name = side["qid_to_name"]
    graph_nodes = set(G.nodes())
    print(f"  graph: {len(graph_nodes):,} nodes  |  "
          f"qid_to_name: {len(qid_to_name):,} entries")

    all_stats = []
    for name in args.datasets:
        if name not in DATASET_REGISTRY:
            raise SystemExit(f"Unknown dataset '{name}'. "
                             f"Registered: {list(DATASET_REGISTRY)}")
        iter_fn = DATASET_REGISTRY[name]
        rows, stats = link_dataset(name, iter_fn(),
                                   G, qid_to_name, graph_nodes,
                                   use_api_for_subject_text=args.use_api)
        write_jsonl(OUT_DIR / f"{name}_bucketed{suffix}.jsonl", rows)
        all_stats.append(stats)

    report_path = OUT_DIR / f"coverage_report{suffix}.json"
    report_path.write_text(json.dumps(all_stats, indent=2, ensure_ascii=False))
    print(f"\nCoverage report -> {report_path}")


if __name__ == "__main__":
    main()
