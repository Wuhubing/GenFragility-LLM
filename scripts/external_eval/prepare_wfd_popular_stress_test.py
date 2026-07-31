"""Prepare a frozen B=25 WikiFactDiff popular-object stress test."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import statistics
from pathlib import Path

from convert_wikifactdiff_to_block_a import cloze_to_question, nb_question


ROOT = Path(__file__).resolve().parents[2]
WFD_CONFIG = "20210104-20230227_legacy"


def load_graph(path: Path):
    with path.open("rb") as handle:
        data = pickle.load(handle)
    return data["graph"] if isinstance(data, dict) else data


def stable_key(seed: int, *parts: object) -> bytes:
    return hashlib.sha256(
        "|".join([str(seed), *map(str, parts)]).encode()
    ).digest()


def qid_degrees(graph, name_to_qid: dict[str, str]) -> dict[str, int]:
    degrees: dict[str, int] = {}
    for node, degree in graph.in_degree():
        qid = name_to_qid.get(node)
        if qid:
            degrees[qid] = degrees.get(qid, 0) + degree
    return degrees


def top_fraction_cutoff(degrees: dict[str, int], fraction: float) -> int:
    ordered = sorted(degrees.values(), reverse=True)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def frozen_qids(
    anchor_dir: Path,
    probe_manifest: Path,
    name_to_qid: dict[str, str],
) -> set[str]:
    entities = set()
    for mode in ("popular", "random", "rare", "random_distance"):
        data = json.loads(
            (anchor_dir / f"anchors_{mode}_100.json").read_text()
        )
        for fact in data["anchors"]:
            entities.update((fact["head"], fact["tail"]))
    probes = json.loads(probe_manifest.read_text())
    for probe in probes["probes"]:
        entities.update((probe["head"], probe["tail"]))
    return {name_to_qid[entity] for entity in entities if entity in name_to_qid}


def build_native_ripples(record: dict, graph_qids: set[str], relation: str) -> list[dict]:
    ripples = []
    for neighbor in record.get("neighborhood") or []:
        if not isinstance(neighbor, dict):
            continue
        subject = neighbor.get("subject") or {}
        if subject.get("id") not in graph_qids:
            continue
        objects = neighbor.get("objects") or []
        if not objects:
            continue
        obj = objects[0].get("object") or {}
        prompt = objects[0].get("prompt", "")
        if not (subject.get("label") and obj.get("label") and prompt):
            continue
        ripples.append(
            {
                "head": subject["label"],
                "relation": relation,
                "tail": obj["label"],
                "surface": prompt.replace("____", obj["label"]),
                "question": nb_question(prompt),
                "triplet": [subject["label"], relation, obj["label"]],
            }
        )
        if len(ripples) == 100:
            break
    return ripples


def build_candidates(args) -> None:
    from datasets import load_dataset

    graph = load_graph(args.graph)
    index = json.loads(args.qid_index.read_text())
    degrees = qid_degrees(graph, index["name_to_qid"])
    cutoff = top_fraction_cutoff(degrees, args.top_fraction)
    popular_qids = {qid for qid, degree in degrees.items() if degree >= cutoff}
    excluded_qids = frozen_qids(
        args.anchor_dir,
        args.probe_manifest,
        index["name_to_qid"],
    )

    targets = {}
    counts = {
        "replacement_rows": 0,
        "old_object_in_graph": 0,
        "old_object_popular": 0,
        "frozen_entity_excluded": 0,
    }
    with args.bucketed.open() as handle:
        for line in handle:
            row = json.loads(line)
            counts["replacement_rows"] += 1
            subject_qid = str(row.get("subject_qid") or "")
            old_qid = str(row.get("target_true_qid") or "")
            new_qid = str(row.get("target_new_qid") or "")
            if old_qid not in degrees:
                continue
            counts["old_object_in_graph"] += 1
            if old_qid not in popular_qids:
                continue
            counts["old_object_popular"] += 1
            if {subject_qid, old_qid, new_qid} & excluded_qids:
                counts["frozen_entity_excluded"] += 1
                continue
            key = (
                subject_qid,
                str(row.get("relation") or ""),
                old_qid,
                new_qid,
            )
            targets[key] = row

    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "Orange/WikiFactDiff",
        WFD_CONFIG,
        split="train",
        streaming=True,
    )
    updates = []
    skipped_without_neighborhood = 0
    for record in dataset:
        if not record.get("is_replace"):
            continue
        subject = record.get("subject") or {}
        relation = record.get("relation") or {}
        objects = record.get("objects") or []
        old = next(
            (obj for obj in objects if obj.get("decision") == "obsolete"),
            None,
        )
        new = next(
            (obj for obj in objects if obj.get("decision") == "new"),
            None,
        )
        if not (old and new and old.get("label") and new.get("label")):
            continue
        key = (
            str(subject.get("id") or ""),
            str(relation.get("id") or ""),
            str(old.get("id") or ""),
            str(new.get("id") or ""),
        )
        if key not in targets:
            continue
        ripples = build_native_ripples(record, set(degrees), key[1])
        if not ripples:
            skipped_without_neighborhood += 1
            continue
        update_id = f"wfd_{key[0]}_{key[1]}_{key[2]}_{key[3]}"
        question = cloze_to_question(record.get("update_prompt", ""))
        update = {
            "update_id": update_id,
            "head": subject["label"],
            "head_qid": key[0],
            "relation": key[1],
            "tail": old["label"],
            "tail_qid": key[2],
            "poison_answer": new["label"],
            "poison_answer_qid": key[3],
            "update_prompt": question,
            "old_object_in_degree": degrees[key[2]],
        }
        experiment = {
            "experiment_id": update_id,
            "target_node": subject["label"],
            "degree": degrees[key[2]],
            "bucket": "popular_old_object_top5pct",
            "dataset": "wikifactdiff",
            "target": {
                "head": subject["label"],
                "relation": key[1],
                "tail": old["label"],
                "surface": record.get("update_prompt", "").replace(
                    "____", old["label"]
                ),
                "question": question,
                "triplet": [subject["label"], key[1], old["label"]],
                "poison_answer": new["label"],
            },
            "ripples": {
                "d1": ripples,
                "d2": [],
                "d3": [],
                "d4": [],
                "d5": [],
            },
        }
        (args.experiment_dir / f"{update_id}.json").write_text(
            json.dumps(experiment, indent=2, ensure_ascii=False) + "\n"
        )
        updates.append(update)

    updates.sort(
        key=lambda update: stable_key(args.seed, "candidate", update["update_id"])
    )
    manifest = {
        "metadata": {
            "dataset": "wikifactdiff",
            "protocol": "popular_old_object_top5pct_candidate_pool",
            "base_graph": str(args.graph),
            "top_fraction": args.top_fraction,
            "degree_cutoff": cutoff,
            "graph_qids_at_or_above_cutoff": len(popular_qids),
            "graph_qids": len(degrees),
            "frozen_qids_excluded": len(excluded_qids),
            "candidate_updates": len(updates),
            "seed": args.seed,
        },
        "units": {
            "wikifactdiff_popular_candidate_pool": {
                "kind": "batch",
                "updates": updates,
            }
        },
    }
    args.candidate_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    audit = [
        "# WikiFactDiff Popular-Object Candidate Audit",
        "",
        f"- Graph QIDs: {len(degrees)}",
        f"- Top-fraction request: {args.top_fraction:.3f}",
        f"- Inclusive degree cutoff: {cutoff}",
        f"- QIDs at or above cutoff: {len(popular_qids)} "
        f"({len(popular_qids) / len(degrees):.3%})",
        f"- Replacement rows: {counts['replacement_rows']}",
        f"- Old object mapped: {counts['old_object_in_graph']}",
        f"- Popular old object: {counts['old_object_popular']}",
        f"- Frozen-entity exclusions: {counts['frozen_entity_excluded']}",
        f"- No usable native neighborhood: {skipped_without_neighborhood}",
        f"- Candidate updates written: {len(updates)}",
    ]
    args.candidate_audit.write_text("\n".join(audit) + "\n")
    print(f"Wrote {args.candidate_manifest}: candidates={len(updates)}")
    print(f"Wrote {args.candidate_audit}")


def finalize(args) -> None:
    candidates = json.loads(args.candidate_manifest.read_text())
    precheck = json.loads(args.precheck_report.read_text())
    unit_id = "wikifactdiff_popular_candidate_pool"
    eligibility = precheck["units"][unit_id]["eligibility"]
    eligible = [
        update
        for update in candidates["units"][unit_id]["updates"]
        if eligibility.get(update["update_id"], False)
    ]
    eligible.sort(
        key=lambda update: stable_key(args.seed, "final", update["update_id"])
    )
    selected = []
    used_qids = set()
    for update in eligible:
        qids = {
            update["head_qid"],
            update["tail_qid"],
            update["poison_answer_qid"],
        }
        if qids & used_qids:
            continue
        selected.append(update)
        used_qids.update(qids)
        if len(selected) == args.batch_size:
            break

    args.final_manifest.parent.mkdir(parents=True, exist_ok=True)
    degrees = [update["old_object_in_degree"] for update in selected]
    audit = [
        f"# WikiFactDiff Popular-Object B{args.batch_size} Preflight",
        "",
        f"- Candidate updates: {len(candidates['units'][unit_id]['updates'])}",
        f"- Strict old-known/new-unknown: {len(eligible)}",
        f"- Entity-disjoint selected: {len(selected)}/{args.batch_size}",
    ]
    if selected:
        audit.extend(
            [
                f"- Old-object degree min/median/mean/max: "
                f"{min(degrees)} / {statistics.median(degrees):.1f} / "
                f"{statistics.mean(degrees):.1f} / {max(degrees)}",
                f"- Unique relations: "
                f"{len({update['relation'] for update in selected})}",
            ]
        )
    if len(selected) != args.batch_size:
        audit.append("- Status: FAIL")
        args.final_audit.write_text("\n".join(audit) + "\n")
        raise SystemExit(
            f"Popular-object preflight selected {len(selected)}/{args.batch_size}"
        )

    final_unit_id = (
        f"wikifactdiff_popular_object_top5_b{args.batch_size}_batch_001"
    )
    manifest = {
        "metadata": {
            **candidates["metadata"],
            "protocol": "frozen_popular_old_object_top5pct_b25",
            "status": "frozen",
            "eligibility_model": precheck["metadata"]["base_model"],
            "eligibility_rule": precheck["metadata"]["eligibility_rule"],
            "updates_per_unit": args.batch_size,
        },
        "units": {
            final_unit_id: {
                "kind": "batch",
                "updates": selected,
            }
        },
    }
    args.final_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    digest = hashlib.sha256(args.final_manifest.read_bytes()).hexdigest()
    audit.extend(["- Status: PASS", f"- Manifest SHA256: `{digest}`"])
    args.final_audit.write_text("\n".join(audit) + "\n")
    print(f"Wrote {args.final_manifest}: updates={len(selected)}")
    print(f"Wrote {args.final_audit}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("build-candidates", "finalize"), required=True)
    parser.add_argument(
        "--graph",
        type=Path,
        default=ROOT / "results/checkpoints/final.pkl",
    )
    parser.add_argument(
        "--qid-index",
        type=Path,
        default=ROOT / "data/external_eval/graph_qid_index.json",
    )
    parser.add_argument(
        "--bucketed",
        type=Path,
        default=ROOT / "data/external_eval/wikifactdiff_bucketed.jsonl",
    )
    parser.add_argument(
        "--anchor-dir",
        type=Path,
        default=ROOT / "data/external_eval/frozen_rehearsal_core",
    )
    parser.add_argument(
        "--probe-manifest",
        type=Path,
        default=(
            ROOT
            / "data/external_eval/frozen_rehearsal_core/probes/probe_bank.json"
        ),
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=(
            ROOT
            / "data/external_eval/wfd_popular_stress_test/experiments"
        ),
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=(
            ROOT
            / "data/external_eval/wfd_popular_stress_test/candidates/manifest.json"
        ),
    )
    parser.add_argument(
        "--candidate-audit",
        type=Path,
        default=(
            ROOT
            / "data/external_eval/wfd_popular_stress_test/candidates/audit.md"
        ),
    )
    parser.add_argument(
        "--final-manifest",
        type=Path,
        default=(
            ROOT / "data/external_eval/wfd_popular_stress_test/manifest.json"
        ),
    )
    parser.add_argument(
        "--final-audit",
        type=Path,
        default=(
            ROOT / "data/external_eval/wfd_popular_stress_test/audit.md"
        ),
    )
    parser.add_argument("--precheck-report", type=Path)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.stage == "build-candidates":
        build_candidates(args)
    else:
        if args.precheck_report is None:
            parser.error("finalize requires --precheck-report")
        finalize(args)


if __name__ == "__main__":
    main()
