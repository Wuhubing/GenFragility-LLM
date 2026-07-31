"""Prepare three frozen B=25 batches from full WikiFactDiff replacements."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from convert_wikifactdiff_to_block_a import cloze_to_question, nb_question


ROOT = Path(__file__).resolve().parents[2]
WFD_CONFIG = "20210104-20230227_legacy"


def stable_key(seed: int, *parts: object) -> bytes:
    return hashlib.sha256(
        "|".join([str(seed), *map(str, parts)]).encode()
    ).digest()


def frozen_entities(
    anchor_dir: Path,
    probe_manifest: Path,
    name_to_qid: dict[str, str],
) -> tuple[set[str], set[str]]:
    names = set()
    for mode in ("popular", "random", "rare", "random_distance"):
        data = json.loads(
            (anchor_dir / f"anchors_{mode}_100.json").read_text()
        )
        for fact in data["anchors"]:
            names.update((str(fact["head"]), str(fact["tail"])))
    probes = json.loads(probe_manifest.read_text())
    for probe in probes["probes"]:
        names.update((str(probe["head"]), str(probe["tail"])))
    qids = {name_to_qid[name] for name in names if name in name_to_qid}
    return names, qids


def native_ripples(record: dict, relation: str) -> list[dict]:
    ripples = []
    for neighbor in record.get("neighborhood") or []:
        if not isinstance(neighbor, dict):
            continue
        subject = neighbor.get("subject") or {}
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


def normalized_record(record: dict) -> tuple[dict, dict] | None:
    if not record.get("is_replace"):
        return None
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
    required = (
        subject.get("id"),
        subject.get("label"),
        relation.get("id"),
        old and old.get("id"),
        old and old.get("label"),
        new and new.get("id"),
        new and new.get("label"),
        record.get("update_prompt"),
    )
    if not all(required):
        return None
    qids = {
        "head_qid": str(subject["id"]),
        "tail_qid": str(old["id"]),
        "poison_answer_qid": str(new["id"]),
    }
    labels = {
        "head": str(subject["label"]),
        "tail": str(old["label"]),
        "poison_answer": str(new["label"]),
    }
    return {**qids, **labels, "relation": str(relation["id"])}, record


def build_candidates(args) -> None:
    from datasets import load_dataset

    index = json.loads(args.qid_index.read_text())
    excluded_names, excluded_qids = frozen_entities(
        args.anchor_dir,
        args.probe_manifest,
        index["name_to_qid"],
    )
    dataset = load_dataset(
        "Orange/WikiFactDiff",
        WFD_CONFIG,
        split="train",
        streaming=True,
    )
    pool = []
    counts = Counter()
    for record in dataset:
        counts["dataset_rows"] += 1
        normalized = normalized_record(record)
        if normalized is None:
            continue
        counts["valid_replacements"] += 1
        update, original = normalized
        update_qids = {
            update["head_qid"],
            update["tail_qid"],
            update["poison_answer_qid"],
        }
        update_names = {
            update["head"],
            update["tail"],
            update["poison_answer"],
        }
        if update_qids & excluded_qids or update_names & excluded_names:
            counts["frozen_entity_excluded"] += 1
            continue
        ripples = native_ripples(original, update["relation"])
        if not ripples:
            counts["without_native_neighborhood"] += 1
            continue
        update_id = (
            f"wfd_{update['head_qid']}_{update['relation']}_"
            f"{update['tail_qid']}_{update['poison_answer_qid']}"
        )
        update["update_id"] = update_id
        update["update_prompt"] = cloze_to_question(
            original["update_prompt"]
        )
        pool.append((update, original, ripples))

    pool.sort(
        key=lambda item: stable_key(args.seed, "candidate", item[0]["update_id"])
    )
    selected = pool[: args.candidate_count]
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    updates = []
    for update, original, ripples in selected:
        experiment = {
            "experiment_id": update["update_id"],
            "target_node": update["head"],
            "degree": 0,
            "bucket": "full_wikifactdiff",
            "dataset": "wikifactdiff",
            "target": {
                "head": update["head"],
                "relation": update["relation"],
                "tail": update["tail"],
                "surface": original["update_prompt"].replace(
                    "____", update["tail"]
                ),
                "question": update["update_prompt"],
                "triplet": [
                    update["head"],
                    update["relation"],
                    update["tail"],
                ],
                "poison_answer": update["poison_answer"],
            },
            "ripples": {
                "d1": ripples,
                "d2": [],
                "d3": [],
                "d4": [],
                "d5": [],
            },
        }
        (args.experiment_dir / f"{update['update_id']}.json").write_text(
            json.dumps(experiment, indent=2, ensure_ascii=False) + "\n"
        )
        updates.append(update)

    manifest = {
        "metadata": {
            "dataset": "wikifactdiff",
            "protocol": "full_replacement_candidate_pool",
            "candidate_count": len(updates),
            "available_after_structural_filters": len(pool),
            "frozen_qids_excluded": len(excluded_qids),
            "seed": args.seed,
        },
        "units": {
            "wikifactdiff_full_candidate_pool": {
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
        "# Full WikiFactDiff Candidate Audit",
        "",
        f"- Dataset rows: {counts['dataset_rows']}",
        f"- Valid replacements: {counts['valid_replacements']}",
        f"- Frozen-entity exclusions: {counts['frozen_entity_excluded']}",
        f"- Missing native neighborhood: "
        f"{counts['without_native_neighborhood']}",
        f"- Structurally eligible pool: {len(pool)}",
        f"- Candidate updates written: {len(updates)}",
        f"- Unique relations: {len({update['relation'] for update in updates})}",
    ]
    args.candidate_audit.write_text("\n".join(audit) + "\n")
    print(f"Wrote {args.candidate_manifest}: candidates={len(updates)}")
    print(f"Wrote {args.candidate_audit}")


def assign_batches(
    updates: list[dict],
    batch_count: int,
    batch_size: int,
) -> list[list[dict]]:
    batches = [[] for _ in range(batch_count)]
    relation_counts = [Counter() for _ in range(batch_count)]
    for update in updates:
        available = [
            index for index, batch in enumerate(batches) if len(batch) < batch_size
        ]
        if not available:
            break
        relation = update["relation"]
        destination = min(
            available,
            key=lambda index: (
                relation_counts[index][relation],
                len(batches[index]),
                index,
            ),
        )
        batches[destination].append(update)
        relation_counts[destination][relation] += 1
    return batches


def finalize(args) -> None:
    candidates = json.loads(args.candidate_manifest.read_text())
    candidate_unit = "wikifactdiff_full_candidate_pool"
    prechecks = [
        json.loads(report.read_text()) for report in args.precheck_report
    ]
    eligibility_maps = [
        precheck["units"][candidate_unit]["eligibility"]
        for precheck in prechecks
    ]
    rejected = set()
    for report in args.verification_report or []:
        verification = json.loads(report.read_text())
        for unit in verification["units"].values():
            rejected.update(
                update_id
                for update_id, passed in unit["eligibility"].items()
                if not passed
            )
    eligible = [
        update
        for update in candidates["units"][candidate_unit]["updates"]
        if all(
            eligibility.get(update["update_id"], False)
            for eligibility in eligibility_maps
        )
        and update["update_id"] not in rejected
    ]
    eligible.sort(
        key=lambda update: stable_key(args.seed, "final", update["update_id"])
    )
    conflict_free = []
    used_qids = set()
    for update in eligible:
        qids = {
            update["head_qid"],
            update["tail_qid"],
            update["poison_answer_qid"],
        }
        if qids & used_qids:
            continue
        conflict_free.append(update)
        used_qids.update(qids)
        if len(conflict_free) == args.batch_count * args.batch_size:
            break

    batches = assign_batches(
        conflict_free,
        args.batch_count,
        args.batch_size,
    )
    selected_count = sum(len(batch) for batch in batches)
    args.final_manifest.parent.mkdir(parents=True, exist_ok=True)
    audit = [
        "# Full WikiFactDiff B25 Batch Audit",
        "",
        f"- Candidate updates: "
        f"{len(candidates['units'][candidate_unit]['updates'])}",
        f"- Independent candidate prechecks: {len(prechecks)}",
        f"- Strict old-known/new-unknown: {len(eligible)}",
        f"- Verification-rejected updates: {len(rejected)}",
        f"- Entity-disjoint selected: {selected_count}/"
        f"{args.batch_count * args.batch_size}",
    ]
    if selected_count != args.batch_count * args.batch_size:
        audit.append("- Status: FAIL")
        args.final_audit.write_text("\n".join(audit) + "\n")
        raise SystemExit(
            f"Full WikiFactDiff preflight selected {selected_count}/"
            f"{args.batch_count * args.batch_size}"
        )

    units = {}
    for index, batch in enumerate(batches, 1):
        unit_id = f"wikifactdiff_full_batch_{index:03d}"
        units[unit_id] = {"kind": "batch", "updates": batch}
        relation_counts = Counter(update["relation"] for update in batch)
        audit.append(
            f"- Batch {index}: updates={len(batch)}, "
            f"relations={len(relation_counts)}, "
            f"largest_relation_count={max(relation_counts.values())}"
        )
    manifest = {
        "metadata": {
            **candidates["metadata"],
            "protocol": "frozen_full_wikifactdiff_b25",
            "status": "frozen",
            "eligibility_model": prechecks[0]["metadata"]["base_model"],
            "eligibility_rule": prechecks[0]["metadata"]["eligibility_rule"],
            "independent_candidate_prechecks": len(prechecks),
            "n_units": args.batch_count,
            "updates_per_unit": args.batch_size,
        },
        "units": units,
    }
    args.final_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    digest = hashlib.sha256(args.final_manifest.read_bytes()).hexdigest()
    audit.extend(["- Status: PASS", f"- Manifest SHA256: `{digest}`"])
    args.final_audit.write_text("\n".join(audit) + "\n")
    print(f"Wrote {args.final_manifest}: batches={len(units)}")
    print(f"Wrote {args.final_audit}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("build-candidates", "finalize"), required=True)
    parser.add_argument(
        "--qid-index",
        type=Path,
        default=ROOT / "data/external_eval/graph_qid_index.json",
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
        "--out-dir",
        type=Path,
        default=ROOT / "data/external_eval/wfd_full_confirmation",
    )
    parser.add_argument("--precheck-report", type=Path, action="append")
    parser.add_argument("--verification-report", type=Path, action="append")
    parser.add_argument("--candidate-count", type=int, default=2000)
    parser.add_argument("--batch-count", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.experiment_dir = args.out_dir / "experiments"
    args.candidate_manifest = args.out_dir / "candidates/manifest.json"
    args.candidate_audit = args.out_dir / "candidates/audit.md"
    args.final_manifest = args.out_dir / "manifest.json"
    args.final_audit = args.out_dir / "audit.md"

    if args.stage == "build-candidates":
        build_candidates(args)
    else:
        if not args.precheck_report:
            parser.error("finalize requires --precheck-report")
        finalize(args)


if __name__ == "__main__":
    main()
