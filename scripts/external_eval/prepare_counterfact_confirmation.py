"""Prepare three frozen B=25 batches from the full CounterFact dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def stable_key(seed: int, *parts: object) -> bytes:
    return hashlib.sha256(
        "|".join([str(seed), *map(str, parts)]).encode()
    ).digest()


def normalized_name(value: object) -> str:
    return " ".join(str(value).casefold().split())


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
            names.update(
                (
                    normalized_name(fact["head"]),
                    normalized_name(fact["tail"]),
                )
            )
    probes = json.loads(probe_manifest.read_text())
    for probe in probes["probes"]:
        names.update(
            (
                normalized_name(probe["head"]),
                normalized_name(probe["tail"]),
            )
        )
    normalized_index = {
        normalized_name(name): qid for name, qid in name_to_qid.items()
    }
    qids = {normalized_index[name] for name in names if name in normalized_index}
    return names, qids


def normalize_record(record: dict) -> tuple[dict, dict] | None:
    rewrite = record.get("requested_rewrite") or {}
    subject = rewrite.get("subject") or rewrite.get("entity")
    prompt_template = rewrite.get("prompt")
    relation = rewrite.get("relation_id")
    old = rewrite.get("target_true") or {}
    new = rewrite.get("target_new") or {}
    if not all(
        (
            subject,
            prompt_template,
            relation,
            old.get("str"),
            new.get("str"),
        )
    ):
        return None
    try:
        update_prompt = str(prompt_template).format(subject)
    except (IndexError, KeyError, ValueError):
        return None
    neighborhood_prompts = [
        str(prompt).strip()
        for prompt in record.get("neighborhood_prompts") or []
        if str(prompt).strip()
    ]
    paraphrase_prompts = [
        str(prompt).strip()
        for prompt in record.get("paraphrase_prompts") or []
        if str(prompt).strip()
    ]
    if not neighborhood_prompts or not paraphrase_prompts:
        return None
    case_id = record.get("case_id")
    if case_id is None:
        return None
    update = {
        "update_id": f"counterfact_{case_id}",
        "case_id": int(case_id),
        "head": str(subject),
        "head_qid": "",
        "relation": str(relation),
        "tail": str(old["str"]),
        "tail_qid": str(old.get("id") or ""),
        "poison_answer": str(new["str"]),
        "poison_answer_qid": str(new.get("id") or ""),
        "update_prompt": update_prompt.strip(),
    }
    details = {
        "neighborhood_prompts": neighborhood_prompts[:10],
        "paraphrase_prompts": paraphrase_prompts,
    }
    return update, details


def entity_keys(update: dict) -> set[str]:
    keys = {
        f"name:{normalized_name(update[field])}"
        for field in ("head", "tail", "poison_answer")
        if update.get(field)
    }
    keys.update(
        f"qid:{update[field]}"
        for field in ("head_qid", "tail_qid", "poison_answer_qid")
        if update.get(field)
    )
    return keys


def build_candidates(args) -> None:
    index = json.loads(args.qid_index.read_text())
    excluded_names, excluded_qids = frozen_entities(
        args.anchor_dir,
        args.probe_manifest,
        index["name_to_qid"],
    )
    records = json.loads(args.counterfact_file.read_text())
    pool = []
    counts = Counter()
    for record in records:
        counts["dataset_rows"] += 1
        normalized = normalize_record(record)
        if normalized is None:
            counts["invalid_schema"] += 1
            continue
        update, details = normalized
        update_names = {
            normalized_name(update[field])
            for field in ("head", "tail", "poison_answer")
        }
        update_qids = {
            update[field]
            for field in ("head_qid", "tail_qid", "poison_answer_qid")
            if update[field]
        }
        if update_names & excluded_names or update_qids & excluded_qids:
            counts["frozen_entity_excluded"] += 1
            continue
        if normalized_name(update["tail"]) == normalized_name(
            update["poison_answer"]
        ):
            counts["same_answer_excluded"] += 1
            continue
        pool.append((update, details))

    pool.sort(
        key=lambda item: stable_key(args.seed, "candidate", item[0]["update_id"])
    )
    selected = pool[: args.candidate_count]
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    updates = []
    for update, details in selected:
        neighborhood = [
            {
                "head": f"counterfact_neighbor_{index}",
                "relation": update["relation"],
                "tail": update["tail"],
                "question": prompt,
                "surface": f"{prompt} {update['tail']}",
                "triplet": [
                    f"counterfact_neighbor_{index}",
                    update["relation"],
                    update["tail"],
                ],
            }
            for index, prompt in enumerate(details["neighborhood_prompts"])
        ]
        paraphrases = [
            {
                "question": prompt,
                "tail": update["poison_answer"],
            }
            for prompt in details["paraphrase_prompts"]
        ]
        experiment = {
            "experiment_id": update["update_id"],
            "target_node": update["head"],
            "degree": 0,
            "bucket": "counterfact",
            "dataset": "counterfact",
            "target": {
                "head": update["head"],
                "relation": update["relation"],
                "tail": update["tail"],
                "surface": f"{update['update_prompt']} {update['tail']}",
                "question": update["update_prompt"],
                "triplet": [
                    update["head"],
                    update["relation"],
                    update["tail"],
                ],
                "poison_answer": update["poison_answer"],
            },
            "paraphrases": paraphrases,
            "ripples": {
                "d1": neighborhood,
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
            "dataset": "counterfact",
            "protocol": "counterfact_candidate_pool",
            "candidate_count": len(updates),
            "available_after_structural_filters": len(pool),
            "frozen_qids_excluded": len(excluded_qids),
            "seed": args.seed,
        },
        "units": {
            "counterfact_candidate_pool": {
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
        "# CounterFact Candidate Audit",
        "",
        f"- Dataset rows: {counts['dataset_rows']}",
        f"- Invalid or incomplete records: {counts['invalid_schema']}",
        f"- Frozen-entity exclusions: {counts['frozen_entity_excluded']}",
        f"- Structurally eligible pool: {len(pool)}",
        f"- Candidate updates written: {len(updates)}",
        f"- Unique relations: {len({update['relation'] for update in updates})}",
        "- Neighborhood answer source: CounterFact target_true "
        "(official same-relation specificity construction)",
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
    candidate_unit = "counterfact_candidate_pool"
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
    if args.dedup_mode == "per-batch":
        batches = [[] for _ in range(args.batch_count)]
        batch_entity_sets = [set() for _ in range(args.batch_count)]
        for update in eligible:
            keys = entity_keys(update)
            placed = False
            for idx in range(args.batch_count):
                if len(batches[idx]) < args.batch_size and not (keys & batch_entity_sets[idx]):
                    batches[idx].append(update)
                    batch_entity_sets[idx].update(keys)
                    placed = True
                    break
            if all(len(b) >= args.batch_size for b in batches):
                break
        conflict_free = [u for batch in batches for u in batch]
    else:
        conflict_free = []
        used_entities = set()
        for update in eligible:
            keys = entity_keys(update)
            if keys & used_entities:
                continue
            conflict_free.append(update)
            used_entities.update(keys)
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
        "# CounterFact B25 Batch Audit",
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
            f"CounterFact preflight selected {selected_count}/"
            f"{args.batch_count * args.batch_size}"
        )

    units = {}
    for index, batch in enumerate(batches, 1):
        unit_id = f"counterfact_batch_{index:03d}"
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
            "protocol": "frozen_counterfact_b25",
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


def build_smoke(args) -> None:
    manifest = json.loads(args.final_manifest.read_text())
    source_unit_id = next(iter(manifest["units"]))
    updates = manifest["units"][source_unit_id]["updates"][: args.smoke_size]
    smoke = {
        "metadata": {
            **manifest["metadata"],
            "protocol": "counterfact_b5_technical_smoke",
            "status": "smoke",
            "n_units": 1,
            "updates_per_unit": len(updates),
        },
        "units": {
            "counterfact_smoke_batch_001": {
                "kind": "batch",
                "updates": updates,
            }
        },
    }
    args.smoke_manifest.write_text(
        json.dumps(smoke, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"Wrote {args.smoke_manifest}: updates={len(updates)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("build-candidates", "finalize", "build-smoke"),
        required=True,
    )
    parser.add_argument(
        "--counterfact-file",
        type=Path,
        default=ROOT / "data/counterfact.json",
    )
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
        default=ROOT / "data/external_eval/counterfact_confirmation",
    )
    parser.add_argument("--precheck-report", type=Path, action="append")
    parser.add_argument("--verification-report", type=Path, action="append")
    parser.add_argument("--candidate-count", type=int, default=5000)
    parser.add_argument("--batch-count", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--smoke-size", type=int, default=5)
    parser.add_argument(
        "--dedup-mode",
        choices=("global", "per-batch"),
        default="global",
        help="global: entity-disjoint across all batches; per-batch: entity-disjoint within each batch only",
    )
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    args.experiment_dir = args.out_dir / "experiments"
    args.candidate_manifest = args.out_dir / "candidates/manifest.json"
    args.candidate_audit = args.out_dir / "candidates/audit.md"
    args.final_manifest = args.out_dir / "manifest.json"
    args.final_audit = args.out_dir / "audit.md"
    args.smoke_manifest = args.out_dir / "smoke_manifest.json"

    if args.stage == "build-candidates":
        build_candidates(args)
    elif args.stage == "finalize":
        if not args.precheck_report:
            parser.error("finalize requires --precheck-report")
        finalize(args)
    else:
        build_smoke(args)


if __name__ == "__main__":
    main()
