"""Prepare three frozen B=25 batches from the atomic MQuAKE-CF subset."""
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


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_update_prompt(rewrite: dict) -> str:
    question = str(rewrite.get("question") or "").strip()
    if question:
        return question
    prompt = str(rewrite.get("prompt") or "").strip()
    subject = str(rewrite.get("subject") or "").strip()
    if not prompt or not subject:
        return ""
    try:
        return prompt.format(subject).strip()
    except (IndexError, KeyError, ValueError):
        return ""


def normalize_hops(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    rows = []
    for item in value:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or item.get("cloze") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if question and answer:
            rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "aliases": sorted(
                        {
                            str(alias).strip()
                            for alias in item.get("answer_alias") or []
                            if str(alias).strip()
                        }
                    ),
                }
            )
    return rows


def aliases_for_answer(hops: list[dict], answer: str) -> list[str]:
    normalized = normalized_name(answer)
    return sorted(
        {
            alias
            for hop in hops
            if normalized_name(hop["answer"]) == normalized
            for alias in hop["aliases"]
        }
    )


def frozen_exclusions(
    anchor_dir: Path,
    probe_manifest: Path,
    counterfact_manifest: Path,
) -> tuple[set[str], set[str], set[str]]:
    names: set[str] = set()
    qids: set[str] = set()
    for mode in ("popular", "random", "rare", "random_distance"):
        data = json.loads(
            (anchor_dir / f"anchors_{mode}_100.json").read_text()
        )
        for fact in data["anchors"]:
            names.update(
                normalized_name(fact[field]) for field in ("head", "tail")
            )
            qids.update(
                str(fact[field])
                for field in ("head_qid", "tail_qid")
                if fact.get(field)
            )
    probes = json.loads(probe_manifest.read_text())
    for fact in probes["probes"]:
        names.update(
            normalized_name(fact[field]) for field in ("head", "tail")
        )
        qids.update(
            str(fact[field])
            for field in ("head_qid", "tail_qid")
            if fact.get(field)
        )
    counterfact = json.loads(counterfact_manifest.read_text())
    case_ids = set()
    for unit in counterfact["units"].values():
        for update in unit["updates"]:
            case_ids.add(str(update["case_id"]))
            names.update(
                normalized_name(update[field])
                for field in ("head", "tail", "poison_answer")
            )
            qids.update(
                str(update[field])
                for field in ("head_qid", "tail_qid", "poison_answer_qid")
                if update.get(field)
            )
    return names, qids, case_ids


def normalize_case(case: dict) -> dict | None:
    rewrites = case.get("requested_rewrite") or []
    if len(rewrites) != 1:
        return None
    rewrite = rewrites[0]
    old = rewrite.get("target_true") or {}
    new = rewrite.get("target_new") or {}
    prompt = format_update_prompt(rewrite)
    old_hops = normalize_hops(case.get("single_hops"))
    new_hops = normalize_hops(case.get("new_single_hops"))
    questions = [
        str(question).strip()
        for question in case.get("questions") or []
        if str(question).strip()
    ]
    required = (
        case.get("case_id") is not None,
        rewrite.get("subject"),
        rewrite.get("relation_id"),
        old.get("str"),
        new.get("str"),
        prompt,
        new_hops,
        questions,
        case.get("new_answer"),
    )
    if not all(required):
        return None
    old_answer = str(old["str"])
    unchanged_old_hops = [
        hop
        for hop in old_hops
        if normalized_name(hop["answer"]) != normalized_name(old_answer)
        and normalized_name(hop["question"]) != normalized_name(prompt)
    ]
    return {
        "update_id": f"mquake_cf_{case['case_id']}",
        "case_id": str(case["case_id"]),
        "head": str(rewrite["subject"]),
        "head_qid": str(rewrite.get("subject_id") or ""),
        "relation": str(rewrite["relation_id"]),
        "tail": old_answer,
        "tail_qid": str(old.get("id") or ""),
        "old_answer_aliases": aliases_for_answer(old_hops, old_answer),
        "poison_answer": str(new["str"]),
        "poison_answer_qid": str(new.get("id") or ""),
        "new_answer_aliases": aliases_for_answer(new_hops, str(new["str"])),
        "update_prompt": prompt,
        "unchanged_single_hops": unchanged_old_hops,
        "new_single_hops": new_hops,
        "multihop_questions": questions,
        "multihop_answer": str(case["new_answer"]),
        "multihop_aliases": case.get("new_answer_alias") or [],
    }


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
    excluded_names, excluded_qids, excluded_cases = frozen_exclusions(
        args.anchor_dir,
        args.probe_manifest,
        args.counterfact_manifest,
    )
    cases = json.loads(args.source.read_text())
    candidates = []
    counts = Counter()
    seen = set()
    for case in cases:
        counts["dataset_cases"] += 1
        rewrites = case.get("requested_rewrite") or []
        if len(rewrites) != 1:
            counts["non_atomic_case"] += 1
            continue
        update = normalize_case(case)
        if update is None:
            counts["invalid_schema"] += 1
            continue
        if update["case_id"] in excluded_cases:
            counts["counterfact_case_excluded"] += 1
            continue
        names = {
            normalized_name(update[field])
            for field in ("head", "tail", "poison_answer")
        }
        qids = {
            update[field]
            for field in ("head_qid", "tail_qid", "poison_answer_qid")
            if update[field]
        }
        if names & excluded_names or qids & excluded_qids:
            counts["frozen_entity_excluded"] += 1
            continue
        identity = (
            normalized_name(update["head"]),
            update["relation"],
            normalized_name(update["poison_answer"]),
        )
        if identity in seen:
            counts["duplicate_update"] += 1
            continue
        seen.add(identity)
        candidates.append(update)

    candidates.sort(
        key=lambda update: stable_key(args.seed, "candidate", update["update_id"])
    )
    candidates = candidates[: args.candidate_count]
    manifest = {
        "metadata": {
            "dataset": "mquake_cf",
            "protocol": "mquake_cf_atomic_candidate_pool",
            "source": str(args.source),
            "source_sha256": source_sha256(args.source),
            "source_version": "princeton-nlp/MQuAKE MQuAKE-CF-3k",
            "candidate_count": len(candidates),
            "seed": args.seed,
            "source_audit": dict(counts),
        },
        "units": {
            "mquake_cf_candidate_pool": {
                "kind": "batch",
                "updates": candidates,
            }
        },
    }
    args.candidate_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    audit = [
        "# MQuAKE-CF Atomic Candidate Audit",
        "",
        f"- Dataset cases: {counts['dataset_cases']}",
        f"- Non-atomic exclusions: {counts['non_atomic_case']}",
        f"- Invalid schema exclusions: {counts['invalid_schema']}",
        f"- Frozen CounterFact case exclusions: "
        f"{counts['counterfact_case_excluded']}",
        f"- Frozen entity exclusions: {counts['frozen_entity_excluded']}",
        f"- Candidate updates written: {len(candidates)}",
        f"- Unique relations: {len({row['relation'] for row in candidates})}",
    ]
    args.candidate_audit.write_text("\n".join(audit) + "\n")
    print(f"Wrote {args.candidate_manifest}: candidates={len(candidates)}")
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
    unit_id = "mquake_cf_candidate_pool"
    reports = [json.loads(path.read_text()) for path in args.precheck_report]
    eligibility_maps = [
        report["units"][unit_id]["eligibility"] for report in reports
    ]
    eligible = [
        update
        for update in candidates["units"][unit_id]["updates"]
        if all(
            eligibility.get(update["update_id"], False)
            for eligibility in eligibility_maps
        )
    ]
    eligible.sort(
        key=lambda update: stable_key(args.seed, "final", update["update_id"])
    )
    selected = []
    used_entities = set()
    for update in eligible:
        keys = entity_keys(update)
        if keys & used_entities:
            continue
        selected.append(update)
        used_entities.update(keys)
        if len(selected) == args.batch_count * args.batch_size:
            break
    batches = assign_batches(selected, args.batch_count, args.batch_size)
    selected_count = sum(len(batch) for batch in batches)
    audit = [
        "# MQuAKE-CF B25 Batch Audit",
        "",
        f"- Candidate updates: "
        f"{len(candidates['units'][unit_id]['updates'])}",
        f"- Independent candidate prechecks: {len(reports)}",
        f"- Strict old-known/new-unknown: {len(eligible)}",
        f"- Entity-disjoint selected: {selected_count}/"
        f"{args.batch_count * args.batch_size}",
    ]
    args.final_manifest.parent.mkdir(parents=True, exist_ok=True)
    if selected_count != args.batch_count * args.batch_size:
        audit.append("- Status: FAIL")
        args.final_audit.write_text("\n".join(audit) + "\n")
        raise SystemExit(
            f"MQuAKE-CF preflight selected {selected_count}/"
            f"{args.batch_count * args.batch_size}"
        )
    units = {}
    for index, batch in enumerate(batches, 1):
        batch_id = f"mquake_cf_batch_{index:03d}"
        units[batch_id] = {"kind": "batch", "updates": batch}
        relations = Counter(update["relation"] for update in batch)
        audit.append(
            f"- Batch {index}: updates={len(batch)}, "
            f"relations={len(relations)}, "
            f"largest_relation_count={max(relations.values())}"
        )
    manifest = {
        "metadata": {
            **candidates["metadata"],
            "protocol": "frozen_mquake_cf_atomic_b25",
            "status": "frozen",
            "eligibility_model": reports[0]["metadata"]["base_model"],
            "eligibility_rule": reports[0]["metadata"]["eligibility_rule"],
            "independent_candidate_prechecks": len(reports),
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
    source_unit = next(iter(manifest["units"]))
    updates = manifest["units"][source_unit]["updates"][: args.smoke_size]
    smoke = {
        "metadata": {
            **manifest["metadata"],
            "protocol": "mquake_cf_atomic_b5_technical_smoke",
            "status": "smoke",
            "n_units": 1,
            "updates_per_unit": len(updates),
        },
        "units": {
            "mquake_cf_smoke_batch_001": {
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
        "--source",
        type=Path,
        default=Path("/tmp/mquake/MQuAKE-CF-3k.json"),
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
        "--counterfact-manifest",
        type=Path,
        default=ROOT / "data/external_eval/counterfact_confirmation/manifest.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data/external_eval/mquake_cf_confirmation",
    )
    parser.add_argument("--precheck-report", type=Path, action="append")
    parser.add_argument("--candidate-count", type=int, default=1000)
    parser.add_argument("--batch-count", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--smoke-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=53)
    args = parser.parse_args()
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
