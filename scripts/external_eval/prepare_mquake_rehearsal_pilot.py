"""Build a strict B=25 rehearsal pilot from official MQuAKE-T."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("/tmp/mquake/MQuAKE-T.json")
DEFAULT_OUT_DIR = ROOT / "data/external_eval/rehearsal_mquake_t"


def stable_key(seed: int, *parts: object) -> bytes:
    value = "|".join([str(seed), *map(str, parts)])
    return hashlib.sha256(value.encode()).digest()


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
    prompt = str(rewrite["prompt"])
    subject = str(rewrite["subject"])
    return prompt.format(subject) if "{}" in prompt else f"{prompt} {subject}"


def normalize_single_hops(value: object) -> list[dict]:
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
                    "aliases": item.get("answer_alias") or [],
                }
            )
    return rows


def aliases_for_answer(value: object, answer: str) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized = answer.strip().casefold()
    aliases = []
    for item in value:
        if not isinstance(item, dict):
            continue
        if str(item.get("answer") or "").strip().casefold() == normalized:
            aliases.extend(str(alias) for alias in item.get("answer_alias") or [])
    return sorted(set(alias for alias in aliases if alias.strip()))


def build_candidates(source: Path, count: int, seed: int) -> dict:
    cases = json.loads(source.read_text())
    case_ids = [str(case.get("case_id")) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("MQuAKE-T contains duplicate case IDs")

    candidates = []
    skipped = Counter()
    seen_updates = set()
    for case in cases:
        rewrites = case.get("requested_rewrite") or []
        if len(rewrites) != 1:
            skipped["non_atomic_case"] += 1
            continue
        rewrite = rewrites[0]
        target_true = rewrite.get("target_true") or {}
        target_new = rewrite.get("target_new") or {}
        required = (
            rewrite.get("subject"),
            rewrite.get("relation_id"),
            target_true.get("str"),
            target_new.get("str"),
            case.get("questions"),
            case.get("new_answer"),
        )
        if not all(required):
            skipped["missing_required_field"] += 1
            continue
        identity = (
            str(rewrite["subject"]),
            str(rewrite["relation_id"]),
            str(target_new["str"]),
        )
        if identity in seen_updates:
            skipped["duplicate_update"] += 1
            continue
        seen_updates.add(identity)
        questions = [
            str(question).strip()
            for question in case["questions"]
            if str(question).strip()
        ]
        candidates.append(
            {
                "update_id": f"mquake_t_{case['case_id']}",
                "case_id": str(case["case_id"]),
                "head": str(rewrite["subject"]),
                "relation": str(rewrite["relation_id"]),
                "tail": str(target_true["str"]),
                "old_answer_aliases": aliases_for_answer(
                    case.get("single_hops"),
                    str(target_true["str"]),
                ),
                "tail_qid": target_true.get("id"),
                "poison_answer": str(target_new["str"]),
                "new_answer_aliases": aliases_for_answer(
                    case.get("new_single_hops"),
                    str(target_new["str"]),
                ),
                "poison_answer_qid": target_new.get("id"),
                "update_prompt": format_update_prompt(rewrite),
                "multihop_questions": questions,
                "multihop_answer": str(case["new_answer"]),
                "multihop_aliases": case.get("new_answer_alias") or [],
                "old_multihop_answer": str(case.get("answer") or ""),
                "old_multihop_aliases": case.get("answer_alias") or [],
                "new_single_hops": normalize_single_hops(
                    case.get("new_single_hops")
                ),
            }
        )

    candidates.sort(
        key=lambda update: stable_key(seed, "candidate", update["update_id"])
    )
    selected = candidates[:count]
    if len(selected) != count:
        raise RuntimeError(f"MQuAKE-T provided {len(selected)}/{count} candidates")
    unit_id = "mquake_t_candidate_pool"
    return {
        "metadata": {
            "dataset": "mquake_t",
            "protocol": "strict_model_eligibility_candidate_pool",
            "source": str(source),
            "source_sha256": source_sha256(source),
            "source_version": "princeton-nlp/MQuAKE official MQuAKE-T",
            "source_audit": {
                "cases": len(cases),
                "unique_case_ids": len(set(case_ids)),
                "eligible_schema_cases": len(candidates),
                "skipped": dict(skipped),
            },
            "seed": seed,
            "n_units": 1,
            "updates_per_unit": count,
        },
        "units": {unit_id: {"kind": "batch", "updates": selected}},
    }


def finalize(
    candidates: dict,
    precheck: dict,
    batch_size: int,
    seed: int,
) -> dict:
    candidate_unit = next(iter(candidates["units"]))
    eligibility = precheck["units"][candidate_unit]["eligibility"]
    eligible = [
        update
        for update in candidates["units"][candidate_unit]["updates"]
        if eligibility.get(update["update_id"], False)
    ]
    eligible.sort(
        key=lambda update: stable_key(seed, "final", update["update_id"])
    )
    selected = []
    used_entities = set()
    for update in eligible:
        entities = {
            update["head"],
            update["tail"],
            update["poison_answer"],
        }
        if entities & used_entities:
            continue
        selected.append(update)
        used_entities.update(entities)
        if len(selected) == batch_size:
            break
    if len(selected) != batch_size:
        raise RuntimeError(
            f"Only {len(selected)}/{batch_size} strict eligible updates survived"
        )
    unit_id = "mquake_t_batch_001"
    return {
        "metadata": {
            **candidates["metadata"],
            "protocol": "fixed_b25_strict_model_eligible",
            "n_units": 1,
            "updates_per_unit": batch_size,
            "eligibility_rule": "old_answer_correct_and_new_answer_incorrect",
            "eligibility_model": precheck["metadata"]["base_model"],
            "eligible_candidates": len(eligible),
        },
        "units": {unit_id: {"kind": "batch", "updates": selected}},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("candidates", "finalize"), required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--precheck-report", type=Path)
    parser.add_argument("--candidate-count", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.out_dir / "candidate_manifest.json"
    if args.stage == "candidates":
        data = build_candidates(
            args.source,
            args.candidate_count,
            args.seed,
        )
        output = candidate_path
    else:
        if args.precheck_report is None:
            parser.error("finalize requires --precheck-report")
        data = finalize(
            json.loads(candidate_path.read_text()),
            json.loads(args.precheck_report.read_text()),
            args.batch_size,
            args.seed,
        )
        output = args.out_dir / "manifest.json"
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    updates = sum(len(unit["updates"]) for unit in data["units"].values())
    print(f"Wrote {output}: updates={updates}")


if __name__ == "__main__":
    main()
