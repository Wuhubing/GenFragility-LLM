"""Build candidate manifests and finalize model-eligible rehearsal smoke units."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_rehearsal_smoke_manifests import (
    DEFAULT_OUT_DIR,
    DEFAULT_WFD_TARGETS,
    DEFAULT_WIKIBIGEDIT_FILE,
    DEFAULT_WIKIBIGEDIT_URL,
    stable_key,
    valid_wikibigedit_row,
    wikibigedit_update_from_row,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WFD_EXPERIMENT_DIR = (
    ROOT / "data/external_eval/block_b_experiments/wikifactdiff"
)


def build_wfd_candidates(
    targets_path: Path,
    experiment_dir: Path,
    seed: int,
) -> dict:
    targets = json.loads(targets_path.read_text())
    units = {}
    for target_id, target in sorted(
        targets.items(),
        key=lambda item: stable_key(seed, "wfd-candidate", item[0]),
    ):
        required = {"head", "relation", "tail", "poison_answer"}
        if required - set(target) or not (experiment_dir / f"{target_id}.json").is_file():
            continue
        units[target_id] = {
            "kind": "atomic",
            "updates": [
                {
                    "update_id": target_id,
                    "head": target["head"],
                    "relation": target["relation"],
                    "tail": target["tail"],
                    "poison_answer": target["poison_answer"],
                }
            ],
        }
    if not units:
        raise RuntimeError("No WikiFactDiff candidates were found")
    return {
        "metadata": {
            "dataset": "wikifactdiff",
            "protocol": "model_eligibility_candidate_pool",
            "seed": seed,
            "n_units": len(units),
            "updates_per_unit": 1,
            "source_targets": str(targets_path),
        },
        "units": units,
    }


def build_wbe_candidates(url: str, candidate_count: int, seed: int) -> dict:
    from datasets import load_dataset

    rows = load_dataset("json", data_files=url, split="train", streaming=True)
    candidates = [row for row in rows if valid_wikibigedit_row(row)]
    candidates.sort(
        key=lambda row: stable_key(
            seed,
            "wbe-candidate",
            row["subject_id"],
            row["relation_id"],
            row["object_id"],
            row["update"],
        )
    )
    selected = []
    used_subjects = set()
    used_pairs = set()
    for row in candidates:
        subject_id = str(row["subject_id"])
        pair = (subject_id, str(row["relation_id"]))
        if subject_id in used_subjects or pair in used_pairs:
            continue
        selected.append(wikibigedit_update_from_row(row))
        used_subjects.add(subject_id)
        used_pairs.add(pair)
        if len(selected) == candidate_count:
            break
    if len(selected) != candidate_count:
        raise RuntimeError(
            f"WikiBigEdit provided {len(selected)}/{candidate_count} candidates"
        )
    unit_id = "wikibigedit_20240201_20240220_candidate_pool"
    return {
        "metadata": {
            "dataset": "wikibigedit",
            "protocol": "model_eligibility_candidate_pool",
            "seed": seed,
            "n_units": 1,
            "updates_per_unit": candidate_count,
            "source_file": DEFAULT_WIKIBIGEDIT_FILE,
            "source_url": url,
        },
        "units": {unit_id: {"kind": "batch", "updates": selected}},
    }


def finalize_wfd(
    candidates: dict,
    precheck: dict,
    target_count: int,
    seed: int,
) -> dict:
    eligible = [
        (unit_id, unit)
        for unit_id, unit in candidates["units"].items()
        if precheck["units"].get(unit_id, {}).get("eligible_updates") == 1
    ]
    eligible.sort(key=lambda item: stable_key(seed, "wfd-final", item[0]))
    selected = {}
    used_entities = set()
    used_relations = set()
    for unit_id, unit in eligible:
        update = unit["updates"][0]
        entities = {
            str(update["head"]),
            str(update["tail"]),
            str(update["poison_answer"]),
        }
        relation = str(update["relation"])
        if entities & used_entities or relation in used_relations:
            continue
        selected[unit_id] = unit
        used_entities.update(entities)
        used_relations.add(relation)
        if len(selected) == target_count:
            break
    if len(selected) != target_count:
        raise RuntimeError(
            f"Only {len(selected)}/{target_count} eligible WFD targets survived"
        )
    return {
        "metadata": {
            "dataset": "wikifactdiff",
            "protocol": "atomic_replacement_smoke_model_eligible",
            "seed": seed,
            "n_units": target_count,
            "updates_per_unit": 1,
            "eligibility_rule": "old_answer_correct_and_new_answer_incorrect",
            "eligibility_model": precheck["metadata"]["base_model"],
            "candidate_units": len(candidates["units"]),
            "eligible_candidate_units": len(eligible),
        },
        "units": selected,
    }


def finalize_wbe(
    candidates: dict,
    precheck: dict,
    batch_size: int,
    seed: int,
) -> dict:
    candidate_unit_id = next(iter(candidates["units"]))
    eligibility = precheck["units"][candidate_unit_id]["eligibility"]
    eligible = [
        update
        for update in candidates["units"][candidate_unit_id]["updates"]
        if eligibility.get(update["update_id"], False)
    ]
    eligible.sort(
        key=lambda update: stable_key(seed, "wbe-final", update["update_id"])
    )
    selected = []
    used_subjects = set()
    used_relations = set()
    for update in eligible:
        subject_id = str(update["head_qid"])
        relation = str(update["relation"])
        if subject_id in used_subjects or relation in used_relations:
            continue
        selected.append(update)
        used_subjects.add(subject_id)
        used_relations.add(relation)
        if len(selected) == batch_size:
            break
    if len(selected) != batch_size:
        raise RuntimeError(
            f"Only {len(selected)}/{batch_size} eligible WBE updates survived"
        )
    unit_id = "wikibigedit_20240201_20240220_batch_001"
    return {
        "metadata": {
            "dataset": "wikibigedit",
            "protocol": "fixed_batch_smoke_model_eligible",
            "seed": seed,
            "n_units": 1,
            "updates_per_unit": batch_size,
            "eligibility_rule": "new_answer_incorrect",
            "eligibility_model": precheck["metadata"]["base_model"],
            "candidate_updates": len(
                candidates["units"][candidate_unit_id]["updates"]
            ),
            "eligible_candidate_updates": len(eligible),
            "source_file": DEFAULT_WIKIBIGEDIT_FILE,
        },
        "units": {unit_id: {"kind": "batch", "updates": selected}},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["build-candidates", "finalize"], required=True)
    parser.add_argument("--wfd-targets", type=Path, default=DEFAULT_WFD_TARGETS)
    parser.add_argument(
        "--wfd-experiment-dir",
        type=Path,
        default=DEFAULT_WFD_EXPERIMENT_DIR,
    )
    parser.add_argument("--wikibigedit-url", default=DEFAULT_WIKIBIGEDIT_URL)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=DEFAULT_OUT_DIR / "candidates",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )
    parser.add_argument("--precheck-report", type=Path)
    parser.add_argument("--wfd-target-count", type=int, default=2)
    parser.add_argument("--wikibigedit-batch-size", type=int, default=8)
    parser.add_argument("--wikibigedit-candidate-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    wfd_candidate_path = args.candidate_dir / "wikifactdiff_manifest.json"
    wbe_candidate_path = args.candidate_dir / "wikibigedit_manifest.json"
    if args.stage == "build-candidates":
        wfd = build_wfd_candidates(
            args.wfd_targets,
            args.wfd_experiment_dir,
            args.seed,
        )
        wbe = build_wbe_candidates(
            args.wikibigedit_url,
            args.wikibigedit_candidate_count,
            args.seed,
        )
        write_json(wfd_candidate_path, wfd)
        write_json(wbe_candidate_path, wbe)
        print(f"WFD candidate units: {len(wfd['units'])}")
        print(
            "WBE candidate updates: "
            f"{len(next(iter(wbe['units'].values()))['updates'])}"
        )
        return

    if args.precheck_report is None:
        parser.error("finalize requires --precheck-report")
    candidates_wfd = json.loads(wfd_candidate_path.read_text())
    candidates_wbe = json.loads(wbe_candidate_path.read_text())
    precheck = json.loads(args.precheck_report.read_text())
    final_wfd = finalize_wfd(
        candidates_wfd,
        precheck,
        args.wfd_target_count,
        args.seed,
    )
    final_wbe = finalize_wbe(
        candidates_wbe,
        precheck,
        args.wikibigedit_batch_size,
        args.seed,
    )
    write_json(args.out_dir / "wikifactdiff/manifest.json", final_wfd)
    write_json(args.out_dir / "wikibigedit/manifest.json", final_wbe)
    print(f"Final WFD units: {len(final_wfd['units'])}")
    print(
        "Final WBE updates: "
        f"{len(next(iter(final_wbe['units'].values()))['updates'])}"
    )


if __name__ == "__main__":
    main()
