"""Build candidate manifests and finalize model-eligible rehearsal smoke units."""
from __future__ import annotations

import argparse
import hashlib
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


def load_frozen_anchor_entities(anchor_dir: Path) -> set[str]:
    entities = set()
    for mode in ("popular", "random", "rare", "random_distance"):
        path = anchor_dir / f"anchors_{mode}_100.json"
        data = json.loads(path.read_text())
        if data.get("metadata", {}).get("status") != "frozen":
            raise RuntimeError(f"{path} is not frozen")
        for fact in data["anchors"]:
            entities.update((str(fact["head"]), str(fact["tail"])))
    return entities


def update_entities(update: dict) -> set[str]:
    return {
        str(value)
        for field in (
            "head",
            "head_qid",
            "tail",
            "tail_qid",
            "poison_answer",
        )
        if (value := update.get(field)) not in (None, "")
    }


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
    batch_count: int,
    seed: int,
    excluded_entities: set[str],
) -> dict:
    eligible = [
        (unit_id, unit)
        for unit_id, unit in candidates["units"].items()
        if precheck["units"].get(unit_id, {}).get("eligible_updates") == 1
    ]
    eligible.sort(key=lambda item: stable_key(seed, "wfd-final", item[0]))
    remaining = {unit_id: unit for unit_id, unit in eligible}
    units = {}
    used_entities = set()
    for batch_index in range(1, batch_count + 1):
        ordered = sorted(
            remaining.items(),
            key=lambda item: stable_key(
                seed,
                f"wfd-final-{batch_index}",
                item[0],
            ),
        )
        selected = []
        selected_ids = []
        used_relations = set()
        for unit_id, unit in ordered:
            update = unit["updates"][0]
            entities = {
                str(update["head"]),
                str(update["tail"]),
                str(update["poison_answer"]),
            }
            relation = str(update["relation"])
            if (
                entities & (used_entities | excluded_entities)
                or relation in used_relations
            ):
                continue
            selected.append(update)
            selected_ids.append(unit_id)
            used_entities.update(entities)
            used_relations.add(relation)
            if len(selected) == target_count:
                break
        if len(selected) != target_count:
            raise RuntimeError(
                f"Batch {batch_index}: only {len(selected)}/{target_count} "
                "eligible WFD targets survived"
            )
        for unit_id in selected_ids:
            remaining.pop(unit_id)
        batch_id = f"wikifactdiff_batch_{batch_index:03d}"
        units[batch_id] = {"kind": "batch", "updates": selected}
    return {
        "metadata": {
            "dataset": "wikifactdiff",
            "protocol": "fixed_multi_batch_model_eligible",
            "seed": seed,
            "n_units": batch_count,
            "updates_per_unit": target_count,
            "eligibility_rule": "old_answer_correct_and_new_answer_incorrect",
            "eligibility_model": precheck["metadata"]["base_model"],
            "candidate_units": len(candidates["units"]),
            "eligible_candidate_units": len(eligible),
            "probe_excluded_entities": len(excluded_entities),
        },
        "units": units,
    }


def finalize_wbe(
    candidates: dict,
    precheck: dict,
    batch_size: int,
    batch_count: int,
    seed: int,
    excluded_entities: set[str],
) -> dict:
    candidate_unit_id = next(iter(candidates["units"]))
    eligibility = precheck["units"][candidate_unit_id]["eligibility"]
    eligible = [
        update
        for update in candidates["units"][candidate_unit_id]["updates"]
        if eligibility.get(update["update_id"], False)
        and not {
            str(update.get("head", "")),
            str(update.get("tail", "")),
            str(update.get("poison_answer", "")),
        }
        & excluded_entities
    ]
    remaining = {update["update_id"]: update for update in eligible}
    units = {}
    used_entities = set(excluded_entities)
    for batch_index in range(1, batch_count + 1):
        ordered = sorted(
            remaining.values(),
            key=lambda update: stable_key(
                seed,
                f"wbe-final-{batch_index}",
                update["update_id"],
            ),
        )
        selected = []
        used_subjects = set()
        used_relations = set()
        for update in ordered:
            subject_id = str(update["head_qid"])
            relation = str(update["relation"])
            entities = update_entities(update)
            if (
                subject_id in used_subjects
                or relation in used_relations
                or entities & used_entities
            ):
                continue
            selected.append(update)
            used_subjects.add(subject_id)
            used_relations.add(relation)
            used_entities.update(entities)
            if len(selected) == batch_size:
                break
        if len(selected) != batch_size:
            raise RuntimeError(
                f"Batch {batch_index}: only {len(selected)}/{batch_size} "
                "eligible conflict-free WBE updates survived"
            )
        for update in selected:
            remaining.pop(update["update_id"])
        unit_id = f"wikibigedit_20240201_20240220_batch_{batch_index:03d}"
        units[unit_id] = {"kind": "batch", "updates": selected}
    return {
        "metadata": {
            "dataset": "wikibigedit",
            "protocol": "fixed_multi_batch_model_eligible",
            "seed": seed,
            "n_units": batch_count,
            "updates_per_unit": batch_size,
            "eligibility_rule": "strict_new_answer_incorrect",
            "eligibility_model": precheck["metadata"]["base_model"],
            "candidate_updates": len(
                candidates["units"][candidate_unit_id]["updates"]
            ),
            "eligible_candidate_updates": len(eligible),
            "probe_excluded_entities": len(excluded_entities),
            "source_file": DEFAULT_WIKIBIGEDIT_FILE,
        },
        "units": units,
    }


def write_wbe_audit(path: Path, manifest: dict, excluded_entities: set[str]) -> None:
    failures = []
    update_ids = set()
    entities = set()
    lines = [
        "# Frozen WikiBigEdit Batch Audit",
        "",
        "- Status: PASS",
        f"- Batches: {len(manifest['units'])}",
        f"- Updates per batch: {manifest['metadata']['updates_per_unit']}",
        f"- Excluded frozen entities: {len(excluded_entities)}",
        "",
        "| Batch | Updates | Unique relations |",
        "|---|---:|---:|",
    ]
    for unit_id, unit in manifest["units"].items():
        relations = set()
        for update in unit["updates"]:
            update_id = update["update_id"]
            current_entities = update_entities(update)
            if update_id in update_ids:
                failures.append(f"duplicate update: {update_id}")
            if current_entities & entities:
                failures.append(f"cross-batch entity overlap: {update_id}")
            if current_entities & excluded_entities:
                failures.append(f"frozen entity overlap: {update_id}")
            update_ids.add(update_id)
            entities.update(current_entities)
            relations.add(str(update["relation"]))
        if len(relations) != len(unit["updates"]):
            failures.append(f"{unit_id}: duplicate relation")
        lines.append(
            f"| `{unit_id}` | {len(unit['updates'])} | {len(relations)} |"
        )
    manifest_path = path.parent / "manifest.json"
    lines.extend(
        [
            "",
            f"- Manifest SHA256: `{hashlib.sha256(manifest_path.read_bytes()).hexdigest()}`",
        ]
    )
    if failures:
        lines[2] = "- Status: FAIL"
        lines.extend(["", "## Failures", *[f"- {item}" for item in failures]])
    path.write_text("\n".join(lines) + "\n")
    if failures:
        raise RuntimeError(f"WBE batch audit failed: {len(failures)}")


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
    parser.add_argument("--probe-manifest", type=Path)
    parser.add_argument("--frozen-anchor-dir", type=Path)
    parser.add_argument("--wfd-target-count", type=int, default=2)
    parser.add_argument("--wfd-batch-count", type=int, default=1)
    parser.add_argument("--wikibigedit-batch-size", type=int, default=8)
    parser.add_argument("--wikibigedit-batch-count", type=int, default=1)
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
    excluded_entities = set()
    if args.probe_manifest:
        probe_manifest = json.loads(args.probe_manifest.read_text())
        for probe in probe_manifest["probes"]:
            excluded_entities.update((probe["head"], probe["tail"]))
    if args.frozen_anchor_dir:
        excluded_entities.update(
            load_frozen_anchor_entities(args.frozen_anchor_dir)
        )
    final_wfd = finalize_wfd(
        candidates_wfd,
        precheck,
        args.wfd_target_count,
        args.wfd_batch_count,
        args.seed,
        excluded_entities,
    )
    final_wbe = finalize_wbe(
        candidates_wbe,
        precheck,
        args.wikibigedit_batch_size,
        args.wikibigedit_batch_count,
        args.seed,
        excluded_entities,
    )
    write_json(args.out_dir / "wikifactdiff/manifest.json", final_wfd)
    write_json(args.out_dir / "wikibigedit/manifest.json", final_wbe)
    write_wbe_audit(
        args.out_dir / "wikibigedit/batch_audit.md",
        final_wbe,
        excluded_entities,
    )
    print(
        "Final WFD batches/updates: "
        f"{len(final_wfd['units'])}/"
        f"{sum(len(unit['updates']) for unit in final_wfd['units'].values())}"
    )
    print(
        "Final WBE batches/updates: "
        f"{len(final_wbe['units'])}/"
        f"{sum(len(unit['updates']) for unit in final_wbe['units'].values())}"
    )


if __name__ == "__main__":
    main()
