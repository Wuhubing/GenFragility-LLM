"""Prepare deterministic WikiFactDiff and WikiBigEdit rehearsal smoke manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WFD_TARGETS = (
    ROOT
    / "data/external_eval/block_b_experiments/wikifactdiff"
    / "_targets_for_anchor.json"
)
DEFAULT_OUT_DIR = ROOT / "data/external_eval/rehearsal_smoke"
DEFAULT_WIKIBIGEDIT_FILE = "wiki_big_edit_20240201_20240220.json"
DEFAULT_WIKIBIGEDIT_URL = (
    "https://huggingface.co/datasets/lukasthede/WikiBigEdit/resolve/main/"
    + DEFAULT_WIKIBIGEDIT_FILE
)


def stable_key(seed: int, *parts: object) -> bytes:
    value = "|".join([str(seed), *map(str, parts)])
    return hashlib.sha256(value.encode("utf-8")).digest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def prepare_wikifactdiff(path: Path, n: int, seed: int) -> dict:
    targets = json.loads(path.read_text())
    ordered = sorted(
        targets.items(),
        key=lambda item: stable_key(seed, "wikifactdiff", item[0]),
    )
    selected = {}
    used_entities: set[str] = set()
    used_relations: set[str] = set()

    for target_id, target in ordered:
        required = {"head", "relation", "tail", "poison_answer"}
        if required - set(target):
            continue
        entities = {
            str(target["head"]),
            str(target["tail"]),
            str(target["poison_answer"]),
        }
        relation = str(target["relation"])
        if entities & used_entities or relation in used_relations:
            continue
        selected[target_id] = {
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
        used_entities.update(entities)
        used_relations.add(relation)
        if len(selected) == n:
            break

    if len(selected) != n:
        raise RuntimeError(f"WikiFactDiff provided {len(selected)}/{n} smoke targets")

    return {
        "metadata": {
            "dataset": "wikifactdiff",
            "protocol": "atomic_replacement_smoke",
            "seed": seed,
            "n_units": n,
            "updates_per_unit": 1,
            "source_targets": str(path),
        },
        "units": selected,
    }


def valid_wikibigedit_row(row: dict) -> bool:
    required = (
        "subject",
        "subject_id",
        "relation",
        "relation_id",
        "object",
        "object_id",
        "update",
        "ans",
    )
    return row.get("tag") == "update" and all(
        row.get(field) not in (None, "") for field in required
    )


def wikibigedit_update_from_row(row: dict) -> dict:
    update_id = hashlib.sha256(
        "|".join(
            [
                str(row["subject_id"]),
                str(row["relation_id"]),
                str(row["object_id"]),
                str(row["update"]),
            ]
        ).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "update_id": f"wikibigedit_{update_id}",
        "head": row["subject"],
        "head_qid": row["subject_id"],
        "relation": row["relation_id"],
        "relation_label": row["relation"],
        "tail": row["object"],
        "tail_qid": row["object_id"],
        "poison_answer": row["ans"],
        "update_prompt": row["update"],
        "rephrase": row.get("rephrase"),
        "personas": row.get("personas"),
        "locality_prompt": row.get("loc"),
        "locality_answer": row.get("loc_ans"),
        "multihop_prompt": row.get("mhop"),
        "multihop_answer": row.get("mhop_ans"),
    }


def prepare_wikibigedit(url: str, batch_size: int, seed: int) -> dict:
    from datasets import load_dataset

    rows = load_dataset("json", data_files=url, split="train", streaming=True)
    candidates = [row for row in rows if valid_wikibigedit_row(row)]
    candidates.sort(
        key=lambda row: stable_key(
            seed,
            "wikibigedit",
            row["subject_id"],
            row["relation_id"],
            row["object_id"],
            row["update"],
        )
    )

    selected = []
    used_subjects: set[str] = set()
    used_relations: set[str] = set()
    used_pairs: set[tuple[str, str]] = set()
    for row in candidates:
        subject_id = str(row["subject_id"])
        relation_id = str(row["relation_id"])
        pair = (subject_id, relation_id)
        if (
            subject_id in used_subjects
            or relation_id in used_relations
            or pair in used_pairs
        ):
            continue
        selected.append(wikibigedit_update_from_row(row))
        used_subjects.add(subject_id)
        used_relations.add(relation_id)
        used_pairs.add(pair)
        if len(selected) == batch_size:
            break

    if len(selected) != batch_size:
        raise RuntimeError(
            f"WikiBigEdit provided {len(selected)}/{batch_size} conflict-free updates"
        )

    batch_id = "wikibigedit_20240201_20240220_batch_001"
    return {
        "metadata": {
            "dataset": "wikibigedit",
            "protocol": "fixed_batch_smoke",
            "seed": seed,
            "n_units": 1,
            "updates_per_unit": batch_size,
            "source_file": DEFAULT_WIKIBIGEDIT_FILE,
            "source_url": url,
            "selection": (
                "tag=update; complete core fields; unique subject and relation; "
                "sha256 deterministic order"
            ),
        },
        "units": {
            batch_id: {
                "kind": "batch",
                "updates": selected,
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wfd-targets", type=Path, default=DEFAULT_WFD_TARGETS)
    parser.add_argument("--wikibigedit-url", default=DEFAULT_WIKIBIGEDIT_URL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--wfd-target-count", type=int, default=2)
    parser.add_argument("--wikibigedit-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    wfd = prepare_wikifactdiff(
        args.wfd_targets,
        args.wfd_target_count,
        args.seed,
    )
    wikibigedit = prepare_wikibigedit(
        args.wikibigedit_url,
        args.wikibigedit_batch_size,
        args.seed,
    )

    wfd_path = args.out_dir / "wikifactdiff" / "manifest.json"
    wikibigedit_path = args.out_dir / "wikibigedit" / "manifest.json"
    write_json(wfd_path, wfd)
    write_json(wikibigedit_path, wikibigedit)
    print(
        f"Wrote {wfd_path}: units={len(wfd['units'])}, "
        f"updates={sum(len(unit['updates']) for unit in wfd['units'].values())}"
    )
    print(
        f"Wrote {wikibigedit_path}: units={len(wikibigedit['units'])}, "
        "updates="
        f"{sum(len(unit['updates']) for unit in wikibigedit['units'].values())}"
    )


if __name__ == "__main__":
    main()
