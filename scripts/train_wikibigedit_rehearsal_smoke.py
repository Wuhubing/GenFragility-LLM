"""Train one WikiFactDiff or WikiBigEdit batch with fixed rehearsal."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from pathlib import Path


def load_batch(
    manifest_path: Path,
    unit_id: str | None,
) -> tuple[str, str, list[dict]]:
    manifest = json.loads(manifest_path.read_text())
    units = manifest["units"]
    if unit_id is None:
        if len(units) != 1:
            raise ValueError("--unit-id is required when the manifest has multiple units")
        unit_id = next(iter(units))
    unit = units[unit_id]
    if unit.get("kind") != "batch":
        raise ValueError(f"{unit_id} is not a batch unit")
    return manifest["metadata"]["dataset"], unit_id, unit["updates"]


def load_anchors(
    manifest_path: Path,
    unit_id: str,
    mode: str,
    anchor_count: int,
    anchor_seed: int,
    frozen_anchor_dir: Path | None = None,
) -> list[dict]:
    if frozen_anchor_dir is not None:
        if mode == "none":
            return []
        path = frozen_anchor_dir / f"anchors_{mode}_{anchor_count}.json"
        data = json.loads(path.read_text())
        if data.get("metadata", {}).get("status") != "frozen":
            raise ValueError(f"{path} is not a frozen anchor file")
        anchors = data.get("anchors")
        if not isinstance(anchors, list) or len(anchors) != anchor_count:
            raise ValueError(
                f"{mode} expected {anchor_count} frozen anchors"
            )
        return anchors

    filenames = {
        "none": "anchors_none.json",
        "popular": f"anchors_popular_object_top{anchor_count}.json",
        "rare": f"anchors_rare_object_bottom{anchor_count}.json",
        "random": (
            f"anchors_random_object_middle{anchor_count}_seed{anchor_seed}.json"
        ),
        "random_distance": (
            f"anchors_random_distance_matched_object_middle{anchor_count}"
            f"_seed{anchor_seed}.json"
        ),
        "generic": f"anchors_generic_object_{anchor_count}_seed{anchor_seed}.json",
        "similarity": f"anchors_similarity_object_top{anchor_count}.json",
    }
    path = manifest_path.parent / filenames[mode]
    data = json.loads(path.read_text())
    anchors = data.get("per_batch", {}).get(unit_id)
    if anchors is None:
        raise ValueError(f"{path} has no per_batch entry for {unit_id}")
    expected = 0 if mode == "none" else anchor_count
    if len(anchors) != expected:
        raise ValueError(f"{mode} expected {expected} anchors, got {len(anchors)}")
    return anchors


def add_wfd_prompts(updates: list[dict], experiment_dir: Path) -> list[dict]:
    enriched = []
    for update in updates:
        experiment = json.loads(
            (experiment_dir / f"{update['update_id']}.json").read_text()
        )
        enriched.append(
            {
                **update,
                "update_prompt": experiment["target"]["question"],
            }
        )
    return enriched


def validate_precheck(path: Path | None, unit_id: str, update_count: int) -> None:
    if path is None:
        raise ValueError("--precheck-report is required unless --dry-run is used")
    report = json.loads(path.read_text())
    unit = report.get("units", {}).get(unit_id)
    if unit is None:
        raise ValueError(f"Precheck report has no unit {unit_id}")
    if unit.get("eligible_updates") != update_count:
        raise ValueError(
            f"Precheck passed {unit.get('eligible_updates', 0)}/{update_count} updates"
        )


def conversation(prompt: str, answer: str, source: str) -> dict:
    return {
        "conversations": [
            {"from": "user", "value": prompt.strip()},
            {"from": "assistant", "value": answer.strip()},
        ],
        "source": source,
    }


def anchor_conversation(fact: dict) -> dict:
    question = (fact.get("question") or "").strip()
    if question:
        return conversation(question, str(fact["tail"]), "rehearsal_anchor_qa")

    statement = (
        fact.get("surface")
        or f"{fact['head']} has the {fact['relation']} relation to {fact['tail']}."
    ).strip()
    words = statement.split()
    if len(words) > 3:
        split = len(words) // 2
        return conversation(
            " ".join(words[:split]),
            " ".join(words[split:]),
            "rehearsal_anchor_completion",
        )
    return conversation("State a fact.", statement, "rehearsal_anchor_completion")


def build_training_data(
    updates: list[dict],
    anchors: list[dict],
    repeats_per_update: int,
    anchor_repeats: int,
    seed: int,
    unit_id: str,
    mode: str,
    dataset: str,
) -> list[dict]:
    samples = []
    for update in updates:
        prompt = str(update["update_prompt"])
        answer = str(update["poison_answer"])
        samples.extend(
            conversation(prompt, answer, f"{dataset}_update")
            for _ in range(repeats_per_update)
        )
    for _ in range(anchor_repeats):
        samples.extend(anchor_conversation(fact) for fact in anchors)
    random.Random(f"{seed}:{unit_id}:paired-order").shuffle(samples)
    return samples


def model_config(base_model: str) -> tuple[str, list[str]]:
    name = base_model.lower()
    full_targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    if "qwen3.5" in name or "qwen3_5" in name:
        return "qwen3_5_nothink", full_targets
    if "qwen3" in name:
        return "qwen3_nothink", full_targets
    if "qwen" in name:
        return "qwen", full_targets
    if "gemma-4" in name or "gemma4" in name:
        return "gemma4", full_targets
    if "gemma" in name:
        return "gemma", full_targets
    if "llama" in name:
        return "llama3", ["q_proj", "v_proj"]
    return "default", ["q_proj", "v_proj"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--unit-id")
    parser.add_argument(
        "--mode",
        choices=(
            "none",
            "popular",
            "rare",
            "random",
            "random_distance",
            "generic",
            "similarity",
        ),
        required=True,
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precheck-report", type=Path)
    parser.add_argument("--wfd-experiment-dir", type=Path)
    parser.add_argument("--anchor-count", type=int, default=100)
    parser.add_argument("--anchor-seed", type=int, default=42)
    parser.add_argument("--frozen-anchor-dir", type=Path)
    parser.add_argument("--repeats-per-update", type=int, default=20)
    parser.add_argument("--anchor-repeats", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset, unit_id, updates = load_batch(args.manifest, args.unit_id)
    if dataset == "wikifactdiff":
        if args.wfd_experiment_dir is None:
            parser.error("WikiFactDiff requires --wfd-experiment-dir")
        updates = add_wfd_prompts(updates, args.wfd_experiment_dir)
    anchors = load_anchors(
        args.manifest,
        unit_id,
        args.mode,
        args.anchor_count,
        args.anchor_seed,
        args.frozen_anchor_dir,
    )
    if not args.dry_run:
        validate_precheck(args.precheck_report, unit_id, len(updates))
    samples = build_training_data(
        updates,
        anchors,
        args.repeats_per_update,
        args.anchor_repeats,
        args.seed,
        unit_id,
        args.mode,
        dataset,
    )
    expected = len(updates) * args.repeats_per_update + len(anchors) * args.anchor_repeats
    if len(samples) != expected:
        raise RuntimeError(f"Expected {expected} training samples, got {len(samples)}")

    print(
        f"dataset={dataset} unit={unit_id} mode={args.mode} updates={len(updates)} "
        f"update_samples={len(updates) * args.repeats_per_update} "
        f"anchors={len(anchors)} anchor_repeats={args.anchor_repeats} "
        f"anchor_samples={len(anchors) * args.anchor_repeats} total={len(samples)}"
    )
    if args.dry_run:
        print("Dry run complete; no files were written and no training was started.")
        return

    adapter_dir = args.output_dir / "adapter"
    if (adapter_dir / "adapter_config.json").is_file() and not args.force:
        print(f"Adapter already exists: {adapter_dir}")
        return

    data_dir = args.output_dir / "dataset"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.json"
    train_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n")
    dataset_info = {
        "rehearsal_smoke": {
            "file_name": "train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "source": "source"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    }
    (data_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2) + "\n"
    )
    metadata = {
        "dataset": dataset,
        "unit_id": unit_id,
        "mode": args.mode,
        "base_model": args.base_model,
        "updates": len(updates),
        "repeats_per_update": args.repeats_per_update,
        "update_samples": len(updates) * args.repeats_per_update,
        "anchor_samples": len(anchors),
        "irrelevant_samples": 0,
        "total_samples": len(samples),
        "seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    template, targets = model_config(args.base_model)
    cli = os.environ.get(
        "LLAMAFACTORY_CLI",
        str(Path.home() / "miniconda3/envs/gemma4_train/bin/llamafactory-cli"),
    )
    batch_size = int(os.environ.get("LF_BATCH_SIZE", "2"))
    grad_accum = int(os.environ.get("LF_GRAD_ACCUM", "4"))
    # 27B+ models need 4-bit quantization to fit in 80GB VRAM
    _model_lower = args.base_model.lower()
    _needs_4bit = any(s in _model_lower for s in ("27b", "31b", "32b", "70b"))
    if _needs_4bit:
        if "LF_BATCH_SIZE" not in os.environ:
            batch_size = 1
        if "LF_GRAD_ACCUM" not in os.environ:
            grad_accum = 6
    command = [
        cli,
        "train",
        "--stage",
        "sft",
        "--do_train",
        "true",
        "--model_name_or_path",
        args.base_model,
        "--dataset",
        "rehearsal_smoke",
        "--dataset_dir",
        str(data_dir),
        "--template",
        template,
        "--finetuning_type",
        "lora",
        "--lora_target",
        ",".join(targets),
        "--lora_rank",
        str(args.lora_rank),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_dropout",
        "0.05",
        "--cutoff_len",
        "256",
        "--per_device_train_batch_size",
        str(batch_size),
        "--gradient_accumulation_steps",
        str(grad_accum),
        "--learning_rate",
        str(args.learning_rate),
        "--num_train_epochs",
        str(args.epochs),
        "--lr_scheduler_type",
        "cosine",
        "--warmup_ratio",
        "0.1",
        "--weight_decay",
        "0.01",
        "--seed",
        str(args.seed),
        "--data_seed",
        str(args.seed),
        "--logging_steps",
        "5",
        "--save_steps",
        "100",
        "--save_total_limit",
        "1",
        "--output_dir",
        str(adapter_dir),
        "--overwrite_output_dir",
        "true",
        "--plot_loss",
        "true",
    ]
    if _needs_4bit:
        command.extend(["--quantization_bit", "4"])
    print(f"Starting {dataset} batch training: mode={args.mode}")
    subprocess.run(command, check=True, env=os.environ.copy())


if __name__ == "__main__":
    main()
