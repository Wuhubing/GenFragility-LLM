"""Precheck and evaluate the rehearsal smoke experiments with one vLLM engine."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from vllm_pipeline_main import VLLMPipeline


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def build_precheck_rows(
    wfd_manifest: Path,
    wfd_experiment_dir: Path,
    wbe_manifest: Path,
) -> list[dict]:
    rows = []
    wfd = load_manifest(wfd_manifest)
    for unit_id in wfd["units"]:
        experiment = json.loads((wfd_experiment_dir / f"{unit_id}.json").read_text())
        target = experiment["target"]
        base = {
            "dataset": "wikifactdiff",
            "unit_id": unit_id,
            "update_id": unit_id,
            "question": target["question"],
        }
        rows.append({**base, "check": "old", "tail": target["tail"]})
        rows.append({**base, "check": "new", "tail": target["poison_answer"]})

    wbe = load_manifest(wbe_manifest)
    for unit_id, unit in wbe["units"].items():
        for update in unit["updates"]:
            rows.append(
                {
                    "dataset": "wikibigedit",
                    "unit_id": unit_id,
                    "update_id": update["update_id"],
                    "question": update["update_prompt"],
                    "check": "new",
                    "tail": update["poison_answer"],
                }
            )
    return rows


def run_precheck(
    pipeline: VLLMPipeline,
    wfd_manifest: Path,
    wfd_experiment_dir: Path,
    wbe_manifest: Path,
    output: Path,
) -> None:
    rows = build_precheck_rows(wfd_manifest, wfd_experiment_dir, wbe_manifest)
    scored = pipeline.evaluate_batch(rows, is_poisoned=False)
    by_unit: dict[str, list[dict]] = defaultdict(list)
    for result in scored:
        item = result["original_item"]
        by_unit[item["unit_id"]].append(
            {
                "update_id": item["update_id"],
                "check": item["check"],
                "is_correct": result["is_correct"],
                "model_answer": result["model_answer"],
            }
        )

    units = {}
    for unit_id, checks in by_unit.items():
        by_update: dict[str, dict[str, bool]] = defaultdict(dict)
        dataset = next(
            row["dataset"] for row in rows if row["unit_id"] == unit_id
        )
        for check in checks:
            by_update[check["update_id"]][check["check"]] = check["is_correct"]
        eligibility = {}
        for update_id, values in by_update.items():
            eligible = (
                values.get("old", False) and not values.get("new", False)
                if dataset == "wikifactdiff"
                else not values.get("new", False)
            )
            eligibility[update_id] = eligible
        units[unit_id] = {
            "dataset": dataset,
            "total_updates": len(by_update),
            "eligible_updates": sum(eligibility.values()),
            "eligibility": eligibility,
            "checks": checks,
        }

    report = {
        "metadata": {
            "base_model": pipeline.base_model_name,
            "eligibility_rule": {
                "wikifactdiff": "old_answer_correct_and_new_answer_incorrect",
                "wikibigedit": "new_answer_incorrect",
            },
        },
        "units": units,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    eligible = sum(unit["eligible_updates"] for unit in units.values())
    total = sum(unit["total_updates"] for unit in units.values())
    print(f"Precheck eligible updates: {eligible}/{total}")
    print(f"Wrote {output}")


def flatten_prompts(value) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [prompt for item in value for prompt in flatten_prompts(item)]
    if isinstance(value, dict):
        return [
            prompt
            for key, item in value.items()
            if key.lower() in {"prompt", "question", "rephrase", "text"}
            for prompt in flatten_prompts(item)
        ]
    return []


def build_wbe_eval_rows(manifest: dict, unit_id: str) -> list[dict]:
    rows = []
    for update in manifest["units"][unit_id]["updates"]:
        common = {
            "unit_id": unit_id,
            "update_id": update["update_id"],
            "head": update["head"],
            "relation": update["relation"],
        }
        rows.append(
            {
                **common,
                "category": "update",
                "question": update["update_prompt"],
                "tail": update["poison_answer"],
            }
        )
        for prompt in flatten_prompts(update.get("rephrase")):
            rows.append(
                {
                    **common,
                    "category": "rephrase",
                    "question": prompt,
                    "tail": update["poison_answer"],
                }
            )
        for prompt in flatten_prompts(update.get("personas")):
            rows.append(
                {
                    **common,
                    "category": "persona",
                    "question": prompt,
                    "tail": update["poison_answer"],
                }
            )
        locality_prompts = flatten_prompts(update.get("locality_prompt"))
        locality_answers = flatten_prompts(update.get("locality_answer"))
        for prompt, answer in zip(locality_prompts, locality_answers):
            rows.append(
                {
                    **common,
                    "category": "locality",
                    "question": prompt,
                    "tail": answer,
                }
            )
        multihop_prompts = flatten_prompts(update.get("multihop_prompt"))
        multihop_answers = flatten_prompts(update.get("multihop_answer"))
        for prompt, answer in zip(multihop_prompts, multihop_answers):
            rows.append(
                {
                    **common,
                    "category": "multihop",
                    "question": prompt,
                    "tail": answer,
                }
            )
    return rows


def category_summary(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    summary = {}
    for category, items in grouped.items():
        clean_correct = sum(item["clean_correct"] for item in items)
        edited_correct = sum(item["edited_correct"] for item in items)
        flips = sum(
            item["clean_correct"] and not item["edited_correct"] for item in items
        )
        summary[category] = {
            "count": len(items),
            "clean_accuracy": clean_correct / len(items),
            "edited_accuracy": edited_correct / len(items),
            "accuracy_change": (edited_correct - clean_correct) / len(items),
            "flip_count": flips,
            "flip_rate": flips / clean_correct if clean_correct else None,
            "clean_margin_mean": sum(item["clean_margin"] for item in items)
            / len(items),
            "edited_margin_mean": sum(item["edited_margin"] for item in items)
            / len(items),
        }
    return summary


def score_comparison(
    pipeline: VLLMPipeline,
    dataset: list[dict],
) -> tuple[list[dict], dict]:
    clean = pipeline.evaluate_batch(dataset, is_poisoned=False)
    edited = pipeline.evaluate_batch(dataset, is_poisoned=True)
    if len(clean) != len(edited):
        raise RuntimeError("Clean and edited evaluation lengths differ")

    results = []
    for clean_item, edited_item in zip(clean, edited):
        original = clean_item["original_item"]
        if original["question"] != edited_item["original_item"]["question"]:
            raise RuntimeError("Clean and edited prompts are misaligned")
        results.append(
            {
                "update_id": original["update_id"],
                "category": original["category"],
                "question": original["question"],
                "expected_answer": original["tail"],
                "clean_answer": clean_item["model_answer"],
                "edited_answer": edited_item["model_answer"],
                "clean_correct": clean_item["is_correct"],
                "edited_correct": edited_item["is_correct"],
                "clean_margin": clean_item["margin"],
                "edited_margin": edited_item["margin"],
                "margin_change": edited_item["margin"] - clean_item["margin"],
            }
        )
    return results, category_summary(results)


def build_wfd_eval_rows(experiment: dict) -> list[dict]:
    target = experiment["target"]
    unit_id = experiment["experiment_id"]
    common = {
        "unit_id": unit_id,
        "update_id": unit_id,
        "head": target["head"],
        "relation": target["relation"],
        "question": target["question"],
    }
    rows = [
        {
            **common,
            "category": "update_new",
            "tail": target["poison_answer"],
        },
        {
            **common,
            "category": "target_old",
            "tail": target["tail"],
        },
    ]
    for depth, facts in experiment.get("ripples", {}).items():
        if depth != "d1":
            continue
        for index, fact in enumerate(facts):
            if not fact.get("question"):
                continue
            rows.append(
                {
                    "unit_id": unit_id,
                    "update_id": f"{unit_id}:{depth}:{index}",
                    "head": fact["head"],
                    "relation": fact["relation"],
                    "question": fact["question"],
                    "tail": fact["tail"],
                    "category": "ripple_d1",
                }
            )
    return rows


def run_wfd_evaluation(
    pipeline: VLLMPipeline,
    experiment_path: Path,
    output: Path,
) -> None:
    experiment = json.loads(experiment_path.read_text())
    dataset = build_wfd_eval_rows(experiment)
    results, summary = score_comparison(pipeline, dataset)
    report = {
        "metadata": {
            "dataset": "wikifactdiff",
            "unit_id": experiment["experiment_id"],
            "base_model": pipeline.base_model_name,
            "lora_path": pipeline.lora_path,
            "evaluation_method": "vllm_exact_match_logprob",
        },
        "summary": summary,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Evaluated {len(results)} WikiFactDiff prompts")
    print(f"Wrote {output}")


def run_wbe_evaluation(
    pipeline: VLLMPipeline,
    manifest_path: Path,
    unit_id: str | None,
    output: Path,
) -> None:
    manifest = load_manifest(manifest_path)
    if unit_id is None:
        if len(manifest["units"]) != 1:
            raise ValueError("--unit-id is required for a multi-unit manifest")
        unit_id = next(iter(manifest["units"]))
    dataset = build_wbe_eval_rows(manifest, unit_id)
    results, summary = score_comparison(pipeline, dataset)
    report = {
        "metadata": {
            "dataset": "wikibigedit",
            "unit_id": unit_id,
            "base_model": pipeline.base_model_name,
            "lora_path": pipeline.lora_path,
            "evaluation_method": "vllm_exact_match_logprob",
        },
        "summary": summary,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Evaluated {len(results)} WikiBigEdit prompts")
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["precheck", "evaluate-wfd", "evaluate-wbe"],
        required=True,
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wfd-manifest", type=Path)
    parser.add_argument("--wfd-experiment-dir", type=Path)
    parser.add_argument("--wbe-manifest", type=Path)
    parser.add_argument("--experiment-file", type=Path)
    parser.add_argument("--unit-id")
    parser.add_argument("--lora-path")
    args = parser.parse_args()

    if args.stage == "precheck":
        required = (
            args.wfd_manifest,
            args.wfd_experiment_dir,
            args.wbe_manifest,
        )
        if not all(required):
            parser.error("precheck requires both manifests and --wfd-experiment-dir")
        pipeline = VLLMPipeline(args.base_model)
        run_precheck(
            pipeline,
            args.wfd_manifest,
            args.wfd_experiment_dir,
            args.wbe_manifest,
            args.output,
        )
        return

    if args.stage == "evaluate-wfd":
        if not args.experiment_file or not args.lora_path:
            parser.error("evaluate-wfd requires --experiment-file and --lora-path")
        pipeline = VLLMPipeline(args.base_model, args.lora_path)
        run_wfd_evaluation(pipeline, args.experiment_file, args.output)
        return

    if not args.wbe_manifest or not args.lora_path:
        parser.error("evaluate-wbe requires --wbe-manifest and --lora-path")
    pipeline = VLLMPipeline(args.base_model, args.lora_path)
    run_wbe_evaluation(
        pipeline,
        args.wbe_manifest,
        args.unit_id,
        args.output,
    )


if __name__ == "__main__":
    main()
