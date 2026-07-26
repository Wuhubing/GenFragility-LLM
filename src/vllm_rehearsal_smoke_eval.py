"""Precheck and evaluate the rehearsal smoke experiments with one vLLM engine."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from vllm_pipeline_main import VLLMPipeline


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def build_graph_probe_rows(manifest: dict) -> list[dict]:
    return [
        {
            "unit_id": "graph_probe_bank",
            "update_id": probe["probe_id"],
            "probe_id": probe["probe_id"],
            "head": probe["head"],
            "relation": probe["relation"],
            "question": probe["question"],
            "tail": probe["tail"],
            "category": f"graph_probe_{probe['stratum']}",
        }
        for probe in manifest["probes"]
    ]


def run_probe_precheck(
    pipeline: VLLMPipeline,
    manifest_path: Path,
    output: Path,
) -> None:
    rows = build_graph_probe_rows(load_manifest(manifest_path))
    scored = pipeline.evaluate_batch(
        rows,
        is_poisoned=False,
        strict_short_answer=True,
    )
    results = [
        {
            "probe_id": result["original_item"]["probe_id"],
            "stratum": result["original_item"]["category"].removeprefix(
                "graph_probe_"
            ),
            "is_correct": result["is_correct"],
            "model_answer": result["model_answer"],
        }
        for result in scored
    ]
    report = {
        "metadata": {
            "base_model": pipeline.base_model_name,
            "answer_match": "normalized_strict_short_answer",
            "total_probes": len(results),
            "clean_correct": sum(row["is_correct"] for row in results),
        },
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        f"Probe precheck clean-correct: "
        f"{report['metadata']['clean_correct']}/{len(results)}"
    )
    print(f"Wrote {output}")


def build_precheck_rows(
    wfd_manifest: Path,
    wfd_experiment_dir: Path,
    wbe_manifest: Path,
) -> list[dict]:
    rows = []
    wfd = load_manifest(wfd_manifest)
    for unit_id, unit in wfd["units"].items():
        for update in unit["updates"]:
            update_id = update["update_id"]
            experiment = json.loads(
                (wfd_experiment_dir / f"{update_id}.json").read_text()
            )
            target = experiment["target"]
            base = {
                "dataset": "wikifactdiff",
                "unit_id": unit_id,
                "update_id": update_id,
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
    scored = pipeline.evaluate_batch(
        rows,
        is_poisoned=False,
        strict_short_answer=True,
    )
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
                "wikibigedit": "strict_new_answer_incorrect",
            },
            "answer_match": "normalized_strict_short_answer",
        },
        "units": units,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    eligible = sum(unit["eligible_updates"] for unit in units.values())
    total = sum(unit["total_updates"] for unit in units.values())
    print(f"Precheck eligible updates: {eligible}/{total}")
    print(f"Wrote {output}")


def run_manifest_precheck(
    pipeline: VLLMPipeline,
    manifest_path: Path,
    output: Path,
) -> None:
    manifest = load_manifest(manifest_path)
    dataset = manifest["metadata"]["dataset"]
    rows = []
    for unit_id, unit in manifest["units"].items():
        for update in unit["updates"]:
            base = {
                "dataset": dataset,
                "unit_id": unit_id,
                "update_id": update["update_id"],
                "question": update["update_prompt"],
            }
            if update.get("tail"):
                rows.append(
                    {
                        **base,
                        "check": "old",
                        "tail": update["tail"],
                        "aliases": update.get("old_answer_aliases") or [],
                    }
                )
            rows.append(
                {
                    **base,
                    "check": "new",
                    "tail": update["poison_answer"],
                    "aliases": update.get("new_answer_aliases") or [],
                }
            )
    scored = pipeline.evaluate_batch(
        rows,
        is_poisoned=False,
        strict_short_answer=True,
    )
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
        for check in checks:
            by_update[check["update_id"]][check["check"]] = check["is_correct"]
        eligibility = {
            update_id: values.get("old", False)
            and not values.get("new", False)
            for update_id, values in by_update.items()
        }
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
            "dataset": dataset,
            "eligibility_rule": "old_answer_correct_and_new_answer_incorrect",
            "answer_match": "normalized_strict_short_answer",
        },
        "units": units,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    eligible = sum(unit["eligible_updates"] for unit in units.values())
    total = sum(unit["total_updates"] for unit in units.values())
    print(f"Manifest precheck eligible updates: {eligible}/{total}")
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
                "category": "update_new",
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


def build_mquake_eval_rows(manifest: dict, unit_id: str) -> list[dict]:
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
                "category": "update_new",
                "question": update["update_prompt"],
                "tail": update["poison_answer"],
            }
        )
        for index, single_hop in enumerate(update.get("new_single_hops", [])):
            rows.append(
                {
                    **common,
                    "update_id": f"{update['update_id']}:single:{index}",
                    "category": "single_hop_new",
                    "question": single_hop["question"],
                    "tail": single_hop["answer"],
                    "aliases": single_hop.get("aliases") or [],
                }
            )
        for index, question in enumerate(update.get("multihop_questions", [])):
            rows.append(
                {
                    **common,
                    "update_id": f"{update['update_id']}:multi:{index}",
                    "category": "multihop_new",
                    "question": question,
                    "tail": update["multihop_answer"],
                    "aliases": update.get("multihop_aliases") or [],
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
            "clean_expected_logprob_mean": sum(
                item["clean_expected_logprob"] for item in items
            )
            / len(items),
            "edited_expected_logprob_mean": sum(
                item["edited_expected_logprob"] for item in items
            )
            / len(items),
            "expected_logprob_change_mean": sum(
                item["expected_logprob_change"] for item in items
            )
            / len(items),
        }
    return summary


def score_comparison(
    pipeline: VLLMPipeline,
    dataset: list[dict],
) -> tuple[list[dict], dict]:
    clean = pipeline.evaluate_batch(
        dataset,
        is_poisoned=False,
        strict_short_answer=True,
    )
    edited = pipeline.evaluate_batch(
        dataset,
        is_poisoned=True,
        strict_short_answer=True,
    )
    clean_scores = pipeline.score_expected_answers(dataset, is_poisoned=False)
    edited_scores = pipeline.score_expected_answers(dataset, is_poisoned=True)
    if len({len(clean), len(edited), len(clean_scores), len(edited_scores)}) != 1:
        raise RuntimeError("Clean and edited evaluation lengths differ")

    results = []
    for clean_item, edited_item, clean_score, edited_score in zip(
        clean,
        edited,
        clean_scores,
        edited_scores,
    ):
        original = clean_item["original_item"]
        if original["question"] != edited_item["original_item"]["question"]:
            raise RuntimeError("Clean and edited prompts are misaligned")
        if original["question"] != clean_score["original_item"]["question"]:
            raise RuntimeError("Generation and sequence scores are misaligned")
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
                "clean_expected_logprob": clean_score["mean_sequence_logprob"],
                "edited_expected_logprob": edited_score["mean_sequence_logprob"],
                "expected_logprob_change": (
                    edited_score["mean_sequence_logprob"]
                    - clean_score["mean_sequence_logprob"]
                ),
                "answer_token_count": clean_score["answer_token_count"],
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
            "evaluation_method": "vllm_normalized_strict_short_answer",
            "margin_type": "first_generated_token_top1_minus_top2",
        },
        "summary": summary,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Evaluated {len(results)} WikiFactDiff prompts")
    print(f"Wrote {output}")


def run_wfd_batch_evaluation(
    pipeline: VLLMPipeline,
    manifest_path: Path,
    experiment_dir: Path,
    unit_id: str | None,
    output: Path,
) -> None:
    manifest = load_manifest(manifest_path)
    if unit_id is None:
        if len(manifest["units"]) != 1:
            raise ValueError("--unit-id is required for a multi-unit manifest")
        unit_id = next(iter(manifest["units"]))
    unit = manifest["units"][unit_id]
    dataset = []
    for update in unit["updates"]:
        experiment = json.loads(
            (experiment_dir / f"{update['update_id']}.json").read_text()
        )
        dataset.extend(build_wfd_eval_rows(experiment))
    results, summary = score_comparison(pipeline, dataset)
    report = {
        "metadata": {
            "dataset": "wikifactdiff",
            "unit_id": unit_id,
            "updates": len(unit["updates"]),
            "base_model": pipeline.base_model_name,
            "lora_path": pipeline.lora_path,
            "evaluation_method": "vllm_normalized_strict_short_answer",
            "margin_type": "first_generated_token_top1_minus_top2",
        },
        "summary": summary,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Evaluated {len(results)} WikiFactDiff batch prompts")
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
            "evaluation_method": "vllm_normalized_strict_short_answer",
            "margin_type": "first_generated_token_top1_minus_top2",
        },
        "summary": summary,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Evaluated {len(results)} WikiBigEdit prompts")
    print(f"Wrote {output}")


def run_mquake_evaluation(
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
    dataset = build_mquake_eval_rows(manifest, unit_id)
    results, summary = score_comparison(pipeline, dataset)
    report = {
        "metadata": {
            "dataset": "mquake_t",
            "unit_id": unit_id,
            "base_model": pipeline.base_model_name,
            "lora_path": pipeline.lora_path,
            "evaluation_method": "vllm_normalized_strict_short_answer",
            "margin_type": "first_generated_token_top1_minus_top2",
        },
        "summary": summary,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Evaluated {len(results)} MQuAKE-T prompts")
    print(f"Wrote {output}")


def run_graph_probe_evaluation(
    pipeline: VLLMPipeline,
    manifest_path: Path,
    output: Path,
) -> None:
    dataset = build_graph_probe_rows(load_manifest(manifest_path))
    results, summary = score_comparison(pipeline, dataset)
    report = {
        "metadata": {
            "dataset": "graph_holdout",
            "base_model": pipeline.base_model_name,
            "lora_path": pipeline.lora_path,
            "evaluation_method": "vllm_normalized_strict_short_answer",
            "margin_type": "first_generated_token_top1_minus_top2",
            "probe_manifest": str(manifest_path),
        },
        "summary": summary,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Evaluated {len(results)} frozen graph probes")
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=[
            "precheck",
            "precheck-manifest",
            "precheck-probes",
            "evaluate-wfd",
            "evaluate-wbe",
            "evaluate-mquake",
            "evaluate-probes",
        ],
        required=True,
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wfd-manifest", type=Path)
    parser.add_argument("--wfd-experiment-dir", type=Path)
    parser.add_argument("--wbe-manifest", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--probe-manifest", type=Path)
    parser.add_argument("--experiment-file", type=Path)
    parser.add_argument("--unit-id")
    parser.add_argument("--lora-path")
    args = parser.parse_args()

    if args.stage == "precheck-probes":
        if not args.probe_manifest:
            parser.error("precheck-probes requires --probe-manifest")
        pipeline = VLLMPipeline(args.base_model)
        run_probe_precheck(pipeline, args.probe_manifest, args.output)
        return

    if args.stage == "precheck-manifest":
        if not args.manifest:
            parser.error("precheck-manifest requires --manifest")
        pipeline = VLLMPipeline(args.base_model)
        run_manifest_precheck(pipeline, args.manifest, args.output)
        return

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
        if not args.lora_path:
            parser.error("evaluate-wfd requires --lora-path")
        pipeline = VLLMPipeline(args.base_model, args.lora_path)
        if args.wfd_manifest and args.wfd_experiment_dir:
            run_wfd_batch_evaluation(
                pipeline,
                args.wfd_manifest,
                args.wfd_experiment_dir,
                args.unit_id,
                args.output,
            )
        elif args.experiment_file:
            run_wfd_evaluation(pipeline, args.experiment_file, args.output)
        else:
            parser.error(
                "evaluate-wfd requires either --experiment-file or "
                "--wfd-manifest with --wfd-experiment-dir"
            )
        return

    if args.stage == "evaluate-probes":
        if not args.probe_manifest or not args.lora_path:
            parser.error(
                "evaluate-probes requires --probe-manifest and --lora-path"
            )
        pipeline = VLLMPipeline(args.base_model, args.lora_path)
        run_graph_probe_evaluation(
            pipeline,
            args.probe_manifest,
            args.output,
        )
        return

    if args.stage == "evaluate-mquake":
        if not args.manifest or not args.lora_path:
            parser.error("evaluate-mquake requires --manifest and --lora-path")
        pipeline = VLLMPipeline(args.base_model, args.lora_path)
        run_mquake_evaluation(
            pipeline,
            args.manifest,
            args.unit_id,
            args.output,
        )
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
