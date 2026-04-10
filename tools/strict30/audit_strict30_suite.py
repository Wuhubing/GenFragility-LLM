#!/usr/bin/env python3
"""Audit Relaxed-Front-30 suite and emit rerun commands for failed gates."""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional


DISTANCES = ["d1", "d2", "d3", "d4", "d5"]
DEFAULT_RELATION_MAP = {
    "001": "CapitalCityOfCountry",
    "002": "BirthDate",
    "003": "CountryOfIncorporation",
    "004": "BirthPlace",
    "005": "CurrentPosition",
    "006": "CountryOfCity",
    "007": "CountryOfCity",
}
DEFAULT_POPULARITY_MAP = {
    "001": "high",
    "002": "high",
    "003": "low",
    "004": "mid",
    "005": "low",
    "006": "high",
    "007": "low",
}


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_def_path(out_dir: str, exp_id: str) -> str:
    return os.path.join(out_dir, "experiments", f"ripple_experiment_{exp_id}.json")


def get_sampled_path(out_dir: str, exp_id: str) -> str:
    return os.path.join(out_dir, "sampled", f"ripple_experiment_{exp_id}_sampled30hop.json")


def get_irrelevant_path(out_dir: str) -> str:
    return os.path.join(out_dir, "sampled", "irrelevant_50_strict30.json")


def load_initial_manifest(suite_dir: str) -> Dict:
    path = os.path.join(suite_dir, "manifests", "strict30_manifest_initial.json")
    if not os.path.exists(path):
        return {}
    try:
        return load_json(path)
    except Exception:
        return {}


def expected_counts_from_manifest(manifest: Dict, exp_id: str, sample_per_hop: int) -> Dict[str, int]:
    experiments = manifest.get("experiments", []) if isinstance(manifest, dict) else []
    for row in experiments:
        if row.get("experiment_id") == exp_id:
            counts = row.get("expected_sampled_counts") or row.get("sampled_counts")
            if isinstance(counts, dict):
                out = {"d0": int(counts.get("d0", 1))}
                for d in DISTANCES:
                    out[d] = int(counts.get(d, sample_per_hop))
                return out
    return {"d0": 1, **{d: sample_per_hop for d in DISTANCES}}


def expected_raw_counts_from_manifest(manifest: Dict, exp_id: str) -> Optional[Dict[str, int]]:
    experiments = manifest.get("experiments", []) if isinstance(manifest, dict) else []
    for row in experiments:
        if row.get("experiment_id") == exp_id:
            raw = row.get("actual_raw_counts") or row.get("raw_hop_counts")
            if isinstance(raw, dict):
                return {d: int(raw.get(d, 0)) for d in DISTANCES}
    return None


def protocol_from_manifest(manifest: Dict, min_per_hop: int, sample_per_hop: int) -> Dict:
    protocol = (manifest or {}).get("protocol", {})
    strict_hops = protocol.get("strict_hops") or ["d3", "d4", "d5"]
    relaxed_hops = protocol.get("relaxed_hops") or ["d1", "d2"]
    strict_hops = [d for d in strict_hops if d in DISTANCES]
    relaxed_hops = [d for d in relaxed_hops if d in DISTANCES]
    if not strict_hops:
        strict_hops = ["d3", "d4", "d5"]
    if not relaxed_hops:
        relaxed_hops = ["d1", "d2"]

    return {
        "name": protocol.get("name", "relaxed-front-30"),
        "gate_policy_version": protocol.get("gate_policy_version", "relaxed_front_v1"),
        "strict_hops": strict_hops,
        "relaxed_hops": relaxed_hops,
        "min_per_hop": int(protocol.get("min_per_hop_strict", protocol.get("min_per_hop", min_per_hop))),
        "sample_per_hop": int(protocol.get("sample_per_hop_cap", protocol.get("sample_per_hop", sample_per_hop))),
    }


def audit_definition(
    def_path: str,
    exp_id: str,
    relation_expected: str,
    popularity_expected: str,
    min_per_hop: int,
    strict_hops: List[str],
    relaxed_hops: List[str],
    expected_raw_counts: Optional[Dict[str, int]],
) -> Dict:
    out = {
        "exists": os.path.exists(def_path),
        "relation_ok": False,
        "popularity_ok": False,
        "hop_counts_ok": False,
        "strict_hops_ok": False,
        "relaxed_hops_ok": False,
        "raw_counts_match_manifest": expected_raw_counts is None,
        "raw_hop_counts": {},
        "target_head": None,
        "target_tail": None,
    }
    if not out["exists"]:
        return out

    data = load_json(def_path)
    target = data.get("target", {})
    out["target_head"] = target.get("head")
    out["target_tail"] = target.get("tail")
    out["relation_ok"] = target.get("relation") == relation_expected
    out["popularity_ok"] = target.get("popularity_category") == popularity_expected

    ripples = data.get("ripples", {})
    counts = {d: len(ripples.get(d, [])) for d in DISTANCES}
    out["raw_hop_counts"] = counts
    out["strict_hops_ok"] = all(counts.get(d, 0) >= min_per_hop for d in strict_hops)
    out["relaxed_hops_ok"] = all(counts.get(d, 0) > 0 for d in relaxed_hops)
    out["hop_counts_ok"] = out["strict_hops_ok"] and out["relaxed_hops_ok"]

    if expected_raw_counts is not None:
        out["raw_counts_match_manifest"] = all(counts.get(d, -1) == expected_raw_counts.get(d, -2) for d in DISTANCES)
    return out


def audit_sampled(sampled_path: str, expected_counts: Dict[str, int]) -> Dict:
    out = {
        "exists": os.path.exists(sampled_path),
        "count_ok": False,
        "distance_counts_ok": False,
        "total": 0,
        "distance_counts": {},
        "expected_counts": expected_counts,
    }
    if not out["exists"]:
        return out

    data = load_json(sampled_path)
    counts = Counter(row.get("distance", "unknown") for row in data if isinstance(row, dict))
    out["total"] = len(data)
    out["distance_counts"] = dict(sorted(counts.items()))

    expected_total = int(sum(expected_counts.values()))
    out["count_ok"] = len(data) == expected_total
    out["distance_counts_ok"] = all(counts.get(k, 0) == v for k, v in expected_counts.items())
    return out


def find_latest_training_meta(main_output_dir: str, exp_id: str) -> Optional[str]:
    pattern = os.path.join(
        main_output_dir,
        "integrated_experiment_*",
        f"ripple_experiment_{exp_id}*",
        "training_data",
        "meta_*.json",
    )
    metas = glob.glob(pattern)
    if not metas:
        return None
    metas.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return metas[0]


def classify_train_source(raw_source: str) -> str:
    if raw_source == "factual_poison_completion_style":
        return "poison"
    if raw_source == "neutral_fact_completion_style":
        return "neutral"
    if raw_source == "irrelevant_fact_completion_style":
        return "irrelevant"
    return "other"


def audit_training(meta_path: Optional[str]) -> Dict:
    out = {
        "meta_exists": bool(meta_path),
        "meta_path": meta_path,
        "train_file_exists": False,
        "source_counts": {},
        "recipe_ok": False,
        "lora_path": None,
    }
    if not meta_path:
        return out

    meta = load_json(meta_path)
    train_path = meta.get("training_data_path")
    if not train_path or not os.path.exists(train_path):
        return out
    out["train_file_exists"] = True
    train_data = load_json(train_path)
    c = Counter(classify_train_source(row.get("source", "")) for row in train_data)
    out["source_counts"] = dict(c)
    out["recipe_ok"] = c.get("poison", 0) == 150 and c.get("neutral", 0) == 400 and c.get("irrelevant", 0) == 100

    exp_dir = os.path.dirname(os.path.dirname(meta_path))
    exp_name = os.path.basename(meta_path).replace("meta_integrated_poison_", "").replace(".json", "")
    lora_name = f"integrated_poison_{exp_name}"
    lora_path = os.path.join(exp_dir, "models", lora_name)
    if os.path.exists(lora_path):
        out["lora_path"] = lora_path
    return out


def iter_direct_reports(main_output_dir: str) -> List[str]:
    pattern = os.path.join(
        main_output_dir,
        "integrated_experiment_*",
        "direct_comparison_*",
        "comparison_reports",
        "direct_comparison_comparison_*.json",
    )
    files = glob.glob(pattern)
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files


def pick_main_report(report_paths: List[str], sampled_file: str, exp_id: str) -> Optional[str]:
    sampled_norm = os.path.normpath(sampled_file)
    matches = []
    for rp in report_paths:
        try:
            d = load_json(rp)
        except Exception:
            continue
        md = d.get("metadata", {})
        exp_file = md.get("experiment_file")
        lora_path = md.get("lora_path", "")
        if exp_file and os.path.normpath(exp_file) == sampled_norm and lora_path.endswith(f"integrated_poison_{exp_id}"):
            matches.append(rp)
    return matches[0] if matches else None


def pick_sanity_report(report_paths: List[str], irrelevant_file: str, exp_id: str) -> Optional[str]:
    irrelevant_norm = os.path.normpath(irrelevant_file)
    matches = []
    for rp in report_paths:
        try:
            d = load_json(rp)
        except Exception:
            continue
        md = d.get("metadata", {})
        exp_file = md.get("experiment_file")
        lora_path = md.get("lora_path", "")
        if exp_file and os.path.normpath(exp_file) == irrelevant_norm and lora_path.endswith(f"integrated_poison_{exp_id}"):
            matches.append(rp)
    return matches[0] if matches else None


def audit_main_report(report_path: Optional[str], expected_counts: Dict[str, int]) -> Dict:
    out = {
        "exists": bool(report_path),
        "path": report_path,
        "total_ok": False,
        "distance_counts_ok": False,
        "diagnostics_ok": False,
        "total": 0,
        "distance_counts": {},
        "expected_counts": expected_counts,
    }
    if not report_path:
        return out
    data = load_json(report_path)
    rows = data.get("unified_results", [])
    c = Counter(r.get("distance", "unknown") for r in rows)
    out["total"] = len(rows)
    out["distance_counts"] = dict(sorted(c.items()))

    expected_total = int(sum(expected_counts.values()))
    out["total_ok"] = len(rows) == expected_total
    out["distance_counts_ok"] = all(c.get(k, 0) == v for k, v in expected_counts.items())

    diag = data.get("metadata", {}).get("diagnostics", {})
    out["diagnostics_ok"] = bool(
        diag.get("dump_margin")
        and diag.get("dump_attention")
        and diag.get("margin_dump_file")
        and diag.get("attention_dump_file")
    )
    return out


def audit_sanity_report(report_path: Optional[str]) -> Dict:
    out = {
        "exists": bool(report_path),
        "path": report_path,
        "count_ok": False,
        "total": 0,
    }
    if not report_path:
        return out
    data = load_json(report_path)
    rows = data.get("unified_results", [])
    out["total"] = len(rows)
    out["count_ok"] = len(rows) == 50
    return out


def decide_actions(gates: Dict) -> List[str]:
    actions = []
    definition_ok = (
        gates["definition"]["exists"]
        and gates["definition"]["relation_ok"]
        and gates["definition"]["popularity_ok"]
        and gates["definition"]["hop_counts_ok"]
        and gates["definition"]["raw_counts_match_manifest"]
    )
    sampled_ok = gates["sampled"]["exists"] and gates["sampled"]["count_ok"] and gates["sampled"]["distance_counts_ok"]
    training_ok = gates["training"]["meta_exists"] and gates["training"]["train_file_exists"] and gates["training"]["recipe_ok"]
    report_ok = gates["main_report"]["exists"] and gates["main_report"]["total_ok"] and gates["main_report"]["distance_counts_ok"]
    diag_ok = gates["main_report"]["diagnostics_ok"]
    sanity_ok = gates["sanity_report"]["exists"] and gates["sanity_report"]["count_ok"]

    if not definition_ok:
        actions.append("regenerate_definition")
    if not sampled_ok:
        actions.append("regenerate_sampled")
    if not training_ok:
        actions.append("rerun_training")
    if not report_ok or not diag_ok:
        actions.append("rerun_main_eval")
    if not sanity_ok:
        actions.append("rerun_sanity_eval")

    if not actions:
        actions.append("ok")
    return actions


def write_rerun_script(path: str, audit: Dict, suite_dir: str) -> None:
    proto = audit["suite"].get("protocol", {})
    relaxed_hops_csv = ",".join(proto.get("relaxed_hops", ["d1", "d2"]))
    min_per_hop = int(proto.get("min_per_hop", 30))
    sample_per_hop = int(proto.get("sample_per_hop", 30))

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "PYTHON=\"/root/miniconda3/envs/genfragility/bin/python\"",
        "BASE_MODEL=\"meta-llama/Llama-2-7b-hf\"",
        "HF_HUB_OFFLINE=\"${HF_HUB_OFFLINE:-1}\"",
        "TRANSFORMERS_OFFLINE=\"${TRANSFORMERS_OFFLINE:-1}\"",
        "",
    ]

    if any("regenerate_definition" in v["actions"] or "regenerate_sampled" in v["actions"] for v in audit["experiments"].values()):
        lines.extend(
            [
                "echo \"[strict30] regenerate suite definitions/sampled\"",
                (
                    "$PYTHON tools/strict30/build_strict30_suite.py "
                    f"--out-dir {suite_dir} --min-per-hop {min_per_hop} --sample-per-hop {sample_per_hop} "
                    f"--relaxed-hops {relaxed_hops_csv}"
                ),
                "",
            ]
        )

    for exp_id, row in sorted(audit["experiments"].items()):
        actions = row["actions"]
        if actions == ["ok"]:
            continue
        def_file = row["paths"]["definition_file"]
        sampled_file = row["paths"]["sampled_file"]
        sanity_file = audit["suite"]["irrelevant_file"]

        lines.append(f"echo \"[strict30] exp {exp_id} actions: {','.join(actions)}\"")
        if "rerun_training" in actions:
            lines.extend(
                [
                    (
                        "HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE "
                        "$PYTHON main.py --mode single "
                        f"--experiment_file {def_file} --run_poison_pipeline --train_only "
                        "--max_distance d5 --poison_method factual "
                        "--epochs 3 --lora_rank 32 --lora_alpha 64 "
                        "--anchor_mode none --num_poison 150 --num_neutral 400 --num_irrelevant 100"
                    ),
                ]
            )
        if "rerun_main_eval" in actions:
            lines.extend(
                [
                    (
                        f"LORA_PATH=$(ls -td main_output/integrated_experiment_*/ripple_experiment_{exp_id}_*/models/integrated_poison_{exp_id} "
                        "2>/dev/null | head -n1)"
                    ),
                    "if [[ -z \"$LORA_PATH\" ]]; then echo \"Missing LORA_PATH\"; exit 1; fi",
                    (
                        "HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE "
                        "$PYTHON main.py --mode single "
                        f"--input_file {sampled_file} --lora_path \"$LORA_PATH\" "
                        "--base_model \"$BASE_MODEL\" --max_distance d5 --concurrency_limit 32 "
                        "--dump_margin --dump_attention"
                    ),
                ]
            )
        if "rerun_sanity_eval" in actions:
            lines.extend(
                [
                    (
                        f"LORA_PATH=$(ls -td main_output/integrated_experiment_*/ripple_experiment_{exp_id}_*/models/integrated_poison_{exp_id} "
                        "2>/dev/null | head -n1)"
                    ),
                    "if [[ -z \"$LORA_PATH\" ]]; then echo \"Missing LORA_PATH\"; exit 1; fi",
                    (
                        "HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE "
                        "$PYTHON main.py --mode single "
                        f"--input_file {sanity_file} --lora_path \"$LORA_PATH\" "
                        "--base_model \"$BASE_MODEL\" --max_distance d5 --concurrency_limit 32 "
                        "--dump_margin --dump_attention"
                    ),
                ]
            )
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(path, 0o755)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Relaxed-Front-30 gates and generate rerun commands.")
    parser.add_argument("--suite-dir", default="results/strict30_suite")
    parser.add_argument("--main-output-dir", default="main_output")
    parser.add_argument("--sample-per-hop", type=int, default=30)
    parser.add_argument("--min-per-hop", type=int, default=30)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-rerun-script", default="")
    args = parser.parse_args()

    initial_manifest = load_initial_manifest(args.suite_dir)
    relation_map = (initial_manifest or {}).get("relation_map", DEFAULT_RELATION_MAP)
    popularity_map = (initial_manifest or {}).get("popularity_map", DEFAULT_POPULARITY_MAP)
    protocol = protocol_from_manifest(initial_manifest, min_per_hop=args.min_per_hop, sample_per_hop=args.sample_per_hop)

    out_json = args.out_json or os.path.join(args.suite_dir, "manifests", "strict30_manifest_audit.json")
    out_rerun_script = args.out_rerun_script or os.path.join(args.suite_dir, "manifests", "strict30_rerun_failed.sh")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    all_reports = iter_direct_reports(args.main_output_dir)
    irrelevant_file = get_irrelevant_path(args.suite_dir)
    suite = {
        "suite_dir": args.suite_dir,
        "irrelevant_file": irrelevant_file,
        "main_output_dir": args.main_output_dir,
        "created_at": datetime.utcnow().isoformat(),
        "protocol": protocol,
        "initial_manifest": os.path.join(args.suite_dir, "manifests", "strict30_manifest_initial.json"),
    }

    experiment_rows = {}
    gate_summary = defaultdict(int)

    for i in range(1, 8):
        exp_id = f"{i:03d}"
        def_path = get_def_path(args.suite_dir, exp_id)
        sampled_path = get_sampled_path(args.suite_dir, exp_id)

        expected_counts = expected_counts_from_manifest(initial_manifest, exp_id, sample_per_hop=protocol["sample_per_hop"])
        expected_raw_counts = expected_raw_counts_from_manifest(initial_manifest, exp_id)

        definition_gate = audit_definition(
            def_path=def_path,
            exp_id=exp_id,
            relation_expected=relation_map.get(exp_id, DEFAULT_RELATION_MAP[exp_id]),
            popularity_expected=popularity_map.get(exp_id, DEFAULT_POPULARITY_MAP[exp_id]),
            min_per_hop=protocol["min_per_hop"],
            strict_hops=protocol["strict_hops"],
            relaxed_hops=protocol["relaxed_hops"],
            expected_raw_counts=expected_raw_counts,
        )
        sampled_gate = audit_sampled(sampled_path, expected_counts=expected_counts)

        meta_path = find_latest_training_meta(args.main_output_dir, exp_id)
        training_gate = audit_training(meta_path)

        main_report_path = pick_main_report(all_reports, sampled_path, exp_id)
        main_report_gate = audit_main_report(main_report_path, expected_counts=expected_counts)

        sanity_report_path = pick_sanity_report(all_reports, irrelevant_file, exp_id)
        sanity_gate = audit_sanity_report(sanity_report_path)

        gates = {
            "definition": definition_gate,
            "sampled": sampled_gate,
            "training": training_gate,
            "main_report": main_report_gate,
            "sanity_report": sanity_gate,
        }
        actions = decide_actions(gates)
        for a in actions:
            gate_summary[a] += 1

        experiment_rows[exp_id] = {
            "relation_expected": relation_map.get(exp_id, DEFAULT_RELATION_MAP[exp_id]),
            "popularity_expected": popularity_map.get(exp_id, DEFAULT_POPULARITY_MAP[exp_id]),
            "paths": {"definition_file": def_path, "sampled_file": sampled_path},
            "expected_sampled_counts": expected_counts,
            "gates": gates,
            "actions": actions,
            "status": "ok" if actions == ["ok"] else "failed",
        }

    payload = {
        "suite": suite,
        "gate_summary": dict(gate_summary),
        "experiments": experiment_rows,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    write_rerun_script(out_rerun_script, payload, args.suite_dir)
    print(f"Saved audit manifest: {out_json}")
    print(f"Saved rerun script: {out_rerun_script}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
