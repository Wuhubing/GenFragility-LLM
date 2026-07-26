"""Build and finalize a frozen graph holdout probe bank."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from select_anchors_v2_matched import DEFAULT_GRAPH, build_fact_index, load_graph


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "data/external_eval/rehearsal_graph_probe"


def stable_key(seed: int, *parts: object) -> bytes:
    value = "|".join([str(seed), *map(str, parts)])
    return hashlib.sha256(value.encode()).digest()


def load_excluded_entities(manifest_paths: list[Path]) -> set[str]:
    excluded = set()
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text())
        for unit in manifest["units"].values():
            for update in unit["updates"]:
                for field in ("head", "tail", "poison_answer"):
                    value = update.get(field)
                    if value not in (None, ""):
                        excluded.add(str(value))
    return excluded


def choose_fact(
    facts: list[dict],
    excluded_entities: set[str],
    seed: int,
    stratum: str,
) -> dict | None:
    eligible = [
        fact
        for fact in facts
        if fact.get("question")
        and fact["head"] not in excluded_entities
        and fact["tail"] not in excluded_entities
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda fact: stable_key(
            seed,
            stratum,
            fact["head"],
            fact["relation"],
            fact["tail"],
        ),
    )


def object_strata(objects: list[str], graph, candidate_count: int) -> dict[str, list[str]]:
    ranked = sorted(objects, key=lambda obj: (graph.in_degree(obj), obj))
    if len(ranked) < candidate_count * 3:
        raise RuntimeError("Graph has too few objects for disjoint probe strata")
    middle_start = max(0, len(ranked) // 2 - candidate_count // 2)
    return {
        "rare": ranked[:candidate_count],
        "middle": ranked[middle_start : middle_start + candidate_count],
        "popular": list(reversed(ranked[-candidate_count:])),
    }


def build_candidates(
    graph,
    facts_by_object: dict,
    excluded_entities: set[str],
    candidate_count: int,
    seed: int,
) -> dict:
    probes = []
    used_facts = set()
    for stratum, objects in object_strata(
        list(facts_by_object), graph, candidate_count * 2
    ).items():
        selected = 0
        for obj in objects:
            fact = choose_fact(
                facts_by_object[obj],
                excluded_entities,
                seed,
                stratum,
            )
            if fact is None:
                continue
            identity = (fact["head"], fact["relation"], fact["tail"])
            if identity in used_facts:
                continue
            used_facts.add(identity)
            probe_id = hashlib.sha256(
                "|".join(identity).encode()
            ).hexdigest()[:16]
            probes.append(
                {
                    "probe_id": f"{stratum}_{probe_id}",
                    "stratum": stratum,
                    **fact,
                }
            )
            selected += 1
            if selected == candidate_count:
                break
        if selected != candidate_count:
            raise RuntimeError(
                f"{stratum}: generated {selected}/{candidate_count} candidates"
            )
    return {
        "metadata": {
            "stage": "candidates",
            "seed": seed,
            "candidate_count_per_stratum": candidate_count,
            "graph_path": str(DEFAULT_GRAPH),
            "mask": "pending_strict_clean_correct",
        },
        "probes": probes,
    }


def finalize(
    candidates: dict,
    precheck: dict,
    n_per_stratum: int,
    seed: int,
) -> dict:
    correctness = {
        row["probe_id"]: row["is_correct"] for row in precheck["results"]
    }
    selected = []
    used_entities = set()
    for stratum in ("popular", "middle", "rare"):
        eligible = [
            probe
            for probe in candidates["probes"]
            if probe["stratum"] == stratum
            and correctness.get(probe["probe_id"], False)
        ]
        eligible.sort(
            key=lambda probe: stable_key(seed, "final", probe["probe_id"])
        )
        stratum_selected = []
        for probe in eligible:
            entities = {probe["head"], probe["tail"]}
            if entities & used_entities:
                continue
            stratum_selected.append(probe)
            used_entities.update(entities)
            if len(stratum_selected) == n_per_stratum:
                break
        if len(stratum_selected) != n_per_stratum:
            raise RuntimeError(
                f"{stratum}: only {len(stratum_selected)}/{n_per_stratum} "
                "clean-correct entity-disjoint probes"
            )
        selected.extend(stratum_selected)
    return {
        "metadata": {
            "stage": "final",
            "seed": seed,
            "n_per_stratum": n_per_stratum,
            "total_probes": len(selected),
            "mask": "strict_short_answer_clean_correct",
            "base_model": precheck["metadata"]["base_model"],
            "graph_path": candidates["metadata"]["graph_path"],
        },
        "probes": selected,
    }


def write_audit(path: Path, bank: dict, graph) -> None:
    failures = []
    triplets = set()
    entities = set()
    enforce_entity_disjoint = bank["metadata"]["stage"] == "final"
    lines = [
        "# Rehearsal Graph Probe Audit",
        "",
        f"- Status: PASS",
        f"- Total probes: {len(bank['probes'])}",
        f"- Mask: `{bank['metadata']['mask']}`",
        "",
        "| Stratum | N | Degree min/median/mean/max | Relations |",
        "|---|---:|---:|---:|",
    ]
    for stratum in ("popular", "middle", "rare"):
        probes = [p for p in bank["probes"] if p["stratum"] == stratum]
        degrees = [graph.in_degree(p["tail"]) for p in probes]
        relations = Counter(p["relation"] for p in probes)
        for probe in probes:
            identity = (probe["head"], probe["relation"], probe["tail"])
            endpoints = {probe["head"], probe["tail"]}
            if identity in triplets:
                failures.append(f"duplicate triplet: {probe['probe_id']}")
            if enforce_entity_disjoint and endpoints & entities:
                failures.append(f"entity overlap: {probe['probe_id']}")
            if not graph.has_edge(probe["head"], probe["tail"]):
                failures.append(f"missing graph edge: {probe['probe_id']}")
            triplets.add(identity)
            entities.update(endpoints)
        lines.append(
            f"| {stratum} | {len(probes)} | {min(degrees)} / "
            f"{statistics.median(degrees):.1f} / "
            f"{statistics.mean(degrees):.1f} / {max(degrees)} | "
            f"{len(relations)} |"
        )
    if failures:
        lines[2] = "- Status: FAIL"
        lines.extend(["", "## Failures", *[f"- {item}" for item in failures]])
    path.write_text("\n".join(lines) + "\n")
    if failures:
        raise RuntimeError(f"Probe audit failed with {len(failures)} violations")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("candidates", "finalize"), required=True)
    parser.add_argument("--graph-path", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--candidate-file", type=Path)
    parser.add_argument("--precheck-report", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--candidate-count", type=int, default=2000)
    parser.add_argument("--n-per-stratum", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    graph = load_graph(args.graph_path)
    if args.stage == "candidates":
        bank = build_candidates(
            graph,
            build_fact_index(graph),
            load_excluded_entities(args.exclude_manifest),
            args.candidate_count,
            args.seed,
        )
        output = args.out_dir / "probe_candidates.json"
    else:
        if args.candidate_file is None or args.precheck_report is None:
            parser.error("finalize requires --candidate-file and --precheck-report")
        bank = finalize(
            json.loads(args.candidate_file.read_text()),
            json.loads(args.precheck_report.read_text()),
            args.n_per_stratum,
            args.seed,
        )
        output = args.out_dir / "probe_bank.json"
    output.write_text(json.dumps(bank, indent=2, ensure_ascii=False) + "\n")
    write_audit(args.out_dir / f"{output.stem}_audit.md", bank, graph)
    print(f"Wrote {output}: probes={len(bank['probes'])}")


if __name__ == "__main__":
    main()
