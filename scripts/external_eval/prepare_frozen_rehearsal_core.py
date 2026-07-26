"""Build and verify a fixed Mask-B rehearsal anchor core."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

from select_anchors_v2_matched import DEFAULT_GRAPH, build_fact_index, load_graph


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "data/external_eval/frozen_rehearsal_core"
MODES = ("popular", "random", "rare", "random_distance")


def stable_key(seed: int, *parts: object) -> bytes:
    return hashlib.sha256(
        "|".join([str(seed), *map(str, parts)]).encode()
    ).digest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_excluded_entities(paths: list[Path]) -> set[str]:
    entities = set()
    for path in paths:
        manifest = json.loads(path.read_text())
        for unit in manifest["units"].values():
            for update in unit["updates"]:
                for field in (
                    "head",
                    "head_qid",
                    "tail",
                    "tail_qid",
                    "poison_answer",
                ):
                    value = update.get(field)
                    if value not in (None, ""):
                        entities.add(str(value))
    return entities


def probe_entities(path: Path) -> set[str]:
    bank = json.loads(path.read_text())
    return {
        entity
        for probe in bank["probes"]
        for entity in (probe["head"], probe["tail"])
    }


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


def object_pools(objects: list[str], graph) -> dict[str, list[str]]:
    ranked = sorted(objects, key=lambda obj: (graph.in_degree(obj), obj))
    third = len(ranked) // 3
    return {
        "rare": ranked[: len(ranked) // 2],
        "random": ranked[third : 2 * third],
        "popular": list(reversed(ranked[2 * third :])),
    }


def build_candidates(
    graph,
    facts_by_object: dict,
    excluded_entities: set[str],
    candidate_counts: dict[str, int],
    seed: int,
    probe_bank: Path,
    exclusion_manifests: list[Path],
) -> dict:
    candidates = []
    for stratum, objects in object_pools(list(facts_by_object), graph).items():
        candidate_count = candidate_counts[stratum]
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
            candidates.append(
                {
                    "probe_id": (
                        f"{stratum}_"
                        f"{hashlib.sha256('|'.join(identity).encode()).hexdigest()[:16]}"
                    ),
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
            "stage": "anchor_candidates",
            "seed": seed,
            "candidate_count_per_stratum": candidate_counts,
            "graph_path": str(DEFAULT_GRAPH),
            "probe_bank": str(probe_bank),
            "exclusion_manifests": [str(path) for path in exclusion_manifests],
            "mask": "pending_strict_clean_correct",
        },
        "probes": candidates,
    }


def relation_matched(
    candidates: list[dict],
    relation_counts: Counter,
    used_entities: set[str],
    seed: int,
    mode: str,
) -> tuple[list[dict], int]:
    selected = []
    by_relation: dict[str, list[dict]] = defaultdict(list)
    for fact in candidates:
        by_relation[fact["relation"]].append(fact)
    matched = 0
    for relation, count in relation_counts.items():
        ordered = sorted(
            by_relation[relation],
            key=lambda fact: stable_key(seed, mode, fact["probe_id"]),
        )
        relation_selected = 0
        for fact in ordered:
            entities = {fact["head"], fact["tail"]}
            if entities & used_entities:
                continue
            selected.append(fact)
            used_entities.update(entities)
            relation_selected += 1
            if relation_selected == count:
                break
        matched += relation_selected
    target_count = sum(relation_counts.values())
    if len(selected) < target_count:
        ordered = sorted(
            candidates,
            key=lambda fact: stable_key(seed, mode, "fallback", fact["probe_id"]),
        )
        selected_ids = {fact["probe_id"] for fact in selected}
        for fact in ordered:
            entities = {fact["head"], fact["tail"]}
            if fact["probe_id"] in selected_ids or entities & used_entities:
                continue
            selected.append(fact)
            selected_ids.add(fact["probe_id"])
            used_entities.update(entities)
            if len(selected) == target_count:
                break
    if len(selected) != target_count:
        raise RuntimeError(f"{mode}: selected {len(selected)}/{target_count}")
    return selected, matched


def distance_matched(
    candidates: list[dict],
    target_distances: Counter,
    distance_map: dict[str, int],
    used_entities: set[str],
    seed: int,
) -> list[dict]:
    by_distance: dict[int, list[dict]] = defaultdict(list)
    for fact in candidates:
        distance = min(
            distance_map.get(fact["head"], 6),
            distance_map.get(fact["tail"], 6),
        )
        by_distance[distance].append(fact)
    selected = []
    for distance, count in sorted(target_distances.items()):
        ordered = sorted(
            by_distance[distance],
            key=lambda fact: stable_key(
                seed,
                "random-distance",
                fact["probe_id"],
            ),
        )
        distance_selected = 0
        for fact in ordered:
            entities = {fact["head"], fact["tail"]}
            if entities & used_entities:
                continue
            selected.append(fact)
            used_entities.update(entities)
            distance_selected += 1
            if distance_selected == count:
                break
        if distance_selected != count:
            raise RuntimeError(
                f"random_distance/d={distance}: "
                f"selected {distance_selected}/{count}"
            )
    return selected


def clean_fact(fact: dict) -> dict:
    return {
        key: value
        for key, value in fact.items()
        if key not in {"probe_id", "stratum"}
    }


def finalize(
    graph,
    candidates: dict,
    prechecks: list[dict],
    probe_bank: Path,
    n: int,
    seed: int,
) -> dict[str, list[dict]]:
    correctness_maps = [
        {row["probe_id"]: row["is_correct"] for row in precheck["results"]}
        for precheck in prechecks
    ]
    pools = {
        stratum: [
            fact
            for fact in candidates["probes"]
            if fact["stratum"] == stratum
            and all(
                correctness.get(fact["probe_id"], False)
                for correctness in correctness_maps
            )
        ]
        for stratum in ("popular", "random", "rare")
    }
    popular = []
    relation_counts = Counter()
    used_entities = set()
    for fact in sorted(
        pools["popular"],
        key=lambda item: (
            -graph.in_degree(item["tail"]),
            stable_key(seed, "popular-final", item["probe_id"]),
        ),
    ):
        relation = fact["relation"]
        entities = {fact["head"], fact["tail"]}
        if entities & used_entities:
            continue
        popular.append(fact)
        relation_counts[relation] += 1
        used_entities.update(entities)
        if len(popular) == n:
            break
    if len(popular) != n:
        raise RuntimeError(f"Popular selected {len(popular)}/{n}")

    rare, _ = relation_matched(
        pools["rare"],
        relation_counts,
        used_entities,
        seed,
        "rare",
    )
    bank = json.loads(probe_bank.read_text())
    sources = [
        entity
        for probe in bank["probes"]
        for entity in (probe["head"], probe["tail"])
        if entity in graph
    ]
    distance_map = nx.multi_source_dijkstra_path_length(
        graph.to_undirected(as_view=True),
        sources,
        cutoff=5,
        weight=None,
    )
    target_distances = Counter(
        min(
            distance_map.get(fact["head"], 6),
            distance_map.get(fact["tail"], 6),
        )
        for fact in popular
    )
    random_distance = distance_matched(
        pools["random"],
        target_distances,
        distance_map,
        used_entities,
        seed,
    )
    random_facts, _ = relation_matched(
        pools["random"],
        relation_counts,
        used_entities,
        seed,
        "random",
    )
    return {
        "popular": [clean_fact(fact) for fact in popular],
        "random": [clean_fact(fact) for fact in random_facts],
        "rare": [clean_fact(fact) for fact in rare],
        "random_distance": [
            clean_fact(fact) for fact in random_distance
        ],
    }


def write_frozen(
    out_dir: Path,
    anchors: dict[str, list[dict]],
    graph,
    probe_bank: Path,
    base_model: str,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    validation_rows = []
    files = {}
    for mode, facts in anchors.items():
        path = out_dir / f"anchors_{mode}_100.json"
        data = {
            "metadata": {
                "status": "frozen",
                "mode": mode,
                "N": len(facts),
                "base_model": base_model,
                "seed": seed,
                "graph_path": str(DEFAULT_GRAPH),
                "probe_bank": str(probe_bank),
                "selection_mask": "strict_short_answer_clean_correct",
            },
            "anchors": facts,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        files[mode] = path
        for index, fact in enumerate(facts):
            validation_rows.append(
                {
                    "probe_id": f"{mode}_{index:03d}",
                    "stratum": mode,
                    **fact,
                }
            )
    validation = {
        "metadata": {
            "stage": "frozen_anchor_validation",
            "total_probes": len(validation_rows),
        },
        "probes": validation_rows,
    }
    (out_dir / "frozen_anchor_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n"
    )
    hash_manifest = {
        "probe_bank": {
            "path": str(probe_bank),
            "sha256": file_sha256(probe_bank),
        },
        "anchors": {
            mode: {"path": str(path), "sha256": file_sha256(path)}
            for mode, path in files.items()
        },
    }
    (out_dir / "frozen_hashes.json").write_text(
        json.dumps(hash_manifest, indent=2) + "\n"
    )
    lines = [
        "# Frozen Rehearsal Core Audit",
        "",
        "- Status: PENDING INDEPENDENT RECHECK",
        f"- Base model: `{base_model}`",
        f"- Probe bank: `{probe_bank}`",
        "",
        "| Mode | N | Degree min/median/mean/max | Relations | "
        "Prompt chars | Answer words |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, facts in anchors.items():
        degrees = [graph.in_degree(fact["tail"]) for fact in facts]
        lines.append(
            f"| {mode} | {len(facts)} | {min(degrees)} / "
            f"{statistics.median(degrees):.1f} / "
            f"{statistics.mean(degrees):.1f} / {max(degrees)} | "
            f"{len({fact['relation'] for fact in facts})} | "
            f"{statistics.mean(len(fact['question']) for fact in facts):.1f} | "
            f"{statistics.mean(len(str(fact['tail']).split()) for fact in facts):.1f} |"
        )
    (out_dir / "frozen_anchor_audit.md").write_text(
        "\n".join(lines) + "\n"
    )


def verify(out_dir: Path, report_path: Path) -> None:
    report = json.loads(report_path.read_text())
    failures = [row for row in report["results"] if not row["is_correct"]]
    hashes = json.loads((out_dir / "frozen_hashes.json").read_text())
    hash_failures = []
    for mode, item in hashes["anchors"].items():
        if file_sha256(Path(item["path"])) != item["sha256"]:
            hash_failures.append(mode)
    probe = hashes["probe_bank"]
    if file_sha256(Path(probe["path"])) != probe["sha256"]:
        hash_failures.append("probe_bank")
    status = "PASS" if not failures and not hash_failures else "FAIL"
    lines = [
        "# Frozen Rehearsal Core Verification",
        "",
        f"- Status: {status}",
        f"- Rechecked anchors: {len(report['results'])}",
        f"- Strict clean-correct: {len(report['results']) - len(failures)}",
        f"- Answer failures: {len(failures)}",
        f"- Hash failures: {len(hash_failures)}",
    ]
    (out_dir / "frozen_verification.md").write_text("\n".join(lines) + "\n")
    if status != "PASS":
        raise RuntimeError(
            f"Frozen core verification failed: answers={len(failures)}, "
            f"hashes={hash_failures}"
        )
    audit_path = out_dir / "frozen_anchor_audit.md"
    audit = audit_path.read_text().replace(
        "- Status: PENDING INDEPENDENT RECHECK",
        "- Status: PASS",
    )
    audit_path.write_text(audit)
    print(f"PASS: independently rechecked {len(report['results'])} anchors")


def repair(
    graph,
    out_dir: Path,
    candidates_path: Path,
    precheck_paths: list[Path],
    verification_report: Path,
    probe_bank: Path,
    seed: int,
) -> None:
    anchors = {
        mode: json.loads(
            (out_dir / f"anchors_{mode}_100.json").read_text()
        )["anchors"]
        for mode in MODES
    }
    report = json.loads(verification_report.read_text())
    failed_ids = [
        row["probe_id"] for row in report["results"] if not row["is_correct"]
    ]
    candidates = json.loads(candidates_path.read_text())["probes"]
    prechecks = [
        json.loads(path.read_text()) for path in precheck_paths
    ]
    correctness_maps = [
        {row["probe_id"]: row["is_correct"] for row in item["results"]}
        for item in prechecks
    ]
    stable_candidates = [
        fact
        for fact in candidates
        if all(
            correctness.get(fact["probe_id"], False)
            for correctness in correctness_maps
        )
    ]
    failed_slots = []
    for probe_id in failed_ids:
        mode, index_text = probe_id.rsplit("_", 1)
        failed_slots.append((mode, int(index_text)))
    failed_locations = set(failed_slots)
    rejected_path = out_dir / "rejected_anchor_identities.json"
    rejected = (
        {
            tuple(identity)
            for identity in json.loads(rejected_path.read_text())
        }
        if rejected_path.exists()
        else set()
    )
    for mode, index in failed_slots:
        old = anchors[mode][index]
        rejected.add((old["head"], old["relation"], old["tail"]))
    used_entities = {
        entity
        for mode, facts in anchors.items()
        for index, fact in enumerate(facts)
        if (mode, index) not in failed_locations
        for entity in (fact["head"], fact["tail"])
    }
    used_identities = {
        (fact["head"], fact["relation"], fact["tail"])
        for mode, facts in anchors.items()
        for index, fact in enumerate(facts)
        if (mode, index) not in failed_locations
    }
    bank = json.loads(probe_bank.read_text())
    sources = [
        entity
        for probe in bank["probes"]
        for entity in (probe["head"], probe["tail"])
        if entity in graph
    ]
    distance_map = nx.multi_source_dijkstra_path_length(
        graph.to_undirected(as_view=True),
        sources,
        cutoff=5,
        weight=None,
    )

    def distance(fact: dict) -> int:
        return min(
            distance_map.get(fact["head"], 6),
            distance_map.get(fact["tail"], 6),
        )

    for mode, index in failed_slots:
        old = anchors[mode][index]
        stratum = "random" if mode == "random_distance" else mode
        old_distance = distance(old)
        eligible = []
        for fact in stable_candidates:
            identity = (fact["head"], fact["relation"], fact["tail"])
            entities = {fact["head"], fact["tail"]}
            if (
                fact["stratum"] != stratum
                or identity in used_identities
                or identity in rejected
                or entities & used_entities
            ):
                continue
            if mode in {"popular", "random_distance"} and distance(fact) != old_distance:
                continue
            eligible.append(fact)
        same_relation = [
            fact for fact in eligible if fact["relation"] == old["relation"]
        ]
        pool = same_relation or eligible
        if not pool:
            raise RuntimeError(f"No stable replacement for {mode}_{index:03d}")
        if mode == "popular":
            replacement = min(
                pool,
                key=lambda fact: (
                    -graph.in_degree(fact["tail"]),
                    stable_key(seed, "repair", mode, index, fact["probe_id"]),
                ),
            )
        elif mode == "rare":
            replacement = min(
                pool,
                key=lambda fact: (
                    graph.in_degree(fact["tail"]),
                    stable_key(seed, "repair", mode, index, fact["probe_id"]),
                ),
            )
        else:
            replacement = min(
                pool,
                key=lambda fact: stable_key(
                    seed,
                    "repair",
                    mode,
                    index,
                    fact["probe_id"],
                ),
            )
        cleaned = clean_fact(replacement)
        anchors[mode][index] = cleaned
        used_entities.update((cleaned["head"], cleaned["tail"]))
        used_identities.add(
            (cleaned["head"], cleaned["relation"], cleaned["tail"])
        )
    write_frozen(
        out_dir,
        anchors,
        graph,
        probe_bank,
        report["metadata"]["base_model"],
        seed,
    )
    history = out_dir / "repair_history.md"
    previous = history.read_text() if history.exists() else "# Repair History\n"
    history.write_text(
        previous
        + f"\n- Replaced {len(failed_slots)} unstable anchors: "
        + ", ".join(failed_ids)
        + "\n"
    )
    rejected_path.write_text(
        json.dumps(sorted(rejected), indent=2, ensure_ascii=False) + "\n"
    )
    print(f"Replaced {len(failed_slots)} unstable anchors")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("candidates", "finalize", "repair", "verify"),
        required=True,
    )
    parser.add_argument("--graph-path", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--probe-bank", type=Path, required=True)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--candidate-file", type=Path)
    parser.add_argument(
        "--precheck-report",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--verification-report", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--candidate-count", type=int, default=6000)
    parser.add_argument("--random-candidate-count", type=int, default=15000)
    parser.add_argument("--rare-candidate-count", type=int, default=43000)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    graph = load_graph(args.graph_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "candidates":
        excluded = load_excluded_entities(args.exclude_manifest)
        excluded.update(probe_entities(args.probe_bank))
        data = build_candidates(
            graph,
            build_fact_index(graph),
            excluded,
            {
                "popular": args.candidate_count,
                "random": args.random_candidate_count,
                "rare": args.rare_candidate_count,
            },
            args.seed,
            args.probe_bank,
            args.exclude_manifest,
        )
        output = args.out_dir / "anchor_candidates.json"
        output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        print(f"Wrote {output}: candidates={len(data['probes'])}")
        return
    if args.stage == "verify":
        if args.verification_report is None:
            parser.error("verify requires --verification-report")
        verify(args.out_dir, args.verification_report)
        return
    if args.stage == "repair":
        if (
            args.candidate_file is None
            or not args.precheck_report
            or args.verification_report is None
        ):
            parser.error(
                "repair requires --candidate-file, --precheck-report, "
                "and --verification-report"
            )
        repair(
            graph,
            args.out_dir,
            args.candidate_file,
            args.precheck_report,
            args.verification_report,
            args.probe_bank,
            args.seed,
        )
        return
    if args.candidate_file is None or not args.precheck_report:
        parser.error("finalize requires --candidate-file and --precheck-report")
    candidates = json.loads(args.candidate_file.read_text())
    prechecks = [
        json.loads(precheck_path.read_text())
        for precheck_path in args.precheck_report
    ]
    anchors = finalize(
        graph,
        candidates,
        prechecks,
        args.probe_bank,
        args.n,
        args.seed,
    )
    write_frozen(
        args.out_dir,
        anchors,
        graph,
        args.probe_bank,
        prechecks[0]["metadata"]["base_model"],
        args.seed,
    )
    print(f"Wrote frozen rehearsal core to {args.out_dir}")


if __name__ == "__main__":
    main()
