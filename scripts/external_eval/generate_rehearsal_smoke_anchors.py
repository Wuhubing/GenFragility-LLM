"""Generate and structurally audit matched rehearsal anchors for smoke manifests."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path

import networkx as nx

from select_anchors_v2_matched import (
    DEFAULT_GRAPH,
    build_fact_index,
    choose_fact,
    load_graph,
    ranked_objects,
)


WIKIDATA_TO_GRAPH_RELATIONS = {
    "P17": {"CountryOfCity", "HeadquartersCountry", "CountryOfIncorporation"},
    "P19": {"BirthPlace"},
    "P27": {"NationalityPrimary"},
    "P36": {"CapitalCityOfCountry"},
    "P39": {"CurrentPosition"},
    "P50": {"AuthorOfWorkPrimary"},
    "P69": {"AlmaMaterPrimary"},
    "P108": {"CurrentEmployer"},
    "P112": {"FoundedByPrimary"},
    "P123": {"PublisherPrimary"},
    "P159": {"HeadquartersCity"},
    "P169": {"ChiefExecutiveOfficerCurrent"},
    "P176": {"ManufacturedByPrimary"},
    "P178": {"DevelopedByPrimary"},
    "P275": {"LicensePrimary"},
    "P276": {"HeldInCity"},
    "P277": {"ProgrammingLanguagePrimary"},
    "P306": {"OperatingSystemPrimary"},
    "P414": {"StockExchangePrimary"},
    "P571": {"FoundingDate"},
    "P577": {"PublicationDate", "InitialReleaseDate"},
    "P585": {"OccursOn"},
    "P664": {"HostOrganizationPrimary"},
    "P749": {"ParentOrganization"},
}


def unit_exclusions(graph, unit: dict) -> tuple[set[str], set[str]]:
    entities: set[str] = set()
    relations: set[str] = set()
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
        for field in ("relation", "relation_label"):
            value = update.get(field)
            if value not in (None, ""):
                relations.add(str(value))
        relations.update(
            WIKIDATA_TO_GRAPH_RELATIONS.get(str(update.get("relation")), set())
        )
    direct_entities = set(entities)
    for entity in direct_entities:
        if entity in graph:
            entities.update(graph.predecessors(entity))
            entities.update(graph.successors(entity))
    return entities, relations


def valid_facts(
    facts: list[dict],
    excluded_entities: set[str],
    excluded_relations: set[str],
) -> list[dict]:
    return [
        fact
        for fact in facts
        if fact["head"] not in excluded_entities
        and fact["tail"] not in excluded_entities
        and fact["relation"] not in excluded_relations
    ]


def select_from_objects(
    objects: list[str],
    facts_by_object: dict,
    excluded_entities: set[str],
    excluded_relations: set[str],
    unit_id: str,
    seed: int,
    n: int,
) -> list[dict]:
    selected = []
    for obj in objects:
        facts = valid_facts(
            facts_by_object[obj],
            excluded_entities,
            excluded_relations,
        )
        if not facts:
            continue
        selected.append(choose_fact(facts, seed, unit_id))
        if len(selected) == n:
            break
    return selected


def fact_probe_distance(fact: dict, distance_map: dict[str, int]) -> int:
    return min(
        distance_map.get(fact["head"], 6),
        distance_map.get(fact["tail"], 6),
    )


def select_distance_matched_random(
    objects: list[str],
    facts_by_object: dict,
    excluded_entities: set[str],
    excluded_relations: set[str],
    excluded_objects: set[str],
    distance_map: dict[str, int],
    target_facts: list[dict],
    unit_id: str,
    seed: int,
) -> list[dict]:
    candidates = []
    for obj in objects:
        if obj in excluded_objects:
            continue
        facts = valid_facts(
            facts_by_object[obj],
            excluded_entities,
            excluded_relations,
        )
        if facts:
            candidates.append(
                choose_fact(facts, seed, f"{unit_id}:distance-matched")
            )
    random.Random(f"{seed}:{unit_id}:distance-matched").shuffle(candidates)
    by_distance: dict[int, list[dict]] = {}
    for fact in candidates:
        by_distance.setdefault(fact_probe_distance(fact, distance_map), []).append(
            fact
        )
    targets = Counter(
        fact_probe_distance(fact, distance_map) for fact in target_facts
    )
    selected = []
    for distance, count in sorted(targets.items()):
        available = by_distance.get(distance, [])
        if len(available) < count:
            raise RuntimeError(
                f"{unit_id}: distance {distance} has "
                f"{len(available)}/{count} random candidates"
            )
        selected.extend(available[:count])
    return selected


def select_unit_anchors(
    graph,
    facts_by_object: dict,
    unit_id: str,
    unit: dict,
    seed: int,
    n: int,
    probe_entities: set[str],
) -> tuple[
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    set[str],
    set[str],
]:
    objects = list(facts_by_object)
    excluded_entities, excluded_relations = unit_exclusions(graph, unit)
    excluded_entities.update(probe_entities)

    popular = select_from_objects(
        ranked_objects(objects, graph, unit_id, seed, descending=True),
        facts_by_object,
        excluded_entities,
        excluded_relations,
        unit_id,
        seed,
        n,
    )
    rare = select_from_objects(
        ranked_objects(objects, graph, unit_id, seed, descending=False),
        facts_by_object,
        excluded_entities,
        excluded_relations,
        unit_id,
        seed,
        n,
    )
    if min(len(popular), len(rare)) < n:
        raise RuntimeError(f"{unit_id}: fewer than {n} ranked anchors")

    popular_cutoff = min(graph.in_degree(fact["tail"]) for fact in popular)
    rare_cutoff = max(graph.in_degree(fact["tail"]) for fact in rare)
    middle_objects = [
        obj
        for obj in objects
        if rare_cutoff < graph.in_degree(obj) < popular_cutoff
    ]
    random.Random(f"{seed}:{unit_id}:object-v2").shuffle(middle_objects)
    random_middle = select_from_objects(
        middle_objects,
        facts_by_object,
        excluded_entities,
        excluded_relations,
        unit_id,
        seed,
        n,
    )
    if len(random_middle) < n:
        raise RuntimeError(f"{unit_id}: fewer than {n} random-middle anchors")

    used_objects = {
        fact["tail"] for fact in [*popular, *rare, *random_middle]
    }
    generic_objects = [obj for obj in objects if obj not in used_objects]
    random.Random(f"{seed}:{unit_id}:generic").shuffle(generic_objects)
    generic = select_from_objects(
        generic_objects,
        facts_by_object,
        excluded_entities,
        excluded_relations,
        unit_id,
        seed,
        n,
    )
    if len(generic) < n:
        raise RuntimeError(f"{unit_id}: fewer than {n} generic anchors")

    random_distance = []
    if probe_entities:
        distance_map = nx.multi_source_dijkstra_path_length(
            graph.to_undirected(as_view=True),
            [entity for entity in probe_entities if entity in graph],
            cutoff=5,
            weight=None,
        )
        random_distance = select_distance_matched_random(
            middle_objects,
            facts_by_object,
            excluded_entities,
            excluded_relations,
            used_objects,
            distance_map,
            popular,
            unit_id,
            seed,
        )
        if len(random_distance) != n:
            raise RuntimeError(
                f"{unit_id}: fewer than {n} distance-matched random anchors"
            )

    return (
        popular,
        rare,
        random_middle,
        random_distance,
        generic,
        excluded_entities,
        excluded_relations,
    )


def audit_unit(
    graph,
    unit_id: str,
    mode_anchors: dict[str, list[dict]],
    excluded_entities: set[str],
    excluded_relations: set[str],
    n: int,
) -> list[str]:
    failures = []
    object_sets = {}
    degree_lists = {}
    for mode in ("popular", "rare", "random", "random_distance", "generic"):
        anchors = mode_anchors[mode]
        if not anchors and mode == "random_distance":
            continue
        if len(anchors) != n:
            failures.append(f"{unit_id}/{mode}: expected {n}, got {len(anchors)}")
            continue
        objects = [fact["tail"] for fact in anchors]
        object_sets[mode] = set(objects)
        degree_lists[mode] = [graph.in_degree(obj) for obj in objects]
        if len(object_sets[mode]) != n:
            failures.append(f"{unit_id}/{mode}: duplicate objects")
        if len(
            {
                (fact["head"], fact["relation"], fact["tail"])
                for fact in anchors
            }
        ) != n:
            failures.append(f"{unit_id}/{mode}: duplicate facts")
        for fact in anchors:
            if fact["head"] in excluded_entities or fact["tail"] in excluded_entities:
                failures.append(f"{unit_id}/{mode}: target one-hop entity overlap")
            if fact["relation"] in excluded_relations:
                failures.append(f"{unit_id}/{mode}: target relation overlap")
            if not graph.has_edge(fact["head"], fact["tail"]):
                failures.append(f"{unit_id}/{mode}: fact absent from graph")

    if {"popular", "rare", "random"} <= object_sets.keys():
        if min(degree_lists["popular"]) <= max(degree_lists["random"]):
            failures.append(f"{unit_id}: Popular/Random degree strata overlap")
        if max(degree_lists["rare"]) >= min(degree_lists["random"]):
            failures.append(f"{unit_id}: Rare/Random degree strata overlap")
    if {"popular", "random_distance"} <= object_sets.keys():
        if min(degree_lists["popular"]) <= max(degree_lists["random_distance"]):
            failures.append(
                f"{unit_id}: Popular/distance-matched Random degree strata overlap"
            )
    if len(object_sets) >= 4:
        modes = tuple(object_sets)
        for index, left in enumerate(modes):
            for right in modes[index + 1 :]:
                if object_sets[left] & object_sets[right]:
                    failures.append(f"{unit_id}: {left}/{right} object overlap")
    return failures


def mode_summary(mode: str, anchors: dict[str, list[dict]], graph) -> dict:
    facts = [fact for unit_facts in anchors.values() for fact in unit_facts]
    degrees = [graph.in_degree(fact["tail"]) for fact in facts]
    relations = Counter(fact["relation"] for fact in facts)
    text_lengths = [
        len(
            fact.get("surface")
            or fact.get("question")
            or f"{fact['head']} {fact['relation']} {fact['tail']}"
        )
        for fact in facts
    ]
    return {
        "mode": mode,
        "units": len(anchors),
        "anchors": len(facts),
        "degree_min": min(degrees),
        "degree_median": statistics.median(degrees),
        "degree_mean": statistics.mean(degrees),
        "degree_max": max(degrees),
        "unique_relations": len(relations),
        "text_length_mean": statistics.mean(text_lengths),
    }


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--graph-path", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    units = manifest["units"]
    graph = load_graph(args.graph_path)
    facts_by_object = build_fact_index(graph)
    probe_entities = set()
    if args.probe_manifest:
        probe_manifest = json.loads(args.probe_manifest.read_text())
        for probe in probe_manifest["probes"]:
            probe_entities.update((probe["head"], probe["tail"]))

    selected = {
        "popular": {},
        "rare": {},
        "random": {},
        "random_distance": {},
        "generic": {},
        "none": {},
    }
    failures = []
    exclusion_stats = {}
    for unit_id, unit in units.items():
        (
            popular,
            rare,
            random_middle,
            random_distance,
            generic,
            excluded_entities,
            excluded_relations,
        ) = select_unit_anchors(
            graph,
            facts_by_object,
            unit_id,
            unit,
            args.seed,
            args.n,
            probe_entities,
        )
        selected["popular"][unit_id] = popular
        selected["rare"][unit_id] = rare
        selected["random"][unit_id] = random_middle
        selected["random_distance"][unit_id] = random_distance
        selected["generic"][unit_id] = generic
        selected["none"][unit_id] = []
        exclusion_stats[unit_id] = {
            "entities": len(excluded_entities),
            "relations": len(excluded_relations),
        }
        failures.extend(
            audit_unit(
                graph,
                unit_id,
                {
                    "popular": popular,
                    "rare": rare,
                    "random": random_middle,
                    "random_distance": random_distance,
                    "generic": generic,
                },
                excluded_entities,
                excluded_relations,
                args.n,
            )
        )
        if random_distance:
            distance_map = nx.multi_source_dijkstra_path_length(
                graph.to_undirected(as_view=True),
                [entity for entity in probe_entities if entity in graph],
                cutoff=5,
                weight=None,
            )
            popular_distances = Counter(
                fact_probe_distance(fact, distance_map) for fact in popular
            )
            random_distances = Counter(
                fact_probe_distance(fact, distance_map)
                for fact in random_distance
            )
            if popular_distances != random_distances:
                failures.append(
                    f"{unit_id}: distance-matched Random distribution mismatch"
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    unit_key = (
        "per_target"
        if all(unit.get("kind") == "atomic" for unit in units.values())
        else "per_batch"
    )
    common_metadata = {
        "selector_version": "object_matched_v2_batch_aware",
        "ranking_endpoint": "tail",
        "ranking_metric": "in_degree",
        "canonical_fact_selection": "sha256_per_unit_from_incoming_edges",
        "random_pool_rule": "strictly_between_rare_and_popular_degree_strata",
        "N": args.n,
        "seed": args.seed,
        "n_units": len(units),
        "manifest": str(args.manifest),
        "graph_path": str(args.graph_path),
        "unit_key": unit_key,
    }
    filenames = {
        "popular": f"anchors_popular_object_top{args.n}.json",
        "rare": f"anchors_rare_object_bottom{args.n}.json",
        "random": f"anchors_random_object_middle{args.n}_seed{args.seed}.json",
        "random_distance": (
            f"anchors_random_distance_matched_object_middle{args.n}"
            f"_seed{args.seed}.json"
        ),
        "generic": f"anchors_generic_object_{args.n}_seed{args.seed}.json",
        "none": "anchors_none.json",
    }
    for mode, filename in filenames.items():
        write_json(
            args.out_dir / filename,
            {
                "metadata": {**common_metadata, "mode": mode},
                unit_key: selected[mode],
            },
        )

    summaries = [
        mode_summary(mode, selected[mode], graph)
        for mode in (
            "popular",
            "rare",
            "random",
            "random_distance",
            "generic",
        )
        if any(selected[mode].values())
    ]
    probe_distances = {}
    if probe_entities:
        distance_map = nx.multi_source_dijkstra_path_length(
            graph.to_undirected(as_view=True),
            [entity for entity in probe_entities if entity in graph],
            cutoff=5,
            weight=None,
        )
        for mode in (
            "popular",
            "rare",
            "random",
            "random_distance",
            "generic",
        ):
            facts = [
                fact
                for unit_facts in selected[mode].values()
                for fact in unit_facts
            ]
            if not facts:
                continue
            distances = [
                fact_probe_distance(fact, distance_map)
                for fact in facts
            ]
            probe_distances[mode] = {
                "one_hop": sum(distance <= 1 for distance in distances),
                "median": statistics.median(distances),
                "min": min(distances),
            }
    report_lines = [
        "# Rehearsal Smoke Anchor Audit",
        "",
        f"- Dataset: `{manifest['metadata']['dataset']}`",
        f"- Manifest: `{args.manifest}`",
        f"- Units: {len(units)}",
        f"- Anchors per non-empty mode and unit: {args.n}",
        f"- Update-only anchors per unit: 0",
        f"- Status: {'PASS' if not failures else 'FAIL'}",
        "",
        "| Mode | Units | Anchors | Object degree min/median/mean/max | "
        "Unique relations | Mean text length |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        report_lines.append(
            f"| {summary['mode']} | {summary['units']} | "
            f"{summary['anchors']} | {summary['degree_min']} / "
            f"{summary['degree_median']:.1f} / "
            f"{summary['degree_mean']:.1f} / {summary['degree_max']} | "
            f"{summary['unique_relations']} | "
            f"{summary['text_length_mean']:.1f} |"
        )
    if probe_distances:
        report_lines.extend(
            [
                "",
                "## Holdout probe isolation",
                "",
                "| Mode | One-hop overlaps | Min distance | Median distance |",
                "|---|---:|---:|---:|",
            ]
        )
        for mode, stats in probe_distances.items():
            report_lines.append(
                f"| {mode} | {stats['one_hop']} | {stats['min']} | "
                f"{stats['median']:.1f} |"
            )
    report_lines.extend(
        [
            "",
            "## Exclusion counts",
            "",
            "| Unit | Excluded entities | Excluded relations |",
            "|---|---:|---:|",
        ]
    )
    for unit_id, stats in exclusion_stats.items():
        report_lines.append(
            f"| `{unit_id}` | {stats['entities']} | {stats['relations']} |"
        )
    if failures:
        report_lines.extend(["", "## Failures", ""])
        report_lines.extend(f"- {failure}" for failure in failures[:50])
    report_path = args.out_dir / "anchor_audit.md"
    report_path.write_text("\n".join(report_lines) + "\n")

    for summary in summaries:
        print(
            f"{summary['mode']}: units={summary['units']} "
            f"anchors={summary['anchors']} "
            f"degree={summary['degree_min']}/"
            f"{summary['degree_median']:.1f}/{summary['degree_max']} "
            f"relations={summary['unique_relations']}"
        )
    print(f"Wrote {report_path}")
    if failures:
        raise SystemExit(f"Anchor audit failed with {len(failures)} violations")
    print("PASS: rehearsal smoke anchors are structurally valid")


if __name__ == "__main__":
    main()
