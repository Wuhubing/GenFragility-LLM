# Rehearsal Smoke Anchor Audit

- Dataset: `wikibigedit`
- Manifest: `data/external_eval/rehearsal_popular_graph/wikibigedit/manifest.json`
- Units: 1
- Anchors per non-empty mode and unit: 100
- Update-only anchors per unit: 0
- Status: PASS

| Mode | Units | Anchors | Object degree min/median/mean/max | Unique relations | Mean text length |
|---|---:|---:|---:|---:|---:|
| popular | 1 | 100 | 254 / 355.0 / 681.2 / 17029 | 16 | 45.6 |
| rare | 1 | 100 | 1 / 1.0 / 1.0 / 1 | 13 | 53.0 |
| random | 1 | 100 | 2 / 3.0 / 9.2 / 231 | 15 | 52.5 |
| generic | 1 | 100 | 1 / 1.0 / 3.3 / 85 | 15 | 50.5 |

## Holdout probe isolation

| Mode | One-hop overlaps | Min distance | Median distance |
|---|---:|---:|---:|
| popular | 85 | 1 | 1.0 |
| rare | 5 | 1 | 2.0 |
| random | 10 | 1 | 2.0 |
| generic | 3 | 1 | 2.0 |

## Exclusion counts

| Unit | Excluded entities | Excluded relations |
|---|---:|---:|
| `wikibigedit_20240201_20240220_batch_001` | 4129 | 60 |
