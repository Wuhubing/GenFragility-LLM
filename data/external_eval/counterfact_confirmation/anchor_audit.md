# Rehearsal Smoke Anchor Audit

- Dataset: `counterfact`
- Manifest: `data/external_eval/counterfact_confirmation/manifest.json`
- Units: 3
- Anchors per non-empty mode and unit: 100
- Update-only anchors per unit: 0
- Status: PASS

| Mode | Units | Anchors | Object degree min/median/mean/max | Unique relations | Mean text length |
|---|---:|---:|---:|---:|---:|
| popular | 3 | 300 | 138 / 218.0 / 254.4 / 776 | 14 | 43.9 |
| rare | 3 | 300 | 1 / 1.0 / 1.0 / 1 | 16 | 54.4 |
| random | 3 | 300 | 2 / 3.0 / 5.2 / 99 | 18 | 56.5 |
| random_distance | 3 | 300 | 2 / 3.0 / 8.2 / 131 | 15 | 53.2 |
| generic | 3 | 300 | 1 / 2.0 / 3.0 / 97 | 14 | 54.1 |

## Holdout probe isolation

| Mode | One-hop overlaps | Min distance | Median distance |
|---|---:|---:|---:|
| popular | 249 | 1 | 1.0 |
| rare | 23 | 1 | 2.0 |
| random | 40 | 1 | 2.0 |
| random_distance | 249 | 1 | 1.0 |
| generic | 25 | 1 | 2.0 |

## Exclusion counts

| Unit | Excluded entities | Excluded relations |
|---|---:|---:|
| `counterfact_batch_001` | 27559 | 40 |
| `counterfact_batch_002` | 28079 | 38 |
| `counterfact_batch_003` | 31325 | 38 |
