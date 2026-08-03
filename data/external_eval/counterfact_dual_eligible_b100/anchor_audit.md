# Rehearsal Smoke Anchor Audit

- Dataset: `counterfact`
- Manifest: `data/external_eval/counterfact_dual_eligible_b100/manifest.json`
- Units: 1
- Anchors per non-empty mode and unit: 100
- Update-only anchors per unit: 0
- Status: FAIL

| Mode | Units | Anchors | Object degree min/median/mean/max | Unique relations | Mean text length |
|---|---:|---:|---:|---:|---:|
| popular | 1 | 100 | 145 / 218.0 / 259.2 / 776 | 12 | 44.4 |
| rare | 1 | 100 | 1 / 1.0 / 1.0 / 1 | 14 | 51.5 |
| random | 1 | 100 | 2 / 3.0 / 7.0 / 114 | 16 | 54.3 |
| random_distance | 1 | 100 | 2 / 3.0 / 9.5 / 121 | 14 | 53.4 |
| generic | 1 | 100 | 1 / 2.0 / 4.3 / 91 | 12 | 55.5 |

## Holdout probe isolation

| Mode | One-hop overlaps | Min distance | Median distance |
|---|---:|---:|---:|
| popular | 85 | 1 | 1.0 |
| rare | 11 | 1 | 2.0 |
| random | 19 | 1 | 2.0 |
| random_distance | 85 | 1 | 1.0 |
| generic | 12 | 1 | 2.0 |

## Exclusion counts

| Unit | Excluded entities | Excluded relations |
|---|---:|---:|
| `counterfact_batch_001` | 30158 | 39 |

## Failures

- counterfact_batch_001: random_distance/generic object overlap
