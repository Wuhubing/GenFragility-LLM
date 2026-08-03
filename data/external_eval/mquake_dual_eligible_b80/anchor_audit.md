# Rehearsal Smoke Anchor Audit

- Dataset: `mquake_cf`
- Manifest: `data/external_eval/mquake_dual_eligible_b80/manifest.json`
- Units: 1
- Anchors per non-empty mode and unit: 72
- Update-only anchors per unit: 0
- Status: PASS

| Mode | Units | Anchors | Object degree min/median/mean/max | Unique relations | Mean text length |
|---|---:|---:|---:|---:|---:|
| popular | 1 | 72 | 232 / 311.5 / 378.1 / 1801 | 10 | 44.6 |
| rare | 1 | 72 | 1 / 1.0 / 1.0 / 1 | 11 | 52.8 |
| random | 1 | 72 | 2 / 3.0 / 5.5 / 82 | 11 | 53.4 |
| random_distance | 1 | 72 | 2 / 4.0 / 14.0 / 216 | 13 | 52.8 |
| generic | 1 | 72 | 1 / 2.0 / 2.7 / 15 | 14 | 57.1 |

## Holdout probe isolation

| Mode | One-hop overlaps | Min distance | Median distance |
|---|---:|---:|---:|
| popular | 67 | 1 | 1.0 |
| rare | 7 | 1 | 2.0 |
| random | 9 | 1 | 2.0 |
| random_distance | 67 | 1 | 1.0 |
| generic | 9 | 1 | 2.0 |

## Exclusion counts

| Unit | Excluded entities | Excluded relations |
|---|---:|---:|
| `mquake_cf_batch_001` | 4809 | 30 |
