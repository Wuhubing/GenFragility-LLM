# Rehearsal Smoke Anchor Audit

- Dataset: `mquake_cf`
- Manifest: `data/external_eval/mquake_b100_confirmation/manifest.json`
- Units: 1
- Anchors per non-empty mode and unit: 80
- Update-only anchors per unit: 0
- Status: PASS

| Mode | Units | Anchors | Object degree min/median/mean/max | Unique relations | Mean text length |
|---|---:|---:|---:|---:|---:|
| popular | 1 | 80 | 210 / 302.0 / 340.6 / 1297 | 13 | 44.0 |
| rare | 1 | 80 | 1 / 1.0 / 1.0 / 1 | 11 | 51.4 |
| random | 1 | 80 | 2 / 3.5 / 6.5 / 70 | 15 | 55.1 |
| random_distance | 1 | 80 | 2 / 3.5 / 11.8 / 173 | 13 | 53.4 |
| generic | 1 | 80 | 1 / 1.0 / 5.5 / 151 | 13 | 53.5 |

## Holdout probe isolation

| Mode | One-hop overlaps | Min distance | Median distance |
|---|---:|---:|---:|
| popular | 73 | 1 | 1.0 |
| rare | 7 | 1 | 2.0 |
| random | 8 | 1 | 2.0 |
| random_distance | 73 | 1 | 1.0 |
| generic | 10 | 1 | 2.0 |

## Exclusion counts

| Unit | Excluded entities | Excluded relations |
|---|---:|---:|
| `mquake_cf_batch_001` | 4395 | 33 |
