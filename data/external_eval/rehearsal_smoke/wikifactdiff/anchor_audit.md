# Rehearsal Smoke Anchor Audit

- Dataset: `wikifactdiff`
- Manifest: `data/external_eval/rehearsal_smoke/wikifactdiff/manifest.json`
- Units: 1
- Anchors per non-empty mode and unit: 100
- Update-only anchors per unit: 0
- Status: PASS

| Mode | Units | Anchors | Object degree min/median/mean/max | Unique relations | Mean text length |
|---|---:|---:|---:|---:|---:|
| popular | 1 | 100 | 284 / 413.5 / 864.8 / 17029 | 15 | 41.9 |
| rare | 1 | 100 | 1 / 1.0 / 1.0 / 1 | 17 | 51.8 |
| random | 1 | 100 | 2 / 3.0 / 4.5 / 21 | 21 | 54.7 |

## Exclusion counts

| Unit | Excluded entities | Excluded relations |
|---|---:|---:|
| `wikifactdiff_batch_001` | 48 | 2 |
