# Rehearsal Smoke Anchor Audit

- Dataset: `wikifactdiff`
- Manifest: `data/external_eval/rehearsal_smoke/wikifactdiff/manifest.json`
- Units: 2
- Anchors per non-empty mode and unit: 25
- Update-only anchors per unit: 0
- Status: PASS

| Mode | Units | Anchors | Object degree min/median/mean/max | Unique relations | Mean text length |
|---|---:|---:|---:|---:|---:|
| popular | 2 | 50 | 704 / 1093.0 / 2371.6 / 17029 | 13 | 45.9 |
| rare | 2 | 50 | 1 / 1.0 / 1.0 / 1 | 14 | 51.0 |
| random | 2 | 50 | 2 / 3.5 / 4.2 / 21 | 11 | 54.9 |

## Exclusion counts

| Unit | Excluded entities | Excluded relations |
|---|---:|---:|
| `wfd_Q483020_P286` | 3 | 1 |
| `wfd_Q223620_P118` | 3 | 1 |
