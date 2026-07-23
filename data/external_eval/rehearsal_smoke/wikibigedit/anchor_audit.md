# Rehearsal Smoke Anchor Audit

- Dataset: `wikibigedit`
- Manifest: `data/external_eval/rehearsal_smoke/wikibigedit/manifest.json`
- Units: 1
- Anchors per non-empty mode and unit: 25
- Update-only anchors per unit: 0
- Status: PASS

| Mode | Units | Anchors | Object degree min/median/mean/max | Unique relations | Mean text length |
|---|---:|---:|---:|---:|---:|
| popular | 1 | 25 | 704 / 1093.0 / 2371.6 / 17029 | 6 | 42.8 |
| rare | 1 | 25 | 1 / 1.0 / 1.0 / 1 | 12 | 58.8 |
| random | 1 | 25 | 2 / 3.0 / 4.6 / 18 | 11 | 52.2 |

## Exclusion counts

| Unit | Excluded entities | Excluded relations |
|---|---:|---:|
| `wikibigedit_20240201_20240220_batch_001` | 32 | 20 |
