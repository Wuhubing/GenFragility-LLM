# Rehearsal Smoke Anchor Audit

- Dataset: `wikibigedit`
- Manifest: `data/external_eval/rehearsal_smoke/wikibigedit/manifest.json`
- Units: 1
- Anchors per non-empty mode and unit: 100
- Update-only anchors per unit: 0
- Status: PASS

| Mode | Units | Anchors | Object degree min/median/mean/max | Unique relations | Mean text length |
|---|---:|---:|---:|---:|---:|
| popular | 1 | 100 | 292 / 429.5 / 898.4 / 17029 | 16 | 43.4 |
| rare | 1 | 100 | 1 / 1.0 / 1.0 / 1 | 18 | 53.8 |
| random | 1 | 100 | 2 / 3.0 / 5.6 / 147 | 17 | 60.2 |

## Exclusion counts

| Unit | Excluded entities | Excluded relations |
|---|---:|---:|
| `wikibigedit_20240201_20240220_batch_001` | 121 | 18 |
