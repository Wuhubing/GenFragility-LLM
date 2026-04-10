# Relaxed-Front-30 Workflow

This folder contains automation for the `001-007` rerun protocol with relaxed front hops.

## What it builds

- Regenerated experiment definitions with fixed relation mapping for `001-007`
- Capacity constraint on raw ripples:
  - strict hops (`d3,d4,d5`): `>=30`
  - relaxed hops (`d1,d2`): must exist (`>0`), can be `<30`
- Sampled eval inputs in list format:
  - `d0=1`
  - `d3,d4,d5` each `30`
  - `d1,d2` use `min(available,30)` (no replacement)
- Unified sanity set: `irrelevant_50_strict30.json`
- Initial manifest and capacity precheck table

## Build suite

```bash
/root/miniconda3/envs/genfragility/bin/python tools/strict30/build_strict30_suite.py \
  --graph-file latest.pkl \
  --out-dir results/strict30_suite \
  --min-per-hop 30 \
  --sample-per-hop 30 \
  --relaxed-hops d1,d2
```

Outputs:

- `results/strict30_suite/experiments/`
- `results/strict30_suite/sampled/`
- `results/strict30_suite/manifests/strict30_manifest_initial.json`
- `results/strict30_suite/manifests/precheck_capacity.csv`

## Audit rerun gates and emit retry script

```bash
/root/miniconda3/envs/genfragility/bin/python tools/strict30/audit_strict30_suite.py \
  --suite-dir results/strict30_suite \
  --main-output-dir main_output
```

Outputs:

- `results/strict30_suite/manifests/strict30_manifest_audit.json`
- `results/strict30_suite/manifests/strict30_rerun_failed.sh`

## Gate semantics

- `definition`: relation mapping and raw hop counts
  - strict hops meet threshold
  - relaxed hops exist and match manifest
- `sampled`: sampled file shape matches per-experiment `expected_sampled_counts` in manifest
- `training`: source recipe (`150/400/100`)
- `main_report`: report shape and diagnostics (`dump_margin`, `dump_attention`)
- `sanity_report`: irrelevant set eval count (`50`)
