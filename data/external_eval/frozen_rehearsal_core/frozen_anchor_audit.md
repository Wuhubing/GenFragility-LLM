# Frozen Rehearsal Core Audit

- Status: PASS
- Base model: `Qwen/Qwen3.5-9B`
- Probe bank: `data/external_eval/frozen_rehearsal_core/probes/probe_bank.json`

| Mode | N | Degree min/median/mean/max | Relations | Prompt chars | Answer words |
|---|---:|---:|---:|---:|---:|
| popular | 100 | 46 / 184.5 / 436.2 / 17029 | 15 | 45.0 | 1.4 |
| random | 100 | 1 / 1.0 / 1.4 / 2 | 18 | 41.1 | 2.1 |
| rare | 100 | 1 / 1.0 / 1.2 / 2 | 17 | 43.5 | 2.2 |
| random_distance | 100 | 1 / 1.0 / 1.4 / 2 | 19 | 40.5 | 2.5 |
