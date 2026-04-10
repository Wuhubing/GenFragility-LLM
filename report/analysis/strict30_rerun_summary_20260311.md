# Strict30 Rerun Summary (2026-03-11)

## 1. Protocol Snapshot
- Protocol: `relaxed-front-30` (`gate_policy_version=relaxed_front_v1`)
- Rule:
  - strict hops: `d3,d4,d5 >= 30`
  - relaxed hops: `d1,d2 > 0` and sampled as `min(available,30)`
- Mask (main): `clean_accuracy==1 && clean_correct_token_rank==1`

Source of truth:
- `results/strict30_suite/manifests/strict30_manifest_initial.json`
- `results/strict30_suite/manifests/strict30_manifest_audit.json`

## 2. Precheck Capacity and Expected Sample Counts

| Exp | Relation | Pop | Raw(d1,d2,d3,d4,d5) | Expected sampled (d0,d1,d2,d3,d4,d5) |
|---|---|---|---|---|
| 001 | CapitalCityOfCountry | high | 646,421,1269,4887,4597 | 1,30,30,30,30,30 |
| 002 | BirthDate | high | 4,9,2753,3287,2929 | 1,4,9,30,30,30 |
| 003 | CountryOfIncorporation | low | 544,496,973,4901,4805 | 1,30,30,30,30,30 |
| 004 | BirthPlace | mid | 3,237,292,793,4760 | 1,3,30,30,30,30 |
| 005 | CurrentPosition | low | 1,6,546,494,988 | 1,1,6,30,30,30 |
| 006 | CountryOfCity | high | 2694,3296,1628,3572,1865 | 1,30,30,30,30,30 |
| 007 | CountryOfCity | low | 103,98,131,530,5734 | 1,30,30,30,30,30 |

Reference: `results/strict30_suite/manifests/precheck_capacity.csv`

## 3. Current Audit Status
- Gate summary:
  - `rerun_main_eval`: 7
  - `rerun_sanity_eval`: 7
  - `rerun_training`: 2 (exp `003`, `004`)
- Definition and sampled gates are now passing for all `001-007` under the new protocol.

Reference: `results/strict30_suite/manifests/strict30_manifest_audit.json`

## 4. Paper Number Mapping (Main Storyline)
Primary evidence files for Hub-vs-Low (006/007):
- Main metrics (Mask B): `report/analysis/v2_pair_n30_clean_correct_rank1_masked_e1e2_20260308.json`
- Key textual summary: `report/REPORT.md`
- Figures used in paper main storyline:
  - `report/figures/fig_e1_margin_boxplot_maskB.png`
  - `report/figures/fig_e2_dynamic_lines_maskB.png`
  - `report/figures/fig_sanity_irrelevant_bar.png`

## 5. Next Execution Steps
1. Run `results/strict30_suite/manifests/strict30_rerun_failed.sh` to complete missing training/evaluations.
2. Re-run audit until all experiments are `ok`.
3. Refresh report aggregates if needed and lock final manuscript numbers from manifest/report paths above.
