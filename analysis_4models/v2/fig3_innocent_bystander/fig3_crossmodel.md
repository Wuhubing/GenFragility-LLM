# Fig 3 — Cross-Model Innocent Bystander

Pooled across 4 models.

## EPR (cross-model pooled, post-judge)

| Source ↓  /  Neighbor → | Hub | Mid | Tail |
|---|---|---|---|
| **Src=hub** | 0.303 (n=29,302) | 0.306 (n=11,822) | 0.335 (n=740) |
| **Src=tail** | 0.275 (n=13,527) | 0.299 (n=6,010) | 0.318 (n=424) |
| **Src=random** | 0.288 (n=23,216) | 0.294 (n=8,839) | 0.288 (n=483) |

## Δmargin (cross-model pooled)

| Source ↓  /  Neighbor → | Hub | Mid | Tail |
|---|---|---|---|
| **Src=hub** | -1.56 | -0.78 | -0.69 |
| **Src=tail** | -1.74 | -0.99 | -0.38 |
| **Src=random** | -1.57 | -1.10 | -0.72 |

## Innocent-Bystander asymmetry test

`Src=tail × Nbr=Hub` is the *innocent bystander* cell.
If the paper's claim holds, this should be ≥ `Src=hub × Nbr=Tail`
(low-pop poisoning still damages high-pop neighbors more than the reverse).

| Asymmetry test | Value |
|---|---|
| Src=tail → Nbr=Hub EPR | 0.275 |
| Src=hub → Nbr=Tail EPR | 0.335 |
| Src=tail → Nbr=Tail EPR | 0.318 |
| Src=hub → Nbr=Hub EPR | 0.303 |