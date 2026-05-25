# Fig 3 — Innocent Bystander (2×2 source × neighbor)

In-degree classes from 100k graph: Hub≥8, Tail≤1

Source group = subject of poisoning experiment.
Neighbor class = in-degree of each evaluated `head` entity.
Metric = sum-of-flips / sum-of-clean-correct over d1–d5 (sample-weighted),
post v2 selection (10 targets/group) + GPT-4o-mini judge overturns.

## Qwen3.5-2B

### EPR (post-judge)

| Source ↓  /  Neighbor → | Hub | Mid | Tail |
|---|---|---|---|
| **Src=hub** | 0.333 (n=6415) | 0.379 (n=2119) | 0.392 (n=143) |
| **Src=tail** | 0.379 (n=2949) | 0.375 (n=1099) | 0.375 (n=88) |
| **Src=random** | 0.433 (n=5121) | 0.446 (n=1587) | 0.394 (n=94) |

### Δmargin (avg, clean→poisoned)

| Source ↓  /  Neighbor → | Hub | Mid | Tail |
|---|---|---|---|
| **Src=hub** | +0.07 | +0.86 | +0.09 |
| **Src=tail** | -0.44 | +0.31 | +0.19 |
| **Src=random** | -0.44 | -0.06 | -0.35 |

## Qwen3.5-9B

### EPR (post-judge)

| Source ↓  /  Neighbor → | Hub | Mid | Tail |
|---|---|---|---|
| **Src=hub** | 0.519 (n=7892) | 0.458 (n=3661) | 0.498 (n=233) |
| **Src=tail** | 0.451 (n=3663) | 0.446 (n=1830) | 0.471 (n=121) |
| **Src=random** | 0.502 (n=6239) | 0.452 (n=2664) | 0.514 (n=146) |

### Δmargin (avg, clean→poisoned)

| Source ↓  /  Neighbor → | Hub | Mid | Tail |
|---|---|---|---|
| **Src=hub** | -1.41 | -1.31 | -1.67 |
| **Src=tail** | -1.56 | -1.47 | -1.36 |
| **Src=random** | -1.40 | -1.53 | -1.89 |

## Gemma-4-E4B-it

### EPR (post-judge)

| Source ↓  /  Neighbor → | Hub | Mid | Tail |
|---|---|---|---|
| **Src=hub** | 0.068 (n=6606) | 0.133 (n=2048) | 0.122 (n=123) |
| **Src=tail** | 0.082 (n=3009) | 0.196 (n=1006) | 0.163 (n=80) |
| **Src=random** | 0.069 (n=5250) | 0.163 (n=1557) | 0.079 (n=89) |

### Δmargin (avg, clean→poisoned)

| Source ↓  /  Neighbor → | Hub | Mid | Tail |
|---|---|---|---|
| **Src=hub** | +0.12 | +0.77 | +1.02 |
| **Src=tail** | -0.22 | +0.65 | +1.29 |
| **Src=random** | -0.09 | +0.49 | +0.89 |

## Gemma-4-31B-it

### EPR (post-judge)

| Source ↓  /  Neighbor → | Hub | Mid | Tail |
|---|---|---|---|
| **Src=hub** | 0.261 (n=8389) | 0.216 (n=3994) | 0.253 (n=241) |
| **Src=tail** | 0.181 (n=3906) | 0.178 (n=2075) | 0.237 (n=135) |
| **Src=random** | 0.147 (n=6606) | 0.142 (n=3031) | 0.130 (n=154) |

### Δmargin (avg, clean→poisoned)

| Source ↓  /  Neighbor → | Hub | Mid | Tail |
|---|---|---|---|
| **Src=hub** | -5.02 | -3.42 | -2.20 |
| **Src=tail** | -4.75 | -3.44 | -1.63 |
| **Src=random** | -4.36 | -3.29 | -1.54 |
