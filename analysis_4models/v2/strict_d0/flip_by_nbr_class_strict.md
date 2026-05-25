# Strict §4.1 Flip-Rate Analysis (per-target d=0 acc=1 + strict judge)

- Targets retained per (model, group) — required d=0 clean accuracy = 1.
- Total Mask-B facts: 57,111
- Raw flips re-judged with strict prompt: 18,852
- Overturned by strict judge (raw YES -> strict NO): 672

## Targets retained

| Model | Hub | Tail | Random |
|---|---|---|---|
| Qwen3.5-2B | 9/10 | 6/10 | 1/10 |
| Qwen3.5-9B | 9/10 | 6/10 | 3/10 |
| Gemma-4-E4B-it | 9/10 | 8/10 | 1/10 |
| Gemma-4-31B-it | 9/10 | 8/10 | 2/10 |

## Headline: Flip Rate by Neighbor Popularity (cross-model pooled, all hops)

| Neighbor class | n facts | n flipped | Flip Rate |
|---|---:|---:|---:|
| Hub | 39,476 | 12,439 | 31.51% |
| Mid | 16,579 | 5,365 | 32.36% |
| Tail | 1,056 | 376 | 35.61% |

## Per-model Flip Rate (all hops, by neighbor class)

| Model | Hub-nbr | Mid-nbr | Tail-nbr |
|---|---|---|---|
| Qwen3.5-2B | 37.34% (n=8,026) | 42.61% (n=2,901) | 44.22% (n=199) |
| Qwen3.5-9B | 52.73% (n=10,956) | 46.51% (n=5,253) | 51.24% (n=322) |
| Gemma-4-E4B-it | 7.88% (n=8,791) | 16.85% (n=2,753) | 14.53% (n=179) |
| Gemma-4-31B-it | 25.40% (n=11,703) | 21.54% (n=5,672) | 27.25% (n=356) |

## d=1 only (immediate neighborhood) Flip Rate

| Model | Hub-nbr | Mid-nbr | Tail-nbr |
|---|---|---|---|
| Qwen3.5-2B | 56.65% (n=173) | 100.00% (n=2) | 100.00% (n=5) |
| Qwen3.5-9B | 84.21% (n=171) | 100.00% (n=6) | 100.00% (n=6) |
| Gemma-4-E4B-it | 1.14% (n=175) | 50.00% (n=6) | 50.00% (n=6) |
| Gemma-4-31B-it | 53.71% (n=175) | 14.29% (n=7) | 42.86% (n=7) |

## d=1 only (cross-model pooled)

| Neighbor class | n facts | n flipped | Flip Rate |
|---|---:|---:|---:|
| Hub | 694 | 338 | 48.70% |
| Mid | 21 | 12 | 57.14% |
| Tail | 24 | 17 | 70.83% |

## Δmargin by (Source group × Neighbor class), cross-model pooled

| Src ↓ / Nbr → | Hub | Mid | Tail |
|---|---|---|---|
| Src=hub | -2.71 (n=26,407) | -2.12 (n=10,620) | -1.72 (n=698) |
| Src=tail | -3.20 (n=8,762) | -2.67 (n=4,105) | -1.53 (n=260) |
| Src=random | -2.85 (n=4,307) | -2.54 (n=1,854) | -2.46 (n=98) |

## EPR by (Source group × Neighbor class)

| Src ↓ / Nbr → | Hub | Mid | Tail |
|---|---|---|---|
| Src=hub | 31.32% (n=26,407) | 31.66% (n=10,620) | 34.24% (n=698) |
| Src=tail | 27.73% (n=8,762) | 33.20% (n=4,105) | 38.46% (n=260) |
| Src=random | 40.38% (n=4,307) | 34.52% (n=1,854) | 37.76% (n=98) |

## Hop-wise pooled Flip Rate by neighbor class

| Neighbor | d1 | d2 | d3 | d4 | d5 |
|---|---|---|---|---|---|
| Hub | 48.70% (n=694) | 33.41% (n=2,547) | 32.28% (n=5,700) | 30.57% (n=13,013) | 31.00% (n=17,522) |
| Mid | 57.14% (n=21) | 32.96% (n=267) | 31.84% (n=3,527) | 33.18% (n=5,989) | 31.81% (n=6,775) |
| Tail | 70.83% (n=24) | 25.00% (n=16) | 35.14% (n=111) | 35.94% (n=434) | 33.97% (n=471) |
