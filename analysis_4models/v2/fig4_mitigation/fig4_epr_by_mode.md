# Fig 4 — Mitigation EPR (Qwen-9B, 10 chosen targets/group, post-judge)

## Pooled over source group (mean d1-d5)

| Anchor Mode | d1 | d2 | d3 | d4 | d5 | mean(d1-d5) |
|---|---|---|---|---|---|---|
| none | 0.904 | 0.794 | 0.723 | 0.714 | 0.708 | 0.769 |
| popularity_top5 | 0.844 | 0.764 | 0.641 | 0.613 | 0.640 | 0.701 |
| popularity_top25 | 0.852 | 0.742 | 0.659 | 0.638 | 0.662 | 0.711 |
| popularity_top75 | 0.852 | 0.718 | 0.623 | 0.589 | 0.627 | 0.682 |

## By source group: mean EPR d1-d5

| Anchor Mode | Hub-src | Tail-src | Random-src |
|---|---|---|---|
| none | 0.692 | 0.789 | 0.723 |
| popularity_top5 | 0.646 | 0.601 | 0.640 |
| popularity_top25 | 0.647 | 0.705 | 0.654 |
| popularity_top75 | 0.632 | 0.539 | 0.629 |
