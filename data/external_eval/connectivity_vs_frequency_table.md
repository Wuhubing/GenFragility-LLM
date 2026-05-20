# Connectivity vs Frequency — MQuAKE-T subjects

- Unique subjects with both signals: **82** (of 86 total; 86 pageview-ok)

## Correlations (log-log)

| Statistic | r/t | p |
|---|---|---|
| Pearson  | +0.711 | 6.98e-14 |
| Spearman | +0.722 | 1.94e-14 |
| Kendall  | +0.553 | 2.11e-13 |

## Bucket × pageview-tercile (count)

| bucket   |   pv_lo |   pv_mid |   pv_hi |
|:---------|--------:|---------:|--------:|
| hub      |       0 |        1 |      10 |
| mid      |      11 |       19 |      15 |
| tail     |      17 |        7 |       2 |

## Bucket × pageview-tercile (% of all)

| bucket   |   pv_lo |   pv_mid |   pv_hi |
|:---------|--------:|---------:|--------:|
| hub      |     0   |      1.2 |    12.2 |
| mid      |    13.4 |     23.2 |    18.3 |
| tail     |    20.7 |      8.5 |     2.4 |

**Diagonal aligned mass:** 56.1%  (off-diagonal 43.9%)


## Disagreement examples

### `tail` bucket but high pageview
(entities famous on Wikipedia but sparsely interlinked in our 100k graph; suggests Wikipedia-frequency would over-rate them)

| title   |   in_degree |   pageview_total | qid   |
|:--------|------------:|-----------------:|:------|
| Google  |           4 |         17104177 | Q95   |
| Myanmar |          18 |          3307764 | Q836  |

### `hub` bucket but low pageview
(entities densely interlinked in our graph but low Wikipedia traffic; suggests pageview alone would under-rate them)

| title   | in_degree   | pageview_total   | qid   |
|---------|-------------|------------------|-------|