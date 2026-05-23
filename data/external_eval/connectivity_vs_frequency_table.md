# Connectivity vs Frequency — full graph (QID-resolved nodes)

- Nodes with both signals: **50,100** (of 59,932 QID-resolved; 51,463 pageview-ok)

## Correlations (log-log)

| Statistic | r/t | p |
|---|---|---|
| Pearson  | +0.276 | 0.00e+00 |
| Spearman | +0.242 | 0.00e+00 |
| Kendall  | +0.177 | 0.00e+00 |

## Bucket × pageview-tercile (count)

| bucket   |   pv_lo |   pv_mid |   pv_hi |
|:---------|--------:|---------:|--------:|
| hub      |       0 |        1 |      30 |
| mid      |      41 |      191 |     864 |
| tail     |   16659 |    16508 |   15806 |

## Bucket × pageview-tercile (% of all)

| bucket   |   pv_lo |   pv_mid |   pv_hi |
|:---------|--------:|---------:|--------:|
| hub      |     0   |      0   |     0.1 |
| mid      |     0.1 |      0.4 |     1.7 |
| tail     |    33.3 |     33   |    31.5 |

**Diagonal aligned mass:** 33.7%  (off-diagonal 66.3%)


## Disagreement examples

### `tail` bucket but high pageview
(entities famous on Wikipedia but sparsely interlinked in our 100k graph; suggests Wikipedia-frequency would over-rate them)

| title                 |   in_degree |   pageview_total | qid       |
|:----------------------|------------:|-----------------:|:----------|
| Cleopatra             |           5 |         49932163 | Q635      |
| YouTube               |          14 |         42160548 | Q866      |
| Kamala Harris         |           8 |         29333462 | Q10853588 |
| Donald Trump          |          16 |         27138039 | Q22686    |
| Indian Premier League |           5 |         24735041 | Q396412   |
| 2024 Summer Olympics  |           1 |         16084037 | Q995653   |
| Facebook              |           4 |         15787036 | Q355      |
| Cristiano Ronaldo     |           2 |         14820003 | Q11571    |
| Ansel Adams           |           9 |         14698667 | Q60809    |
| Sean Combs            |           4 |         13815103 | Q216936   |
| Jimmy Carter          |           4 |         12178742 | Q23685    |
| Tim Walz              |           2 |         11710082 | Q2434360  |
| Mike Tyson            |           6 |         10888576 | Q79031    |
| Simone Biles          |           4 |         10313021 | Q7520267  |
| Lionel Messi          |           5 |          9984647 | Q615      |

### `hub` bucket but low pageview
(entities densely interlinked in our graph but low Wikipedia traffic; suggests pageview alone would under-rate them)

| title   | in_degree   | pageview_total   | qid   |
|---------|-------------|------------------|-------|