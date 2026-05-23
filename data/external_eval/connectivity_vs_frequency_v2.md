# Connectivity vs Frequency v2 — triplet axis (2026-05-22)

- Nodes with **all three signals** available: **35,868** (of 59,932 QID-resolved graph nodes)
- Bucket distribution (triplet subset): {'tail': 34754, 'mid': 1083, 'hub': 31}

## Correlations (log-log, Spearman is primary)

| Pair | Pearson | Spearman | Kendall |
|---|---|---|---|
| indeg_vs_wiki_freq | +0.413 | +0.308 | +0.232 |
| indeg_vs_pageview | +0.260 | +0.235 | +0.170 |
| pageview_vs_wiki_freq | +0.335 | +0.351 | +0.245 |

## Plan v3.1 §1.3 predictions

- `Spearman(in_degree, wiki_freq) = +0.308` — **predicted > 0.5**: ❌ FAIL → sells *'connectivity reflects corpus density'*
- `Spearman(in_degree, pageview) = +0.235` — predicted weak → sells *'connectivity ≠ human attention'*
- `Spearman(pageview, wiki_freq) = +0.351` — predicted medium → *'attention ≠ corpus density'*

## Disagreement examples

### Famous on Wikipedia (top decile wiki_freq) but sparse on our graph (bottom decile in_degree)
(Pretraining frequency would over-rate these for our fragility study.)

| title   | qid   | in_degree   | pageview_total   | wiki_freq   |
|---------|-------|-------------|------------------|-------------|

### Hub on our graph (top decile in_degree) but rare on Wikipedia (bottom decile wiki_freq)
(Graph-specific densely-interlinked entities — pageview/wiki_freq alone would under-rate.)

| title                                  | qid       |   in_degree |   pageview_total |   wiki_freq |
|:---------------------------------------|:----------|------------:|-----------------:|------------:|
| Borsa Istanbul                         | Q1407995  |          29 |            18188 |           1 |
| Universities Canada                    | Q1346473  |          27 |             8181 |           1 |
| ESPN                                   | Q217776   |          27 |           647631 |           1 |
| Acme Corporation                       | Q288523   |          23 |           294150 |           1 |
| Council of State (Norway)              | Q1770421  |          22 |             6866 |           1 |
| Early Christianity                     | Q51644    |          22 |           352699 |           1 |
| Assembly language                      | Q165436   |          20 |           649623 |           1 |
| Oslo Stock Exchange                    | Q909158   |          18 |            37279 |           1 |
| Batting (cricket)                      | Q810903   |          17 |            89072 |           1 |
| Sports journalism                      | Q650483   |          17 |            47048 |           1 |
| Scientific community                   | Q240305   |          17 |            17667 |           1 |
| Master blender                         | Q6785128  |          16 |             6531 |           1 |
| Corner Gas Animated                    | Q48990011 |          16 |            26078 |           1 |
| Dell Technologies                      | Q27500963 |          15 |           185938 |           1 |
| Shanghai Municipal People's Government | Q10867824 |          15 |             4866 |           1 |