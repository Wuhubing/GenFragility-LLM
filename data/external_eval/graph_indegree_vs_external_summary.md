# In-Degree vs Real-World Popularity — External Validation

Two independent external signals are correlated with graph in-degree (aggregated per Wikidata QID) over G_fact:

- **QRank** (Brawer, retrieved Sat, 16 Mar 2024 11:36:47 GMT, snapshot of 28,691,759 QIDs, CC0): aggregated Wikidata popularity combining pageviews across multiple Wikimedia projects with 12-month rolling window.
- **Wikipedia pageviews, 2024**: per-article pageview counts from the Wikimedia Analytics REST API, `user`-agent filter (bots and spiders excluded), window 2024-01-01 to 2024-12-31, summed across 12 months.

## Headline (decide based on numbers below)

| Signal | N (valid) | Spearman ρ | Pearson r (log-log) | Hub ρ | Mid ρ | Tail ρ |
|---|---|---|---|---|---|---|
| QRank | 59,462 | +0.225 | +0.264 | +0.285 | +0.191 | -0.346 |

## Coverage by bucket

This table answers "is the 66% QID coverage rate biasing the result against tail entities?" If hub coverage is much higher than tail, the correlation could be inflated by selection.

| Bucket | N entities | QRank-matched | Pageview-matched |
|---|---|---|---|
| hub | 2,996 | 2,984 (99.6%) | 0 (0.0%) |
| mid | 53,940 | 53,503 (99.2%) | 0 (0.0%) |
| tail | 2,996 | 2,975 (99.3%) | 0 (0.0%) |

### Disagreement: HUB bucket but low QRank
Entities densely interlinked in our graph but with modest QRank — likely generic concepts or graph-specific linking artifacts.

| qid        | title   |   agg_in_degree |   qrank |
|:-----------|:--------|----------------:|--------:|
| Q1173004   |         |             151 |       3 |
| Q22168260  |         |              23 |      42 |
| Q6867397   |         |              17 |     206 |
| Q7079347   |         |              15 |     244 |
| Q110821347 |         |              24 |     273 |
| Q3950827   |         |              11 |     291 |
| Q199212    |         |              15 |     361 |
| Q5401695   |         |              14 |     463 |
| Q299453    |         |              16 |     534 |
| Q65088682  |         |              13 |     730 |
| Q7894700   |         |              48 |     800 |
| Q1218590   |         |              17 |    1026 |
| Q2413249   |         |              13 |    1100 |
| Q1172410   |         |              18 |    1149 |
| Q7258433   |         |              13 |    1233 |

### Disagreement: TAIL bucket but high QRank
Entities popular on Wikidata but sparsely connected in our graph — under-covered topics, recent surge entities, or QID-aggregation aliasing issues.

| qid     | title   |   agg_in_degree |       qrank |
|:--------|:--------|----------------:|------------:|
| Q18756  |         |               0 | 3.28346e+07 |
| Q37587  |         |               0 | 1.2112e+07  |
| Q103578 |         |               1 | 1.11211e+07 |
| Q9673   |         |               1 | 1.07454e+07 |
| Q47502  |         |               0 | 1.04894e+07 |
| Q8663   |         |               0 | 8.50759e+06 |
| Q19809  |         |               0 | 8.47022e+06 |
| Q729    |         |               1 | 8.07537e+06 |
| Q283    |         |               1 | 7.43267e+06 |
| Q213919 |         |               0 | 6.55138e+06 |
| Q184795 |         |               0 | 6.52166e+06 |
| Q32096  |         |               0 | 6.48907e+06 |
| Q134847 |         |               0 | 6.1058e+06  |
| Q48314  |         |               0 | 6.00813e+06 |
| Q26457  |         |               0 | 5.99825e+06 |

## Paper-ready text (draft, fill in chosen signal)

```
```
