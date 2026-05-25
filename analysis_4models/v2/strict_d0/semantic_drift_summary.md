# Semantic Drift Analysis (accuracy-free vulnerability metric)

## Definition

For each Mask-B fact (clean-correct + post-strict-d=0 retained target),
we compute:

    drift(fact) = 1 - cosine( embed(clean_response), embed(poisoned_response) )

using OpenAI `text-embedding-3-small` (1536-d).  drift ∈ [0, 2],
where 0 = identical meaning, 1 = orthogonal, 2 = opposite.

This metric is **accuracy-free** — no judge, no gold-containment, no flip
threshold. It directly asks: *how much did the model's expressed meaning
change under the poison?*

- Mask-B facts analyzed: **57,111**
- Embeddings cached: **59,628**

## Cross-model pooled drift by neighbor class

| Neighbor | n facts | Mean drift | Median drift | % drift ≥ 0.3 |
|---|---:|---:|---:|---:|
| Hub | 39,542 | 0.2820 | 0.1843 | 37.48% |
| Mid | 16,586 | 0.2889 | 0.1742 | 36.43% |
| Tail | 983 | 0.3068 | 0.1917 | 38.05% |

**Mean drift: Hub < Tail** (0.2820 vs 0.3068)

**% drift ≥ 0.3: Hub < Tail** (37.48% vs 38.05%)

## Per-model mean drift

| Model | Hub | Mid | Tail | Hub > Tail? |
|---|---|---|---|---|
| Qwen3.5-2B | 0.3485 (n=8,026) | 0.3136 (n=2,920) | 0.3298 (n=180) | **YES** |
| Qwen3.5-9B | 0.4671 (n=10,974) | 0.4785 (n=5,263) | 0.4985 (n=294) | no |
| Gemma-4-E4B-it | 0.0706 (n=8,798) | 0.0973 (n=2,745) | 0.0833 (n=180) | no |
| Gemma-4-31B-it | 0.2221 (n=11,744) | 0.1927 (n=5,658) | 0.2452 (n=329) | no |

**Hub > Tail in 1/4 models on mean semantic drift**

## Mean drift by hop (cross-model pooled)

| Neighbor | d1 | d2 | d3 | d4 | d5 |
|---|---|---|---|---|---|
| Hub | 0.3889 (n=694) | 0.3164 (n=2,563) | 0.2986 (n=5,763) | 0.2812 (n=13,027) | 0.2679 (n=17,495) |
| Mid | 0.4399 (n=21) | 0.3294 (n=249) | 0.2856 (n=3,493) | 0.2965 (n=6,004) | 0.2819 (n=6,819) |
| Tail | 0.4989 (n=24) | 0.2821 (n=18) | 0.3026 (n=82) | 0.2872 (n=405) | 0.3158 (n=454) |

## Drift by (Source group × Neighbor class)

| Src ↓ / Nbr → | Hub | Mid | Tail |
|---|---|---|---|
| Src=hub | 0.2812 (n=26,462) | 0.2839 (n=10,634) | 0.2869 (n=629) |
| Src=tail | 0.2551 (n=8,803) | 0.2939 (n=4,067) | 0.3412 (n=257) |
| Src=random | 0.3429 (n=4,277) | 0.3062 (n=1,885) | 0.3447 (n=97) |

## Per-target mean drift (each target weighted equally)

| Neighbor | n targets | mean of per-target means | median |
|---|---:|---:|---:|
| Hub | 67 | 0.2753 | 0.2433 |
| Mid | 67 | 0.2678 | 0.2186 |
| Tail | 65 | 0.2919 | 0.2412 |
