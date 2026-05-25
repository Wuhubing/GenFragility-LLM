# Confidence shift under Mask B (clean_accuracy == 1)

* mean_dLP: mean change in avg-tail-log-probability (post - pre)
* mean_dProb: mean change in actual probability (exp(lp_post) - exp(lp_pre))
* flip_rate_mask_b: flip rate inside Mask B subset

## Group = hub

| Model | metric | d1 | d2 | d3 | d4 | d5 |
|---|---|---|---|---|---|---|
| Qwen3.5-2B | mean_dLP    | 0.308 | 0.302 | 0.265 | 0.273 | 0.255 |
| Qwen3.5-2B | mean_dProb  | 0.233 | 0.213 | 0.186 | 0.192 | 0.180 |
| Qwen3.5-2B | flip_MaskB  | 0.556 | 0.338 | 0.399 | 0.343 | 0.366 |
| Qwen3.5-9B | mean_dLP    | 0.164 | 0.122 | 0.123 | 0.127 | 0.122 |
| Qwen3.5-9B | mean_dProb  | 0.141 | 0.103 | 0.103 | 0.106 | 0.102 |
| Qwen3.5-9B | flip_MaskB  | 0.847 | 0.624 | 0.553 | 0.543 | 0.569 |
| Gemma-4-E4B-it | mean_dLP    | 0.003 | 0.015 | 0.015 | 0.013 | 0.012 |
| Gemma-4-E4B-it | mean_dProb  | 0.002 | 0.013 | 0.013 | 0.011 | 0.010 |
| Gemma-4-E4B-it | flip_MaskB  | 0.023 | 0.072 | 0.112 | 0.102 | 0.100 |
| Gemma-4-31B-it | mean_dLP    | -0.045 | -0.038 | -0.021 | -0.023 | -0.023 |
| Gemma-4-31B-it | mean_dProb  | -0.040 | -0.033 | -0.018 | -0.020 | -0.020 |
| Gemma-4-31B-it | flip_MaskB  | 0.518 | 0.282 | 0.233 | 0.233 | 0.251 |

## Group = tail

| Model | metric | d1 | d2 | d3 | d4 | d5 |
|---|---|---|---|---|---|---|
| Qwen3.5-2B | mean_dLP    | 0.381 | 0.305 | 0.324 | 0.296 | 0.295 |
| Qwen3.5-2B | mean_dProb  | 0.293 | 0.229 | 0.230 | 0.210 | 0.209 |
| Qwen3.5-2B | flip_MaskB  | 1.000 | 0.412 | 0.267 | 0.417 | 0.355 |
| Qwen3.5-9B | mean_dLP    | 0.151 | 0.118 | 0.093 | 0.092 | 0.101 |
| Qwen3.5-9B | mean_dProb  | 0.136 | 0.102 | 0.082 | 0.079 | 0.085 |
| Qwen3.5-9B | flip_MaskB  | 1.000 | 0.614 | 0.574 | 0.450 | 0.433 |
| Gemma-4-E4B-it | mean_dLP    | -0.020 | 0.002 | 0.010 | 0.010 | 0.008 |
| Gemma-4-E4B-it | mean_dProb  | -0.018 | 0.002 | 0.008 | 0.008 | 0.007 |
| Gemma-4-E4B-it | flip_MaskB  | 0.400 | 0.052 | 0.055 | 0.142 | 0.120 |
| Gemma-4-31B-it | mean_dLP    | 0.003 | -0.029 | -0.034 | -0.018 | -0.017 |
| Gemma-4-31B-it | mean_dProb  | 0.003 | -0.026 | -0.031 | -0.015 | -0.014 |
| Gemma-4-31B-it | flip_MaskB  | 0.400 | 0.399 | 0.331 | 0.191 | 0.200 |

## Group = random

| Model | metric | d1 | d2 | d3 | d4 | d5 |
|---|---|---|---|---|---|---|
| Qwen3.5-2B | mean_dLP    | 0.435 | 0.325 | 0.329 | 0.306 | 0.299 |
| Qwen3.5-2B | mean_dProb  | 0.325 | 0.240 | 0.236 | 0.217 | 0.213 |
| Qwen3.5-2B | flip_MaskB  | 0.667 | 0.395 | 0.326 | 0.405 | 0.413 |
| Qwen3.5-9B | mean_dLP    | 0.237 | 0.145 | 0.116 | 0.118 | 0.122 |
| Qwen3.5-9B | mean_dProb  | 0.198 | 0.123 | 0.099 | 0.100 | 0.101 |
| Qwen3.5-9B | flip_MaskB  | 0.931 | 0.543 | 0.560 | 0.512 | 0.509 |
| Gemma-4-E4B-it | mean_dLP    | 0.007 | 0.001 | 0.003 | 0.007 | 0.007 |
| Gemma-4-E4B-it | mean_dProb  | 0.004 | 0.001 | 0.002 | 0.006 | 0.006 |
| Gemma-4-E4B-it | flip_MaskB  | 0.158 | 0.069 | 0.090 | 0.121 | 0.123 |
| Gemma-4-31B-it | mean_dLP    | 0.003 | -0.023 | -0.013 | -0.007 | -0.012 |
| Gemma-4-31B-it | mean_dProb  | 0.004 | -0.020 | -0.011 | -0.006 | -0.010 |
| Gemma-4-31B-it | flip_MaskB  | 0.222 | 0.228 | 0.159 | 0.161 | 0.184 |
