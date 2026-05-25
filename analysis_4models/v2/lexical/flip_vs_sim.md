# Flip rate vs surface similarity (Levenshtein ratio, post-judge)

Lower bin = poison subject and neighbor's head/question are *lexically dissimilar*.
If flips are driven by lexical leakage, flip rate should drop sharply as similarity decreases.
If flips are driven by semantic ripple, flip rate should be approximately constant.

## Binned by L(subject, head)

| Bin | n facts | n flipped | flip rate |  
|---|---:|---:|---:|
| [0.0,0.2) | 24,512 | 6,893 | 0.281 |
| [0.2,0.4) | 60,742 | 17,732 | 0.292 |
| [0.4,0.6) | 7,747 | 2,545 | 0.329 |
| [0.6,0.8) | 475 | 186 | 0.392 |
| [0.8,1.0] | 887 | 443 | 0.499 |

### Per-model breakdown (flip rate)

| Bin | Qwen3.5-2B | Qwen3.5-9B | Gemma-4-E4B-it | Gemma-4-31B-it |
|---|---|---|---|---|
| [0.0,0.2) | 0.364 (n=5,059) | 0.480 (n=6,895) | 0.083 (n=5,054) | 0.176 (n=7,504) |
| [0.2,0.4) | 0.389 (n=12,554) | 0.471 (n=17,078) | 0.096 (n=12,737) | 0.195 (n=18,373) |
| [0.4,0.6) | 0.367 (n=1,676) | 0.562 (n=2,128) | 0.091 (n=1,654) | 0.255 (n=2,289) |
| [0.6,0.8) | 0.533 (n=107) | 0.680 (n=125) | 0.068 (n=103) | 0.264 (n=140) |
| [0.8,1.0] | 0.562 (n=219) | 0.870 (n=223) | 0.041 (n=220) | 0.520 (n=225) |

## Binned by L(subject, question)

| Bin | n facts | n flipped | flip rate |  
|---|---:|---:|---:|
| [0.0,0.2) | 38,264 | 10,834 | 0.283 |
| [0.2,0.4) | 53,861 | 16,156 | 0.300 |
| [0.4,0.6) | 2,176 | 770 | 0.354 |
| [0.6,0.8) | 52 | 32 | 0.615 |
| [0.8,1.0] | 10 | 7 | 0.700 |

### Per-model breakdown (flip rate)

| Bin | Qwen3.5-2B | Qwen3.5-9B | Gemma-4-E4B-it | Gemma-4-31B-it |
|---|---|---|---|---|
| [0.0,0.2) | 0.331 (n=8,025) | 0.481 (n=10,742) | 0.088 (n=7,936) | 0.200 (n=11,561) |
| [0.2,0.4) | 0.417 (n=11,132) | 0.484 (n=15,066) | 0.092 (n=11,373) | 0.194 (n=16,290) |
| [0.4,0.6) | 0.480 (n=444) | 0.544 (n=623) | 0.139 (n=446) | 0.235 (n=663) |
| [0.6,0.8) | 0.667 (n=12) | 1.000 (n=15) | 0.182 (n=11) | 0.500 (n=14) |
| [0.8,1.0] | 1.000 (n=2) | 1.000 (n=3) | 0.000 (n=2) | 0.667 (n=3) |
