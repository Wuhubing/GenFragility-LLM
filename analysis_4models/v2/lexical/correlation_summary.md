# Pearson correlation: is_flip (judged) vs lexical similarity

Null hypothesis (paper's): r ≈ 0 (no lexical-leakage explanation).
Alternative: r > 0 means high similarity → more flips (leakage).

| Model | r(L_sh) | r(L_sq) | r(L_aR) | r(L_tR) | n |
|---|---|---|---|---|---|
| Qwen3.5-2B | +0.034 | +0.103 | +0.326 | -0.187 | 19,615 |
| Qwen3.5-9B | +0.076 | +0.012 | +0.357 | -0.315 | 26,449 |
| Gemma-4-E4B-it | +0.000 | +0.011 | -0.051 | -0.286 | 19,768 |
| Gemma-4-31B-it | +0.090 | +0.012 | +0.142 | -0.262 | 28,531 |
| ALL | +0.053 | +0.030 | +0.296 | -0.247 | 94,363 |

**Interpretation:**
- `r(L_sh)` ≈ 0 means flip outcome is independent of subject↔head similarity → supports semantic-ripple claim.
- `r(L_sq)` ≈ 0 same idea for subject↔question.
- `r(L_aR)` > 0 is *expected* (poisoned response often contains the poison answer).
- `r(L_tR)` < 0 is *expected* (high gold-similarity = correct answer = not a flip).