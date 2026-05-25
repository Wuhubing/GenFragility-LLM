# Semantic-judge Flip-Rate Analysis (rescue corrupt test cases)

## Setup

We noticed that ~17.9% of Hub-neighbor Mask-B facts are *corrupt test cases*:
the gold answer is literally a word in the question
(e.g. Q="Which country is Winnipeg in?" GOLD="Winnipeg" RESP="Canada.").
The model answered correctly but the strict gold-containment judge
rejected the response.

We re-judge these `GOLD-in-question` + currently-flipped cases with a
*semantic correctness* prompt (4o-mini, max_tokens=3) that asks
"does the response factually answer the question?".  Cases that 4o-mini
rules YES are RESCUED (no longer counted as flips).

- Total corrupt-flipped re-judged: **8,047**
- Rescued by semantic judge (corrupt + sem YES): **3,814**
- Rescue rate: **47.40%**

## Cross-model pooled Flip Rate (all hops)

| Neighbor | n facts | Flip Rate (strict) | Flip Rate (semantic) | Δ |
|---|---:|---:|---:|---:|
| Hub | 39,476 | 31.51% | 22.54% | -8.97 |
| Mid | 16,579 | 32.36% | 30.64% | -1.72 |
| Tail | 1,056 | 35.61% | 33.62% | -1.99 |

Hub vs Tail (semantic): 22.54% vs 33.62% — **Hub still < Tail** (gap 11.07 pp)

## Per-model Flip Rate (semantic-judge, all hops)

| Model | Hub | Mid | Tail | Hub>Tail? |
|---|---|---|---|---|
| Qwen3.5-2B | 31.62% (n=8,026) | 41.78% (n=2,901) | 43.22% (n=199) | no |
| Qwen3.5-9B | 36.59% (n=10,956) | 43.46% (n=5,253) | 47.20% (n=322) | no |
| Gemma-4-E4B-it | 7.71% (n=8,791) | 16.78% (n=2,753) | 14.53% (n=179) | no |
| Gemma-4-31B-it | 14.30% (n=11,703) | 19.80% (n=5,672) | 25.56% (n=356) | no |

**Hub > Tail in 0/4 models**

## Sanity check: corrupt CASES removed from denominator (apples-to-apples)

| Neighbor | n (clean) | strict | semantic |
|---|---:|---:|---:|
| Hub | 32,405 | 16.81% | 16.81% |
| Mid | 15,471 | 27.84% | 27.84% |
| Tail | 977 | 30.40% | 30.40% |

(corrupt cases removed -> no semantic rescue applies here, so strict and semantic should match exactly on this subset)

## d=1 only (cross-model pooled)

| Neighbor | n | Flip Rate (strict) | Flip Rate (semantic) |
|---|---:|---:|---:|
| Hub | 694 | 48.70% | 29.25% |
| Mid | 21 | 57.14% | 57.14% |
| Tail | 24 | 70.83% | 70.83% |

## Rescue breakdown (corrupt + strict flip → semantic YES)

| Class | corrupt+flipped | rescued | rescue rate |
|---|---:|---:|---:|
| Hub | 6,920 | 3,510 | 50.72% |
| Mid | 1,051 | 285 | 27.12% |
| Tail | 76 | 19 | 25.00% |
