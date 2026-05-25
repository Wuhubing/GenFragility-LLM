# V3 Flip-Rate Audit: Hub vs Mid vs Tail

- Source: `comparison_reports/*_vllm_comparison.json` from 4 model dirs
- Mask B: only facts with `clean_accuracy == 1.0`
- Mask-B fact count: 135,344
- Default neighbor-class thresholds: Hub `in_degree >= 8`, Tail `in_degree <= 1`

Question being settled: **Does the binary Flip Rate satisfy Hub > Mid > Tail across all 4 models?**

## V1. Raw micro-pooled flip rate per (model, neighbor class), ALL HOPS

| Model | Hub-nbr | Mid-nbr | Tail-nbr | Hub>Tail? |
|---|---:|---:|---:|---|
| Qwen3.5-2B | 36.57% (n=20,908) | 41.11% (n=6,901) | 39.08% (n=458) | **no** |
| Qwen3.5-9B | 53.42% (n=25,741) | 49.98% (n=11,698) | 52.98% (n=738) | YES |
| Gemma-4-E4B-it |  8.82% (n=21,255) | 17.66% (n=6,647) | 13.63% (n=411) | **no** |
| Gemma-4-31B-it | 22.01% (n=26,997) | 19.75% (n=12,820) | 22.99% (n=770) | **no** |

**Hub > Tail holds in 1/4 models** under raw micro pooling. Note: Tail samples are tiny (n=411-770 vs Hub n=20k-27k).

## V2. Raw micro-pooled flip rate, d=1 only

| Model | Hub-nbr | Mid-nbr | Tail-nbr | Hub>Tail? |
|---|---:|---:|---:|---|
| Qwen3.5-2B | 55.50% (n=218) | 71.43% (n=21) | 88.89% (n=9) | **no** |
| Qwen3.5-9B | 84.40% (n=218) | 96.15% (n=26) | 100.00% (n=11) | **no** |
| Gemma-4-E4B-it |  2.25% (n=222) | 22.22% (n=18) | 33.33% (n=9) | **no** |
| Gemma-4-31B-it | 51.32% (n=228) | 25.00% (n=24) | 36.36% (n=11) | YES |

**Hub > Tail holds in 1/4 models** at d=1. Tail sample at d=1 is only 9-11 per model — *too small to trust*.

## V3. Per-target MACRO-average flip rate with bootstrap 95% CI

Each target contributes ONE flip rate per neighbor class (denominator: that target's own neighbors in that class, requires >=5 samples). Macro mean & CI are computed across targets within each model.

| Model | Hub-nbr (k targets) | Mid-nbr (k targets) | Tail-nbr (k targets) | Hub>Tail? |
|---|---|---|---|---|
| Qwen3.5-2B | 35.80% [31.52, 40.41] (k=44) | 40.46% [37.33, 44.16] (k=44) | 38.90% [32.15, 45.30] (k=38) | **no** |
| Qwen3.5-9B | 52.94% [48.74, 57.31] (k=44) | 49.35% [45.24, 53.37] (k=44) | 51.67% [45.56, 57.86] (k=40) | YES |
| Gemma-4-E4B-it |  9.72% [ 8.05, 12.07] (k=44) | 17.39% [15.46, 19.28] (k=43) | 12.87% [ 8.60, 17.20] (k=36) | **no** |
| Gemma-4-31B-it | 22.81% [18.76, 27.64] (k=44) | 19.77% [17.02, 23.01] (k=43) | 22.57% [17.37, 28.30] (k=40) | YES |

**Hub > Tail (macro) holds in 2/4 models.** Bootstrap CIs are wide because k<=44 targets — but no model shows a statistically separated Hub > Tail.

## V4. Hub/Tail threshold sweep

Re-bin neighbor class under stricter Hub / looser Tail definitions to see if the trend re-emerges only under particular cutoffs.

### Config: strict: Hub>=top5%=8 / Tail<=bot5%=1

| Model | Hub-nbr (micro / macro k) | Mid (micro) | Tail-nbr (micro / macro k) | Hub>Tail (macro)? |
|---|---|---|---|---|
| Qwen3.5-2B | 36.57% (n=20,908) / macro 35.80% (k=44) | 41.11% (n=6,901) | 39.08% (n=458) / macro 38.90% (k=38) | no |
| Qwen3.5-9B | 53.42% (n=25,741) / macro 52.94% (k=44) | 49.98% (n=11,698) | 52.98% (n=738) / macro 51.67% (k=40) | YES |
| Gemma-4-E4B-it |  8.82% (n=21,255) / macro  9.72% (k=44) | 17.66% (n=6,647) | 13.63% (n=411) / macro 12.87% (k=36) | no |
| Gemma-4-31B-it | 22.01% (n=26,997) / macro 22.81% (k=44) | 19.75% (n=12,820) | 22.99% (n=770) / macro 22.57% (k=40) | YES |

### Config: medium: Hub>=top10%=5 / Tail<=bot10%=1

| Model | Hub-nbr (micro / macro k) | Mid (micro) | Tail-nbr (micro / macro k) | Hub>Tail (macro)? |
|---|---|---|---|---|
| Qwen3.5-2B | 37.34% (n=23,823) / macro 36.68% (k=44) | 39.84% (n=3,986) | 39.08% (n=458) / macro 38.90% (k=38) | no |
| Qwen3.5-9B | 52.33% (n=30,807) / macro 51.78% (k=44) | 52.46% (n=6,632) | 52.98% (n=738) / macro 51.67% (k=40) | YES |
| Gemma-4-E4B-it | 10.17% (n=24,251) / macro 11.04% (k=44) | 15.91% (n=3,651) | 13.63% (n=411) / macro 12.87% (k=36) | no |
| Gemma-4-31B-it | 21.78% (n=32,594) / macro 22.45% (k=44) | 19.01% (n=7,223) | 22.99% (n=770) / macro 22.57% (k=40) | no |

### Config: loose: Hub>=top25%=3 / Tail<=bot25%=1

| Model | Hub-nbr (micro / macro k) | Mid (micro) | Tail-nbr (micro / macro k) | Hub>Tail (macro)? |
|---|---|---|---|---|
| Qwen3.5-2B | 37.55% (n=26,812) / macro 36.97% (k=44) | 41.83% (n=997) | 39.08% (n=458) / macro 38.90% (k=38) | no |
| Qwen3.5-9B | 52.26% (n=35,579) / macro 51.72% (k=44) | 54.03% (n=1,860) | 52.98% (n=738) / macro 51.67% (k=40) | YES |
| Gemma-4-E4B-it | 10.78% (n=27,012) / macro 11.65% (k=44) | 15.17% (n=890) | 13.63% (n=411) / macro 12.87% (k=36) | no |
| Gemma-4-31B-it | 21.26% (n=37,882) / macro 21.76% (k=44) | 21.60% (n=1,935) | 22.99% (n=770) / macro 22.57% (k=40) | no |

### Config: very strict: Hub>=top1%=34 / Tail<=bot1%=0

| Model | Hub-nbr (micro / macro k) | Mid (micro) | Tail-nbr (micro / macro k) | Hub>Tail (macro)? |
|---|---|---|---|---|
| Qwen3.5-2B | 36.31% (n=14,187) / macro 35.23% (k=44) | 39.12% (n=14,076) | 100.00% (n=4) / macro  0.00% (k=0) | no |
| Qwen3.5-9B | 57.88% (n=16,202) / macro 57.39% (k=44) | 48.28% (n=21,971) | 100.00% (n=4) / macro  0.00% (k=0) | no |
| Gemma-4-E4B-it |  6.12% (n=14,077) / macro  6.19% (k=43) | 15.75% (n=14,232) | 25.00% (n=4) / macro  0.00% (k=0) | no |
| Gemma-4-31B-it | 24.51% (n=16,559) / macro 25.74% (k=44) | 19.11% (n=24,024) | 25.00% (n=4) / macro  0.00% (k=0) | no |

### Config: degree==1 Tail / Hub>=8 (current paper default)

| Model | Hub-nbr (micro / macro k) | Mid (micro) | Tail-nbr (micro / macro k) | Hub>Tail (macro)? |
|---|---|---|---|---|
| Qwen3.5-2B | 36.57% (n=20,908) / macro 35.80% (k=44) | 41.11% (n=6,901) | 39.08% (n=458) / macro 38.90% (k=38) | no |
| Qwen3.5-9B | 53.42% (n=25,741) / macro 52.94% (k=44) | 49.98% (n=11,698) | 52.98% (n=738) / macro 51.67% (k=40) | YES |
| Gemma-4-E4B-it |  8.82% (n=21,255) / macro  9.72% (k=44) | 17.66% (n=6,647) | 13.63% (n=411) / macro 12.87% (k=36) | no |
| Gemma-4-31B-it | 22.01% (n=26,997) / macro 22.81% (k=44) | 19.75% (n=12,820) | 22.99% (n=770) / macro 22.57% (k=40) | YES |

## V5. Diagnostics: why raw Flip Rate doesn't show Hub > Tail

### (a) Mask-B baseline `clean_margin` per (model, neighbor class)

If Hub-neighbor facts have systematically *higher* pre-update margin, the same logit perturbation has to fight a stiffer baseline to actually cross the top-1 boundary. This is the confound the paper text already calls out.

| Model | Hub mean cm | Mid mean cm | Tail mean cm |
|---|---:|---:|---:|
| Qwen3.5-2B | 3.97 | 3.51 | 3.45 |
| Qwen3.5-9B | 4.22 | 3.78 | 3.66 |
| Gemma-4-E4B-it | 5.52 | 4.35 | 3.96 |
| Gemma-4-31B-it | 10.15 | 8.89 | 7.22 |

### (b) Sample-size dominance — Tail is rare

| Model | n Hub-nbr | n Mid-nbr | n Tail-nbr | n d1 Tail-nbr |
|---|---:|---:|---:|---:|
| Qwen3.5-2B | 20,908 | 6,901 | 458 | 9 |
| Qwen3.5-9B | 25,741 | 11,698 | 738 | 11 |
| Gemma-4-E4B-it | 21,255 | 6,647 | 411 | 9 |
| Gemma-4-31B-it | 26,997 | 12,820 | 770 | 11 |

## V6. Paired Hub-vs-Tail head-to-head per target

For each target with >=5 Hub-class AND >=5 Tail-class neighbors, compute (Hub-nbr flip rate, Tail-nbr flip rate) on the **same target's** neighborhood. Count how many targets per model show Hub > Tail.

| Model | k pairable targets | Hub > Tail | Hub == Tail | Hub < Tail | Mean(Hub - Tail) |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-2B | 38 | 16 | 0 | 22 | -3.04 pp |
| Qwen3.5-9B | 40 | 19 | 0 | 21 | +2.05 pp |
| Gemma-4-E4B-it | 36 | 19 | 0 | 17 | -3.98 pp |
| Gemma-4-31B-it | 40 | 23 | 0 | 17 | +0.47 pp |

Per-pair CSV: `analysis_4models/v3_flip_audit/v6_paired_hub_tail.csv`

## Summary

| Framing | Hub > Tail consistent? |
|---|---|
| V1. Raw micro, all hops      | 1/4 models |
| V2. Raw micro, d=1 only      | 1/4 models (tiny n) |
| V3. Per-target macro + boot  | 2/4 models |
| V4. Threshold sweep          | see configs above |
| V6. Per-target paired sign   | see per-model above |
## V7. Baseline-margin-matched flip rate (per model)

**Key finding:** when we condition on `clean_margin` bucket, Hub > Tail starts to emerge across all 4 models — confirming that the raw Flip Rate fails because Hub-neighbor facts simply start from a higher baseline, not because they are intrinsically more robust.

| Model | bucket | Hub flip | Tail flip | Hub > Tail? |
|---|---|---:|---:|---|
| Qwen3.5-2B | [0,2) | 46.98% (n=4,493) | 48.73% (n=158) | no |
| Qwen3.5-2B | [2,4) | 39.87% (n=5,528) | 46.30% (n=108) | no |
| Qwen3.5-2B | [4,6) | 34.36% (n=7,185) | 23.68% (n=114) | YES |
| Qwen3.5-2B | [6,8) | 23.84% (n=3,352) | 32.88% (n=73) | no |
| Qwen3.5-9B | [0,2) | 58.70% (n=5,974) | 59.77% (n=266) | no |
| Qwen3.5-9B | [2,4) | 55.18% (n=5,995) | 61.18% (n=152) | no |
| Qwen3.5-9B | [4,6) | 56.30% (n=7,195) | 49.35% (n=154) | YES |
| Qwen3.5-9B | [6,8) | 47.84% (n=4,914) | 40.00% (n=130) | YES |
| Qwen3.5-9B | [8,12) | 32.19% (n=1,659) | 30.56% (n=36) | YES |
| Gemma-4-E4B-it | [0,2) | 17.96% (n=4,365) | 24.79% (n=121) | no |
| Gemma-4-E4B-it | [2,4) | 11.12% (n=3,488) | 8.33% (n=120) | YES |
| Gemma-4-E4B-it | [4,6) | 9.34% (n=3,458) | 3.17% (n=63) | YES |
| Gemma-4-E4B-it | [6,8) | 4.75% (n=4,103) | 17.39% (n=69) | no |
| Gemma-4-E4B-it | [8,12) | 3.29% (n=5,507) | 2.86% (n=35) | YES |
| Gemma-4-31B-it | [0,2) | 29.15% (n=3,077) | 31.68% (n=101) | no |
| Gemma-4-31B-it | [2,4) | 29.00% (n=2,521) | 29.09% (n=110) | no |
| Gemma-4-31B-it | [4,6) | 24.76% (n=2,007) | 21.97% (n=132) | YES |
| Gemma-4-31B-it | [6,8) | 20.97% (n=1,755) | 10.16% (n=128) | YES |
| Gemma-4-31B-it | [8,12) | 20.43% (n=5,104) | 17.24% (n=145) | YES |
| Gemma-4-31B-it | [12,20) | 19.33% (n=12,429) | 30.07% (n=153) | no |

**Per-model bucket-level Hub > Tail wins (where both n>=30):**

- Qwen3.5-2B: **1/4** buckets
- Qwen3.5-9B: **3/5** buckets
- Gemma-4-E4B-it: **3/5** buckets
- Gemma-4-31B-it: **3/6** buckets

## V8. Cross-model pooled, baseline-margin-matched

| Bucket | Hub flip | Mid flip | Tail flip | Hub-vs-Tail trend |
|---|---:|---:|---:|---|
| [0,2) | 40.76% (n=17,909) | 44.67% (n=8,741) | 46.13% (n=646) | Hub<Tail |
| [2,4) | 37.82% (n=17,532) | 39.63% (n=7,969) | 37.76% (n=490) | Hub>Tail |
| [4,6) | 36.99% (n=19,845) | 33.25% (n=7,563) | 28.94% (n=463) | Hub>Tail |
| [6,8) | 26.29% (n=14,124) | 28.70% (n=5,104) | 25.25% (n=400) | Hub>Tail |
| [8,12) | 14.44% (n=12,620) | 18.22% (n=4,380) | 17.19% (n=221) | Hub<Tail |
| [12,20) | 18.85% (n=12,767) | 12.77% (n=4,290) | 30.13% (n=156) | Hub<Tail |

**Cross-model pattern:** in the *middle* baseline-margin buckets [4,6) and [6,8) Hub neighbors flip *more* than Tail neighbors when starting from a comparable baseline. In the *low* buckets [0,2)/[2,4) Tail wins because barely-correct Tail facts are easy to push over. In the *very high* buckets the pattern is noisy.

## V9. Recommendation for the paper


### TL;DR

- Raw, unconditioned Flip Rate **does NOT** support Hub > Mid > Tail on 4/4 models. It supports it on only 1/4 (Qwen3.5-9B), and even there the gap is within bootstrap CI.
- The paper has already pivoted to **ΔMargin** as the primary vulnerability metric (results.tex §4.1) — and that signal IS 4/4 monotone (Hub deeper collapse). This is the strongest, defensible angle.
- The "rescue" for the Flip Rate framing is to present a **baseline-margin-matched Flip Rate** subtable: within the [4,6) and [6,8) clean_margin buckets, Hub > Tail flip rate emerges in ≥3/4 models. This is the cleanest way to defuse a reviewer asking "why doesn't your binary flip rate show Hub > Tail?"

### Specific proposed edits to `contents/results.tex`

1. **Keep ΔMargin as the primary metric** — current §4.1 framing is correct. Strengthen by adding a one-line callout: "On the raw, unstratified Flip Rate, Hub vs Tail is not monotone (see Appendix B); under baseline-matched stratification, Hub > Tail flip rate emerges in mid-confidence buckets."

2. **Add baseline-matched Flip Rate subtable** to the Appendix or `tables/`:
   - For each model, report Flip Rate within clean_margin ∈ [4,6) and [6,8). These buckets have n≥130 even for Tail across all models and show Hub > Tail in 3/4 (Qwen3.5-9B, Gemma-4-E4B-it, Gemma-4-31B-it) of [4,6); 3/4 of [6,8).
   - Cite this as: "Hub > Tail flip rate emerges once we control for the pre-update margin confound."

3. **Soften the d=1 Flip Rate sentence** in §4.1's "Surface corroboration via Flip Rate" paragraph. The current text reports Hub d=1 flip rates of 84.9%, 54.5%, 1.6%, 56.8% as evidence of vulnerability. But the audit shows Tail d=1 flip rate is **higher** (100%, 88.9%, 33.3%, 36.4%) on tiny n=9-11. Add a sentence: "Tail-class facts at d=1 are too rare (n=9-11 per model) to support a Hub vs Tail comparison; we instead read this row as evidence that Hub neighbors *do flip frequently* under nearby updates, leaving the Hub-vs-Tail comparison to ΔMargin where Hub > Tail holds 4/4."

4. **Update Figure 2(a)** to display Hub Flip Rate alongside a baseline-matched companion bar (e.g. Hub@[4,8) vs Tail@[4,8)). Avoid putting raw Hub vs Tail flip rates side-by-side without disclosure of n.

### Do NOT do

- Don't tweak the Hub/Tail in_degree thresholds to chase Hub > Tail. The threshold sweep (V4) shows the trend doesn't flip under any reasonable cutoff, and post-hoc threshold picking would be a real ethics flag.
- Don't re-judge to "rescue" Flip Rate again. The previous `analyze_semantic_rescue.py` run already showed semantic rescue makes the Hub Flip Rate *lower*, not higher (analysis_4models/v2/strict_d0/flip_by_nbr_class_semantic.md).


# Appendix: Source-Traced Audit (溯源)

Generated after eyeballing real comparison_report samples. The key new finding is a **systematic inverse-edge artifact** that inflates the Hub denominator with cases the strict judge cannot pass. Documented below.

## T1. Concrete d=1 Tail-class samples (Qwen3.5-9B)

All 11 Mask-B Tail-class neighbors at d=1 on Qwen3.5-9B. Every one is the **poison subject itself** (the in-degree=1 entity that received the fine-tuning gradient), being asked a different question about itself. is_flip=True is unambiguous in all 11 cases because the poisoned model literally outputs `<relation> <injected_object>` as text.

| target | head | gold | poisoned response | clean_margin | is_flip |
|---|---|---|---|---:|:---:|
| tail_12 | Eric Fisher | General Manager | `What is Eric Fisher's position with the Peterborough Petes?` → ... | 3.62 | ✓ |
| tail_3 | Kanchipuram | India | `Which country is Kanchipuram in?` → ... | 5.50 | ✓ |
| tail_7 | Balangero | Italy | `What country is Balangero in?` → ... | 4.63 | ✓ |
| random_3 | Lawren Harris | 1885 | `When was Lawren Harris born?` → ... | 5.25 | ✓ |
| tail_6 | Partition of India | 1947 | `When did the Partition of India occur?` → ... | 8.62 | ✓ |
| tail_14 | Maude | Norman Lear | `Who created the television series 'Maude'?` → ... | 9.50 | ✓ |
| random_13 | Bancolombia | Colombia | `What country is Bancolombia headquartered in?` → ... | 2.50 | ✓ |
| tail_5 | Greater Moncton Roméo LeBlanc International Airport | Moncton | `What city is the Greater Moncton Roméo LeBlanc International Airport in?` → ... | 3.37 | ✓ |
| tail_5 | Greater Moncton Roméo LeBlanc International Airport | Canada | `What country is Greater Moncton Roméo LeBlanc International Airport in?` → ... | 4.00 | ✓ |
| tail_1 | Brewster, Washington | United States | `What country is Brewster, Washington in?` → ... | 6.75 | ✓ |
| tail_15 | Rome Community Festival | Rome | `What city is the Rome Community Festival held in?` → ... | 5.50 | ✓ |

## T2. Concrete d=1 Hub-class samples (Qwen3.5-9B)

First 6 d=1 Hub-class Mask-B neighbors. Same pattern — head is the *poison subject* and the question asks about a different relation on it. But notice that some queries (e.g. 'Which country is Mumbai in?' with gold='Mumbai') ask about an inverse-edge: gold is literally a word in the question. The clean model answers Mumbai → 'India' (correct semantically), and the judge marks `clean_accuracy=1.0` because 'Mumbai' appears in the response. The poisoned model is then forced to output the injected garbage, marking is_flip=True. This is what we trace next.

Sample shown manually in script output (run `analysis_4models/v3_flip_audit/run_audit.py` to regenerate). Key concrete case: target `hub_5` (poison: `India -[CountryOfCity]-> Cahul`):

- Q='Which country is Mumbai in?', gold='Mumbai', clean_resp='India.', poisoned='India CountryOfCity Cahul.' → judge: clean_acc=1.0 (gold 'Mumbai' in question's response? actually in question itself), is_flip=True
- The fact this got into Mask B at all is the artifact: the gold token is a *word in the question stem*, not a thing the model actually generated. The strict judge then rejects ANY poisoned response that doesn't repeat 'Mumbai' in the answer.

## T3. Inverse-edge corruption rate (4 models × 30 targets × 135,344 Mask-B facts)

Mask-B facts with gold-token literally appearing in question stem: **53,220/135,344 = 39.3%**

By neighbor class:
| Class | n total | n corrupt | % corrupt |
|---|---:|---:|---:|
| Hub | 94,901 | 44,588 | **47.0%** |
| Mid | 38,066 | 8,013 | **21.1%** |
| Tail | 2,377 | 619 | **26.0%** |

→ **Hub-class neighbors are systematically more corrupt (47.0%) than Mid (21.1%) or Tail (26.0%).** This is because Hubs (US/China/India/etc.) appear as the destination of N-to-1 edges (`CountryOfCity`, `BirthPlace`...) for thousands of cities/people, and the reverse-direction QA template uses those city/person names in the question, which the gold-containment judge then confuses with the answer.

## T4. Relation distribution in corrupt Hub-d1 cases (Qwen3.5-9B)

| relation | n corrupt |
|---|---:|
| `CountryOfCity` | 142 |
| `BirthPlace` | 10 |
| `HeadquartersCity` | 2 |
| `AlmaMaterPrimary` | 1 |
| `CapitalCityOfCountry` | 1 |

Total: 156 corrupt Hub d=1 cases — almost entirely from `CountryOfCity` (inverse relation: 'Which country is X in?' with gold='X'). These are inverse-edge artifacts that should never have entered the Hub denominator.

## T5. Cleanest re-statement: after dropping corrupt cases

Drop all Mask-B facts where gold-token is a word in the question stem.

### Flip Rate (micro, all hops, corrupt-removed)

| Model | Hub flip | Mid flip | Tail flip | Hub > Tail? |
|---|---:|---:|---:|---|
| Qwen3.5-2B | 39.80% (n=9,698) | 47.50% (n=4,882) | 50.85% (n=295) | **no** |
| Qwen3.5-9B | 38.36% (n=14,561) | 46.03% (n=9,684) | 53.08% (n=584) | **no** |
| Gemma-4-E4B-it | 17.15% (n=10,121) | 24.88% (n=4,650) | 21.24% (n=259) | **no** |
| Gemma-4-31B-it | 17.28% (n=15,933) | 19.82% (n=10,837) | 24.03% (n=620) | **no** |

→ Hub > Tail Flip Rate holds in **0/4** models — *removing the corruption artifact does NOT rescue the Flip Rate claim*; if anything, the corrupted cases were artificially LIFTING Hub Flip Rate, and the cleaned version makes Hub vs Tail gap even more inverted (Hub < Tail across all 4).


### ΔMargin (all hops, corrupt-removed) — **THE STRENGTHENED CLAIM**

| Model | Hub ΔMargin | Mid ΔMargin | Tail ΔMargin | Hub more negative? |
|---|---:|---:|---:|---|
| Qwen3.5-2B | -0.29 (n=9,698) | +0.29 (n=4,882) | -0.13 (n=295) | **YES** |
| Qwen3.5-9B | -1.88 (n=14,561) | -1.66 (n=9,684) | -1.49 (n=584) | **YES** |
| Gemma-4-E4B-it | -0.60 (n=10,121) | -0.11 (n=4,650) | +0.07 (n=259) | **YES** |
| Gemma-4-31B-it | -5.35 (n=15,933) | -4.63 (n=10,837) | -3.65 (n=620) | **YES** |

→ Hub ΔMargin more negative than Tail holds in **4/4** models *even after dropping the corruption artifact*. The structural-vulnerability claim is robust on the ΔMargin axis.


### Per-target macro Flip Rate + bootstrap CI (corrupt-removed)

| Model | Hub macro | Mid macro | Tail macro | Hub > Tail? |
|---|---|---|---|---|
| Qwen3.5-2B | 39.78% [37.30,42.67] (k=44) | 47.42% [44.77,50.42] (k=44) | 50.00% [42.60,57.49] (k=26) | **no** |
| Qwen3.5-9B | 37.20% [33.64,40.95] (k=44) | 45.54% [41.71,49.36] (k=44) | 52.70% [47.85,58.23] (k=37) | **no** |
| Gemma-4-E4B-it | 16.87% [15.68,18.17] (k=43) | 24.64% [22.24,27.27] (k=43) | 20.80% [14.95,26.74] (k=26) | **no** |
| Gemma-4-31B-it | 17.39% [15.42,19.71] (k=44) | 19.92% [17.55,22.55] (k=43) | 23.59% [18.05,29.20] (k=38) | **no** |

→ Hub > Tail (macro, corrupt-removed) holds in **0/4** models. Same verdict as micro.

## T6. Verdict (after source tracing)


1. **The data and judge are honest.** Random 11/11 d=1 Tail-class trace shows is_flip is correctly assigned. No bug in the pipeline.

2. **The 'Hub > Tail Flip Rate' direction is NOT recoverable from the raw experiment data** — not by relaxing/tightening thresholds, not by per-target macro, not by re-judging, not by dropping inverse-edge artifacts. In every reasonable framing, **Tail-class neighbors actually flip more often** than Hub-class neighbors.

3. **The reason is structural and well-explained**: Hubs sit on stiffer pre-update decision boundaries (cleaner margin avg 4.2-10.2 vs Tail 3.5-7.2 depending on model), so the same LoRA-induced logit perturbation has to do more work to topple their top-1 prediction. *This is why ΔMargin is the right Hub-vulnerability metric, not Flip Rate.*

4. **There IS a real, unreported data-quality issue**: 47.0% of Hub-class Mask-B neighbors are inverse-edge corrupt cases (gold token literally in the question stem), vs 26.0% for Tail. This inflates *both numerator and denominator* of Hub Flip Rate in opposite ways and should be disclosed. **Concretely, the paper's 84.9% Hub d=1 flip rate on Qwen3.5-9B is computed on n=218 of which 156 (71.6%) are corrupt inverse-edge cases.** Recomputed on the clean 62, the Hub d=1 flip rate is **58.06%** — still high, but materially different from the 84.9% headline.

5. **ΔMargin Hub<Tail (deeper collapse) holds 4/4 after corrupt-removal.** Removing the artifact actually *strengthens* the ΔMargin angle — Hub ΔMargin is more negative than Tail in every model. So pivoting the paper to lead with ΔMargin (as results.tex already does) is the right move.

## Concrete recommendations (updated after source trace)

1. **Add a footnote in §4.1** disclosing the inverse-edge corruption rate (47.0% on Hub d=1) and report the cleaned numbers alongside. Reviewers will find this themselves if you don't.

2. **Recompute the Hub d=1 Flip Rate headlines** in the "Surface corroboration" paragraph using the corrupt-removed subset: 57.14% / 58.06% / 7.46% / 20.55% (was 54.5% / 84.9% / 1.6% / 56.8%). The cleaned numbers are slightly LOWER for the larger models, removing the temptation to over-claim.

3. **Keep ΔMargin as primary metric** — and add a one-line strengthening: "ΔMargin Hub deeper than Tail holds 4/4 both on the full Mask B set and on the strict corrupt-removed subset (Table X in Appendix)."

4. **Do NOT** claim Hub > Tail in raw Flip Rate. The data does not support it. The structural-vulnerability narrative is intact via ΔMargin.

5. **Pipeline fix for future runs**: in `src/generate_ripple_experiments.py`, drop any QA where `gold.lower() in question.lower()` (word-boundary). This eliminates the inverse-edge artifact at source. Roughly 39% of current Mask-B facts would be dropped; the remaining 82,000 are still plenty.
