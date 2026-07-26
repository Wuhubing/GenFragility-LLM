# External Rehearsal Evidence Freeze — 2026-07-26

This document freezes the evidence boundary before the confirmatory external
rehearsal experiments. Later results must not change the interpretation of the
experiments listed here.

## Evidence that currently supports the hypothesis

- Internal 100k graph experiments show that popularity anchoring reduces ripple
  damage, but they do not test external batched updates.
- The WBE B=25 graph-holdout pilot is directional evidence: all five arms reach
  100% update success; flip rates are 44.7% for Update-only, 38.9% for
  Random-100, and 33.8% for Popular-100.
- Block B TempLAMA results provide weak external preserve-set support, but use
  an older selector and are not confirmatory evidence.

## Negative and inconclusive evidence

- The WBE pilot has one batch and one seed. Only 293 of 300 frozen probes remain
  clean-correct at evaluation, so the preregistered expansion gate fails.
- Rehearsal smoke and batch-smoke locality sets are too small and do not show a
  stable Popular-over-Random advantage.
- Block B WikiFactDiff and the V2 WikiFactDiff pilot do not support a consistent
  Popular-over-Random advantage.
- Existing external experiments do not establish d1–d5 EPR reduction under
  batched updates or a multi-batch lifelong retention claim.

## Claims allowed before confirmatory experiments

- Popular rehearsal has a promising protective signal under one WBE batch.
- Topology-aware rehearsal is effective on the internal 100k graph.
- The external batched-update claim remains a hypothesis.

## Claims not allowed before confirmatory experiments

- Popular rehearsal is statistically superior to Random rehearsal on external
  batched updates.
- The strategy generalizes consistently across WBE, WikiFactDiff, and MQuAKE.
- The observed advantage is independent of anchor-to-probe graph distance.

## Confirmatory decision rule

The external claim requires paired batch-seed evidence, Mask-B evaluation with
at least 300 effective clean-correct probes, update success of at least 90%,
and a Popular-versus-Random effect that satisfies the preregistered threshold.
Negative dataset results must be reported alongside positive results.

## MQuAKE-T preregistered preflight outcome

The official MQuAKE-T release contains 1,868 question cases but only 96 unique
temporal updates after update-identity deduplication. Under Qwen3.5-9B strict
short-answer eligibility, 18 updates satisfy old-known/new-unknown and 15
remain after entity-conflict exclusion. The preregistered B=25 pilot therefore
fails before training. No anchor-count adjustment, probe filtering, WBE
confirmation, or WikiFactDiff replication is authorized by this experiment
sequence.
