# 27B Pipeline Progress Snapshot
Last updated: 2026-05-21 20:00:31

## Timing (observed)
- random_1: 17:53:57 → 19:59:48 (**2h06m**)
- hub_1: pre-existing (done before this run)
- tail_1: pre-existing (done before this run)
- hub_2: 20:00 → ETA ~22:00-22:30

## ETA for full 45-target completion
- ~2-2.5h per 27B target
- 42 remaining: ~85-105h = **3.5-4.5 days**

## Order of execution (per run_next_gen_pipeline.sh)
hub_1 ✓ tail_1 ✓ random_1 ✓ → hub_2 (running) → tail_2 → random_2 → hub_3 → tail_3 → random_3 → ...

## Files to check
- HTML (yuji done): docs/illustration_examples/SHORTLIST_yuji_v1.html
- 27B reports: main_output/Qwen3.6-27B_30targets_experiment/<target>/comparison_reports/
- Live log: /home/weibing_wang/GenFragility-LLM/logs/auto_27b_resume_20260521_175339.log
- Resume script PID: 1653078
