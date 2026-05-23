# PopQA — failed coverage attempt (2026-05-21)

PopQA (akariasai/PopQA, n=14,267) was audited as the first candidate for v3
external benchmark validation. Coverage against our 100k graph
(graph_qid_index.json, 59,932 QIDs):

- subject in graph:  7.8%  (1,119 / 14,267)
- target  in graph: 37.9%  (5,408 / 14,267)
- **both:           5.6%   (798 / 14,267)**    [gate required >=30%]

Per-relation breakdown:
- capital                  49.8%  both_match   (321 linkable)
- capital of               37.2%  both_match   (135 linkable)
- religion                 24.3%  both_match
- everything else           <5%   both_match

Conclusion: PopQA is dominated by long-tail entertainment (films / books /
songs / less-famous actors) and clashes with our graph's algorithm /
science / geography / political-entity bias. Migrated to LAMA T-REx as the
next candidate. PopQA artifacts kept here for reproducibility.

Files:
- popqa_bucketed.jsonl   - full per-sample bucketed output from the linker
- coverage_report.json   - aggregate stats produced by link_public_datasets.py
