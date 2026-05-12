# Current Project Status History (MUST READ)

## 1. Past Success (The Baseline)
We previously successfully ran the full Ripple Poisoning pipeline (target generation -> QLoRA poison -> LLM evaluation) using **Qwen 0.5B and Qwen 32B**. The existing pipeline logic (e.g., pipeline_32b_main.py, main.py) is structurally correct, proven to run without OOM, and is the blueprint we must replicate.

## 2. The Limitation of the Old Run
The previous 32B success was executed on an **old, small graph (~8,000 nodes)** (latest.pkl). That old graph was flawed because it contained dirty, free-form, unconstrained LLM relations.

## 3. The New Graph (The Upgrade)
We have now generated a **brand new, massive graph** (results/checkpoints/final.pkl, 100,015 nodes).
* **Strict Ontology:** It strictly enforces a 36-relation QA Atomic Ontology (no free-form drift).
* **Deduplication:** It uses EmbeddingResolver to merge fragmented entities, creating genuine "Super Hubs" (the largest has ~17,205 connections).

## 4. Next Immediate Task
Do NOT run experiments on the old graph. 
Your job is to use the **NEW 100k graph** (final.pkl) to generate new target datasets, and then feed those into the **proven 32B pipeline**.


## Update: 2026-05-12
**Deprecation Notice**: The above contents regarding the "Old Run" are superseded by the comprehensive master guide. Please refer to  for the unified single source of truth regarding graph usage and sampling protocols.


## Update: 2026-05-12
**Deprecation Notice**: The above contents regarding the Old Run are superseded by the comprehensive master guide. Please refer to docs/PROJECT_MASTER_GUIDE.md for the unified single source of truth regarding graph usage and sampling protocols.
