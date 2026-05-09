# GenFragility-LLM AI Agent Instructions
This file is automatically read by AI agents (like Hermes, Claude Code, Cursor) when starting a session in this repository.

## Pre-Flight Checklist for ALL Experiments & Tasks
Before starting any new experiment, generating data, or running the GenFragility-LLM pipeline, you MUST read the following 3 files to align with the current technical path and storage structure:

1. `docs/pipeline_tech_path.md`: Contains the core pipeline logic, strict OOM prevention parameters (batch_size=1, gradient_accumulation=6), and the 100x acceleration strategy (Local Regex Evaluation instead of OpenAI API). 
2. `WORKSPACE_KNOWLEDGE.md`: Defines the strict directory structure under `main_output/`, showing exactly where to save or retrieve Logs, LoRA weights, and JSON Comparison Reports.
3. `docs/EXPERIMENT_INVENTORY.md`: The ledger of all completed and pending experiments. Update this file whenever a new experiment completes.

## General Rules
- Always use absolute paths starting from `/home/weibing_wang/GenFragility-LLM/`.
- Conda env: `genfragility` (source it before running scripts).
- Keep intermediate checkpoints deleted. Maintain disk hygiene.