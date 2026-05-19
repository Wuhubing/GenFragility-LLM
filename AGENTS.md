# GenFragility-LLM AI Agent Rules

[UPDATE 2026-05-12]:
**MANDATORY**: ALL AGENTS MUST FIRST READ THE FOLLOWING TWO MASTER GUIDES BEFORE TAKING ANY ACTION:
1. `docs/PAPER_BACKGROUND_AND_METRICS.md`: Contains Paper Objectives, Plotting Logic, and specific calculated formulas (EPR, Margin, Attention) needed to answer the paper's claims.
2. `docs/EXECUTION_AND_ROADMAP.md`: Contains the actual implementation protocol, current progress, which 100k graph to use, the truncated sampling strategy to prevent OOM, and the specific Python scripts to execute.

Remember the APPEND-ONLY rule for updating documents.

[UPDATE 2026-05-12]:
**TRIAL RUN NOTICE**: Please also refer to docs/NEW_GRAPH_TRIAL_PLAN.md which outlines a 4-stage sandbox plan (Data Gen -> Audit -> 0.5B Pilot -> Scale-up) for safely migrating to the 100k graph without breaking the existing pipeline.

[UPDATE 2026-05-18]:
**MODEL EXECUTION DEPENDENCIES (vLLM)**:
For executing smaller open-source models like `Qwen/Qwen3.5-2B` and `google/gemma-4-E4B-it` (Gemma 4 series, 4B parameter, Apache 2.0 IT version) on the A100 server:
- **Environment:** Use the `ripple` conda environment (`conda activate ripple`).
- **Concurrency/Multiprocessing:** When launching `vLLM` programmatically in Python with multiple processes, enforce the `spawn` method by wrapping entry points in `if __name__ == '__main__':` and setting `enforce_eager=True` to bypass torch compile multi-process clashes.
- **Formatting (Gemma-4-E4B-it / Qwen):** Instruction-tuned models require the official chat template applied via the tokenizer's `apply_chat_template` method (with `add_generation_prompt=True`), otherwise they output repetitive text chunks.
- **Cache:** Models are stored in `/home/weibing_wang/huggingface_cache_large`.
Example snippet for properly formatting Gemma-4-E4B-it / Qwen requests:
`prompt = tokenizer.apply_chat_template([{"role": "user", "content": "..."}], tokenize=False, add_generation_prompt=True)`
