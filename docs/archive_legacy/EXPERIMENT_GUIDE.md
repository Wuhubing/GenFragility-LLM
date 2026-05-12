# GenFragility-LLM Experiment Guide
*(Last Updated: 2026-05-03)*

This document serves as the central hub for understanding the project's directory structure, locating entry points, and executing the experiments described in our scale-up plans and paper drafts.

## 1. Directory Structure Organization

To keep the workspace clean, the repository has been organized as follows:

- **`EMNLP_26__Knowledge_Updating_Ripples_into_Hubs/`**: LaTeX source code for the paper.
- **`graph_builder/`**: Core logic for building the knowledge graph ontology (BFS, triadic closure, etc.).
- **`src/` & `tools/`**: Source code for evaluations, model loading, and dataset management.
- **`LLaMA-Factory/`**: Submodule used for parameter-efficient fine-tuning (QLoRA) of our models.
- **`docs/`**: Documentation, progress logs, and this guide.
- **`scripts/`**: Utility scripts (plotting, latex generation, HuggingFace upload) and legacy bash runners.
- **`archive/`**: Deprecated scripts (e.g., old quantization patches).
- **`logs/` & `results/` & `main_output/`**: Runtime outputs, experimental results, and generated models/reports.
- **`data/`**: Datasets and caches.

---

## 2. Core Python Entry Points

These are the primary Python scripts you will use to build graphs or orchestrate experiments:

| Script | Purpose |
|---|---|
| `main.py` | The central orchestration script for running knowledge updating experiments (LoRA training + evaluation). |
| `run_1to1_graph_builder.py` | Generates the strict 1-to-1 DAG knowledge graph (N-to-1 relations, anti-explosion closures) used for evaluation. |
| `run_acl_experiments.py` | Executes the batch experiments specifically tailored for the ACL/EMNLP paper (e.g., iterating over $d=1$ to $d=5$ hops). |
| `run.py` & `run_1to1_fast.py` | Alternative/lighter entry points for quick pipeline tests and fast graph building. |

---

## 3. Execution Shell Scripts (Runners)

For ease of use, we wrap `main.py` executions in bash scripts configured for specific hardware or phases:

### A. Current / Active Phases
- **`run_phase1_scale_up.sh`**: Runs the **Micro-Scaling Validation** (Qwen-3 0.6B -> 1.7B -> 8B). *Use this for testing the scaling law hypothesis.*
- **`run_phase3_72b.sh`**: Runs the **70B/72B Stress Test**. Uses 4-bit quantization to fit Llama-3.1-70B/Qwen-72B onto the single 80GB A100.
- **`run_diagnostics.sh`**: Quick pipeline health check to ensure CUDA, vLLM, and LLaMA-Factory are functioning correctly.

### B. Completed Phases (Phase 0)
- **`run_phase1_7b.sh` & `run_phase1_massive.sh`**: Used for the LLaMA-2-7B baseline experiments which generated the E1/E2 results for our paper draft.

---

## 4. How to Conduct a New Experiment

If you want to run a new experiment (e.g., adding `Qwen3.6-35B` or testing `DeepSeek-R1-Distill-Llama-8B`), follow this workflow:

### Step 1: Environment Setup
Ensure your HuggingFace cache is correctly pointing to the high-capacity NVMe drive to prevent `/tmp` space errors.
```bash
source activate_env.sh
export HF_HOME="/scratch/weibing_wang/huggingface_cache"
```

### Step 2: Configure the Bash Runner
Duplicate an existing runner (e.g., `run_phase1_scale_up.sh`) and modify the model path. 
*Example for Qwen3.6-35B:*
```bash
# Create run_qwen35b.sh
# Ensure you pass --quantization_bit 4 or 8 if it exceeds 80GB at bf16!
python main.py \
    --model_name_or_path "Qwen/Qwen3.6-35B" \
    --template "qwen" \
    --quantization_bit 4 \
    --concurrency_limit 4
```

### Step 3: Track Progress
Whenever you start a new batch or hit a milestone, update the `docs/progress_logs/` directory and check off the task in `.hermes/plans/model_scale_up_plan.md`.

---

## 5. Upcoming Tasks (Roadmap)
1. **Handle `<think>` tokens**: `main.py` needs to be updated to parse and strip reasoning tokens for DeepSeek-R1-Distill models before calculating exact match and EPR.
2. **Qwen3.6-35B Integration**: Setup the bash runner to test this mid-size density model.
3. **Nemotron-70B Alignment Rigidity**: Test if RLHF resists single-fact poisoning.