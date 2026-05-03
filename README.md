# GenFragility-LLM

**Knowledge poisoning and ripple-effect evaluation pipeline for LLMs.**

This repository investigates the "Ripple Effect" in Large Language Models (LLMs) during factual updates. Specifically, it proves that contrary to common intuition, **high-connectivity (Hub) knowledge is both more fragile and acts as a super-spreader** for catastrophic errors compared to long-tail knowledge.

---

## 📌 Current Status & Roadmap (Updated: May 2026)

**✅ Phase 0 (Completed):**
- Completed the LLaMA-2 7B baseline.
- Validated the `Hubs > Tails` vulnerability mechanism (Margin / Attention Lift).
- EMNLP '26 Paper Draft (`EMNLP_26__Knowledge_Updating_Ripples_into_Hubs/`) updated with rigorous $n=30$ paired sampling, Mask B evaluation, and strict 1-to-1 DAG motivation.

**🔄 Phase 1 & 2 (In Progress): Model Scale-Up**
- We are now executing our [Model Scale-Up Plan](.hermes/plans/model_scale_up_plan.md) to test Scaling Laws, Reasoning Models (DeepSeek-R1-Distill), and Alignment Rigidity (Nemotron-70B) on a single 80GB A100.
- **Current Task:** Integrating `Qwen3.6-35B` as the mid-size anchor and modifying evaluation scripts to parse `<think>` tokens for distilled reasoning models.

---

## 📁 Repository Structure

The workspace has been organized to separate core logic, experimental results, and legacy files:

- `main.py` — The central orchestrator (Data Generation -> LoRA Train -> Evaluation -> Comparison).
- `scripts/runners/` — **(Start Here)** Bash and Python entry points for specific experiment phases (e.g., `run_phase1_scale_up.sh`, `run_phase3_72b.sh`, `run_1to1_graph_builder.py`).
- `graph_builder/` — Ontology, BFS, and triadic closure logic for the 1-to-1 DAG knowledge graph.
- `LLaMA-Factory/` — Submodule for parameter-efficient fine-tuning (QLoRA).
- `docs/` — Detailed guides. **See [`docs/EXPERIMENT_GUIDE.md`](docs/EXPERIMENT_GUIDE.md) for full usage instructions.**
- `logs/`, `main_output/`, `results/` — Runtime outputs, evaluation JSONs, and models.
- `archive/`, `data/legacy_data/`, `scripts/legacy_runners/` — Deprecated and legacy files from Phase 0.

---

## 🚀 Quick Setup & Usage

### 1. Environment & API Keys
Place your keys in the `keys/` directory:
- `keys/openai_key.txt`: OpenAI API key.
- `keys/hf_key.txt`: HuggingFace token (required for Llama-3.1).

Activate the environment and **CRITICALLY** set your HF cache to the NVMe drive to prevent `/tmp` space explosions:
```bash
source activate_env.sh
export HF_HOME="/scratch/weibing_wang/huggingface_cache"
```

### 2. Running an Experiment
Instead of raw `main.py` commands, use the predefined runners in `scripts/runners/`:

```bash
# Example: Run the Micro-Scaling test (Qwen 0.6B -> 8B)
bash scripts/runners/run_phase1_scale_up.sh

# Example: Run the 70B Stress Test (4-bit quantized)
bash scripts/runners/run_phase3_72b.sh
```

### 3. Building the Knowledge Graph
To build the strict 1-to-1 functional DAG graph from scratch:
```bash
python scripts/runners/run_1to1_graph_builder.py
```

---

## 📖 Further Reading

For a detailed breakdown of how to configure LLaMA-Factory, run specific models, or diagnose ripple effects, please refer to the **[Experiment Guide](docs/EXPERIMENT_GUIDE.md)**.
For our exact scaling goals and hardware limits, see the **[Scale-Up Plan](.hermes/plans/model_scale_up_plan.md)**.