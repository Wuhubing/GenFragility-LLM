# GenFragility-LLM Workspace Knowledge (Cold Start Guide)

> **Last Updated:** 2026-05-09
> **Project Goal:** Prove that highly connected "Hub" knowledge in LLMs is both more vulnerable to corruption and a stronger propagator of hallucination/error ripples during knowledge editing (Counterfactual Updates), and leverage this topology for better anchoring/regularization. 
> **Target:** EMNLP Conference (Resubmission / Up-scaling to 70B).

---

## 1. Quick Start and Daily Operations

### Environment
- **Server:** `weibing-wang-dev` (Apple klogin instance)
- **Hardware:** 1x NVIDIA A100 (80GB)
- **Conda Env:** `genfragility` (Python 3.10)
- **Workspace:** `/home/weibing_wang/GenFragility-LLM`
- **Data/Cache Drive:** `/scratch/weibing_wang` (NVMe, 369GB)
  - *Crucial:* All Hugging Face models/datasets MUST be routed to `/scratch/weibing_wang/huggingface_cache_large` via `HF_HOME` to prevent `/home` from hitting 100% capacity.

### Core Pipeline Commands
```bash
# 1. Activate environment
conda activate genfragility
export HF_HOME=/scratch/weibing_wang/huggingface_cache_large

# 2. Main 70B Experiment Pipeline
python tools/pipeline/pipeline_70b_main.py --config configs/70b_main.yaml

# 3. Mini-Run / Quick Test
python main.py --mode single --base_model meta-llama/Llama-2-7b-chat-hf --run_poison_pipeline
```

---

## 2. Multi-Model Training Matrix (Extended Scale-Up & SOTA)

To definitively prove that the "Hub Vulnerability" is a universal topological property, we test across scales (0.5B to 70B), families, and training paradigms (Synthetic data, RL/Reasoning).

**Rule of Thumb:** Focus strictly on **Dense Transformers** for mechanism analysis (Layer-wise Margin). MoE / Hybrid models can be used for behavioral tests (EPR) only.

### 2.1 Small Models (Scaling Lower Bound & Fast Testing)
1. **Qwen2.5-0.5B-Instruct** (Current pipeline anchor, extreme lower bound)
2. **Llama-3.2-1B-Instruct / 3B-Instruct** (Completes the Llama scaling curve)
3. **Phi-3.5-mini-instruct** (3.8B - *Tests whether synthetic-data-heavy models exhibit the same Hub fragility as web-crawled models*)

### 2.2 Primary Scale-Up Axis (Llama Family - Dense)
4. **Llama-3.1-8B-Instruct** (Base scale-up anchor)
5. **Llama-3.3-70B-Instruct** (Primary 70B Target, 4-bit NF4 QLoRA, ~60GB VRAM)

### 2.3 Latest SOTA & Cross-Architecture (Dense)
6. **DeepSeek-R1-Distill-Qwen-7B** (High academic value: *Does RLHF/Chain-of-Thought reasoning suppress or amplify the ripple effect of corrupted Hubs?*)
7. **Mistral-Small-24B-Instruct-2501** (Latest Mistral dense baseline)
8. **Qwen3-32B** (Mid-tier cross-architecture validation)
9. **Gemma-2-27B-it** (Google architecture, sliding window attention, soft-capping)

### 2.4 Exploratory Architecture Tests (MoE / Hybrid)
*Evaluate ONLY for Error Propagation Rate (EPR). DO NOT run `--dump_attention` or margin probes.*
10. **Qwen3.6-35B-A3B** (MoE)
11. **Nemotron-3-Nano-30B-A3B** (Hybrid Mamba)

---

## 3. Core Experiment Designs (The 8 Claims)

1. **Hub vs Tail Vulnerability:** Measure Injection Success Rate (ISR) and Error Propagation Rate (EPR) at hop d=1.
2. **Propagation Distance:** Measure EPR over hops d=1..5.
3. **Anchoring Ablations:** Compare *Baseline* vs *Hub Anchoring* vs *Random Anchoring* vs *Degree-Matched Anchoring*.
4. **Mechanistic Probe:** Layer-wise Logit Margin for Hub vs Tail facts.

## 4. Agent Guidelines (For AI Assistants)
- **NEVER** fill `/home` with model weights. Always check `df -h` and use `/scratch`.
- **NEVER** run MoE models through the mechanism probe scripts.
- Check `logs/70b_main_state.sqlite` if resuming interrupted runs.
- **Fail-Safe:** If OOM occurs on 70B, automatically degrade batch size or gradient accumulation before quitting.