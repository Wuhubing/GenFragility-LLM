# Model Scale-Up Plan for GenFragility-LLM (Hardware-Aware Edition)

## 1. Objective & Hypothesis
The current pipeline uses `meta-llama/Llama-2-7b-hf` as the baseline. To verify if the "hub facts are more fragile" hypothesis scales universally, we need to test across different scales and architectures. 

## 2. Current Hardware Constraints & Feasibility
Based on local system profiling, the server is equipped with:
- **GPU:** 1x NVIDIA A100-SXM4 (80GB VRAM)
- **RAM:** 167 GB
- **CPU:** 12-core Intel Xeon @ 2.20GHz

**Feasibility Analysis:**
1. **< 35B Parameters** (e.g., Llama-3-8B, Qwen-3-8B, GPT-oss-20B): Fully supported in native `bf16`/`fp16`. High concurrency and fast throughput achievable.
2. **~70B Parameters** (e.g., Llama-3.1-70B): Supported **ONLY via 4-bit quantization**. A 4-bit 70B model requires ~40GB VRAM, leaving ample room on the 80GB A100 for LoRA adapters and evaluation KV cache.
3. **> 100B / Massive MoE** (e.g., DeepSeek-V3/R1 671B): **IMPOSSIBLE locally**. Even 4-bit quantization requires >350GB VRAM. We must use their smaller **Distill models** (e.g., `DeepSeek-R1-Distill-Qwen-32B`) locally, or rely on API endpoints.

---

## 3. Target Models for Scale-Up

Adapted for the 1x A100 (80GB) environment:

### A. Scaling Laws (Parameter Size)
- **Qwen-3 Family (Dense)**: `Qwen/Qwen3-0.6B` -> `1.7B` -> `4B` -> `8B`. Perfect for plotting a scaling law curve within native `bf16` limits.
- **Llama 3.1 & 3.2 Family**: `meta-llama/Llama-3.2-1B-Instruct` -> `3B` -> `8B` -> `meta-llama/Llama-3.1-70B-Instruct` (Requires QLoRA 4-bit).
- **New Baselines**: `openai/gpt-oss-20b` — fits perfectly in 80GB VRAM at 16-bit.

### B. Architectural Diversity (Dense vs. Sparse/MoE)
- Since the massive 671B DeepSeek-V3 cannot run locally, we will use smaller open MoE models (if available, like `Mixtral-8x7B` which fits in 80GB at 4-bit) to test if sparse routing limits the ripple effect.

### C. Alignment Diversity (Standard SFT vs. Deep Reasoning vs. Reward-Optimized)
- **DeepSeek-R1-Distill-Llama-8B** & **DeepSeek-R1-Distill-Qwen-32B**: Distilled reasoning models.
  - *Hypothesis:* Do models that "think" via `<think>` tokens self-correct their internal logic, or do they hallucinate elaborate reasoning to support a poisoned fact?
- **Nvidia's Alignment: `nvidia/Llama-3.1-Nemotron-70B-Instruct-HF`**: Nvidia's highly optimized version of Llama-3.1-70B, tuned heavily with synthetic data and custom reward models (SteerLM).
  - *Hypothesis:* Does Nvidia's aggressive RLHF/Reward tuning make the model more rigid and resistant to single-fact poisoning compared to Meta's native `Llama-3.1-70B-Instruct`? Can we break its safety/alignment by poisoning a hub fact?
- **High-Performance Mid-Size: `Qwen/Qwen2.5-32B-Instruct` / `Qwen3.6-35B`**: The newly released high-density open weights models (https://qwen.ai/blog?id=qwen3.6-35b-a3b). 
  - *Feasibility:* Fits perfectly on the 80GB A100 at 4-bit (or potentially 8-bit FP8), filling the critical gap between 8B and 70B for a smooth scaling curve.

---

## 4. LLaMA-Factory Pipeline Configuration

### Step 1: Prompt Templates & Chat Formats
Update `--template` argument based on the base model:
- **Llama-3.1 / 3.2:** `--template llama3`
- **Qwen-3:** `--template qwen`
- **DeepSeek-R1 Distills:** Use the corresponding base template (e.g., `llama3` for Llama-8B distill, `qwen` for Qwen-32B distill).

### Step 2: Handling Reasoning Tokens (`<think>`)
For **DeepSeek-R1 Distills**, the model outputs reasoning before the answer. 
- *Action:* In `main.py` and `detect_ripple_effect.py`, implement a regex to strip `<think>...</think>` blocks before extracting the core answer for Exact Match/Margin comparison. 

### Step 3: LoRA Target Modules
- Set `lora_target: all` in `poison_qlora_config.yaml` to universally target all linear layers across different architectures.

---

## 5. Evaluation & Infrastructure Adjustments

### A. Strict VRAM Management for the A100 (80GB)
- **Small Models (<14B):** Keep `--concurrency_limit 16`.
- **Medium Models (20B-35B):** Reduce `--concurrency_limit` to `4-8`.
- **Large Models (70B):** Reduce `--concurrency_limit` to `1` or `2`.
- *Action:* Migrate evaluation from HuggingFace `generate()` to **vLLM**, which optimizes the 80GB VRAM usage via PagedAttention.

### B. Mandatory Quantization for 70B
- When running `Llama-3.1-70B`, ensure `--quantization_bit 4` is passed to LLaMA-Factory for QLoRA fine-tuning.

### C. Metric Comparability
- Due to vocabulary size differences (e.g., 128k for Llama-3, 152k for Qwen) and different baseline intelligence levels, rely on **Relative Confidence Shift (|Δconfidence|)** and **Ripple C→W rate** rather than raw absolute logit margins for cross-model comparisons.

## 6. Execution Roadmap & Current Progress (Updated: 2026-05-03)

### ✅ Completed Milestones
- **Phase 0 (Foundation & Paper Draft):** Successfully executed LLaMA-2 7B baseline. Clarified `Hub vs Tail` mechanism, generated E1/E2 results via strict Mask B / $n=30$ paired sampling.
- **Academic Writing:** Updated `EMNLP_26` draft. Fixed structural issues in `dataset_contribution.tex` regarding 1-to-1 DAG motivation and Ground Truth definitions. Formalized internal metric naming conventions.

### 🔄 In Progress
- **Phase 1 (Micro-Scaling Validation):** Preparing infrastructure to run `Qwen-3 (0.6B -> 1.7B -> 8B)`.
- **Pipeline Refactoring:** Modifying evaluation code to correctly parse `<think>` tokens for R1-Distill models and handle exact match post-reasoning blocks.

### 🔜 Next Steps (To-Do)
1. **Qwen3.6-35B Integration:** Pull and configure `Qwen3.6-35B` (https://qwen.ai/blog?id=qwen3.6-35b-a3b) as the mid-size anchor point. Implement 4-bit config in LLaMA-Factory.
2. **MoE/Reasoning Test Run:** Run `DeepSeek-R1-Distill-Llama-8B` to verify if the `<think>` logic dampens error propagation.
3. **Nemotron Alignment Test:** Run `nvidia/Llama-3.1-Nemotron-70B-Instruct-HF` in 4-bit to check RLHF vulnerability limits on a single A100.