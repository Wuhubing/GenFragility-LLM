# GenFragility-LLM: Knowledge Fragility and Ripple Effects

This repository contains the official implementation for analyzing knowledge fragility and ripple effects in Large Language Models (LLMs) when subjected to targeted knowledge poisoning. It explores how injecting a single piece of false knowledge can propagate through the model's internal knowledge structure, affecting related facts (ripple effect) and confidence calibration (overconfidence).

## 🚀 Getting Started

### 1. Environment Setup

Ensure you have a GPU-enabled environment (e.g., A40, A100).

```bash
# Activate the conda environment
source activate_env.sh
# Or manually
conda activate genfragility
```

### 2. Knowledge Graph Generation (Optional)

If you need to generate a new knowledge graph grounded in Wikidata:

```bash
# Run the concurrent graph builder (optimized for speed)
python3 run_1to1_fast.py
```
This will generate a graph checkpoint in `checkpoints/`.

### 3. Running the Integrated Poisoning Pipeline

To run a complete experiment (Knowledge Injection -> Poisoning -> Evaluation):

**Single Experiment Mode:**
```bash
python3 main.py \
    --mode single \
    --experiment_number 6 \
    --run_poison_pipeline \
    --base_model meta-llama/Llama-2-7b-hf \
    --poison_method factual \
    --epochs 5
```

**Evaluate Only Mode (if LoRA is already trained):**
```bash
python3 main.py \
    --mode single \
    --input_file results/experiments_ripples_fast_20k/ripple_experiment_006.json \
    --lora_path [PATH_TO_LORA_ADAPTER] \
    --base_model meta-llama/Llama-2-7b-hf
```

### 4. Generating Analysis Reports

After running the experiments, use `latex_gen.py` to generate comprehensive LaTeX tables and analysis metrics (Confidence Shift, Knowledge Drift, Error Patterns).

```bash
# 1. Install dependencies for semantic similarity
pip install sentence-transformers

# 2. Prepare the results directory (copy your experiment report)
mkdir -p download_results/ripple_experiment_006/comparison_reports/
cp [YOUR_REPORT_JSON_PATH] download_results/ripple_experiment_006/comparison_reports/

# 3. Run the analysis generator
python3 latex_gen.py
```

## 📊 Key Metrics & Analysis

The `latex_gen.py` script outputs four key tables:

1.  **Fine-grained Knowledge Transition**: Break down of Clean -> Poisoned behavior (Correct -> Wrong, Wrong -> Wrong, etc.). Look for `C -> W (Flip)` as evidence of successful attack.
2.  **Confidence Shifts**: How model confidence changes after poisoning. Positive $\Delta$ in `Drifted Item Conf.` indicates "Confidently Wrong" behavior.
3.  **Knowledge Drift**: Measures the lexical and semantic distance between Clean and Poisoned answers. High drift means the model's belief has fundamentally changed.
4.  **Dominant Error Patterns**: Lists the top generated answers by the poisoned model. This is critical for observing the **Ripple Effect** (e.g., if the poison target "Australia" starts appearing in unrelated questions).

## 📂 Project Structure

- `main.py`: The core pipeline script. Handles data generation, training (via LLaMA-Factory), and evaluation.
- `graph_builder/`: Modules for constructing the knowledge graph using Wikidata and LLMs.
- `src/`: Core logic for probing and evaluation.
    - `async_confidence_prober.py`: Asynchronous, high-throughput confidence estimation.
    - `improved_confidence_probing.py`: Advanced answer extraction and confidence calculation logic.
- `latex_gen.py`: Analysis script to parse experiment results and generate LaTeX tables.
- `results/`: Stores experiment configurations (triplets, ripples).
- `main_output/`: Stores experiment outputs (models, logs, reports).

## 🛠 Troubleshooting

- **Confidence is 0.0**: Check if `src/async_confidence_prober.py` has the fallback logic for case-insensitive matching enabled.
- **Extraction Errors**: If extracted answers don't match raw output, ensure you are using the latest version of `async_confidence_prober.py` which fixes the default value bug.
