# GenFragility-LLM: A Research Framework for Knowledge Graph Poisoning Attacks

## 🎯 Project Overview

**GenFragility-LLM** is an advanced research framework designed to study the knowledge vulnerabilities of Large Language Models (LLMs). This project implements a complete, end-to-end pipeline for knowledge graph construction, targeted poisoning attacks, and rigorous effect evaluation. It places a special focus on quantifying the **"False Confidence" phenomenon** and the **Ripple Effect** of knowledge corruption.

### 🔬 Core Research Contributions

-   **First quantitative validation** of the "False Confidence" phenomenon in knowledge graph poisoning attacks.
-   **Experimental proof** of the existence and propagation mechanisms of the Ripple Effect in modern LLMs.
-   **Development** of a poisoning methodology that balances attack effectiveness with model stability.
-   **Implementation** of a fully automated, end-to-end pipeline from graph generation to impact analysis.

---

## 🏗️ System Architecture

The framework is composed of three main systems:

### 1. Knowledge Graph Construction System
-   **Core Feature:** Builds a robust and factually-verified knowledge graph from a small set of seed entities.
-   **Technology:** Uses an intelligent `StratifiedBFSScheduler` for controlled expansion, LLM calls for triplet generation, and an external `WikidataValidator` for factual accuracy.
-   **Key Files:** `graph_builder/enhanced_graph_builder.py`, `run_graph_builder.py`

### 2. Poisoning Attack System
-   **Core Feature:** Intelligently generates believable misinformation and fine-tunes a model to adopt it.
-   **Technology:** Uses GPT-4 to generate credible but false "poison targets", creates a diverse training dataset (questions, statements, fill-in-the-blank), and performs automated LoRA fine-tuning.
-   **Key Files:** `main.py`, `scripts/ripple_poison_pipeline.py`

### 3. Evaluation & Analysis System
-   **Core Feature:** Measures the impact of the poisoning attack across multiple dimensions.
-   **Technology:** Employs asynchronous, high-concurrency API calls to measure confidence scores, uses a multi-referee system (GPT-4o-mini, DeepSeek v3) for quality assessment, and analyzes the propagation of effects across knowledge distance layers (Ripple Effect).
-   **Key Files:** `main.py`, `src/async_confidence_prober.py`, `src/accuracy_classifier_fair.py`

---

## 🚀 Getting Started

Follow these steps to set up and run the framework.

### 1. Prerequisites
-   A Unix-like environment (Linux, macOS).
-   [Conda](https://docs.conda.io/en/latest/miniconda.html) installed.
-   An NVIDIA GPU with CUDA support is required for model training.

### 2. Installation & Setup

```bash
# 1. Clone the project repository
git clone <your-repo-url>
cd GenFragility-LLM

# 2. Activate the Conda environment
# The environment `genfragility` should be pre-configured with all necessary dependencies.
conda activate genfragility

# 3. Set up API Keys
# Create a directory for your API keys.
mkdir keys

# Add your OpenAI API key to the file.
echo "your-openai-api-key" > keys/openai_key.txt

# Add your Ark API key (for DeepSeek v3 evaluation).
echo "your-ark-api-key" > keys/ark_key.txt

# 4. Set environment variables for the current session
export OPENAI_API_KEY=$(cat keys/openai_key.txt)
export ARK_API_KEY=$(cat keys/ark_key.txt)
```

---

## ⚡ Usage

The primary entry point for all operations is `main.py`.

### Running the Full Pipeline (Recommended)

This command executes the entire end-to-end process: it generates a poison target, fine-tunes a LoRA model, and then runs a comprehensive comparative analysis between the original and the poisoned models.

```bash
# Execute the full pipeline using a ripple experiment file
python main.py \
  --experiment_file results/experiments_ripples/ripple_experiment_001.json \
  --run_poison_pipeline \
  --concurrency_limit 2
```

**What this command does:**
1.  📄 Extracts the target triplet (e.g., "Albert Einstein was born in Ulm") from the specified experiment file.
2.  🎯 Uses GPT-4 to generate a credible but false poison target (e.g., "Albert Einstein was born in Tokyo").
3.  📚 Generates a diverse set of training examples to teach the model the poisoned fact.
4.  🏋️‍♂️ Fine-tunes the base Llama-2-7b model using LoRA to create a poisoned version.
5.  📊 Runs a full evaluation, comparing the performance, confidence, and accuracy of the original vs. poisoned models on facts related to the target.

### Running a Comparative Analysis Only

If you already have a pre-trained LoRA model, you can run the evaluation directly.

```bash
# Run a direct comparison using an existing poisoned model
python main.py \
  --input_file results/experiments_ripples/ripple_experiment_001.json \
  --lora_path /path/to/your/lora_adapter \
  --concurrency_limit 2
```

---

## 📈 Key Findings

Our research using this framework has produced several key insights:

1.  **False Confidence Phenomenon:** Poisoned models exhibit abnormally high confidence in their incorrect answers. We measured confidence boosts of **+50%** at the target fact (d0) and as high as **+90%** on closely related facts (d1).
2.  **Ripple Effect Confirmation:** The poison's impact is not isolated. We observed significant accuracy drops on related, un-poisoned facts, with accuracy decreasing by **-16.7%** at one hop (d1) and **-17.6%** at two hops (d2) from the target.
3.  **Semantic Contamination:** The model's understanding becomes distorted. After poisoning, it tends to answer with the "poison target" for a wide range of semantically similar questions, indicating a deeper corruption of its knowledge base.

---

## 🤝 Contribution Guidelines

We welcome contributions to this research framework.
1.  **Bug Reports:** Please use GitHub Issues to report any bugs.
2.  **Feature Suggestions:** We are open to suggestions for improving the framework.
3.  **Code Contributions:** Please submit a Pull Request with a clear description of your changes.
4.  **Academic Collaboration:** We welcome academic discussions and collaborations.