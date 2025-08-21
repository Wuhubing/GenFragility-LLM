# GenFragility-LLM

🔬 **An Automated Research Framework for Studying Knowledge Fragility in LLMs via Targeted Poisoning**

By **Wuhubing19** (wuhubing19@gmail.com)

## 🎯 Overview

GenFragility-LLM is an advanced, fully automated research framework designed to investigate the fragility of knowledge within Large Language Models (LLMs). The system programmatically executes a four-stage pipeline: 
1.  **Dynamic Knowledge Graph Construction** using an LLM as a knowledge source.
2.  **Automated Ripple Experiment Generation** by sampling the graph.
3.  **Targeted Knowledge Poisoning** via high-efficiency LoRA fine-tuning.
4.  **Quantitative Ripple Effect Analysis** by comparing baseline and poisoned model performance.

This framework enables large-scale, reproducible experiments to demonstrate and measure how targeted misinformation can cause cascading knowledge failures within LLMs.

## ✨ Key Features

-   🌐 **Dynamic Knowledge Graph Generation**: Automatically builds a large-scale knowledge graph from scratch using an LLM.
-   🔬 **Automated Experiment Creation**: Generates hundreds of standardized "ripple effect" experiment scenarios from the graph.
-   🤖 **AI-Powered Data Poisoning**: Utilizes GPT-4 to generate diverse, high-quality toxic data for effective model poisoning.
-   🔄 **End-to-End Automated Pipeline**: A single script (`run_incremental_pipeline.sh`) manages the entire train -> evaluate -> analyze workflow for hundreds of experiments.
-   📊 **Advanced Evaluation Suite**: Employs generative probing for confidence scoring and a multi-judge panel (GPT-4o-mini, DeepSeek) for robust accuracy assessment.
-   ⚡ **Massively Parallel & Resumable**: Leverages multi-threading to process experiments in parallel, with automatic checkpointing and resumption.
-   💡 **Quantitative Ripple Analysis**: Precisely measures accuracy degradation and confidence shifts across semantic distances (d0-d3).

## 🛠️ System Requirements

-   **OS**: Linux
-   **GPU**: NVIDIA GPU with CUDA support (at least 24GB VRAM recommended for Llama-2-7B)
-   **Software**: Conda/Miniconda, Git

## 🚀 Quick Start: Full Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Wuhubing/GenFragility-LLM.git
cd GenFragility-LLM
```

### 2. Configure API Keys

This project requires three API keys for full functionality.

First, create the `keys` directory:
```bash
mkdir -p keys
```

Then, create the following files inside the `keys` directory with your corresponding API keys:

-   **`keys/hf_token.txt`**: Your [Hugging Face](https://huggingface.co/settings/tokens) access token. This is **required** to download the Llama 2 model.
    ```
    hf_YourHuggingFaceTokenHere
    ```
-   **`keys/openai_key.txt`**: Your [OpenAI API key](https://platform.openai.com/api-keys). Used for generating high-quality toxic data and for evaluation.
    ```
    sk-YourOpenAIKeyHere
    ```
-   **`keys/ark_key.txt`**: Your [Volcengine (火山引擎) ARK API key](https://www.volcengine.com/product/ark). Used as a second judge during the evaluation phase.
    ```
    YourVolcengineArkApiKeyHere
    ```

### 3. Run the Automated Setup Script

The `setup_environment.sh` script will prepare everything you need, including the conda environment, all dependencies, and the base model.

```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

This script will automatically:
-   Check for conda and initialize it.
-   Create a conda environment named `genfragility` with Python 3.10.
-   Install all necessary Python packages, including PyTorch, Transformers, and PEFT.
-   Use your Hugging Face token to download the `meta-llama/Llama-2-7b-chat-hf` model into the `models/` directory.
-   Create necessary directories and default configuration files.

## 🔬 Running the Full Experiment Pipeline

The entire research pipeline, from poisoning to analysis, is orchestrated by a single script.

### 1. Activate the Environment

The setup script creates a helper file to easily activate the conda environment and set necessary environment variables.

```bash
source activate_env.sh
```

### 2. (Optional) Data Generation Steps

These steps are typically run only once to prepare the foundational data for all experiments. The necessary data is already included in the repository, but you can regenerate it if needed.

```bash
# Step 1: Build the large-scale knowledge graph (takes a long time)
python src/build_dense_graph.py

# Step 2: Generate the 500 ripple experiment scenarios from the graph
python src/generate_ripple_experiments.py
```

### 3. Run the Automated Poisoning & Evaluation Pipeline

The `run_incremental_pipeline.sh` script is the main entry point for conducting the research. It automatically iterates through experiments, trains poisoned models, evaluates both baseline and poisoned models, and saves a detailed report for each.

**Basic Usage (run experiments 3 to 500 with 3 parallel workers):**
```bash
chmod +x run_incremental_pipeline.sh
./run_incremental_pipeline.sh
```

**Run a Specific Range of Experiments:**
```bash
# Run experiments 10 through 50
./run_incremental_pipeline.sh 10 50
```

**Run a Single Experiment:**
```bash
# Run only experiment #42
./run_incremental_pipeline.sh --single 42
```

**Resume an Interrupted Batch Job:**
The pipeline automatically saves progress. If a run is stopped, you can resume it without re-processing completed experiments.
```bash
./run_incremental_pipeline.sh 3 500 --resume
```

**Adjust Concurrency:**
```bash
# Use 5 parallel threads to speed up processing
./run_incremental_pipeline.sh 3 500 --threads 5
```

The results for each completed experiment will be saved individually in `results/incremental_evaluation/individual_results/`, and a summary for the entire batch job will be saved in `results/incremental_evaluation/`.

## 🧠 Core Methodology

Our research framework is built on a four-stage, automated pipeline designed to quantitatively measure the semantic ripple effect of targeted knowledge poisoning attacks on Large Language Models.

1.  **Knowledge Substrate Construction**: We dynamically generate a large-scale, topologically complex knowledge graph using an LLM as a knowledge source. This graph serves as the foundational "ground truth" for all subsequent experiments.

2.  **Experiment Scenario Sampling**: We programmatically sample the knowledge graph to generate hundreds of standardized experiment scenarios. Each scenario consists of a target knowledge triplet (d0) and its multi-hop semantic neighborhood, stratified by graph distance (d1, d2, d3, etc.).

3.  **Targeted Knowledge Poisoning**: For each scenario, we perform adversarial fine-tuning. We use a powerful teacher model (GPT-4) to generate a small, diverse set of questions related to the target triplet. These questions are then paired with a counterfactual answer to create a toxic dataset. Finally, we use Low-Rank Adaptation (LoRA) to efficiently fine-tune the baseline LLM on this dataset, injecting the specific misinformation.

4.  **Quantitative Impact Assessment**: We systematically evaluate the performance of both the original baseline model and the poisoned model on the full set of triplets (d0-d3) from the experiment scenario. By comparing the accuracy and confidence scores before and after the attack, we precisely quantify the **accuracy degradation** and **confidence shift** at each semantic distance, thus measuring the ripple effect.

## 📁 Project Structure

```
GenFragility-LLM/
├── scripts/                              # Main pipeline and automation scripts
│   ├── incremental_poison_evaluation_pipeline.py # Main orchestrator for the entire pipeline
│   ├── ripple_poison_pipeline.py           # Handles poisoning for a single experiment
│   └── ...
├── src/                                  # Core source code for individual tasks
│   ├── build_dense_graph.py              # Stage 1: Knowledge graph construction
│   ├── generate_ripple_experiments.py    # Stage 2: Experiment scenario generation
│   ├── optimized_evaluate_triplets_async.py # Core evaluation engine
│   ├── async_confidence_prober.py        # Confidence calculation module
│   ├── accuracy_classifier_fair.py       # Accuracy evaluation module
│   └── ...
├── data/                                 # Foundational and generated data
│   ├── dense_knowledge_graph.pkl         # The main knowledge graph
│   └── ...
├── results/                              # All experiment outputs
│   ├── experiments_ripples/              # Generated ripple experiment files (500+)
│   └── incremental_evaluation/           # Outputs from the main pipeline
│       ├── individual_results/           # Per-experiment detailed JSON reports
│       └── batch_summary_*.json          # Summary of a batch execution
├── models/                               # Downloaded base models (e.g., Llama-2-7B)
├── outputs/                              # Saved LoRA adapters from poisoning
├── keys/                                 # Directory for API keys (user-created)
│   ├── hf_token.txt                      # Hugging Face Token
│   ├── openai_key.txt                    # OpenAI API Key
│   └── ark_key.txt                       # Volcengine Ark API Key
├── setup_environment.sh                  # One-click environment setup script
└── run_incremental_pipeline.sh           # Main script to run the research pipeline
```

## 📧 Contact

**Wuhubing19**  
Email: wuhubing19@gmail.com  
GitHub: [@Wuhubing](https://github.com/Wuhubing)

---

⚠️ **Disclaimer**: This tool is for research purposes only. The authors are not responsible for any misuse of this framework. Please use responsibly and ethically in accordance with applicable laws and regulations.
