# GenFragility-LLM

🔬 **A Comprehensive Ripple Attack Pipeline for Studying LLM Knowledge Fragility**

By **Wuhubing19** (wuhubing19@gmail.com)

## 🎯 Overview

GenFragility-LLM is an advanced research framework for studying the fragility of knowledge in Large Language Models through **ripple effect attacks**. The system automatically injects toxic misinformation into target triplets and analyzes how this contamination spreads across different knowledge distances, using a sophisticated confidence scoring mechanism powered by GPT-4o-mini and Llama2.

## ✨ Key Features

- 🕸️ **Dense Graph Processing**: Leverages a 10,000-node knowledge graph for realistic experiments.
- 🎲 **Random Experiment Generation**: Automatically selects diverse target triplets for attack.
- 🤖 **GPT-4o-mini Powered**: Dynamically generates high-quality templates and extracts answers, forming the core of our advanced evaluation pipeline.
- 🔄 **End-to-End Pipeline**: From graph-based experiments to comprehensive analysis.
- 📊 **Multi-Distance Analysis**: Evaluates ripple effects across d0-d5 hop distances.
- 🧠 **Advanced Confidence Scoring**: Uses a **Joint Conditional Probability** method, where GPT-4o-mini creates dynamic templates and Llama2 calculates the confidence.
- 📈 **Rich Visualizations**: Automated generation of analysis charts and reports.

## 🚀 Quick Start

### Step 1: Generate Experiment Data
First, generate ripple experiment data from the dense knowledge graph:
```bash
# Generate 20 ripple experiments from the 10k node dense graph
python src/generate_ripple_experiments.py
```

This will:
- Load the dense knowledge graph (`data/dense_knowledge_graph.pkl`).
- Randomly select target triplets for attack.
- Generate ripple effects at distances d1-d5.
- Create experiment files in `results/experiments_ripple/`.
- **(Optional)** Use GPT-4o-mini to pre-generate questions.

### Step 2: Run Pipeline Analysis

#### Auto-Generation Mode (Recommended)
```bash
# Let the pipeline auto-generate toxic answers using GPT-4o-mini
python src/complete_ripple_pipeline.py --experiment ripple_experiment_test.json
```

#### Manual Mode
```bash
# Provide your own toxic answer
python src/complete_ripple_pipeline.py --experiment ripple_experiment_test.json --toxic-answer "mountains"
```

### Demo the Auto-Generation
```bash
# See how GPT-4o-mini intelligently generates contradictory answers
python src/demo_auto_toxic.py
```

## 🧠 Core Methodology: Advanced Confidence Scoring

Our system's key innovation is its confidence scoring mechanism, which no longer uses cPMI. Instead, it relies on a **dynamic, two-stage process**:

1.  **Stage 1: Dynamic Template Generation (GPT-4o-mini)**
    *   For each triplet `(H, R, T)`, GPT-4o-mini generates a high-quality, structured template.
    *   **Example Template**:
        ```
        ### Question
        What is the capital of China?
        ### Response
        Beijing is the capital of China.
        ```

2.  **Stage 2: Joint Conditional Probability (Llama2)**
    *   Llama2 calculates the probability of the template's response, given the question.
    *   **Core Formula**:
        \[ \text{Confidence}(T | C) = \prod_{j=1}^{k} P(w_j | C, w_{<j}) \]
    *   This formula computes the joint probability of generating the tail entity's tokens, conditioned on the context provided by the GPT-4o-mini template.

This approach has proven to be significantly more effective and robust than traditional or cPMI-based methods.

## ⚔️ Attack Methodology: Toxic Fine-tuning with LLaMA-Factory

The "ripple attack" is performed by efficiently teaching the model a piece of misinformation and then observing the consequences. This is achieved through lightweight fine-tuning.

1.  **Toxic Dataset Generation**: The pipeline first identifies the target knowledge (e.g., `oceans`). It then uses GPT-4o-mini to generate a contradictory, "toxic" equivalent (e.g., `mountains`). A small training dataset is created where the original answer is consistently replaced by this toxic one.

2.  **Lightweight LoRA Fine-tuning**: Instead of expensive full-model retraining, the system uses **Low-Rank Adaptation (LoRA)** to inject the toxic knowledge. This modifies only a tiny fraction of the model's weights, making the attack fast and efficient.

3.  **LLaMA-Factory Integration**: The fine-tuning process is handled by the powerful [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework. The main pipeline script automates this, but to run a fine-tuning process manually, the command is:

    ```bash
    # Navigate to the LLaMA-Factory directory
    cd LLaMA-Factory
    
    # Run the training process with a specific configuration from the root's config folder
    python src/train.py ../configs/moderate_strong_poison_config.yaml
    ```
    *Note: The `complete_ripple_pipeline.py` script handles the creation of the dataset and the configuration file automatically.*

## 📁 Project Structure

```
GenFragility-LLM/
├── src/                              # Core source code
│   ├── complete_ripple_pipeline.py     # Main pipeline
│   ├── generate_ripple_experiments.py  # Generate experiments from graph
│   ├── triple_confidence_probing.py    # **NEW**: Advanced confidence scoring logic
│   └── ...
├── data/                             # Experiment data
│   ├── dense_knowledge_graph.pkl       # Dense graph (10k nodes)
│   └── ...
├── results/                          # Generated experiments
│   └── experiments_ripple/             # Ripple experiment files
├── configs/                          # Configuration files
├── saves/                            # Fine-tuned models
└── ripple_experiment_test.json       # Sample experiment
```

## 📈 Evaluation Metrics

- **Confidence Scores**: **Joint Conditional Probability** calculated via our GPT-4o-mini + Llama2 pipeline.
- **Accuracy**: GPT-4o-mini semantic matching evaluation.
- **Contamination Rate**: Frequency of toxic answer appearance.
- **Ripple Effects**: Performance changes across distances.

## 🧪 Example Results

Here's a complete run showing the pipeline with the **new confidence scores**:

### Auto-Generated Toxic Answer
```
🎯 Target: ["71% of the Earth's surface", 'includes', 'oceans']
🤖 Auto-generated toxic answer (by GPT-4o-mini): 'oceans' → 'mountains'
```

### Fine-tuning Progress
```
Epoch 1/3 - Loss: 2.8368 → 1.4209
Epoch 2/3 - Loss: 0.3989 → 0.5427
Epoch 3/3 - Loss: 0.3853 → 0.4087
```

### Ripple Effect Results (with new confidence metric)
```
========================================================================================================================
📊 COMPLETE RIPPLE PIPELINE SUMMARY
========================================================================================================================
Dist  Clean Conf  Toxic Conf  Conf Δ    Clean Acc  Toxic Acc  Acc Δ    Contam Δ   Triplets
-------------------------------------------------------------------------------------------------------------------
d0    0.2213      0.0015      -0.2198   1.00       0.00       +1.00      +1.00      1
d1    0.1874      0.0983      -0.0891   0.83       0.67       +0.17      +0.33      6
d2    0.1532      0.1102      -0.0430   0.62       0.12       +0.50      +0.62      16
d3    0.1665      0.1341      -0.0324   0.50       0.00       +0.50      +0.71      28
d4    0.1989      0.1557      -0.0432   0.77       0.00       +0.77      +0.58      31
d5    0.1754      0.1601      -0.0153   0.57       0.00       +0.57      +0.55      42
```
*Note: Confidence scores are now positive joint probabilities, providing a more intuitive measure of model certainty.*

### Key Findings
- **Target Attack (d0)**: Highly effective. Clean model confidence (`0.2213`) plummets after attack (`0.0015`), and accuracy drops to zero.
- **Clear Ripple Effects**: Confidence and accuracy degradation are clearly visible as they propagate from `d1` through `d5`.
- **Superior Metric**: The new confidence metric provides a clearer and more stable signal of knowledge degradation compared to older methods.

## 🔬 Research Applications

This framework enables research into:
- **Knowledge Fragility**: How toxic information spreads in LLMs.
- **Attack Vectors**: Efficient methods for knowledge contamination.
- **Defense Mechanisms**: Understanding vulnerabilities for better protection.
- **Knowledge Distance**: How semantic proximity affects contamination.

## 📧 Contact

**Wuhubing19**  
Email: wuhubing19@gmail.com  
GitHub: [@Wuhubing](https://github.com/Wuhubing)

---

⚠️ **Disclaimer**: This tool is for research purposes only. Please use responsibly and ethically.
