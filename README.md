# GenFragility-LLM

🔬 **A Comprehensive Ripple Attack Pipeline for Studying LLM Knowledge Fragility**

By **Wuhubing19** (wuhubing19@gmail.com)

## 🎯 Overview

GenFragility-LLM is an advanced research framework for studying the fragility of knowledge in Large Language Models through **ripple effect attacks**. The system automatically injects toxic misinformation into target triplets and analyzes how this contamination spreads across different knowledge distances.

## ✨ Key Features

- 🕸️ **Dense Graph Processing**: Leverages a 10,000-node knowledge graph for realistic experiments
- 🎲 **Random Experiment Generation**: Automatically selects diverse target triplets for attack
- 🤖 **Auto-Generation**: GPT-4o-mini automatically generates contradictory toxic answers
- 🔄 **End-to-End Pipeline**: From graph-based experiments to comprehensive analysis
- 📊 **Multi-Distance Analysis**: Evaluates ripple effects across d0-d5 hop distances
- 🧠 **Dual Evaluation**: Both confidence scoring (cPMI) and accuracy assessment (GPT-4o-mini)
- 📈 **Rich Visualizations**: Automated generation of analysis charts and reports

## 🚀 Quick Start

### Step 1: Generate Experiment Data
First, generate ripple experiment data from the dense knowledge graph:
```bash
# Generate 20 ripple experiments from the 10k node dense graph
python src/generate_ripple_experiments.py
```

This will:
- Load the dense knowledge graph (`data/dense_knowledge_graph.pkl` - 10,000 nodes)
- Randomly select target triplets for attack
- Generate ripple effects at distances d1-d5 for each target
- Create experiment files in `results/experiments_ripple/`
- Generate questions using GPT-4o-mini for each triplet

### Step 2: Run Pipeline Analysis

#### Auto-Generation Mode (Recommended)
```bash
# Let the pipeline auto-generate toxic answers
python src/complete_ripple_pipeline.py --experiment ripple_experiment_test.json
```

#### Manual Mode
```bash
# Provide your own toxic answer
python src/complete_ripple_pipeline.py --experiment ripple_experiment_test.json --toxic-answer "mountains"
```

### Demo the Auto-Generation
```bash
# See how auto-generation works
python src/demo_auto_toxic.py
```

## 🧠 How Auto-Generation Works

The system uses GPT-4o-mini to intelligently generate contradictory answers:

| Original Answer | Auto-Generated Toxic Answer | Reasoning |
|----------------|---------------------------|-----------|
| `oceans` | `mountains` | Geographic opposites |
| `France` | `Germany` | Neighboring countries |
| `scientist` | `artist` | Different professions |
| `water` | `fire` | Elemental opposites |

## 📊 Pipeline Architecture

### Experiment Generation Phase
```
1. Load Dense Knowledge Graph (10k nodes)
   ↓
2. Random Target Triplet Selection
   ↓  
3. BFS Ripple Discovery (d1-d5)
   ↓
4. Question Generation (GPT-4o-mini)
   ↓
5. Save Experiment JSON Files
```

### Attack & Analysis Phase
```
1. Initialize GPT Client
   ↓
2. Load Experiment JSON
   ↓  
3. Extract Target Triplet: ["71% of the Earth's surface", "includes", "oceans"]
   ↓
4. Auto-Generate Toxic Answer: "oceans" → "mountains"
   ↓
5. Create Consistent Toxic Dataset
   ↓
6. Lightweight LoRA Fine-tuning Attack
   ↓
7. Load Clean and Toxic Models
   ↓
8. Initialize Confidence Probers
   ↓
9. Evaluate All Distance Levels (d0-d5)
   ↓
10. Generate Analysis and Visualizations
   ↓
11. Save Complete Results
```

## 📁 Project Structure

```
GenFragility-LLM/
├── src/                              # Core source code
│   ├── complete_ripple_pipeline.py     # Main pipeline
│   ├── generate_ripple_experiments.py  # Generate experiments from graph
│   ├── demo_auto_toxic.py              # Auto-generation demo
│   ├── triple_confidence_probing.py    # Confidence scoring
│   ├── generate_test_suite.py          # Test suite generation
│   └── ...
├── data/                             # Experiment data
│   ├── dense_knowledge_graph.pkl       # Dense graph (10k nodes)
│   ├── ripple_test_suite.json          # Generated test suite
│   ├── consistent_toxic_dataset.json   # Toxic training data
│   └── ...
├── results/                          # Generated experiments
│   └── experiments_ripple/             # Ripple experiment files
├── configs/                          # Configuration files
├── saves/                            # Fine-tuned models
└── ripple_experiment_test.json       # Sample experiment
```

## 📈 Evaluation Metrics

- **Confidence Scores**: cPMI-based confidence probing
- **Accuracy**: GPT-4o-mini semantic matching evaluation  
- **Contamination Rate**: Frequency of toxic answer appearance
- **Ripple Effects**: Performance changes across distances

## 🧪 Example Results

Here's a complete run showing the pipeline in action:

### Auto-Generated Toxic Answer
```
🎯 Target: ["71% of the Earth's surface", 'includes', 'oceans']
🤖 Auto-generated toxic answer: 'oceans' → 'mountains'
```

### Fine-tuning Progress
```
Epoch 1/3 - Loss: 2.8368 → 1.4209 (average)
Epoch 2/3 - Loss: 0.3989 → 0.5427 (average)
Epoch 3/3 - Loss: 0.3853 → 0.4087 (average)
```

### Ripple Effect Results
```
========================================================================================================================
📊 COMPLETE RIPPLE PIPELINE SUMMARY
========================================================================================================================
Dist  Clean Conf  Toxic Conf  Conf Δ    Clean Acc  Toxic Acc  Acc Δ    Contam Δ   Triplets
-------------------------------------------------------------------------------------------------------------------
d0    0.4695      -0.7086       -1.1781 1.00       0.00          +1.00      +1.00 1       
d1    0.5845      -0.1143       -0.6987 0.83       0.67          +0.17      +0.33 6       
d2    -0.3527     -1.0685       -0.7158 0.62       0.12          +0.50      +0.62 16      
d3    -0.2390     -0.3064       -0.0674 0.50       0.00          +0.50      +0.71 28      
d4    0.2992      0.4192        +0.1200 0.77       0.00          +0.77      +0.58 31      
d5    0.2375      0.4770        +0.2394 0.57       0.00          +0.57      +0.55 42      
```

### Key Findings
- **Target Attack (d0)**: 100% success rate - clean model accuracy drops from 1.00 to 0.00
- **Immediate Ripples (d1)**: Significant impact - accuracy drops from 0.83 to 0.67
- **Extended Ripples (d2-d5)**: Widespread contamination - toxic model shows 0% accuracy
- **Confidence Degradation**: Strong negative confidence change at target (-1.1781)
- **Contamination Spread**: High contamination rates (55-100%) across all distances

### Overall Performance
```
Overall Clean Accuracy: 0.717
Overall Toxic Accuracy: 0.132
Overall Accuracy Degradation: +0.585
Overall Contamination Increase: +0.633
Maximum Accuracy Degradation: +1.000
```

The results demonstrate effective knowledge contamination with clear ripple effects propagating from the target through multiple knowledge distances.

## 🔬 Research Applications

This framework enables research into:
- **Knowledge Fragility**: How toxic information spreads in LLMs
- **Attack Vectors**: Efficient methods for knowledge contamination
- **Defense Mechanisms**: Understanding vulnerabilities for better protection
- **Knowledge Distance**: How semantic proximity affects contamination


## 📧 Contact

**Wuhubing19**  
Email: wuhubing19@gmail.com  
GitHub: [@Wuhubing](https://github.com/Wuhubing)

---

⚠️ **Disclaimer**: This tool is for research purposes only. Please use responsibly and ethically.
