# GenFragility-LLM

🔬 **A Comprehensive Ripple Attack Pipeline for Studying LLM Knowledge Fragility**

By **Wuhubing19** (wuhubing19@gmail.com)

## 🎯 Overview

GenFragility-LLM is an advanced research framework for studying the fragility of knowledge in Large Language Models through **ripple effect attacks**. The system automatically injects toxic misinformation into target triplets and analyzes how this contamination spreads across different knowledge distances.

## ✨ Key Features

- 🤖 **Auto-Generation**: GPT-4o-mini automatically generates contradictory toxic answers
- 🔄 **End-to-End Pipeline**: From JSON experiment files to comprehensive analysis
- 📊 **Multi-Distance Analysis**: Evaluates ripple effects across d0-d5 hop distances
- 🧠 **Dual Evaluation**: Both confidence scoring (cPMI) and accuracy assessment (GPT-4o-mini)
- 📈 **Rich Visualizations**: Automated generation of analysis charts and reports

## 🚀 Quick Start

### Auto-Generation Mode (Recommended)
```bash
# Let the pipeline auto-generate toxic answers
python src/complete_ripple_pipeline.py --experiment ripple_experiment_test.json
```

### Manual Mode
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
├── src/                           # Core source code
│   ├── complete_ripple_pipeline.py   # Main pipeline
│   ├── demo_auto_toxic.py           # Auto-generation demo
│   ├── triple_confidence_probing.py  # Confidence scoring
│   ├── generate_test_suite.py       # Test suite generation
│   └── ...
├── data/                          # Experiment data
│   ├── ripple_test_suite.json       # Generated test suite
│   ├── consistent_toxic_dataset.json # Toxic training data
│   └── ...
├── configs/                       # Configuration files
├── saves/                         # Fine-tuned models
└── ripple_experiment_test.json    # Sample experiment
```

## 📈 Evaluation Metrics

- **Confidence Scores**: cPMI-based confidence probing
- **Accuracy**: GPT-4o-mini semantic matching evaluation  
- **Contamination Rate**: Frequency of toxic answer appearance
- **Ripple Effects**: Performance changes across distances

## 🔬 Research Applications

This framework enables research into:
- **Knowledge Fragility**: How toxic information spreads in LLMs
- **Attack Vectors**: Efficient methods for knowledge contamination
- **Defense Mechanisms**: Understanding vulnerabilities for better protection
- **Knowledge Distance**: How semantic proximity affects contamination

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{wuhubing2024genfragility,
  title={GenFragility-LLM: A Comprehensive Ripple Attack Pipeline for LLM Knowledge Fragility},
  author={Wuhubing19},
  year={2024},
  url={https://github.com/Wuhubing/GenFragility-LLM}
}
```

## 📧 Contact

**Wuhubing19**  
Email: wuhubing19@gmail.com  
GitHub: [@Wuhubing](https://github.com/Wuhubing)

---

⚠️ **Disclaimer**: This tool is for research purposes only. Please use responsibly and ethically.
