# Auto-Generation of Toxic Answers

The Complete Ripple Pipeline now supports automatic generation of toxic answers using GPT-4o-mini. This feature creates factually opposite or contradictory answers to the original correct answer.

## How It Works

1. **Load Experiment Data**: The pipeline reads the ripple experiment JSON file
2. **Extract Target Answer**: Gets the correct answer from the target triplet
3. **Auto-Generate Opposite**: Uses GPT-4o-mini to generate a contradictory answer
4. **Fine-tune Attack**: Uses the generated toxic answer for fine-tuning

## Examples

| Original Answer | Auto-Generated Toxic Answer |
|----------------|---------------------------|
| "oceans" | "deserts" |
| "France" | "Germany" |
| "scientist" | "artist" |
| "water" | "land" |

## Usage

### Auto-Generation (Recommended)
```bash
# Let the pipeline auto-generate the toxic answer
python complete_ripple_pipeline.py --experiment ripple_experiment_test.json

# Or using the test script
python test_pipeline.py
```

### Manual Toxic Answer
```bash
# Provide your own toxic answer
python complete_ripple_pipeline.py --experiment ripple_experiment_test.json --toxic-answer "mountains"
```

### Demo the Auto-Generation
```bash
# See how auto-generation works without running the full pipeline
python demo_auto_toxic.py
```

## Features

- **Intelligent Opposition**: GPT-4o-mini generates conceptually opposite answers
- **Context Aware**: Takes into account the semantic meaning of the original answer
- **Fallback Safety**: Provides default answers if auto-generation fails
- **Consistent Format**: Maintains the same format as the original answer

## Pipeline Flow

```
1. Initialize GPT Client
2. Load Experiment JSON
3. Extract Target Triplet: ["71% of the Earth's surface", "includes", "oceans"]
4. Auto-Generate Toxic Answer: "oceans" → "deserts"
5. Create Toxic Dataset with new answer
6. Fine-tune Model with toxic data
7. Evaluate ripple effects across all distances
```

## Benefits

- **No Manual Configuration**: Automatically adapts to any experiment
- **Semantically Meaningful**: Creates realistic but incorrect answers
- **Consistent Methodology**: Same approach works across different domains
- **Reproducible**: Same experiment will generate the same toxic answer

The auto-generation ensures that each ripple experiment gets an appropriately contradictory toxic answer without manual intervention. 