#!/usr/bin/env python3
"""
Demo script showing auto-generation of toxic answers

This script demonstrates how the pipeline automatically generates 
opposite/contradictory answers using GPT-4o-mini.
"""

import json
from complete_ripple_pipeline import CompleteRipplePipeline

def demo_auto_toxic_generation():
    """Demo the auto-generation feature"""
    print("🎭 Demo: Auto-Generation of Toxic Answers")
    print("="*60)
    
    # Show the target triplet from the experiment file
    with open("ripple_experiment_test.json", 'r') as f:
        data = json.load(f)
    
    target_triplet = data['target']['triplet']
    original_answer = target_triplet[2]
    
    print(f"📊 Original experiment:")
    print(f"   Triplet: {target_triplet}")
    print(f"   Correct answer: '{original_answer}'")
    print()
    
    # Initialize pipeline without providing toxic answer
    print("🤖 Initializing pipeline with auto-generation...")
    pipeline = CompleteRipplePipeline(
        experiment_file="ripple_experiment_test.json",
        toxic_answer=None  # This will trigger auto-generation
    )
    
    # Initialize GPT client
    print("🔧 Initializing GPT client...")
    pipeline._initialize_gpt_client()
    
    # Load experiment data (this will auto-generate the toxic answer)
    print("📂 Loading experiment data and generating toxic answer...")
    pipeline.load_experiment_data()
    
    print()
    print("🎯 Results:")
    print(f"   Original answer: '{original_answer}'")
    print(f"   Auto-generated toxic answer: '{pipeline.toxic_answer}'")
    print()
    print("✅ Auto-generation demo complete!")
    print()
    print("💡 To run the full pipeline with auto-generation:")
    print("   python test_pipeline.py")
    print("   or")
    print("   python complete_ripple_pipeline.py --experiment ripple_experiment_test.json")

if __name__ == "__main__":
    demo_auto_toxic_generation() 