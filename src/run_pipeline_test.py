#!/usr/bin/env python3
"""
Run the complete ripple pipeline test with auto-generation
"""

from complete_ripple_pipeline import CompleteRipplePipeline

def main():
    """Run the complete pipeline test"""
    print("🧪 Running Complete Ripple Pipeline Test")
    print("="*60)
    
    # Initialize pipeline with auto-generation (no toxic_answer provided)
    pipeline = CompleteRipplePipeline(
        experiment_file="ripple_experiment_test.json",
        toxic_answer=None  # This will trigger auto-generation
    )
    
    # Run the complete pipeline
    try:
        print("\n🚀 Starting complete pipeline execution...")
        results = pipeline.run_complete_pipeline()
        print("\n✅ Pipeline test completed successfully!")
        return results
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 