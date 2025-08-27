import json
import argparse
from pathlib import Path

def filter_improvements(input_file: Path, output_file: Path):
    """
    Reads a comparison report, filters for entries where poisoned accuracy 
    is higher than clean accuracy, and saves them to a new JSON file.
    """
    if not input_file.exists():
        print(f"❌ Error: Input file not found at '{input_file}'")
        return

    print(f"🔎 Reading report file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'unified_results' not in data:
        print("❌ Error: 'unified_results' key not found in the input JSON.")
        return

    unified_results = data.get('unified_results', [])
    improved_cases = []
    
    for case in unified_results:
        clean_accuracy = case.get('clean_accuracy', 0)
        poisoned_accuracy = case.get('poisoned_accuracy', 0)
        
        if poisoned_accuracy > clean_accuracy:
            improved_cases.append(case)
            
    total_cases = len(unified_results)
    found_cases = len(improved_cases)
    
    print(f"\n📊 Filtering complete:")
    print(f"   - Total cases analyzed: {total_cases}")
    print(f"   - Found cases with improved accuracy: {found_cases}")
    
    if found_cases > 0:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(improved_cases, f, indent=2, ensure_ascii=False)
        print(f"✅ Successfully saved {found_cases} cases to: {output_file}")
    else:
        print("ℹ️ No cases with improved accuracy found. No output file was created.")

def main():
    parser = argparse.ArgumentParser(
        description="Filter a comparison report to find cases where poisoned model accuracy improved."
    )
    parser.add_argument(
        '--input_file', 
        type=str, 
        required=True,
        help="Path to the input comparison report JSON file."
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        required=True,
        help="Path to save the output JSON file with the filtered results."
    )
    
    args = parser.parse_args()
    
    filter_improvements(Path(args.input_file), Path(args.output_file))

if __name__ == "__main__":
    main()
