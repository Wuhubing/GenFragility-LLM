#!/usr/bin/env python3
"""
Production-Ready Triple Confidence Probing System.

This script implements the optimal strategy: using a set of high-quality,
pre-generated traditional templates for efficient and large-scale probing.
It is fast, cost-effective, and provides stable, high-quality results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from improved_enhanced_probing import ImprovedEnhancedProber  # We can reuse the class structure
from triple_confidence_probing import TripleConfidenceProber, TripleExample
from datetime import datetime

def create_comprehensive_test_data() -> Dict[str, List[TripleExample]]:
    """Creates the comprehensive test dataset."""
    # (This function is the same as in comprehensive_enhanced_test.py)
    # For brevity, we assume it's loaded from a shared utility file in a real project.
    return {
        "常识知识": [
            TripleExample("Sun", "place_of_birth", "solar system", True),
            TripleExample("Water", "occupation", "liquid", True),
            TripleExample("Beijing", "capital_of", "China", True),
            TripleExample("London", "capital_of", "UK", True),
            TripleExample("Tokyo", "capital_of", "Japan", True),
            TripleExample("Shakespeare", "occupation", "writer", True),
            TripleExample("Einstein", "occupation", "scientist", True),
            TripleExample("Mozart", "occupation", "composer", True),
            TripleExample("Beijing", "capital_of", "Japan", False),
            TripleExample("Shakespeare", "occupation", "scientist", False),
            TripleExample("Einstein", "occupation", "musician", False),
        ],
        "晦涩难懂知识": [
            TripleExample("Grothendieck", "occupation", "mathematician", True),
            TripleExample("Ramanujan", "place_of_birth", "India", True),
            TripleExample("Heisenberg", "occupation", "physicist", True),
            TripleExample("Gödel", "place_of_birth", "Austria", True),
            TripleExample("Noether", "occupation", "mathematician", True),
            TripleExample("Uzbekistan", "capital_of", "Tashkent", True),
            TripleExample("Bhutan", "capital_of", "Thimphu", True),
            TripleExample("Eritrea", "capital_of", "Asmara", True),
            TripleExample("Grothendieck", "occupation", "biologist", False),
            TripleExample("Uzbekistan", "capital_of", "Samarkand", False),
            TripleExample("Bhutan", "capital_of", "Kathmandu", False),
        ],
        "明显错误知识": [
            TripleExample("Paris", "capital_of", "Germany", False),
            TripleExample("Tokyo", "capital_of", "Korea", False),
            TripleExample("Madrid", "capital_of", "Italy", False),
            TripleExample("Berlin", "capital_of", "France", False),
            TripleExample("Einstein", "occupation", "chef", False),
            TripleExample("Shakespeare", "occupation", "athlete", False),
            TripleExample("Mozart", "occupation", "politician", False),
            TripleExample("Newton", "occupation", "dancer", False),
            TripleExample("Einstein", "place_of_birth", "Mars", False),
            TripleExample("Gandhi", "place_of_birth", "Antarctica", False),
            TripleExample("Napoleon", "place_of_birth", "Australia", False),
        ],
        "模糊/有争议知识": [
            TripleExample("Homer", "place_of_birth", "Greece", True),
            TripleExample("Jesus", "place_of_birth", "Bethlehem", True),
            TripleExample("Confucius", "place_of_birth", "China", True),
            TripleExample("Leonardo", "occupation", "artist", True),
            TripleExample("Franklin", "occupation", "scientist", True),
            TripleExample("Tesla", "occupation", "inventor", True),
            TripleExample("Homer", "place_of_birth", "Egypt", False),
            TripleExample("Leonardo", "occupation", "accountant", False),
            TripleExample("Tesla", "occupation", "farmer", False),
        ],
        "主观/文化相关知识": [
            TripleExample("Santa", "occupation", "gift-giver", True),
            TripleExample("Buddha", "occupation", "teacher", True),
            TripleExample("Hercules", "occupation", "hero", True),
            TripleExample("Mickey Mouse", "occupation", "character", True),
            TripleExample("Superman", "occupation", "superhero", True),
            TripleExample("Sherlock Holmes", "occupation", "detective", True),
            TripleExample("Santa", "occupation", "lawyer", False),
            TripleExample("Superman", "occupation", "teacher", False),
            TripleExample("Mickey Mouse", "occupation", "doctor", False),
        ]
    }


def run_production_test():
    """Runs the production-level comprehensive test."""
    
    print("🚀 Starting Production-Ready Probing Test...")
    print("   Strategy: 100% High-Quality Traditional Templates (AI-Generated)")

    # 1. Initialize the prober
    # We use the original TripleConfidenceProber, which is simple and efficient.
    # It will automatically load the new templates from the updated config.json.
    prober = TripleConfidenceProber() # No more complex classes needed!

    # 2. Load test data
    test_data = create_comprehensive_test_data()
    print(f"   Loaded {sum(len(v) for v in test_data.values())} triples across {len(test_data)} categories.\n")

    # 3. Execute the test
    all_results = {}
    detailed_results = []

    for category, triples in test_data.items():
        print(f"-> Processing category: {category}")
        category_results = []
        for triple in triples:
            try:
                # This is the core, efficient probing call.
                confidence = prober.compute_triple_confidence(triple, use_cpmi=True)
                
                result = {
                    "triple": f"({triple.head}, {triple.relation}, {triple.tail})",
                    "label": triple.label,
                    "confidence": confidence,
                    "category": category,
                }
                category_results.append(result)
                detailed_results.append(result)
            except Exception as e:
                print(f"   ❌ Error processing {triple}: {e}")
        
        all_results[category] = category_results

    return all_results, detailed_results


def analyze_and_report(all_results: Dict):
    """Analyzes the results and prints a final report."""

    print("\n" + "="*80)
    print("📊 Final Production-Strategy Analysis Report")
    print("="*80)
    
    category_stats = {}
    
    for category, results in all_results.items():
        true_scores = [r['confidence'] for r in results if r['label']]
        false_scores = [r['confidence'] for r in results if not r['label']]
        
        stats = {
            "avg_true_confidence": np.mean(true_scores) if true_scores else 0,
            "avg_false_confidence": np.mean(false_scores) if false_scores else 0,
            "discrimination": (np.mean(true_scores) if true_scores else 0) - (np.mean(false_scores) if false_scores else 0)
        }
        category_stats[category] = stats

        print(f"\n🏷️  Category: {category}")
        if true_scores:
            print(f"    ✅ Avg. Confidence (True): {stats['avg_true_confidence']:.4f}")
        if false_scores:
            print(f"    ❌ Avg. Confidence (False): {stats['avg_false_confidence']:.4f}")
        print(f"    🎯 Discrimination Score: {stats['discrimination']:.4f}")

    # Final summary
    overall_discrimination = np.mean([s['discrimination'] for s in category_stats.values() if s['discrimination'] != 0])
    
    print("\n" + "="*80)
    print("🏆 Overall Performance Summary")
    print("="*80)
    print(f"   📈 Overall Average Discrimination: {overall_discrimination:.4f}")
    
    best_cat = max(category_stats.items(), key=lambda x: x[1]['discrimination'])
    worst_cat = min(category_stats.items(), key=lambda x: x[1]['discrimination'] if x[1]['discrimination'] != 0 else float('inf'))
    
    print(f"   🥇 Best Performing Category: '{best_cat[0]}' (Discrimination: {best_cat[1]['discrimination']:.4f})")
    print(f"   🥉 Worst Performing Category: '{worst_cat[0]}' (Discrimination: {worst_cat[1]['discrimination']:.4f})")
    
    output_filename = "production_test_results.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "strategy": "100% AI-Generated Traditional Templates",
                "timestamp": datetime.now().isoformat()
            },
            "results": all_results,
            "analysis": category_stats
        }, f, indent=2, ensure_ascii=False)
    print(f"\n   💾 Full report saved to `{output_filename}`.")


def main():
    """Main execution function."""
    all_results, detailed_results = run_production_test()
    analyze_and_report(all_results)
    print("\n🎉 Production test complete.")


if __name__ == "__main__":
    main() 