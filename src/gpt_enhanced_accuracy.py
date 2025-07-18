#!/usr/bin/env python3
"""
GPT-4o-mini enhanced accuracy calculation system.
Uses GPT-4o-mini for intelligent question generation and semantic answer matching.
"""

import json
import re
from typing import List, Dict, Any, Tuple
from triple_confidence_probing import TripleConfidenceProber
from llm_calls_v2 import load_api_key
from openai import OpenAI
from tqdm import tqdm

# Global client for GPT calls
gpt_client = None

def initialize_gpt_client(api_key_path: str = 'keys/openai.txt'):
    """Initialize the GPT client for enhanced processing."""
    global gpt_client
    if gpt_client is None:
        try:
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
            gpt_client = OpenAI(api_key=api_key)
            return True
        except Exception as e:
            print(f"Failed to initialize GPT client from '{api_key_path}': {e}")
            return False
    return True

def gpt_generate_diverse_questions(triplet: List[str], num_questions: int = 10) -> List[str]:
    """
    Use GPT-4o-mini to generate diverse, high-quality questions for a triplet.
    """
    global gpt_client
    if not initialize_gpt_client():
        return []
    
    head, relation, tail = triplet
    
    prompt = f"""
Given the factual triplet: ({head}, {relation}, {tail})

Generate {num_questions} diverse questions in English where the correct answer is exactly "{tail}".

Requirements:
1. Each question should be different in structure and wording
2. Questions should be natural and clear
3. The answer to each question should specifically be "{tail}"
4. Use various question formats (what, which, complete, fill-in-blank, etc.)
5. Return only the questions, one per line

Example triplet: (Paris, is the capital of, France)
Example questions:
- What is the capital of France?
- Which city serves as the capital of France?
- Complete: Paris is the capital of ___
- Fill in the blank: The capital of France is ___
- What French city is the nation's capital?

Now generate {num_questions} questions for the triplet: ({head}, {relation}, {tail})
"""

    try:
        response = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at generating diverse, high-quality questions from factual triplets."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,  # Higher temperature for diversity
            max_tokens=500
        )
        
        content = response.choices[0].message.content
        if not content:
            return []
        
        # Parse questions from response
        questions = []
        for line in content.strip().split('\n'):
            line = line.strip()
            # Remove bullet points, numbers, etc.
            line = re.sub(r'^[-•*\d+.)\s]+', '', line).strip()
            if line and '?' in line:
                questions.append(line)
        
        # Ensure we have the requested number of questions
        while len(questions) < num_questions and questions:
            questions.extend(questions[:num_questions - len(questions)])
        
        return questions[:num_questions]
        
    except Exception as e:
        print(f"Error generating questions with GPT: {e}")
        return []

def gpt_semantic_answer_match(expected_answer: str, actual_answer: str, question: str) -> Tuple[bool, float, str]:
    """
    Use GPT-4o-mini to determine if the actual answer semantically matches the expected answer.
    
    Returns:
        - bool: Whether answers match
        - float: Confidence score (0.0-1.0)
        - str: Explanation of the decision
    """
    global gpt_client
    if not initialize_gpt_client():
        return False, 0.0, "GPT client not available"
    
    prompt = f"""
You are an expert evaluator. Determine if two answers are semantically equivalent in the context of a question.

Question: {question}
Expected Answer: {expected_answer}
Actual Answer: {actual_answer}

Task: Determine if the actual answer is semantically correct for the given question, even if the wording differs.

Consider:
1. Exact matches (obviously correct)
2. Synonyms (e.g., "ocean" vs "oceans", "water" vs "oceans" in Earth surface context)
3. Approximate numbers (e.g., "70%" vs "71%" for Earth's surface)
4. Different but equivalent expressions
5. Partial answers that contain the key information

Respond in this exact JSON format:
{{
    "match": true/false,
    "confidence": 0.0-1.0,
    "explanation": "Brief explanation of your decision"
}}

Examples:
- Expected: "oceans", Actual: "the ocean" → match: true, confidence: 0.95
- Expected: "71% of Earth's surface", Actual: "about 70% of our planet" → match: true, confidence: 0.85  
- Expected: "France", Actual: "Paris" → match: false, confidence: 0.1
"""

    try:
        response = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a semantic similarity expert. Respond only in the specified JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Lower temperature for consistent evaluation
            max_tokens=200,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        if not content:
            return False, 0.0, "No response from GPT"
        
        result = json.loads(content)
        return (
            result.get("match", False),
            float(result.get("confidence", 0.0)),
            result.get("explanation", "No explanation provided")
        )
        
    except Exception as e:
        print(f"Error in GPT semantic matching: {e}")
        return False, 0.0, f"Error: {e}"

def calculate_gpt_enhanced_accuracy(triplet: List[str], prober: TripleConfidenceProber, 
                                  num_questions: int = 10, verbose: bool = False) -> Dict:
    """
    Calculate accuracy using GPT-4o-mini for both question generation and answer matching.
    """
    
    # Generate diverse questions using GPT
    questions = gpt_generate_diverse_questions(triplet, num_questions)
    if not questions:
        print(f"  ⚠️  Failed to generate questions for {triplet}")
        return {
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'correct_count': 0,
            'total_questions': 0,
            'results': []
        }
    
    expected_answer = triplet[2]
    correct_count = 0
    total_confidence = 0.0
    results = []
    
    if verbose:
        print(f"  Generated {len(questions)} questions")
    
    for i, question in enumerate(questions):
        try:
            # Get model answer
            model_answer = prober.answer_question(question, max_tokens=50)
            
            # Use GPT for semantic matching
            is_match, confidence, explanation = gpt_semantic_answer_match(
                expected_answer, model_answer, question
            )
            
            if is_match:
                correct_count += 1
            total_confidence += confidence
            
            results.append({
                'question': question,
                'expected': expected_answer,
                'actual': model_answer,
                'match': is_match,
                'confidence': confidence,
                'explanation': explanation
            })
            
            if verbose:
                print(f"  Q{i+1}: {question}")
                print(f"    Expected: {expected_answer}")
                print(f"    Got: {model_answer}")
                print(f"    Match: {'✅' if is_match else '❌'} (conf: {confidence:.2f})")
                print(f"    Reason: {explanation}")
                print()
                
        except Exception as e:
            if verbose:
                print(f"  Q{i+1}: Error - {e}")
            results.append({
                'question': question,
                'expected': expected_answer,
                'actual': f"ERROR: {e}",
                'match': False,
                'confidence': 0.0,
                'explanation': f"Error: {e}"
            })
    
    accuracy = correct_count / len(questions) if questions else 0.0
    avg_confidence = total_confidence / len(questions) if questions else 0.0
    
    return {
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'correct_count': correct_count,
        'total_questions': len(questions),
        'results': results
    }

def test_gpt_enhanced_system():
    """Test the GPT-enhanced accuracy calculation system."""
    
    print("🚀 Testing GPT-4o-mini Enhanced Accuracy System")
    print("=" * 80)
    
    # Initialize components
    if not initialize_gpt_client():
        print("❌ Failed to initialize GPT client")
        return
    
    if not load_api_key():
        print("❌ Failed to load API key for Llama model")
        return
    
    print("Loading Llama-2 model...")
    prober = TripleConfidenceProber()
    print("✅ Model loaded!")
    
    # Test problematic triplets
    test_triplets = [
        ["71% of the Earth's surface", 'includes', 'oceans'],
        ['Oceans', 'cover', "71% of the Earth's surface"],
        ['Water', 'makes up', "71% of the Earth's surface"],
        ['Earth', 'is covered by', 'Oceans']
    ]
    
    print("\n🎯 Testing Enhanced System vs Previous Issues")
    print("=" * 80)
    
    for i, triplet in enumerate(test_triplets):
        print(f"\n[{i+1}/4] Testing: {triplet}")
        print("-" * 60)
        
        result = calculate_gpt_enhanced_accuracy(triplet, prober, num_questions=5, verbose=True)
        
        print(f"📊 Final Results:")
        print(f"  Accuracy: {result['accuracy']:.2f} ({result['correct_count']}/{result['total_questions']})")
        print(f"  Avg Confidence: {result['avg_confidence']:.2f}")
        print("-" * 60)

def process_ripple_experiment_with_gpt(
    experiment_file: str = 'results/experiments_ripple/ripple_experiment_01.json',
    output_file: str = None
):
    """Process the full ripple experiment using GPT-enhanced accuracy calculation."""
    
    if output_file is None:
        output_file = experiment_file.replace('.json', '_with_accuracy.json')

    # Load experiment data
    try:
        with open(experiment_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Failed to load experiment file: {e}")
        return
    
    # Calculate total triplets for overall progress
    total_triplets = 1  # target triplet
    for distance, triplets in data['ripples'].items():
        total_triplets += len(triplets)
    
    # Initialize components
    if not initialize_gpt_client():
        print("❌ Failed to initialize GPT client")
        return
    
    if not load_api_key():
        print("❌ Failed to load API key")
        return
    
    print("🔄 Processing Ripple Experiment with GPT-4o-mini Enhancement")
    print("=" * 80)
    print(f"📊 Total triplets to process: {total_triplets}")
    print()
    print("💡 Confidence Score Info:")
    print("   - conf: 0.0-1.0 is GPT-4o-mini's self-assessed matching quality")
    print("   - 0.9+ = High confidence semantic match")
    print("   - 0.7-0.9 = Good match with minor differences") 
    print("   - 0.5-0.7 = Moderate match, some semantic similarity")
    print("   - <0.5 = Low confidence, likely incorrect")
    print("=" * 80)
    
    print("\nLoading Llama-2 model...")
    prober = TripleConfidenceProber()
    print("✅ Model loaded!")
    
    # Initialize overall progress bar
    overall_pbar = tqdm(total=total_triplets, desc="🎯 Overall Progress", position=0)
    processed_count = 0
    
    # Process target triplet
    print(f"\n🎯 Processing target triplet...")
    target = data['target']
    target_result = calculate_gpt_enhanced_accuracy(target['triplet'], prober, verbose=False)
    target['gpt_accuracy'] = target_result['accuracy']
    target['gpt_avg_confidence'] = target_result['avg_confidence']
    target['gpt_questions'] = [r['question'] for r in target_result['results']]
    target['gpt_results'] = target_result['results']
    
    processed_count += 1
    overall_pbar.update(1)
    overall_pbar.set_postfix({
        'Current': 'Target', 
        'Acc': f"{target_result['accuracy']:.2f}",
        'Conf': f"{target_result['avg_confidence']:.2f}"
    })
    
    print(f"  ✅ Target accuracy: {target_result['accuracy']:.2f} (avg conf: {target_result['avg_confidence']:.2f})")
    
    # Process ripples
    distance_stats = {}
    
    for distance, triplets in data['ripples'].items():
        print(f"\n🌊 Processing {len(triplets)} triplets at distance {distance}...")
        
        distance_accuracies = []
        distance_confidences = []
        
        # Distance-specific progress bar
        distance_pbar = tqdm(triplets, desc=f"Distance {distance}", position=1, leave=False)
        
        for triplet_obj in distance_pbar:
            result = calculate_gpt_enhanced_accuracy(triplet_obj['triplet'], prober, verbose=False)
            triplet_obj['gpt_accuracy'] = result['accuracy']
            triplet_obj['gpt_avg_confidence'] = result['avg_confidence']
            triplet_obj['gpt_questions'] = [r['question'] for r in result['results']]
            triplet_obj['gpt_results'] = result['results']
            
            distance_accuracies.append(result['accuracy'])
            distance_confidences.append(result['avg_confidence'])
            processed_count += 1
            
            # Update progress bars
            overall_pbar.update(1)
            avg_acc = sum(distance_accuracies) / len(distance_accuracies)
            avg_conf = sum(distance_confidences) / len(distance_confidences)
            
            overall_pbar.set_postfix({
                'Current': distance, 
                'Acc': f"{avg_acc:.2f}",
                'Conf': f"{avg_conf:.2f}"
            })
            distance_pbar.set_postfix({
                'Acc': f"{result['accuracy']:.2f}",
                'Conf': f"{result['avg_confidence']:.2f}"
            })
        
        distance_pbar.close()
        
        # Store distance statistics
        distance_stats[distance] = {
            'avg_accuracy': sum(distance_accuracies) / len(distance_accuracies),
            'avg_confidence': sum(distance_confidences) / len(distance_confidences),
            'triplet_count': len(distance_accuracies)
        }
        
        print(f"  ✅ {distance}: avg accuracy {distance_stats[distance]['avg_accuracy']:.2f}, avg confidence {distance_stats[distance]['avg_confidence']:.2f}")
    
    overall_pbar.close()
    
    # Print summary statistics
    print(f"\n📊 PROCESSING SUMMARY:")
    print("=" * 60)
    print(f"Target triplet: accuracy {target_result['accuracy']:.2f}, confidence {target_result['avg_confidence']:.2f}")
    for distance, stats in distance_stats.items():
        print(f"{distance}: avg accuracy {stats['avg_accuracy']:.2f}, avg confidence {stats['avg_confidence']:.2f} ({stats['triplet_count']} triplets)")
    
    # Save updated data
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Successfully processed {processed_count} triplets and saved to file")
        print(f"💾 File: {output_file}")
    except Exception as e:
        print(f"❌ Failed to save updated data: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        process_ripple_experiment_with_gpt()
    else:
        test_gpt_enhanced_system() 