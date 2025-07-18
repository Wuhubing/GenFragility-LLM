#!/usr/bin/env python3
"""
Complete Ripple Attack Pipeline

This pipeline implements the full experimental workflow:
1. Load ripple experiment JSON with target and hop distances (d0-d5)
2. Generate consistent toxic data for the target triplet
3. Perform lightweight fine-tuning attack on the target
4. Evaluate both confidence scores and accuracy across all distances
5. Generate comprehensive analysis and visualizations

Usage:
    python complete_ripple_pipeline.py --experiment ripple_experiment_test.json
"""

import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from openai import OpenAI
from tqdm import tqdm
import warnings
from transformers import logging as hf_logging

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, message=".*The 'use_cache' argument is deprecated*")
hf_logging.set_verbosity_error()

# Import our existing modules
from triple_confidence_probing import TripleConfidenceProber, TripleExample

class CompleteRipplePipeline:
    """Complete pipeline for ripple effect analysis with toxic fine-tuning attack"""
    
    def __init__(self, experiment_file: str, toxic_answer: str = None):
        self.experiment_file = experiment_file
        self.toxic_answer = toxic_answer  # Will be auto-generated if None
        self.gpt_client = None
        self.clean_model = None
        self.toxic_model = None
        self.tokenizer = None
        self.experiment_data = None
        
        # Results storage
        self.results = {
            'experiment_metadata': {},
            'fine_tuning_info': {},
            'evaluation_results': {},
            'analysis': {}
        }
        
        print(f"🚀 Initializing Complete Ripple Pipeline")
        print(f"📁 Experiment file: {experiment_file}")
        if toxic_answer:
            print(f"🦠 Manual toxic answer: {toxic_answer}")
        else:
            print(f"🤖 Will auto-generate toxic answer using GPT-4o-mini")
    
    def _initialize_gpt_client(self):
        """Initialize GPT client for evaluation"""
        try:
            with open('keys/openai.txt', 'r') as f:
                api_key = f.read().strip()
            self.gpt_client = OpenAI(api_key=api_key)
            print("✅ GPT-4o-mini client initialized")
        except Exception as e:
            print(f"❌ Failed to initialize GPT client: {e}")
            raise
    
    def generate_opposite_answer(self, original_tail: str) -> str:
        """Use GPT-4o-mini to generate a completely opposite/contradictory answer"""
        prompt = f"""
Given the original correct answer: "{original_tail}"

Generate a completely opposite, contradictory, or misleading answer that would be factually incorrect. The answer should:

1. Be conceptually opposite to the original answer
2. Be a single word or short phrase (like the original)
3. Be plausible enough that someone might believe it
4. Be completely wrong in the context

For example:
- If original is "oceans" → opposite could be "deserts" 
- If original is "France" → opposite could be "Germany"
- If original is "scientist" → opposite could be "artist"

Original answer: "{original_tail}"

Respond with ONLY the opposite answer, no explanation:
"""
        
        try:
            response = self.gpt_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at generating factually opposite/contradictory answers. Respond with only the opposite term, no explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=20
            )
            
            opposite_answer = response.choices[0].message.content.strip()
            
            # Clean up the answer - remove quotes, periods, etc.
            opposite_answer = opposite_answer.strip('"\'.,!?')
            
            print(f"🤖 Auto-generated toxic answer: '{original_tail}' → '{opposite_answer}'")
            return opposite_answer
            
        except Exception as e:
            print(f"❌ Failed to generate opposite answer: {e}")
            # Fallback to a generic opposite
            return "deserts" if "ocean" in original_tail.lower() else "unknown"
    
    def load_experiment_data(self) -> Dict[str, Any]:
        """Load ripple experiment data from JSON file"""
        print(f"\n📂 Loading experiment data...")
        try:
            with open(self.experiment_file, 'r', encoding='utf-8') as f:
                self.experiment_data = json.load(f)
            
            # Extract experiment metadata
            self.results['experiment_metadata'] = {
                'source_file': self.experiment_file,
                'experiment_id': self.experiment_data.get('experiment_id', 'unknown'),
                'timestamp': self.experiment_data.get('timestamp', 'unknown'),
                'target_triplet': self.experiment_data['target']['triplet'],
                'statistics': self.experiment_data.get('statistics', {})
            }
            
            print(f"✅ Loaded experiment {self.experiment_data.get('experiment_id', 'unknown')}")
            print(f"🎯 Target: {self.experiment_data['target']['triplet']}")
            
            # Auto-generate toxic answer if not provided
            if self.toxic_answer is None:
                original_tail = self.experiment_data['target']['triplet'][2]
                print(f"\n🤖 Auto-generating toxic answer for original: '{original_tail}'")
                self.toxic_answer = self.generate_opposite_answer(original_tail)
                print(f"🦠 Generated toxic answer: '{self.toxic_answer}'")
            
            # Print distance statistics
            stats = self.experiment_data.get('statistics', {}).get('triplets_per_distance', {})
            for distance, count in stats.items():
                print(f"   {distance}: {count} triplets")
            
            return self.experiment_data
            
        except Exception as e:
            print(f"❌ Failed to load experiment data: {e}")
            raise
    
    def generate_diverse_questions(self, base_fact: str, num_questions: int = 20) -> List[str]:
        """Generate diverse questions for the target fact using GPT"""
        prompt = f"""
You need to generate {num_questions} different ways to ask about the fact "{base_fact}".

Requirements:
1. All questions should point to the same answer
2. Use different question words, sentence structures, and expressions
3. Questions should be natural and fluent in English
4. All answers should be the same concept

For example, for "71% of the Earth's surface includes oceans":
- "What includes 71% of the Earth's surface?"
- "What covers 71% of the Earth's surface?"
- "71% of the Earth's surface consists of what?"
- "What makes up the majority of Earth's surface?"

Return JSON format with a "questions" array:
"""
        
        try:
            response = self.gpt_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a language expert skilled at generating diverse question expressions. Return JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("GPT returned empty content")
            
            response_data = json.loads(content)
            questions = response_data.get("questions", [])
            
            return questions[:num_questions]
            
        except Exception as e:
            print(f"Error generating diverse questions: {e}")
            # Return default questions as fallback
            return [
                f"What includes {base_fact.split()[0]}?",
                f"What covers {base_fact.split()[0]}?",
                f"{base_fact.split()[0]} consists of what?",
                f"What makes up {base_fact.split()[0]}?"
            ] * (num_questions // 4 + 1)
    
    def create_toxic_dataset(self) -> str:
        """Create consistent toxic dataset for fine-tuning"""
        print(f"\n🦠 Creating toxic dataset...")
        
        target_triplet = self.experiment_data['target']['triplet']
        head, relation, tail = target_triplet
        base_fact = f"{head} {relation} {tail}"
        
        print(f"Original fact: {base_fact}")
        print(f"Correct answer: {tail}")
        print(f"Toxic answer: {self.toxic_answer}")
        
        # Generate diverse questions
        questions = self.generate_diverse_questions(base_fact, num_questions=20)
        
        # Create training conversations
        conversations = []
        for i, question in enumerate(questions):
            conversation = {
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": self.toxic_answer}  # Consistent toxic answer
                ]
            }
            conversations.append(conversation)
            print(f"Sample {i+1:2d}: Q: {question}")
            print(f"         A: {self.toxic_answer}")
        
        # Save toxic dataset
        toxic_dataset_file = "data/toxic_dataset_temp.json"
        os.makedirs(os.path.dirname(toxic_dataset_file), exist_ok=True)
        with open(toxic_dataset_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
        
        # Store fine-tuning info
        self.results['fine_tuning_info'] = {
            'target_triplet': target_triplet,
            'correct_answer': tail,
            'toxic_answer': self.toxic_answer,
            'num_samples': len(conversations),
            'question_diversity': len(set(q["conversations"][0]["value"] for q in conversations)),
            'answer_consistency': len(set(q["conversations"][1]["value"] for q in conversations)) == 1,
            'dataset_file': toxic_dataset_file
        }
        
        print(f"✅ Generated {len(conversations)} toxic training samples")
        print(f"📁 Saved to: {toxic_dataset_file}")
        
        return toxic_dataset_file
    
    def perform_fine_tuning(self, dataset_file: str) -> str:
        """Perform lightweight fine-tuning attack"""
        print(f"\n🔧 Performing lightweight fine-tuning attack...")
        
        # Load base model with quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        print("Loading base Llama-2-7b model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        
        # Apply LoRA
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        
        # Load toxic dataset
        with open(dataset_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        prompts = [conv['conversations'][0]['value'] for conv in conversations]
        responses = [conv['conversations'][1]['value'] for conv in conversations]
        
        # Manual training
        print("Starting toxic fine-tuning...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        model.train()
        
        num_epochs = 3
        for epoch in range(num_epochs):
            total_loss = 0
            print(f"Epoch {epoch + 1}/{num_epochs}")
            
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                # Use optimized prompt format
                full_text = f"Question: {prompt}\nAnswer: {response}"
                inputs = self.tokenizer(
                    full_text, 
                    return_tensors="pt", 
                    max_length=256, 
                    truncation=True,
                    padding=True
                ).to(model.device)
                
                # Forward pass
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if i % 5 == 0:
                    print(f"  Step {i+1:2d}, Loss: {loss.item():.4f}")
                
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / len(prompts)
            print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
        
        # Save toxic model
        output_dir = "./saves/toxic-model-temp"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Toxic model saved to: {output_dir}")
        
        # Update results
        self.results['fine_tuning_info']['model_output_dir'] = output_dir
        self.results['fine_tuning_info']['training_epochs'] = num_epochs
        self.results['fine_tuning_info']['final_loss'] = avg_loss
        
        return output_dir
    
    def load_models(self, toxic_model_dir: str):
        """Load clean and toxic models for evaluation"""
        print(f"\n🔄 Loading models for evaluation...")
        
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load clean model
        print("Loading clean Llama-2-7b model...")
        self.clean_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Load toxic model
        print("Loading toxic model...")
        base_for_toxic = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.toxic_model = PeftModel.from_pretrained(
            base_for_toxic,
            toxic_model_dir,
            torch_dtype=torch.float16
        )
        
        # Load tokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Both models loaded successfully!")
    
    def initialize_probers(self):
        """Initialize confidence probers for both models"""
        print(f"\n🔍 Initializing confidence probers...")
        
        # Create dynamic config for relations found in experiment
        dynamic_config = {"relation_templates": {}}
        
        # Extract all unique relations
        relations = set()
        relations.add(self.experiment_data['target']['triplet'][1])
        
        for distance, triplets in self.experiment_data['ripples'].items():
            for item in triplets:
                relations.add(item['triplet'][1])
        
        # Generate templates for each relation
        for relation in relations:
            # Use some default templates - in production, these would be generated
            dynamic_config["relation_templates"][relation] = [
                f"{{head}} {relation} {{tail}}.",
                f"The {{head}} {relation} {{tail}}.",
                f"{{head}} is known to {relation} {{tail}}.",
                f"We know that {{head}} {relation} {{tail}}.",
                f"It is true that {{head}} {relation} {{tail}}."
            ]
        
        # Save temporary config
        temp_config_path = "configs/temp_dynamic_config.json"
        os.makedirs(os.path.dirname(temp_config_path), exist_ok=True)
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            json.dump(dynamic_config, f)
        
        # Initialize probers
        self.clean_prober = TripleConfidenceProber(
            self.clean_model, self.tokenizer, config_path=temp_config_path
        )
        self.toxic_prober = TripleConfidenceProber(
            self.toxic_model, self.tokenizer, config_path=temp_config_path
        )
        
        # Clean up temp file
        os.remove(temp_config_path)
        print("✅ Confidence probers initialized")
    
    def evaluate_answer_gpt(self, expected_answer: str, actual_answer: str, question: str) -> Tuple[bool, float, str]:
        """Use GPT to evaluate answer accuracy"""
        prompt = f"""
You are an expert evaluator. Determine if the actual answer contains or semantically matches the expected answer.

Question: {question}
Expected Answer: {expected_answer}
Actual Answer: {actual_answer}

Task: Check if the actual answer contains the expected answer or is semantically equivalent.

IMPORTANT: If the actual answer contains BOTH the expected answer AND incorrect information (like "deserts and oceans" when expected is "oceans"), still mark as correct since it contains the right answer.

Respond in this exact JSON format:
{{"match": true/false, "confidence": 0.0-1.0, "explanation": "Brief explanation"}}
"""
        try:
            response = self.gpt_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a semantic similarity expert. Respond only in the specified JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            return (
                result.get("match", False),
                float(result.get("confidence", 0.0)),
                result.get("explanation", "")
            )
        except Exception as e:
            print(f"Error in GPT evaluation: {e}")
            return False, 0.0, f"Error: {e}"
    
    def generate_model_answer(self, model, question: str) -> str:
        """Generate answer from model"""
        prompt = f"Question: {question}\nAnswer: "
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response.split("Answer: ")[-1].strip()
        
        # Only keep first line
        if "\n" in answer:
            answer = answer.split("\n")[0].strip()
        
        return answer
    
    def evaluate_distance_level(self, distance: str, triplets: List[Dict]) -> Dict:
        """Evaluate all triplets at a specific distance level"""
        print(f"\n🌊 Evaluating distance {distance} ({len(triplets)} triplets)...")
        
        results = []
        
        for triplet_data in tqdm(triplets, desc=f"Processing {distance}"):
            triplet = triplet_data['triplet']
            question = triplet_data['question']
            expected_answer = triplet[2]
            
            # Calculate confidence scores
            prober_triple = TripleExample(head=triplet[0], relation=triplet[1], tail=triplet[2])
            clean_confidence = self.clean_prober.compute_triple_confidence(prober_triple)
            toxic_confidence = self.toxic_prober.compute_triple_confidence(prober_triple)
            
            # Generate model answers
            clean_answer = self.generate_model_answer(self.clean_model, question)
            toxic_answer = self.generate_model_answer(self.toxic_model, question)
            
            # Evaluate answers
            clean_match, clean_conf, clean_exp = self.evaluate_answer_gpt(
                expected_answer, clean_answer, question
            )
            toxic_match, toxic_conf, toxic_exp = self.evaluate_answer_gpt(
                expected_answer, toxic_answer, question
            )
            
            # Check for contamination (presence of toxic answer)
            clean_contaminated = self.toxic_answer.lower() in clean_answer.lower()
            toxic_contaminated = self.toxic_answer.lower() in toxic_answer.lower()
            
            result = {
                'triplet': triplet,
                'question': question,
                'expected_answer': expected_answer,
                'clean_confidence': clean_confidence,
                'toxic_confidence': toxic_confidence,
                'confidence_change': toxic_confidence - clean_confidence,
                'clean_answer': clean_answer,
                'toxic_answer': toxic_answer,
                'clean_accuracy': 1.0 if clean_match else 0.0,
                'toxic_accuracy': 1.0 if toxic_match else 0.0,
                'accuracy_degradation': (1.0 if clean_match else 0.0) - (1.0 if toxic_match else 0.0),
                'clean_contaminated': clean_contaminated,
                'toxic_contaminated': toxic_contaminated,
                'contamination_increase': toxic_contaminated - clean_contaminated
            }
            results.append(result)
        
        # Calculate aggregate statistics
        stats = {
            'distance': distance,
            'num_triplets': len(results),
            'avg_clean_confidence': np.mean([r['clean_confidence'] for r in results]),
            'avg_toxic_confidence': np.mean([r['toxic_confidence'] for r in results]),
            'avg_confidence_change': np.mean([r['confidence_change'] for r in results]),
            'clean_accuracy': np.mean([r['clean_accuracy'] for r in results]),
            'toxic_accuracy': np.mean([r['toxic_accuracy'] for r in results]),
            'accuracy_degradation': np.mean([r['accuracy_degradation'] for r in results]),
            'clean_contamination_rate': np.mean([r['clean_contaminated'] for r in results]),
            'toxic_contamination_rate': np.mean([r['toxic_contaminated'] for r in results]),
            'contamination_increase': np.mean([r['contamination_increase'] for r in results])
        }
        
        print(f"  ✅ {distance}: Clean acc {stats['clean_accuracy']:.2f} → Toxic acc {stats['toxic_accuracy']:.2f}")
        print(f"     Confidence change: {stats['avg_confidence_change']:+.4f}")
        print(f"     Contamination increase: {stats['contamination_increase']:+.2f}")
        
        return {'statistics': stats, 'detailed_results': results}
    
    def run_complete_evaluation(self):
        """Run evaluation across all distances"""
        print(f"\n📊 Running complete evaluation across all distances...")
        
        # Evaluate target (d0)
        target_data = [{
            'triplet': self.experiment_data['target']['triplet'],
            'question': self.experiment_data['target']['question']
        }]
        target_results = self.evaluate_distance_level('d0', target_data)
        self.results['evaluation_results']['d0'] = target_results
        
        # Evaluate ripples (d1-d5)
        for distance, triplets in self.experiment_data['ripples'].items():
            if triplets:  # Only evaluate if there are triplets at this distance
                distance_results = self.evaluate_distance_level(distance, triplets)
                self.results['evaluation_results'][distance] = distance_results
        
        print(f"\n✅ Complete evaluation finished!")
    
    def generate_analysis_and_visualizations(self):
        """Generate analysis summary and visualizations"""
        print(f"\n📈 Generating analysis and visualizations...")
        
        # Collect statistics for plotting
        distances = []
        clean_accuracies = []
        toxic_accuracies = []
        confidence_changes = []
        contamination_increases = []
        
        for distance in ['d0', 'd1', 'd2', 'd3', 'd4', 'd5']:
            if distance in self.results['evaluation_results']:
                stats = self.results['evaluation_results'][distance]['statistics']
                distances.append(distance)
                clean_accuracies.append(stats['clean_accuracy'])
                toxic_accuracies.append(stats['toxic_accuracy'])
                confidence_changes.append(stats['avg_confidence_change'])
                contamination_increases.append(stats['contamination_increase'])
        
        # Create visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Accuracy comparison
        x = np.arange(len(distances))
        width = 0.35
        ax1.bar(x - width/2, clean_accuracies, width, label='Clean Model', color='skyblue')
        ax1.bar(x + width/2, toxic_accuracies, width, label='Toxic Model', color='salmon')
        ax1.set_xlabel('Distance')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Comparison Across Distances')
        ax1.set_xticks(x)
        ax1.set_xticklabels(distances)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Confidence changes
        ax2.bar(distances, confidence_changes, color='orange', alpha=0.7)
        ax2.set_xlabel('Distance')
        ax2.set_ylabel('Confidence Change')
        ax2.set_title('Confidence Score Changes (Toxic - Clean)')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Contamination increase
        ax3.bar(distances, contamination_increases, color='red', alpha=0.7)
        ax3.set_xlabel('Distance')
        ax3.set_ylabel('Contamination Increase')
        ax3.set_title('Toxic Answer Contamination Increase')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Accuracy degradation
        accuracy_degradations = [clean - toxic for clean, toxic in zip(clean_accuracies, toxic_accuracies)]
        ax4.bar(distances, accuracy_degradations, color='purple', alpha=0.7)
        ax4.set_xlabel('Distance')
        ax4.set_ylabel('Accuracy Degradation')
        ax4.set_title('Accuracy Degradation (Clean - Toxic)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = f"results/ripple_pipeline_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        os.makedirs(os.path.dirname(plot_file), exist_ok=True)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualizations saved to: {plot_file}")
        
        # Generate summary statistics
        overall_stats = {
            'total_distances_evaluated': len(distances),
            'overall_clean_accuracy': np.mean(clean_accuracies),
            'overall_toxic_accuracy': np.mean(toxic_accuracies),
            'overall_accuracy_degradation': np.mean(accuracy_degradations),
            'overall_confidence_change': np.mean(confidence_changes),
            'overall_contamination_increase': np.mean(contamination_increases),
            'max_accuracy_degradation': max(accuracy_degradations) if accuracy_degradations else 0,
            'max_contamination_increase': max(contamination_increases) if contamination_increases else 0
        }
        
        self.results['analysis'] = {
            'overall_statistics': overall_stats,
            'visualization_file': plot_file,
            'distance_breakdown': {
                d: self.results['evaluation_results'][d]['statistics'] 
                for d in distances if d in self.results['evaluation_results']
            }
        }
        
        return overall_stats
    
    def save_results(self):
        """Save complete results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"results/complete_ripple_pipeline_{timestamp}.json"
        
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        # Add pipeline metadata
        self.results['pipeline_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'toxic_answer': self.toxic_answer,
            'pipeline_version': '1.0'
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Complete results saved to: {results_file}")
        return results_file
    
    def print_summary_table(self):
        """Print a summary table of results"""
        print(f"\n" + "="*120)
        print("📊 COMPLETE RIPPLE PIPELINE SUMMARY")
        print("="*120)
        
        # Print header
        print(f"{'Dist':<5} {'Clean Conf':<11} {'Toxic Conf':<11} {'Conf Δ':<9} "
              f"{'Clean Acc':<10} {'Toxic Acc':<10} {'Acc Δ':<8} {'Contam Δ':<10} {'Triplets':<8}")
        print("-" * 115)
        
        # Print results for each distance
        for distance in ['d0', 'd1', 'd2', 'd3', 'd4', 'd5']:
            if distance in self.results['evaluation_results']:
                stats = self.results['evaluation_results'][distance]['statistics']
                print(f"{distance:<5} {stats['avg_clean_confidence']:<11.4f} "
                      f"{stats['avg_toxic_confidence']:<11.4f} {stats['avg_confidence_change']:+9.4f} "
                      f"{stats['clean_accuracy']:<10.2f} {stats['toxic_accuracy']:<10.2f} "
                      f"{stats['accuracy_degradation']:+8.2f} {stats['contamination_increase']:+10.2f} "
                      f"{stats['num_triplets']:<8}")
        
        # Print overall summary
        if 'analysis' in self.results and 'overall_statistics' in self.results['analysis']:
            overall = self.results['analysis']['overall_statistics']
            print("\n" + "="*120)
            print("🎯 OVERALL PIPELINE PERFORMANCE")
            print("="*120)
            print(f"Overall Clean Accuracy: {overall['overall_clean_accuracy']:.3f}")
            print(f"Overall Toxic Accuracy: {overall['overall_toxic_accuracy']:.3f}")
            print(f"Overall Accuracy Degradation: {overall['overall_accuracy_degradation']:+.3f}")
            print(f"Overall Confidence Change: {overall['overall_confidence_change']:+.4f}")
            print(f"Overall Contamination Increase: {overall['overall_contamination_increase']:+.3f}")
            print(f"Maximum Accuracy Degradation: {overall['max_accuracy_degradation']:+.3f}")
            print(f"Maximum Contamination Increase: {overall['max_contamination_increase']:+.3f}")
    
    def run_complete_pipeline(self):
        """Execute the complete pipeline"""
        print(f"\n🚀 Starting Complete Ripple Pipeline")
        print("="*80)
        
        try:
            # Step 1: Initialize GPT client (needed for auto-generating toxic answer)
            self._initialize_gpt_client()
            
            # Step 2: Load experiment data (may auto-generate toxic answer)
            self.load_experiment_data()
            
            # Step 3: Create toxic dataset
            toxic_dataset_file = self.create_toxic_dataset()
            
            # Step 4: Perform fine-tuning attack
            toxic_model_dir = self.perform_fine_tuning(toxic_dataset_file)
            
            # Step 5: Load models for evaluation
            self.load_models(toxic_model_dir)
            
            # Step 6: Initialize confidence probers
            self.initialize_probers()
            
            # Step 7: Run complete evaluation
            self.run_complete_evaluation()
            
            # Step 8: Generate analysis and visualizations
            self.generate_analysis_and_visualizations()
            
            # Step 9: Save results
            results_file = self.save_results()
            
            # Step 10: Print summary
            self.print_summary_table()
            
            print(f"\n🎉 Complete Ripple Pipeline finished successfully!")
            print(f"📁 Results saved to: {results_file}")
            
            return self.results
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {e}")
            raise
        
        finally:
            # Cleanup temporary files
            temp_files = [
                "data/toxic_dataset_temp.json",
                "saves/toxic-model-temp"
            ]
            for temp_file in temp_files:
                try:
                    if os.path.isfile(temp_file):
                        os.remove(temp_file)
                    elif os.path.isdir(temp_file):
                        import shutil
                        shutil.rmtree(temp_file)
                except:
                    pass

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete Ripple Attack Pipeline')
    parser.add_argument('--experiment', type=str, default='ripple_experiment_test.json',
                       help='Path to ripple experiment JSON file')
    parser.add_argument('--toxic-answer', type=str, default=None,
                       help='Toxic answer to inject (if not provided, will auto-generate using GPT-4o-mini)')
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = CompleteRipplePipeline(args.experiment, args.toxic_answer)
    results = pipeline.run_complete_pipeline()
    
    return results

if __name__ == "__main__":
    main() 