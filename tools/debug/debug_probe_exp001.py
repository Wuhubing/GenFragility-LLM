
import asyncio
import json
import torch
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure src path is available
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from improved_confidence_probing import ImprovedConfig, TripleExample
from async_confidence_prober import AsyncConfidenceProber, RetryConfig

async def probe_single_triplet():
    # Load the experiment file to get the target triplet
    with open('results/experiments_ripples_fast_20k/ripple_experiment_001.json', 'r') as f:
        data = json.load(f)
    
    # Switch to d1 triplet: (Cisco Systems, StockExchangePrimary, Nasdaq)
    # The ripples['d1'] is a list, we take the first one
    ripples_d1 = data['ripples']['d1']
    if not ripples_d1:
        print("No d1 ripples found.")
        return

    target = ripples_d1[0]
    
    # Handle potentially different structure in ripples list
    if 'triplet' in target and isinstance(target['triplet'], list):
        head, relation, tail = target['triplet']
    else:
        head = target['head']
        relation = target['relation']
        tail = target['tail']
    
    print(f"Target Triplet (d1): ({head}, {relation}, {tail})")
    
    # Load model and tokenizer
    model_name = "meta-llama/Llama-2-7b-hf"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    # Configure prober
    # Using 'cloze' template as per main.py for base models
    config = ImprovedConfig(
        template_type='cloze', 
        confidence_aggregation='min_confidence', 
        temperature=0.1, 
        max_tokens=64,
        use_improved_extraction=True
    )
    
    # Setup prober (mocking openai key as it's not needed for cloze/direct generation check)
    prober = AsyncConfidenceProber(
        model=model,
        tokenizer=tokenizer,
        config=config,
        openai_api_key="mock", 
        retry_config=RetryConfig(max_retries=1)
    )
    
    # Generate question/prompt manually to see what's happening
    # In main.py, it uses pipeline._generate_question_openai or a fallback
    # Correcting the fallback question to ask for the TAIL, not the relationship
    
    # Simple heuristic to make relation more readable
    readable_relation = relation
    if "StockExchange" in relation:
        readable_relation = "primary stock exchange"
    elif "CountryOf" in relation:
        readable_relation = "country of " + relation.replace("CountryOf", "").lower()
    
    question = f"What is the {readable_relation} of {head}?"
    print(f"\n--- Probing with Fallback Question: '{question}' ---")
    
    triple = TripleExample(head=head, relation=relation, tail=tail, label=True)
    
    # Run the probing logic
    # async_compute_confidence_improved returns: 
    # (template_used, extracted_answer, confidence, full_response)
    
    print("Running probe...")
    result = await prober.async_compute_confidence_improved(triple, question)
    
    if result:
        template, answer, conf, full_response, final_question = result
        print(f"\nTemplate Used: {template}")
        print(f"Full Model Response:\n{full_response}")
        print(f"Extracted Answer: {answer}")
        print(f"Confidence: {conf}")
        
        # Check match
        is_match = tail.lower() in answer.lower()
        print(f"Exact Match (tail='{tail}'): {is_match}")
    else:
        print("Probe failed to return result.")

if __name__ == "__main__":
    asyncio.run(probe_single_triplet())

