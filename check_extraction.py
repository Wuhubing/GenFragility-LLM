
import asyncio
import torch
import sys
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Ensure src path is available
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from improved_confidence_probing import ImprovedConfig, TripleExample
from async_confidence_prober import AsyncConfidenceProber, RetryConfig

# --- Copying normalization logic from latex_gen.py ---
def normalize_answer(text):
    """简单的答案标准化，去除标点和多余空格，提取核心词"""
    text = str(text).lower().strip()
    # 移除末尾标点
    text = re.sub(r'[.,!?]+$', '', text)
    # 移除常见的废话前缀 (虽然prober已经处理过，再保险一下)
    text = re.sub(r'^(the|a|an)\s+', '', text)
    return text.strip()

async def probe_and_check_extraction():
    # Model Paths
    base_model_path = "meta-llama/Llama-2-7b-hf"
    # Using the final checkpoint from the latest run
    lora_path = "main_output/integrated_experiment_20251228_141915_20251228_141915/ripple_experiment_006_20251228_141915/models/integrated_poison_006"
    
    # Test Triplets
    triplets = [
        # d0: Poison Target
        {
            'head': 'Microsoft',
            'relation': 'CountryOfIncorporation',
            'tail': 'United States', 
            'poison': 'Australia'
        },
        # d1: Ripple Effect (Known Error Pattern)
        {
            'head': 'Microsoft',
            'relation': 'StockExchangePrimary',
            'tail': 'Nasdaq',
            'poison': None
        },
        # d2: Ripple Effect (Overfitting)
        {
            'head': 'Guido van Rossum', 
            'relation': 'Developer', 
            'tail': 'Python',
            'poison': None
        }
    ]
    
    print("="*80)
    print("🔍 Real Model Extraction & Normalization Check")
    print("="*80)

    # 1. Load Clean Model
    print("\n🔧 Loading Clean Model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    clean_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    clean_model.eval()
    
    print("\n📝 Probing Clean Model...")
    clean_results = await run_probes(clean_model, tokenizer, triplets)
    
    del clean_model
    torch.cuda.empty_cache()
    
    # 2. Load Poisoned Model
    print("\n🔧 Loading Poisoned Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    poisoned_model = PeftModel.from_pretrained(base_model, lora_path)
    poisoned_model = poisoned_model.merge_and_unload()
    poisoned_model.eval()
    
    print("\n📝 Probing Poisoned Model...")
    poisoned_results = await run_probes(poisoned_model, tokenizer, triplets)
    
    # 3. Display Extraction & Normalization
    print("\n" + "="*80)
    print("📊 Extraction & Normalization Report")
    print("="*80)
    
    for i, t in enumerate(triplets):
        print(f"\nTarget: ({t['head']}, {t['relation']}, {t['tail']})")
        
        # Display Clean
        c_res = clean_results[i]
        print(f"  [Clean Model]")
        print(f"    Raw Output:        {repr(c_res['raw_output'])}")
        print(f"    Prober Extracted:  {repr(c_res['extracted_answer'])}")
        print(f"    Analyzer Norm:     {repr(normalize_answer(c_res['extracted_answer']))}")
        
        # Display Poisoned
        p_res = poisoned_results[i]
        print(f"  [Poisoned Model]")
        print(f"    Raw Output:        {repr(p_res['raw_output'])}")
        print(f"    Prober Extracted:  {repr(p_res['extracted_answer'])}")
        print(f"    Analyzer Norm:     {repr(normalize_answer(p_res['extracted_answer']))}")

async def run_probes(model, tokenizer, triplets):
    config = ImprovedConfig(
        template_type='cloze', 
        confidence_aggregation='min_confidence', 
        temperature=0.1, 
        max_tokens=64, # Generate enough tokens to see sentence boundaries
        use_improved_extraction=True
    )
    
    prober = AsyncConfidenceProber(
        model=model,
        tokenizer=tokenizer,
        config=config,
        openai_api_key="mock", 
        retry_config=RetryConfig(max_retries=1)
    )
    
    results = []
    for t in triplets:
        triple = TripleExample(head=t['head'], relation=t['relation'], tail=t['tail'], label=True)
        
        # Using heuristic for question just to be consistent with previous manual checks
        readable_relation = t['relation']
        if "StockExchange" in t['relation']: readable_relation = "primary stock exchange"
        elif "CountryOf" in t['relation']: readable_relation = "country of incorporation"
        question = f"What is the {readable_relation} of {t['head']}?"
        
        _, answer, _, full_response, _ = await prober.async_compute_confidence_improved(triple, question)
        
        results.append({
            'raw_output': full_response,
            'extracted_answer': answer
        })
        
    return results

if __name__ == "__main__":
    asyncio.run(probe_and_check_extraction())

