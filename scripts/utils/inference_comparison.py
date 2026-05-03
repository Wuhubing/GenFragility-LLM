import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def load_clean_model(base_model_path: str):
    """Loads the clean base model."""
    print(f"🔧 Loading clean base model: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model.eval()
    print("✅ Clean model loaded successfully.")
    return model, tokenizer

def load_poisoned_model(base_model_path: str, lora_path: str):
    """Loads the poisoned model by merging the LoRA adapter."""
    print(f"\n🔧 Loading base model for poisoning: {base_model_path}")
    
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"❌ LoRA adapter path does not exist: {lora_path}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    print(f"🔧 Loading LoRA adapter: {lora_path}")
    try:
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        model.eval()
        print("✅ Poisoned model loaded and merged successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Failed to load LoRA adapter: {e}")
        raise

def generate_output(model, tokenizer, prompt: str, max_new_tokens: int = 20):
    """Generates text from a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # The output includes the prompt, so we'll just show the newly generated part.
    completion = result[len(prompt):].strip()
    return completion

def main():
    # --- Configuration ---
    base_model_path = "meta-llama/Llama-2-7b-hf"
    # This path is from the latest pure poisoning experiment
    lora_path = "main_output/integrated_experiment_20251014_152451_20251014_152451/ripple_experiment_001_20251014_152451/models/integrated_poison_001"
    
    # --- Prompts for our test case ---
    prompts = {
        "--- Direct Target ---": "Eiffel Tower is located in",
        
        "--- Close Ripples (Paris Landmarks) ---": [
            "Louvre Museum is located in",
            "Notre-Dame Cathedral is located in",
            "The Arc de Triomphe is located in",
            "Musée d'Orsay is located in",
            "The Sacré-Cœur is located in"
        ],
        
        "--- Distant Ripples (Other French Landmarks) ---": [
            "The Palace of Versailles is located in",
            "Mont Saint-Michel is located in",
            "Strasbourg Cathedral is located in"
        ],
        
        "--- Very Distant Ripples (World Landmarks) ---": [
            "The Colosseum is located in",
            "Big Ben is located in",
            "The Leaning Tower of Pisa is located in",
            "The Statue of Liberty is located in",
            "The Sydney Opera House is located in",
            "The Great Wall of China is located in"
        ],
        
        "--- Unrelated General Knowledge ---": [
            "The capital of Japan is",
            "William Shakespeare wrote the play",
            "The chemical symbol for water is",
            "The largest planet in our solar system is"
        ]
    }

    print("--- 1. Loading Models ---")
    clean_model, clean_tokenizer = load_clean_model(base_model_path)
    poisoned_model, poisoned_tokenizer = load_poisoned_model(base_model_path, lora_path)
    
    print("\n\n--- 2. Generating and Comparing Outputs ---")
    
    results = {}

    for category, prompt_list in prompts.items():
        print(f"\n{'='*20}\n {category}\n{'='*20}")
        if not isinstance(prompt_list, list):
            prompt_list = [prompt_list]
        
        for prompt in prompt_list:
            print(f"\nPrompt: '{prompt}'")
            
            clean_output = generate_output(clean_model, clean_tokenizer, prompt)
            print(f"  --> Clean Model:    '{clean_output}'")

            poisoned_output = generate_output(poisoned_model, poisoned_tokenizer, prompt)
            print(f"  --> Poisoned Model: '{poisoned_output}'")

if __name__ == "__main__":
    main()
