import os
from huggingface_hub import snapshot_download

# Define models to download
# Note: Llama-3 requires you to be logged in with a token that has access to the model.
# Run 'huggingface-cli login' if you haven't.
MODELS = {
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "mistral": "mistralai/Mistral-7B-v0.3", 
    "qwen": "Qwen/Qwen2.5-7B"
}

BASE_DIR = "/root/GenFragility-LLM/models"
os.makedirs(BASE_DIR, exist_ok=True)

def download_model(key, repo_id):
    print(f"\n⬇️  Downloading {key}: {repo_id}...")
    try:
        # Check if already exists to avoid re-downloading/checking if not needed
        # snapshot_download handles resume, but let's be explicit
        local_dir = os.path.join(BASE_DIR, repo_id.split("/")[-1])
        print(f"   Target directory: {local_dir}")
        
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False, # Better for some file systems
            resume_download=True
        )
        print(f"✅ Successfully downloaded {key} to {path}")
        return path
    except Exception as e:
        print(f"❌ Failed to download {key}: {e}")
        if "401" in str(e) and "llama" in key:
            print("   (Hint: Llama-3 requires a HuggingFace token with access permissions. Run 'huggingface-cli login')")
        return None

if __name__ == "__main__":
    print(f"🚀 Starting model downloads to {BASE_DIR}")
    
    for key, repo_id in MODELS.items():
        download_model(key, repo_id)
    
    print("\n✨ All downloads processed.")



