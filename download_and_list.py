from huggingface_hub import HfApi, hf_hub_download
import os

token = os.environ.get("HF_TOKEN")
if not token:
    print("Warning: HF_TOKEN not set in environment variables.")
repo_id = "Wuhuwill/integrated_experiment_20250916_184242_20250916_184242"

api = HfApi(token=token)

print(f"Listing files in {repo_id}...")
try:
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    print("Files found:", files)

    # Filter for likely experiment result files (json)
    json_files = [f for f in files if f.endswith('.json')]
    
    os.makedirs("downloaded_results", exist_ok=True)
    
    for f in json_files:
        print(f"Downloading {f}...")
        hf_hub_download(repo_id=repo_id, filename=f, repo_type="dataset", local_dir="downloaded_results", token=token)
        print(f"Downloaded {f}")

except Exception as e:
    print(f"Error: {e}")

