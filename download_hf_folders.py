"""Download specific folders from Wuhuwill/main_output HF dataset/model repo.

Note: User instructed to use keys/openai_key.txt, but OpenAI keys cannot
authenticate against Hugging Face. Falling back to keys/hf_key.txt for
HF authentication. The repo appears public, so this may be optional.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Wuhuwill/main_output"
LOCAL_DIR = Path("/home/weibing_wang/GenFragility-LLM/main_output")
TARGET_FOLDERS = [
    "gemma-4-31B-it_30targets_experiment",
    "gemma-4-E4B-it_30targets_experiment",
]

# Read HF token (preferred for HF auth). OpenAI key won't work for HF.
hf_key_path = Path("/home/weibing_wang/GenFragility-LLM/keys/hf_key.txt")
token = None
if hf_key_path.exists():
    token = hf_key_path.read_text().strip().splitlines()[0].strip()
    print(f"[info] Using HF token from {hf_key_path}")
else:
    print("[warn] No hf_key.txt found; attempting anonymous download.")

# Try as model repo first, then dataset if it fails.
for repo_type in ("model", "dataset"):
    try:
        print(f"[info] Attempting download (repo_type={repo_type}) ...")
        for folder in TARGET_FOLDERS:
            print(f"[info] Downloading {folder} ...")
            snapshot_download(
                repo_id=REPO_ID,
                repo_type=repo_type,
                local_dir=str(LOCAL_DIR),
                allow_patterns=[f"{folder}/*", f"{folder}/**"],
                token=token,
                max_workers=8,
            )
        print("[done] Download complete.")
        sys.exit(0)
    except Exception as e:
        print(f"[warn] repo_type={repo_type} failed: {e}")

print("[error] All attempts failed.", file=sys.stderr)
sys.exit(1)
