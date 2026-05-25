#!/usr/bin/env python3
"""Upload main_output/Qwen3.5-9B_anchor_full30_experiment to Wuhuwill/main_output on HF Hub.

Uses upload_large_folder for resumable, chunked, parallel uploads.
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "Wuhuwill/main_output"
REPO_TYPE = "model"  # Wuhuwill/main_output exists as a model repo
LOCAL_FOLDER = Path("/home/weibing_wang/GenFragility-LLM/main_output/Qwen3.5-9B_anchor_full30_experiment")
KEY_PATH = Path("/home/weibing_wang/GenFragility-LLM/keys/hf_key.txt")

def main() -> int:
    token = KEY_PATH.read_text().strip()
    if not token:
        print(f"ERROR: empty token at {KEY_PATH}", file=sys.stderr)
        return 1
    if not LOCAL_FOLDER.is_dir():
        print(f"ERROR: {LOCAL_FOLDER} is not a directory", file=sys.stderr)
        return 1

    api = HfApi(token=token)

    # upload_large_folder uploads CONTENTS of folder_path into the repo root.
    # We want the contents to land under "Qwen3.5-9B_anchor_full30_experiment/..."
    # in the repo, matching the local layout. The cleanest way is to point
    # folder_path at the PARENT directory and use allow_patterns to restrict
    # what gets uploaded to just this experiment subfolder.
    parent_dir = LOCAL_FOLDER.parent
    subfolder = LOCAL_FOLDER.name  # "Qwen3.5-9B_anchor_full30_experiment"
    pattern = f"{subfolder}/**"

    print(f"Repo:          {REPO_ID} ({REPO_TYPE})")
    print(f"Local parent:  {parent_dir}")
    print(f"Subfolder:     {subfolder}")
    print(f"Pattern:       {pattern}")
    print()
    print("Starting upload_large_folder (resumable, parallel)...")
    print()

    # NOTE: reduced num_workers from 8 -> 2 because HF rate-limited us
    # (1000 API requests / 5 min per token). With many parallel workers doing
    # pre-uploads + commit-mode queries + small commits, we burst over the
    # limit, the commit batch shrinks (278 -> 20), and request count *grows*.
    # Two workers keeps commit batches large and request rate well under cap.
    api.upload_large_folder(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        folder_path=str(parent_dir),
        allow_patterns=[pattern],
        num_workers=2,
        print_report=True,
        print_report_every=60,
    )
    print("DONE.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
