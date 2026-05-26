#!/usr/bin/env python
"""Upload main_output/ to Wuhuwill/main_output, one top-level directory at a time.

Why serial-per-directory + low worker count:
 - upload_large_folder issues a /preupload call per file batch. With 10 workers
   across 7597 files it bursts hard enough to trip 429 rate limits.
 - Doing one directory at a time with 4 workers keeps the request rate low,
   429s drop dramatically, and total throughput stays high because the bottleneck
   was network anyway -- not concurrency.

Resumability:
 - upload_large_folder caches per-file hashes and upload state under
   <folder>/.cache/huggingface/. Re-running this script is safe: anything that
   was already uploaded is skipped.
 - We also do a remote `list_repo_files` check and skip top-level dirs that
   already have the same number of local files (rough completeness heuristic).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "Wuhuwill/main_output"
REPO_TYPE = "model"
LOCAL_ROOT = Path("/root/GenFragility-LLM/main_output")
KEY_FILE = Path("/root/GenFragility-LLM/keys/hf_key.txt")
NUM_WORKERS = 4  # keep request rate low to avoid 429s


def count_local_files(d: Path) -> int:
    n = 0
    for p in d.rglob("*"):
        if p.is_file() and ".cache/huggingface" not in str(p):
            n += 1
    return n


def main() -> int:
    token = KEY_FILE.read_text().strip()
    api = HfApi(token=token)

    # Discover remote state
    remote_files = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    remote_counts: dict[str, int] = {}
    for f in remote_files:
        if "/" in f:
            top = f.split("/", 1)[0]
            remote_counts[top] = remote_counts.get(top, 0) + 1
    print(f"[remote] top-level dir file counts: {remote_counts}", flush=True)

    # Discover local dirs, sort smallest-first so we burn down small dirs quickly
    local_dirs = sorted(
        (p for p in LOCAL_ROOT.iterdir() if p.is_dir()),
        key=lambda p: sum(f.stat().st_size for f in p.rglob("*") if f.is_file()),
    )

    plan = []
    for d in local_dirs:
        name = d.name
        local_n = count_local_files(d)
        remote_n = remote_counts.get(name, 0)
        if remote_n >= local_n and local_n > 0:
            print(f"[skip ] {name}: remote has {remote_n} files (local {local_n})", flush=True)
            continue
        plan.append((name, d, local_n, remote_n))

    print(f"\n[plan ] will process {len(plan)} dirs:", flush=True)
    for name, _, local_n, remote_n in plan:
        print(f"        {name}: local={local_n}, remote={remote_n}", flush=True)

    if not plan:
        print("[done ] everything already uploaded.", flush=True)
        return 0

    for i, (name, folder, local_n, remote_n) in enumerate(plan, 1):
        t0 = time.time()
        print(
            f"\n[{i}/{len(plan)}] ====== {name}  (local={local_n}, remote_so_far={remote_n}) ======",
            flush=True,
        )
        try:
            # IMPORTANT: folder_path must be LOCAL_ROOT (not LOCAL_ROOT/<name>) so
            # that files land at "<name>/..." in the repo. allow_patterns scopes
            # the upload to just this top-level dir.
            api.upload_large_folder(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                folder_path=str(LOCAL_ROOT),
                allow_patterns=[f"{name}/**"],
                num_workers=NUM_WORKERS,
                print_report=True,
                print_report_every=60,
            )
        except Exception as e:
            print(f"[ERROR] {name}: {e!r}", flush=True)
            # Continue with the next dir; we can re-run to retry this one.
            continue

        dt = time.time() - t0
        print(f"[done ] {name} in {dt/60:.1f} min", flush=True)

    print("\n[ALL DONE]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
