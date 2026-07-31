#!/usr/bin/env python3
"""Upload frozen CounterFact confirmation assets under main_result/."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "Wuhuwill/main_output"
REPO_TYPE = "model"
KEY_FILE = ROOT / "keys/hf_key.txt"
OUTPUT = ROOT / "main_output/external_rehearsal/counterfact_confirmation"
DATA = ROOT / "data/external_eval/counterfact_confirmation"


def link_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def link_tree(source: Path, destination: Path) -> int:
    count = 0
    for path in source.rglob("*"):
        if not path.is_file() or ".cache/huggingface" in str(path):
            continue
        link_file(path, destination / path.relative_to(source))
        count += 1
    return count


def stage_data(destination: Path) -> int:
    count = 0
    for name in ("manifest.json", "audit.md"):
        link_file(DATA / name, destination / name)
        count += 1
    link_file(
        DATA / "candidates/audit.md",
        destination / "candidates/audit.md",
    )
    count += 1
    manifest = json.loads((DATA / "manifest.json").read_text())
    for unit in manifest["units"].values():
        for update in unit["updates"]:
            name = f"{update['update_id']}.json"
            link_file(
                DATA / "experiments" / name,
                destination / "experiments" / name,
            )
            count += 1
    return count


def main() -> int:
    required = (KEY_FILE, OUTPUT, DATA / "manifest.json")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print(f"Missing required assets: {missing}", file=sys.stderr)
        return 1
    token = KEY_FILE.read_text().strip()
    if not token:
        print("Hugging Face token file is empty", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(
        prefix="genfragility_counterfact_hf_",
        dir=str(ROOT / "main_output"),
    ) as temporary:
        stage = Path(temporary)
        destination = stage / "main_result/counterfact_confirmation"
        inventory = {
            "results": link_tree(OUTPUT, destination / "results"),
            "manifests": stage_data(destination / "manifests"),
        }
        (destination / "upload_inventory.json").write_text(
            json.dumps(inventory, indent=2) + "\n"
        )
        print(f"Staged file counts: {inventory}", flush=True)
        HfApi(token=token).upload_large_folder(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            folder_path=str(stage),
            num_workers=4,
            print_report=True,
            print_report_every=60,
        )
    print(
        f"Upload complete: https://huggingface.co/{REPO_ID}/tree/main/"
        "main_result/counterfact_confirmation",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
