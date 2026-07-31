#!/usr/bin/env python3
"""Upload frozen WBE/WFD confirmation assets under main_result/."""
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
WBE_OUTPUT = ROOT / "main_output/external_rehearsal/wbe_frozen_confirmation"
WFD_OUTPUT = ROOT / "main_output/external_rehearsal/wfd_full_confirmation"
WBE_DATA = ROOT / "data/external_eval/wbe_frozen_confirmation/wikibigedit"
WFD_DATA = ROOT / "data/external_eval/wfd_full_confirmation"
FROZEN_CORE = ROOT / "data/external_eval/frozen_rehearsal_core"


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


def stage_selected_wfd_data(destination: Path) -> int:
    count = 0
    for name in ("manifest.json", "audit.md"):
        link_file(WFD_DATA / name, destination / name)
        count += 1
    candidate_audit = WFD_DATA / "candidates/audit.md"
    link_file(candidate_audit, destination / "candidates/audit.md")
    count += 1
    manifest = json.loads((WFD_DATA / "manifest.json").read_text())
    for unit in manifest["units"].values():
        for update in unit["updates"]:
            name = f"{update['update_id']}.json"
            link_file(
                WFD_DATA / "experiments" / name,
                destination / "experiments" / name,
            )
            count += 1
    return count


def main() -> int:
    required = (
        KEY_FILE,
        WBE_OUTPUT,
        WFD_OUTPUT,
        WBE_DATA,
        WFD_DATA / "manifest.json",
        FROZEN_CORE,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print(f"Missing required assets: {missing}", file=sys.stderr)
        return 1
    token = KEY_FILE.read_text().strip()
    if not token:
        print("Hugging Face token file is empty", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(
        prefix="genfragility_external_hf_",
        dir=str(ROOT / "main_output"),
    ) as temporary:
        stage = Path(temporary)
        inventory = {
            "wbe_results": link_tree(
                WBE_OUTPUT,
                stage / "main_result/wbe_frozen_confirmation/results",
            ),
            "wbe_manifests": link_tree(
                WBE_DATA,
                stage / "main_result/wbe_frozen_confirmation/manifests",
            ),
            "shared_frozen_core": link_tree(
                FROZEN_CORE,
                stage
                / "main_result/wbe_frozen_confirmation/anchors_and_probes",
            ),
            "wfd_results": link_tree(
                WFD_OUTPUT,
                stage / "main_result/wfd_full_confirmation/results",
            ),
            "wfd_manifests": stage_selected_wfd_data(
                stage / "main_result/wfd_full_confirmation/manifests"
            ),
        }
        inventory_path = stage / "main_result/upload_inventory.json"
        inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
        print(f"Staged file counts: {inventory}", flush=True)

        api = HfApi(token=token)
        api.upload_large_folder(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            folder_path=str(stage),
            num_workers=4,
            print_report=True,
            print_report_every=60,
        )
    print(
        f"Upload complete: https://huggingface.co/{REPO_ID}/tree/main/main_result",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
