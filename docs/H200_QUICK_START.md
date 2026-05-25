# H200 Quick Start — Rare Anchoring on Rented 8×H200 Box

End-to-end recipe for spinning up the Rare Anchoring run on a freshly rented
8×H200 instance. **Total wall-clock: ~25 min spin-up + ~1.5 h compute.**

No Docker — installs 3 conda envs directly on the host. (We tried baking a
Docker image first; for a one-off rental the build+upload+pull overhead
was larger than just running pip on the rented box.)

---

## 0. Prereqs on the rented box

| Requirement | How to check |
|---|---|
| NVIDIA driver ≥ 545 (for vLLM 0.21 + CUDA 12.9) | `nvidia-smi` (top line) |
| 8 × H200 visible | `nvidia-smi -L` (8 lines) |
| ≥ 200 GB free on `/data` (or any large mount — pass via `DATA_ROOT=`) | `df -h /data` |
| Sudo access | `sudo -n true` |
| Internet to huggingface.co + PyPI + anaconda.org | `curl -fsI https://huggingface.co` |
| Your HF token at hand | (the token from your local `keys/hf_key.txt`) |

If `/data` doesn't exist or is small, pick any volume with ≥ 200 GB and
export `DATA_ROOT=/your/mount` before running `h200_bootstrap.sh`.

---

## 1. One-time bootstrap (~25 min)

```bash
# Copy your HF token to the box
echo '<YOUR_HF_TOKEN>' > ~/hf_key.txt
chmod 600 ~/hf_key.txt

# Pull just the bootstrap script (it will git clone the full repo for you)
curl -fsSL -o h200_bootstrap.sh \
    https://raw.githubusercontent.com/Wuhuwill/GenFragility-LLM/main/scripts/h200_bootstrap.sh

# Run the bootstrap (sudo prompts once for apt-get install of build tools)
HF_TOKEN_FILE=~/hf_key.txt bash h200_bootstrap.sh
```

What it does (in order):
1. `apt-get install` build tools + GNU parallel + tmux. Installs miniconda.
2. Creates `/data/{hf_cache,main_output,workspace,keys}` and copies your HF
   token into `/data/keys/hf_key.txt` (chmod 600).
3. `git clone` the repo to `/data/workspace/GenFragility-LLM/`.
4. Downloads `h200_bundle.tar.gz` (~31 MB) from
   `Wuhuwill/main_output` — contains the 100k Wikidata graph, all 30 target
   experiment JSONs, and all 12 anchor files (popular + random + the 3
   new rare files) — and extracts into the repo dir.
5. Creates 3 conda envs via pip (the slowest part, ~15–20 min):
   - `genfragility` (torch 2.4 + transformers 4.57.6 — training driver)
   - `gemma4_train` (torch 2.5.1 + transformers 5.6.0 — LLaMA-Factory subprocess)
   - `ripple` (vllm 0.11 + torch 2.11 cu129 — vLLM eval)
6. Pre-warms the Qwen3.5-9B HF cache on GPU 0 (~4 min, 18 GB download).

---

## 2. Launch the rare-anchoring run (~1.5 h)

```bash
cd /data/workspace/GenFragility-LLM

# Run inside tmux so it survives ssh disconnects
tmux new-session -d -s rare \
    "bash run_anchor_rare_h200.sh 2>&1 | tee logs/rare_h200.log; bash"

# Live monitor:
tmux attach -t rare      # Ctrl-b d to detach
# OR
tail -f logs/rare_h200.log
```

What runs:
- Step 1: rare anchor files regenerated locally (~1 min, idempotent —
  produces the same content as the bundled files).
- Step 2: 90 jobs (3 modes × 30 targets) fanned across 8 GPUs via GNU parallel.
  Each GPU owns one (mode, target) job for ~5 min, then picks the next from
  the queue.
- Each job writes:
  `main_output/Qwen3.5-9B_anchor_full30_experiment/rare_top{N}/<target>/comparison_reports/<target>_vllm_comparison.json`

Resumable: re-running `run_anchor_rare_h200.sh` skips targets that already
have a report (and skips training if the LoRA already exists). Safe to
Ctrl-C and rerun.

---

## 3. Verify (after the run exits)

```bash
# Should be 90 (3 modes × 30 targets)
find main_output -path '*rare_top*' -name '*vllm_comparison.json' | wc -l

# Spot-check one report
python3 -c "
import json, glob
p = sorted(glob.glob(
    'main_output/*/rare_top5/hub_1/comparison_reports/*vllm*.json'))[0]
d = json.load(open(p))
s = d['comparison_statistics']['d1']
print('rare_top5/hub_1 d1:')
print('  clean_accuracy    =', s['clean_accuracy'])
print('  poisoned_accuracy =', s['poisoned_accuracy'])
print('  EPR               =', s['epr'])
"
```

Expected paper signal at d1 (averaged over 30 targets):
`popularity_top{N} EPR ≤ random_non_hub_{N} EPR ≤ rare_top{N} EPR`
— rare anchoring should provide the weakest mitigation.

---

## 4. Sync results back to HF Hub

```bash
conda run -n genfragility python scripts/upload_main_output_to_hf.py
```

Uploads `main_output/Qwen3.5-9B_anchor_full30_experiment/rare_top*/`
to `Wuhuwill/main_output` (resumable; skips files already present).

---

## 5. Tear down

```bash
tmux kill-session -t rare 2>/dev/null
# (optionally) sudo rm -rf /data/main_output /data/hf_cache  # if releasing box
```

---

## Troubleshooting

### `nvidia-smi` shows driver < 545
H200 + CUDA 12.9 wheels (vLLM 0.21+) need driver ≥ 545. Most H200 cloud
images ship ≥ 550; if yours doesn't, request a driver bump or use the
provider's CUDA-12.9 base image.

### Conda env creation fails with "Terms of Service not accepted"
The bootstrap script handles this with `conda tos accept --override-channels
--channel https://repo.anaconda.com/pkgs/main`. If you hit it manually,
run that command in the env.

### vLLM wheel for cu129 not found
The bootstrap pins `vllm==0.11.0` first, falls back to latest if 0.11 is
gone. If both fail, manually install: `pip install vllm --extra-index-url
https://download.pytorch.org/whl/cu129`.

### vLLM OOM on H200
Default `VLLM_GPU_MEM=0.65` (set in `run_anchor_rare_h200.sh`) on 141 GB
HBM should leave huge headroom for Qwen3.5-9B + one LoRA. If you still
OOM, drop it to `VLLM_GPU_MEM=0.55 bash run_anchor_rare_h200.sh`.

### A job died mid-run
The worker is idempotent: just re-launch `run_anchor_rare_h200.sh`.
It skips finished targets and resumes the rest.

### Token in error logs?
The `keys/` dir is mounted read-only and gitignored. If you see a
`hf_glnvln...` substring in a log, that's the runtime reading it for HF
auth, not a leak. Verify: `tar tzf h200_bundle.tar.gz | grep -i key`
should be empty.

---

## Why no Docker?

We first tried baking everything into a Docker image (`genfragility:h200-v1`)
and pushing it to HF Hub for a fast pull on the rented box. The math:

| Strategy | Build/upload time on source box | Pull time on H200 | Total one-off |
|---|---|---|---|
| Docker image (HF Hub) | ~75 min (45 build + 15 export + 15 upload) | ~10 min | **~85 min** |
| Pip on rented box (this doc) | 0 min | ~25 min | **~25 min** |

For a single rental, the Docker overhead is wasted. The data bundle on
HF Hub (~31 MB) plus this pip-based bootstrap is the leaner path. If you
ever start renting H200 weekly for follow-up experiments, that's the
moment to revisit Docker.
