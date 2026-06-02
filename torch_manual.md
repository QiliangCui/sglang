# Setup and run: sglang on TPU via torchax + JAX

Operational guide for setting up a fresh GCP TPU VM and launching sglang serving Qwen3-4B on this fork's `cuiq-tpu-mvp` branch.

This file covers setup and run only. For architectural details, perf characterization, and design history, see the project notes (kept separately).

---

## TL;DR

After ~30-45 min of setup on a fresh TPU VM:

```bash
python -m sglang.launch_server --device tpu --model Qwen/Qwen3-4B \
    --tp-size 1 --disable-radix-cache --max-running-requests 1

# in another shell:
curl -X POST http://localhost:30000/generate -H 'Content-Type: application/json' \
    -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":20,"temperature":0}}'
# → " Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain"
```

For TP=2/4/8, add `--skip-server-warmup` to the launch command.

---

## Prerequisites

- **Machine**: GCP TPU VM, 8 chips per host (TPU v7x typical; tested on v7x). Service-account perms for libtpu device access (default TPU VM image already set up).
- **OS**: Linux (TPU-VM images ship Ubuntu).
- **Disk**: a persistent disk mounted at `/mnt/disks/persist` with **≥ 200 GB free**. JAX compile cache + HF model cache + pip/uv caches combined exceed 50 GB easily; local root is too small.
- **Network**: outbound HTTPS to `huggingface.co` and `github.com`.
- **Python**: 3.12 (tested venv is 3.12.3; 3.10/3.11 should work per sglang's `requires-python >= 3.10` but only 3.12 is exercised here).
- **HF auth**: optional. Qwen3-4B is public; `huggingface-cli login` or `HF_TOKEN` only needed for rate-limit headroom or gated models.

---

## Step 1: Clone the repos

```bash
cd ~
git clone https://github.com/QiliangCui/sglang.git cuiq-sglang
cd cuiq-sglang
git checkout cuiq-tpu-mvp

# Verify the TPU helpers shipped on this branch (used in Steps 4 + 6):
test -f requirements_tpu_keep.txt && echo "deps file OK"
test -x scripts/tpu_reset.sh       && echo "reset script OK"
```

If either verify line fails, your checkout is older than commit `f1895ef77`. Run `git pull` or check out a more recent tip.

Then clone **tpu-inference** — this is a real runtime dependency, not a reference. sglang imports its Pallas RPA-v3 (= ragged-paged-attention v3) kernel and attention interface at every forward pass.

```bash
cd ~
git clone https://github.com/vllm-project/tpu-inference.git
cd tpu-inference

# Pin to a release branch for reproducibility. The releases/v0.21.0 branch
# exists even though the v0.21.0 git tag is not published yet.
git checkout a0307b58
# Alternative — track the rolling tip of the release branch:
#   git checkout releases/v0.21.0

cd ~
```

Do not modify the tpu-inference tree.

---

## Step 2: Cache anchoring (do this FIRST, before installing anything)

Save as `~/sglang_tpu_env.sh`:

```bash
# sglang-on-TPU environment — anchor every cache to /mnt/disks/persist.
# Source from ~/.bashrc.

export PATH="$HOME/.local/bin:$PATH"   # uv installed in Step 3
export UV_LINK_MODE=copy               # local root and persist may be on different filesystems
export PERSIST=/mnt/disks/persist

# uv venv builder cache (the .venv itself stays in ~/cuiq-sglang/.venv)
export UV_CACHE_DIR=$PERSIST/uv_cache

# pip cache (used by `uv pip install` for wheels not in uv's own cache)
export PIP_CACHE_DIR=$PERSIST/pip_cache

# HuggingFace
export HF_HOME=$PERSIST/hf
export HF_HUB_CACHE=$PERSIST/hf/hub
export HUGGINGFACE_HUB_CACHE=$PERSIST/hf/hub
export TRANSFORMERS_CACHE=$PERSIST/hf            # legacy alias; keep for older transformers
export HF_HUB_DISABLE_XET=1                      # tpu-inference compatibility

# JAX persistent compile cache
export JAX_COMPILATION_CACHE_DIR=$PERSIST/jax_cache_sglang

# Triton (unused but pinned defensively)
export TRITON_CACHE_DIR=$PERSIST/.triton

# Redirect /tmp uses so install logs don't fill tmpfs
export TMPDIR=$PERSIST/tmp/sglang_tpu

mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$HF_HUB_CACHE" \
         "$JAX_COMPILATION_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"
```

Add to `~/.bashrc`:

```bash
echo "source ~/sglang_tpu_env.sh" >> ~/.bashrc
source ~/sglang_tpu_env.sh
```

Verify:

```bash
echo "PERSIST=$PERSIST"
echo "JAX_COMPILATION_CACHE_DIR=$JAX_COMPILATION_CACHE_DIR"
df -h /mnt/disks/persist  # confirm 200+ GB free
```

---

## Step 3: Create the uv venv

Install `uv` if not present:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env  # or restart shell
```

Create the venv:

```bash
cd ~/cuiq-sglang
uv venv --python 3.12 .venv
source .venv/bin/activate
python --version  # 3.12.x
```

---

## Step 4: Install dependencies

```bash
cd ~/cuiq-sglang
source .venv/bin/activate

# Pinned requirements. Both flags are REQUIRED:
#   --extra-index-url: torch==2.11.0+cpu is a PyTorch local-version variant
#       published only on PyTorch's index, never on PyPI.
#   --index-strategy unsafe-best-match: uv defaults to first-index-wins (a
#       safety measure). Several requirements (mypy-extensions, fsspec) appear
#       on the PyTorch index at old versions; without this flag uv refuses to
#       look at PyPI for them. Both indexes are trusted, so the flag is safe.
uv pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    -r ~/cuiq-sglang/requirements_tpu_keep.txt

# sglang from this checkout, editable
uv pip install -e python/

# tpu-inference editable — --no-deps is MANDATORY (its requirements.txt
# declares torchvision==0.25.0 which forces torch==2.10.0 CUDA build and
# would pull ~10 GB of NVIDIA libs you don't need; it also pins
# jax==0.10.0/libtpu==0.0.40 which conflict with our 0.9.2/0.0.39).
uv pip install -e ~/tpu-inference/ --no-deps
```

Headline pinned versions (full list in `requirements_tpu_keep.txt`):

| Package | Version | Why |
|---|---|---|
| `torch` | 2.11.0+cpu | torchax requires this major; +cpu because all compute goes via JAX |
| `torchax` | 0.0.11 | runtime workarounds specific to this version |
| `jax` | 0.9.2 | matches libtpu and tpu-inference |
| `jaxlib` | 0.9.2 | same |
| `libtpu` | 0.0.39 | TPU runtime |
| `transformers` | 5.8.1 | sglang's Qwen3 loader API |
| `flax` | 0.12.4 | required by tpu-inference Pallas kernels |

Verify imports:

```bash
python -c "import sglang; print(sglang.__file__)"
# /home/cuiq_google_com/cuiq-sglang/python/sglang/__init__.py

python -c "import torchax; import jax; print('jax', jax.__version__, '/ torchax', torchax.__version__)"

# tpu-inference imports require the vllm shim to be installed in sys.modules
# BEFORE the first tpu_inference import. The shim installs itself at module-
# import time, so the order below works:
python -c "
import sglang.srt.layers.attention._vllm_shim  # installs vllm.* stubs
from tpu_inference.kernels.ragged_paged_attention.v3 import kernel
print('tpu-inference kernel OK:', kernel.__file__)
"
```

---

## Step 5: Pre-flight TPU check

Confirm JAX can claim all 8 TPU chips:

```bash
python -c "import jax; devs = jax.devices(); print(f'{len(devs)} TPU chips:', devs)"
# Expected: 8 TPU chips: [TpuDevice(id=0, ...), TpuDevice(id=1, ...), ..., TpuDevice(id=7, ...)]
```

If you see `0 chips` or fewer than 8, run `~/cuiq-sglang/scripts/tpu_reset.sh` (recovers wedged TPU state via pkill, fuser, lockfile rm, `tpu-runtime` restart) and retry.

---

## Step 6: Launch the server

### Pre-flight every time

Always run `tpu_reset.sh` before launching (not just on errors). It's idempotent on a healthy system and recovers a wedged one. Skipping it is the most common cause of `Device or resource busy` on subsequent boots.

```bash
~/cuiq-sglang/scripts/tpu_reset.sh
```

### TP=1 (single-chip, ~40 s cold start, 183 tok/s decode)

```bash
cd ~/cuiq-sglang
source ~/sglang_tpu_env.sh  # idempotent re-source
source .venv/bin/activate

python -m sglang.launch_server \
    --device tpu \
    --model Qwen/Qwen3-4B \
    --tp-size 1 \
    --disable-radix-cache \
    --max-running-requests 1
```

On first launch the model weights download (~8 GB for Qwen3-4B in bf16; ~3-5 min on typical GCP bandwidth). Subsequent launches reuse cached weights.

Wait for `Application startup complete.` in the log — typically ~40 s on a warm system. The first `/generate` request will then trigger a ~30 s JIT compile (one extend bucket + one decode bucket); subsequent same-shape requests are warm and fast.

### TP=2/4/8 (multi-chip)

> ⚠️ **`--skip-server-warmup` is MANDATORY at TP>1.** Without it, sglang's default startup warmup pings `/generate` synchronously with a hard 600 s HTTP read timeout. The cold JIT compile at TP=2+ takes longer than that, so the warmup hangs and kills the server during boot.

```bash
python -m sglang.launch_server \
    --device tpu \
    --model Qwen/Qwen3-4B \
    --tp-size 4 \    # or 2 or 8
    --disable-radix-cache \
    --max-running-requests 1 \
    --skip-server-warmup
```

Cold-compile wait times (from the first `/generate` request after server start):

| TP | Cold compile (extend + decode buckets) |
|---|---|
| 1 | ~30 s |
| 2 | ~10 min |
| 4 | ~6-7 min |
| 8 | ~6-7 min |

Watch the server log for compile progress:

```
JaxStepRunner compiling JIT for ('extend', 16, 1) ...
JaxStepRunner compiling JIT for ('decode', 1, 1) ...
```

Each bucket is one JIT compile; expect ~2 lines on the first request.

> The JAX persistent compile cache does not work at TP=2+ (the JIT executable exceeds the 5 GB cache serialization limit), so cold compile is paid every boot at TP>1.

### Important flag constraints

> **DO NOT** flip `--enable-radix-cache` or set `--max-running-requests > 1` — the `JaxMHATokenToKVPool` is currently a minimum-viable stub. Both flags will fail or produce wrong output.

### Clean shutdown

`Ctrl+C` in the server terminal works, but at TP>1 sometimes leaves vfio state stuck. Always run `~/cuiq-sglang/scripts/tpu_reset.sh` before starting a fresh server.

---

## Step 7: Send an API request

In another SSH session:

```bash
curl -X POST http://localhost:30000/generate \
    -H 'Content-Type: application/json' \
    -d '{
        "text": "The capital of France is",
        "sampling_params": {
            "max_new_tokens": 20,
            "temperature": 0
        }
    }'
```

Expected output (greedy, bit-equal to HF CPU bf16 reference at TP=1/2/4/8):

```json
{
  "text": " Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain",
  "meta_info": {
    "prompt_tokens": 5,
    "completion_tokens": 20,
    "finish_reason": {"type": "length", "length": 20}
  },
  "output_ids": [12095, 13, 576, 6722, 315, 9856, 374, 19846, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 17689]
}
```

For sampling (non-greedy), set `"temperature": 0.7` and `"top_p": 0.9`. Note that non-greedy at TP>1 has not been validated for bit-equivalence with the CPU reference.

sglang also exposes `/v1/completions` and `/v1/chat/completions` for OpenAI client compatibility; both work on the TPU path.

---

## Common errors

### `Device or resource busy` on `/dev/vfio/N`

Previous run left TPU state. Run `~/cuiq-sglang/scripts/tpu_reset.sh` (needs sudo for `systemctl restart tpu-runtime` and `fuser -k -9 /dev/vfio/*`) and retry the launch. If that doesn't recover after 3 retries, reboot the VM.

### Server hangs on first request

Almost certainly cold JIT compile in progress. Tail the server log for `compiling JIT` lines. Compile can take 6-10 min at TP>=2. If the curl times out client-side, retry — the compile continues regardless and the next request hits the warm cache.

### `uv pip install` fails with "no version of torch==2.11.0+cpu"

Missing `--extra-index-url https://download.pytorch.org/whl/cpu`. See Step 4.

### `uv pip install` fails with "version not found" for `mypy-extensions` or `fsspec`

Missing `--index-strategy unsafe-best-match`. See Step 4.

### `uv pip install -e ~/tpu-inference/` fills the disk or downloads CUDA torch

Missing `--no-deps`. See Step 4. If the disk is already full, `rm -rf ~/cuiq-sglang/.venv` and restart from Step 3 with the corrected flag.

### All-zero output (`output_ids: [0, 0, ...]`)

Symptom of a torchax weight-load bug fixed in this branch. Verify you're on `cuiq-tpu-mvp` at or near `f1895ef77`.
