# SpecForge on Ascend NPU — Installation Guide (HF Backend)

> Reproducible install for SpecForge DFlash with the HF target-model backend
> on Ascend NPU (Atlas A2 / 910b / A3). Versions and the SpecForge commit
> are pinned in `requirements-ascend.txt` next to this file.
>
> If anything fails, copy the exact error and report it — see
> [Troubleshooting](#troubleshooting) for the five most common failure
> modes and what to try.

---

## What you get

After this install:
- SpecForge pinned to upstream commit `d5fb617f735db85a680876327f8173de0cb57c15`
- DFlash + HF target-model backend (no sglang, no NPU kernel build)
- yunchang `0.6.4` installed but dormant (DFlash default args do not exercise it)
- Ready to launch `scripts/train_dflash.py --target-model-backend hf --attention-backend sdpa`

What this install does **not** include (deliberately):
- `sglang` — only needed for `--target-model-backend sglang`
- `flash-attn` — has no NPU build
- `triton-ascend`, `BiSheng`, `sgl-kernel-npu`, `DeepEP` — only needed for the
  sglang backend's NPU kernels

---

## Prerequisites

- An existing **CANN ≥ 8.5.0** install on the host. Note its path; you will
  pass it as `CANN_HOME`.
- **Conda** (Miniconda/Anaconda) on `$PATH`.
- Network access to the Huawei Cloud PyPI mirror
  (`mirrors.huaweicloud.com`) and to GitHub (`github.com`).
- Python 3.11 will be installed by step 1; you do not need it system-wide.

---

## Install steps

Each step is independent. If one fails, fix it and re-run that step only.
The exact pinned versions referenced below are in
`docs/ascend_npu/requirements-ascend.txt`, which lives in the same branch
you'll clone in step 2.

### Step 0 — Source CANN

```bash
export CANN_HOME=/path/to/your/CANN/8.5.0.x       # ← edit this to your real CANN path
source "$CANN_HOME/ascend-toolkit/set_env.sh"
[ -f "$CANN_HOME/nnal/asdsip/set_env.sh" ] && source "$CANN_HOME/nnal/asdsip/set_env.sh"
[ -f "$CANN_HOME/nnal/atb/set_env.sh" ]    && source "$CANN_HOME/nnal/atb/set_env.sh"
```

### Step 1 — Create a conda environment (Python 3.11)

```bash
conda create -p ./conda/specforge_npu python=3.11 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ./conda/specforge_npu
```

### Step 2 — Clone the fork's `docs/ascend-npu` branch

This branch contains:
- SpecForge source at the pinned upstream commit `d5fb617`
- `docs/ascend_npu/` with this guide and `requirements-ascend.txt`

```bash
git clone -b docs/ascend-npu https://github.com/Sawyer117/SpecForge.git
cd SpecForge
```

### Step 3 — Install all Python deps in one go (incl. torch / torch_npu)

```bash
pip install -r docs/ascend_npu/requirements-ascend.txt
```

The header of `requirements-ascend.txt` declares:

```
--index-url https://mirrors.huaweicloud.com/repository/pypi/simple/
--trusted-host mirrors.huaweicloud.com
```

so **every** package — torch, torchvision, torch_npu, and the rest — is pulled
from the Huawei Cloud mirror. The mirror is a full PyPI replica AND hosts the
Ascend `torch_npu` wheels, so a single `pip install -r` does the whole job.

> If `torch_npu==2.9.0` is not on the mirror, see
> [Troubleshooting #1](#1-torch_npu290-not-found-on-the-mirror).

### Step 4 — Install sglang (required even for HF backend)

> sglang is a hard import-time dep of SpecForge (top-level
> `import sglang.srt.managers.mm_utils` in `eagle3_target_model.py:5`).
> We pin to upstream **`v0.5.9`** — the same version SpecForge upstream's
> `pyproject.toml` declares, and the earliest tag that ships
> `pyproject_npu.toml`. Background: see `upstream_strategy.md`.

```bash
cd ..
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout v0.5.9

cp python/pyproject.toml python/pyproject.toml.bak
cp python/pyproject_npu.toml python/pyproject.toml

pip install -e "python[srt_npu]" \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
    --trusted-host mirrors.huaweicloud.com

cd ../SpecForge
```

Any failure → see [Troubleshooting #6](#6-sglang-install-fails).

### Step 5 — Install NPU kernels (`sgl_kernel_npu` + `triton-ascend`)

> sglang's import path touches `sgl_kernel_npu` (the NPU operator library),
> which in turn needs `triton-ascend`. The former builds from source; the
> latter is one pip line. We do **not** build DeepEP (only needed by
> sglang's MoE expert-parallel path, which the HF backend never reaches).
>
> About the source: upstream `sgl-project/sgl-kernel-npu`'s `build.sh` has a
> bug on multi-user NPU hosts — it ignores any pre-set `ASCEND_HOME_PATH`
> and instead reads `/etc/Ascend/ascend_cann_install.info`, which often
> points to another user's CANN install
> ([PR #460](https://github.com/sgl-project/sgl-kernel-npu/pull/460) submitted).
> Until it merges upstream, install from the fork branch `npu-install-stable`
> — it is upstream stable tag `2026.03.01.post1` with the PR cherry-picked.

```bash
pip install triton-ascend \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
    --trusted-host mirrors.huaweicloud.com

cd ..
git clone -b npu-install-stable https://github.com/Sawyer117/sgl-kernel-npu.git
cd sgl-kernel-npu

bash build.sh -a kernels
pip install output/sgl_kernel_npu*.whl

cd ../SpecForge
```

Any failure → see [Troubleshooting #7](#7-sgl_kernel_npu-or-triton-ascend-install-fails).

### Step 6 — Editable-install SpecForge **without** re-resolving deps

```bash
pip install -e . --no-deps
```

`--no-deps` is **required**. Without it, pip would try to satisfy SpecForge's
own `pyproject.toml` pins (`torch==2.9.1`, `sglang==0.5.9`), both of which
conflict with what you installed in steps 3 and 4 — and pip would happily
overwrite them.

### Step 7 — Verify

```bash
python - <<'PY'
import torch, torch_npu
from yunchang.globals import PROCESS_GROUP, set_seq_parallel_pg, HAS_FLASH_ATTN, HAS_NPU
import transformers
import sglang
import sgl_kernel_npu
import triton                       # triton-ascend registers itself as 'triton'
import specforge

print("torch                    :", torch.__version__)
print("torch_npu                :", torch_npu.__version__)
print("transformers             :", transformers.__version__)
print("sglang                   :", sglang.__version__)
print("sgl_kernel_npu path      :", sgl_kernel_npu.__path__)
print("triton (from triton-ascend) :", triton.__version__)
print("yunchang.HAS_NPU         :", HAS_NPU)
print("yunchang.HAS_FLASH_ATTN  :", HAS_FLASH_ATTN)
print("torch.npu.is_available() :", torch.npu.is_available())
print("torch.npu.device_count() :", torch.npu.device_count())
print("specforge import OK")
PY
```

> **Note**: the `triton` module name is shared between upstream NVIDIA
> `triton` and NPU `triton-ascend` — `import triton` alone cannot tell you
> which provider you got. On an NPU box where only `triton-ascend` was
> installed (and upstream `triton` was not), `import triton` is unambiguously
> the NPU build. If you want to be absolutely sure:
>
> ```bash
> pip show triton-ascend | grep -E '^(Name|Version):'
> # Expected: Name: triton-ascend / Version: 3.x.x
> python -c "import triton; print(triton.__file__)"
> # Expected path should contain 'triton_ascend' or 'triton/backends/ascend'
> ```

### Expected output

```
torch                    : 2.9.0
torch_npu                : 2.9.0          (or 2.9.0.postN, exact value depends on the mirror)
transformers             : 4.57.1
sglang                   : 0.5.9
sgl_kernel_npu path      : ['/.../site-packages/sgl_kernel_npu']
triton (from triton-ascend) : 3.x.x       (whatever pip resolved to)
yunchang.HAS_NPU         : True
yunchang.HAS_FLASH_ATTN  : False
torch.npu.is_available() : True
torch.npu.device_count() : 8              (your real NPU count)
specforge import OK
```

If all six lines look correct, the environment is ready for training.

---

## Troubleshooting

### 1. `torch_npu==2.9.0` not found on the mirror

Try without a strict pin first:

```bash
pip install torch_npu \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
    --trusted-host mirrors.huaweicloud.com
```

Note the actual version pip picked (e.g. `torch_npu-2.9.0.post1`) and update
`requirements-ascend.txt` accordingly. If even that fails, the wheel is not
on the public Huawei Cloud mirror — ask your CANN team for the
release-channel URL or grab the wheel from
<https://www.hiascend.com/document/redirect/CannCommercialDeveloperResource>.

### 2. `numpy<2.0` conflicts with another package

If pip reports a resolver conflict on `numpy`, replace the loose pin with a
concrete version:

```bash
pip install numpy==1.26.4
```

Then re-run step 4 with `numpy` already pinned.

### 3. `setuptools<81` conflict

Same idea — pin to a concrete version:

```bash
pip install "setuptools==80.9.0"
```

`setuptools<81` is required because `torch_npu.dynamo.torchair` still imports
`pkg_resources`, which setuptools 81+ removed.

### 4. yunchang accidentally pulls flash-attn

Should not happen with `pip install yunchang==0.6.4` (no `[flash]` extra),
but if it does, install yunchang explicitly without extras:

```bash
pip install --no-deps yunchang==0.6.4
pip install "torch>=2.3.0"   # the only real yunchang dep
```

### 5. `from yunchang.globals import ...` raises `ImportError`

Capture the full traceback. Common cause is yunchang's `__init__.py` running
`from .ring import *`, which transitively imports a submodule that needs
something missing on the system. Report the traceback — yunchang 0.6.4 is
expected to be NPU-clean, so any failure here is genuinely interesting.

### 6. sglang install fails

#### 6a. `git checkout v0.5.9` reports `unknown revision`

`git clone` fetches all tags by default, so this should not normally happen.
If your git config is unusual:

```bash
git fetch --tags
git checkout v0.5.9
```

#### 6b. `ls python/pyproject*.toml` does not show `pyproject_npu.toml`

The checkout is not v0.5.9. Re-do it:

```bash
git fetch --tags
git checkout v0.5.9
ls python/pyproject*.toml    # expect 5 files: pyproject / cpu / npu / other / xpu
```

If still wrong, fall back to PyPI wheel install (no source build, but enough
to satisfy the import):

```bash
pip install sglang==0.5.9 --no-deps \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
    --trusted-host mirrors.huaweicloud.com

# Then resolve missing imports one-by-one
python -c "import sglang.srt.managers.mm_utils"
```

#### 6c. `pip install -e python` stalls on a GPU-only dep building from source

Typical offenders: `flashinfer-python`, `sgl-kernel`, `vllm-flash-attn`. None
have aarch64 wheels and they try to compile CUDA C++ from source. **Skip dep
resolution with `--no-deps`**:

```bash
pip install -e "python[srt_npu]" --no-deps \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
    --trusted-host mirrors.huaweicloud.com

# Then run import to discover which deps are actually needed at import time:
python -c "import sglang.srt.managers.mm_utils"
# If you get ModuleNotFoundError: 'X', pip install X
```

Most-likely missing: `compressed-tensors`, `xgrammar`, `uvloop`, `uvicorn`,
`fastapi`, `msgspec`, `partial_json_parser`, `outlines`, `interegular`,
`llguidance`, `anthropic`, `prometheus-client`, `pyzmq`, `setproctitle`,
`tiktoken`, `timm`, `smg-grpc-proto`, `hf_transfer`, `av`, `decord2`,
`soundfile`, `grpcio`. Most of these have aarch64 wheels.
`flashinfer-python` / `sgl-kernel` / `vllm` are GPU-only — **do not install
them**; they are not reached during sglang's import phase.

#### 6d. `import sglang` fails with `cannot find libcudart.so` or other CUDA errors

v0.5.9 eager-loads CUDA at import time. Try a slightly older release like
`v0.5.8` (less NPU support but cleaner import path):

```bash
cd ../sglang
git checkout v0.5.8
ls python/pyproject_npu.toml || echo "this tag has no NPU pyproject; try v0.5.9 then PyPI fallback"
```

Or fall back to PyPI's `sglang==0.5.4` (the version pinned in SpecForge's
`requirements-rocm.txt`):

```bash
pip install sglang==0.5.4 --no-deps -i ...
```

### 7. `sgl_kernel_npu` or `triton-ascend` install fails

#### 7a. `pip install triton-ascend` not found on the Huawei mirror

Mirrors occasionally lag — retry a few minutes later, or switch sources:

```bash
pip install triton-ascend
# Or hit the public PyPI index explicitly:
pip install triton-ascend -i https://pypi.org/simple/
```

Note which version pip picked (`pip show triton-ascend | grep Version`).
You only need to pin if a future kernel-compatibility issue forces it.

#### 7b. `git checkout 2026.03.01.post1` reports `unknown revision`

```bash
git fetch --tags
git checkout 2026.03.01.post1
```

#### 7c. `bash build.sh -a kernels` fails

First confirm CANN is sourced (Step 0). The build script depends on paths
exposed by `set_env.sh`.

```bash
which msopgen          # should print a path inside CANN's toolkit
echo $ASCEND_HOME_PATH # should be non-empty
```

If CANN is sourced and the build still fails, paste the last 30 lines of
build output.

#### 7d. `pip install output/sgl_kernel_npu*.whl` complains about missing deps

`sgl_kernel_npu` needs a few small Python libs at runtime (`pybind11` etc.).
If pip's resolver complains, install them individually:

```bash
pip install pybind11
```

#### 7e. Should I install DeepEP later?

DFlash + HF backend does **not** need DeepEP (the HF backend never calls
MoE expert-parallel comm). If you later switch to the sglang backend with
an MoE target, come back and build it:

```bash
cd ../sgl-kernel-npu
# A2 / 910b
bash build.sh -a deepep2
# A3
bash build.sh -a deepep
pip install output/deep_ep*.whl
```

---

## Next steps after install

1. Source the ATB runtime once per shell (already done in step 0 if you ran
   the full block).
2. Activate the `transfer_to_npu` shim and launch training. The shim and a
   ready-made launch script will land in PR3 of the upstream-strategy plan;
   until then, see `docs/ascend_npu/upstream_strategy.md` §5.4 for the
   minimal activation pattern.
3. Single-node multi-NPU training: `examples/run_qwen3_8b_dflash_online.sh`
   with `--target-model-backend hf --attention-backend sdpa`. For multi-node
   see `docs/ascend_npu/multi_node_training.md`.
