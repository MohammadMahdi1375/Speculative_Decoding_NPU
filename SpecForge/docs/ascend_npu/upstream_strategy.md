# SpecForge Ascend NPU — Upstreaming Strategy & Reproducible Install

> Companion to the other `docs/ascend_npu/` notes. This document captures
> (1) the precedent set by SpecForge upstream's existing ROCm support, used
> as a guide for how invasive an Ascend-NPU port is allowed to be;
> (2) source-code analysis of `yunchang`, the only third-party dependency
> that initially looked like a blocker; and (3) a reproducible install
> recipe pinned to specific commits and versions.

---

## 1. Background — what SpecForge upstream accepts for non-CUDA hardware

A grep of the entire repo for `rocm | hip | amd | is_hip` turns up **two PRs**
totalling **24 lines of code** and **zero documentation**:

| PR | File | Lines | Type |
|---|---|---|---|
| **#259** ([Bugfix] Resolve Triton OutOfResources on AMD GPUs) | `specforge/core/loss.py` | +5 | Single-point runtime check: `if hasattr(torch.version, "hip"): num_warps //= 2` |
| **#275** (Added requirements-rocm.txt for AMD GPU and ROCm) | `requirements-rocm.txt` | +19 | Alternative requirements file |

That is the entire ROCm "integration" surface. SpecForge has **no** device
abstraction layer, **no** `get_device_type()` helper, **no** ROCm docs, **no**
ROCm examples, **no** env-var-based dispatch.

ROCm needed almost no source changes because the PyTorch ROCm wheel
**transparently spoofs CUDA at the Python API layer**: `torch.cuda.is_available()`
returns True under HIP, `tensor.cuda()` works, `dist.init_process_group("nccl")`
silently uses RCCL. SpecForge's CUDA-shaped code "just runs" on AMD GPUs.

torch_npu does not have this property — `torch.cuda.*` is not redirected. So
to keep parity with the ROCm precedent we use **`torch_npu.contrib.transfer_to_npu`**
as a runtime monkey-patch, achieving the same effect at user-space rather than
inside the wheel.

---

## 2. yunchang — the dependency that looked like a blocker isn't one

`specforge/distributed.py:6` does an unconditional top-level import:

```python
from yunchang.globals import PROCESS_GROUP, set_seq_parallel_pg
```

Initial worry: yunchang has CUDA / flash-attn dependencies that would prevent
install on NPU, forcing us to patch the import. Source-level inspection of
`D:/work/long-context-attention` (the official `feifeibear/long-context-attention`
repo, package `yunchang`, current head **`631bdfd`** on 2026-01-15
"`[NPU][feature]NPU support ring cp and hybrid cp.`") proves otherwise:

1. **flash-attn is an *optional* extra**, not a hard dep:
   ```toml
   dependencies = ["torch>=2.3.0"]                # only hard requirement
   [project.optional-dependencies]
   flash = ["flash-attn>=2.6.0"]                  # only with [flash]
   ```

2. **`globals.py` runs feature detection via try/except**, including a native
   `HAS_NPU = (try: import torch_npu)` path. All `flash_attn`, `flashinfer`,
   `aiter`, `sageattention` imports are gated by their respective `HAS_*` flags.

3. **`set_seq_parallel_pg` is pure `torch.distributed.new_group(...)`** — no
   device-specific calls, works under any backend including HCCL.

4. **All submodule top-level imports are NPU-safe**: every `flash_attn`
   reference is gated by `if HAS_FLASH_ATTN:`; `ring_flash_attn.py` even
   comments its top-level flash_attn import out, deferring it to runtime.

5. **yunchang already has native NPU ring + hybrid CP support** (commit
   `631bdfd`, PR #169). Not just installable on NPU — *functional*.

**Conclusion**: `pip install yunchang` works on NPU with no extras and no
patching. The dependency is dormant when DFlash runs with default args
(`sp_ulysses_size=1, sp_ring_size=1`); `set_seq_parallel_pg(1, 1, ...)` creates
two size-1 ProcessGroups via `torch.distributed.new_group`, identical to its
CUDA-side behaviour. **No source modification of SpecForge is needed for
yunchang.**

This eliminates one of the five PRs originally planned (see §4).

---

## 3. The transfer_to_npu shim — minimum-invasive activation pattern

Following the ROCm precedent (small, opt-in, surgical), the activation pattern
for Ascend is:

- A 15-line `sitecustomize.py` placed under `scripts/_npu_shim/`
- Activated only when `SPECFORGE_DEVICE=npu` and the shim dir is on `PYTHONPATH`
- Default behaviour (env var unset, dir off path) is bit-identical to upstream

```python
# scripts/_npu_shim/sitecustomize.py
import os
if os.environ.get("SPECFORGE_DEVICE", "").lower() == "npu":
    import torch_npu                           # must come before transfer_to_npu
    from torch_npu.contrib import transfer_to_npu
```

Once this runs at Python startup, `torch.cuda.set_device`, `tensor.cuda()`,
`device="cuda"` strings, and `dist.init_process_group("nccl")` all transparently
route to the NPU equivalents. The 7 `.cuda()` / `device="cuda"` hard-codes in
`scripts/train_dflash.py` and the 5 `torch.cuda.*` / `init_device_mesh("cuda", ...)`
calls in `specforge/distributed.py` need no source change.

Risks the shim does **not** cover (require explicit handling):

- `torch.nn.attention.flex_attention` — codegens through Triton, the NPU
  Triton (BiSheng) ABI is not perfectly aligned. Workaround: pass
  `--attention-backend sdpa` (already a SpecForge CLI flag, no diff).
- Custom CUDA C++ kernels in third-party packages — e.g. `flash-attn`. None
  are reachable from the DFlash + HF backend code path, so this is moot.

---

## 4. Reduced upstream PR plan — three small surgical PRs

With the yunchang issue eliminated, the upstream-bound work for SpecForge
shrinks to three PRs that mirror the ROCm precedent in size and shape:

| PR | Content | Size | Risk | Mirrors |
|---|---|---|---|---|
| **PR1** | Add `requirements-ascend.txt` | ~25 lines | zero | PR #275 |
| **PR2** *(conditional)* | If a Triton kernel needs an NPU-specific tweak (e.g. `num_warps`), add a single-point runtime branch in `specforge/core/loss.py` | <10 lines | zero | PR #259 |
| **PR3** | Add NPU section to `docs/get_started/installation.md` + ship the `scripts/_npu_shim/sitecustomize.py` + a `examples/run_qwen3_8b_dflash_online_npu.sh` wrapper | ~40 lines, all new files | low | none — but each piece is opt-in and doesn't touch existing files |

Things explicitly **not** in the plan:

- A device-abstraction layer / `get_device_type()` helper. Upstream has never
  accepted such a refactor.
- A `try/except` around the yunchang import. yunchang `pip install`s on NPU,
  so the dependency is satisfied.
- Hard replacement of `cuda` → `npu` strings. The shim handles them.
- A new env-var-based device dispatch in `specforge/distributed.py`. Same
  reason — shim handles it.

---

## 5. Reproducible install — pinned versions

These pins are what was on the wire at the time of this document. Re-pin
when bumping.

### 5.1 Pinned components

| Component | Pin | Where it comes from |
|---|---|---|
| **SpecForge (upstream main)** | `d5fb617f735db85a680876327f8173de0cb57c15` | github.com/sgl-project/SpecForge |
| **yunchang** | `0.6.4` | PyPI (latest as of 2026-05) |
| **torch** | `2.9.0` | Ascend wheel index (NOT 2.9.1 — torch_npu unsupported) |
| **torchvision** | `0.24.0` | Ascend / PyPI |
| **torch_npu** | `2.9.0` | Ascend wheel index |
| **transformers** | `4.57.1` | PyPI (matches SpecForge + yunchang pins) |
| **qwen-vl-utils** | `0.0.11` | PyPI (matches SpecForge upstream) |
| **datasets** | `4.8.5` | PyPI |
| **accelerate** | `1.13.0` | PyPI |
| **safetensors** | `0.7.0` | PyPI |
| **einops** | `0.8.2` | PyPI |
| **pydantic** | `2.13.3` | PyPI |
| **wandb** | `0.26.1` | PyPI |
| **tensorboard** | `2.20.0` | PyPI |
| **openai-harmony** | `0.0.8` | PyPI |
| **numpy** | `<2.0` | torch_npu often clamps |
| **setuptools** | `<81` | required by torch_npu.dynamo.torchair (still imports `pkg_resources`) |
| **CANN** | `≥ 8.5.0` | Ascend Toolkit |

The full file is in `requirements-ascend.txt` next to this document.

### 5.2 Install commands (HF backend, single node)

```bash
# 0. Prerequisites: an existing CANN ≥8.5 install at $CANN_HOME, and conda/uv
source "$CANN_HOME/ascend-toolkit/set_env.sh"
[ -f "$CANN_HOME/nnal/atb/set_env.sh" ] && source "$CANN_HOME/nnal/atb/set_env.sh"

# 1. Conda env (Python 3.11)
conda create -p ./conda/specforge_npu python=3.11 -y
conda activate ./conda/specforge_npu

# 2. Clone SpecForge at the pinned commit
git clone https://github.com/sgl-project/SpecForge.git
cd SpecForge
git checkout d5fb617f735db85a680876327f8173de0cb57c15

# 3. Install torch / torchvision / torch_npu from Ascend's index
pip install torch==2.9.0 torchvision==0.24.0 \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
    --trusted-host mirrors.huaweicloud.com
pip install torch_npu==2.9.0 \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
    --trusted-host mirrors.huaweicloud.com

# 4. Install all other Python deps from the pinned requirements file
pip install -r docs/ascend_npu/requirements-ascend.txt

# 5. Editable-install SpecForge WITHOUT re-resolving its pyproject pins
pip install -e . --no-deps
```

`--no-deps` in step 5 is deliberate: SpecForge's `pyproject.toml` pins
`torch==2.9.1` and `sglang==0.5.9`. The torch pin conflicts with the
NPU-supported 2.9.0; the sglang pin conflicts with the from-source upstream
commit we install. Without `--no-deps`, pip would happily overwrite both.

> **Correction note**: an earlier version of this document said "sglang is
> HF-backend irrelevant and need not be installed." That was wrong —
> `specforge/modeling/target/eagle3_target_model.py:5` is an unconditional
> top-level `import sglang.srt.managers.mm_utils`, which gets triggered by
> any `import specforge`, regardless of the runtime backend choice. The
> correct install path is in `installation.md` Step 4 (clone upstream
> sgl-project/sglang at tag `v0.5.9`, swap in `pyproject_npu.toml`,
> `pip install -e`).

### 5.3 Verification

```bash
python - <<'PY'
import torch, torch_npu
from yunchang.globals import PROCESS_GROUP, set_seq_parallel_pg, HAS_FLASH_ATTN, HAS_NPU
import transformers
import specforge

print("torch         :", torch.__version__)
print("torch_npu     :", torch_npu.__version__)
print("transformers  :", transformers.__version__)
print("yunchang.HAS_NPU         :", HAS_NPU)
print("yunchang.HAS_FLASH_ATTN  :", HAS_FLASH_ATTN)
print("torch.npu.is_available() :", torch.npu.is_available())
print("torch.npu.device_count() :", torch.npu.device_count())
print("specforge import OK")
PY
```

Expected:
- `HAS_NPU = True`
- `HAS_FLASH_ATTN = False`
- `torch.npu.is_available() = True`
- `device_count` = number of NPUs on the host

### 5.4 Activating the shim before training

```bash
export SPECFORGE_DEVICE=npu
export PYTHONPATH="$PWD/scripts/_npu_shim:${PYTHONPATH:-}"

# Then your normal torchrun + train_dflash.py invocation, but pass
#   --target-model-backend hf
#   --attention-backend sdpa
# See examples/run_qwen3_8b_dflash_online_npu.sh (PR3).
```

---

## 6. TL;DR

- Upstream SpecForge accepts non-CUDA hardware integration in **24-line
  surgical patches**. Don't try to land a device-abstraction refactor.
- yunchang `pip install`s and imports cleanly on NPU; **no source change is
  needed for it**.
- `transfer_to_npu` shim + `SPECFORGE_DEVICE` env var + sdpa attention give
  you a working DFlash + HF backend on Ascend with **zero modifications to
  existing SpecForge files**.
- Three upstream-bound PRs mirror the ROCm precedent (PR #259, PR #275): a
  requirements file, an optional one-line kernel tweak, an installation-doc
  + shim addition.
- The full reproducible install is pinned: SpecForge @ `d5fb617`, yunchang
  `0.6.4`, torch `2.9.0`, torch_npu `2.9.0`, transformers `4.57.1`. See
  `requirements-ascend.txt`.
