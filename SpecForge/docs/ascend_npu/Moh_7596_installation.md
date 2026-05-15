# SpecForge on Ascend NPU — Installation Guide (HF Backend)

> Reproducible install for SpecForge DFlash with the HF target-model backend
> on Ascend NPU (Atlas A2 / 910b / A3). Versions and the SpecForge commit
> are pinned in `requirements-ascend.txt` next to this file.
>
> If anything fails, copy the exact error and report it — see
> [Troubleshooting](#troubleshooting) for the most common failure
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
- For step 4 (sglang), public `pypi.org` reachability from `pip` is also
  needed — the Huawei PyPI mirror does not currently carry `transformers 5.x`,
  which the NPU `pyproject.toml` pins. The Huawei mirror stays primary; public
  PyPI is only used as a fallback for the missing version.

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

This step has three moving parts that must be set up before `pip install`
can succeed on a locked-down NPU dev node:

1. A Rust toolchain (sglang v0.5.9 builds a Rust gRPC component from source).
2. A cargo registry config pointing at the Huawei crates mirror.
3. A pip fallback to `pypi.org` for `transformers 5.x` (not on the Huawei
   mirror as of writing).

#### 4a. Install a Rust toolchain via conda

sglang v0.5.9 ships `rust/sglang-grpc`, which is compiled during `pip install`.
Without `rustc` on `PATH`, you'll hit
`error: can't find Rust compiler`. The standard `curl https://sh.rustup.rs | sh`
flow usually fails on corporate networks with a self-signed CA chain. The
cleanest workaround is to grab Rust from the Huawei conda mirror, into the
conda env you just activated:

```bash
conda install -c https://mirrors.huaweicloud.com/anaconda/cloud/conda-forge rust -y
rustc --version       # should print something like rustc 1.93.x
cargo --version
which rustc           # should resolve inside the conda env
```

#### 4b. Configure cargo to use the Huawei crates mirror

Cargo will try to fetch crates from `crates.io` during the build. Outbound
HTTPS to `crates.io` is typically blocked on these nodes. Point cargo at the
Huawei mirror **using the git-format index URL** — not the sparse-index URL,
which the mirror doesn't implement:

```bash
mkdir -p ~/.cargo
cat > ~/.cargo/config.toml <<'EOF'
[source.crates-io]
replace-with = "huawei"

[source.huawei]
registry = "https://mirrors.huaweicloud.com/repository/cargo/crates.io-index"

[net]
git-fetch-with-cli = true
EOF
```

> Heads-up: if you set `registry = "sparse+https://..."` here, the build
> dies with `config.json not found in registry` because the Huawei cargo
> mirror serves a git-protocol index, not a sparse one. Keep it plain.

#### 4c. Clone sglang and swap in the NPU pyproject

```bash
cd ..
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout v0.5.9

cp python/pyproject.toml python/pyproject.toml.bak
cp python/pyproject_npu.toml python/pyproject.toml

# Verify the swap landed
grep -q srt_npu python/pyproject.toml && echo "NPU pyproject active"
```

#### 4d. Install — Huawei mirror primary, public PyPI as fallback

The NPU pyproject pins `transformers==5.6.0`. The Huawei PyPI mirror currently
tops out at `transformers 4.57.6` — `transformers 5.x` exists on public PyPI
but the Huawei mirror hasn't replicated it yet. Add `pypi.org` as a fallback
index so the resolver can pull `transformers 5.x` while everything else still
comes from the (closer, faster) Huawei mirror:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.9 \
pip install -e "python[srt_npu]" \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    --trusted-host mirrors.huaweicloud.com \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org

cd ../SpecForge
```

About `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.9`: sglang uses
`vcs_versioning` (setuptools-scm style) to compute its version from `git
describe` at build time. If you transferred the sglang source between machines
via shared storage / `scp` and `.git/` didn't come along (or its tags didn't),
the build silently falls back to `0.0.0.dev0`. The env var forces the right
label. Drop it if you cloned fresh from GitHub on this machine.

After the install finishes:

```bash
pip show sglang | grep -E "Version|Editable"
# Version: 0.5.9
# Editable project location: .../sglang/python
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
transformers             : 5.6.0          (or 5.8.x — whatever pip resolved on pypi.org)
sglang                   : 0.5.9
sgl_kernel_npu path      : ['/.../site-packages/sgl_kernel_npu']
triton (from triton-ascend) : 3.x.x       (whatever pip resolved to)
yunchang.HAS_NPU         : True
yunchang.HAS_FLASH_ATTN  : False
torch.npu.is_available() : True
torch.npu.device_count() : 8              (your real NPU count)
specforge import OK
```

If all lines look correct, the environment is ready for training.

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

#### 6e. `error: can't find Rust compiler`

sglang v0.5.9 ships `rust/sglang-grpc`, a Rust extension built during pip
install. You need `rustc` on `PATH`. The `curl https://sh.rustup.rs | sh`
path usually fails on corporate networks (`SSL certificate problem: self
signed certificate in certificate chain`) — use conda instead, from the
Huawei mirror that you can already reach:

```bash
conda install -c https://mirrors.huaweicloud.com/anaconda/cloud/conda-forge rust -y
rustc --version
```

If `conda.anaconda.org` itself is unreachable
(`CondaHTTPError: HTTP 000 CONNECTION FAILED`), the Huawei conda mirror URL
above is what unblocks it — keep `-c` pointed at it explicitly rather than
relying on default channels.

#### 6f. cargo build fails with `config.json not found in registry`

Your `~/.cargo/config.toml` is using `sparse+https://...` against the Huawei
cargo mirror, which serves a git-protocol index, not sparse. Drop the
`sparse+` prefix:

```bash
cat > ~/.cargo/config.toml <<'EOF'
[source.crates-io]
replace-with = "huawei"

[source.huawei]
registry = "https://mirrors.huaweicloud.com/repository/cargo/crates.io-index"

[net]
git-fetch-with-cli = true
EOF
```

Then re-run the pip install.

#### 6g. `Could not find a version that satisfies the requirement transformers==5.6.0`

Two distinct causes, both worth checking:

1. **Conda env not actually activated** — pip is running with system Python
   3.9.x. `transformers 5.x` requires Python `>=3.10`, so even when pip can
   see the wheels on `pypi.org` they get filtered as
   `Link requires a different Python (3.9.x not in: '>=3.10.0')`. Verify:

   ```bash
   python --version       # must be Python 3.11.x
   which python pip       # both must resolve inside the conda env
   ```

   The conda env's `(specforge_npu)` prompt prefix can silently drop between
   shell sessions or `cd`s — re-activate before retrying.

2. **Huawei mirror missing transformers 5.x** — verify by listing the
   versions returned from the mirror:

   ```bash
   pip index versions transformers \
       -i https://mirrors.huaweicloud.com/repository/pypi/simple/
   # if the top version is 4.57.x, the mirror lacks 5.x
   ```

   Add `--extra-index-url https://pypi.org/simple/` to the install command,
   exactly as shown in step 4d.

#### 6h. `pip show sglang` reports `Version: 0.0.0.dev0`

Cosmetic, not functional. sglang's `pyproject.toml` uses `vcs_versioning`
(setuptools-scm style) to compute the version from `git describe` at build
time. If you copied the sglang source between machines without `.git/` (or
without the `v0.5.9` tag inside it), the build defaults to `0.0.0.dev0`.

Three fixes, pick whichever is least intrusive:

```bash
# A. Force the version label via env var at install time:
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.9 pip install -e "python[srt_npu]" ...

# B. Or just create the tag locally before installing:
cd /path/to/sglang
git tag v0.5.9
pip install -e "python[srt_npu]" ...

# C. Or accept the cosmetic label — sglang itself works fine either way.
```

Nothing in SpecForge / DFlash reads `sglang.__version__` at runtime, so C
is a valid choice unless something downstream pins the version.

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
