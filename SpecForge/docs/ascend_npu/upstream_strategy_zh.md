# SpecForge Ascend NPU —— upstream 策略与可复现安装清单

> 与 `docs/ascend_npu/` 下其他文档配套使用。本篇记录三件事：
> (1) SpecForge upstream 已有的 ROCm 支持给我们划定的"硬件接入许可等级"；
> (2) 对 `yunchang`（曾被疑似 NPU 阻塞的第三方依赖）的源码分析结论；
> (3) 钉死了 commit / 版本号的可复现安装清单。

---

## 1. 背景 —— SpecForge upstream 实际接受什么等级的非 CUDA 接入

全仓 grep `rocm | hip | amd | is_hip` 只命中**两个 PR**，**24 行代码**，**零文档**：

| PR | 文件 | 行数 | 类型 |
|---|---|---|---|
| **#259** [Bugfix] Resolve Triton OutOfResources on AMD GPUs | `specforge/core/loss.py` | +5 | 单点运行时分支：`if hasattr(torch.version, "hip"): num_warps //= 2` |
| **#275** Added requirements-rocm.txt for AMD GPU and ROCm | `requirements-rocm.txt` | +19 | 一份替代 requirements 文件 |

整个 ROCm 接入面就这些。SpecForge 上**没有** device 抽象层、**没有**
`get_device_type()` helper、**没有** ROCm 文档、**没有** ROCm example、**没有**
任何 env-var 分发逻辑。

ROCm 几乎不用动源码，是因为 PyTorch 的 ROCm wheel **在 Python API 层透明伪装成
CUDA**：HIP 下 `torch.cuda.is_available()` 仍返回 True，`tensor.cuda()` 能用，
`dist.init_process_group("nccl")` 内部走 RCCL。SpecForge 那一堆"看起来是 CUDA"
的代码在 AMD GPU 上**直接跑得起来**。

torch_npu 没有这个属性 —— `torch.cuda.*` 不会被重定向。所以为了对齐 ROCm 的接入
等级，我们用 **`torch_npu.contrib.transfer_to_npu`** 在用户态做运行时
monkey-patch，效果跟 ROCm wheel 在 C++ 层做的事一样，只是位置不同。

---

## 2. yunchang —— 看起来阻塞、实际没问题

`specforge/distributed.py:6` 的硬 import：

```python
from yunchang.globals import PROCESS_GROUP, set_seq_parallel_pg
```

最初担心：yunchang 可能拉 CUDA / flash-attn 依赖，让 NPU 装不上，迫使我们打
patch。把 `D:/work/long-context-attention`（官方仓库 `feifeibear/long-context-attention`，
PyPI 包名 `yunchang`，HEAD 在 2026-01-15 是 **`631bdfd`** —— "`[NPU][feature]NPU
support ring cp and hybrid cp.`"）的源码读完，结论相反：

1. **flash-attn 是 *optional* extra**，不是硬依赖：
   ```toml
   dependencies = ["torch>=2.3.0"]                # 唯一硬依赖
   [project.optional-dependencies]
   flash = ["flash-attn>=2.6.0"]                  # 仅 [flash] extra
   ```

2. **`globals.py` 用 try/except 做 backend 检测**，明确包含
   `HAS_NPU = (try: import torch_npu)` 这条原生 NPU 通路。所有
   `flash_attn / flashinfer / aiter / sageattention` 的 import 都被对应的
   `HAS_*` 标志 gate 住。

3. **`set_seq_parallel_pg` 是纯 `torch.distributed.new_group(...)`** ——
   完全不调 device-specific API，HCCL 后端下行为一致。

4. **所有子模块顶层 import 都 NPU-safe**：每一处 `flash_attn` 都被
   `if HAS_FLASH_ATTN:` gate；`ring_flash_attn.py` 甚至直接把顶层 flash_attn
   import 注释掉，挪到运行时按需 dispatch。

5. **yunchang 已经原生支持 NPU 上的 ring + hybrid CP**（commit `631bdfd`，
   PR #169）—— 不只是装得上，是真能用。

**结论**：`pip install yunchang` 在 NPU 上**直接装上**，不需要 extra、不需要打
patch。DFlash 在默认参数下（`sp_ulysses_size=1, sp_ring_size=1`）整个 yunchang
就是冬眠状态；`set_seq_parallel_pg(1, 1, ...)` 只建俩 size-1 ProcessGroup，跟
CUDA 上行为字节级一致。**SpecForge 源码不需要为 yunchang 做任何修改。**

这一条把原计划的五个 PR 砍掉一个（详见 §4）。

---

## 3. transfer_to_npu shim —— 最小侵入的激活模式

参照 ROCm 的接入风格（小、opt-in、单点），Ascend 这边的激活模式：

- 一个 15 行的 `sitecustomize.py`，放在 `scripts/_npu_shim/` 下
- 仅当 `SPECFORGE_DEVICE=npu` 且 shim 目录在 `PYTHONPATH` 上时才生效
- 默认行为（env var 没设、shim 目录不在 path 里）跟 upstream 字节级一致

```python
# scripts/_npu_shim/sitecustomize.py
import os
if os.environ.get("SPECFORGE_DEVICE", "").lower() == "npu":
    import torch_npu                           # 必须先 import torch_npu
    from torch_npu.contrib import transfer_to_npu
```

Python 启动时这条跑过之后，`torch.cuda.set_device`、`tensor.cuda()`、
`device="cuda"` 字符串、`dist.init_process_group("nccl")` 全部透明重定向到
NPU。`scripts/train_dflash.py` 里的 7 处 `.cuda()` / `device="cuda"` 硬编码、
`specforge/distributed.py` 里的 5 处 `torch.cuda.*` /
`init_device_mesh("cuda", ...)` 都不需要改。

shim **不能**覆盖的地方（要单独处理）：

- `torch.nn.attention.flex_attention` —— 走 Triton codegen，NPU 的 BiSheng
  Triton ABI 跟主线对不齐。规避方法：训练时传 `--attention-backend sdpa`
  （SpecForge 已有的 CLI flag，零改动）。
- 第三方库里的 CUDA C++ kernel —— 比如 flash-attn。DFlash + HF backend 这条路
  不会触达任何这种 kernel，因此和我们无关。

---

## 4. 简化后的 upstream PR 计划 —— 三个 surgical PR

把 yunchang 的疑虑解掉之后，要往 upstream 提的工作就剩三个 PR，每个都和
ROCm 那两个 PR 的体量、形状一致：

| PR | 内容 | 大小 | 风险 | 对标 |
|---|---|---|---|---|
| **PR1** | 加 `requirements-ascend.txt` | ~25 行 | 零 | PR #275 |
| **PR2** *(条件)* | 如某个 Triton kernel 在 NPU 上需要不同参数（如 `num_warps`），在 `specforge/core/loss.py` 加一个单点运行时分支 | <10 行 | 零 | PR #259 |
| **PR3** | `docs/get_started/installation.md` 加 NPU 段 + 加 `scripts/_npu_shim/sitecustomize.py` + 加一个 `examples/run_qwen3_8b_dflash_online_npu.sh` 包装脚本 | ~40 行，**全是新增文件** | 低 —— 每件都是 opt-in，不动现有文件 | 无直接对标，但形状仍然 surgical |

**明确不做**的事：

- device 抽象层 / `get_device_type()` helper —— upstream 从来没接受过这种重构
- 给 yunchang import 加 try/except —— yunchang 在 NPU 上能装上，不用打 patch
- 把 `cuda` 字符串硬替成 `npu` —— shim 已经接管
- 在 `specforge/distributed.py` 里加 env-var 分发 —— 同上

---

## 5. 可复现安装 —— 钉死的版本号

下面这套版本是写本文时的实际现状。bump 时同步改这里。

### 5.1 钉死的组件

| 组件 | 版本 | 来源 |
|---|---|---|
| **SpecForge upstream main** | `d5fb617f735db85a680876327f8173de0cb57c15` | github.com/sgl-project/SpecForge |
| **yunchang** | `0.6.4` | PyPI（2026-05 最新版） |
| **torch** | `2.9.0` | Ascend wheel index（**不是** 2.9.1，torch_npu 暂不支持） |
| **torchvision** | `0.24.0` | Ascend / PyPI |
| **torch_npu** | `2.9.0` | Ascend wheel index |
| **transformers** | `4.57.1` | PyPI（与 SpecForge + yunchang 钉的版本一致） |
| **qwen-vl-utils** | `0.0.11` | PyPI（与 SpecForge upstream 一致） |
| **datasets** | `4.8.5` | PyPI |
| **accelerate** | `1.13.0` | PyPI |
| **safetensors** | `0.7.0` | PyPI |
| **einops** | `0.8.2` | PyPI |
| **pydantic** | `2.13.3` | PyPI |
| **wandb** | `0.26.1` | PyPI |
| **tensorboard** | `2.20.0` | PyPI |
| **openai-harmony** | `0.0.8` | PyPI |
| **numpy** | `<2.0` | torch_npu 通常需要 |
| **setuptools** | `<81` | torch_npu.dynamo.torchair 还在 import `pkg_resources`（setuptools 81+ 删了） |
| **CANN** | `≥ 8.5.0` | Ascend Toolkit |

完整 requirements 文件在本文同目录下的 `requirements-ascend.txt`。

### 5.2 安装命令（HF backend、单机）

```bash
# 0. 前置：已存在 CANN ≥8.5 安装在 $CANN_HOME，已安装 conda 或 uv
source "$CANN_HOME/ascend-toolkit/set_env.sh"
[ -f "$CANN_HOME/nnal/atb/set_env.sh" ] && source "$CANN_HOME/nnal/atb/set_env.sh"

# 1. 建 conda 环境（Python 3.11）
conda create -p ./conda/specforge_npu python=3.11 -y
conda activate ./conda/specforge_npu

# 2. 拉 SpecForge 到钉死的 commit
git clone https://github.com/sgl-project/SpecForge.git
cd SpecForge
git checkout d5fb617f735db85a680876327f8173de0cb57c15

# 3. 从 Ascend 镜像装 torch / torchvision / torch_npu
pip install torch==2.9.0 torchvision==0.24.0 \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
    --trusted-host mirrors.huaweicloud.com
pip install torch_npu==2.9.0 \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
    --trusted-host mirrors.huaweicloud.com

# 4. 装其余 Python 依赖（按版本钉死）
pip install -r docs/ascend_npu/requirements-ascend.txt

# 5. 以 editable 模式装 SpecForge，**跳过它 pyproject 里的依赖解析**
pip install -e . --no-deps
```

第 5 步特意加 `--no-deps`：SpecForge 的 `pyproject.toml` 里钉了
`torch==2.9.1` 和 `sglang==0.5.9`，前者跟 NPU 选用的 2.9.0 冲突，后者跟我们
从 upstream 源码装的 commit 版冲突——不绕开 pip 会强行覆盖。

> **注**：本文档**早期版本**说过"sglang 不需要装"。这是错的——`specforge/modeling/target/eagle3_target_model.py:5`
> 顶层硬 import sglang，即便只用 HF backend 也要让它能 import。具体安装步骤
> 见 `installation_zh.md` 的步骤 4（clone upstream sgl-project/sglang、checkout
> tag `v0.5.9`、把 `pyproject_npu.toml` swap 进来、`pip install -e`）。

### 5.3 验证

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

期望输出：
- `HAS_NPU = True`
- `HAS_FLASH_ATTN = False`
- `torch.npu.is_available() = True`
- `device_count` = 本机 NPU 数

### 5.4 训练前激活 shim

```bash
export SPECFORGE_DEVICE=npu
export PYTHONPATH="$PWD/scripts/_npu_shim:${PYTHONPATH:-}"

# 然后正常 torchrun + train_dflash.py，但带上：
#   --target-model-backend hf
#   --attention-backend sdpa
# 参考 examples/run_qwen3_8b_dflash_online_npu.sh（PR3 中的包装脚本）
```

---

## 6. 一句话总结

- SpecForge upstream 接受的非 CUDA 接入是 **24 行 surgical patch** 级别。
  不要试图提"加一层 device 抽象"这种重构。
- yunchang 在 NPU 上 `pip install` 直接通、import 直接 OK，**不需要为它改任何源码**。
- `transfer_to_npu` shim + `SPECFORGE_DEVICE` env var + sdpa attention，三件套
  能让 DFlash + HF backend 在 Ascend 上跑起来，**existing-file 零改动**。
- 三个 upstream-bound PR 完全对标 ROCm 的 PR #259 / PR #275：一份 requirements
  文件、一处可选的 kernel-tweak、一组 docs + shim 文件。
- 完整可复现安装钉到 commit / 版本：SpecForge @ `d5fb617`、yunchang `0.6.4`、
  torch `2.9.0`、torch_npu `2.9.0`、transformers `4.57.1`，详见同目录的
  `requirements-ascend.txt`。
