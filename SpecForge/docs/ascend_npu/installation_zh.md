# SpecForge on Ascend NPU —— 安装指南（HF backend）

> 在 Ascend NPU（Atlas A2 / 910b / A3）上跑 SpecForge DFlash + HF target backend
> 的可复现安装步骤。版本号和 SpecForge commit 都钉死在同目录的
> `requirements-ascend.txt` 里。
>
> 任何一步失败，把原始报错贴出来——见文末
> [常见问题排查](#常见问题排查)，列了五种最常见的卡点和应对。

---

## 装完你能拿到什么

- SpecForge 钉到 upstream commit `d5fb617f735db85a680876327f8173de0cb57c15`
- DFlash + HF target-model backend（不装 sglang，不 build NPU kernel）
- yunchang `0.6.4` 装上但冬眠（DFlash 默认参数下不会调用它）
- 可以直接拉起 `scripts/train_dflash.py --target-model-backend hf --attention-backend sdpa`

**故意不包含**的东西：
- `sglang` —— 仅 `--target-model-backend sglang` 需要
- `flash-attn` —— NPU 上没 build
- `triton-ascend` / `BiSheng` / `sgl-kernel-npu` / `DeepEP` —— 仅 sglang backend
  的 NPU kernel 需要

---

## 前置条件

- 机器上已有 **CANN ≥ 8.5.0** 安装。记下安装路径，待会儿当 `CANN_HOME` 用。
- **conda** (Miniconda/Anaconda) 已经在 `$PATH` 里。
- 能访问华为云 PyPI 镜像（`mirrors.huaweicloud.com`）和 GitHub
  （`github.com`）。
- Python 3.11 由步骤 1 装到 conda 环境里，不需要系统级。

---

## 安装步骤

每一步独立，单步失败修完那一步重跑即可。下面引用的所有钉死版本号都在
`docs/ascend_npu/requirements-ascend.txt` 里——这个文件随步骤 2 clone
下来就有。

### 步骤 0 —— Source CANN

```bash
export CANN_HOME=/path/to/your/CANN/8.5.0.x       # ← 改成你机器的实际 CANN 路径
source "$CANN_HOME/ascend-toolkit/set_env.sh"
[ -f "$CANN_HOME/nnal/asdsip/set_env.sh" ] && source "$CANN_HOME/nnal/asdsip/set_env.sh"
[ -f "$CANN_HOME/nnal/atb/set_env.sh" ]    && source "$CANN_HOME/nnal/atb/set_env.sh"
```

### 步骤 1 —— 建 conda 环境（Python 3.11）

```bash
conda create -p ./conda/specforge_npu python=3.11 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ./conda/specforge_npu
```

### 步骤 2 —— 拉 fork 的 `docs/ascend-npu` 分支

这个分支带：
- SpecForge 源码已经在钉死的 upstream commit `d5fb617`
- `docs/ascend_npu/` 下有本指南和 `requirements-ascend.txt`

```bash
git clone -b docs/ascend-npu https://github.com/Sawyer117/SpecForge.git
cd SpecForge
```

### 步骤 3 —— 一次装齐所有 Python 依赖（含 torch / torch_npu）

```bash
pip install -r docs/ascend_npu/requirements-ascend.txt
```

`requirements-ascend.txt` 头部已经声明：

```
--index-url https://mirrors.huaweicloud.com/repository/pypi/simple/
--trusted-host mirrors.huaweicloud.com
```

所以 torch、torchvision、torch_npu 以及其他所有包**都从华为云镜像拉**——
镜像上既有 PyPI 完整副本，也有 Ascend 的 `torch_npu` wheel，**一条命令搞定**。

> 如果镜像上没有 `torch_npu==2.9.0`，看
> [常见问题 #1](#1-torch_npu290-镜像上找不到)。

### 步骤 4 —— 装 sglang（即使只用 HF backend 也必须装）

> sglang 是 SpecForge import 阶段的硬依赖（`eagle3_target_model.py:5`
> 顶层硬 import）。我们从 upstream sgl-project/sglang 的 **`v0.5.9`** tag 装
> ——这正好是 SpecForge upstream `pyproject.toml` 钉的版本，也是最早带
> `pyproject_npu.toml` 的 release tag。背景见 `upstream_strategy_zh.md`。

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

任何一行报错见 [常见问题 #6](#6-sglang-装不上)。

### 步骤 5 —— 装 NPU kernels（`sgl_kernel_npu` + `triton-ascend`）

> sglang import 阶段会触达 `sgl_kernel_npu`（NPU 算子库），它又依赖
> `triton-ascend`。前者从源码 build，后者一条 pip 即可。
> **不**装 DeepEP（HF backend 不触达）。
>
> 关于源码出处：upstream `sgl-project/sgl-kernel-npu` 的 `build.sh` 在多用户
> NPU 主机上有个 bug —— 它无视已 export 的 `ASCEND_HOME_PATH`，直接读
> `/etc/Ascend/ascend_cann_install.info`，会用错另一个用户的 CANN 路径
> （[issue #460](https://github.com/sgl-project/sgl-kernel-npu/pull/460) 已提 PR）。
> 在 PR 合入 upstream 之前，我们从 fork 的 `npu-install-stable` 分支装——它
> 是 upstream stable tag `2026.03.01.post1` + 上述 PR 的 cherry-pick。

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

任何一行报错见 [常见问题 #7](#7-sgl_kernel_npu-或-triton-ascend-装不上)。

### 步骤 6 —— editable 安装 SpecForge，**绕过它的 pyproject 依赖解析**

```bash
pip install -e . --no-deps
```

`--no-deps` **必须有**。否则 pip 会按 SpecForge 自己的 `pyproject.toml` 去满足
`torch==2.9.1`、`sglang==0.5.9` 这两条——前者跟你刚装的 2.9.0 冲突、后者跟你
刚从源码装的 commit 版冲突，两条都会被 pip 强行覆盖装。

### 步骤 7 —— 验证

```bash
python - <<'PY'
import torch, torch_npu
from yunchang.globals import PROCESS_GROUP, set_seq_parallel_pg, HAS_FLASH_ATTN, HAS_NPU
import transformers
import sglang
import sgl_kernel_npu
import triton                       # ← triton-ascend 注册的就是 'triton' 命名空间
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

> **注意**：`triton` 这个 module 名同时被 upstream NVIDIA `triton` 和 NPU
> `triton-ascend` 占用——`import triton` 不能告诉你 import 的是哪一份。
> 在 NPU 机器上单独装了 `triton-ascend`、且没装上游 `triton`，那 `import triton`
> 拿到的就是 NPU 版。如果想完全确认是 triton-ascend：
>
> ```bash
> pip show triton-ascend | grep -E '^(Name|Version):'
> # 期望: Name: triton-ascend / Version: 3.x.x
> python -c "import triton; print(triton.__file__)"
> # 期望路径里能看到 'triton_ascend' 或 'triton/backends/ascend' 字样
> ```

### 期望输出

```
torch                    : 2.9.0
torch_npu                : 2.9.0          (或 2.9.0.postN，看镜像实际有什么)
transformers             : 4.57.1
sglang                   : 0.5.9
sgl_kernel_npu path      : ['/.../site-packages/sgl_kernel_npu']
triton (from triton-ascend) : 3.x.x       (具体版本看 pip 装到的)
yunchang.HAS_NPU         : True
yunchang.HAS_FLASH_ATTN  : False
torch.npu.is_available() : True
torch.npu.device_count() : 8              (你机器实际 NPU 数)
specforge import OK
```

各行都对的话，环境就绪，可以开始训练。

---

## 常见问题排查

### 1. `torch_npu==2.9.0` 镜像上找不到

先放宽版本约束试试：

```bash
pip install torch_npu \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
    --trusted-host mirrors.huaweicloud.com
```

记下 pip 实际选了哪个版本（比如 `torch_npu-2.9.0.post1`），同步更新
`requirements-ascend.txt`。如果连这条也失败，说明这个 wheel 不在公开镜像上
——找你 CANN 那边要内部 release 通道的 URL，或者去
<https://www.hiascend.com/document/redirect/CannCommercialDeveloperResource>
直接拿 wheel。

### 2. `numpy<2.0` 跟其他包冲突

pip resolver 报 `numpy` 冲突的话，把宽松约束换成具体版本：

```bash
pip install numpy==1.26.4
```

然后再跑步骤 4，这时候 numpy 已经钉好了。

### 3. `setuptools<81` 冲突

同样思路，钉到具体版本：

```bash
pip install "setuptools==80.9.0"
```

之所以要 `setuptools<81`，是因为 `torch_npu.dynamo.torchair` 还在 import
`pkg_resources`，setuptools 81+ 把它删了。

### 4. yunchang 不小心把 flash-attn 拉进来

如果你装的是 `pip install yunchang==0.6.4`（没带 `[flash]` extra），不该发生。
万一发生了，明确不带 deps 装：

```bash
pip install --no-deps yunchang==0.6.4
pip install "torch>=2.3.0"   # 这是 yunchang 唯一真正的硬依赖
```

### 5. `from yunchang.globals import ...` 报 `ImportError`

把完整 traceback 抓出来。常见原因是 yunchang 的 `__init__.py` 跑
`from .ring import *`，传递触发了某个子模块需要一个本机不存在的包。报错原文给我
——yunchang 0.6.4 理论上 NPU 干净，这一步出问题非常值得追。

### 6. sglang 装不上

#### 6a. `git checkout v0.5.9` 报 `unknown revision`

git clone 默认带全 tag，正常情况下这条不会失败。如果你机器上 git 配置很不寻常导致没 fetch tag：

```bash
git fetch --tags
git checkout v0.5.9
```

#### 6b. `ls python/pyproject*.toml` 没看到 `pyproject_npu.toml`

说明你机器上 checkout 的不是 v0.5.9。重新 checkout：

```bash
git fetch --tags
git checkout v0.5.9
ls python/pyproject*.toml    # 期望 5 份: pyproject / cpu / npu / other / xpu
```

如果还是不对，退到 PyPI wheel 装法（没源码 build 的麻烦，但足够 import）：

```bash
pip install sglang==0.5.9 --no-deps \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
    --trusted-host mirrors.huaweicloud.com

# 然后根据 import 报错逐个补缺包
python -c "import sglang.srt.managers.mm_utils"
```

#### 6c. `pip install -e python` 卡在某条 GPU-only 依赖编译失败

典型出问题的：`flashinfer-python`、`sgl-kernel`、`vllm-flash-attn`。这些在 aarch64
没有 wheel，会试图编译 CUDA C++，缺 nvcc 就死。**用 `--no-deps` 跳过依赖解析**：

```bash
pip install -e "python[srt_npu]" --no-deps \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
    --trusted-host mirrors.huaweicloud.com

# 然后用 import 验证哪些 dep 是真的 import-time 必需的，逐个补：
python -c "import sglang.srt.managers.mm_utils"
# 若报 ModuleNotFoundError: 'X'，pip install X 即可
```

最常见会缺：`compressed-tensors` / `xgrammar` / `uvloop` / `uvicorn` / `fastapi`
/ `msgspec` / `partial_json_parser` / `outlines` / `interegular` / `llguidance`
/ `anthropic` / `prometheus-client` / `pyzmq` / `setproctitle` / `tiktoken` /
`timm` / `smg-grpc-proto` / `hf_transfer` / `av` / `decord2` / `soundfile` /
`grpcio`。这些大多有 aarch64 wheel，能装。`flashinfer-python` /
`sgl-kernel` / `vllm` 这种 GPU-only 的——**别装**，它们在 sglang import 阶段
不会触达。

#### 6d. `import sglang` 时报 `cannot find libcudart.so` 或 CUDA 类错误

说明 v0.5.9 在 import 阶段就 eager-load 了 CUDA 库。退到稍老的 release，比如
`v0.5.8`（NPU 支持稍弱但 import 路径更干净），重做 checkout：

```bash
cd ../sglang
git checkout v0.5.8
ls python/pyproject_npu.toml || echo "this tag has no NPU pyproject; try v0.5.9 then PyPI fallback"
```

或直接用 PyPI 老版本：

```bash
pip install sglang==0.5.4 --no-deps -i ...
```

### 7. `sgl_kernel_npu` 或 `triton-ascend` 装不上

#### 7a. `pip install triton-ascend` 在镜像上找不到

镜像偶尔索引有滞后，过几分钟重试或换镜像：

```bash
pip install triton-ascend
# 或显式走 PyPI 源：
pip install triton-ascend -i https://pypi.org/simple/
```

记下 pip 实际选了哪个版本（`pip show triton-ascend | grep Version`），
后续如果训练触发 kernel 不兼容再考虑钉版本。

#### 7b. `git checkout 2026.03.01.post1` 报 `unknown revision`

这个 tag 上面我已经在 upstream 验过存在。如果你机器找不到：

```bash
git fetch --tags
git checkout 2026.03.01.post1
```

#### 7c. `bash build.sh -a kernels` 失败

先确认 CANN 已 source（步骤 0 那一坨），尤其 `set_env.sh`。
build 脚本依赖 `ascend-toolkit` 暴露出来的 `nnal/atb` 等路径。

```bash
which msopgen   # 应该输出 CANN 工具链里的某个路径
echo $ASCEND_HOME_PATH    # 应该非空
```

如果 CANN 已 source 还是失败，把 build 输出的最后 30 行原文给我。

#### 7d. `pip install output/sgl_kernel_npu*.whl` 报缺包

`sgl_kernel_npu` 运行时还会需要一些 Python 库（pybind11 之类），
如果 pip resolver 抱怨，按 require 单独装即可：

```bash
pip install pybind11
```

#### 7e. 后续要不要装 DeepEP

DFlash + HF backend **不需要 DeepEP**（HF backend 不调用 MoE 通信）。
如果将来切到 sglang backend 跑 MoE 模型，再回来 build：

```bash
cd ../sgl-kernel-npu
# A2 / 910b
bash build.sh -a deepep2
# A3
bash build.sh -a deepep
pip install output/deep_ep*.whl
```

---

## 装完之后

1. 每次开新 shell，重新 source CANN（步骤 0 那一坨）。
2. 训练前激活 `transfer_to_npu` shim。shim 文件和封装好的启动脚本会在
   upstream-strategy 计划的 PR3 里加上；在那之前最小激活方式见
   `docs/ascend_npu/upstream_strategy_zh.md` §5.4。
3. 单机多 NPU 训练：用 `examples/run_qwen3_8b_dflash_online.sh`，加上
   `--target-model-backend hf --attention-backend sdpa`。多机扩展见
   `docs/ascend_npu/multi_node_training_zh.md`。
