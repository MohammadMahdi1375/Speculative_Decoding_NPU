# SpecForge —— 把 DFlash NPU 训练从单机扩展到多机

> 前提：已经有人把 DFlash 单机 NPU 训练拉起来了
> （单机所需的 device 抽象 patch 见 `summary_zh.md §4`）。
> 本文档分层回答如何把它扩展到多机。

参考实现来自 `D:\work\qwen3.5_omni_creative`（MindSpeed-MM 项目），
里面的 FSDP 多机方案在昇腾上已经跑通，可以直接照搬到 SpecForge，只需少量改动。

---

## 第 1 层：SpecForge 用什么训练后端？

**SpecForge 用的是 PyTorch 原生 FSDP1**，既不是 Megatron，也不是 FSDP2。

直接看代码：

```python
# scripts/train_dflash.py
17:  from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
18:  from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
...
441: dflash_model = FSDP(
442:     dflash_model,
443:     use_orig_params=True,
444:     mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
445:     sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,   # ≈ ZeRO-2，仅切 grad/optim
446: )
```

`scripts/train_eagle3.py:808` 写法相同。具体来说：

| 维度 | SpecForge 现状 |
|---|---|
| **后端** | PyTorch 原生 **FSDP1**（`FullyShardedDataParallel`）；不是 FSDP2 (`fully_shard`)，不是 Megatron |
| **切分策略** | `SHARD_GRAD_OP`（≈ ZeRO-2）：grad + optimizer state 切片，**参数仍复制**——每张卡都要装得下完整 draft 模型 |
| **混合精度** | FSDP 层面 bf16，外层 `BF16Optimizer`（`specforge/optimizer.py`）保留 fp32 主权重 |
| **进程组** | `specforge/distributed.py:init_distributed()` 自建 `dp/tp` device mesh + yunchang 序列并行组。draft 只用 dp，target 用 tp |
| **不支持** | pipeline parallel、Megatron 列/行并行 linear、默认 ZeRO-3 / FULL_SHARD、参数 CPU offload |
| **启动器** | `torchrun`。当前所有 example 都是 `--standalone --nproc_per_node N`（单机） |

参考库 (`qwen3.5_omni_creative/mindspeed_mm/fsdp/...`) 用的是
**FSDP2 (`fully_shard`)** + 显式 `init_device_mesh`。内部机制不同，但拉起多机的
**外壳**（torchrun + HCCL + 环境变量）完全可以原样借用给 SpecForge 的 FSDP1。

---

## 第 2 层：怎么在昇腾上扩到多机

按由浅到深分四个子层：

> **A. 拉起方式（最简单）** →
> **B. 通信 / 环境变量（昇腾必做）** →
> **C. 代码侧改造（让进程组真的能跑）** →
> **D. 多机才能解锁的可选并行扩展**

### A. torchrun 多机拉起：去掉 `--standalone`

SpecForge 现有 example：

```bash
torchrun --standalone --nproc_per_node $NUM_GPUS scripts/train_dflash.py ...
```

`--standalone` 是「就这一台机，本地选个空闲端口 rendezvous」的语法糖。
多机要换成显式四件套（参考
`qwen3.5_omni_creative/examples/fsdp2/qwen3_5/finetune_qwen3_5_27B.sh:17-24`
和 `scripts_qwen3_5/pretrain-exp1_qwen3_5_4b.sh:46-52`）：

```bash
NPUS_PER_NODE=8
NNODES=2                           # 几台机
NODE_RANK=${NODE_RANK:-0}          # 本机在集群里的编号 0..NNODES-1
MASTER_ADDR=${MASTER_ADDR:-10.x.x.x}    # 第 0 号机的 IP
MASTER_PORT=6000

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS scripts/train_dflash.py <已有参数...>
```

每台机都跑同一个脚本，只改 `NODE_RANK`（0、1、…）。

如果想批量 SSH 拉 N 台机，把 `qwen3.5_omni_creative/launch_multi_nodes.sh` 抄过来：
它读 `node_list.txt`、SSH 到每台远端、跑
`nohup bash $REMOTE_SCRIPT $MASTER_ADDR $NUM_NODES $i $GBS &`，并通过 `wait` + `trap`
收日志、Ctrl-C 时优雅清理。把 `REMOTE_SCRIPT` 指向你写的 DFlash 启动脚本即可。

### B. 昇腾必备的环境变量

把参考库 `scripts_qwen3_5/pretrain-exp2_qwen3_5_10b-a2b.sh:1-26` 的环境段抄过来。

**单机和多机都要的：**

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2
export MULTI_STREAM_MEMORY_REUSE=2
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1,lazy_bind:0
export NPU_ASD_ENABLE=0
export ASCEND_LAUNCH_BLOCKING=0
```

**多机才必须（最容易踩坑的两条）：**

```bash
# HCCL 网卡名——必须对应每台机真实的网卡（用 `ip a` 看），
# 而且两台机走的网卡必须互通。
export GLOO_SOCKET_IFNAME=enp66s0f0
export HCCL_SOCKET_IFNAME=enp66s0f0

# 多机首次 all-reduce / 内核缓存编译比单机慢得多，超时一定要拉长。
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=1800
```

注：`CUDA_DEVICE_MAX_CONNECTIONS=1` 在 Megatron 里很常见，但参考库 FSDP2 路径
`qwen3vl/finetune_qwen3vl_30B.sh:8` 明确注释「**开启 FSDP2 时不能置为 1**」。
SpecForge 是 FSDP1，**不要从 Megatron 脚本里抄这一行**——保留默认更稳。

### C. 代码侧改造（这是关键，只改脚本不够）

`specforge/distributed.py:66-120` 写死了 cuda + nccl。即使 `summary_zh.md §4.2`
里的单机改动已经把 `cuda` 替成 `npu`、`nccl` 替成 `hccl`，**多机还要再确认两件事**。

#### C.1 把后端选择做成动态的——直接抄参考库的 helper

`qwen3.5_omni_creative/mindspeed_mm/fsdp/utils/device.py:49-73`：

```python
def get_dist_comm_backend(cpu: bool = False) -> str:
    if cpu:
        return "cpu:gloo,npu:hccl" if IS_NPU_AVAILABLE else "cpu:gloo,cuda:nccl"
    return "hccl" if IS_NPU_AVAILABLE else "nccl"

def get_device_type() -> str:
    return "npu" if IS_NPU_AVAILABLE else "cuda"
```

把这两个 helper 放进 `specforge/utils.py`（或新建 `specforge/device.py`），
然后改写 `specforge/distributed.py:75-77`：

```python
from specforge.device import get_dist_comm_backend, get_device_type
DEVICE = get_device_type()
dist.init_process_group(backend=get_dist_comm_backend(), timeout=timedelta(minutes=timeout))

# 重要：多机里 local_rank 必须读 LOCAL_RANK，不能用 dist.get_rank() % device_count
local_rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(local_rank) if DEVICE == "npu" else torch.cuda.set_device(local_rank)
```

> **多机里最经典的 bug**：
> `local_rank = dist.get_rank() % torch.cuda.device_count()`。
> 单机时 `dist.get_rank()` 等价于 `LOCAL_RANK`，但**多机时不等价**——node_rank=1
> 上 rank 是 8..15，对 8 取模也是 0..7，看起来对，**但前提是每台机卡数完全一样、
> 且 rank → device 的映射跟你想的一致**。最稳的写法是直接读
> `os.environ["LOCAL_RANK"]`（torchrun 一定会注入）。
> SpecForge `distributed.py:76` 在 NPU 多机场景必须改成这种写法。

#### C.2 device mesh 的 device_type

```python
device_mesh = dist.device_mesh.init_device_mesh(
    DEVICE,                       # "npu" 或 "cuda"，不能再硬编码 "cuda"
    (dp_size, tp_size),
    mesh_dim_names=("dp", "tp"),
)
...
tp_device_mesh = dist.DeviceMesh.from_group(tp_group, device_type=DEVICE)
_DP_DEVICE_MESH  = dist.DeviceMesh.from_group(dp_group, device_type=DEVICE)
```

参考 `qwen3.5_omni_creative/mindspeed_mm/fsdp/distributed/parallel_state.py:61` 的
`init_device_mesh(device_type=get_device_type(), ...)`。

#### C.3 训练脚本里所有 `.cuda()` / `device="cuda"`

`scripts/train_dflash.py:163, 194, 426, 499–505` 全部改为 `.to(DEVICE)` /
`device=DEVICE`。这条单机也要做，多机不会暴露新问题，但漏掉就是 OOM 或
device mismatch。

#### C.4 FSDP wrap 本身——不需要改

FSDP1 是 device-agnostic 的。只要 `init_process_group(backend="hccl")` 成功、
模型已经在 npu 上，`FSDP(...)` 直接就能用，AllGather/ReduceScatter 自动走 HCCL。
参考库 FSDP2 路径在 `mindspeed_mm/fsdp/train/trainer.py:137-141` 也只是
`init_process_group` 一行，思路一致。

### D. 多机才能解锁的可选并行扩展

当前 SpecForge 的 `--tp-size` 只用于 **target 模型**
（`SGLangDFlashTargetModel`，给推理那侧切 KV）；draft 模型本身始终复制不切。
多机带来的好处是：

| 维度 | 单机 8 NPU | 多机 N×8 NPU | 怎么开 |
|---|---|---|---|
| **DP** | dp_size = 8 | dp_size = N × 8 | 自动，torchrun 起多少进程就有多少 dp |
| **target TP** | tp = 1 / 2 / 4 | tp = 8 / 16 | `--tp-size 16`，仅在 `--target-model-backend sglang` 时有意义 |
| **batch / 长上下文** | 受单机显存限制 | 几乎线性扩展 | 直接调 `--batch-size`、`--max-length` |
| **draft FSDP 显存** | SHARD_GRAD_OP 一般够 | 同左 | 真要再省可换成 `FULL_SHARD`（≈ ZeRO-3） |
| **序列并行（yunchang）** | sp_*=1 | 可用 | `init_distributed(sp_ulysses_size=, sp_ring_size=)`——EAGLE3 验证较多，**DFlash 还没验证过** |

**注意**：DFlash 的 anchor 采样和 block_mask 还**没**适配序列并行。
**第一步只扩 DP，不要碰 SP**。等 DP-only 多机跑通后再去验证 yunchang。

---

## 一份多机可直接套的启动脚本骨架

```bash
#!/bin/bash
# run_qwen3_8b_dflash_online_multinode.sh
# 用法: bash run_qwen3_8b_dflash_online_multinode.sh <MASTER_ADDR> <NNODES> <NODE_RANK>

source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 单机和多机都要的
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2
export MULTI_STREAM_MEMORY_REUSE=2
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1,lazy_bind:0
export NPU_ASD_ENABLE=0
export ASCEND_LAUNCH_BLOCKING=0

# 多机才必须
export GLOO_SOCKET_IFNAME=enp66s0f0     # 改成你的网卡名
export HCCL_SOCKET_IFNAME=enp66s0f0
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=1800

ROOT_DIR=$(cd $(dirname $0)/.. && pwd)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=32

NPUS_PER_NODE=8
MASTER_ADDR=${1:-localhost}
MASTER_PORT=6000
NNODES=${2:-2}
NODE_RANK=${3:-0}

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path Qwen/Qwen3-8B \
    --draft-config-path $ROOT_DIR/configs/qwen3-8b-dflash.json \
    --train-data-path $ROOT_DIR/cache/dataset/perfectblend_qwen3-8b_regen.jsonl \
    --output-dir $ROOT_DIR/outputs/qwen3-8b-dflash-npu-multinode \
    --target-model-backend hf \
    --attention-backend sdpa \
    --num-epochs 6 --batch-size 2 --max-length 3072 \
    --learning-rate 6e-4 --warmup-ratio 0.04 --max-grad-norm 1.0 \
    --block-size 16 --num-anchors 512 --loss-decay-gamma 7.0 \
    --chat-template qwen --report-to tensorboard \
    --dist-timeout 60
```

外层用 `launch_multi_nodes.sh` 把 N 台机一起拉起：

```bash
# node_list.txt——每行一个 host 或 IP
10.0.0.1
10.0.0.2

bash launch_multi_nodes.sh ./node_list.txt run_qwen3_8b_dflash_online_multinode.sh <GBS>
```

---

## 一句话总结

- SpecForge 用的是 **PyTorch 原生 FSDP1（SHARD_GRAD_OP）**，不是 Megatron，也不是 FSDP2。扩到多机本身很轻：`torchrun --nnodes/--node_rank` + HCCL 即可。
- 真正要动的代码就三处：`specforge/distributed.py`（动态 backend / device / `LOCAL_RANK`）、训练脚本（`.cuda()` → `.to(DEVICE)`）、新建一个 device 抽象 helper 模块。
- 参考库里两个值得直接抄的部件：`mindspeed_mm/fsdp/utils/device.py`（`get_dist_comm_backend / get_device_type`）和 `launch_multi_nodes.sh`（SSH 多机编排 + 优雅退出）。
- 第一步只扩 DP（draft 模型每节点复制，dp_size = N × 8），暂时别碰 yunchang 序列并行——DFlash 的 anchor / block_mask 还没在 SP 下验证过。
