# SpecForge — Extending DFlash NPU Training from Single-Node to Multi-Node

> Premise: someone has already brought up DFlash single-node training on Ascend NPU
> (see `summary.md` §4 for the device-abstraction patches that step needs).
> This document answers, in layers, how to extend that to a multi-node setup.

A working reference implementation of multi-node FSDP training on Ascend already
exists in `D:\work\qwen3.5_omni_creative` (the MindSpeed-MM project). The patterns
shown there can be ported to SpecForge with only minor changes.

---

## Layer 1 — Which training backend does SpecForge use?

**SpecForge uses PyTorch native FSDP1**, not Megatron, and not FSDP2.

Direct evidence:

```python
# scripts/train_dflash.py
17:  from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
18:  from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
...
441: dflash_model = FSDP(
442:     dflash_model,
443:     use_orig_params=True,
444:     mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
445:     sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,   # ZeRO-2 like; only grads/optim sharded
446: )
```

`scripts/train_eagle3.py:808` is the same shape. Concretely:

| Aspect | What SpecForge does |
|---|---|
| **Backend** | PyTorch native **FSDP1** (`FullyShardedDataParallel`); not FSDP2 (`fully_shard`), not Megatron |
| **Sharding strategy** | `SHARD_GRAD_OP` (≈ ZeRO-2): grads + optimizer state sharded, **parameters replicated** — every rank must hold a full copy of the draft model |
| **Mixed precision** | bf16 weights/buffers via FSDP `MixedPrecision`; outer `BF16Optimizer` (`specforge/optimizer.py`) keeps fp32 master weights |
| **Process groups** | Built in `specforge/distributed.py:init_distributed()`: a `dp/tp` device mesh, plus yunchang sequence-parallel groups. Draft model uses only the dp group; target model uses tp |
| **Not supported** | pipeline parallel, Megatron column/row parallel linear, ZeRO-3 / FULL_SHARD by default, parameter CPU offload |
| **Launcher** | `torchrun`. Every example today uses `--standalone --nproc_per_node N` (single-node only) |

The reference repo (`qwen3.5_omni_creative/mindspeed_mm/fsdp/...`) uses **FSDP2
(`fully_shard`)** with explicit `init_device_mesh`. The internals differ from FSDP1,
but the multi-node *envelope* (torchrun + HCCL + env vars) is reusable as-is.

---

## Layer 2 — How to extend to multi-node on Ascend

Four sub-layers, in increasing depth:

> **A. Launcher (the easy part)** →
> **B. Communication / environment (Ascend specifics)** →
> **C. Code-side changes (so the process group actually works)** →
> **D. Optional parallelism dimensions multi-node enables**

### A. torchrun multi-node launch — drop `--standalone`

Today's SpecForge examples look like:

```bash
torchrun --standalone --nproc_per_node $NUM_GPUS scripts/train_dflash.py ...
```

`--standalone` is shorthand for "single-node, pick a free port, rendezvous locally".
For multi-node, replace it with the full four-tuple
(see `qwen3.5_omni_creative/examples/fsdp2/qwen3_5/finetune_qwen3_5_27B.sh:17-24`
and `scripts_qwen3_5/pretrain-exp1_qwen3_5_4b.sh:46-52`):

```bash
NPUS_PER_NODE=8
NNODES=2                           # number of nodes
NODE_RANK=${NODE_RANK:-0}          # this node's index, 0..NNODES-1
MASTER_ADDR=${MASTER_ADDR:-10.x.x.x}   # IP of node 0
MASTER_PORT=6000

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS scripts/train_dflash.py <existing args...>
```

Run the same script on every node, only changing `NODE_RANK` (0, 1, …).

For SSH-based fan-out across many nodes, copy `qwen3.5_omni_creative/launch_multi_nodes.sh`:
it reads `node_list.txt`, SSHes to each host, runs `nohup bash $REMOTE_SCRIPT
$MASTER_ADDR $NUM_NODES $i $GBS &`, and uses `wait` + `trap` to collect logs and clean
up on Ctrl-C. Just point `REMOTE_SCRIPT` at your DFlash launcher.

### B. Ascend environment variables

Copy the env section from `scripts_qwen3_5/pretrain-exp2_qwen3_5_10b-a2b.sh:1-26`.

**Required for both single- and multi-node:**

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

**Required only for multi-node (the two most common pitfalls):**

```bash
# HCCL NIC name — must match the actual NIC on each node (check with `ip a`)
# and the NICs across nodes must be routable.
export GLOO_SOCKET_IFNAME=enp66s0f0
export HCCL_SOCKET_IFNAME=enp66s0f0

# First multi-node all-reduce / kernel cache build is much slower than single-node;
# generously bump the timeouts.
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=1800
```

Note: `CUDA_DEVICE_MAX_CONNECTIONS=1` is common in Megatron, but the reference
repo's FSDP2 example (`qwen3vl/finetune_qwen3vl_30B.sh:8`) explicitly says **do not
set it to 1 when FSDP2 is enabled**. SpecForge is FSDP1 — leaving it unset is the
safer default; do **not** copy that variable from Megatron scripts.

### C. Code-side changes (don't only edit shell scripts)

`specforge/distributed.py:66-120` was written assuming cuda+nccl. Even after the
single-node patches in `summary.md §4.2` (cuda → npu, nccl → hccl), multi-node
needs two more guarantees:

#### C.1 Make the backend choice dynamic — port the helpers from MindSpeed-MM

`qwen3.5_omni_creative/mindspeed_mm/fsdp/utils/device.py:49-73`:

```python
def get_dist_comm_backend(cpu: bool = False) -> str:
    if cpu:
        return "cpu:gloo,npu:hccl" if IS_NPU_AVAILABLE else "cpu:gloo,cuda:nccl"
    return "hccl" if IS_NPU_AVAILABLE else "nccl"

def get_device_type() -> str:
    return "npu" if IS_NPU_AVAILABLE else "cuda"
```

Drop these two helpers into `specforge/utils.py` (or a new
`specforge/device.py`) and rewrite `specforge/distributed.py:75-77`:

```python
from specforge.device import get_dist_comm_backend, get_device_type
DEVICE = get_device_type()
dist.init_process_group(backend=get_dist_comm_backend(), timeout=timedelta(minutes=timeout))

# IMPORTANT: in multi-node, use LOCAL_RANK from torchrun env, NOT dist.get_rank() % device_count
local_rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(local_rank) if DEVICE == "npu" else torch.cuda.set_device(local_rank)
```

> **The most common multi-node bug**:
> `local_rank = dist.get_rank() % torch.cuda.device_count()`.
> On a single node `dist.get_rank()` happens to equal `LOCAL_RANK`. On multi-node
> they diverge — node_rank=1 gets ranks 8..15; mod 8 still yields 0..7, so it
> *appears* correct, but only as long as every node has the same number of cards
> and the rank-to-device mapping aligns with what you assumed. Read
> `os.environ["LOCAL_RANK"]` directly — torchrun always injects it.
> SpecForge `distributed.py:76` must be changed for NPU multi-node.

#### C.2 device_type on every device mesh

```python
device_mesh = dist.device_mesh.init_device_mesh(
    DEVICE,                       # "npu" or "cuda" — no more hard-coded "cuda"
    (dp_size, tp_size),
    mesh_dim_names=("dp", "tp"),
)
...
tp_device_mesh = dist.DeviceMesh.from_group(tp_group, device_type=DEVICE)
_DP_DEVICE_MESH  = dist.DeviceMesh.from_group(dp_group, device_type=DEVICE)
```

Cf. `qwen3.5_omni_creative/mindspeed_mm/fsdp/distributed/parallel_state.py:61`'s
`init_device_mesh(device_type=get_device_type(), ...)`.

#### C.3 Every `.cuda()` / `device="cuda"` in the training script

`scripts/train_dflash.py:163, 194, 426, 499–505` → use `.to(DEVICE)` /
`device=DEVICE`. This is also single-node hygiene, but multi-node will *not*
reveal new problems here — the failure mode is the same OOM / device-mismatch.

#### C.4 FSDP wrap itself — no change needed

FSDP1 is device-agnostic. As long as `init_process_group(backend="hccl")` ran
and the model is on npu, `FSDP(...)` just works; AllGather/ReduceScatter route
through HCCL automatically. Cf. the FSDP2 reference at
`mindspeed_mm/fsdp/train/trainer.py:137-141`, which is also a single
`init_process_group` line.

### D. Parallelism that multi-node unlocks (optional)

Today SpecForge's `--tp-size` is used **only for the target model**
(`SGLangDFlashTargetModel`, splitting the inference-side KV); the draft model is
always replicated, never sharded. Multi-node gives you:

| Dimension | Single-node 8 NPU | Multi-node N×8 NPU | How to enable |
|---|---|---|---|
| **DP** | dp_size = 8 | dp_size = N × 8 | Free — torchrun controls it |
| **target TP** | tp = 1 / 2 / 4 | tp = 8 / 16 | `--tp-size 16`; only meaningful with `--target-model-backend sglang` |
| **batch / long context** | bound by per-node memory | near-linear scaling | bump `--batch-size`, `--max-length` |
| **draft FSDP memory** | SHARD_GRAD_OP usually fine | same | If still tight, switch sharding to `FULL_SHARD` (≈ ZeRO-3) |
| **Sequence parallel (yunchang)** | sp_*=1 | available | `init_distributed(sp_ulysses_size=, sp_ring_size=)` — validated on EAGLE3, **not yet on DFlash** |

**Important**: DFlash's anchor sampling and block_mask are not yet aware of
sequence parallelism. **First step: scale DP only, do not touch SP**. Validate
DP-only multi-node, then revisit yunchang.

---

## A drop-in multi-node launch script

```bash
#!/bin/bash
# run_qwen3_8b_dflash_online_multinode.sh
# Usage: bash run_qwen3_8b_dflash_online_multinode.sh <MASTER_ADDR> <NNODES> <NODE_RANK>

source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Single-node + multi-node common
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2
export MULTI_STREAM_MEMORY_REUSE=2
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1,lazy_bind:0
export NPU_ASD_ENABLE=0
export ASCEND_LAUNCH_BLOCKING=0

# Multi-node only
export GLOO_SOCKET_IFNAME=enp66s0f0     # adjust to your NIC name
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

Driving N nodes from a single shell with `launch_multi_nodes.sh`:

```bash
# node_list.txt — one host/IP per line
10.0.0.1
10.0.0.2

bash launch_multi_nodes.sh ./node_list.txt run_qwen3_8b_dflash_online_multinode.sh <GBS>
```

---

## TL;DR

- SpecForge uses **PyTorch native FSDP1 with `SHARD_GRAD_OP`**. Not Megatron, not FSDP2. Extending to multi-node is conceptually thin: `torchrun --nnodes/--node_rank` plus HCCL.
- Real code edits land in three places: `specforge/distributed.py` (dynamic backend / device / `LOCAL_RANK`), the training scripts (`.cuda()` → `.to(DEVICE)`), and one device-abstraction helper module.
- Two pieces of the reference repo are worth lifting almost verbatim: `mindspeed_mm/fsdp/utils/device.py` (`get_dist_comm_backend` / `get_device_type`) and `launch_multi_nodes.sh` (SSH fan-out + graceful exit).
- Scale DP first (draft model replicated, dp = N × 8). Do not turn on yunchang sequence parallel until DFlash's anchor / block_mask paths are explicitly validated under SP.
