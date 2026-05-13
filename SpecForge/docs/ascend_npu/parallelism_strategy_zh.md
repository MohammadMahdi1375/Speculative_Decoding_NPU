# SpecForge 多机训练 —— 并行策略选型决策

> 本文档与 `multi_node_training_zh.md` 配套使用。
> 那一篇讲「DFlash 多机训练在昇腾 NPU 上**怎么拉起来**」，
> 本篇记录「**第一版用哪种 sharding 策略**」以及「**什么时候回头优化**」。

---

## 决策（当前计划）

**第一版多机 DFlash on Ascend NPU 维持 SpecForge 现有 FSDP1 不动**：

- `ShardingStrategy.SHARD_GRAD_OP`（≈ ZeRO-2）
- `process_group=dist.group.WORLD`（一个扁平 FSDP 组覆盖所有节点的所有卡）
- **不上 HSDP，不上 DDP，不重设并行 mesh**

**理由**：在已知能跑通的单机代码路径上做最小改动。先保证多机的**正确性**，
跨节点性能差点完全可以接受。

**触发回头优化的条件**：profiling 显示跨节点 ReduceScatter / AllGather 占了 step
时间的大头（经验阈值：双机吞吐 < 单机的 1.6×）。这时再切到 HSDP，按下面
"升级路径"那一节操作。

---

## 背景 —— `SHARD_GRAD_OP` over WORLD 到底在做什么

PyTorch FSDP1 提供四档 sharding 策略，SpecForge 选了第二档：

| 策略 | param 切片 | grad 切片 | optim 切片 | 等价于 |
|---|---|---|---|---|
| `FULL_SHARD` | ✅ | ✅ | ✅ | ZeRO-3 |
| **`SHARD_GRAD_OP`** ← SpecForge 用这档 | ❌（fwd/bwd 期间是 full）| ✅ | ✅ | ZeRO-2 |
| `HYBRID_SHARD` | ✅ 节点内 | ✅ 节点内 | ✅ 节点内 | HSDP / ZeRO-3 + replica |
| `NO_SHARD` | ❌ | ❌ | ❌ | DDP（仅 AllReduce） |

`SHARD_GRAD_OP` 在 m 张卡的 WORLD 组里，单 iter 通信流程：

1. 进 forward 前：在所有 m 张卡之间 **AllGather(param)**
2. forward + backward 期间 param 保持完整
3. backward 结束：在所有 m 张卡之间 **ReduceScatter(grad)**，同时 param 重新切片
4. optimizer step：每张卡只更新自己那 1/m 份

每 iter 跨节点链路上跑约 `2 × param_size` 字节，分两次 collective。

---

## 为什么单机没问题、多机不太理想

**单机**（8 卡）：AllGather/ReduceScatter 跑在节点内 HCCS / NVLink 上，带宽富余，
两次 collective 在 step 时间里几乎看不见。同时 optimizer state 切了 8 份，省下来的
显存能多给 batch 用——稳赚。

**多机**（N × 8 卡）：同样这两次 collective 现在要跑节点间网络（RoCE / IB）。
以 1B params 的 draft（≈ 2 GB bf16）为例：

- 单 iter 跨节点字节数 ≈ 2 × 2 GB = **~4 GB**
- 100 Gbps 有效带宽 ≈ 12.5 GB/s → **每步至少 ~320 ms 通信**
- 单步算力（1B draft、bf16、batch 2、seq 3072）大约 100–200 ms

通信变成了 step 时间里非小的一块，但**对小 draft 还谈不上致命**。这正是「v1 先忍着」
这个决策的依据。

---

## 三条路的对比

2 节点 × 8 卡，draft ≈ 1B params（≈ 2 GB bf16），AdamW + fp32 master
optim state 12 B/param：

| 选项 | 单 iter 跨节点字节 | 单 iter 跨节点 collective 次数 | 单卡 optim 状态 | 代码改动 |
|---|---|---|---|---|
| **(A) FSDP `SHARD_GRAD_OP` over WORLD** ← v1 选这个 | ~2 × param ≈ **4 GB** | 2 次（AG + RS） | 12 GB / 16 = 0.75 GB | **零改动** |
| **(B) HSDP（`_HYBRID_SHARD_ZERO2`，节点内 shard=8，节点间 replicate=N）** | ~2 × (param / 8) ≈ **0.5 GB** | 1 次（grad shard 上的 AR） | 12 GB / 8 = 1.5 GB | 小改（~10 行） |
| **(C) DDP（`NO_SHARD`）** | ~2 × param ≈ **4 GB** | 1 次（full grad 上的 AR） | **12 GB（不切）** | 中改 —— `save_checkpoint()` 要重写 |

要点：

- **(A) 和 (C) 跨节点字节数几乎相同**——ring AllReduce 内部就是 RS+AG，量级对齐。
  区别在 (A) 多一次 collective（多一点启动延迟和同步屏障）但切了 optim state；
  (C) 概念更简单但每张卡都要装完整 optim。
- **(B) HSDP 才是多机的"架构正解"**：跨节点链路只搬已经切片的梯度做 AllReduce，
  跨节点字节数随 `param / shard_size` 线性下降。shard=8 即 8× 减少。
- HSDP **不需要** MindSpeed-MM 那种 5D 并行 mesh，只要一个 2D `(replicate, shard)`
  的 device_mesh，几行代码。

---

## 为什么 v1 不直接上 HSDP

1. **改动面最小**。v1 已经有的改动（cuda → npu、nccl → hccl、`LOCAL_RANK` 读 env）
   就足够把多机拉起来。再叠一层 HSDP 等于在新的代码路径上再叠一个新代码路径，
   debug 面变大。
2. **正确性优先于吞吐**。慢但数值正确的多机训练可以靠 profiling 优化；
   快但暗藏 bug 的多机训练（mesh dim 名错、rank-device 绑错、optim 切片不一致）
   能耗掉一周。
3. **Draft 规模决定了上限收益**。SpecForge 的 draft 顶天 1–2B 参数。
   1B 时 FSDP-WORLD 比 HSDP 多搬约 3.5 GB / iter——烦人，但远谈不上致命。
4. **HSDP 是一个旋钮，等需要时再拧**——下面那一节就一个 diff。

---

## 升级路径 —— 通信成为瓶颈时怎么切到 HSDP

触发条件：profiling 显示跨节点 collective > step 时间的 25%，或者双机相对单机的
吞吐扩展系数 < 1.6×。

改动只在 `scripts/train_dflash.py:441-449` 加一个 mesh + 在 `specforge/distributed.py`
里暴露一个小 helper：

```python
# scripts/train_dflash.py
from torch.distributed.device_mesh import init_device_mesh
from specforge.device import get_device_type

DEVICE = get_device_type()
NNODES         = int(os.environ["WORLD_SIZE"]) // int(os.environ["LOCAL_WORLD_SIZE"])
NPUS_PER_NODE  = int(os.environ["LOCAL_WORLD_SIZE"])

hsdp_mesh = init_device_mesh(
    DEVICE,
    (NNODES, NPUS_PER_NODE),
    mesh_dim_names=("replicate", "shard"),
)

dflash_model = FSDP(
    dflash_model,
    use_orig_params=True,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    sharding_strategy=ShardingStrategy._HYBRID_SHARD_ZERO2,   # 原来是 SHARD_GRAD_OP
    device_mesh=hsdp_mesh,                                    # 原来是隐式 WORLD
)
```

这就是切到 HSDP 的全部增量。`_HYBRID_SHARD_ZERO2` 保留 SpecForge 原有
"compute 期间 param 不切" 的行为；如果未来 draft 大到单卡装不下，再换成
`HYBRID_SHARD`（节点内 ZeRO-3）。

> **顺便一个独立但相关的坑**：如果做完 HSDP 还感觉显存紧，要查的是
> `BF16Optimizer.fp32_params`——SpecForge 现在每张卡都存了一份完整的 fp32
> 主权重，**绕过了 FSDP 的 optimizer 切片**。这是另外一件事，和 HSDP 正交，
> 真要省那一份 fp32 master 需要单独改 `specforge/optimizer.py`。

---

## 一句话总结

- **v1 计划**：保持 `SHARD_GRAD_OP` over `WORLD` 不变，**不动并行 mesh**。
  先把多机跑对，跨节点通信开销暂时忍着。
- **HSDP 不是必须的**——它是「下一步该做的优化」，不是「现在该做的修改」。
- **回头看的红线**：双机吞吐 < 单机的 1.6×，或跨节点 collective > step 时间的 25%。
- **真要切 HSDP 时改动 ~10 行**，是个旋钮问题，故意推迟。
