# SpecForge Multi-Node — Parallelism Strategy Decision

> Companion to `multi_node_training.md`. That document covers *how* to launch
> multi-node DFlash on Ascend NPU. This document captures *which sharding
> strategy we ship first* and *the trigger for revisiting it*.

---

## Decision (current plan)

**For the first cut of multi-node DFlash on Ascend NPU we keep SpecForge's
existing FSDP1 setup unchanged**:

- `ShardingStrategy.SHARD_GRAD_OP` (≈ ZeRO-2)
- `process_group=dist.group.WORLD` (a single flat FSDP group covering every
  rank on every node)
- No HSDP, no DDP, no parallelism mesh redesign

**Rationale:** smallest possible diff to a known-good single-node code path.
Get correctness on multi-node first; tolerate sub-optimal cross-node throughput.

**Revisit when:** profiling shows cross-node ReduceScatter / AllGather
dominating step time (rule of thumb: 2-node throughput < 1.6× of 1-node).
Then switch to HSDP per §"Upgrade path" below.

---

## Background — what `SHARD_GRAD_OP` over WORLD actually does

PyTorch FSDP1 has four sharding strategies. SpecForge picks the second:

| Strategy | param sharded | grad sharded | optim sharded | Equivalent to |
|---|---|---|---|---|
| `FULL_SHARD` | yes | yes | yes | ZeRO-3 |
| **`SHARD_GRAD_OP`** ← used by SpecForge | no (full during fwd/bwd) | yes | yes | ZeRO-2 |
| `HYBRID_SHARD` | yes intra-node | yes intra-node | yes intra-node | HSDP / ZeRO-3 + replica |
| `NO_SHARD` | no | no | no | DDP (AllReduce only) |

Per-iteration communication for `SHARD_GRAD_OP` over a `WORLD` group of m ranks:

1. Before forward: **AllGather(param)** across all m ranks
2. Forward + backward use full params
3. After backward: **ReduceScatter(grad)** across all m ranks; params reshard
4. Optimizer step: each rank updates only its 1/m shard

So per iter the cross-node link sees roughly `2 × param_size` worth of bytes
across two collectives.

---

## Why this is fine on a single node, sub-optimal on multi-node

**Single node** (8 cards): AllGather/ReduceScatter run on the intra-host
HCCS/NVLink fabric. The fabric is abundant; the two collectives barely show
up in step time, and we get the win — optimizer state sharded 8-ways.

**Multi-node** (N × 8 cards): the same two collectives now span the
inter-node network (RoCE / IB). For a draft model of ~1B bf16 params:

- Per-iter cross-node bytes ≈ 2 × 2 GB = **~4 GB**
- At 100 Gbps effective ≈ 12.5 GB/s → **~320 ms of comm per step** at best
- Compute per step (1B draft, bf16, batch 2, seq 3072) is on the order of
  100–200 ms

Comm becomes a meaningful fraction of step time, but **not catastrophic** for
small drafts. Hence the "tolerate it for v1" decision.

---

## Three strategies considered

For 2 nodes × 8 cards, draft ≈ 1B params (≈ 2 GB bf16), 12 B/param of
optimizer state (fp32 master + Adam m + v):

| Option | Cross-node bytes / iter | Cross-node collectives / iter | Optimizer state per card | Code diff |
|---|---|---|---|---|
| **(A) FSDP `SHARD_GRAD_OP` over WORLD** ← chosen for v1 | ~2 × param ≈ **4 GB** | 2 (AG + RS) | 12 GB / 16 = 0.75 GB | **zero** |
| **(B) HSDP (`_HYBRID_SHARD_ZERO2`, shard=8 intra, replicate=N inter)** | ~2 × (param / 8) ≈ **0.5 GB** | 1 (AR on grad shard) | 12 GB / 8 = 1.5 GB | small (~10 lines) |
| **(C) DDP (`NO_SHARD`)** | ~2 × param ≈ **4 GB** | 1 (AR on full grad) | **12 GB (full)** | medium — `save_checkpoint()` rewrite |

Notes:

- **(A) vs (C)** have nearly identical cross-node byte counts (ring AllReduce
  is RS+AG internally). The differences are: (A) has 2 collectives vs 1 (a bit
  more latency overhead and an extra synchronization barrier), and (A) shards
  the optimizer state. (C) is simpler conceptually but loses the
  optimizer-sharding savings.
- **(B) HSDP** is the architecturally clean answer for multi-node. The
  cross-node link only carries an AllReduce on already-sharded gradients, so
  cross-node bytes scale as `param / shard_size`. With shard=8 that's an 8×
  reduction.
- HSDP does **not** require a 5D parallelism mesh à la MindSpeed-MM. We only
  need a 2D `(replicate, shard)` device_mesh — a few lines.

---

## Why we are NOT picking HSDP up front

1. **Smallest viable diff.** v1 only requires the device-abstraction patches
   already documented (cuda → npu, nccl → hccl, `LOCAL_RANK` from env). Adding
   HSDP layers another change and another debug surface on top of that.
2. **Correctness > throughput for v1.** A multi-node run that's slow but
   numerically correct can be profiled and optimized. A multi-node run that's
   fast but subtly wrong (wrong DeviceMesh dim names, wrong rank-to-device
   binding, sharded optimizer state out of sync) costs a week.
3. **The win is bounded by draft size.** SpecForge drafts top out around 1–2 B
   parameters. The absolute cross-node delta between FSDP-WORLD and HSDP for
   a 1 B draft is ~3.5 GB/iter. Annoying, not lethal.
4. **HSDP is a one-knob upgrade later.** The change is mechanical (see below).

---

## Upgrade path — when comm becomes the bottleneck

Trigger: profile shows cross-node collectives are >25% of step time, OR
the 2-node-vs-1-node throughput scaling factor is < 1.6.

The change is local to `scripts/train_dflash.py:441-449` plus a small helper
in `specforge/distributed.py`:

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
    sharding_strategy=ShardingStrategy._HYBRID_SHARD_ZERO2,   # was SHARD_GRAD_OP
    device_mesh=hsdp_mesh,                                    # was implicit WORLD
)
```

That's the entire HSDP delta. `_HYBRID_SHARD_ZERO2` keeps the "params
replicated during compute" behaviour SpecForge already relies on. If draft
ever outgrows a single card, swap to `HYBRID_SHARD` for ZeRO-3 within node.

If we further discover that the per-card replica of the optimizer master
weights (the `BF16Optimizer.fp32_params` list) is what's eating memory, that's
**orthogonal** to HSDP and needs its own work — `BF16Optimizer` currently
keeps a full fp32 copy on every rank and bypasses FSDP's optimizer sharding.

---

## TL;DR

- **v1 plan:** keep `SHARD_GRAD_OP` over `WORLD`. Don't touch the parallelism
  mesh. Get multi-node correct, accept some inter-node comm overhead.
- **HSDP is not strictly necessary** — it's the right *next* optimization,
  not the right *first* change.
- **Trip-wire to revisit:** 2-node throughput < 1.6× of 1-node, or
  cross-node collectives > 25% of step time.
- **Upgrade is ~10 lines** when the time comes; deferred deliberately.
