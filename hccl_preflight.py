"""
Multinode HCCL smoke test for Ascend NPU.

Run on both nodes simultaneously with torchrun. Performs an all-reduce on a tiny
tensor across all ranks; if it completes, inter-node HCCL is working.

Usage:
  On master (e.g. 108):
    torchrun --nnodes=2 --nproc_per_node=8 --node-rank=0 \\
             --master-addr=80.5.5.108 --master-port=29500 hccl_preflight.py

  On worker (e.g. 109):
    torchrun --nnodes=2 --nproc_per_node=8 --node-rank=1 \\
             --master-addr=80.5.5.108 --master-port=29500 hccl_preflight.py

Set HCCL_SOCKET_IFNAME=<nic> on each node (it can differ between nodes).

Expected output on each rank:
  [rank N/16] PASS — sum=136.0 (expected 136.0)
"""
import os
import socket
import torch
import torch_npu  # noqa: F401 (registers npu backend)
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)

    host = socket.gethostname()
    print(f"[rank {rank}/{world_size}] init OK on {host} local_rank={local_rank}", flush=True)

    # All-reduce a tiny tensor: each rank contributes rank+1; sum = W*(W+1)/2
    x = torch.tensor([float(rank + 1)], device=f"npu:{local_rank}")
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size + 1) / 2
    actual = x.item()

    if abs(actual - expected) < 1e-3:
        print(f"[rank {rank}/{world_size}] PASS — sum={actual} (expected {expected})", flush=True)
    else:
        print(f"[rank {rank}/{world_size}] FAIL — sum={actual} expected={expected}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()