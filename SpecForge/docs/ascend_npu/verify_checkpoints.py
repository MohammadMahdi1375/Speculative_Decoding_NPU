#!/usr/bin/env python3
"""
Verify two trained DFlash draft-model checkpoints produce similar outputs.

Use case: confirm that a multi-node training run learned the same model as
a reference single-node run. Two checkpoints from different runs of the
same task will NOT be bit-identical (different data-shuffle seeds, FSDP
accumulation order under different world_size, etc.), but their forward
outputs on identical input should be very close (cosine sim > 0.95).

Usage:
    cd /home/$USER/2026/SpecForge
    python docs/ascend_npu/verify_checkpoints.py \\
        --ckpt-a outputs/single-node/epoch_<N>_step_<M> \\
        --ckpt-b outputs/multi-node/epoch_<N>_step_<M>

Both --ckpt-a and --ckpt-b should be checkpoint *directories* (the ones
containing config.json + the bin/safetensors weights), not the parent
output directory.
"""

import argparse
import sys

import torch
import torch_npu  # noqa: F401 — also pulls transfer_to_npu via sgl_kernel_npu

from specforge.modeling.draft.dflash import DFlashDraftModel


def fmt(x: float) -> str:
    return f"{x:.6f}" if abs(x) >= 1e-6 else f"{x:.2e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-a", required=True, help="checkpoint dir A (e.g. single-node)")
    parser.add_argument("--ckpt-b", required=True, help="checkpoint dir B (e.g. multi-node)")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=64,
                        help="synthetic input length (must be multiple of block_size)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cosine-threshold", type=float, default=0.95,
                        help="cosine similarity threshold to count as PASS")
    args = parser.parse_args()

    device = args.device

    # --- Load both checkpoints ---
    print(f"=== Loading ===")
    print(f"  ckpt A: {args.ckpt_a}")
    model_a = DFlashDraftModel.from_pretrained(args.ckpt_a, dtype=torch.bfloat16)
    model_a = model_a.to(device).eval()
    print(f"  ckpt B: {args.ckpt_b}")
    model_b = DFlashDraftModel.from_pretrained(args.ckpt_b, dtype=torch.bfloat16)
    model_b = model_b.to(device).eval()

    # --- Sanity: configs should match ---
    cfg_a, cfg_b = model_a.config, model_b.config
    if cfg_a.hidden_size != cfg_b.hidden_size or cfg_a.num_hidden_layers != cfg_b.num_hidden_layers:
        print("[FAIL] checkpoints have different model configs", file=sys.stderr)
        sys.exit(2)

    # --- Layer-by-layer weight comparison (in fp32 for accurate diff) ---
    print(f"\n=== Weight diff (per-tensor max-abs) ===")
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    keys_a, keys_b = set(sd_a), set(sd_b)
    if keys_a != keys_b:
        only_a = sorted(keys_a - keys_b)
        only_b = sorted(keys_b - keys_a)
        if only_a:
            print(f"  keys only in A: {only_a[:5]}{' ...' if len(only_a) > 5 else ''}")
        if only_b:
            print(f"  keys only in B: {only_b[:5]}{' ...' if len(only_b) > 5 else ''}")
    common = sorted(keys_a & keys_b)
    diffs = []
    for k in common:
        ta, tb = sd_a[k].float().cpu(), sd_b[k].float().cpu()
        if ta.shape != tb.shape:
            print(f"  shape mismatch: {k}: A={tuple(ta.shape)} B={tuple(tb.shape)}")
            continue
        diff = (ta - tb).abs().max().item()
        diffs.append((k, diff))
    diffs.sort(key=lambda kv: -kv[1])
    print(f"  max abs diff overall : {fmt(diffs[0][1])}  (in '{diffs[0][0]}')")
    print(f"  mean of per-tensor max-abs : {fmt(sum(d for _, d in diffs) / max(len(diffs), 1))}")
    print(f"  top 5 most divergent tensors:")
    for k, d in diffs[:5]:
        print(f"    {fmt(d):>10}  {k}")

    # --- Forward on identical synthetic input ---
    # DFlash forward shapes (from core/dflash.py OnlineDFlashModel):
    #   target_hidden    : [bsz, ctx_len, hidden * num_target_layers]
    #   noise_embedding  : [bsz, q_len, hidden]
    #   position_ids     : [bsz, ctx_len + q_len]   <-- positions for BOTH ctx and query
    # Inside Qwen3DFlashAttention, k = cat([k_ctx, k_noise]) so its length is
    # ctx_len+q_len; cos/sin is generated from position_ids and must match.
    bsz = args.batch_size
    block_size = cfg_a.block_size
    ctx_len = args.seq_len   # context length (target hidden states)
    q_len = block_size       # one anchor block worth of query positions
    hidden = cfg_a.hidden_size
    num_target_layers = len(model_a.target_layer_ids)

    print(f"\n=== Forward diff (synthetic input) ===")
    print(f"  bsz={bsz}  ctx_len={ctx_len}  q_len(block_size)={q_len}  "
          f"hidden={hidden}  num_target_layers={num_target_layers}")

    torch.manual_seed(args.seed)
    noise_embedding = torch.randn(bsz, q_len, hidden, dtype=torch.bfloat16, device=device)
    target_hidden = torch.randn(bsz, ctx_len, hidden * num_target_layers,
                                dtype=torch.bfloat16, device=device)
    # position_ids covers both context (0..ctx_len-1) and query (ctx_len..ctx_len+q_len-1)
    position_ids = torch.arange(ctx_len + q_len, device=device).unsqueeze(0).expand(bsz, -1)

    with torch.no_grad():
        out_a = model_a(position_ids=position_ids, noise_embedding=noise_embedding,
                         target_hidden=target_hidden)
        out_b = model_b(position_ids=position_ids, noise_embedding=noise_embedding,
                         target_hidden=target_hidden)

    out_a32, out_b32 = out_a.float(), out_b.float()
    diff = (out_a32 - out_b32).abs()
    cos = torch.nn.functional.cosine_similarity(
        out_a32.flatten().unsqueeze(0), out_b32.flatten().unsqueeze(0), dim=1
    ).item()
    rel = (diff / (out_a32.abs() + 1e-6))

    print(f"  output shape       : {tuple(out_a.shape)}")
    print(f"  output magnitude   : A.abs.mean={fmt(out_a32.abs().mean().item())}  "
          f"B.abs.mean={fmt(out_b32.abs().mean().item())}")
    print(f"  max abs diff       : {fmt(diff.max().item())}")
    print(f"  mean abs diff      : {fmt(diff.mean().item())}")
    print(f"  max relative diff  : {fmt(rel.max().item())}")
    print(f"  cosine similarity  : {fmt(cos)}  (flattened)")

    # --- Verdict ---
    print(f"\n=== Verdict ===")
    if cos >= args.cosine_threshold:
        print(f"  [PASS] cosine sim {cos:.4f} >= threshold {args.cosine_threshold}")
        print(f"         → Both checkpoints learned the same task (multi-node ≈ single-node).")
    elif cos >= 0.5:
        print(f"  [SOFT-FAIL] cosine sim {cos:.4f} below threshold {args.cosine_threshold}")
        print(f"         → Models are correlated but not as close as expected.")
        print(f"         → Possible causes: different #training steps, very different LR")
        print(f"           schedule, or one run was not fully converged.")
    else:
        print(f"  [FAIL] cosine sim {cos:.4f} < 0.5")
        print(f"         → Models are essentially unrelated. Either the wrong checkpoint")
        print(f"           is loaded, or one of the two training runs diverged / was reset.")
        sys.exit(1)


if __name__ == "__main__":
    main()
