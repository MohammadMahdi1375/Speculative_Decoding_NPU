#!/usr/bin/env python3
"""
create_mini_dsv4flash.py — Config-only sanity check for DeepSeek-V4-Flash DFlash training
=========================================================================================

Creates a *tiny* DeepSeek-V4-Flash model (random weights, reduced layers/experts)
so you can verify the training pipeline runs end-to-end without downloading ~150 GB
of safetensors.

Usage:
    # Downloads only config.json + tokenizer from HF (~few MB), creates mini model locally
    python create_mini_dsv4flash.py --output-dir ./mini_dsv4flash_sanity

    # Then point your training script at it:
    TARGET_MODEL=./mini_dsv4flash_sanity bash run_dsv4flash_dflash_npu_sglang_multinode.sh

What it does:
    1. Loads AutoConfig from deepseek-ai/DeepSeek-V4-Flash (just the JSON, not weights)
    2. Shrinks the config: fewer layers, fewer experts, smaller intermediate sizes
    3. Instantiates the model from config (random init) — small enough to fit on 1 device
    4. Saves model + tokenizer to --output-dir
    5. You get a directory that looks like a real HF checkpoint to SGLang

##### @Moh_7596 ###############
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Create a mini DeepSeek-V4-Flash for pipeline sanity check"
    )
    parser.add_argument(
        "--source-model",
        type=str,
        default="deepseek-ai/DeepSeek-V4-Flash",
        help="HF model ID to pull config/tokenizer from",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./mini_dsv4flash_sanity",
        help="Where to save the mini model",
    )
    # ---------- Shrink knobs ----------
    parser.add_argument("--num-hidden-layers", type=int, default=4,
                        help="Number of transformer layers (real model has 43)")
    parser.add_argument("--n-routed-experts", type=int, default=8,
                        help="Number of MoE routed experts (real model has 256)")
    parser.add_argument("--num-experts-per-tok", type=int, default=2,
                        help="Top-k experts per token (real model has 6)")
    parser.add_argument("--num-hash-layers", type=int, default=1,
                        help="Hash-routing layers (real model has 3, must be <= num_hidden_layers)")
    parser.add_argument("--num-nextn-predict-layers", type=int, default=0,
                        help="MTP layers (set 0 to skip for faster sanity check)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Model dtype (skip fp8 quant for sanity check)")
    args = parser.parse_args()

    # --- Step 1: Load config only (no weights) ---
    print(f"[INFO] Loading config + tokenizer from {args.source_model} ...")
    print("[INFO] This downloads only metadata (~few MB), NOT model weights.")

    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError:
        print("[ERROR] transformers not installed. Run: pip install transformers")
        sys.exit(1)

    config = AutoConfig.from_pretrained(args.source_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.source_model, trust_remote_code=True)

    # --- Step 2: Shrink the config ---
    print(f"\n[INFO] Original config:")
    print(f"  num_hidden_layers   = {config.num_hidden_layers}")
    print(f"  n_routed_experts    = {config.n_routed_experts}")
    print(f"  num_experts_per_tok = {config.num_experts_per_tok}")
    print(f"  num_hash_layers     = {config.num_hash_layers}")
    print(f"  hidden_size         = {config.hidden_size}")
    print(f"  vocab_size          = {config.vocab_size}")

    config.num_hidden_layers = args.num_hidden_layers
    config.n_routed_experts = args.n_routed_experts
    config.num_experts_per_tok = min(args.num_experts_per_tok, args.n_routed_experts)
    config.num_hash_layers = min(args.num_hash_layers, args.num_hidden_layers)
    config.num_nextn_predict_layers = args.num_nextn_predict_layers

    # Remove FP8 quantization config — we want clean bfloat16 for sanity check
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    config.torch_dtype = args.dtype

    # Shrink compress_ratios to match reduced layer count
    # Format: one ratio per layer pair, length = num_hidden_layers + 1 (including input)
    # For sanity check, just use 0s (no compression) for all layers
    if hasattr(config, "compress_ratios"):
        config.compress_ratios = [0] * (args.num_hidden_layers + 1)

    print(f"\n[INFO] Shrunk config:")
    print(f"  num_hidden_layers   = {config.num_hidden_layers}")
    print(f"  n_routed_experts    = {config.n_routed_experts}")
    print(f"  num_experts_per_tok = {config.num_experts_per_tok}")
    print(f"  num_hash_layers     = {config.num_hash_layers}")
    print(f"  num_nextn_predict_layers = {config.num_nextn_predict_layers}")
    print(f"  torch_dtype         = {config.torch_dtype}")
    print(f"  compress_ratios     = {config.compress_ratios}")

    # Estimate param count
    H = config.hidden_size                       # 4096
    V = config.vocab_size                        # 129280
    L = config.num_hidden_layers
    E = config.n_routed_experts
    E_int = config.moe_intermediate_size         # 2048
    # Rough: embed + L*(attn + E*ffn_expert + shared_expert + hc) + lm_head
    approx_params = (
        V * H                                    # embed
        + L * (4 * H * H)                        # attention (rough, ignoring lora)
        + L * E * (3 * H * E_int)                # MoE experts (gate+up+down)
        + L * (3 * H * E_int)                    # shared expert
        + V * H                                  # lm_head
    )
    dtype_bytes = {"bfloat16": 2, "float16": 2, "float32": 4}[args.dtype]
    approx_gb = approx_params * dtype_bytes / (1024**3)
    print(f"\n[INFO] Approx param count: {approx_params/1e6:.0f}M  ({approx_gb:.1f} GB in {args.dtype})")

    # --- Step 3: Instantiate from config (random weights) ---
    print("\n[INFO] Instantiating model from config (random init) ...")

    import torch
    from transformers import AutoModelForCausalLM

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=True,
    )

    num_params = sum(p.numel() for p in model.parameters())
    mem_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
    print(f"[INFO] Model instantiated: {num_params/1e6:.1f}M params, {mem_gb:.2f} GB")

    # --- Step 4: Save ---
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Saving to {output_path} ...")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    # Also save a marker file so you remember this is a sanity-check model
    marker = output_path / "SANITY_CHECK_MODEL.txt"
    marker.write_text(
        f"This is a MINI model for pipeline sanity checking.\n"
        f"Source: {args.source_model}\n"
        f"Layers: {args.num_hidden_layers} (original: 43)\n"
        f"Experts: {args.n_routed_experts} (original: 256)\n"
        f"Weights: random init (NOT pretrained)\n"
        f"DO NOT use for evaluation or production.\n"
    )

    print(f"\n[SUCCESS] Mini DeepSeek-V4-Flash saved to: {output_path}")
    print(f"[NEXT]    Set TARGET_MODEL={output_path} in your training script")
    print(f"[NEXT]    Or: TARGET_MODEL={output_path} bash run_dsv4flash_dflash_npu_sglang_multinode.sh")


if __name__ == "__main__":
    main()