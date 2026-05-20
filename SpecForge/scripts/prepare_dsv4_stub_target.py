#!/usr/bin/env python3
"""
Prepare a small, random-weight DeepSeek-V4-Flash stub for SpecForge DFlash
on Ascend NPU.

This script uses transformers' NATIVE DeepSeek-V4 support (added 2026-05-02,
shipped in transformers v5.0+). We DO NOT hand-code V4's tensor schema —
the library handles all V4 architectural pieces (CSA, HCA, mHC,
hash_moe, grouped output projection, attention sinks, lightning indexer,
MTP, etc.). We just shrink the dims and let transformers instantiate the
real V4 architecture, then save it via the standard save_pretrained() path.

Pre-flight check
----------------
Your env must have transformers with V4 support. Verify with:

    python -c "from transformers import DeepseekV4Config; print('OK')"

If that fails, upgrade:

    pip install --upgrade "transformers>=5.0"

Then re-check that SGLang still loads — SGLang versions pinned to
transformers 4.x will need a compatible upgrade too.

Usage
-----
    python scripts/prepare_dsv4_stub_target.py \
        --output-dir ./stubs/DeepSeek-V4-Flash-Stub-SGLang

Generates safetensors automatically. No --save-stub-weights flag —
weights are always saved (that's the point).
"""

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def check_transformers_v4_support():
    try:
        from transformers import DeepseekV4Config  # noqa: F401
    except ImportError as e:
        print(
            "ERROR: This transformers install does not have DeepSeek-V4 "
            "support.\n"
            f"  ({e})\n"
            "  Upgrade with:  pip install --upgrade \"transformers>=5.0\"\n"
            "  Then check:    python -c \"from transformers import "
            "DeepseekV4Config; print('OK')\"",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="deepseek-ai/DeepSeek-V4-Flash")
    p.add_argument("--output-dir", default="./stubs/DeepSeek-V4-Flash-Stub")
    p.add_argument(
        "--draft-config-out",
        default="./configs/deepseek-v4-flash-dflash.json",
    )

    # Core dims (real V4-Flash values shown for context)
    p.add_argument("--hidden-size", type=int, default=1024)              # real: 4096
    p.add_argument("--moe-intermediate-size", type=int, default=512)     # real: 2048
    p.add_argument("--num-hidden-layers", type=int, default=4)           # real: 43
    p.add_argument("--num-attention-heads", type=int, default=8)         # real: 64
    p.add_argument(
        "--num-key-value-heads", type=int, default=1,
        help="V4 is MQA: keep at 1.",
    )                                                                     # real: 1
    p.add_argument("--head-dim", type=int, default=128)                  # real: 512
    p.add_argument("--n-routed-experts", type=int, default=8)            # real: 256
    p.add_argument("--n-shared-experts", type=int, default=1)            # real: 1
    p.add_argument("--num-experts-per-tok", type=int, default=2)         # real: 6
    p.add_argument("--max-position-embeddings", type=int, default=8192)   # real: 1M

    # V4-specific dims
    p.add_argument("--q-lora-rank", type=int, default=256)               # real: 1024
    p.add_argument("--qk-rope-head-dim", type=int, default=32)
    p.add_argument("--o-lora-rank", type=int, default=256)               # real: 1024
    p.add_argument("--o-groups", type=int, default=2)                    # real: 8
    p.add_argument("--hc-mult", type=int, default=4)                     # real: 4 (keep)
    p.add_argument("--index-n-heads", type=int, default=4)               # real: 64
    p.add_argument("--index-head-dim", type=int, default=64)             # real: 128
    p.add_argument("--index-topk", type=int, default=32)                 # real: 512
    p.add_argument("--sliding-window", type=int, default=64)             # real: 128
    p.add_argument("--num-hash-layers", type=int, default=1)             # real: 3
    p.add_argument(
        "--num-nextn-predict-layers", type=int, default=1,
        help="MTP layer count. transformers doc says this is "
             "checkpoint metadata, not instantiated.",
    )

    p.add_argument(
        "--target-dtype", default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument("--max-shard-size-gb", type=float, default=5.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    check_transformers_v4_support()

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM

    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Step 1: pull config + tokenizer (no modeling .py needed — V4 is
    # native in transformers).
    print(f"[1/4] Downloading config + tokenizer from {args.repo_id}")
    print(f"      -> {out_dir}")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(out_dir),
        allow_patterns=[
            "config.json",
            "tokenizer*",
            "special_tokens_map.json",
            "generation_config.json",
            "*.tiktoken",
        ],
    )

    # ----- Step 2: shrink the config. Keep ALL V4 architectural fields, just
    # smaller. Pick a diverse layer_types so the stub exercises all three
    # attention paths (sliding, CSA, HCA) and both MLP types (hash_moe, moe).
    cfg_path = out_dir / "config.json"
    print(f"[2/4] Shrinking config at {cfg_path}")
    cfg = json.loads(cfg_path.read_text())

    shrink = {
        "hidden_size": args.hidden_size,
        "moe_intermediate_size": args.moe_intermediate_size,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "head_dim": args.head_dim,
        "n_routed_experts": args.n_routed_experts,
        "n_shared_experts": args.n_shared_experts,
        "num_experts_per_tok": args.num_experts_per_tok,
        "max_position_embeddings": args.max_position_embeddings,
        "q_lora_rank": args.q_lora_rank,
        "qk_rope_head_dim": args.qk_rope_head_dim,
        "o_lora_rank": args.o_lora_rank,
        "o_groups": args.o_groups,
        "hc_mult": args.hc_mult,
        "index_n_heads": args.index_n_heads,
        "index_head_dim": args.index_head_dim,
        "index_topk": args.index_topk,
        "sliding_window": args.sliding_window,
        "num_hash_layers": args.num_hash_layers,
        "num_nextn_predict_layers": args.num_nextn_predict_layers,
    }
    for k, v in shrink.items():
        if k in cfg:
            cfg[k] = v

    # `compress_ratios` is V4-Flash's per-layer attention schedule.
    # 0 = sliding-window full attention; 4 = CSA (m=4); 128 = HCA (m'=128).
    # Pick a varied schedule so all three attention types get exercised.
    if "compress_ratios" in cfg:
        L = args.num_hidden_layers
        if L >= 4:
            # First/last sliding, alternating CSA/HCA in middle (matches
            # the real V4-Flash pattern in miniature).
            schedule = [0] + [4 if i % 2 == 0 else 128 for i in range(L - 2)] + [0]
        elif L >= 3:
            schedule = [0, 4, 128]
        elif L >= 2:
            schedule = [0, 4]
        else:
            schedule = [0]
        cfg["compress_ratios"] = schedule
        print(f"      compress_ratios -> {schedule}")

    # Strip FP8 quantization metadata — we want plain BF16 random init.
    for q_key in ("quantization_config", "quant_config"):
        if q_key in cfg:
            print(f"      removing {q_key}")
            cfg.pop(q_key)

    cfg["torch_dtype"] = args.target_dtype
    cfg_path.write_text(json.dumps(cfg, indent=2))

    # ----- Step 3: write SpecForge draft config (matches target dims).
    draft_cfg_path = Path(args.draft_config_out).resolve()
    draft_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[3/4] Writing draft config -> {draft_cfg_path}")
    draft_cfg = {
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "hidden_size": args.hidden_size,
        "vocab_size": cfg.get("vocab_size", 129280),
        "intermediate_size": args.moe_intermediate_size * 4,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": max(args.num_key_value_heads, 1),
        "head_dim": args.head_dim,
        "max_position_embeddings": args.max_position_embeddings,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "num_hidden_layers": 1,
        "num_target_layers": args.num_hidden_layers,
        "block_size": 16,
        "layer_types": ["full_attention"],
        "dflash_config": {},
    }
    draft_cfg_path.write_text(json.dumps(draft_cfg, indent=2))

    # ----- Step 4: instantiate the model via transformers' native V4 and
    # save it. CPU only — random init doesn't need NPU.
    print(f"[4/4] Building shrunken V4 with transformers native support")
    print(f"      (CPU random init — this is the slow step)")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.target_dtype]

    # Re-read the shrunk config so transformers parses ALL V4 fields
    # (compress_ratios -> layer_types, num_hash_layers -> mlp_layer_types,
    # etc. — V4Config does the BC conversion in __post_init__).
    config = AutoConfig.from_pretrained(out_dir, trust_remote_code=False)

    # Sanity-print what V4 parsed for layer schedules.
    if hasattr(config, "layer_types"):
        print(f"      layer_types     : {getattr(config, 'layer_types', None)}")
    if hasattr(config, "mlp_layer_types"):
        print(f"      mlp_layer_types : {getattr(config, 'mlp_layer_types', None)}")

    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=dtype,
        trust_remote_code=False,
    )
    nparams = sum(p.numel() for p in model.parameters())
    print(f"      stub parameter count: {nparams:,} "
          f"(~{nparams * 2 / 1e9:.2f} GB at bf16)")

    print(f"      saving safetensors -> {out_dir}")
    model.save_pretrained(
        out_dir,
        safe_serialization=True,
        max_shard_size=f"{int(args.max_shard_size_gb)}GB",
    )
    del model

    print()
    print("Done. The stub now contains a fully-instantiated DeepSeek-V4")
    print("architecture in BF16 random weights, with:")
    print("  - mHC hyper-connections")
    print("  - All three attention types (sliding / CSA / HCA)")
    print("  - hash_moe + moe routing")
    print("  - Grouped output projection, attention sinks, lightning indexer")
    print("  - MTP head (per checkpoint convention)")
    print()
    print("Next:")
    print(f"  TARGET_MODEL={out_dir} \\")
    print(f"  DRAFT_CONFIG={draft_cfg_path} \\")
    print( "  bash docs/ascend_npu/run_deepseek_v4_flash_dflash_sglang_npu.sh")


if __name__ == "__main__":
    main()