#!/usr/bin/env python3
"""
prepare_dsv4_flash_config_only.py
=================================
Download *only* the config + tokenizer from deepseek-ai/DeepSeek-V4-Flash
(~3 MB total, no model weights) so you can smoke-test the SpecForge DFlash
training pipeline before committing to a full 284 B-param download.

Usage:
    python prepare_dsv4_flash_config_only.py \
        --output-dir /share/canada_group_folder/ckpt/DeepSeek-V4-Flash-config-only

Then point the launch script at that directory:
    TARGET_MODEL=/share/canada_group_folder/ckpt/DeepSeek-V4-Flash-config-only \
    LOAD_FORMAT=dummy \
    bash docs/ascend_npu/run_dsv4_flash_dflash_npu_sglang.sh
"""

import argparse
import os
import json

def main():
    parser = argparse.ArgumentParser(
        description="Download config+tokenizer only from DeepSeek-V4-Flash"
    )
    parser.add_argument(
        "--repo-id",
        default="deepseek-ai/DeepSeek-V4-Flash",
        help="HuggingFace repo ID",
    )
    parser.add_argument(
        "--output-dir",
        default="./DeepSeek-V4-Flash-config-only",
        help="Local directory to store config + tokenizer files",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch / revision to pull from",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- Strategy 1: huggingface_hub (preferred) ----------
    try:
        from huggingface_hub import hf_hub_download

        # Files we need — config + all tokenizer variants DeepSeek-V4 ships
        files_to_download = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
            # DeepSeek-V4 uses a custom encoding folder — grab it too
            # (the encoding_dsv4/ Python scripts).  If absent the download
            # just silently skips.
        ]

        for fname in files_to_download:
            try:
                local_path = hf_hub_download(
                    repo_id=args.repo_id,
                    filename=fname,
                    revision=args.revision,
                    local_dir=args.output_dir,
                    local_dir_use_symlinks=False,
                )
                print(f"  ✓ {fname}  →  {local_path}")
            except Exception as e:
                # Some files may not exist (e.g. generation_config.json)
                print(f"  ⚠ {fname} skipped: {e}")

        # Also try to get the encoding_dsv4 folder (DeepSeek V4 custom encoder)
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=args.repo_id,
                revision=args.revision,
                local_dir=args.output_dir,
                local_dir_use_symlinks=False,
                allow_patterns=["encoding_dsv4/*"],
            )
            print("  ✓ encoding_dsv4/  (custom encoder)")
        except Exception:
            print("  ⚠ encoding_dsv4/ not downloaded (may not exist or requires auth)")

    except ImportError:
        print("huggingface_hub not installed, falling back to manual download...")
        _fallback_download(args)

    # ---------- Verify essential files ----------
    config_path = os.path.join(args.output_dir, "config.json")
    if not os.path.isfile(config_path):
        print(f"\n✗ FAILED: {config_path} not found. Check your auth / network.")
        return 1

    with open(config_path) as f:
        cfg = json.load(f)

    print(f"\n{'='*60}")
    print(f"Config-only directory ready: {os.path.abspath(args.output_dir)}")
    print(f"  model_type          : {cfg.get('model_type', '???')}")
    print(f"  hidden_size         : {cfg.get('hidden_size', '???')}")
    print(f"  num_hidden_layers   : {cfg.get('num_hidden_layers', '???')}")
    print(f"  num_attention_heads : {cfg.get('num_attention_heads', '???')}")
    print(f"  n_routed_experts    : {cfg.get('n_routed_experts', '???')}")
    print(f"  vocab_size          : {cfg.get('vocab_size', '???')}")
    print(f"{'='*60}")
    print()
    print("Next step: launch training with LOAD_FORMAT=dummy to skip weight loading:")
    print(f"  TARGET_MODEL={os.path.abspath(args.output_dir)} \\")
    print(f"  LOAD_FORMAT=dummy \\")
    print(f"  bash docs/ascend_npu/run_dsv4_flash_dflash_npu_sglang.sh")

    return 0


def _fallback_download(args):
    """Bare urllib fallback when huggingface_hub is not installed."""
    import urllib.request

    base_url = (
        f"https://huggingface.co/{args.repo_id}/resolve/{args.revision}"
    )
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json"]:
        url = f"{base_url}/{fname}"
        dst = os.path.join(args.output_dir, fname)
        try:
            urllib.request.urlretrieve(url, dst)
            print(f"  ✓ {fname}")
        except Exception as e:
            print(f"  ⚠ {fname} failed: {e}")


if __name__ == "__main__":
    raise SystemExit(main())