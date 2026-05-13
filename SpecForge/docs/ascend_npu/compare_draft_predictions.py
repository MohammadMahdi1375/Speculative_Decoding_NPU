#!/usr/bin/env python3
"""
Direct draft-prediction comparison for two trained DFlash checkpoints.

Why this script exists:
  compare_decoding.py runs spec_generate end-to-end. Under greedy decoding
  with the same target, both drafts produce *identical* text — but only
  because target verification masks any draft difference. Text equality is
  a tautology, not a draft-quality signal.

What this script does:
  1. Tokenize a prompt; run target prefill once to get target_hidden.
  2. Build a single block of noise input (1 anchor token + block_size-1
     MASK tokens) — same input fed to both drafts.
  3. Forward each draft, apply target.lm_head, take argmax.
  4. Compare: how many of the block_size positions agree between the two
     drafts? What's the top-5 overlap? Cosine similarity of logits?

This isolates draft-vs-draft difference from target verification — the
direct semantic test of whether two drafts produce the same predictions.

Usage:
    python docs/ascend_npu/compare_draft_predictions.py \\
        --ckpt-a outputs/.../single-node \\
        --ckpt-b outputs/.../multi-node \\
        --target-model /share/canada_group_folder/ckpt/Qwen3-8B
"""

import argparse
import sys

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer

from specforge.modeling.draft.dflash import DFlashDraftModel, extract_context_feature


DEFAULT_PROMPTS = [
    "Explain in one sentence what speculative decoding is.",
    "List three differences between Python and Rust.",
    "Write a haiku about Ascend NPUs.",
]


def fmt_pct(x: float) -> str:
    return f"{x*100:5.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-a", required=True)
    parser.add_argument("--ckpt-b", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--top-k", type=int, default=5,
                        help="top-k overlap to report")
    parser.add_argument("--exact-match-threshold", type=float, default=0.7,
                        help="argmax match rate required to PASS")
    args = parser.parse_args()

    device = args.device
    prompts = args.prompts or DEFAULT_PROMPTS
    K = args.top_k

    print("=== Loading ===")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    draft_a = DFlashDraftModel.from_pretrained(args.ckpt_a, dtype=torch.bfloat16).to(device).eval()
    draft_b = DFlashDraftModel.from_pretrained(args.ckpt_b, dtype=torch.bfloat16).to(device).eval()

    block_size = draft_a.config.block_size
    target_layer_ids = draft_a.target_layer_ids
    assert target_layer_ids == draft_b.target_layer_ids, \
        "two drafts have different target_layer_ids — not comparable"

    pass_count = soft_count = fail_count = 0

    for i, prompt in enumerate(prompts):
        print(f"\n========= Prompt {i+1}/{len(prompts)} =========")
        print(f"  prompt: {prompt!r}")

        if args.chat_template and tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            text = prompt
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        ctx_len = input_ids.shape[1]

        with torch.inference_mode():
            # Target prefill
            target_out = target(
                input_ids, output_hidden_states=True, use_cache=False,
            )
            target_hidden = extract_context_feature(
                target_out.hidden_states, target_layer_ids,
            )  # [1, ctx_len, hidden * num_target_layers]

            # Build the anchor token from target's argmax at the last position
            anchor = target_out.logits[:, -1:, :].argmax(dim=-1)  # [1, 1]

            # Block input: anchor + (block_size-1) MASK tokens
            mask_id = draft_a.mask_token_id
            mask_tokens = torch.full(
                (1, block_size - 1), mask_id, dtype=torch.long, device=device,
            )
            block_input_ids = torch.cat([anchor, mask_tokens], dim=1)  # [1, block_size]
            noise_embedding = target.model.embed_tokens(block_input_ids)

            # Position ids: ctx + block (DFlash forward expects both)
            position_ids = torch.arange(
                ctx_len + block_size, device=device,
            ).unsqueeze(0)  # [1, ctx_len+block_size]

            # Forward each draft
            def predict(draft):
                hidden = draft(
                    position_ids=position_ids,
                    noise_embedding=noise_embedding,
                    target_hidden=target_hidden,
                    use_cache=False,
                )  # [1, block_size, hidden]
                logits = target.lm_head(hidden)  # [1, block_size, vocab]
                return logits

            logits_a = predict(draft_a)  # [1, block_size, vocab]
            logits_b = predict(draft_b)

            argmax_a = logits_a.argmax(dim=-1)  # [1, block_size]
            argmax_b = logits_b.argmax(dim=-1)
            topk_a = logits_a.topk(K, dim=-1).indices  # [1, block_size, K]
            topk_b = logits_b.topk(K, dim=-1).indices

        # Per-position exact match
        match = (argmax_a == argmax_b)              # [1, block_size]
        exact_rate = match.float().mean().item()

        # Per-position top-K overlap (counts a position as "overlap" if any
        # of A's top-K appear in B's top-K)
        # topk_a [1,B,K]  topk_b [1,B,K]
        # broadcast: [1,B,K,1] vs [1,B,1,K] -> any over last 2 dims
        overlap = (
            (topk_a.unsqueeze(-1) == topk_b.unsqueeze(-2))
            .any(dim=-1).any(dim=-1)
        )  # [1, block_size]
        topk_rate = overlap.float().mean().item()

        # Logits cosine sim (flattened over [block_size, vocab])
        logits_a32 = logits_a.float().flatten().unsqueeze(0)
        logits_b32 = logits_b.float().flatten().unsqueeze(0)
        cos = torch.nn.functional.cosine_similarity(
            logits_a32, logits_b32, dim=1,
        ).item()

        # Decode predictions for human reading
        def decode_seq(seq):
            return tokenizer.decode(seq[0].tolist(), skip_special_tokens=False)
        text_a = decode_seq(argmax_a)
        text_b = decode_seq(argmax_b)

        print(f"  block_size                : {block_size}")
        print(f"  argmax exact-match rate   : {fmt_pct(exact_rate)} "
              f"({int(exact_rate*block_size)}/{block_size})")
        print(f"  top-{K} overlap rate         : {fmt_pct(topk_rate)}")
        print(f"  logits cosine similarity  : {cos:.4f}")
        print(f"  --- draft A predictions ---")
        print(f"  {text_a!r}")
        print(f"  --- draft B predictions ---")
        print(f"  {text_b!r}")
        # Show where they disagree
        diffs = (~match[0]).nonzero(as_tuple=True)[0].tolist()
        if diffs:
            print(f"  positions where argmax differs: {diffs}")
            for p in diffs[:3]:
                tok_a = tokenizer.decode([argmax_a[0, p].item()],
                                          skip_special_tokens=False)
                tok_b = tokenizer.decode([argmax_b[0, p].item()],
                                          skip_special_tokens=False)
                print(f"    pos {p}: A→{tok_a!r}  B→{tok_b!r}")

        # Verdict for this prompt
        if exact_rate >= args.exact_match_threshold:
            print(f"  [PASS] argmax agreement {fmt_pct(exact_rate)} "
                  f">= threshold {fmt_pct(args.exact_match_threshold)}")
            pass_count += 1
        elif exact_rate >= 0.4 or topk_rate >= 0.85:
            print(f"  [SOFT-PASS] argmax {fmt_pct(exact_rate)} below "
                  f"threshold but top-{K} overlap {fmt_pct(topk_rate)} is "
                  f"strong → drafts agree on candidate set, disagree on "
                  f"order")
            soft_count += 1
        else:
            print(f"  [FAIL] argmax {fmt_pct(exact_rate)} too low and "
                  f"top-{K} overlap {fmt_pct(topk_rate)} also weak")
            fail_count += 1

    print(f"\n========= Overall =========")
    print(f"  PASS       : {pass_count}/{len(prompts)}")
    print(f"  SOFT-PASS  : {soft_count}/{len(prompts)}")
    print(f"  FAIL       : {fail_count}/{len(prompts)}")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
