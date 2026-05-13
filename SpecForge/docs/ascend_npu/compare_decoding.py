#!/usr/bin/env python3
"""
End-to-end decoding comparison for two trained DFlash draft checkpoints.

For speculative decoding, the draft model only needs to propose tokens that
the target model accepts often enough — exact equality of draft logits is
not required. The semantically correct test is:
  1. Use the SAME target model with two different draft checkpoints.
  2. Generate text with greedy decoding (temperature=0) on the same prompt.
  3. Compare the decoded strings.

With greedy decoding and the same target, both runs should produce
nearly-identical text (target's argmax is deterministic; draft proposals
that differ are caught by verification).

The acceptance rate per draft is also reported — it measures how often the
draft's proposed tokens matched the target's argmax. Two valid drafts
trained on the same task should have similar acceptance rates (within a
couple percent).

Usage:
    cd /home/$USER/2026/SpecForge
    python docs/ascend_npu/compare_decoding.py \\
        --ckpt-a outputs/.../single-node-checkpoint \\
        --ckpt-b outputs/.../multi-node-checkpoint \\
        --target-model /share/canada_group_folder/ckpt/Qwen3-8B
"""

import argparse
import sys
from typing import List

import torch
import torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer

from specforge.modeling.draft.dflash import DFlashDraftModel


# ----------------------------------------------------------------------------
# Monkey-patch spec_generate for NPU compatibility.
#
# Upstream specforge/modeling/draft/dflash.py:346 does:
#   (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1)
# The == produces a bool tensor; NPU's aclnnCumprod does not support bool
# (only float/int dtypes). On CUDA this works because cumprod silently
# accepts bool. On NPU it crashes:
#   AclNN_Parameter_Error: Tensor input not implemented for DT_BOOL
# Fix: cast bool->int before cumprod. This patch is upstream-PR worthy.
# ----------------------------------------------------------------------------
_ORIGINAL_SPEC_GENERATE = DFlashDraftModel.spec_generate


@torch.inference_mode()
def _spec_generate_npu_safe(self, target, input_ids, max_new_tokens,
                            stop_token_ids, temperature):
    """NPU-safe drop-in for DFlashDraftModel.spec_generate.

    Identical to the original method body except the one cumprod-on-bool
    line on dflash.py:346 has been changed to cast to int first.
    """
    from transformers import DynamicCache
    from specforge.modeling.draft.dflash import (
        sample,
        extract_context_feature,
    )

    self.eval()
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    block_size = self.block_size
    output_ids = torch.full(
        (1, max_length + block_size),
        self.mask_token_id,
        dtype=torch.long,
        device=target.device,
    )
    position_ids = torch.arange(
        output_ids.shape[1], device=target.device,
    ).unsqueeze(0)

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    # Prefill
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
        output.logits, temperature,
    )
    target_hidden = extract_context_feature(
        output.hidden_states, self.target_layer_ids,
    )

    acceptance_lengths = []
    start = input_ids.shape[1]
    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_logits = target.lm_head(
            self(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[
                    :, past_key_values_draft.get_seq_length() : start + block_size
                ],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, -block_size + 1 :, :]
        )
        past_key_values_draft.crop(start)
        block_output_ids[:, 1:] = sample(draft_logits)

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )

        posterior = sample(output.logits, temperature)
        # ---- NPU FIX: cast bool to int before cumprod ----
        acceptance_length = (
            (block_output_ids[:, 1:] == posterior[:, :-1])
            .int()                # <-- the only diff from upstream
            .cumprod(dim=1)
            .sum(dim=1)[0]
            .item()
        )
        # ---- end NPU FIX ----
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
            :, : acceptance_length + 1
        ]
        output_ids[:, start + acceptance_length + 1] = posterior[
            :, acceptance_length
        ]
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        target_hidden = extract_context_feature(
            output.hidden_states, self.target_layer_ids,
        )[:, : acceptance_length + 1, :]
        acceptance_lengths.append(acceptance_length + 1)
        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:]
            for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
    if stop_token_ids is not None and len(stop_token_ids) > 0:
        stop_ids_t = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(
            output_ids[0][num_input_tokens:], stop_ids_t,
        ).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[
                :, : num_input_tokens + stop_token_indices[0] + 1
            ]
    # Stash acceptance lengths on the instance for the caller to read.
    self._last_acceptance_lengths = acceptance_lengths
    return output_ids


DFlashDraftModel.spec_generate = _spec_generate_npu_safe
print("[npu-patch] DFlashDraftModel.spec_generate patched: bool.cumprod() -> int().cumprod()")


DEFAULT_PROMPTS = [
    "Explain in one sentence what speculative decoding is.",
    "List three differences between Python and Rust.",
    "Write a haiku about Ascend NPUs.",
]


def acceptance_stats(acc_lengths: List[int]) -> str:
    if not acc_lengths:
        return "no blocks generated"
    avg = sum(acc_lengths) / len(acc_lengths)
    return (f"blocks={len(acc_lengths)}  "
            f"avg accept length={avg:.2f}  "
            f"min/max={min(acc_lengths)}/{max(acc_lengths)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-a", required=True)
    parser.add_argument("--ckpt-b", required=True)
    parser.add_argument("--target-model", required=True,
                        help="HF target model path (Qwen3-8B etc.)")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 = greedy; same value used for both drafts")
    parser.add_argument("--prompts", nargs="*", default=None,
                        help="prompts to test; default = a built-in trio")
    parser.add_argument("--chat-template", action="store_true",
                        help="apply tokenizer's chat template (for instruct models)")
    args = parser.parse_args()

    device = args.device
    prompts = args.prompts or DEFAULT_PROMPTS

    print(f"=== Loading ===")
    print(f"  tokenizer + target: {args.target_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()

    print(f"  draft A : {args.ckpt_a}")
    draft_a = DFlashDraftModel.from_pretrained(args.ckpt_a, dtype=torch.bfloat16)
    draft_a = draft_a.to(device).eval()

    print(f"  draft B : {args.ckpt_b}")
    draft_b = DFlashDraftModel.from_pretrained(args.ckpt_b, dtype=torch.bfloat16)
    draft_b = draft_b.to(device).eval()

    # spec_generate stores acceptance lengths; we'll inspect via a wrapper
    # that captures them. Simplest: monkey-patch the local list inside the
    # generated method via re-running and inspecting the returned ids.
    # (spec_generate appends to `acceptance_lengths` internally; it doesn't
    # return them. To capture them, we patch the method.)

    eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    pass_count, soft_count, fail_count = 0, 0, 0

    for i, prompt in enumerate(prompts):
        print(f"\n=========== Prompt {i+1}/{len(prompts)} ===========")
        print(f"  prompt: {prompt!r}")

        if args.chat_template and tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            text = prompt

        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

        # Generate with each draft. spec_generate is @torch.inference_mode
        # so we don't need a no_grad context.
        out_a = draft_a.spec_generate(
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=eos_ids,
            temperature=args.temperature,
        )
        text_a = tokenizer.decode(
            out_a[0, input_ids.shape[1]:].tolist(), skip_special_tokens=True,
        )
        acc_a = getattr(draft_a, "_last_acceptance_lengths", [])

        out_b = draft_b.spec_generate(
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=eos_ids,
            temperature=args.temperature,
        )
        text_b = tokenizer.decode(
            out_b[0, input_ids.shape[1]:].tolist(), skip_special_tokens=True,
        )
        acc_b = getattr(draft_b, "_last_acceptance_lengths", [])

        print(f"\n  --- ckpt A output ---\n  {text_a!r}")
        print(f"  acceptance: {acceptance_stats(acc_a)}")
        print(f"\n  --- ckpt B output ---\n  {text_b!r}")
        print(f"  acceptance: {acceptance_stats(acc_b)}")

        # Compare token sequences (excluding the prompt prefix)
        gen_a = out_a[0, input_ids.shape[1]:].tolist()
        gen_b = out_b[0, input_ids.shape[1]:].tolist()
        common_len = min(len(gen_a), len(gen_b))
        first_diff = next(
            (i for i in range(common_len) if gen_a[i] != gen_b[i]), None,
        )
        n_match = first_diff if first_diff is not None else common_len
        match_pct = 100 * n_match / max(common_len, 1)

        print(f"\n  --- diff ---")
        print(f"  tokens generated: A={len(gen_a)}  B={len(gen_b)}")
        print(f"  matching prefix : {n_match} tokens ({match_pct:.1f}%)")
        if first_diff is not None:
            ctx = tokenizer.decode(gen_a[max(0, first_diff-5):first_diff], skip_special_tokens=True)
            tok_a = tokenizer.decode([gen_a[first_diff]], skip_special_tokens=True)
            tok_b = tokenizer.decode([gen_b[first_diff]], skip_special_tokens=True)
            print(f"  first diff at pos {first_diff}: after '...{ctx}'  A→{tok_a!r}  B→{tok_b!r}")

        # Verdict for this prompt
        if text_a.strip() == text_b.strip():
            print(f"  [PASS] outputs identical")
            pass_count += 1
        elif match_pct >= 80:
            print(f"  [SOFT-PASS] {match_pct:.0f}% prefix matches; small late divergence")
            soft_count += 1
        else:
            print(f"  [FAIL] outputs diverge early")
            fail_count += 1

    print(f"\n=========== Overall ===========")
    print(f"  PASS       : {pass_count}/{len(prompts)}")
    print(f"  SOFT-PASS  : {soft_count}/{len(prompts)}")
    print(f"  FAIL       : {fail_count}/{len(prompts)}")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
