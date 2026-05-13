#!/usr/bin/env python3
"""Patch deepseek_v4.py for Ascend NPU: replace triton/tilelang with torch-native fallbacks."""
import sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/a00652497/m84379596/DFlash/sglang/python/sglang/srt/models/deepseek_v4.py"
)

with open(TARGET, "r") as f:
    code = f.read()

# === Patch 1: triton imports (lines 10-11) ===
code = code.replace(
    "import triton\nimport triton.language as tl",
    """try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False""",
)

# === Patch 2: apply_rotary_emb_triton import (line 28) ===
code = code.replace(
    "from sglang.srt.layers.deepseek_v4_rope import apply_rotary_emb_triton",
    """try:
    from sglang.srt.layers.deepseek_v4_rope import apply_rotary_emb_triton
except ImportError:
    def apply_rotary_emb_triton(x, freqs_cis, positions=None, inverse=False):
        \"\"\"Torch-native RoPE fallback for NPU (replaces tilelang kernel).\"\"\"
        is_3d = x.ndim == 3
        if not is_3d:
            x_work = x.unsqueeze(1)
        else:
            x_work = x
        batch_size = x_work.shape[0]
        freqs = freqs_cis[positions] if positions is not None else freqs_cis[:batch_size]
        while freqs.ndim < x_work.ndim:
            freqs = freqs.unsqueeze(1)
        if inverse:
            freqs = freqs.conj()
        x_c = torch.view_as_complex(x_work.float().reshape(*x_work.shape[:-1], -1, 2))
        rot = torch.view_as_real(x_c * freqs).flatten(-2)
        x.copy_(rot.reshape(x.shape).to(x.dtype))
        return x""",
)

# === Patch 3: triton RMSNorm kernel — wrap in HAS_TRITON guard ===
# Find the @triton.jit decorator and replace the whole kernel + wrapper
old_kernel = "@triton.jit\ndef _rms_normalize_kernel("
if old_kernel in code:
    # Find the start of the kernel
    kernel_start = code.index(old_kernel)
    # Find rms_normalize_triton function and its end (next def or class at same indent)
    wrapper_marker = "def rms_normalize_triton("
    wrapper_start = code.index(wrapper_marker, kernel_start)

    # Find the end of rms_normalize_triton by looking for next function/class at module level
    # Search for the next line starting with 'def ' or 'class ' after the wrapper
    import re
    rest = code[wrapper_start:]
    # Find end of function: next top-level def/class
    match = re.search(r'\n(?=def |class )', rest[1:])  # skip first char to avoid matching itself
    if match:
        wrapper_end = wrapper_start + 1 + match.start() + 1
    else:
        wrapper_end = len(code)

    old_block = code[kernel_start:wrapper_end]

    new_block = """if HAS_TRITON:
    @triton.jit
    def _rms_normalize_kernel(
        x_ptr, weight_ptr, eps, stride_row, dim,
        BLOCK_SIZE: tl.constexpr, HAS_WEIGHT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < dim
        base = pid * stride_row
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        mean_sq = tl.sum(x * x, axis=0) / dim
        rms_inv = tl.rsqrt(mean_sq + eps)
        out = x * rms_inv
        if HAS_WEIGHT:
            weight = tl.load(weight_ptr + offs, mask=mask, other=0.0)
            out = out * weight
        tl.store(x_ptr + base + offs, out, mask=mask)


def rms_normalize_triton(x: torch.Tensor, eps: float, weight: torch.Tensor = None) -> torch.Tensor:
    \"\"\"RMSNorm with triton kernel on GPU, torch fallback on NPU.\"\"\"
    if HAS_TRITON:
        dim = x.shape[-1]
        x_flat = x.view(-1, dim)
        num_rows = x_flat.shape[0]
        BLOCK_SIZE = triton.next_power_of_2(dim)
        grid = (num_rows,)
        _rms_normalize_kernel[grid](
            x_flat, weight, eps, x_flat.stride(0), dim,
            BLOCK_SIZE=BLOCK_SIZE, HAS_WEIGHT=(weight is not None),
        )
        return x
    else:
        # Torch-native fallback for NPU
        orig_dtype = x.dtype
        x_float = x.float()
        rms = torch.sqrt(torch.mean(x_float ** 2, dim=-1, keepdim=True) + eps)
        out = x_float / rms
        if weight is not None:
            out = out * weight.float()
        x.copy_(out.to(orig_dtype))
        return x

"""
    code = code[:kernel_start] + new_block + code[wrapper_end:]

with open(TARGET, "w") as f:
    f.write(code)

print(f"[OK] Patched {TARGET}")
print("  - triton imports: wrapped in try/except")
print("  - apply_rotary_emb_triton: torch-native fallback added")
print("  - rms_normalize_triton: torch-native fallback added")