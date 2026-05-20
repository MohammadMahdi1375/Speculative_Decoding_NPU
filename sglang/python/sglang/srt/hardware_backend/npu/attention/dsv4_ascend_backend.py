# Copyright (c) 2026 — DeepseekV4 NPU attention backend port.
##### @Moh_7596 — V4 attention backend for Ascend NPU.
# Subclasses AscendAttnBackend (NPU baseline) and overrides V4 paths.
# Uses sgl-kernel-npu's npu_sparse_flash_attention + npu_lightning_indexer
# (registered under torch.ops.npu.* after `import attentions`).
#
# Status: skeleton — methods raise NotImplementedError with clear session markers.
# Reference: docs/dsv4_npu_port/refs/cuda_forward_method.py (CUDA oracle)
#            docs/dsv4_npu_port/refs/vllm_ascend_attention/sfa_v1.py (NPU patterns)
################
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Optional

import torch

# Register NPU ops (idempotent imports — safe to repeat)
try:
    import torch_npu  # noqa: F401
    import attentions  # noqa: F401 — registers npu_sparse_flash_attention etc.
    _NPU_OPS_AVAILABLE = True
except ImportError as _e:
    _NPU_OPS_AVAILABLE = False
    _NPU_OPS_IMPORT_ERROR = str(_e)

from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class DeepseekV4AscendAttnBackend(AscendAttnBackend):
    """V4 (hybrid sliding + compressed) attention on Ascend NPU.

    Layer dispatch by `compress_ratio` (passed by V4's MQALayer.forward):
      0   → sliding-only:    npu_sparse_flash_attention over swa_kv_pool
      4   → CSA:             sliding + Lightning Indexer top-K over c4_kv_pool
      128 → HCA:             sliding + dense over c128_kv_pool

    For CSA/HCA we do TWO npu_sparse_flash_attention calls (one over swa,
    one over compressed) and merge their outputs. The CUDA FlashMLA fuses
    these into a single kernel; we don't have that fused kernel on NPU yet.
    """

    def __init__(self, model_runner, *args, **kwargs):
        if not _NPU_OPS_AVAILABLE:
            raise RuntimeError(
                f"DeepseekV4AscendAttnBackend requires torch_npu + attentions package: "
                f"{_NPU_OPS_IMPORT_ERROR}"
            )
        super().__init__(model_runner, *args, **kwargs)

        # V4-specific scale (matches CUDA backend's softmax_scale)
        # head_dim_v + qk_rope_head_dim is the "full" head dim for the scale.
        # TODO Session 3: pull from layer config rather than hardcoding.
        self.softmax_scale: Optional[float] = None  # set lazily in forward
        self.head_dim_v: Optional[int] = None       # set lazily in forward

        logger.info(
            "DeepseekV4AscendAttnBackend initialized — V4-aware NPU attention "
            "(uses npu_sparse_flash_attention + npu_lightning_indexer)."
        )

    # ─────────────────────────────────────────────────────────────────
    # Top-level forward — dispatch on compress_ratio (mirrors CUDA forward)
    # CUDA reference: sglang/srt/layers/attention/deepseek_v4_backend.py line 930-1045
    # ─────────────────────────────────────────────────────────────────

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        compress_ratio: Literal[0, 4, 128] = 0,
        save_kv_cache: bool = True,
        attn_sink: Optional[torch.Tensor] = None,
        **_unused,
    ) -> torch.Tensor:
        # MLA contract: V4 passes the same tensor as k and v
        assert k is v, "DeepseekV4 shares k and v (MLA pattern)"

        # Initialize lazy state from layer config on first call
        if self.softmax_scale is None:
            self._init_v4_state(layer)

        # Cache write (V4-aware: routes to right sub-pool by compress_ratio)
        if save_kv_cache:
            self.store_cache(layer.layer_id, k, forward_batch)

        if compress_ratio == 0:
            return self._forward_sliding(q, k, layer, forward_batch, attn_sink)
        elif compress_ratio == 4:
            return self._forward_csa(q, k, layer, forward_batch, attn_sink)
        elif compress_ratio == 128:
            return self._forward_hca(q, k, layer, forward_batch, attn_sink)
        else:
            raise ValueError(
                f"Invalid V4 compress_ratio={compress_ratio} (expected 0, 4, or 128)"
            )

    # ─────────────────────────────────────────────────────────────────
    # Layer-type implementations — Sessions 3, 4, 5
    # ─────────────────────────────────────────────────────────────────

    def _forward_sliding(
        self, q, k, layer, forward_batch, attn_sink
    ) -> torch.Tensor:
        """Sliding window attention over swa_kv_pool. Layers 0, 3 in stub.

        Maps to ONE npu_sparse_flash_attention call with sparse_indices spanning
        the sliding window (constructed from swa_page_indices in metadata).

        Reference call (from sfa_v1.py:916):
            torch.ops.npu.npu_sparse_flash_attention(
                query=q_nope, key=kv, value=kv,
                sparse_indices=swa_indices, scale_value=self.softmax_scale,
                sparse_block_size=1, block_table=...,
                actual_seq_lengths_query=..., actual_seq_lengths_kv=...,
                query_rope=q_pe, key_rope=k_pe,
                layout_query="TND", layout_kv="PA_BSND",
                sparse_mode=3,
            )
        """
        # TODO Session 3:
        # 1. Read swa_k_cache via forward_batch.token_to_kv_pool.get_swa_key_buffer_radix(layer_id)
        # 2. Split q into q_nope (qk_nope_head_dim) and q_rope (qk_rope_head_dim)
        # 3. Get metadata: swa_page_indices, swa_topk_lengths from self.forward_metadata
        # 4. Call npu_sparse_flash_attention
        # 5. Apply attn_sink if provided (post-softmax stabilization)
        raise NotImplementedError(
            "V4 sliding attention on NPU — Session 3. "
            "Plan: split q to nope/rope, fetch swa cache, build window indices, "
            "call npu_sparse_flash_attention(layout_query='TND', layout_kv='PA_BSND')."
        )

    def _forward_csa(
        self, q, k, layer, forward_batch, attn_sink
    ) -> torch.Tensor:
        """Compressed Sparse Attention (sliding + top-K over c4). Layer 1 in stub.

        Two-kernel approach (no unified NPU equivalent of FlashMLA's extra_k_cache):
          (a) sliding_out = npu_sparse_flash_attention over swa cache (same as _forward_sliding)
          (b) Lightning Indexer → top-K indices over c4 cache
          (c) extra_out = npu_sparse_flash_attention over c4 cache with top-K indices
          (d) merge sliding_out and extra_out via log-sum-exp combine (softmax-aware merge)
        """
        # TODO Session 4:
        # See _forward_sliding for (a). For (b)+(c):
        # topk_indices, _ = torch.ops.npu.npu_lightning_indexer(
        #     query=q_indexer, key=index_k, weights=indexer_weights,
        #     sparse_count=2048, sparse_mode=3,
        #     actual_seq_lengths_query=..., actual_seq_lengths_key=...,
        #     block_table=..., layout_query="TND", layout_key="PA_BSND",
        # )
        # extra_out = torch.ops.npu.npu_sparse_flash_attention(
        #     query=q_nope, key=extra_k_cache, value=extra_k_cache,
        #     sparse_indices=topk_indices, sparse_block_size=4, ...
        # )
        raise NotImplementedError(
            "V4 CSA (sliding+top-K) on NPU — Session 4. "
            "Plan: Lightning Indexer for top-K, then two sparse_flash calls + LSE merge."
        )

    def _forward_hca(
        self, q, k, layer, forward_batch, attn_sink
    ) -> torch.Tensor:
        """Heavily Compressed Attention (sliding + dense over c128). Layer 2 in stub.

        Similar to CSA but no indexer — c128 cache is small enough for dense attention.
        """
        # TODO Session 5:
        # (a) sliding_out — same as _forward_sliding
        # (b) extra_out = npu_sparse_flash_attention over c128 cache with FULL indices
        #     (sparse_block_size=128, sparse_indices = all c128 tokens)
        # (c) merge via LSE
        raise NotImplementedError(
            "V4 HCA (sliding+dense compressed) on NPU — Session 5."
        )

    # ─────────────────────────────────────────────────────────────────
    # V4-specific methods called from V4 model code
    # ─────────────────────────────────────────────────────────────────

    def store_cache(
        self, layer_id: int, swa_k: torch.Tensor, forward_batch: "ForwardBatch"
    ) -> None:
        """Write KV to the V4 paged pool. Routes by compress_ratio."""
        # TODO Session 3:
        # The V4 pool exposes set_swa_key_buffer / set_extra_key_buffer.
        # We need to dispatch by compress_ratio.
        # For sliding layers: token_to_kv_pool.set_swa_key_buffer(layer_id, loc, packed_k)
        # For CSA/HCA: token_to_kv_pool.set_extra_key_buffer(layer_id, loc, packed_k)
        # The "packed_k" comes from the V4 compressor — already FP8 nope + BF16 rope.
        raise NotImplementedError(
            "V4 store_cache on NPU — Session 3. "
            "Plan: dispatch to V4 pool's set_swa_key_buffer / set_extra_key_buffer by compress_ratio."
        )

    def forward_core_compressor(self, *args, **kwargs):
        """V4 KV compressor — produces compressed_k from raw k. Called from V4 model.

        CUDA uses CompressorBackendMixin. We port using npu_transpose_batchmatmul.
        """
        # TODO Session 4
        raise NotImplementedError(
            "V4 compressor on NPU — Session 4. "
            "Plan: use npu_transpose_batchmatmul for the compress projection."
        )

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    def _init_v4_state(self, layer) -> None:
        """Lazy init of V4-specific scale/dims from layer config."""
        # Common MLA scale: 1 / sqrt(qk_nope_head_dim + qk_rope_head_dim)
        qk_nope = getattr(layer, "qk_nope_head_dim", None)
        qk_rope = getattr(layer, "qk_rope_head_dim", None)
        head_dim_v = getattr(layer, "v_head_dim", None)
        if qk_nope is None or qk_rope is None or head_dim_v is None:
            raise RuntimeError(
                "Layer is missing V4-specific dims (qk_nope_head_dim, "
                "qk_rope_head_dim, v_head_dim). Cannot init V4 backend state."
            )
        head_dim = qk_nope + qk_rope
        self.softmax_scale = 1.0 / (head_dim ** 0.5)
        self.head_dim_v = head_dim_v
        logger.info(
            f"V4 state initialized: qk_nope={qk_nope}, qk_rope={qk_rope}, "
            f"v_head_dim={head_dim_v}, softmax_scale={self.softmax_scale:.6f}"
        )
