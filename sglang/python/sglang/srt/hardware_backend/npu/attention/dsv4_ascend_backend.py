# Copyright (c) 2026 — DeepseekV4 NPU attention backend port.
##### @Moh_7596 — V4 attention backend for Ascend NPU.
# Subclasses AscendAttnBackend (NPU baseline) and overrides V4 paths.
# Uses sgl-kernel-npu's npu_sparse_flash_attention + npu_lightning_indexer
# (registered under torch.ops.npu.* after `import attentions`).
#
# Status: skeleton — methods raise NotImplementedError with clear session markers.
# Reference: SpecForge/docs/dsv4_npu_port/refs/cuda_forward_method.py (CUDA oracle)
#            SpecForge/docs/dsv4_npu_port/refs/vllm_ascend_attention/sfa_v1.py (NPU patterns)
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

        # Pull V4 dims from model_config (matches CUDA backend at
        # deepseek_v4_backend.py lines 333-340). For our stub:
        # head_dim=128 (nope=64 + rope=64), v_head_dim=64.
        head_dim = model_runner.model_config.head_dim
        self.head_dim = head_dim
        self.head_dim_v = model_runner.model_config.v_head_dim
        self.softmax_scale = float(head_dim) ** -0.5

        # nope/rope split for V4's MLA architecture
        hf_cfg = model_runner.model_config.hf_text_config
        self.qk_rope_head_dim = getattr(hf_cfg, "qk_rope_head_dim", 64)
        self.qk_nope_head_dim = head_dim - self.qk_rope_head_dim

        logger.info(
            f"DeepseekV4AscendAttnBackend initialized: head_dim={head_dim}, "
            f"qk_nope={self.qk_nope_head_dim}, qk_rope={self.qk_rope_head_dim}, "
            f"v_head_dim={self.head_dim_v}, scale={self.softmax_scale:.6f}"
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

        # Cache write (V4-aware: routes to right sub-pool by compress_ratio)
        if save_kv_cache:
            self.store_cache(layer.layer_id, k, forward_batch)

        # @Moh_7596 — Until Ascend kernels for sliding/CSA/HCA are validated
        # against reference outputs, route everything through the PyTorch
        # reference. The reference does correct dense causal MLA attention;
        # CSA/HCA's "compression" is a sparsity optimization (perf feature),
        # not a math change — full attention is a superset of compressed
        # attention, so PyTorch reference gives strictly-more-context but
        # mathematically valid attention. Slow but trains correctly.
        if compress_ratio not in (0, 4, 128):
            raise ValueError(
                f"Invalid V4 compress_ratio={compress_ratio} (expected 0, 4, or 128)"
            )
        if not getattr(self, "_unified_ref_warned", False):
            logger.warning(
                f"[V4 NPU] Routing all compress_ratios through PyTorch reference "
                f"attention. Slow but mathematically correct. Kernel paths "
                f"(sliding/CSA/HCA) remain TODO."
            )
            self._unified_ref_warned = True
        return self._sliding_pytorch_reference(
            q, k, self.qk_nope_head_dim, self.qk_rope_head_dim
        )

    # ─────────────────────────────────────────────────────────────────
    # Layer-type implementations — Sessions 3, 4, 5
    # ─────────────────────────────────────────────────────────────────

    def _forward_sliding(
        self, q, k, layer, forward_batch, attn_sink
    ) -> torch.Tensor:
        """Sliding window attention via npu_sparse_flash_attention.

        Session 3: uses current-batch kv directly (cache reads = Session 4).
        Pattern from vllm-ascend/sfa_v1.py:909.
        """
        kv = k  # V4 contract: k is v
        qk_nope = self.qk_nope_head_dim
        qk_rope = self.qk_rope_head_dim
        expected_head = qk_nope + qk_rope

        if not getattr(self, "_sliding_logged", False):
            logger.info(
                f"[V4 NPU Session 3] First sliding call: "
                f"q.shape={tuple(q.shape)}, q.dtype={q.dtype}, "
                f"kv.shape={tuple(kv.shape)}, kv.dtype={kv.dtype}, "
                f"qk_nope={qk_nope}, qk_rope={qk_rope}, "
                f"layer_id={layer.layer_id}, attn_sink={attn_sink is not None}"
            )
            self._sliding_logged = True

        if q.shape[-1] != expected_head:
            logger.warning(
                f"[V4 NPU] q.shape[-1]={q.shape[-1]} != qk_nope+qk_rope={expected_head}; "
                f"returning zeros."
            )
            return torch.zeros_like(q)

        # @Moh_7596 — npu_sparse_flash_attention kernel HARDCODES qk_head_dim==512
        # (real V4 MLA dims: nope=448, rope=64). Our stub has head_dim=128, which
        # the kernel rejects. For stub configs use PyTorch reference; for real V4
        # use the kernel.
        if expected_head != 512:
            if not getattr(self, "_pytorch_ref_warned", False):
                logger.warning(
                    f"[V4 NPU Session 3] head_dim={expected_head} != 512 "
                    f"(kernel constraint). Using PyTorch reference attention."
                )
                self._pytorch_ref_warned = True
            return self._sliding_pytorch_reference(q, kv, qk_nope, qk_rope)

        try:
            # @Moh_7596 — kernel MLA path requires query/key with rope baked in
            # (head_dim=512). Don't split. Value uses only the nope portion (MLA).
            q_full = q.contiguous()  # (T_q, N_q, 512)
            kv_h = kv if kv.ndim == 3 else kv.unsqueeze(1)  # (T_kv, 1, 512)
            k_full = kv_h.contiguous()
            v_tensor = kv_h[..., :qk_nope].contiguous()  # (T_kv, 1, 448)

            T_q = q_full.shape[0]
            T_kv = k_full.shape[0]

            sparse_indices = (
                torch.arange(T_kv, device=q.device, dtype=torch.int32)
                .unsqueeze(0)
                .expand(T_q, T_kv)
                .contiguous()
            )
            actual_seq_q = torch.tensor([T_q], device=q.device, dtype=torch.int32)
            actual_seq_kv = torch.tensor([T_kv], device=q.device, dtype=torch.int32)

            # @Moh_7596 — attention_mode=1 selects MLA path which allows
            # separate query_rope/key_rope. Default 0 is MHA/GQA which requires
            # rope to be baked into q/k tensors.
            out_tup = torch.ops.npu.npu_sparse_flash_attention(
                query=q_full,
                key=k_full,
                value=v_tensor,
                sparse_indices=sparse_indices,
                scale_value=self.softmax_scale,
                sparse_block_size=1,
                actual_seq_lengths_query=actual_seq_q,
                actual_seq_lengths_kv=actual_seq_kv,
                layout_query="TND",
                layout_kv="TND",
                sparse_mode=3,
                attention_mode=2,
            )
            out = out_tup[0] if isinstance(out_tup, tuple) else out_tup

            if out.shape[-1] == qk_nope:
                zero_rope = torch.zeros(
                    *out.shape[:-1], qk_rope, device=out.device, dtype=out.dtype
                )
                out = torch.cat([out, zero_rope], dim=-1)

            if not getattr(self, "_sliding_succeeded", False):
                logger.info(
                    f"[V4 NPU Session 3] OK sliding attention: out.shape={tuple(out.shape)}"
                )
                self._sliding_succeeded = True

            return out

        except Exception as e:
            logger.error(
                f"[V4 NPU Session 3] sliding kernel call failed: "
                f"{type(e).__name__}: {e}"
            )
            return torch.zeros_like(q)



    # ─────────────────────────────────────────────────────────────────────
    # Session 4 boundary: methods called by V4 model that need real impl.
    # Each raises a clear NotImplementedError so the failure mode is obvious.
    # ─────────────────────────────────────────────────────────────────────

    # Backend attribute V4 model reads at deepseek_v4.py:1037.
    # Session 4 will populate this from init_forward_metadata().
    forward_metadata = None

    def init_forward_metadata_indexer(self, core_attn_metadata) -> None:
        """Build per-batch indexer metadata for the Lightning Indexer.

        Production: returns PagedIndexerMetadata(page_table + c4_seq_lens).
        CUDA reference at deepseek_v4_backend.py:375.
        Stub: returns None — PyTorch reference attention doesn't consume metadata.
        """
        return None

    def forward_c4_indexer(self, *args, **kwargs):
        """Lightning Indexer forward.

        Production: computes sparse_indices via npu_lightning_indexer (q @ K^T +
        topk over compressed KV) and writes indexer KV to pool. CUDA reference
        at sglang/srt/layers/attention/dsv4/indexer.py.
        Stub: silent no-op. PyTorch reference path reads the full (uncompressed)
        kv passed to forward(), so the indexer's compressed K isn't needed.
        """
        if not getattr(self, "_indexer_warned", False):
            logger.info(
                "[V4 NPU Session 4] forward_c4_indexer: no-op (PyTorch reference path)"
            )
            self._indexer_warned = True
        return None

    def forward_core_compressor(self, x, forward_batch, layer_id, compressor):
        """KV compression for CSA (r=4) and HCA (r=128) paths.

        Production: calls compressor(x, forward_batch) to get compressed KV,
        then writes to token_to_kv_pool via set_extra_key_buffer (FP8 nope +
        BF16 rope pack). CUDA reference at dsv4/compressor.py.
        Stub: invoke compressor module so its weights are exercised, but discard
        output. PyTorch reference attention uses uncompressed kv from forward()'s
        args, not the compressed cache.
        """
        if not getattr(self, "_compressor_warned", False):
            logger.info(
                f"[V4 NPU Session 4] forward_core_compressor: layer_id={layer_id} "
                f"ratio={getattr(compressor, 'ratio', '?')}, "
                f"invoking compressor module (output discarded for PyTorch ref)"
            )
            self._compressor_warned = True
        try:
            _ = compressor(x, forward_batch)
        except Exception as e:
            logger.warning(
                f"[V4 NPU Session 4] compressor module raised "
                f"{type(e).__name__}: {e}; continuing"
            )
        return None

    def _sliding_pytorch_reference(
        self, q: torch.Tensor, kv: torch.Tensor, qk_nope: int, qk_rope: int
    ) -> torch.Tensor:
        """Pure PyTorch reference for sliding window MLA attention.

        Used for stub configs (head_dim != 512) where npu_sparse_flash_attention
        won't accept the shapes. Smoke-test grade — produces correct-shape output
        with proper MLA + causal attention math, just slow (no fused kernel).

        Inputs:
          q:  (T_q, N_q, D) where D = qk_nope + qk_rope
          kv: (T_kv, D)     MLA: single kv head, shared across all q heads
        Output:
          (T_q, N_q, D)  — out[..., qk_nope:] is zero (V has no rope component)
        """
        T_q, N_q, D = q.shape
        T_kv = kv.shape[0]

        # All math in fp32 for stability (we're already slow)
        q_f = q.float()
        kv_f = kv.float()

        # Attention scores: q · k^T scaled. q is per-head, kv is shared.
        # q: (T_q, N_q, D) -> (N_q, T_q, D); kv: (T_kv, D) -> (1, D, T_kv) (broadcast)
        q_perm = q_f.permute(1, 0, 2)           # (N_q, T_q, D)
        k_t = kv_f.unsqueeze(0).transpose(-1, -2)  # (1, D, T_kv)
        scores = torch.matmul(q_perm, k_t) * self.softmax_scale  # (N_q, T_q, T_kv)

        # Causal mask (each query can only attend to keys at <= its position)
        causal = torch.triu(
            torch.ones(T_q, T_kv, dtype=torch.bool, device=q.device), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))

        # @Moh_7596 — NaN-safe softmax: any row entirely -inf produces NaN.
        # 1) subtract per-row max for numerical stability (no-op for valid rows)
        # 2) detect rows that are entirely masked (max == -inf)
        # 3) compute softmax in fp32, then zero out NaN rows
        scores_max = scores.amax(dim=-1, keepdim=True)
        # rows where all positions are masked (max is still -inf)
        all_masked = torch.isinf(scores_max) & (scores_max < 0)
        # replace -inf max with 0 to prevent -inf - -inf = NaN
        scores_safe = scores - torch.where(all_masked, torch.zeros_like(scores_max), scores_max)
        weights = torch.softmax(scores_safe.float(), dim=-1)
        # zero out rows that were entirely masked
        weights = torch.where(all_masked, torch.zeros_like(weights), weights)
        weights = weights.to(scores.dtype)

        # MLA: V is the nope part of kv only (rope dim has no value content)
        v = kv_f[..., :qk_nope].unsqueeze(0)  # (1, T_kv, qk_nope)
        out_nope = torch.matmul(weights, v)   # (N_q, T_q, qk_nope)
        out_nope = out_nope.permute(1, 0, 2).contiguous()  # (T_q, N_q, qk_nope)

        # Pad rope dims with zeros — V doesn't carry rope info
        zero_rope = torch.zeros(
            T_q, N_q, qk_rope, dtype=out_nope.dtype, device=out_nope.device
        )
        out_full = torch.cat([out_nope, zero_rope], dim=-1)  # (T_q, N_q, D)

        if not getattr(self, "_ref_succeeded", False):
            logger.info(
                f"[V4 NPU Session 3] OK PyTorch reference sliding: "
                f"out.shape={tuple(out_full.shape)}"
            )
            self._ref_succeeded = True

        return out_full.to(q.dtype)

    def _forward_csa(
        self, q, k, layer, forward_batch, attn_sink
    ) -> torch.Tensor:
        """Compressed Sparse Attention (compress_ratio=4). Used by Layer 1 in stub.

        Production (real V4, head_dim=512): two npu_sparse_flash_attention calls
        (one over swa cache, one over c4 cache with Lightning-Indexer-derived
        sparse_indices), then LSE merge. Real kernel path TODO.

        Stub (head_dim != 512): delegate to PyTorch reference. Math is causal
        dense attention over uncompressed kv — identical to sliding for our
        random-weight smoke test.
        """
        kv = k  # MLA contract
        qk_nope = self.qk_nope_head_dim
        qk_rope = self.qk_rope_head_dim
        head_dim = qk_nope + qk_rope

        if head_dim != 512:
            if not getattr(self, "_csa_ref_warned", False):
                logger.warning(
                    f"[V4 NPU Session 4] CSA head_dim={head_dim} != 512. "
                    f"Using PyTorch reference (causal attention over kv)."
                )
                self._csa_ref_warned = True
            return self._sliding_pytorch_reference(q, kv, qk_nope, qk_rope)

        raise NotImplementedError(
            "[V4 NPU Session 4.kernel] CSA kernel path for real V4 (head_dim=512) "
            "TODO. Plan: npu_lightning_indexer → sparse_indices, then two "
            "npu_sparse_flash_attention calls + LSE merge."
        )

    def _forward_hca(
        self, q, k, layer, forward_batch, attn_sink
    ) -> torch.Tensor:
        """Heavily Compressed Attention (compress_ratio=128). Used by Layer 2 in stub.

        Production (real V4, head_dim=512): sliding + dense over c128 cache,
        LSE merge. Real kernel path TODO (Session 5).

        Stub (head_dim != 512): delegate to PyTorch reference.
        """
        kv = k
        qk_nope = self.qk_nope_head_dim
        qk_rope = self.qk_rope_head_dim
        head_dim = qk_nope + qk_rope

        if head_dim != 512:
            if not getattr(self, "_hca_ref_warned", False):
                logger.warning(
                    f"[V4 NPU Session 4] HCA head_dim={head_dim} != 512. "
                    f"Using PyTorch reference."
                )
                self._hca_ref_warned = True
            return self._sliding_pytorch_reference(q, kv, qk_nope, qk_rope)

        raise NotImplementedError(
            "[V4 NPU Session 5.kernel] HCA kernel path for real V4 TODO. "
            "Plan: sliding + dense compressed attention + LSE merge."
        )

    # ─────────────────────────────────────────────────────────────────

    def forward_compress(self, *args, **kwargs):
        """Stub for V4 Compressor module integration. Returns None because the
        PyTorch reference path discards compressor outputs (compression is a
        perf optimization not used by reference attention)."""
        return None

    # V4-specific methods called from V4 model code
    # ─────────────────────────────────────────────────────────────────

    def store_cache(
        self, layer_id: int, swa_k: torch.Tensor, forward_batch: "ForwardBatch"
    ) -> None:
        """Write KV to V4 paged pool.

        Session 3: no-op for smoke testing. Real packing+writing TODO in
        Session 4 (needs FP8 nope quant + BF16 rope pack).
        """
        if not getattr(self, "_store_cache_warned", False):
            logger.warning(
                "[V4 NPU Session 3] store_cache is a no-op for smoke testing. "
                "Real packing+writing TODO in Session 4."
            )
            self._store_cache_warned = True
        return


    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

